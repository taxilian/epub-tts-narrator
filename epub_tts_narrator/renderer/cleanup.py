"""Conservative cleanup of renderer-owned speech caches.

Callers must hold the book audio-render lock and render-work lock for the whole
operation. Voice stores and all reference/casting directories are outside this
module's scope. The default keeps accepted and rejected normalized speech, the
success index, mixes, and every chapter named by a checkpoint, so speech can resume.
"""

from __future__ import annotations

from pathlib import Path
import re
import stat

from .common import digest, file_hash, read_json


_HASH = re.compile(r"[0-9a-f]{64}")
# Per-run retry limits do not limit cumulative take IDs. Keep canonical
# minimum-three-digit names, including attempts beyond 999, recognizable.
_ATTEMPT_NUMBER = r"(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2,})"
_RAW = re.compile(rf"attempt-({_ATTEMPT_NUMBER})-(?:raw|recovery-raw|regenerated-[0-9a-f]{{16}})\.wav")
_ATTEMPT = re.compile(rf"attempt-({_ATTEMPT_NUMBER})\.(?:wav|json)")
_SPEECH = {"speech", "paragraph_speech"}
_STORES = ("clips", "chapters", "mixes", "completed-chapters", "speech-cache")


def _directory(path):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Cleanup requires a regular, nonsymlink directory: {path}")


def _file(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Cleanup refuses nonregular files or symlinks: {path}")
    return info


def _json(path):
    if _file(path).st_size > 64 * 1024 * 1024:
        raise ValueError(f"Cleanup metadata is oversized: {path}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Cleanup metadata must be an object: {path}")
    return value


def _exists(path):
    return path.exists() or path.is_symlink()


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


class _Plan:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.directories = set()
        self.checked = {}

    def check_file(self, path, checksum=None):
        info = _file(path)
        if checksum is not None and (not _hash(checksum) or file_hash(path) != checksum):
            raise ValueError(f"Cleanup refused a file with an invalid checksum: {path}")
        self.checked[path] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        return info

    def remove_file(self, path, category):
        info = self.check_file(path)
        self.files[path] = {"path": str(path.relative_to(self.root)),
                            "bytes": info.st_size, "category": category}

    def remove_directory(self, path):
        _directory(path)
        self.directories.add(path)

    def finish(self, *, dry_run, preserves_resumability=True):
        # Preflight the entire operation before deleting anything. The caller's
        # locks prevent normal writers; stat checks also catch unexpected edits.
        for path, expected in self.checked.items():
            actual = _file(path)
            if (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) != expected:
                raise ValueError(f"A cleanup input changed during inspection: {path}")
        for path in self.directories:
            _directory(path)
        categories = {}
        for item in self.files.values():
            counts = categories.setdefault(item["category"], {"files_removed": 0, "bytes_removed": 0})
            counts["files_removed"] += 1
            counts["bytes_removed"] += item["bytes"]
        result = {"dry_run": dry_run, "preserves_resumability": preserves_resumability,
                  "files_removed": len(self.files),
                  "bytes_removed": sum(item["bytes"] for item in self.files.values()),
                  "directories_removed": len(self.directories), "categories": categories,
                  "files": sorted(self.files.values(), key=lambda item: item["path"])}
        if not dry_run:
            for path in self.files:
                path.unlink()
            for path in sorted(self.directories, key=lambda item: len(item.parts), reverse=True):
                path.rmdir()
        return result


def _request(record, path):
    request = record.get("request")
    if not isinstance(request, dict) or record.get("request_hash") != digest(request):
        raise ValueError(f"Cleanup found invalid synthesis request metadata: {path}")
    return request


def _normalized(record, directory, plan, *, request_hash, attempt=None):
    number = record.get("attempt")
    if (type(number) is not int or number < 1
            or attempt is not None and number != attempt
            or record.get("request_hash") != request_hash
            or record.get("audio_file") != f"attempt-{number:03d}.wav"
            or not isinstance(record.get("report"), dict)):
        raise ValueError(f"Cleanup found invalid normalized take metadata: {directory}")
    wav = directory / record["audio_file"]
    plan.check_file(wav, record.get("audio_sha256"))
    if not _hash(record.get("audio_sha256")):
        raise ValueError(f"Cleanup needs a normalized audio checksum: {wav}")


def _take(directory, plan, *, require_directory_identity=False):
    _directory(directory)
    accepted_path = directory / "accepted.json"
    if not _exists(accepted_path):
        return
    accepted = _json(accepted_path)
    plan.check_file(accepted_path)
    request = _request(accepted, accepted_path)
    # A caller accidentally passing a reference directory must never cause
    # cleanup of its audio, even if its filenames resemble speech takes.
    if request.get("kind") not in _SPEECH:
        return
    if require_directory_identity and directory.name != accepted["request_hash"]:
        raise ValueError(f"Cleanup found a speech cache with the wrong request identity: {directory}")
    _normalized(accepted, directory, plan, request_hash=accepted["request_hash"])
    if accepted["report"].get("passed") is not True:
        return
    for raw in sorted(directory.iterdir()):
        match = _RAW.fullmatch(raw.name)
        if match is None:
            continue
        _file(raw)
        attempt = int(match[1])
        metadata = directory / f"attempt-{attempt:03d}.json"
        if _exists(metadata):
            record = _json(metadata)
            plan.check_file(metadata)
            candidate_request = _request(record, metadata)
            if candidate_request.get("kind") not in _SPEECH:
                raise ValueError(f"Cleanup found reference voice metadata in a speech take: {metadata}")
            # Interrupted generation can leave a diagnostic JSON without a
            # normalized recording. Retain the raw until a checked copy exists.
            if record.get("audio_file") is None and record.get("audio_sha256") is None:
                continue
            _normalized(record, directory, plan, request_hash=accepted["request_hash"], attempt=attempt)
        elif attempt != accepted["attempt"]:
            continue
        plan.remove_file(raw, "speech_intermediates")


def clean_take_intermediates(directory, *, dry_run=False):
    """Remove redundant raw speech takes after acceptance, under caller-held locks.

    All normalized attempts, verification records, and failed/pending-only take
    directories are retained. Dry runs report the same paths and byte counts
    without changing any file.
    """
    directory = Path(directory)
    _directory(directory.parent)
    plan = _Plan(directory)
    _take(directory, plan)
    return plan.finish(dry_run=dry_run)


def discard_interrupted_raw(directory):
    """Discard unfinished raw speech before a replacement generation starts.

    The caller must identify this as a speech directory and hold its take/work
    locks. Known reference metadata opts the entire directory out. Normalized
    candidates, successful verification records, and unknown files are retained;
    no recognized unsafe file is followed or removed.
    """
    directory = Path(directory)
    _directory(directory)
    plan = _Plan(directory)
    protected = set()
    accepted_path = directory / "accepted.json"
    if _exists(accepted_path):
        accepted = _json(accepted_path)
        plan.check_file(accepted_path)
        if _request(accepted, accepted_path).get("kind") not in _SPEECH:
            return 0
        if isinstance(accepted.get("report"), dict) and accepted["report"].get("passed") is True:
            number = accepted.get("attempt")
            if type(number) is not int or number < 1:
                raise ValueError(f"Cleanup found an invalid accepted attempt: {accepted_path}")
            protected.add(number)
    paths = sorted(directory.iterdir())
    for metadata in paths:
        match = _ATTEMPT.fullmatch(metadata.name)
        if match is None or metadata.suffix != ".json":
            continue
        record = _json(metadata)
        plan.check_file(metadata)
        if _request(record, metadata).get("kind") not in _SPEECH:
            return 0
        number = int(match[1])
        if type(record.get("attempt")) is not int or record["attempt"] != number:
            raise ValueError(f"Cleanup found mismatched attempt metadata: {metadata}")
        if not isinstance(record.get("report"), dict):
            raise ValueError(f"Cleanup found invalid verification metadata: {metadata}")
        if record["report"].get("passed") is True:
            protected.add(number)
    for raw in paths:
        match = _RAW.fullmatch(raw.name)
        if match is None:
            continue
        _file(raw)
        number = int(match[1])
        normalized = directory / f"attempt-{number:03d}.wav"
        if _exists(normalized):
            plan.check_file(normalized)
            continue
        if number not in protected:
            plan.remove_file(raw, "interrupted_speech")
    return plan.finish(dry_run=False)["files_removed"]


def _signed(path, expected_format, plan):
    record = _json(path)
    plan.check_file(path)
    checksum = record.pop("record_sha256", None)
    if (checksum != digest(record) or record.get("format") != expected_format
            or record.get("version") != 1):
        raise ValueError(f"Cleanup found invalid cache metadata: {path}")
    return record


def _chapter(directory, plan):
    _directory(directory)
    record = _signed(directory / "chapter.json", "audiobook-renderer-chapter", plan)
    identity, chapter = record.get("identity"), record.get("chapter")
    if (not isinstance(identity, dict) or digest(identity) != directory.name
            or not isinstance(identity.get("context"), dict)
            or not isinstance(identity.get("settings"), dict)
            or not isinstance(chapter, dict) or type(chapter.get("frames")) is not int
            or chapter["frames"] < 0):
        raise ValueError(f"Cleanup found invalid chapter identity: {directory}")
    kind = identity["settings"].get("chapter_format")
    if kind not in {"m4a", "mp3"}:
        raise ValueError(f"Cleanup found invalid chapter format: {directory}")
    expected = {"chapter.json", "chapter.wav"}
    if not _hash(record.get("pcm_sha256")):
        raise ValueError(f"Cleanup needs a chapter PCM checksum: {directory}")
    plan.check_file(directory / "chapter.wav", record["pcm_sha256"])
    if chapter["frames"]:
        expected.add(f"chapter.{kind}")
        if not _hash(record.get("encoded_sha256")):
            raise ValueError(f"Cleanup needs an encoded chapter checksum: {directory}")
        plan.check_file(directory / f"chapter.{kind}", record["encoded_sha256"])
    elif record.get("encoded_sha256") is not None:
        raise ValueError(f"Cleanup found encoded audio for an empty chapter: {directory}")
    children = {path.name for path in directory.iterdir()}
    if ".DS_Store" in children:
        plan.check_file(directory / ".DS_Store")
    if children - {".DS_Store"} != expected:
        raise ValueError(f"Cleanup found unrecognized files in a complete chapter: {directory}")
    return record


def _chapters(work, plan):
    checkpoints, chapters = work / "completed-chapters", work / "chapters"
    referenced = {}
    if _exists(checkpoints):
        _directory(checkpoints)
        for path in sorted(checkpoints.iterdir()):
            if path.name == ".DS_Store":
                _file(path)
                continue
            if path.suffix != ".json":
                raise ValueError(f"Cleanup found an unrecognized chapter checkpoint: {path}")
            record = _signed(path, "audiobook-chapter-checkpoint", plan)
            if (not _hash(record.get("cache_key")) or not isinstance(record.get("context"), dict)
                    or not isinstance(record.get("acceptance"), list)):
                raise ValueError(f"Cleanup found an invalid chapter checkpoint: {path}")
            referenced.setdefault(record["cache_key"], []).append(record)
    if not _exists(chapters):
        if referenced:
            raise ValueError("Cleanup found chapter checkpoints with missing chapter audio")
        return
    _directory(chapters)
    complete = {}
    for directory in sorted(chapters.iterdir()):
        if directory.name == ".DS_Store":
            _file(directory)
            continue
        _directory(directory)
        if _HASH.fullmatch(directory.name) is None:
            # Interrupted staging directories are not complete chapter caches.
            continue
        if not _exists(directory / "chapter.json"):
            continue
        complete[directory.name] = _chapter(directory, plan)
    for key, records in referenced.items():
        if key not in complete:
            raise ValueError(f"Cleanup found a checkpoint referencing an incomplete chapter: {key}")
        cached = complete[key]
        for record in records:
            if record["context"] != cached["identity"]["context"]:
                raise ValueError(f"Cleanup found a checkpoint/chapter context mismatch: {key}")
            audio = record.get("audio_file")
            expected = f"chapter.{cached['identity']['settings']['chapter_format']}"
            if ((cached["chapter"]["frames"] and
                    (not isinstance(audio, str) or Path(audio).name != expected or Path(audio).parent.name != key))
                    or not cached["chapter"]["frames"] and audio is not None):
                raise ValueError(f"Cleanup found an invalid checkpoint audio reference: {key}")
    for key in complete.keys() - referenced.keys():
        directory = chapters / key
        for path in sorted(directory.iterdir()):
            plan.remove_file(path, "obsolete_chapters")
        plan.remove_directory(directory)


def _purge_clip(directory, plan):
    _directory(directory)
    for path in sorted(directory.iterdir()):
        if path.name == ".DS_Store":
            plan.remove_file(path, "render_cache")
            continue
        match = _ATTEMPT.fullmatch(path.name)
        if path.name != "accepted.json" and match is None and _RAW.fullmatch(path.name) is None:
            raise ValueError(f"Cleanup found an unrecognized speech cache file: {path}")
        _file(path)
        if path.suffix == ".json":
            record = _json(path)
            request = _request(record, path)
            if request.get("kind") not in _SPEECH:
                raise ValueError(f"Cleanup refuses to discard a voice/reference sample: {path}")
            if digest(request) != directory.name:
                raise ValueError(f"Cleanup found a speech cache with the wrong request identity: {path}")
        plan.remove_file(path, "render_cache")
    plan.remove_directory(directory)


def _purge_cache(directory, plan, store, *, staging=False):
    _directory(directory)
    names = ({"chapter.json", "chapter.wav", "chapter.m4a", "chapter.mp3", "normalized.wav"}
             if store == "chapters" else {"mix.json", "mix.wav"}) | {".DS_Store"}
    for path in sorted(directory.iterdir()):
        if staging and path.name == "ready":
            _purge_cache(path, plan, store)
            continue
        if path.name not in names:
            raise ValueError(f"Cleanup found an unrecognized {store} cache file: {path}")
        plan.remove_file(path, "render_cache")
    plan.remove_directory(directory)


def _speech_cache(work, plan, *, discard=False):
    """Keep the success index normally; discard only its validated owned files.

    An index can outlive deliberately deleted clip audio. Validate the index's
    own binding without requiring its referenced waveform to remain available.
    """
    directory = work / "speech-cache"
    if not _exists(directory):
        return
    _directory(directory)
    for path in sorted(directory.iterdir()):
        if path.name == ".DS_Store":
            plan.check_file(path)
        else:
            if path.suffix != ".json" or not _hash(path.stem):
                raise ValueError(f"Cleanup found an unrecognized speech cache index entry: {path}")
            record = _signed(path, "audiobook-speech-cache", plan)
            identity = record.get("identity")
            if (set(record) != {"format", "version", "identity", "request_hash",
                                "accepted_sha256", "audio_sha256"}
                    or not isinstance(identity, dict) or digest(identity) != path.stem
                    or identity.get("kind") not in _SPEECH
                    or not all(_hash(record.get(key)) for key in
                               ("request_hash", "accepted_sha256", "audio_sha256"))):
                raise ValueError(f"Cleanup found invalid speech cache index metadata: {path}")
        if discard:
            plan.remove_file(path, "render_cache")
    if discard:
        plan.remove_directory(directory)


def _purge(work, plan):
    for store in _STORES:
        if store == "speech-cache":
            _speech_cache(work, plan, discard=True)
            continue
        directory = work / store
        if not _exists(directory):
            continue
        _directory(directory)
        for path in sorted(directory.iterdir()):
            if path.name == ".DS_Store":
                plan.remove_file(path, "render_cache")
                continue
            if store == "completed-chapters":
                if path.suffix != ".json":
                    raise ValueError(f"Cleanup found an unrecognized chapter checkpoint: {path}")
                _signed(path, "audiobook-chapter-checkpoint", plan)
                plan.remove_file(path, "render_cache")
            elif store == "clips" and _HASH.fullmatch(path.name):
                _purge_clip(path, plan)
            elif store in {"chapters", "mixes"}:
                staging = bool(re.fullmatch(r"\.(?:chapter|mix)-[a-z0-9_]{8}", path.name))
                if not _HASH.fullmatch(path.name) and not staging:
                    raise ValueError(f"Cleanup found an unrecognized cache directory: {path}")
                _purge_cache(path, plan, store, staging=staging)
            else:
                raise ValueError(f"Cleanup found an unrecognized speech cache directory: {path}")
        plan.remove_directory(directory)


def cleanup_work(work, *, dry_run=False, discard_render_cache=False):
    """Clean speech intermediates and obsolete chapters under caller-held locks.

    ``discard_render_cache=True`` explicitly removes speech/chapter cache stores
    and the successful speech index, requiring regeneration on the next render.
    It never traverses voice stores, casting/reference candidates, or published output.
    Every path to remove is listed in the report; in dry-run mode its counts
    describe what would be removed. No changes occur until preflight succeeds.
    """
    work = Path(work)
    plan = _Plan(work)
    if not _exists(work):
        return plan.finish(dry_run=dry_run, preserves_resumability=not discard_render_cache)
    _directory(work)
    if discard_render_cache:
        _purge(work, plan)
    else:
        _speech_cache(work, plan)
        clips = work / "clips"
        if _exists(clips):
            _directory(clips)
            for directory in sorted(clips.iterdir()):
                if directory.name == ".DS_Store":
                    _file(directory)
                    continue
                _directory(directory)
                if _HASH.fullmatch(directory.name) is None:
                    raise ValueError(f"Cleanup found an unrecognized speech cache directory: {directory}")
                _take(directory, plan, require_directory_identity=True)
        _chapters(work, plan)
    return plan.finish(dry_run=dry_run, preserves_resumability=not discard_render_cache)
