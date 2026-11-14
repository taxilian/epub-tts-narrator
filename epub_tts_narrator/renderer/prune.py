"""Delete render audio and records a book no longer needs (``prune``, ``just cleanup``).

Three groups, all on by default; ``--only`` picks a subset:

- ``voices``: each voice's current local choice stays. This book's other
  recordings of that voice (older candidates, and the first recording a later
  choice replaced), the casting takes that created them under
  ``render-work/casting-candidates``, and loudness previews of recordings that
  are no longer current are removed. A voice whose regeneration is pending or
  failed, or that has no local choice, keeps everything. Earlier books' voice
  stores are never touched, so a continued book keeps every inherited voice,
  and a casting take whose hosted voice is not yet released is kept.
- ``chapters``: for each chapter, the newest complete version stays, even when
  it is out of date, together with the version its checkpoint names; older
  versions and abandoned assembly staging are removed.
- ``clips``: a take directory stays when the next render reuses or resumes it
  (the speech cache decides, exactly as ``cache-report`` replays it), when a
  kept chapter was built from it, or when a person judged one of its takes or a
  failed chapter names it; recovery parts and other directories a kept one
  names stay too. Inside a kept, accepted directory the rejected takes' audio
  and continuity codes go; its records stay. Every other speech take directory,
  its speech-cache index entries and pause timings, and old run plans go.

Never traverses the published output, never deletes an ``.m4b``, a lock or a
lease, and keeps anything it does not recognize (listed in the report). The
caller holds no locks: ``prune_book`` takes the render-work, book audio and
voice store locks itself and refuses to run while any is held.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat

from .cleanup import _Plan, _chapter_staging, _directory, _file, _json, _signed, _SPEECH
from .common import digest, read_json


GROUPS = ("voices", "chapters", "clips")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX24 = re.compile(r"[0-9a-f]{24}")
_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_CODES = re.compile(r"[0-9a-f]{64}\.codes\.json")
# A take is a WAV or its stored FLAC (``stored_audio``); records name the WAV.
_TAKE_AUDIO = re.compile(r"(?:attempt|selection)-.+\.(?:wav|flac)")
# A casting take with any of these and no cleanup record still owns a hosted voice.
_HOSTED = ("gemini-voice-cleanup-request.json", "gemini-voice.json", "gemini-voice-pending.json")
_LABELS = {
    "voices.superseded_recordings": "older recordings",
    "voices.casting_takes": "casting takes",
    "voices.previews": "previews",
    "chapters.superseded": "older versions",
    "chapters.interrupted_chapters": "interrupted assembly",
    "clips.unused_takes": "unused take folders",
    "clips.rejected_takes": "rejected takes",
    "clips.rejected_codes": "rejected takes' codes",
    "clips.intermediates": "raw audio",
    "clips.index": "speech index entries",
    "clips.pause_timings": "pause timings",
    "clips.run_plans": "old run plans",
}


def parse_groups(text):
    """``None`` → every group; else a comma-separated subset, in canonical order."""
    if text is None:
        return GROUPS
    names = {part.strip() for part in text.split(",") if part.strip()}
    if not names or names - set(GROUPS):
        raise ValueError(f"--only takes one or more of {', '.join(GROUPS)}, separated by commas")
    return tuple(group for group in GROUPS if group in names)


def _present(path):
    return path.exists() or path.is_symlink()


def _guarded(name):
    return name.endswith((".lease", ".lock", ".m4b")) or ".lock." in name


def _tree(directory):
    """Every file and directory below ``directory`` (itself included), or None if any is unsafe."""
    files, directories, pending = [], [], [Path(directory)]
    while pending:
        current = pending.pop()
        if current.is_symlink() or not current.is_dir() or _guarded(current.name):
            return None
        directories.append(current)
        for entry in os.scandir(current):
            path = Path(entry.path)
            if _guarded(entry.name) or entry.is_symlink():
                return None
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                files.append(path)
            else:
                return None
    return files, directories


def _tokens(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return set(_TOKEN.findall(handle.read()))


class _Removal(_Plan):
    def __init__(self, root):
        super().__init__(root)
        self.group = None
        self.directory_groups = {}
        self.retained = []

    def _label(self, path):
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def remove_file(self, path, category):
        if _guarded(path.name):
            raise ValueError(f"Cleanup never removes books, locks or leases: {path}")
        info = self.check_file(path)
        self.files[path] = {"path": self._label(path), "bytes": info.st_size,
                            "category": category if "." in category else f"{self.group}.{category}"}

    def remove_directory(self, path):
        super().remove_directory(path)
        self.directory_groups[path] = self.group

    def remove_tree(self, directory, category):
        found = _tree(directory)
        if found is None:
            self.keep(directory, "holds a link, lock or lease")
            return False
        files, directories = found
        for path in files:
            self.remove_file(path, category)
        for path in directories:
            self.remove_directory(path)
        return True

    def keep(self, path, reason):
        self.retained.append({"group": self.group, "path": self._label(path), "reason": reason})

    def report(self, *, dry_run, groups):
        result = self.finish(dry_run=dry_run)
        summary = {group: {"files": 0, "bytes": 0, "directories": 0, "categories": {}} for group in groups}
        for item in self.files.values():
            group = item["category"].split(".", 1)[0]
            entry = summary[group]
            entry["files"] += 1
            entry["bytes"] += item["bytes"]
            counts = entry["categories"].setdefault(item["category"].split(".", 1)[1], {"files": 0, "bytes": 0})
            counts["files"] += 1
            counts["bytes"] += item["bytes"]
        for group in self.directory_groups.values():
            summary[group]["directories"] += 1
        result.pop("preserves_resumability", None)
        result.pop("categories", None)
        return dict(result, groups=summary, retained=list(self.retained))


# Chapters ------------------------------------------------------------------------------------------

def _chapter_versions(work, plan):
    """Complete chapter versions by name, each with its chapter slot and age."""
    root, versions = work / "chapters", {}
    if not _present(root):
        return versions
    _directory(root)
    for directory in sorted(root.iterdir()):
        if _HEX64.fullmatch(directory.name) is None:
            continue
        try:
            _directory(directory)
            path = directory / "chapter.json"
            info = _file(path)
            record = _json(path)
            body = {key: value for key, value in record.items() if key != "record_sha256"}
            identity = body.get("identity")
            chapter = identity.get("chapter") if isinstance(identity, dict) else None
            slot = (chapter.get("id") if isinstance(chapter, dict) and isinstance(chapter.get("id"), str)
                    else chapter.get("number") if isinstance(chapter, dict) else None)
            if (record.get("record_sha256") != digest(body) or body.get("format") != "audiobook-renderer-chapter"
                    or digest(identity) != directory.name or slot is None):
                raise ValueError("not a complete chapter")
        except (ValueError, OSError):
            plan.keep(directory, "unrecognized chapter folder")
            continue
        versions[directory.name] = {"slot": slot, "age": (info.st_mtime_ns, directory.name),
                                    "path": directory}
    return versions


def _checkpoints(work, plan):
    root, records = work / "completed-chapters", []
    if not _present(root):
        return records
    _directory(root)
    for path in sorted(root.glob("*.json")):
        record = _signed(path, "audiobook-chapter-checkpoint", plan)
        if not isinstance(record.get("cache_key"), str) or _HEX64.fullmatch(record["cache_key"]) is None:
            raise ValueError(f"Cleanup found an invalid chapter checkpoint: {path}")
        records.append((path, record))
    return records


def _kept_chapters(versions, checkpoints):
    kept = {record["cache_key"] for _, record in checkpoints} & versions.keys()
    newest = {}
    for name, version in versions.items():
        if version["slot"] not in newest or version["age"] > versions[newest[version["slot"]]]["age"]:
            newest[version["slot"]] = name
    return kept | set(newest.values())


def _prune_chapters(work, plan, versions, kept):
    for name, version in sorted(versions.items()):
        if name not in kept:
            plan.remove_tree(version["path"], "superseded")
    root = work / "chapters"
    if not _present(root):
        return
    for directory in sorted(root.iterdir()):
        if directory.name.startswith(".chapter-"):
            try:
                if not _chapter_staging(directory, plan):
                    plan.keep(directory, "unrecognized assembly staging")
            except (ValueError, OSError):
                plan.keep(directory, "unrecognized assembly staging")


# Clips ---------------------------------------------------------------------------------------------

def _take_record(directory):
    """(accepted record or None, its JSON text) for a speech take directory; None if not speech."""
    accepted, text = None, None
    path = directory / "accepted.json"
    if _present(path):
        record = _json(path)
        request = record.get("request")
        if (not isinstance(request, dict) or digest(request) != directory.name
                or record.get("request_hash") != directory.name):
            return None
        if request.get("kind") not in _SPEECH:
            return None
        accepted = record
        text = path.read_text(encoding="utf-8")
    else:
        attempts = sorted(directory.glob("attempt-*.json"))
        if not attempts:
            return None
        request = _json(attempts[0]).get("request")
        if not isinstance(request, dict) or digest(request) != directory.name or request.get("kind") not in _SPEECH:
            return None
    passed = accepted is not None and isinstance(accepted.get("report"), dict) and accepted["report"].get("passed") is True
    return {"accepted": accepted if passed else None, "text": text, "path": directory}


def _whole_takes(work):
    """Take directories a person or a failure record points at: kept exactly as they are."""
    names = set()
    for path in (work / "take-feedback.json", work / "status.json"):
        if _present(path):
            _file(path)
            names |= _tokens(path)
    return names


def _prune_clips(work, plan, needed, chapter_tokens):
    reused, resumed = needed
    root = work / "clips"
    takes = {}
    if _present(root):
        _directory(root)
        for directory in sorted(root.iterdir()):
            if _HEX64.fullmatch(directory.name) is None or directory.is_symlink() or not directory.is_dir():
                if directory.name != ".DS_Store":
                    plan.keep(directory, "unrecognized clip folder")
                continue
            try:
                take = _take_record(directory)
            except (ValueError, OSError, UnicodeDecodeError):
                take = None
            if take is None:
                plan.keep(directory, "not a recognized speech take")
                continue
            takes[directory.name] = take
    by_audio = {}
    for name, take in takes.items():
        if take["accepted"] is not None and isinstance(take["accepted"].get("audio_sha256"), str):
            by_audio.setdefault(take["accepted"]["audio_sha256"], set()).add(name)
    whole = (_whole_takes(work) | set(resumed)) & takes.keys()
    fallbacks = work / "paragraph-fallbacks"
    if _present(fallbacks):
        _directory(fallbacks)
        whole |= {path.stem for path in fallbacks.glob("*.json")} & takes.keys()
    kept = set(reused) & takes.keys()
    kept |= chapter_tokens & takes.keys()
    for token in chapter_tokens:
        kept |= by_audio.get(token, set())
    kept |= whole
    pending = list(kept)
    while pending:
        directory = takes[pending.pop()]["path"]
        found = _tree(directory)
        if found is None:
            continue
        for path in found[0]:
            if path.suffix == ".json":
                for token in _tokens(path) & takes.keys():
                    if token not in kept:
                        kept.add(token)
                        pending.append(token)
    removed = set()
    for name, take in sorted(takes.items()):
        directory = take["path"]
        if name not in kept:
            if plan.remove_tree(directory, "unused_takes"):
                removed.add(name)
            continue
        accepted = take["accepted"]
        if accepted is None or name in whole:
            continue  # Pending work and judged takes stay exactly as they are.
        raw = accepted.get("raw_file")
        for path in sorted(directory.iterdir()):
            logical = path.name[:-len(".flac")] + ".wav" if path.name.endswith(".flac") else path.name
            if path.is_symlink() or not path.is_file() or logical == accepted.get("audio_file"):
                continue
            named = f'"{logical}"' in take["text"]
            if _TAKE_AUDIO.fullmatch(path.name) and (not named or logical == raw):
                plan.remove_file(path, "intermediates" if logical == raw else "rejected_takes")
            elif _CODES.fullmatch(path.name) and not named:
                plan.remove_file(path, "rejected_codes")
    kept_audio = {take["accepted"].get("audio_sha256") for name, take in takes.items()
                  if name in kept and take["accepted"] is not None} | chapter_tokens
    _prune_index(work, plan, removed)
    _prune_timings(work, plan, kept_audio)
    _prune_run_plans(work, plan)


def _prune_index(work, plan, removed):
    root = work / "speech-cache"
    if not removed or not _present(root):
        return
    _directory(root)
    for path in sorted(root.glob("*.json")):
        try:
            record = _json(path)
        except (ValueError, OSError):
            plan.keep(path, "unrecognized speech index entry")
            continue
        if record.get("format") == "audiobook-speech-cache" and record.get("request_hash") in removed:
            plan.remove_file(path, "index")


def _prune_timings(work, plan, kept_audio):
    root = work / "pause-word-timings"
    if not _present(root):
        return
    _directory(root)
    for path in sorted(root.glob("*.json")):
        try:
            record = _json(path)
            audio = record["binding"]["audio_sha256"]
            if record.get("format") != "epub-tts-take-pause-words" or not isinstance(audio, str):
                raise ValueError
        except (ValueError, OSError, KeyError, TypeError):
            plan.keep(path, "unrecognized pause timing")
            continue
        if audio not in kept_audio:
            plan.remove_file(path, "pause_timings")


def _prune_run_plans(work, plan):
    root = work / "runs"
    if not _present(root):
        return
    _directory(root)
    current = None
    status = work / "status.json"
    if _present(status):
        record = read_json(status)
        current = record.get("plan_hash") if isinstance(record, dict) else None
    runs = []
    for directory in sorted(root.iterdir()):
        path = directory / "plan.json"
        if (_HEX64.fullmatch(directory.name) is None or directory.is_symlink() or not directory.is_dir()
                or {item.name for item in directory.iterdir()} != {"plan.json"}):
            continue
        runs.append((_file(path).st_mtime_ns, directory))
    newest = max(runs)[1] if runs else None
    for _, directory in runs:
        if directory != newest and directory.name != current:
            plan.remove_tree(directory, "run_plans")


# Voices --------------------------------------------------------------------------------------------

def _voice_ids(root):
    """Voice IDs with local state in this store, by store folder name."""
    found = {}

    def add(path):
        record = _json(path)
        voice_id = record.get("voice_id")
        if not isinstance(voice_id, str):
            raise ValueError(f"Voice record without a voice ID: {path}")
        found.setdefault(digest(voice_id)[:24], voice_id)

    selections = root / "selections"
    if _present(selections):
        for path in sorted(selections.glob("*.json")):
            add(path)
    for directory in sorted(root.iterdir()):
        if _HEX24.fullmatch(directory.name) and _present(directory / "voice.json"):
            add(directory / "voice.json")
    candidates = root / "candidates"
    if _present(candidates):
        for directory in sorted(candidates.iterdir()):
            for path in sorted(directory.glob("*/voice.json")):
                add(path)
    for key, voice_id in found.items():
        if digest(voice_id)[:24] != key:
            raise ValueError(f"Voice folder {key} does not belong to its recorded voice")
    return found


def _casting_request(directory):
    for name in ("casting-request.json", "accepted.json", *sorted(path.name for path in directory.glob("attempt-*.json"))):
        path = directory / name
        if not _present(path):
            continue
        try:
            record = _json(path)
        except (ValueError, OSError):
            return None
        request = record if name == "casting-request.json" else record.get("request")
        if isinstance(request, dict) and digest(request) == directory.name:
            return request
    return None


def _hosted_voice_pending(directory):
    if _present(directory / "gemini-cleanup-pending.json"):
        return True
    return (any(_present(directory / name) for name in _HOSTED)
            and not _present(directory / "gemini-voice-cleanup.json"))


def _prune_voices(bundle, store, work, plan):
    from .voices import VoiceStore
    root = store.directory
    current, published, folders = {}, {}, {}
    if _present(root):
        _directory(root)
        for key, voice_id in _voice_ids(root).items():
            binding, marker = store._resolve_local(voice_id)
            if marker is None and binding is not None:
                current[key] = binding
            legacy = root / key
            if _present(legacy / "voice.json"):
                folders.setdefault(key, []).append((VoiceStore.LEGACY_CANDIDATE, legacy))
            for path in sorted((root / "candidates" / key).glob("*/voice.json")):
                folders.setdefault(key, []).append((path.parent.name, path.parent))
        for key, candidates in folders.items():
            for candidate_id, folder in candidates:
                record = _json(folder / "voice.json")
                request_hash = (record.get("creation") or {}).get("request_hash")
                if isinstance(request_hash, str):
                    published[request_hash] = (key, candidate_id)
        for key, candidates in sorted(folders.items()):
            if key not in current:
                continue  # Pending, failed or never chosen here: nothing is superseded yet.
            chosen = current[key]["candidate_id"]
            removed = {folder for candidate_id, folder in candidates if candidate_id != chosen
                       and plan.remove_tree(folder, "superseded_recordings")}
            revisions = root / "candidates" / key
            if (chosen == VoiceStore.LEGACY_CANDIDATE and revisions.is_dir() and not revisions.is_symlink()
                    and set(revisions.iterdir()) <= removed):
                plan.remove_directory(revisions)  # Every revision went; so does their folder.
    references = {binding["reference_sha256"] for binding in current.values()}
    for voice in bundle["cast"]["voices"]:
        loaded = store.load(voice["id"])
        if loaded is not None:
            references.add(loaded["reference_sha256"])
    takes = work / "casting-candidates"
    if _present(takes):
        _directory(takes)
        for directory in sorted(takes.iterdir()):
            if _HEX64.fullmatch(directory.name) is None or directory.is_symlink() or not directory.is_dir():
                continue
            if _hosted_voice_pending(directory):
                plan.keep(directory, "its hosted voice is not released yet")
                continue
            owner = published.get(directory.name)
            if owner is None:
                request = _casting_request(directory)
                if (request is None or not isinstance(request.get("voice_id"), str)
                        or request.get("series_id", store.series_id) != store.series_id):
                    plan.keep(directory, "unrecognized casting take")
                    continue
                owner = (digest(request["voice_id"])[:24], request.get("nonce"))
            key, candidate_id = owner
            if key in current and candidate_id != current[key]["candidate_id"]:
                plan.remove_tree(directory, "casting_takes")
    previews = work / "voice-previews"
    if _present(previews):
        _directory(previews)
        for directory in sorted(previews.iterdir()):
            if _HEX64.fullmatch(directory.name) is None or directory.is_symlink() or not directory.is_dir():
                continue
            try:
                identity = _json(directory / "normalization.json").get("identity")
                if digest(identity) != directory.name or not isinstance(identity.get("source_sha256"), str):
                    raise ValueError
            except (ValueError, OSError, AttributeError):
                plan.keep(directory, "unrecognized preview")
                continue
            if identity["source_sha256"] not in references:
                plan.remove_tree(directory, "previews")


# Entry points --------------------------------------------------------------------------------------

def plan_takes(bundle, config):
    """(reused, resumed) take directories of the next render, from the current plan and speech cache."""
    if config.get("remote_chapters", {}).get("url"):
        raise ValueError("Cleaning clips needs this book's own render plan; it cannot tell which takes a "
                         "remote chapter service will reuse. Use --only voices,chapters")
    if config["engine"]["type"] != "drama3_mlx":
        raise ValueError("Cleaning clips supports the built-in Drama/Fish engines only; use --only voices,chapters")
    from .cache_report import cache_report
    from .pool import create_render_engine
    engine = create_render_engine(config)
    identity = (engine.get_identity(cache_fingerprints=False)
                if hasattr(engine, "get_identity") else engine.identity)
    reused, resumed = set(), set()
    try:
        cache_report(bundle, config, identity, multi_speaker=engine.capabilities.get("multi_speaker", False),
                     max_examples=0, needed=reused, resumed=resumed)
    except ValueError as exc:
        raise ValueError(f"Cleaning clips needs a render plan, and this book's plan has problems: {exc}") from exc
    return reused, resumed


def prune_book(bundle, config, *, groups=GROUPS, dry_run=False, confirm=None, needed_takes=None):
    """Remove what a book no longer needs; returns the report (``deleted`` says whether it did).

    Holds the render-work, book audio and voice store locks throughout, so a
    render, assembly or casting session on this book refuses to start, and
    this refuses while one runs. ``confirm(report)`` sees the full dry-run
    report and returns True to delete; without it, deletion follows ``dry_run``.
    """
    from .book_context import resolve_book_config, voice_store
    from .run_locks import acquire_directory_lock
    from .workflow import RenderBlocked, _lock_book_audio
    groups = tuple(group for group in GROUPS if group in groups)
    config = resolve_book_config(bundle, config)
    book = Path(bundle["directory"]).resolve()
    work = Path(config["render"]["work_dir"])
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise RenderBlocked("Cleanup work directory must not use symlinks")
    store = voice_store(bundle, config)
    locks = []
    busy = ("Another render, assembly, casting session or cleanup is using this book; "
            "retry after it finishes: {}")
    try:
        if work.is_dir():
            try:
                locks.append(acquire_directory_lock(work / ".render.lock"))
            except FileExistsError as exc:
                raise RenderBlocked(busy.format(work / ".render.lock")) from exc
        locks.append(_lock_book_audio(bundle))
        if store.directory.is_dir():
            try:
                locks.append(acquire_directory_lock(store.directory / ".lock"))
            except FileExistsError as exc:
                raise RenderBlocked(busy.format(store.directory / ".lock")) from exc
        plan = _Removal(book)
        if work.is_dir():
            plan.group = "chapters"
            versions = _chapter_versions(work, plan)
            checkpoints = _checkpoints(work, plan)
            kept = _kept_chapters(versions, checkpoints) if "chapters" in groups else set(versions)
            if "chapters" in groups:
                _prune_chapters(work, plan, versions, kept)
        if "voices" in groups:
            plan.group = "voices"
            _prune_voices(bundle, store, work, plan)
        if "clips" in groups and work.is_dir():
            plan.group = "clips"
            tokens = set()
            for name in kept:
                tokens |= _tokens(versions[name]["path"] / "chapter.json")
            for path, _ in checkpoints:
                tokens |= _tokens(path)
            _prune_clips(work, plan, (needed_takes or plan_takes)(bundle, config), tokens)
        report = plan.report(dry_run=True, groups=groups)
        delete = (not dry_run and bool(plan.files)
                  and (confirm is None or confirm(dict(report, book=str(book)))))
        if delete:
            report = plan.report(dry_run=False, groups=groups)
        return dict(report, book=str(book), dry_run=dry_run, deleted=delete)
    finally:
        for lock in reversed(locks):
            lock.rmdir()


def _size(count):
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.0f} {unit}" if unit == "bytes" else f"{count:.1f} {unit}"
        count /= 1024


def summary_lines(report):
    lines = [f"Book: {report['book']}"]
    for group, entry in report["groups"].items():
        parts = ", ".join(f"{_LABELS.get(group + '.' + name, name)} {counts['files']}"
                          for name, counts in sorted(entry["categories"].items()))
        lines.append(f"  {group:<9}{entry['files']:>7} file(s) {_size(entry['bytes']):>11}"
                     + (f"  ({parts})" if parts else ""))
    lines.append(f"  Total {report['files_removed']} file(s), {_size(report['bytes_removed'])}; "
                 f"about {_size(report['estimated_bytes_freed'])} of disk space.")
    if report["retained"]:
        lines.append(f"  Kept {len(report['retained'])} item(s) it does not recognize or must keep "
                     "(--json lists them).")
    return lines


def prune_command(args, config):
    import json
    import sys
    from .book_context import resolve_book_config
    from .bundle import load_bundle
    books = args.bundle
    if len(books) > 1 and (args.work_dir or args.output or args.epub):
        raise ValueError("--work-dir, --output and --epub apply to one book; name a single book with them")
    groups = parse_groups(args.only)
    asking = not args.yes and not args.dry_run
    interactive = sys.stdin.isatty()
    # With --json, standard output carries only the reports; people read standard error.
    say = sys.stderr if args.json else sys.stdout
    failed = 0
    for path in books:
        try:
            bundle = load_bundle(path)
            book_config = resolve_book_config(bundle, config, output=args.output,
                                              work_dir=args.work_dir, epub=args.epub)

            def confirm(report):
                for line in summary_lines(report):
                    print(line, file=say)
                if not interactive:
                    print("  Not deleted: no terminal to confirm on. Rerun with --yes to delete.", file=say)
                    return False
                print("Delete these files? [y/N] ", end="", file=say, flush=True)
                answer = sys.stdin.readline().strip().lower()
                return answer in {"y", "yes"}

            report = prune_book(bundle, book_config, groups=groups,
                                dry_run=args.dry_run, confirm=confirm if asking else None)
        except (ValueError, OSError) as exc:
            failed += 1
            print(f"{path}: {exc}", file=sys.stderr)
            continue
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        if not asking or not report["files_removed"]:
            for line in summary_lines(report):
                print(line, file=say)
        if report["deleted"]:
            print(f"  Deleted {report['files_removed']} file(s).", file=say)
        elif args.dry_run:
            print("  Dry run: nothing was deleted.", file=say)
        elif not report["files_removed"]:
            print("  Nothing to delete.", file=say)
        else:
            print("  Nothing was deleted.", file=say)
    return 2 if failed else 0  # The renderer's error status.
