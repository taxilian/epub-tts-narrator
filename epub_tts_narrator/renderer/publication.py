"""Replace only intact, renderer-owned books after their replacement is ready."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from .common import digest, file_hash, read_json, regular_file


def published_book(directory, book_id):
    """Validate ownership and the entire published inventory before any replacement."""
    directory = Path(directory)
    if not directory.exists() and not directory.is_symlink():
        return None
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("output must be a regular directory")
        record = read_json(regular_file(directory / "render.json"))
        if (not isinstance(record, dict) or record.get("format") != "audiobook-render"
                or record.get("version") != 1 or record.get("status") != "complete"):
            raise ValueError("not a completed renderer output")
        files = record.get("files_sha256")
        required = {"source.json", "cast.json", "verification.json", "audio-manifest.json",
                    "sync.json", "book.m4b"}
        if not isinstance(files, dict) or not required.issubset(files):
            raise ValueError("published file inventory is incomplete")
        expected_dirs = set()
        for name, checksum in files.items():
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or path.as_posix() != name
                    or ".." in path.parts or "\\" in name or name == "render.json"
                    or not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)):
                raise ValueError("invalid published file inventory")
            expected_dirs.update(parent.as_posix() for parent in path.parents if parent != PurePosixPath("."))
        found = set()
        for path in directory.rglob("*"):
            name = path.relative_to(directory).as_posix()
            if path.is_symlink():
                if name.startswith("chapter-links/"):
                    continue
                raise ValueError(f"published output contains a symlink: {name}")
            if path.is_dir():
                if name != "chapter-links" and name not in expected_dirs:
                    raise ValueError(f"output contains an untracked directory: {name}")
            elif path.is_file():
                if name.startswith("chapter-links/"):
                    raise ValueError(f"chapter-links entries must be symlinks: {name}")
                found.add(name)
            else:
                raise ValueError(f"output contains a nonregular file: {name}")
        if found != set(files) | {"render.json"}:
            raise ValueError("published files are missing or untracked files were added")
        for name, checksum in files.items():
            if file_hash(regular_file(directory / name)) != checksum:
                raise ValueError(f"published file was modified: {name}")
        source = read_json(regular_file(directory / "source.json"))
        audio = record.get("audio", {})
        if (not isinstance(source, dict) or not isinstance(source.get("book"), dict)
                or source["book"].get("id") != book_id
                or not isinstance(audio, dict) or not isinstance(audio.get("book"), dict)
                or audio["book"].get("id") != book_id):
            raise ValueError("output belongs to a different book")
        chapters = audio.get("chapters")
        if (type(record.get("partial_book")) is not bool or not isinstance(chapters, list)
                or not chapters or any(not isinstance(chapter, dict)
                    or not isinstance(chapter.get("id"), str) or not chapter["id"] for chapter in chapters)
                or len({chapter["id"] for chapter in chapters}) != len(chapters)):
            raise ValueError("invalid published chapter metadata")
        if audio != read_json(regular_file(directory / "audio-manifest.json")):
            raise ValueError("published audio manifest differs from render.json")
        return record
    except (ValueError, OSError, TypeError) as exc:
        raise ValueError(f"Output already exists and cannot be safely updated: {directory}: {exc}. "
                         "Restore the published files or choose another output directory.") from exc


def check_selection(previous, chapter_ids, *, partial):
    """An explicitly partial render must not silently replace a larger audiobook."""
    if previous is not None and partial:
        saved = [chapter.get("id") for chapter in previous["audio"]["chapters"]]
        if saved != list(chapter_ids):
            raise ValueError("The existing audiobook covers different chapters. Use --chapters-only "
                             "to update selected chapters, then render or assemble the whole book; "
                             "or choose a separate output directory for this partial audiobook.")


_UNSPECIFIED = object()


def publish(stage, destination, *, expected=_UNSPECIFIED):
    """Publish completed staging; roll back the previous book on a failed rename."""
    stage, destination = Path(stage), Path(destination)
    reservation = destination.parent / ("." + destination.name + ".publish.lock")
    try:
        reservation.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"Another process may be publishing this output. If it was forcibly terminated, "
            f"confirm it has stopped before removing the stale lock: {reservation}") from exc
    backup = None
    try:
        if destination.exists() or destination.is_symlink():
            incoming = read_json(regular_file(stage / "render.json"))
            book_id = incoming["audio"]["book"]["id"]
            previous = published_book(destination, book_id)
            check_selection(previous, (c["id"] for c in incoming["audio"]["chapters"]),
                            partial=incoming["partial_book"])
            if expected is not _UNSPECIFIED and digest(previous) != expected:
                raise ValueError("Published output changed during rendering; rerun to use its current state")
            backup = Path(tempfile.mkdtemp(prefix="." + destination.name + ".previous-",
                                           dir=destination.parent))
            backup.rmdir()
            os.rename(destination, backup)
        elif expected is not _UNSPECIFIED and expected is not None:
            raise ValueError("Published output was removed during rendering; rerun the command")
        try:
            os.rename(stage, destination)
        except BaseException:
            if backup is not None:
                try:
                    os.rename(backup, destination)
                except OSError as exc:
                    raise ValueError(f"Publication failed and rollback failed. The previous complete "
                                     f"audiobook is saved at {backup}; restore it to {destination}.") from exc
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        reservation.rmdir()


def publication_key(bundle, config, plan, voices, chapters, acceptance, engine, verifier):
    """Include all package inputs, but not transient paths to cached audio."""
    path_fields = {"wav_path", "encoded_path", "cache_directory"}
    render = config["render"]
    return digest({
        "version": 1, "bundle_hash": bundle["bundle_hash"],
        "source_fingerprint": plan["source_fingerprint"], "partial_book": plan["partial_book"],
        "chapters": [{k: v for k, v in chapter.items() if k not in path_fields} for chapter in chapters],
        "voice_assets": {key: value["asset_hash"] for key, value in voices.items()},
        "acceptance": acceptance, "engine": engine, "verifier": verifier,
        "quality": config["quality"], "review_items": plan["review_items"],
        "allow_review": render["allow_review"],
        "cover_epub": file_hash(render["epub"]) if render["epub"] else None,
        "encoding": {key: render.get(key) for key in
                     ("chapter_format", "aac_bitrate", "mp3_bitrate", "ffmpeg", "ffprobe")},
    })
