"""Repair a published book's artwork/tags without synthesizing or encoding audio."""

from copy import deepcopy
from pathlib import Path
import re
import shutil
import tempfile

from .assembly import _check_encoded, _run
from .book_metadata import check_book_tags, check_publication_source, metadata_arguments, publication_assets
from .book_output import resolve_audiobook_filename
from .common import digest, file_hash, read_json, regular_file, write_json
from .publication import published_book, publish


def _audio_payload_hash(path, ffmpeg):
    """Hash encoded audio packets, excluding changed container tags/artwork."""
    result = _run([ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
                   "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-"]).strip()
    if re.fullmatch(r"SHA256=[0-9a-f]{64}", result) is None:
        raise ValueError("Could not verify the encoded audio checksum; previous audiobook preserved.")
    return result


def refresh_metadata(bundle, config, *, notify=print):
    from .book_context import resolve_book_config
    from .workflow import _check_analysis_update, _lock_book_audio, _publish_chapter_link
    from epub_tts_narrator.source_epub import remember_source_epub

    config = resolve_book_config(bundle, config)
    config["render"]["require_cover"] = True
    check_publication_source(bundle, config)
    lock = _lock_book_audio(bundle)
    try:
        _check_analysis_update(bundle)
        output = Path(config["render"]["output_dir"])
        previous = published_book(output, bundle["source"]["book"]["id"])
        if previous is None:
            raise ValueError("No completed audiobook exists at this output; render or assemble it first.")
        expected = digest(previous)
        filename = resolve_audiobook_filename(previous_audio=previous["audio"], output_dir=output)
        book_path = output / filename
        ffmpeg, ffprobe = config["render"]["ffmpeg"], config["render"]["ffprobe"]
        with tempfile.TemporaryDirectory(prefix=".audiobook-metadata-", dir=output.parent) as temporary:
            stage = Path(temporary) / "ready"
            stage.mkdir()
            staged_book = stage / filename
            # Keep chapters, verification, and all source timing data byte-exact.
            # The previous published directory stays intact until every check passes.
            replaced = {filename, "sync.json", "audio-manifest.json"}
            for name in previous["files_sha256"]:
                if name in replaced or name.startswith("artwork/"):
                    continue
                source = output / name
                if source.is_symlink() or not source.is_file():
                    continue
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            book, cover, artwork = publication_assets(bundle, config, stage / "artwork", notify=notify)
            notify(f"Updating {filename} cover and metadata; copying encoded audio unchanged.")
            _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                  "-i", str(book_path), "-i", str(cover),
                  "-map", "0:a:0", "-map", "1:v:0", "-map_metadata", "0", "-map_chapters", "0",
                  "-c:a", "copy", "-c:v", "copy", "-disposition:v:0", "attached_pic",
                  *metadata_arguments(book), "-movflags", "+faststart", "-f", "ipod", str(staged_book)])
            audio = deepcopy(previous["audio"])
            inspection = _check_encoded(
                staged_book, audio["frames"], audio["sample_rate"], ffprobe=ffprobe,
                chapters=[c for c in audio["chapters"] if not c["skipped"]], cover=True)
            check_book_tags(inspection, book)
            if _audio_payload_hash(book_path, ffmpeg) != _audio_payload_hash(staged_book, ffmpeg):
                raise ValueError("Metadata refresh changed the encoded audio; previous audiobook preserved.")
            checksum = file_hash(staged_book)
            try:
                sync = read_json(regular_file(output / "sync.json"))
            except (OSError, ValueError) as exc:
                raise ValueError("The published sync.json is missing or unreadable; render the book to "
                                 "recreate its timing before refreshing metadata.") from exc
            sync["audio_file"] = filename
            sync["audio_sha256"] = checksum
            write_json(stage / "sync.json", sync)
            audio["book"] = book
            audio["audiobook"].update(path=filename, sha256=checksum, cover_embedded=True, artwork=artwork)
            audio["sync"]["sha256"] = file_hash(stage / "sync.json")
            write_json(stage / "audio-manifest.json", audio)
            result = deepcopy(previous)
            # Rendering can subsequently recompute the normal package identity.
            result.pop("publication_key", None)
            result["audio"] = audio
            result["files_sha256"] = {p.relative_to(stage).as_posix(): file_hash(p)
                                      for p in sorted(stage.rglob("*")) if p.is_file()}
            write_json(stage / "render.json", result)
            publish(stage, output, expected=expected)
        for chapter in audio["chapters"]:
            if not chapter["skipped"] and chapter.get("path") in result["files_sha256"]:
                _publish_chapter_link(config, {
                    "track_number": chapter.get("number", chapter["ordinal"]),
                    "encoded_path": str(output / chapter["path"]),
                }, notify=notify)
        remember_source_epub(bundle["directory"], config["render"]["epub"], book["id"],
                             saved_source=bundle["source"])
        notify(f"Updated audiobook: {book_path} (audio and chapter timing unchanged)")
        return result
    finally:
        lock.rmdir()
