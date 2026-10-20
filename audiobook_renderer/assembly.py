"""Lossless chapter assembly, exact frame timing, and audiobook containers.

The renderer must verify speech before passing clips here. Encoding never uses
the lossy chapter files as the input to the final audiobook.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Sequence
import wave
import zipfile

from epub_audiobook.epub import (
    MAX_ARCHIVE_BYTES, _Archive, _parse_xml, _resolve_href, _tag,
)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(arguments: list[str], *, timeout: int = 3600) -> str:
    try:
        result = subprocess.run(arguments, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Unable to run {Path(arguments[0]).name}: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise ValueError(f"{Path(arguments[0]).name} failed: {detail.strip()}")
    return result.stdout.decode("utf-8", errors="strict")


def _rate(sample_rate: int) -> int:
    if type(sample_rate) is not int or not 8000 <= sample_rate <= 192000:
        raise ValueError("sample_rate must be an integer from 8000 to 192000")
    return sample_rate


def _pause_frames(milliseconds: float, sample_rate: int) -> int:
    if (isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float))
            or not math.isfinite(milliseconds) or not 0 <= milliseconds <= 5000):
        raise ValueError("Pauses must be finite milliseconds between 0 and 5000")
    return round(milliseconds * sample_rate / 1000)


def _wave_info(path: Path, sample_rate: int | None = None) -> dict:
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            if (audio.getcomptype() != "NONE" or audio.getsampwidth() != 2
                    or audio.getnchannels() != 1):
                raise ValueError(f"Expected uncompressed mono PCM16 WAV: {path}")
            if sample_rate is not None and rate != sample_rate:
                raise ValueError(f"WAV sample rate {rate} differs from {sample_rate}: {path}")
            _rate(rate)
            return {"sample_rate": rate, "channels": 1, "frames": audio.getnframes()}
    except (OSError, wave.Error, EOFError) as exc:
        raise ValueError(f"Invalid PCM WAV {path}: {exc}") from exc


def normalize_clip(input_path: str | Path, output_path: str | Path, *,
                   sample_rate: int = 44100, ffmpeg: str = "ffmpeg") -> dict:
    """Decode one engine result into the renderer's common PCM format.

    Silence/content checks must run on this normalized result, before assembly.
    Existing files are never overwritten.
    """
    _rate(sample_rate)
    source, target = Path(input_path).resolve(), Path(output_path)
    if target.exists() or target.is_symlink():
        raise ValueError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".normalize-", dir=target.parent) as temporary:
        staged = Path(temporary) / "audio.wav"
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
              "-protocol_whitelist", "file,pipe", "-i", str(source),
              "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar",
              str(sample_rate), "-c:a", "pcm_s16le", "-f", "wav", str(staged)],
             timeout=600)
        info = _wave_info(staged, sample_rate)
        if info["frames"] == 0:
            raise ValueError("TTS result contains no audio frames")
        # Link publication is exclusive, including when another process wins.
        try:
            target.hardlink_to(staged)
        except OSError as exc:
            raise ValueError(f"Unable to publish normalized audio {target}: {exc}") from exc
    return dict(info, wav_path=str(target))


def compose_chapter(clips: Sequence[dict], output_wav: str | Path, *,
                    sample_rate: int = 44100, paragraph_pause_ms: float = 250,
                    blank_pause_ms: float = 500, bracket_pause_ms: float = 120) -> dict:
    """Join verified PCM clips, retaining exact source and sample-frame ranges.

    ``wav_path=None`` is reserved for deliberate no-speech controls (for
    example punctuation only). It is never a synthesis-error fallback.
    Adjacent paragraph/blank-line pauses use the larger cue, not their sum.
    """
    _rate(sample_rate)
    paragraph_frames = _pause_frames(paragraph_pause_ms, sample_rate)
    blank_frames = _pause_frames(blank_pause_ms, sample_rate)
    bracket_frames = _pause_frames(bracket_pause_ms, sample_rate)
    target = Path(output_wav)
    if target.exists() or target.is_symlink():
        raise ValueError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    timings, cursor, pending_pause = [], 0, 0
    with tempfile.TemporaryDirectory(prefix=".compose-", dir=target.parent) as temporary:
        staged = Path(temporary) / "chapter.wav"
        with wave.open(str(staged), "wb") as destination:
            destination.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))

            def silence(frames: int) -> None:
                nonlocal cursor
                remaining = frames
                while remaining:
                    count = min(remaining, 65536)
                    destination.writeframesraw(b"\x00\x00" * count)
                    remaining -= count
                cursor += frames

            for clip in clips:
                row = clip["segment"]
                spoken = row.get("spoken_text", row["text"])
                start, end = clip["spoken_start"], clip["spoken_end"]
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start <= end <= len(spoken)
                        or clip["spoken_text"] != spoken[start:end]):
                    raise ValueError(f"Invalid spoken-text range for {clip['chunk_id']}")
                ranges = clip.get("source_ranges", [])
                if not isinstance(ranges, list) or any(
                    not isinstance(item, dict)
                    or type(item.get("start")) is not int
                    or type(item.get("end")) is not int
                    or not row["start"] <= item["start"] <= item["end"] <= row["end"]
                    for item in ranges
                ):
                    raise ValueError(f"Invalid source ranges for {clip['chunk_id']}")
                pause_before = (blank_frames if start == 0
                                and row.get("break_before") == "blank_paragraph" else 0)
                performance = row.get("performance", {})
                if (start == 0 and performance.get("kind") == "bracketed"
                        and performance.get("enter") is True):
                    pause_before = max(pause_before, bracket_frames)
                pending_pause = max(pending_pause, pause_before)
                if clip.get("wav_path") is not None:
                    silence(pending_pause)
                    pending_pause = 0
                first_frame = cursor
                if clip.get("wav_path") is not None:
                    source = Path(clip["wav_path"])
                    info = _wave_info(source, sample_rate)
                    if not info["frames"]:
                        raise ValueError(f"Verified clip has no audio frames: {source}")
                    count = 0
                    with wave.open(str(source), "rb") as audio:
                        while chunk := audio.readframes(65536):
                            if len(chunk) % 2:
                                raise ValueError(f"Truncated PCM frame: {source}")
                            destination.writeframesraw(chunk)
                            count += len(chunk) // 2
                    if count != info["frames"]:
                        raise ValueError(f"Truncated PCM audio: {source}")
                    cursor += count
                timing = {
                    "chunk_id": clip["chunk_id"], "segment_id": row["id"],
                    "block_id": row["block_id"], "voice_id": row.get("voice_id"),
                    "role_id": row.get("role_id"), "source_text": row["text"],
                    "source_start": row["start"], "source_end": row["end"],
                    "source_ranges": deepcopy(ranges),
                    "spoken_text": clip["spoken_text"],
                    "spoken_start": start, "spoken_end": end,
                    "chapter_start_frame": first_frame, "chapter_end_frame": cursor,
                    "no_speech": clip.get("wav_path") is None,
                }
                if performance:
                    timing["performance"] = deepcopy(performance)
                timings.append(timing)
                if end == len(spoken) and row.get("break_after") == "paragraph":
                    pending_pause = max(pending_pause, paragraph_frames)
                if (end == len(spoken) and performance.get("kind") == "bracketed"
                        and performance.get("exit") is True):
                    pending_pause = max(pending_pause, bracket_frames)
            # Empty scripts deliberately remain frame-empty and are skipped
            # during packaging. A real chapter keeps its final paragraph pause.
            if cursor:
                silence(pending_pause)
        try:
            target.hardlink_to(staged)
        except OSError as exc:
            raise ValueError(f"Unable to publish chapter PCM {target}: {exc}") from exc
    return {"wav_path": str(target), "sample_rate": sample_rate, "channels": 1,
            "frames": cursor, "segments": timings}


def extract_cover(epub: str | Path, output_dir: str | Path,
                  expected_book_id: str) -> Path | None:
    """Copy the EPUB's declared JPEG/PNG cover without changing its artwork."""
    path, destination = Path(epub), Path(output_dir)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("EPUB archive exceeds safety size limit")
    if _hash(path) != expected_book_id:
        raise ValueError("Cover EPUB does not match the analyzed book ID")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid cover EPUB: {exc}") from exc
    with archive:
        resources = _Archive(archive)
        encrypted = set()
        if "META-INF/encryption.xml" in resources.entries:
            root = _parse_xml(resources.read("META-INF/encryption.xml"), "encryption.xml")
            encrypted = {_resolve_href("", node.get("URI", "")) for node in root.iter()
                         if _tag(node) == "cipherreference"}

        def xml(name: str):
            if name in encrypted:
                raise ValueError(f"Encrypted cover resource is unsupported: {name}")
            return _parse_xml(resources.read(name), name)

        container = xml("META-INF/container.xml")
        rootfiles = [node for node in container.iter()
                     if _tag(node) == "rootfile"
                     and node.get("media-type") == "application/oebps-package+xml"]
        if not rootfiles:
            raise ValueError("EPUB container has no OPF package")
        package_path = _resolve_href("", rootfiles[0].get("full-path", ""))
        package = xml(package_path)
        manifest = [node for node in package if _tag(node) == "manifest"]
        metadata = [node for node in package if _tag(node) == "metadata"]
        if len(manifest) != 1 or len(metadata) != 1:
            raise ValueError("EPUB must have exactly one manifest and metadata section")
        items = [node for node in manifest[0] if _tag(node) == "item"]
        modern = [node for node in items
                  if "cover-image" in node.get("properties", "").split()]
        identifiers = {node.get("content") for node in metadata[0]
                       if _tag(node) == "meta" and node.get("name") == "cover"}
        candidates = modern or [node for node in items if node.get("id") in identifiers]
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError("EPUB declares multiple cover images")
        item = candidates[0]
        kind = item.get("media-type")
        if kind not in {"image/jpeg", "image/png"}:
            # SVG and other formats need deliberate rendering support.
            return None
        href = _resolve_href(package_path, item.get("href", ""))
        if href in encrypted:
            raise ValueError("Encrypted EPUB cover is unsupported")
        data = resources.read(href)
        if kind == "image/jpeg" and not data.startswith(b"\xff\xd8\xff"):
            raise ValueError("EPUB JPEG cover has an invalid signature")
        if kind == "image/png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("EPUB PNG cover has an invalid signature")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / ("cover.jpg" if kind == "image/jpeg" else "cover.png")
        with target.open("xb") as stream:
            stream.write(data)
        return target


def _metadata(value: str) -> str:
    if "\x00" in value:
        raise ValueError("NUL is not allowed in audio metadata")
    return re.sub(r"([\\=;#\r\n])", r"\\\1", value)


def _concat_path(path: Path) -> str:
    value = str(path.resolve())
    if any(ord(character) < 32 for character in value):
        raise ValueError("Control characters are not allowed in audio paths")
    return "'" + value.replace("'", "'\\''") + "'"


def probe_audio(path: str | Path, *, ffprobe: str = "ffprobe") -> dict:
    """Return ffprobe's inspected streams, chapters and container metadata."""
    raw = _run([ffprobe, "-v", "error", "-show_streams", "-show_chapters",
                "-show_format", "-of", "json", str(Path(path).resolve())], timeout=120)
    try:
        return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonfinite ffprobe value: {value}")))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid ffprobe output") from exc


def _check_encoded(path: Path, frames: int, sample_rate: int, *, ffprobe: str,
                   chapters: Sequence[dict] | None = None, cover: bool = False) -> dict:
    inspection = probe_audio(path, ffprobe=ffprobe)
    audio = [stream for stream in inspection.get("streams", [])
             if stream.get("codec_type") == "audio"]
    if len(audio) != 1 or int(audio[0].get("sample_rate", 0)) != sample_rate:
        raise ValueError(f"Encoded audio has an unexpected audio stream: {path}")
    duration = float(audio[0].get("duration", inspection.get("format", {}).get("duration", "nan")))
    if not math.isfinite(duration) or abs(duration - frames / sample_rate) > max(.12, 4096 / sample_rate):
        raise ValueError(f"Encoded duration differs from PCM timeline: {path}")
    if cover and not any(stream.get("disposition", {}).get("attached_pic") == 1
                         for stream in inspection.get("streams", [])):
        raise ValueError(f"Encoded audiobook is missing its cover: {path}")
    if chapters is not None:
        actual = inspection.get("chapters", [])
        if len(actual) != len(chapters):
            raise ValueError("Encoded audiobook chapter count is incorrect")
        for expected, found in zip(chapters, actual):
            if found.get("tags", {}).get("title") != expected["title"]:
                raise ValueError("Encoded audiobook chapter title is incorrect")
            for edge in ("start", "end"):
                seconds = float(found.get(f"{edge}_time", "nan"))
                if (not math.isfinite(seconds)
                        or abs(seconds - expected[f"{edge}_frame"] / sample_rate) > .002):
                    raise ValueError("Encoded audiobook chapter timing is incorrect")
    return inspection


def package_audio(chapters: Sequence[dict], output_dir: str | Path, *,
                  book: dict, cover_path: str | Path | None = None,
                  chapter_format: str = "m4a", aac_bitrate: str = "128k",
                  mp3_bitrate: str = "128k", ffmpeg: str = "ffmpeg",
                  ffprobe: str = "ffprobe", source_fingerprint: str | None = None) -> dict:
    """Encode chapter files and a single AAC M4B from the same lossless WAVs.

    The caller owns the staging directory and publishes it only after success.
    Empty chapters are explicit skipped records. A book with no audio fails.
    """
    if chapter_format not in {"m4a", "mp3"}:
        raise ValueError("chapter_format must be m4a or mp3")
    for bitrate in (aac_bitrate, mp3_bitrate):
        if not isinstance(bitrate, str) or not re.fullmatch(r"[1-9][0-9]{0,2}k", bitrate):
            raise ValueError("Audio bitrate must be a positive value such as 128k")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("chapters", "book.m4b", "sync.json", "audio-manifest.json"):
        if (target / name).exists() or (target / name).is_symlink():
            raise ValueError(f"Audio package output already exists: {target / name}")
    if not chapters:
        raise ValueError("Audiobook has no chapters")
    sample_rate = _rate(chapters[0]["sample_rate"])
    records, audio_chapters, timings, cursor = [], [], [], 0
    for ordinal, chapter in enumerate(chapters, 1):
        info = _wave_info(Path(chapter["wav_path"]), sample_rate)
        if info["frames"] != chapter["frames"] or chapter["sample_rate"] != sample_rate:
            raise ValueError("Chapter PCM no longer matches its synchronization data")
        record = {
            "id": chapter["id"], "title": chapter["title"],
            "script_path": chapter.get("script_path"),
            "ordinal": ordinal, "start_frame": cursor,
            "end_frame": cursor + info["frames"], "frames": info["frames"],
            "skipped": info["frames"] == 0,
            "path": f"chapters/chapter-{ordinal:04d}.{chapter_format}" if info["frames"] else None,
        }
        records.append(record)
        for segment in chapter["segments"]:
            start, end = segment["chapter_start_frame"], segment["chapter_end_frame"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start <= end <= info["frames"]):
                raise ValueError("Chapter synchronization frame range is invalid")
            timings.append(dict(deepcopy(segment), chapter_ordinal=ordinal,
                                chapter_id=chapter["id"],
                                book_start_frame=cursor + start, book_end_frame=cursor + end))
        if info["frames"]:
            audio_chapters.append((chapter, record))
        cursor += info["frames"]
    if cursor == 0:
        raise ValueError("Audiobook has no speech audio to package")
    chapter_dir = target / "chapters"
    chapter_dir.mkdir()
    cover = Path(cover_path).resolve() if cover_path is not None else None
    title = str(book.get("title", "Audiobook"))
    authors = "; ".join(book.get("authors", []))
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    for chapter, record in audio_chapters:
        codec = (["-c:a", "aac", "-b:a", aac_bitrate, "-movflags", "+faststart"]
                 if chapter_format == "m4a"
                 else ["-c:a", "libmp3lame", "-b:a", mp3_bitrate])
        output = target / record["path"]
        _run(base + ["-i", str(Path(chapter["wav_path"]).resolve()), "-map", "0:a:0"]
             + codec + ["-metadata", f"title={record['title']}",
                        "-metadata", f"album={title}", "-metadata", f"artist={authors}",
                        "-metadata", f"track={record['ordinal']}/{len(records)}", str(output)])
        _check_encoded(output, record["frames"], sample_rate, ffprobe=ffprobe)
        record["sha256"] = _hash(output)
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=target) as temporary:
        scratch = Path(temporary)
        concat = scratch / "chapters.ffconcat"
        concat.write_text("ffconcat version 1.0\n" + "".join(
            f"file {_concat_path(Path(chapter['wav_path']))}\n"
            f"duration {chapter['frames'] / sample_rate:.12f}\n"
            for chapter, _ in audio_chapters), encoding="utf-8")
        metadata = scratch / "book.ffmetadata"
        metadata.write_text(
            f";FFMETADATA1\ntitle={_metadata(title)}\nartist={_metadata(authors)}\n"
            "genre=Audiobook\n" + "".join(
                f"[CHAPTER]\nTIMEBASE=1/{sample_rate}\n"
                f"START={record['start_frame']}\nEND={record['end_frame']}\n"
                f"title={_metadata(record['title'])}\n" for _, record in audio_chapters),
            encoding="utf-8")
        command = base + ["-f", "concat", "-safe", "0", "-protocol_whitelist", "file,pipe",
                          "-i", str(concat), "-f", "ffmetadata", "-i", str(metadata)]
        if cover is not None:
            command += ["-i", str(cover)]
        command += ["-map", "0:a:0", "-map_metadata", "1", "-map_chapters", "1",
                    "-c:a", "aac", "-b:a", aac_bitrate]
        if cover is not None:
            command += ["-map", "2:v:0", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
        command += ["-movflags", "+faststart", "-f", "ipod", str(target / "book.m4b")]
        _run(command)
    inspection = _check_encoded(target / "book.m4b", cursor, sample_rate,
                                chapters=[record for _, record in audio_chapters],
                                cover=cover is not None, ffprobe=ffprobe)
    sync = {
        "format": "audiobook-renderer-sync", "version": 1, "book_id": book.get("id"),
        "source_fingerprint": source_fingerprint, "sample_rate": sample_rate,
        "timebase": f"1/{sample_rate}", "range_end": "exclusive",
        "timing_basis": "PCM sample frames; segment ranges exclude inserted pauses",
        "precision": "chunk", "audio_file": "book.m4b",
        "audio_sha256": _hash(target / "book.m4b"),
        "playback_note": "Use codec delay/padding-aware playback. Chapter tags may round to milliseconds.",
        "chapters": records, "segments": timings,
    }
    (target / "sync.json").write_text(json.dumps(sync, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    manifest = {
        "format": "audiobook-renderer-audio", "version": 1, "book": deepcopy(book),
        "source_fingerprint": source_fingerprint, "sample_rate": sample_rate,
        "channels": 1, "frames": cursor, "duration_seconds": cursor / sample_rate,
        "chapter_format": chapter_format, "chapters": records,
        "audiobook": {"path": "book.m4b", "sha256": sync["audio_sha256"], "codec": "aac",
                      "cover_embedded": cover is not None},
        "sync": {"path": "sync.json", "sha256": _hash(target / "sync.json")},
        "encoded_audio_duration_seconds": float(next(
            stream for stream in inspection["streams"] if stream["codec_type"] == "audio")["duration"]),
    }
    (target / "audio-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
