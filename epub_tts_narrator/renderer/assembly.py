"""Lossless chapter assembly, exact frame timing, and audiobook containers.

The renderer must verify speech before passing clips here. Encoding never uses
the lossy chapter files as the input to the final audiobook.
"""

from __future__ import annotations

from copy import deepcopy
from array import array
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence
import wave
import zipfile

from epub_tts_narrator.epub import (
    MAX_ARCHIVE_BYTES, _Archive, _parse_xml, _resolve_href, _tag,
)
from .pacing import audio_edges, punctuation_pause
from .interrupts import run_subprocess


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(arguments: list[str], *, timeout: int = 3600) -> str:
    try:
        result = run_subprocess(arguments, stdin=subprocess.DEVNULL,
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


def mix_speakers(lanes: Sequence[dict], output_wav: str | Path, *, sample_rate=44100) -> dict:
    """Align verified mono lanes at frame zero and average with constant headroom.

    Every lane keeps its original timing. Shorter lanes are padded with zeros;
    no time stretching, automatic gain changes, or mixture ASR is performed.
    """
    _rate(sample_rate)
    if not isinstance(lanes, (list, tuple)) or len(lanes) < 2:
        raise ValueError("A simultaneous mix requires at least two speaker lanes")
    target = Path(output_wav)
    if target.exists() or target.is_symlink():
        raise ValueError(f"Output already exists: {target}")
    info = [_wave_info(Path(lane["wav_path"]), sample_rate) for lane in lanes]
    if any(item["frames"] <= 0 for item in info):
        raise ValueError("A simultaneous speaker lane cannot be empty")
    hashes = [_hash(Path(lane["wav_path"])) for lane in lanes]
    if any(lane.get("audio_sha256") != checksum for lane, checksum in zip(lanes, hashes)):
        raise ValueError("Verified simultaneous speaker audio changed before mixing")
    frames = max(item["frames"] for item in info)
    target.parent.mkdir(parents=True, exist_ok=True)
    peak = 0
    with tempfile.TemporaryDirectory(prefix=".mix-", dir=target.parent) as temporary:
        staged = Path(temporary) / "mix.wav"
        with ExitStack() as handles:
            inputs = [handles.enter_context(wave.open(str(lane["wav_path"]), "rb")) for lane in lanes]
            destination = handles.enter_context(wave.open(str(staged), "wb"))
            destination.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            cursor = 0
            while cursor < frames:
                count = min(32768, frames - cursor)
                totals = [0] * count
                for source, item in zip(inputs, info):
                    expected = max(0, min(count, item["frames"] - cursor))
                    data = source.readframes(expected)
                    if len(data) != expected * 2:
                        raise ValueError("Truncated simultaneous speaker PCM")
                    samples = array("h")
                    samples.frombytes(data)
                    if sys.byteorder != "little":
                        samples.byteswap()
                    for index, value in enumerate(samples):
                        totals[index] += value
                mixed = array("h", (round(value / len(lanes)) for value in totals))
                peak = max(peak, max((abs(value) for value in mixed), default=0))
                if sys.byteorder != "little":
                    mixed.byteswap()
                destination.writeframesraw(mixed.tobytes())
                cursor += count
        for lane, expected in zip(lanes, hashes):
            if _hash(Path(lane["wav_path"])) != expected:
                raise ValueError("Speaker lane changed during mixing")
        if peak == 0:
            raise ValueError("Simultaneous mixture is silent; speaker lanes cancelled each other")
        if _wave_info(staged, sample_rate)["frames"] != frames:
            raise ValueError("Simultaneous mixture has an invalid PCM timeline")
        target.hardlink_to(staged)
    return {"wav_path": str(target), "sample_rate": sample_rate, "channels": 1,
            "frames": frames, "method": "aligned_fixed_average", "weight": 1 / len(lanes),
            "peak_amplitude": peak, "audio_sha256": _hash(target)}


def compose_chapter(clips: Sequence[dict], output_wav: str | Path, *,
                    sample_rate: int = 44100, paragraph_pause_ms: float = 400,
                    blank_pause_ms: float = 600, bracket_pause_ms: float = 120,
                    normalize_pauses: bool = True, sentence_pause_ms: float = 260,
                    clause_pause_ms: float = 120) -> dict:
    """Join verified PCM clips, retaining exact source and sample-frame ranges.

    ``wav_path=None`` is reserved for deliberate no-speech controls (for
    example punctuation only). It is never a synthesis-error fallback.
    Adjacent paragraph/blank-line pauses use the larger cue, not their sum.
    """
    _rate(sample_rate)
    paragraph_frames = _pause_frames(paragraph_pause_ms, sample_rate)
    blank_frames = _pause_frames(blank_pause_ms, sample_rate)
    bracket_frames = _pause_frames(bracket_pause_ms, sample_rate)
    sentence_frames = _pause_frames(sentence_pause_ms, sample_rate)
    clause_frames = _pause_frames(clause_pause_ms, sample_rate)
    if type(normalize_pauses) is not bool:
        raise ValueError("normalize_pauses must be boolean")
    target = Path(output_wav)
    if target.exists() or target.is_symlink():
        raise ValueError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    timings, cursor, pending_pause, previous_tail = [], 0, 0, 0
    pending_punctuation = 0
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
                edges = None
                if clip.get("wav_path") is not None:
                    source = Path(clip["wav_path"])
                    info = _wave_info(source, sample_rate)
                    if not info["frames"]:
                        raise ValueError(f"Verified clip has no audio frames: {source}")
                    # Unison lanes retain their original common origin and timing.
                    edges = (audio_edges(source, trim="speaker_lanes" not in clip)
                             if normalize_pauses else
                             dict(start=0, end=info["frames"], leading=0, trailing=0))
                    silence(max(0, max(pending_pause, pending_punctuation)
                                - previous_tail - edges["leading"]))
                    pending_pause = 0
                    pending_punctuation = 0
                first_frame = cursor
                if clip.get("wav_path") is not None:
                    count = 0
                    with wave.open(str(source), "rb") as audio:
                        audio.setpos(edges["start"])
                        remaining = edges["end"] - edges["start"]
                        while remaining:
                            requested = min(65536, remaining)
                            chunk = audio.readframes(requested)
                            if len(chunk) != requested * 2:
                                raise ValueError(f"Truncated PCM frame: {source}")
                            destination.writeframesraw(chunk)
                            count += len(chunk) // 2
                            remaining -= requested
                    cursor += count
                    previous_tail = edges["trailing"]
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
                if edges is not None:
                    timing.update(source_audio_start_frame=edges["start"],
                                  source_audio_end_frame=edges["end"])
                if "paragraph_turns" in clip:
                    # A single continuous performance has paragraph timing. Do not
                    # invent word/speaker timestamps from unreliable ASR alignment.
                    timing["precision"] = "paragraph"
                    timing["source_segments"] = [{
                        "chunk_id": turn["chunk_id"], "segment_id": turn["segment"]["id"],
                        "block_id": turn["segment"]["block_id"],
                        "voice_id": turn["segment"]["voice_id"],
                        "role_id": turn["segment"]["role_id"],
                        "source_ranges": deepcopy(turn["source_ranges"]),
                        "spoken_text": turn["spoken_text"],
                        "spoken_start": turn["spoken_start"], "spoken_end": turn["spoken_end"],
                    } for turn in clip["paragraph_turns"]]
                if performance:
                    timing["performance"] = deepcopy(performance)
                if row.get("simultaneous_speakers"):
                    timing["simultaneous_speakers"] = deepcopy(row["simultaneous_speakers"])
                if "speaker_lanes" in clip:
                    lanes = clip["speaker_lanes"]
                    if (not isinstance(lanes, list) or len(lanes) < 2
                            or any(not isinstance(lane, dict) for lane in lanes)
                            or [{"role_id": lane.get("role_id"), "voice_id": lane.get("voice_id")}
                                for lane in lanes] != row.get("simultaneous_speakers")
                            or any(type(lane.get("start_frame")) is not int or lane["start_frame"] != 0
                                   or type(lane.get("end_frame")) is not int
                                   or not 0 < lane["end_frame"] <= cursor - first_frame for lane in lanes)
                            or max(lane["end_frame"] for lane in lanes) != cursor - first_frame):
                        raise ValueError("Invalid simultaneous speaker timing")
                    timing["speaker_lanes"] = [dict(deepcopy(lane),
                        chapter_start_frame=first_frame + lane["start_frame"],
                        chapter_end_frame=first_frame + lane["end_frame"]) for lane in lanes]
                    timing["mix"] = deepcopy(clip["mix"])
                timings.append(timing)
                if normalize_pauses:
                    pending_punctuation = max(pending_punctuation, punctuation_pause(
                        clip["spoken_text"], sentence_frames, clause_frames))
                if end == len(spoken) and row.get("break_after") == "paragraph":
                    pending_pause = max(pending_pause, paragraph_frames)
                if (end == len(spoken) and performance.get("kind") == "bracketed"
                        and performance.get("exit") is True):
                    pending_pause = max(pending_pause, bracket_frames)
            # Empty scripts deliberately remain frame-empty and are skipped
            # during packaging. A real chapter keeps its final paragraph pause.
            if cursor:
                silence(max(0, pending_pause - previous_tail))
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


def aac_codec_arguments(bitrate: str) -> list[str]:
    """AAC faststart codec flags shared by chapter encoding and deployment smoke."""
    return ["-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart"]


def encode_chapter(chapter: dict, output_path: str | Path, *, book: dict,
                   ordinal: int, total_chapters: int, chapter_format: str = "m4a",
                   aac_bitrate: str = "128k", mp3_bitrate: str = "128k",
                   ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    """Encode and inspect one lossless chapter, without changing its timeline."""
    if chapter_format not in {"m4a", "mp3"}:
        raise ValueError("chapter_format must be m4a or mp3")
    for bitrate in (aac_bitrate, mp3_bitrate):
        if not isinstance(bitrate, str) or not re.fullmatch(r"[1-9][0-9]{0,2}k", bitrate):
            raise ValueError("Audio bitrate must be a positive value such as 128k")
    if (type(ordinal) is not int or type(total_chapters) is not int
            or not 1 <= ordinal <= total_chapters):
        raise ValueError("Chapter track numbers must be positive and within the book")
    target = Path(output_path)
    if target.exists() or target.is_symlink():
        raise ValueError(f"Audio chapter output already exists: {target}")
    rate = _rate(chapter["sample_rate"])
    info = _wave_info(Path(chapter["wav_path"]), rate)
    if info["frames"] != chapter["frames"] or not info["frames"]:
        raise ValueError("Chapter PCM is empty or differs from its synchronization data")
    target.parent.mkdir(parents=True, exist_ok=True)
    codec = (aac_codec_arguments(aac_bitrate) if chapter_format == "m4a"
             else ["-c:a", "libmp3lame", "-b:a", mp3_bitrate])
    with tempfile.TemporaryDirectory(prefix=".chapter-", dir=target.parent) as temporary:
        staged = Path(temporary) / f"chapter.{chapter_format}"
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
              "-i", str(Path(chapter["wav_path"]).resolve()), "-map", "0:a:0"]
             + codec + ["-metadata", f"title={chapter['title']}",
                        "-metadata", f"album={book.get('title', 'Audiobook')}",
                        "-metadata", f"artist={'; '.join(book.get('authors', []))}",
                        "-metadata", f"track={ordinal}/{total_chapters}", str(staged)])
        inspection = _check_encoded(staged, info["frames"], rate, ffprobe=ffprobe)
        target.hardlink_to(staged)
    return {"path": str(target), "sha256": _hash(target), **info,
            "duration_seconds": float(next(stream for stream in inspection["streams"]
                                             if stream["codec_type"] == "audio")["duration"])}


def package_audio(chapters: Sequence[dict], output_dir: str | Path, *,
                  book: dict, cover_path: str | Path | None = None,
                  chapter_format: str = "m4a", aac_bitrate: str = "128k",
                  mp3_bitrate: str = "128k", ffmpeg: str = "ffmpeg",
                  ffprobe: str = "ffprobe", source_fingerprint: str | None = None,
                  notify=lambda message: None) -> dict:
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
    chapter_numbers = set()
    for ordinal, chapter in enumerate(chapters, 1):
        info = _wave_info(Path(chapter["wav_path"]), sample_rate)
        if info["frames"] != chapter["frames"] or chapter["sample_rate"] != sample_rate:
            raise ValueError("Chapter PCM no longer matches its synchronization data")
        number = chapter.get("number", ordinal)
        if type(number) is not int or number < 1 or number in chapter_numbers:
            raise ValueError("Chapter source numbers must be distinct positive integers")
        chapter_numbers.add(number)
        record = {
            "id": chapter["id"], "title": chapter["title"],
            "script_path": chapter.get("script_path"),
            "ordinal": ordinal, "number": number, "start_frame": cursor,
            "end_frame": cursor + info["frames"], "frames": info["frames"],
            "skipped": info["frames"] == 0,
            "path": f"chapters/chapter-{number:04d}.{chapter_format}" if info["frames"] else None,
        }
        if "normalization" in chapter:
            record["normalization"] = deepcopy(chapter["normalization"])
        if "pcm_sha256" in chapter:
            if _hash(Path(chapter["wav_path"])) != chapter["pcm_sha256"]:
                raise ValueError("Chapter PCM failed integrity checks before packaging")
            record["pcm_sha256"] = chapter["pcm_sha256"]
        records.append(record)
        for segment in chapter["segments"]:
            start, end = segment["chapter_start_frame"], segment["chapter_end_frame"]
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start <= end <= info["frames"]):
                raise ValueError("Chapter synchronization frame range is invalid")
            timing = dict(deepcopy(segment), chapter_ordinal=ordinal,
                          chapter_id=chapter["id"],
                          book_start_frame=cursor + start, book_end_frame=cursor + end)
            for lane in timing.get("speaker_lanes", []):
                lane["book_start_frame"] = cursor + lane["chapter_start_frame"]
                lane["book_end_frame"] = cursor + lane["chapter_end_frame"]
            timings.append(timing)
        if info["frames"]:
            audio_chapters.append((chapter, record))
        cursor += info["frames"]
    if cursor == 0:
        raise ValueError("Audiobook has no speech audio to package")
    chapter_dir = target / "chapters"
    chapter_dir.mkdir()
    cover = Path(cover_path).resolve() if cover_path is not None else None
    from .book_metadata import metadata_arguments, check_book_tags
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
    for chapter, record in audio_chapters:
        output = target / record["path"]
        chapter_label = f"{record['number']} / {chapter.get('total_chapters', len(records))}"
        if chapter.get("encoded_path") is not None:
            cached = Path(chapter["encoded_path"])
            if (cached.is_symlink() or not cached.is_file()
                    or cached.suffix != f".{chapter_format}"
                    or _hash(cached) != chapter.get("encoded_sha256")):
                raise ValueError("Cached encoded chapter failed integrity checks")
            _check_encoded(cached, record["frames"], sample_rate, ffprobe=ffprobe)
            notify(f"Reusing chapter audio {chapter_label}: {record['title']}")
            with output.open("xb") as destination, cached.open("rb") as source:
                shutil.copyfileobj(source, destination)
            if _hash(output) != chapter["encoded_sha256"]:
                raise ValueError("Cached encoded chapter changed during packaging")
        else:
            notify(f"Encoding chapter {chapter_label}: {record['title']}")
            encode_chapter(chapter, output, book=book,
                           ordinal=chapter.get("track_number", record["ordinal"]),
                           total_chapters=chapter.get("total_chapters", len(records)),
                           chapter_format=chapter_format,
                           aac_bitrate=aac_bitrate, mp3_bitrate=mp3_bitrate,
                           ffmpeg=ffmpeg, ffprobe=ffprobe)
        record["sha256"] = _hash(output)
    notify(f"Combining {len(audio_chapters)} chapters into book.m4b")
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=target) as temporary:
        scratch = Path(temporary)
        concat = scratch / "chapters.ffconcat"
        concat.write_text("ffconcat version 1.0\n" + "".join(
            f"file {_concat_path(Path(chapter['wav_path']))}\n"
            f"duration {chapter['frames'] / sample_rate:.12f}\n"
            for chapter, _ in audio_chapters), encoding="utf-8")
        metadata = scratch / "book.ffmetadata"
        metadata.write_text(
            ";FFMETADATA1\n" + "".join(
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
        command += metadata_arguments(book)
        command += ["-movflags", "+faststart", "-f", "ipod", str(target / "book.m4b")]
        _run(command)
    inspection = _check_encoded(target / "book.m4b", cursor, sample_rate,
                                chapters=[record for _, record in audio_chapters],
                                cover=cover is not None, ffprobe=ffprobe)
    check_book_tags(inspection, book)
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
