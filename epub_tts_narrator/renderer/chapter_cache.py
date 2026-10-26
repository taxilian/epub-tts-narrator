"""Durable, independently playable chapter checkpoints from verified clips."""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .assembly import _check_encoded, _wave_info, compose_chapter, encode_chapter, mix_speakers
from .chapter_records import signed_record
from .common import digest, file_hash, read_json, regular_file, write_json
from .config import RENDER_CONTRACT
from .interrupts import run_subprocess


def chapter_settings(render, *, censorship_policy=None):
    """Return only the settings that affect chapter composition or encoding."""
    settings = {"version": 2}
    for key, spec in RENDER_CONTRACT.items():
        if not spec.get("chapter"):
            continue
        value = render.get(key, spec["default"])
        settings[key] = str(value) if spec["kind"] == "text" else value
    if settings["chapter_format"] not in RENDER_CONTRACT["chapter_format"]["choices"]:
        raise ValueError("chapter_format must be m4a or mp3")
    for key, spec in RENDER_CONTRACT.items():
        if spec.get("checked") and spec["kind"] == "bool" and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key, spec in RENDER_CONTRACT.items():
        if spec.get("checked") and spec["kind"] == "int":
            low, high = spec["range"]
            if type(settings[key]) is not int or not low <= settings[key] <= high:
                raise ValueError(f"Invalid chapter {key}")
    for key, spec in RENDER_CONTRACT.items():
        if spec.get("checked") and spec["kind"] == "float":
            low, high = spec["range"]
            value = settings[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid chapter {key}")
    if censorship_policy is not None:
        from .censorship import policy_identity
        policy = policy_identity(censorship_policy)
        if policy is not None:
            settings["censorship"] = policy
    return settings


def _loudness_pass(source, audio_filter, destination, settings):
    command = [settings["ffmpeg"], "-hide_banner", "-nostats", "-nostdin", "-n",
               "-protocol_whitelist", "file,pipe", "-i", str(source),
               "-map", "0:a:0", "-vn", "-sn", "-dn", "-af", audio_filter]
    if destination is None:
        command += ["-f", "null", "-"]
    else:
        command += ["-map_metadata", "-1", "-ac", "1", "-ar", str(settings["sample_rate"]),
                    "-c:a", "pcm_s16le", "-f", "wav", str(destination)]
    try:
        result = run_subprocess(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Unable to normalize chapter with FFmpeg: {exc}") from exc
    report = result.stderr.decode("utf-8", errors="replace")
    if result.returncode:
        raise ValueError(f"Chapter loudness normalization failed: {report[-2000:].strip()}")
    matches = re.findall(r'\{\s*"input_i"\s*:.*?\}', report, flags=re.DOTALL)
    try:
        stats = json.loads(matches[-1])
        for key in ("input_i", "input_tp", "input_lra", "input_thresh",
                    "output_i", "output_tp", "target_offset"):
            if not math.isfinite(float(stats[key])):
                raise ValueError("non-finite loudness")
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Chapter has no measurable loudness, or FFmpeg returned invalid measurements") from exc
    return stats


def _normalize(source, settings):
    """Replace only staged assembled PCM; keep every sample frame and source clip."""
    info = _wave_info(source, settings["sample_rate"])
    before = file_hash(source)
    common = {"input_sha256": before, "target_lufs": settings["target_lufs"],
              "true_peak_db": settings["true_peak_db"]}
    if not settings["normalize_loudness"]:
        return dict(common, status="disabled")
    if info["frames"] == 0:
        return dict(common, status="skipped", reason="empty_chapter")
    # EBU R128 integrated loudness needs a 400 ms analysis window. Such a
    # chapter is still usable if its clips passed speech verification; record
    # explicitly that its gain is unchanged instead of inventing a LUFS value.
    if info["frames"] < math.ceil(info["sample_rate"] * .4):
        return dict(common, status="skipped", reason="too_short_for_integrated_loudness")
    base = (f"loudnorm=I={settings['target_lufs']}:TP={settings['true_peak_db']}:"
            "LRA=50:print_format=json")
    measured = _loudness_pass(source, base, None, settings)
    audio_filter = (base + ":linear=true"
                    f":measured_I={float(measured['input_i'])}"
                    f":measured_TP={float(measured['input_tp'])}"
                    f":measured_LRA={float(measured['input_lra'])}"
                    f":measured_thresh={float(measured['input_thresh'])}"
                    f":offset={float(measured['target_offset'])}")
    destination = source.with_name("normalized.wav")
    result = _loudness_pass(source, audio_filter, destination, settings)
    if _wave_info(destination, settings["sample_rate"]) != info:
        raise ValueError("Chapter loudness normalization changed the PCM timeline")
    if (abs(float(result["output_i"]) - settings["target_lufs"]) > 1.0
            or float(result["output_tp"]) > settings["true_peak_db"] + .2):
        raise ValueError("Chapter normalization could not meet the loudness/peak targets")
    os.replace(destination, source)
    return dict(common, status="normalized", method="ebu_r128_two_pass",
                input_lufs=float(measured["input_i"]), output_lufs=float(result["output_i"]),
                input_true_peak_db=float(measured["input_tp"]),
                output_true_peak_db=float(result["output_tp"]),
                normalization_type=result.get("normalization_type"))


def _timings(chapter):
    previous = 0
    if not isinstance(chapter.get("segments"), list):
        raise ValueError("Cached chapter synchronization data is invalid")
    for segment in chapter["segments"]:
        if not isinstance(segment, dict):
            raise ValueError("Cached chapter synchronization data is invalid")
        start, end = segment.get("chapter_start_frame"), segment.get("chapter_end_frame")
        if (type(start) is not int or type(end) is not int
                or not previous <= start <= end <= chapter["frames"]):
            raise ValueError("Cached chapter synchronization frame range is invalid")
        if "source_audio_start_frame" in segment or "source_audio_end_frame" in segment:
            audio_start, audio_end = (segment.get("source_audio_start_frame"),
                                      segment.get("source_audio_end_frame"))
            if (type(audio_start) is not int or type(audio_end) is not int
                    or not 0 <= audio_start <= audio_end or audio_end - audio_start != end - start):
                raise ValueError("Cached source audio synchronization frame range is invalid")
        if "source_segments" in segment:
            sources = segment["source_segments"]
            if not isinstance(sources, list) or len(sources) < 2:
                raise ValueError("Cached paragraph source segments are invalid")
            for source in sources:
                if (not isinstance(source, dict)
                        or any(not isinstance(source.get(key), str) or not source[key]
                               for key in ("chunk_id", "segment_id", "block_id", "voice_id", "role_id",
                                           "spoken_text"))
                        or type(source.get("spoken_start")) is not int
                        or type(source.get("spoken_end")) is not int
                        or not 0 <= source["spoken_start"] <= source["spoken_end"]
                        or source["spoken_end"] - source["spoken_start"] != len(source["spoken_text"])
                        or not isinstance(source.get("source_ranges"), list)
                        or not source["source_ranges"]
                        or any(not isinstance(span, dict) or type(span.get("start")) is not int
                               or type(span.get("end")) is not int or not 0 <= span["start"] < span["end"]
                               for span in source["source_ranges"])):
                    raise ValueError("Cached paragraph source segments are invalid")
        if "speaker_lanes" in segment:
            lanes = segment["speaker_lanes"]
            if (not isinstance(lanes, list) or len(lanes) < 2
                    or any(not isinstance(lane, dict)
                           or lane.get("start_frame") != 0
                           or type(lane.get("end_frame")) is not int
                           or not 0 < lane["end_frame"] <= end - start
                           or lane.get("chapter_start_frame") != start
                           or lane.get("chapter_end_frame") != start + lane["end_frame"] for lane in lanes)
                    or max(lane["chapter_end_frame"] for lane in lanes) != end):
                raise ValueError("Cached simultaneous speaker timing is invalid")
        if "censorship" in segment:
            _censorship_timing(segment, chapter.get("sample_rate"))
        previous = end


def _censorship_timing(segment, sample_rate):
    """Check every exported mask against its original PCM and chapter offset."""
    from .censorship import policy_identity, validate_mask_plan
    value = segment["censorship"]
    if (not isinstance(value, dict) or set(value) != {"version", "policy", "lanes"}
            or type(value["version"]) is not int or value["version"] != 1
            or policy_identity(value["policy"]) != value["policy"]
            or value["policy"] is None or not isinstance(value["lanes"], list)):
        raise ValueError("Cached censorship timing is invalid")
    expected_voices = ([lane["voice_id"] for lane in segment["speaker_lanes"]]
                       if "speaker_lanes" in segment else [segment["voice_id"]])
    if ([lane.get("voice_id") if isinstance(lane, dict) else None for lane in value["lanes"]]
            != expected_voices):
        raise ValueError("Cached censorship timing differs from its performers")
    source_start, source_end = segment["source_audio_start_frame"], segment["source_audio_end_frame"]
    chapter_start = segment["chapter_start_frame"]
    for lane in value["lanes"]:
        if set(lane) != {"voice_id", "audit", "chapter_masks"} or not isinstance(lane["audit"], dict):
            raise ValueError("Cached censorship lane audit is invalid")
        audit = lane["audit"]
        validate_mask_plan(audit, segment["spoken_text"], value["policy"], sample_rate=sample_rate,
                           frames=audit.get("frames"), audio_sha256=audit.get("audio_sha256"),
                           targets=audit.get("targets"))
        if "speaker_lanes" in segment:
            source_lane = next(item for item in segment["speaker_lanes"] if item["voice_id"] == lane["voice_id"])
            if (audit["frames"] != source_lane["end_frame"]
                    or audit["audio_sha256"] != source_lane["audio_sha256"]):
                raise ValueError("Cached censorship timing differs from its original speaker audio")
        elif audit["frames"] < source_end:
            raise ValueError("Cached censorship timing leaves its original audio")
        expected = [{"source_start": mask["source_start"], "source_end": mask["source_end"],
                     "chapter_start_frame": chapter_start + max(mask["mute_start_frame"], source_start) - source_start,
                     "chapter_end_frame": chapter_start + min(mask["mute_end_frame"], source_end) - source_start}
                    for mask in audit["masks"]
                    if mask["mute_end_frame"] > source_start and mask["mute_start_frame"] < source_end]
        if lane["chapter_masks"] != expected:
            raise ValueError("Cached censorship frame mapping differs from its original audio")


LANE_FIELDS = ("role_id", "voice_id", "audio_sha256", "request_hash", "validation_key",
               "start_frame", "end_frame")


def mixture_identity(lanes, sample_rate):
    """Bind the fixed mixing recipe to each independently verified performer."""
    return {"version": 1, "method": "aligned_fixed_average", "sample_rate": sample_rate,
            "speaker_lanes": [{key: lane[key] for key in LANE_FIELDS} for lane in lanes]}


def mixture_validation_key(lanes):
    return digest({"method": "independently_verified_simultaneous_lanes", "version": 1,
                   "lanes": [{key: lane[key] for key in ("role_id", "voice_id", "audio_sha256", "validation_key")}
                             for lane in lanes]})


def prepare_mixture(lanes, directory, *, sample_rate=44100, notify=print, preserve_lanes=False):
    """Cache a mix only after every lane has passed independent verification."""
    if not isinstance(lanes, list) or len(lanes) < 2:
        raise ValueError("Simultaneous speech requires at least two verified lanes")
    prepared = []
    for lane in lanes:
        if (not isinstance(lane, dict) or not isinstance(lane.get("report"), dict)
                or lane["report"].get("passed") is not True):
            raise ValueError("Every simultaneous speaker must pass verification before mixing")
        path = regular_file(lane["wav_path"])
        info = _wave_info(path, sample_rate)
        if not info["frames"] or file_hash(path) != lane["audio_sha256"]:
            raise ValueError("Verified simultaneous speaker audio changed or is empty")
        prepared.append(dict(lane, start_frame=0, end_frame=info["frames"]))
    if (len({lane["role_id"] for lane in prepared}) != len(prepared)
            or len({lane["voice_id"] for lane in prepared}) != len(prepared)):
        raise ValueError("Simultaneous speaker roles and voices must be distinct")
    identity = mixture_identity(prepared, sample_rate)
    key, frames = digest(identity), max(lane["end_frame"] for lane in prepared)
    directory = Path(directory)
    for parent in (directory, directory.parent):
        if parent.is_symlink():
            raise ValueError("Simultaneous mixture cache directories must not be symlinks")
    folder = directory / key

    def load():
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Simultaneous mixture cache must be a regular directory")
        record = read_json(regular_file(folder / "mix.json"))
        if not isinstance(record, dict):
            raise ValueError("Simultaneous mixture metadata must be an object")
        checksum = record.pop("record_sha256", None)
        wav = regular_file(folder / "mix.wav")
        if (checksum != signed_record(record) or record.get("identity") != identity
                or record.get("format") != "audiobook-renderer-mix" or record.get("version") != 1
                or record.get("audio_sha256") != file_hash(wav)
                or _wave_info(wav, sample_rate)["frames"] != frames
                or record.get("frames") != frames):
            raise ValueError("Simultaneous mixture cache failed integrity checks")
        mix = {"version": 1, "method": "aligned_fixed_average", "sample_rate": sample_rate,
               "frames": frames, "cache_key": key, "audio_sha256": record["audio_sha256"],
               "weight": 1 / len(prepared), "peak_amplitude": record["peak_amplitude"]}
        result = {"wav_path": str(wav.resolve()), "mix": mix,
                "speaker_lanes": [{key: value for key, value in lane.items() if key != "wav_path"}
                                  for lane in prepared]}
        if preserve_lanes:
            # These paths are private execution inputs, never voice assets or
            # transport artifacts. Retain original lanes for post-verification
            # masking before averaging, without changing the original mix key.
            result["censorship_lanes"] = deepcopy(prepared)
        return result

    if folder.exists() or folder.is_symlink():
        return load()
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mix-", dir=directory) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        notify(f"Mixing {len(prepared)} independently verified simultaneous speakers")
        mixed = mix_speakers(prepared, stage / "mix.wav", sample_rate=sample_rate)
        record = {"format": "audiobook-renderer-mix", "version": 1, "identity": identity,
                  "frames": frames, "audio_sha256": mixed["audio_sha256"],
                  "peak_amplitude": mixed["peak_amplitude"]}
        write_json(stage / "mix.json", dict(record, record_sha256=digest(record)))
        try:
            os.rename(stage, folder)
        except OSError:
            if not folder.exists():
                raise
    return load()


def load_chapter(directory, *, ffprobe="ffprobe"):
    """Validate a complete checkpoint before exposing its audio or timings."""
    folder = Path(directory)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Chapter cache must be a regular directory")
    try:
        record = read_json(regular_file(folder / "chapter.json"))
        if not isinstance(record, dict):
            raise ValueError("Chapter cache metadata must be an object")
        checksum = record.pop("record_sha256")
        if (checksum != signed_record(record) or record.get("format") != "audiobook-renderer-chapter"
                or record.get("version") != 1 or digest(record["identity"]) != folder.name):
            raise ValueError("Chapter cache metadata failed integrity checks")
        chapter = deepcopy(record["chapter"])
        wav = regular_file(folder / "chapter.wav")
        if (file_hash(wav) != record["pcm_sha256"]
                or _wave_info(wav, chapter["sample_rate"]) != {
                    "sample_rate": chapter["sample_rate"], "channels": chapter["channels"],
                    "frames": chapter["frames"]}):
            raise ValueError("Chapter PCM failed integrity checks")
        _timings(chapter)
        encoded = None
        if chapter["frames"]:
            kind = record["identity"]["settings"]["chapter_format"]
            if kind not in {"m4a", "mp3"}:
                raise ValueError("Chapter cache encoding format is invalid")
            encoded = regular_file(folder / f"chapter.{kind}")
            if file_hash(encoded) != record["encoded_sha256"]:
                raise ValueError("Encoded chapter failed integrity checks")
            _check_encoded(encoded, chapter["frames"], chapter["sample_rate"], ffprobe=ffprobe)
        elif record["encoded_sha256"] is not None:
            raise ValueError("Empty chapter unexpectedly has encoded audio")
        return dict(chapter, wav_path=str(wav.resolve()),
                    encoded_path=str(encoded.resolve()) if encoded else None,
                    encoded_sha256=record["encoded_sha256"],
                    pcm_sha256=record["pcm_sha256"], normalization=record["normalization"],
                    cache_directory=str(folder.resolve()), cache_key=folder.name,
                    track_number=record["identity"]["ordinal"],
                    total_chapters=record["identity"]["total_chapters"],
                    context=deepcopy(record["identity"]["context"]),
                    settings=deepcopy(record["identity"]["settings"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete or invalid chapter cache: {folder}") from exc


def prepare_chapter(chapter, clips, directory, *, book, render_settings, context, notify=print,
                    censorship_policy=None):
    """Compose, level, and encode once; leave a complete chapter after later failures."""
    settings = chapter_settings(render_settings, censorship_policy=censorship_policy)
    directory = Path(directory)
    for candidate in (directory, directory.parent):
        if candidate.is_symlink():
            raise ValueError("Chapter cache directories must not be symlinks")
    clip_identity = []
    for clip in clips:
        item = {key: deepcopy(value) for key, value in clip.items()
                if key not in {"wav_path", "censorship", "censorship_lanes"}}
        if "censorship" in settings:
            if "censorship" in clip:
                item["censorship"] = deepcopy(clip["censorship"])
            if "censorship_lanes" in clip:
                item["censorship_lanes"] = [
                    {key: deepcopy(value) for key, value in lane.items() if key != "wav_path"}
                    for lane in clip["censorship_lanes"]]
        item["audio_sha256"] = (file_hash(regular_file(clip["wav_path"]))
                                if clip.get("wav_path") is not None else None)
        clip_identity.append(item)
    ordinal = chapter.get("_ordinal", chapter.get("number", 1))
    total = chapter.get("_total_chapters", context.get("total_chapters", ordinal))
    metadata = {key: deepcopy(chapter[key]) for key in ("id", "title", "number", "script_path")
                if key in chapter}
    chapter_label = f"{metadata.get('number', ordinal)} / {total}"
    identity = {"context": deepcopy(context), "chapter": metadata, "book": deepcopy(book),
                "settings": settings, "clips": clip_identity, "ordinal": ordinal, "total_chapters": total}
    key = digest(identity)
    folder = directory / key
    if folder.exists() or folder.is_symlink():
        result = load_chapter(folder, ffprobe=settings["ffprobe"])
        notify(f"Reusing completed chapter {chapter_label}: {metadata['title']}")
        return result
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".chapter-", dir=directory) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        notify(f"Assembling chapter {chapter_label}: {metadata['title']}")
        composed = compose_chapter(
            clips, stage / "chapter.wav", sample_rate=settings["sample_rate"],
            paragraph_pause_ms=settings["paragraph_pause_ms"],
            blank_pause_ms=settings["blank_pause_ms"], bracket_pause_ms=settings["bracket_pause_ms"],
            normalize_pauses=settings["normalize_pauses"],
            sentence_pause_ms=settings["sentence_pause_ms"], clause_pause_ms=settings["clause_pause_ms"],
            censorship_policy=censorship_policy)
        composed.update(metadata)
        normalization = _normalize(stage / "chapter.wav", settings)
        encoded = None
        if composed["frames"]:
            notify(f"Encoding chapter {chapter_label}: {metadata['title']}")
            encoded = encode_chapter(
                composed, stage / f"chapter.{settings['chapter_format']}", book=book,
                ordinal=ordinal, total_chapters=total, chapter_format=settings["chapter_format"],
                aac_bitrate=settings["aac_bitrate"], mp3_bitrate=settings["mp3_bitrate"],
                ffmpeg=settings["ffmpeg"], ffprobe=settings["ffprobe"])
        for clip, expected in zip(clips, clip_identity):
            if (clip.get("wav_path") is not None
                    and file_hash(regular_file(clip["wav_path"])) != expected["audio_sha256"]):
                raise ValueError("Verified clip changed during chapter preparation")
        record = {"format": "audiobook-renderer-chapter", "version": 1, "identity": identity,
                  "chapter": {key: value for key, value in composed.items() if key != "wav_path"},
                  "normalization": normalization, "pcm_sha256": file_hash(stage / "chapter.wav"),
                  "encoded_sha256": encoded["sha256"] if encoded else None}
        write_json(stage / "chapter.json", dict(record, record_sha256=digest(record)))
        try:
            os.rename(stage, folder)
        except OSError:
            if not folder.exists():
                raise
            # A concurrent producer must have published exactly the same identity.
            return load_chapter(folder, ffprobe=settings["ffprobe"])
    result = load_chapter(folder, ffprobe=settings["ffprobe"])
    if result["encoded_path"]:
        notify(f"Chapter {chapter_label} ready: {result['encoded_path']}")
    else:
        notify(f"Chapter {chapter_label} is empty; recorded as skipped")
    return result


def reencode_chapter(chapter, directory, *, book, render_settings, context, notify=print,
                     censorship_policy=None):
    """Publish a new encoding checkpoint from existing, validated chapter PCM."""
    settings = chapter_settings(render_settings, censorship_policy=censorship_policy)
    source = load_chapter(chapter["cache_directory"], ffprobe=settings["ffprobe"])
    encoding_keys = {"chapter_format", "aac_bitrate", "mp3_bitrate", "ffmpeg", "ffprobe"}
    if ({key: value for key, value in source["settings"].items() if key not in encoding_keys}
            != {key: value for key, value in settings.items() if key not in encoding_keys}):
        raise ValueError("Re-encoding cannot change chapter PCM settings or censorship")
    original = read_json(regular_file(Path(source["cache_directory"]) / "chapter.json"))
    identity = deepcopy(original["identity"])
    if identity["book"] != book:
        raise ValueError("Re-encoding requires the original chapter book metadata")
    identity.update(settings=settings, context=deepcopy(context))
    directory = Path(directory)
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ValueError("Chapter cache directories must not be symlinks")
    folder = directory / digest(identity)
    if folder.exists() or folder.is_symlink():
        return load_chapter(folder, ffprobe=settings["ffprobe"])
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".chapter-", dir=directory) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        wav = stage / "chapter.wav"
        wav.hardlink_to(regular_file(source["wav_path"]))
        encoded = None
        if source["frames"]:
            notify(f"Encoding chapter {source['number']} / {source['total_chapters']}: {source['title']}")
            encoded = encode_chapter(
                dict(source, wav_path=str(wav)), stage / f"chapter.{settings['chapter_format']}",
                book=book, ordinal=identity["ordinal"], total_chapters=identity["total_chapters"],
                chapter_format=settings["chapter_format"], aac_bitrate=settings["aac_bitrate"],
                mp3_bitrate=settings["mp3_bitrate"], ffmpeg=settings["ffmpeg"], ffprobe=settings["ffprobe"])
        if file_hash(wav) != original["pcm_sha256"]:
            raise ValueError("Chapter PCM changed during re-encoding")
        record = {key: deepcopy(value) for key, value in original.items() if key != "record_sha256"}
        record.update(identity=identity, encoded_sha256=encoded["sha256"] if encoded else None)
        write_json(stage / "chapter.json", dict(record, record_sha256=digest(record)))
        try:
            os.rename(stage, folder)
        except OSError:
            if not folder.exists():
                raise
    return load_chapter(folder, ffprobe=settings["ffprobe"])
