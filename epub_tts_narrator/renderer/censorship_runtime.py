"""Prepare hash-bound censor masks without changing verified speech or source."""

from copy import deepcopy
from pathlib import Path

from ..spoken_text import build_spoken
from .assembly import _wave_info
from .censorship import CensorshipAlignmentRequired, matches, plan_masks, validate_policy
from .common import file_hash, read_json, regular_file


def censorship_targets(chunk, policy):
    """Map whole source words through approved spellings into this spoken slice."""
    policy = validate_policy(policy)
    if not policy["enabled"]:
        return []
    if "paragraph_turns" in chunk:
        result, offset = [], 0
        for turn in chunk["paragraph_turns"]:
            result.extend(dict(item, start=item["start"] + offset, end=item["end"] + offset)
                          for item in censorship_targets(turn, policy))
            offset += len(turn["spoken_text"])
        if "".join(t["spoken_text"] for t in chunk["paragraph_turns"]) != chunk["spoken_text"]:
            raise ValueError("Censorship paragraph does not reconstruct its approved speech")
        return result
    row = chunk["segment"]
    spoken, mapping = build_spoken(row)
    start, end = chunk["spoken_start"], chunk["spoken_end"]
    if spoken[start:end] != chunk["spoken_text"]:
        raise ValueError("Censorship slice differs from approved speech")
    targets = []
    for target in matches(row["text"], policy):
        a, b = row["start"] + target["start"], row["start"] + target["end"]
        spans = []
        for span in mapping:
            left, right = max(a, span["source_start"]), min(b, span["source_end"])
            if left >= right:
                continue
            if span["kind"] == "source":
                spans.append((span["spoken_start"] + left - span["source_start"],
                              span["spoken_start"] + right - span["source_start"]))
            else:
                if not a <= span["source_start"] < span["source_end"] <= b:
                    if span["spoken_start"] < end and start < span["spoken_end"]:
                        raise CensorshipAlignmentRequired("source_word_inside_multiword_replacement")
                    continue
                spans.append((span["spoken_start"], span["spoken_end"]))
        if not spans:
            continue
        left, right = spans[0][0], spans[-1][1]
        if right <= start or left >= end:
            continue
        if left < start or right > end:
            raise CensorshipAlignmentRequired("source_word_split_between_audio_clips")
        targets.append({"text": spoken[left:right], "start": left - start, "end": right - start,
                        "normalized": target["normalized"]})
    return targets


def prepare_chapter_censorship(clips, acceptance, policy, *, verifier=None,
                               cache_dir=None, verifier_factory=None, notify=print):
    """Resolve masks after verification, before inference-free chapter assembly.

    Optional forced alignment is separate from acceptance. It never regenerates
    speech or overwrites accepted reports. All originals remain immutable.
    """
    policy = validate_policy(policy)
    if not policy["enabled"]:
        return clips
    accepted_by_id = {entry["chunk_id"]: entry for entry in acceptance}
    if len(accepted_by_id) != len(acceptance):
        raise ValueError("Duplicate censorship verification records")
    result, censored = [], 0

    def prepare(chunk, path, accepted):
        nonlocal verifier, censored
        path = regular_file(path)
        checksum = file_hash(path)
        if (checksum != accepted.get("audio_sha256")
                or accepted.get("report", {}).get("passed") is not True):
            raise ValueError("Censorship requires unchanged independently accepted audio")
        info = _wave_info(path)
        targets = censorship_targets(chunk, policy)
        arguments = dict(sample_rate=info["sample_rate"], frames=info["frames"],
                         audio_sha256=checksum, targets=targets)
        try:
            audit = plan_masks(chunk["spoken_text"], accepted["report"], policy, **arguments)
        except CensorshipAlignmentRequired:
            if cache_dir is None:
                raise
            from .censorship_alignment import align_saved_audio, CensorshipAlignmentUnavailable
            notify("  Aligning a saved clip for censorship; accepted speech is retained.")
            try:
                aligned = align_saved_audio(path, chunk["spoken_text"], verifier, cache_dir,
                                            report=accepted["report"])
            except CensorshipAlignmentUnavailable:
                if verifier is not None or verifier_factory is None:
                    raise
                verifier = verifier_factory()
                aligned = align_saved_audio(path, chunk["spoken_text"], verifier, cache_dir,
                                            report=accepted["report"])
            audit = plan_masks(chunk["spoken_text"], aligned, policy, **arguments)
        censored += len(audit["masks"])
        return audit

    for original in clips:
        chunk = deepcopy(original)
        if chunk.get("wav_path") is None:
            result.append(chunk)
            continue
        accepted = accepted_by_id.get(chunk["chunk_id"])
        if accepted is None:
            raise ValueError("Censorship clip has no matching verification")
        if "speaker_lanes" in chunk:
            lanes = chunk.get("censorship_lanes")
            accepted_lanes = accepted.get("speaker_lanes", [])
            if not isinstance(lanes, list) or len(lanes) != len(accepted_lanes):
                raise ValueError("Censorship requires each original simultaneous speaker lane")
            for lane, verified in zip(lanes, accepted_lanes):
                if any(lane.get(key) != verified.get(key) for key in ("voice_id", "role_id", "audio_sha256")):
                    raise ValueError("Censorship simultaneous speaker bindings differ")
                lane["censorship"] = prepare(chunk, lane["wav_path"], verified)
        else:
            chunk["censorship"] = prepare(chunk, chunk["wav_path"], accepted)
        result.append(chunk)
    notify(f"Prepared censorship: {censored} word occurrence(s); original audio retained.")
    return result


def restore_verified_clips(chapter, acceptance, work, *, sample_rate, notify=print):
    """Recover original takes, including unison lanes, for CPU recomposition."""
    from .chapter_cache import prepare_mixture
    work = Path(work)
    accepted_by_id = {item["chunk_id"]: item for item in acceptance}

    def audio(entry):
        request = entry["request_hash"]
        if not isinstance(request, str) or len(request) != 64 or any(c not in "0123456789abcdef" for c in request):
            raise ValueError("Invalid accepted audio request hash")
        directory = work / "clips" / request
        saved = read_json(regular_file(directory / "accepted.json"))
        name = saved.get("audio_file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
            raise ValueError("Invalid original accepted audio filename")
        path = regular_file(directory / name)
        if (saved.get("request_hash") != request or saved.get("audio_sha256") != entry["audio_sha256"]
                or file_hash(path) != entry["audio_sha256"] or saved.get("report", {}).get("passed") is not True):
            raise ValueError("Original accepted audio changed before recomposition")
        return str(path)

    clips = []
    for original in chapter["chunks"]:
        chunk = deepcopy(original)
        if chunk["silent_control"]:
            clips.append(dict(chunk, wav_path=None))
            continue
        accepted = accepted_by_id[chunk["chunk_id"]]
        if "speaker_lanes" in accepted:
            lanes = [dict(lane, wav_path=audio(lane)) for lane in accepted["speaker_lanes"]]
            mix = prepare_mixture(lanes, work / "mixes", sample_rate=sample_rate,
                                  notify=notify, preserve_lanes=True)
            if mix["mix"] != accepted["mix"]:
                raise ValueError("Restored simultaneous mixture differs from accepted audio")
            from .chapter_cache import LANE_FIELDS
            chunk.update(wav_path=mix["wav_path"], mix=mix["mix"],
                         speaker_lanes=[{key: lane[key] for key in LANE_FIELDS} for lane in mix["speaker_lanes"]],
                         censorship_lanes=mix["censorship_lanes"])
        else:
            chunk["wav_path"] = audio(accepted)
        clips.append(chunk)
    return clips
