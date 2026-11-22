"""Preserve synthesis identities across export-only sequence renumbering."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from .common import digest, read_json, regular_file


def accepted_requests(work):
    """Read bound request metadata, without accepting or trusting its audio."""
    requests = []
    for path in sorted((Path(work) / "clips").glob("*/accepted.json")):
        if path.parent.is_symlink():
            raise ValueError(f"Cached accepted audio must not use symlinks: {path.parent}")
        request = _check_accepted(path, read_json(regular_file(path)))
        if request is not None:
            requests.append(request)
    return requests


ANCHOR_SNAPSHOT = "accepted-plan-rows-snapshot.json"


def _check_accepted(path, record):
    """``record``'s request when it passed, else None; raise as ``accepted_requests`` does for a bad record."""
    request = record.get("request") if isinstance(record, dict) else None
    if (not isinstance(request, dict) or record.get("request_hash") != digest(request)
            or path.parent.name != record["request_hash"]):
        raise ValueError(f"Invalid accepted synthesis request: {path}")
    if not isinstance(record.get("report"), dict) or record["report"].get("passed") is not True:
        return None
    return request


def _anchors(chunk, found):
    """``[key, sequence, turn sequences]`` of ``chunk`` and each of its turns, as ``reanchor_export_sequences`` remembers them.

    ``key`` is the chunk's digest without sequences; a turn sequence is
    ``[value]``, or ``[]`` when the turn has none.
    """
    if not isinstance(chunk, dict) or not isinstance(chunk.get("segment"), dict):
        return found
    if type(chunk["segment"].get("sequence")) is int:
        turns = [[turn["segment"]["sequence"]] if isinstance(turn, dict) and isinstance(turn.get("segment"), dict)
                 and "sequence" in turn["segment"] else [] for turn in chunk.get("paragraph_turns", [])]
        found.append([digest(_without_sequence(chunk)), chunk["segment"]["sequence"], turns])
    for turn in chunk.get("paragraph_turns", []):
        _anchors(turn, found)
    return found


def _request_anchors(request):
    return _anchors(request.get("chunk"), []) if request.get("kind") in {"speech", "paragraph_speech"} else []


def _plan_row(request):
    """What planning needs from one passing accepted request (``workflow.make_render_plan``).

    ``anchors`` for ``reanchor_export_sequences``; ``speech`` (``[chunk ID,
    reuse digest]``) or ``paragraph`` (one per turn) for the saved boundaries
    planning keeps while their audio is still reusable (``workflow._reuse_digest``).
    """
    from .workflow import _reuse_digest
    row = {"anchors": _request_anchors(request)}
    chunk = request.get("chunk", {})
    if request.get("kind") == "speech":
        if isinstance(chunk, dict) and isinstance(chunk.get("chunk_id"), str):
            row["speech"] = [chunk["chunk_id"], _reuse_digest(chunk)]
    elif request.get("kind") == "paragraph_speech":
        turns = chunk.get("paragraph_turns", []) if isinstance(chunk, dict) else []
        if isinstance(turns, list) and len(turns) >= 2 and all(isinstance(turn, dict)
                and isinstance(turn.get("chunk_id"), str) for turn in turns):
            row["paragraph"] = [[turn["chunk_id"], _reuse_digest(turn)] for turn in turns]
    return row


def accepted_plan_rows(work):
    """``_plan_row`` of every passing accepted take, in ``accepted_requests`` order.

    Kept in a stat-checked snapshot (``file_snapshot``): planning a render
    read every take's ``accepted.json`` of the book (tens of thousands) only
    for these few values.
    """
    import os
    from . import file_snapshot
    clips = Path(work) / "clips"
    try:
        names = sorted(name for name in os.listdir(clips) if not name.startswith("."))
    except (FileNotFoundError, NotADirectoryError):
        return []
    snapshot = Path(work) / ANCHOR_SNAPSHOT
    previous = file_snapshot.load(snapshot, "accepted-plan-rows", None)
    current, rows = {}, []
    for name in names:
        path = clips / name / "accepted.json"
        key = file_snapshot.stat_key(path)
        if key is None:
            continue
        if (clips / name).is_symlink():
            raise ValueError(f"Cached accepted audio must not use symlinks: {path.parent}")
        saved = previous.get(name)
        if isinstance(saved, list) and len(saved) == 2 and saved[0] == key:
            row = saved[1]
            current[name] = saved
        else:
            request = _check_accepted(path, read_json(regular_file(path)))
            row = None if request is None else _plan_row(request)
            if file_snapshot.stat_key(path) == key:  # unchanged while it was read
                current[name] = [key, row]
        if row is not None:
            rows.append(row)
    if current != previous:
        file_snapshot.save(snapshot, "accepted-plan-rows", None, current)
    return rows


def completed_chunks(work, chapters):
    """Read checksum-bound old plan chunks, including nonspoken pause controls.

    Audio, verification, and current synthesis settings still need the normal
    checkpoint restoration checks. These records authorize only retention of
    export sequence numbers when every other chunk field matches exactly.
    """
    chunks = []
    for chapter in chapters:
        path = Path(work) / "completed-chapters" / (chapter["id"] + ".json")
        if not path.exists() and not path.is_symlink():
            continue
        checkpoint = read_json(regular_file(path))
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a JSON object: {path}")
        checksum = checkpoint.pop("record_sha256", None)
        key = checkpoint.get("cache_key")
        if (checksum != digest(checkpoint)
                or checkpoint.get("format") != "audiobook-chapter-checkpoint"
                or checkpoint.get("version") != 1
                or not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)):
            raise ValueError(f"Checkpoint integrity check failed: {path}")
        folder = Path(work) / "chapters" / key
        if folder.is_symlink() or folder.parent.is_symlink():
            raise ValueError(f"Chapter cache must not use symlinks: {folder}")
        if not folder.exists():
            continue
        record = read_json(regular_file(folder / "chapter.json"))
        if not isinstance(record, dict):
            raise ValueError(f"Invalid chapter cache: {folder}")
        checksum = record.pop("record_sha256", None)
        identity = record.get("identity")
        if (checksum != digest(record) or not isinstance(identity, dict)
                or digest(identity) != key or identity.get("context") != checkpoint.get("context")
                or not isinstance(identity.get("clips"), list)):
            raise ValueError(f"Invalid chapter cache: {folder}")
        for clip in identity["clips"]:
            if not isinstance(clip, dict):
                raise ValueError(f"Invalid cached chapter clip: {folder}")
            chunks.append({name: value for name, value in clip.items()
                           if name not in {"audio_sha256", "speaker_lanes", "mix"}})
    return chunks


def _without_sequence(chunk):
    result = deepcopy(chunk)
    row = result.get("segment")
    if isinstance(row, dict):
        row.pop("sequence", None)
    for turn in result.get("paragraph_turns", []):
        row = turn.get("segment") if isinstance(turn, dict) else None
        if isinstance(row, dict):
            row.pop("sequence", None)
    return result


def reanchor_export_sequences(plan, requests=(), *, completed=(), anchors=()):
    """Keep old request hashes for identical speech after an earlier row splits.

    A render plan's sequence is bookkeeping, not text or a synchronization
    coordinate. Source rows remain untouched; published sync uses their exact
    IDs, offsets, roles, and text. The normal request/voice/audio integrity gates
    still apply. No other metadata is normalized or copied across differences.
    ``anchors`` (``accepted_anchors``) stand for accepted ``requests`` without
    reading them: a chunk matching one apart from its sequences takes the
    saved sequences, which is exactly the saved chunk.
    """
    known = {}
    for key, sequence, turns in [*anchors, *(row for request in requests for row in _request_anchors(request))]:
        known[key] = (sequence, turns)
    # A current checkpoint wins if several historical requests have different
    # sequence numbers, retaining the chapter's original composition identity.
    for chunk in completed:
        for key, sequence, turns in _anchors(chunk, []):
            known[key] = (sequence, turns)
    result = deepcopy(plan)
    for chapter in result["chapters"]:
        for index, chunk in enumerate(chapter["chunks"]):
            saved = known.get(digest(_without_sequence(chunk)))
            if saved is not None:
                chapter["chunks"][index] = _with_sequences(chunk, *saved)
    return result


def _with_sequences(chunk, sequence, turns):
    result = deepcopy(chunk)
    result["segment"]["sequence"] = sequence
    for turn, saved in zip(result.get("paragraph_turns", []), turns):
        if isinstance(turn, dict) and isinstance(turn.get("segment"), dict):
            turn["segment"].pop("sequence", None)
            if saved:
                turn["segment"]["sequence"] = saved[0]
    return result
