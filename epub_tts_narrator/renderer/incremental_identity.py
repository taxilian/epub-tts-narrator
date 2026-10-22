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
        record = read_json(regular_file(path))
        request = record.get("request") if isinstance(record, dict) else None
        if (not isinstance(request, dict) or record.get("request_hash") != digest(request)
                or path.parent.name != record["request_hash"]):
            raise ValueError(f"Invalid accepted synthesis request: {path}")
        requests.append(request)
    return requests


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
        if folder.is_symlink():
            raise ValueError(f"Chapter cache must not use symlinks: {folder}")
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


def reanchor_export_sequences(plan, requests, *, completed=()):
    """Keep old request hashes for identical speech after an earlier row splits.

    A render plan's sequence is bookkeeping, not text or a synchronization
    coordinate. Source rows remain untouched; published sync uses their exact
    IDs, offsets, roles, and text. The normal request/voice/audio integrity gates
    still apply. No other metadata is normalized or copied across differences.
    """
    known = {}

    def remember(chunk):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("segment"), dict):
            return
        if type(chunk["segment"].get("sequence")) is int:
            known[digest(_without_sequence(chunk))] = chunk
        for turn in chunk.get("paragraph_turns", []):
            remember(turn)

    for request in requests:
        if request.get("kind") in {"speech", "paragraph_speech"}:
            remember(request.get("chunk"))
    # A current checkpoint wins if several historical requests have different
    # sequence numbers, retaining the chapter's original composition identity.
    for chunk in completed:
        remember(chunk)
    result = deepcopy(plan)
    for chapter in result["chapters"]:
        for index, chunk in enumerate(chapter["chunks"]):
            saved = known.get(digest(_without_sequence(chunk)))
            if saved is not None:
                chapter["chunks"][index] = deepcopy(saved)
    return result
