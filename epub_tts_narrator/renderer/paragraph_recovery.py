"""Persist a failed paragraph's original passage boundaries for safe resumption."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

from .common import digest, read_json, regular_file, write_json


_FORMAT = "audiobook-paragraph-fallback"
_FIELDS = {"format", "version", "request_hash", "paragraph_chunk_id",
           "member_chunk_ids", "record_sha256"}


def _chunk_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32,64}", value) is not None


def _paragraph_id(members):
    return digest({"paragraph_group_version": 1, "member_chunk_ids": members})[:32]


def _directory(work):
    work = Path(work)
    directory = work / "paragraph-fallbacks"
    if work.is_symlink() or directory.is_symlink():
        raise ValueError("Paragraph fallback directories must not use symlinks")
    if directory.exists() and not directory.is_dir():
        raise ValueError("Paragraph fallback location must be a directory")
    return directory


def _validate_record(record, path):
    if (not isinstance(record, dict) or set(record) != _FIELDS
            or record.get("format") != _FORMAT
            or type(record.get("version")) is not int or record["version"] != 1):
        raise ValueError(f"Invalid paragraph fallback record: {path}")
    request_hash, members = record["request_hash"], record["member_chunk_ids"]
    if (not isinstance(request_hash, str) or re.fullmatch(r"[0-9a-f]{64}", request_hash) is None
            or path.name != request_hash + ".json"
            or not isinstance(members, list) or not 2 <= len(members) <= 1000
            or any(not _chunk_id(member) for member in members)
            or len(set(members)) != len(members)
            or record["paragraph_chunk_id"] != _paragraph_id(members)):
        raise ValueError(f"Invalid paragraph fallback identity: {path}")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if record["record_sha256"] != digest(body):
        raise ValueError(f"Paragraph fallback integrity check failed: {path}")
    return record


def _request_record(request):
    if not isinstance(request, dict) or request.get("kind") != "paragraph_speech":
        raise ValueError("Paragraph fallback requires a paragraph synthesis request")
    chunk = request.get("chunk")
    if not isinstance(chunk, dict) or chunk.get("silent_control") is not False:
        raise ValueError("Paragraph fallback requires a spoken paragraph")
    turns, text = chunk.get("paragraph_turns"), chunk.get("spoken_text")
    if (not isinstance(turns, list) or not 2 <= len(turns) <= 1000
            or not isinstance(text, str) or not text
            or any(not isinstance(turn, dict) or "paragraph_turns" in turn
                   or turn.get("silent_control") is not False
                   or not _chunk_id(turn.get("chunk_id"))
                   or not isinstance(turn.get("spoken_text"), str)
                   or not turn["spoken_text"]
                   for turn in turns)):
        raise ValueError("Invalid paragraph fallback passages")
    members = [turn["chunk_id"] for turn in turns]
    if len(set(members)) != len(members) or chunk.get("chunk_id") != _paragraph_id(members):
        raise ValueError("Paragraph fallback does not match its original passage identities")
    if "".join(turn["spoken_text"] for turn in turns) != text:
        raise ValueError("Paragraph fallback text differs from its original passages")
    row = chunk.get("segment")
    rows = [turn.get("segment") for turn in turns]
    if (not isinstance(row, dict) or row.get("text") != text
            or any(not isinstance(part, dict) or not part.get("block_id")
                   or part.get("block_id") != row.get("block_id") for part in rows)):
        raise ValueError("Paragraph fallback must retain one source block and its exact text")
    body = {"format": _FORMAT, "version": 1, "request_hash": digest(request),
            "paragraph_chunk_id": chunk["chunk_id"], "member_chunk_ids": members}
    return dict(body, record_sha256=digest(body))


def record_paragraph_fallback(work, request):
    """Record original boundaries before any replacement passages are generated.

    The marker contains only identifiers and checksums. The request hash points
    to the retained failed attempts for audit; no script or voice data is copied.
    Existing records are immutable, and identical calls are idempotent.
    """
    record = _request_record(request)
    directory = _directory(work)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (record["request_hash"] + ".json")

    def existing():
        saved = _validate_record(read_json(regular_file(path)), path)
        if saved != record:
            raise ValueError(f"Paragraph fallback conflicts with the saved request: {path}")

    if path.exists() or path.is_symlink():
        existing()
        return path
    # Publish exclusively so concurrent attempts cannot replace a recorded choice.
    with tempfile.TemporaryDirectory(prefix=".fallback-", dir=directory) as temporary:
        staged = Path(temporary) / "record.json"
        write_json(staged, record)
        try:
            os.link(staged, path)
        except FileExistsError:
            existing()
    return path


def load_paragraph_fallbacks(work):
    """Return original passage IDs that must stay ungrouped when replanning."""
    directory = _directory(work)
    if not directory.exists():
        return set()
    members = set()
    for path in sorted(directory.glob("*.json")):
        record = _validate_record(read_json(regular_file(path)), path)
        members.update(record["member_chunk_ids"])
    return members


def paragraph_retry_episode(work, chunk):
    """Bind a paragraph and its fallback children to one run-local allowance.

    This scheduling identity stays outside synthesis requests: introducing a
    shared budget must not move failed takes or discard their saved evidence.
    Existing immutable fallback markers restore the relation on a later run.
    Overlapping legacy groups share one conservative allowance for their whole
    connected component instead of granting each child a separate budget.
    """
    if not isinstance(chunk, dict):
        raise ValueError("Paragraph retry budget requires a chunk object")
    if "paragraph_turns" in chunk:
        record = _request_record({"kind": "paragraph_speech", "chunk": chunk})
        groups = {record["paragraph_chunk_id"]}
        members = set(record["member_chunk_ids"])
    else:
        # Ordinary caller-supplied chunks need not use the planner's digest ID
        # format. They cannot match validated fallback members and therefore
        # have no shared paragraph allowance. Grouped requests remain strict.
        if not _chunk_id(chunk.get("chunk_id")):
            return None
        members, groups = {chunk["chunk_id"]}, set()
    directory = _directory(work)
    if directory.exists():
        records = [_validate_record(read_json(regular_file(path)), path)
                   for path in sorted(directory.glob("*.json"))]
        while True:
            discovered = [record for record in records
                          if record["paragraph_chunk_id"] not in groups
                          and members.intersection(record["member_chunk_ids"])]
            if not discovered:
                break
            for record in discovered:
                groups.add(record["paragraph_chunk_id"])
                members.update(record["member_chunk_ids"])
    if not groups:
        return None
    return digest({"paragraph_retry_groups": sorted(groups)})
