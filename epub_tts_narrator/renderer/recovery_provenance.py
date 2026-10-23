"""Read-only reconstruction of hash-bound child evidence for older speech joins."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re

from .common import digest, file_hash, read_json, regular_file


def bind_legacy_recovery_metadata(clips, request, metadata):
    """Restore omitted evidence in memory; the normal identity validator checks it."""
    if (request.get("kind") != "speech" or "recovery" not in request
            or not isinstance(metadata, Mapping) or "recovery" not in metadata):
        return metadata
    audit = metadata["recovery"]
    if not isinstance(audit, Mapping):
        raise ValueError("Invalid sentence recovery generation audit")
    if "child_acceptance_version" in audit:
        return metadata
    rows = audit.get("parts")
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, Mapping) or "acceptance" in row for row in rows)):
        raise ValueError("Legacy sentence recovery has incomplete child evidence")
    directory = Path(clips) / digest(request)
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ValueError("Sentence recovery directories must not use symlinks")
    result = deepcopy(metadata)
    for row in result["recovery"]["parts"]:
        name, child_hash = row.get("audio_file"), row.get("request_hash")
        if (not isinstance(name, str) or not isinstance(child_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", child_hash) is None):
            raise ValueError("Legacy sentence recovery has invalid child identifiers")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3
                or relative.parts[:2] != ("parts", child_hash)):
            raise ValueError("Legacy sentence recovery has an invalid child path")
        audio = directory / relative
        if audio.parent.is_symlink() or audio.parent.parent.is_symlink():
            raise ValueError("Sentence recovery parts must not use symlinks")
        try:
            acceptance = regular_file(audio.parent / "accepted.json")
            if file_hash(acceptance) != row.get("accepted_sha256"):
                raise ValueError("Legacy sentence recovery child acceptance changed")
            if file_hash(regular_file(audio)) != row.get("audio_sha256"):
                raise ValueError("Legacy sentence recovery child audio changed")
            row["acceptance"] = read_json(acceptance)
        except OSError as exc:
            raise ValueError("Legacy sentence recovery child evidence is unavailable") from exc
    result["recovery"]["child_acceptance_version"] = 1
    return result
