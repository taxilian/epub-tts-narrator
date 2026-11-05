"""Bounded, immutable codec artifacts; never recover codec tokens from a waveform.

Engines return ``generated_codes`` in their metadata. The take loop persists
them here, next to the raw take, before normalization and verification, and
replaces the raw tokens with a hash-bound marker. Local adapters, HTTP workers
and chapter services therefore share one persistence and validation path.
"""

import json
import hashlib
import os
from pathlib import Path
import re
import tempfile

from .common import digest, file_hash, regular_file
from .continuity import MAX_SPEAKERS, SPEAKER_TAG, current


NUM_CODEBOOKS = 10
MAX_FRAMES = 2048
_PROVENANCE = ("text", "speakers", "raw_audio_sha256", "context", "request_binding")


def validate_codes(codes, codebook_size):
    if (type(codebook_size) is not int or not 1 <= codebook_size <= 65536
            or not isinstance(codes, list) or len(codes) != NUM_CODEBOOKS
            or any(not isinstance(row, list) for row in codes)
            or not 1 <= len(codes[0]) <= MAX_FRAMES
            or any(len(row) != len(codes[0]) for row in codes)
            or any(type(value) is not int or not 0 <= value < codebook_size for row in codes for value in row)):
        raise ValueError("Invalid continuity codes dimensions or range")


def _validate_turn(text, speakers):
    if (not isinstance(text, str) or not text or len(text) > 100000
            or not isinstance(speakers, list) or not 1 <= len(speakers) <= MAX_SPEAKERS
            or any(not isinstance(asset, str) or re.fullmatch(r"[0-9a-f]{64}", asset) is None
                   for asset in speakers)
            or len(set(speakers)) != len(speakers)):
        raise ValueError("Invalid continuity turn text or speakers")
    tags = [int(index) for index in SPEAKER_TAG.findall(text)]
    if not tags or not text.startswith("<|speaker:") or set(tags) - set(range(len(speakers))):
        raise ValueError("Continuity turn speaker tags must name its declared speakers")


def _validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"version", "codes", "codebook_size", *_PROVENANCE}:
        raise ValueError("Invalid continuity codes schema")
    if (type(payload["version"]) is not int or payload["version"] != 2
            or not isinstance(payload["context"], dict)
            or any(not isinstance(payload[key], str) or re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None
                   for key in ("raw_audio_sha256", "request_binding"))):
        raise ValueError("Invalid continuity codes provenance")
    _validate_turn(payload["text"], payload["speakers"])
    validate_codes(payload["codes"], payload["codebook_size"])
    return payload


def load_codes(path, marker):
    path = Path(path)
    if (not isinstance(marker, dict) or not isinstance(marker.get("file"), str)
            or re.fullmatch(r"[a-f0-9]{64}\.codes\.json", marker["file"]) is None
            or path.name != marker["file"] or path.parent.is_symlink()):
        raise ValueError("Continuity codes artifact checksum or path changed")
    with regular_file(path).open("rb") as handle:
        blob = handle.read(2 * 1024 * 1024 + 1)
    if len(blob) > 2 * 1024 * 1024 or hashlib.sha256(blob).hexdigest() != marker.get("sha256"):
        raise ValueError("Continuity codes artifact checksum or path changed")
    payload = _validate(json.loads(blob))
    if (payload["codebook_size"] != marker.get("codebook_size")
            or any(payload[key] != marker.get(key) for key in _PROVENANCE)):
        raise ValueError("Continuity codec semantics changed")
    return payload


def persist(directory, raw, metadata):
    """Replace ``generated_codes`` with an immutable hash-named artifact marker.

    Outside a chapter transaction (voice creation, previews) codes are not
    history evidence and are simply dropped.
    """
    if "generated_codes" not in metadata:
        return metadata
    generated = metadata["generated_codes"]
    metadata = {key: value for key, value in metadata.items() if key != "generated_codes"}
    scope = current()
    if scope is None:
        return metadata
    if not isinstance(generated, dict) or set(generated) != {"codes", "codebook_size", "text", "speakers"}:
        raise ValueError("Engine returned invalid generated codes")
    payload = _validate({"version": 2, "codes": generated["codes"],
        "codebook_size": generated["codebook_size"], "text": generated["text"],
        "speakers": generated["speakers"], "raw_audio_sha256": file_hash(regular_file(raw)),
        "context": scope["identity"], "request_binding": scope["request_binding"]})
    blob = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    path = Path(directory) / (hashlib.sha256(blob).hexdigest() + ".codes.json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".codes-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if file_hash(regular_file(path)) != hashlib.sha256(blob).hexdigest():
                raise ValueError("Existing immutable continuity codes changed") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    marker = {"version": 2, "file": path.name, "sha256": file_hash(path),
              "codebook_size": payload["codebook_size"], "frames": len(payload["codes"][0]),
              **{key: payload[key] for key in _PROVENANCE}}
    return {**metadata, "continuity_codes": marker}


def history_digest(history):
    """Payload-free identity of the turns a worker actually conditioned on."""
    return digest([{"text_sha256": digest(text), "codes_sha256": digest(codes)} for text, codes in history])
