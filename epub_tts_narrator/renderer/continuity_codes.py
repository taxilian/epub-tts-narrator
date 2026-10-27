"""Bounded, immutable codec artifacts; never recover codec tokens from a waveform."""

import json
import hashlib
import os
from pathlib import Path
import re
import tempfile

from .common import digest, file_hash, regular_file
from .continuity import current


_PROVENANCE = ("text", "runtime", "raw_audio_sha256", "context", "request_binding")


def _validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"version", "codes", "codebook_size", *_PROVENANCE}:
        raise ValueError("Invalid continuity codes schema")
    if (not isinstance(payload["text"], str) or not payload["text"]
            or len(payload["text"]) > 100000 or not isinstance(payload["context"], dict)
            or any(not isinstance(payload[key], str) or re.fullmatch(r"[0-9a-f]{64}", payload[key]) is None
                   for key in ("runtime", "raw_audio_sha256", "request_binding"))):
        raise ValueError("Invalid continuity codes provenance")
    size, codes = payload["codebook_size"], payload["codes"]
    if (type(payload["version"]) is not int or payload["version"] != 1
            or type(size) is not int or not 1 <= size <= 65536
            or not isinstance(codes, list) or len(codes) != 10
            or any(not isinstance(row, list) for row in codes)
            or not 1 <= len(codes[0]) <= 2048
            or any(len(row) != len(codes[0]) for row in codes)
            or any(type(value) is not int or not 0 <= value < size for row in codes for value in row)):
        raise ValueError("Invalid continuity codes dimensions or range")
    return payload


def load_codes(path, marker):
    path = Path(path)
    if (not isinstance(marker, dict) or not isinstance(marker.get("file"), str)
            or re.fullmatch(r"[a-zA-Z0-9-]+\.codes\.json", marker["file"]) is None
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


def previous_takes(engine, model):
    scope = current()
    if scope is None:
        return {}
    import numpy as np
    previous = []
    for entry in scope["takes"]:
        marker = entry["marker"]
        if marker["runtime"] != digest(engine.identity):
            raise ValueError("Continuity model/runtime changed; cannot reuse codec history")
        payload = load_codes(entry["path"], marker)
        if (model.model.config.audio_decoder.num_codebooks != 10
                or payload["codebook_size"] != model.model.config.audio_decoder.vocab_size):
            raise ValueError("Continuity codec configuration differs from the current model")
        previous.append((marker["text"], np.array(payload["codes"], dtype=np.int64)))
    return {"previous_takes": previous}


def save_codes(engine, model, segment, audio):
    scope = current()
    if scope is None:
        return {}
    codes = segment.codes
    if (getattr(codes, "ndim", None) != 2 or getattr(codes.dtype, "kind", None) not in ("i", "u")
            or codes.shape[0] != 10 or not 1 <= codes.shape[1] <= 2048):
        raise ValueError("Continuity requires generated integer codec tokens")
    if model.model.config.audio_decoder.num_codebooks != 10:
        raise ValueError("Continuity requires the ten-codebook Drama codec")
    provenance = {"text": segment.text, "raw_audio_sha256": file_hash(audio),
        "runtime": digest(engine.identity), "context": scope["identity"],
        "request_binding": scope["request_binding"]}
    payload = _validate({"version": 1, "codes": codes.tolist(), **provenance,
                         "codebook_size": model.model.config.audio_decoder.vocab_size})
    blob = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    path = Path(audio).parent / (hashlib.sha256(blob).hexdigest() + ".codes.json")
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
    return {"continuity_codes": {"version": 1, "file": path.name,
        "sha256": file_hash(path), "codebook_size": payload["codebook_size"],
        **provenance}}
