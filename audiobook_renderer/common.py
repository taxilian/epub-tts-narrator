"""Small deterministic storage primitives shared by the renderer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def read_json(path):
    def reject(_):
        raise ValueError("Non-finite JSON number")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def write_json(path, value):
    """Atomic replace, intended only for renderer-owned work state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def regular_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, nonsymlink file")
    return path
