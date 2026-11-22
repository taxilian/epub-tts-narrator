"""Remember what was read from many small files, keyed by each file's stat, so an unchanged file is not read again.

A render's start reads every take's ``accepted.json`` and speech-cache index
record (about 30,000 of each per book, and earlier series books' too), which
took minutes while chapters waited. Each loader keeps one snapshot file of what
it derived from those files: per file, its ``stat_key`` and the result. On the
next start a file whose key is unchanged gives its saved result; any other
file is read, parsed and checked as before.

The key is the file's type, size, modification and change times (ns) and
inode, so an edit, an atomic replacement (a new inode) or a rename shows. The
snapshot is also bound to ``code_fingerprint``, a hash of the package's source,
so a change in how results are derived (validation, identities, history
rules) reads everything again once. A missing, corrupt or foreign snapshot is
simply an empty one. Snapshots are derived state only, never a record of what
was accepted: deleting one only makes the next start slower.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading

FORMAT = "audiobook-file-snapshot"
VERSION = 1

_fingerprint = None
_fingerprint_lock = threading.Lock()


def stat_key(path):
    """``[type, size, mtime_ns, ctime_ns, inode]`` of ``path`` itself (not a link's target), or None when absent."""
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return [stat.S_IFMT(info.st_mode), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino]


def code_fingerprint():
    """A hash of every Python source file of the package (computed once per process)."""
    global _fingerprint
    with _fingerprint_lock:
        if _fingerprint is None:
            package = Path(__file__).resolve().parent.parent
            hasher = hashlib.sha256()
            for path in sorted(package.rglob("*.py")):
                hasher.update(str(path.relative_to(package)).encode() + b"\0")
                hasher.update(path.read_bytes() + b"\0")
            _fingerprint = hasher.hexdigest()
        return _fingerprint


def load(path, kind, binding, *, code=True):
    """The saved ``{name: [key, result]}`` for ``kind`` and ``binding`` (anything JSON), or ``{}``.

    ``code=False`` for results that never depend on how this package derives
    them (a file's own hash); ``binding`` then carries their format version.
    """
    try:
        with open(path, "rb") as stream:
            saved = json.loads(stream.read())
    except (OSError, ValueError):
        return {}
    if (not isinstance(saved, dict) or saved.get("format") != FORMAT or saved.get("version") != VERSION
            or saved.get("kind") != kind or saved.get("code") != (code_fingerprint() if code else None)
            or saved.get("binding") != binding or not isinstance(saved.get("files"), dict)):
        return {}
    return saved["files"]


def save(path, kind, binding, files, *, code=True):
    """Replace the snapshot atomically; a failure to write only loses the speed-up."""
    path = Path(path)
    data = json.dumps({"format": FORMAT, "version": VERSION, "kind": kind, "code": code_fingerprint() if code else None,
                       "binding": binding, "files": files}, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)
    temporary = None
    try:
        if path.is_symlink() or not path.parent.is_dir():
            return
        fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
        os.replace(temporary, path)
        temporary = None
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
