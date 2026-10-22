"""Recoverable renderer locks that retain the existing directory sentinels.

A stable sibling file carries an OS lease; a directory remains visible to
older readers such as analysis revision checks. Only a marked directory bound
to this exact lease and inode can be reclaimed after the owning process dies.
Legacy locks and unexpected files always require the operator's attention.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
from pathlib import Path
import stat
import uuid

from .common import write_json
from .interrupts import defer_force_stop


_OWNER = "owner.json"
_FORMAT = "audiobook-render-directory-lock"


def _occupied(path):
    return FileExistsError(errno.EEXIST, "Renderer lock is held or cannot be safely recovered", str(path))


def _identity(info):
    return info.st_dev, info.st_ino


def _read_owner(path, lease_info):
    """Return proven ownership, or refuse without changing the directory."""
    try:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or set(item.name for item in path.iterdir()) != {_OWNER}:
            raise _occupied(path)
        marker = path / _OWNER
        marker_info = marker.lstat()
        if (not stat.S_ISREG(marker_info.st_mode) or marker_info.st_nlink != 1
                or marker_info.st_size > 4096):
            raise _occupied(path)
        owner = json.loads(marker.read_text(encoding="utf-8"))
        expected = {"format", "version", "path", "token", "lease_device", "lease_inode",
                    "directory_device", "directory_inode"}
        if (not isinstance(owner, dict) or set(owner) != expected
                or owner["format"] != _FORMAT or type(owner["version"]) is not int
                or owner["version"] != 1 or owner["path"] != str(path)
                or not isinstance(owner["token"], str) or len(owner["token"]) != 32
                or any(char not in "0123456789abcdef" for char in owner["token"])
                or any(type(owner[key]) is not int for key in
                       ("lease_device", "lease_inode", "directory_device", "directory_inode"))
                or (owner["lease_device"], owner["lease_inode"]) != _identity(lease_info)
                or (owner["directory_device"], owner["directory_inode"]) != _identity(info)):
            raise _occupied(path)
        return owner
    except (OSError, ValueError, UnicodeError):
        raise _occupied(path) from None


def _remove_owned(path, lease_info, owner):
    if _read_owner(path, lease_info) != owner:
        raise _occupied(path)
    (path / _OWNER).unlink()
    path.rmdir()


class DirectoryLock:
    """Path-like ownership handle; rmdir() releases both sentinel and lease."""

    def __init__(self, path, descriptor, lease_info, owner):
        self.path = path
        self._descriptor = descriptor
        self._lease_info = lease_info
        self._owner = owner

    def __fspath__(self):
        return os.fspath(self.path)

    def __str__(self):
        return str(self.path)

    def __getattr__(self, name):
        return getattr(self.path, name)

    def rmdir(self):
        if self._descriptor is None:
            return
        with defer_force_stop():
            descriptor, self._descriptor = self._descriptor, None
            try:
                _remove_owned(self.path, self._lease_info, self._owner)
            finally:
                os.close(descriptor)


def acquire_directory_lock(path: Path) -> DirectoryLock:
    """Acquire a directory lock, recovering only proven abandoned ownership.

    The lease inode is never removed or replaced: unlinking it could allow a
    new process to lock a different inode while an existing owner still writes.
    Explicit force-stop signals are deferred only across marker publication
    and removal. An unmarked lock left by SIGKILL is deliberately not guessed.
    """
    path = Path(path).absolute()
    lease = path.with_name(path.name + ".lease")
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    descriptor = None
    try:
        try:
            descriptor = os.open(lease, flags, 0o600)
            lease_info = os.fstat(descriptor)
            if (not stat.S_ISREG(lease_info.st_mode) or lease_info.st_nlink != 1
                    or lease_info.st_size or stat.S_IMODE(lease_info.st_mode) & 0o077):
                raise _occupied(path)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            current = lease.lstat()
            if not stat.S_ISREG(current.st_mode) or _identity(current) != _identity(lease_info):
                raise _occupied(path)
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ELOOP, errno.EAGAIN, errno.EACCES,
                             errno.ENXIO, errno.EISDIR, errno.ENOTDIR):
                raise _occupied(path) from None
            raise
        with defer_force_stop():
            if path.exists() or path.is_symlink():
                owner = _read_owner(path, lease_info)
                _remove_owned(path, lease_info, owner)
            path.mkdir(mode=0o700)
            info = path.lstat()
            owner = {"format": _FORMAT, "version": 1, "path": str(path), "token": uuid.uuid4().hex,
                     "lease_device": lease_info.st_dev, "lease_inode": lease_info.st_ino,
                     "directory_device": info.st_dev, "directory_inode": info.st_ino}
            try:
                write_json(path / _OWNER, owner)
            except BaseException:
                # Only remove our own empty/marked directory; failed atomic
                # writes may leave unknown artifacts, which remain protected.
                if _identity(path.lstat()) == _identity(info):
                    contents = set(item.name for item in path.iterdir())
                    if not contents:
                        path.rmdir()
                    elif contents == {_OWNER}:
                        _remove_owned(path, lease_info, owner)
                raise
            result = DirectoryLock(path, descriptor, lease_info, owner)
            descriptor = None
            return result
    finally:
        if descriptor is not None:
            os.close(descriptor)
