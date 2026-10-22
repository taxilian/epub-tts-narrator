"""Persist a server credential without rewriting the user's global TOML."""

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import tomllib

from .config import _credential


def _read(path):
    try:
        text = path.read_bytes().decode("utf-8")
        supplied = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid TOML configuration.") from None
    server = supplied.get("server", {})
    if not isinstance(server, dict):
        raise ValueError("server must be a configuration table")
    key = server.get("api_key", "")
    _credential(key, "server.api_key")
    return text, supplied, key


def _with_key(text, supplied, key):
    """Make one lossless edit, accepting it only after a full TOML comparison.

    Candidate positions cover table, dotted-key and inline-table spellings.
    They need not distinguish comments/string contents: the parser gate rejects
    every edit that doesn't change exactly server.api_key and nothing else.
    """
    expected = deepcopy(supplied)
    expected.setdefault("server", {})["api_key"] = key
    literal = json.dumps(key)
    newline = "\r\n" if "\r\n" in text else "\n"

    def candidates():
        if "api_key" in supplied.get("server", {}):
            # All empty TOML string spellings, including multiline strings and
            # whitespace-trimming backslash continuations. Other empty settings
            # or quote-like content are excluded by the full-document check.
            for match in re.finditer(r'"""[\s\\]*?"""|\'\'\'[\s]*?\'\'\'|""|\'\'', text):
                yield text[:match.start()] + literal + text[match.end():]
            return
        yield text + newline + "[server]" + newline + f"api_key = {literal}" + newline
        yield f"server.api_key = {literal}" + newline + text
        for match in re.finditer(r"(?m)^[ \t]*\[[^\n]+\][ \t]*(?:#[^\n]*)?(?:\r?\n|$)", text):
            offset = match.end()
            separator = "" if text[:offset].endswith("\n") else newline
            yield text[:offset] + separator + f"api_key = {literal}" + newline + text[offset:]
        for match in re.finditer(r"\{", text):
            offset = match.end()
            for separator in (", ", " "):
                yield text[:offset] + f" api_key = {literal}" + separator + text[offset:]

    for candidate in candidates():
        try:
            parsed = tomllib.loads(candidate)
        except tomllib.TOMLDecodeError:
            continue
        if parsed == expected:
            return candidate
    raise ValueError("Could not safely add server.api_key to the TOML configuration; "
                     "use a [server] table with api_key = \"\"")


def _write_private(path, content):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ensure_server_api_key(path: Path) -> tuple[str, bool]:
    """Return the saved key and whether it was generated, or fail before serving.

    The stable sibling lock serializes first starts across processes even when
    the config is atomically replaced. It contains no credentials and must not
    be unlinked on unlock, which could let a third process bypass a waiter.
    """
    path = Path(path).resolve()
    _, _, key = _read(path)
    if key:
        return key, False
    lock_path = path.with_name("." + path.name + ".server-key.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            text, supplied, key = _read(path)
            if key:
                return key, False
            key = secrets.token_urlsafe(32)
            updated = _with_key(text, supplied, key)
            # An editor may not honor our lock. Abort if we observe an edit
            # since the snapshot used to construct the credential insertion.
            if path.read_bytes() != text.encode("utf-8"):
                raise ValueError("Configuration changed while creating server.api_key; rerun serve")
            _write_private(path, updated)
            return key, True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
