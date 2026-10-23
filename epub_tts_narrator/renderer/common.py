"""Small deterministic storage and audio-verification primitives shared by the renderer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import HTTPRedirectHandler
import wave


USER_AGENT = "epub-tts-narrator/1"


class RejectRedirects(HTTPRedirectHandler):
    """Refuse redirects by raising the exception a caller's factory builds.

    The factory receives (request, fp, code, headers) so each transport keeps
    its own redaction and error-classification behavior.
    """

    def __init__(self, failure):
        self._failure = failure

    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise self._failure(request, fp, code, headers)


def write_private(path, content):
    """Atomically replace a credential file, private (mode 0600) even on first write."""
    descriptor, temporary = tempfile.mkstemp(prefix="." + Path(path).name + ".", dir=Path(path).parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return Path(path)


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


def pcm_sha256(path, *, format_error=None):
    """SHA-256 over a WAV file's PCM frames; raise format_error unless mono 16-bit."""
    result = hashlib.sha256()
    with wave.open(str(path), "rb") as stream:
        if format_error is not None and (stream.getnchannels() != 1 or stream.getsampwidth() != 2):
            raise format_error
        while data := stream.readframes(65536):
            result.update(data)
    return result.hexdigest()


def verify_flac_pcm(path, expected, ffmpeg, *, decode_error, mismatch_error):
    """Verify a FLAC decodes to the expected pcm_s16le samples via ffmpeg's stream hash."""
    result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                             "-protocol_whitelist", "file,pipe", "-i", str(path),
                             "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "hash", "-hash", "sha256", "-"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=3600, check=False)
    if result.returncode:
        raise decode_error
    if result.stdout.decode("ascii", errors="replace").strip().lower() != "sha256=" + expected:
        raise mismatch_error
