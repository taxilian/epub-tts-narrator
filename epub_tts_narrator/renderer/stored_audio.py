"""Keep renderer WAVs on disk as FLAC files that restore them byte for byte.

A record that names ``something.wav`` names a *logical* WAV: the file on disk
is either that WAV or ``something.flac`` beside it. The FLAC holds the WAV's
16-bit samples losslessly and, in one APPLICATION metadata block, the WAV's
bytes before and after its sample data. Decoding it rebuilds the original WAV
exactly, so every checksum a record or identity binds (``audio_sha256``,
``raw_audio_sha256``, ``reference_sha256``, ``pcm_sha256``) keeps meaning the
same bytes, and storing a WAV this way changes no cache, request or voice
identity.

``store`` converts one WAV and removes it only after the finished FLAC has
been decoded back to the identical bytes. WAVs it cannot hold exactly (float,
24-bit or unusual layouts) are left as they are. Readers use ``locate``,
``audio_exists``, ``audio_hash``, ``open_wave``, ``materialize`` and
``remove_audio`` with the logical name; while both files exist (between the
FLAC being finished and the WAV being removed) the WAV is used.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import wave

from .common import file_hash


STORED_SUFFIX = ".flac"
LOGICAL_SUFFIX = ".wav"
FORMAT = "epub-tts-narrator-stored-wav"
VERSION = 1
_APPLICATION_ID = b"eTTN"
_MAX_WRAPPER_BYTES = 1024 * 1024
_SPOOL_BYTES = 64 * 1024 * 1024
_READ_FRAMES = 1 << 18


class StoredAudioError(ValueError):
    """A stored FLAC is missing its WAV wrapper, is damaged, or does not restore its WAV."""


def stored_path(path: str | Path) -> Path:
    """The FLAC that may hold the logical WAV ``path``."""
    path = Path(path)
    return path.with_suffix(STORED_SUFFIX)


def _logical(path: Path) -> bool:
    return path.suffix.lower() == LOGICAL_SUFFIX


def _regular(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def locate(path: str | Path) -> Path:
    """The file holding the logical audio: the WAV if present, else its stored FLAC, else ``path``."""
    path = Path(path)
    if not _logical(path) or path.exists() or path.is_symlink():
        return path
    stored = stored_path(path)
    return stored if _regular(stored) else path


def is_stored(path: str | Path) -> bool:
    """True when the logical WAV ``path`` currently lives only as its stored FLAC."""
    path = Path(path)
    return _logical(path) and locate(path) != path


def audio_exists(path: str | Path) -> bool:
    """True when the logical audio is present as a regular file in either form."""
    return _regular(locate(path))


def audio_present(path: str | Path) -> bool:
    """``path.exists() or path.is_symlink()`` for a logical WAV in either form."""
    return bool(audio_files(path))


def audio_files(path: str | Path) -> list[Path]:
    """Every file on disk that holds this logical audio (WAV and/or stored FLAC)."""
    path = Path(path)
    files = [path] if path.exists() or path.is_symlink() else []
    if _logical(path):
        stored = stored_path(path)
        if stored.exists() or stored.is_symlink():
            files.append(stored)
    return files


def regular_audio(path: str | Path) -> Path:
    """``common.regular_file`` for a logical WAV in either form; returns the logical path."""
    if not _regular(locate(path)):
        raise ValueError("Expected a regular, nonsymlink file")
    return Path(path)


def remove_audio(path: str | Path) -> list[Path]:
    """Remove the logical audio in both forms; return what was removed."""
    removed = []
    for item in audio_files(path):
        item.unlink(missing_ok=True)
        removed.append(item)
    return removed


def _retry_missing(path: Path, action):
    """Run ``action(located)``; if the WAV vanished because it was just stored, use the FLAC."""
    located = locate(path)
    try:
        return action(located)
    except FileNotFoundError:
        if located == path and _logical(path):
            again = locate(path)
            if again != located:
                return action(again)
        raise


# --- WAV layout --------------------------------------------------------------------------------------

def _wav_layout(head: bytes, size: int) -> dict:
    """Parse a RIFF/WAVE header; return sample layout and data payload bounds."""
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise StoredAudioError("not a RIFF WAVE file")
    position, layout = 12, {}
    while position + 8 <= len(head):
        chunk, length = head[position:position + 4], struct.unpack("<I", head[position + 4:position + 8])[0]
        body = position + 8
        if chunk == b"fmt ":
            if length < 16 or body + 16 > len(head):
                raise StoredAudioError("short WAV format chunk")
            tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", head[body:body + 16])
            if tag == 0xFFFE and length >= 40 and body + 40 <= len(head):
                tag = struct.unpack("<H", head[body + 24:body + 26])[0]
            layout.update(tag=tag, channels=channels, rate=rate, align=align, bits=bits)
        elif chunk == b"data":
            end = min(body + length, size)
            layout.update(start=body, end=end)
            return layout
        position = body + length + (length & 1)
    raise StoredAudioError("WAV header too large or missing its data chunk")


def _storable(layout: dict) -> bool:
    return (layout.get("tag") == 1 and layout.get("bits") == 16 and layout.get("channels") in (1, 2)
            and layout.get("align") == 2 * layout.get("channels", 0) and 1 <= layout.get("rate", 0) <= 655350
            and "start" in layout and (layout["end"] - layout["start"]) % layout["align"] == 0
            and layout["end"] > layout["start"])


# --- FLAC metadata -----------------------------------------------------------------------------------

def _metadata_blocks(stream) -> tuple[list[tuple[int, bytes]], int]:
    """Read FLAC metadata blocks; return them (type, body) and the offset of the first audio frame."""
    if stream.read(4) != b"fLaC":
        raise StoredAudioError("not a FLAC file")
    blocks, offset = [], 4
    while True:
        header = stream.read(4)
        if len(header) != 4:
            raise StoredAudioError("truncated FLAC metadata")
        kind, length = header[0] & 0x7F, int.from_bytes(header[1:], "big")
        if kind == 127:
            raise StoredAudioError("invalid FLAC metadata block")
        body = stream.read(length) if kind != 1 else (stream.seek(length, 1), b"")[1]
        if kind != 1 and len(body) != length:
            raise StoredAudioError("truncated FLAC metadata")
        blocks.append((kind, body))
        offset += 4 + length
        if header[0] & 0x80:
            return blocks, offset


def _wrapper(path: Path) -> dict:
    with path.open("rb") as stream:
        blocks, _ = _metadata_blocks(stream)
    for kind, body in blocks:
        if kind == 2 and body[:4] == _APPLICATION_ID:
            try:
                record = json.loads(body[4:].decode("utf-8"))
                header = base64.b64decode(record["header"], validate=True)
                trailer = base64.b64decode(record["trailer"], validate=True)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                raise StoredAudioError("unreadable stored-WAV wrapper") from exc
            if (record.get("format") != FORMAT or record.get("version") != VERSION
                    or type(record.get("wav_bytes")) is not int or not isinstance(record.get("wav_sha256"), str)):
                raise StoredAudioError("unsupported stored-WAV wrapper")
            return {**record, "header": header, "trailer": trailer}
    raise StoredAudioError("FLAC holds no stored WAV wrapper")


def is_stored_flac(path: str | Path) -> bool:
    """True when ``path`` is a FLAC carrying this module's WAV wrapper (not proof it restores)."""
    try:
        _wrapper(Path(path))
        return True
    except (StoredAudioError, OSError):
        return False


def _with_application_block(flac: bytes, block: bytes) -> bytes:
    """Insert one APPLICATION block straight after STREAMINFO."""
    stream = io.BytesIO(flac)
    blocks, offset = _metadata_blocks(stream)
    if not blocks or blocks[0][0] != 0:
        raise StoredAudioError("FLAC must start with STREAMINFO")
    stream.seek(0)
    raw = stream.read(offset)
    output = bytearray(b"fLaC")
    position = 4
    entries = []
    while position < offset:
        header = raw[position:position + 4]
        length = int.from_bytes(header[1:], "big")
        entries.append(bytearray(raw[position:position + 4 + length]))
        position += 4 + length
    body = _APPLICATION_ID + block
    if len(body) >= 1 << 24:
        raise StoredAudioError("WAV wrapper too large")
    inserted = bytearray([2]) + len(body).to_bytes(3, "big") + body
    entries.insert(1, inserted)
    for entry in entries:
        entry[0] &= 0x7F
    entries[-1][0] |= 0x80
    for entry in entries:
        output += entry
    return bytes(output) + flac[offset:]


# --- decoding ----------------------------------------------------------------------------------------

def _samples(path: Path) -> Iterator[bytes]:
    import soundfile
    try:
        with soundfile.SoundFile(str(path)) as audio:
            if audio.format != "FLAC":
                raise StoredAudioError("stored audio is not FLAC")
            while True:
                block = audio.read(_READ_FRAMES, dtype="int16", always_2d=True)
                if not len(block):
                    return
                yield block.astype("<i2", copy=False).tobytes()
    except (RuntimeError, soundfile.SoundFileError) as exc:
        raise StoredAudioError(f"stored FLAC cannot be decoded: {path.name}") from exc


def _restore(path: Path, sink) -> tuple[str, int]:
    """Write the exact WAV held by stored FLAC ``path`` into ``sink``; verify it; return (sha256, size)."""
    wrapper = _wrapper(path)
    digest_ = hashlib.sha256()
    size = 0
    for part in _iter_restored(path, wrapper):
        digest_.update(part)
        size += len(part)
        if sink is not None:
            sink.write(part)
    checksum = digest_.hexdigest()
    if size != wrapper["wav_bytes"] or checksum != wrapper["wav_sha256"]:
        raise StoredAudioError(f"stored FLAC does not restore its WAV exactly: {path.name}")
    return checksum, size


def _iter_restored(path: Path, wrapper: dict) -> Iterator[bytes]:
    yield wrapper["header"]
    yield from _samples(path)
    yield wrapper["trailer"]


def audio_hash(path: str | Path) -> str:
    """SHA-256 of the logical WAV bytes, wherever they are stored; raise unless a regular file."""
    path = Path(path)

    def run(located: Path) -> str:
        if located.is_symlink() or not located.is_file():
            raise ValueError("Expected a regular, nonsymlink file")
        if located != path:
            return _restore(located, None)[0]
        return file_hash(located)
    return _retry_missing(path, run)


def pcm_sha256(path: str | Path, *, format_error=None) -> str:
    """SHA-256 over the logical WAV's PCM frames (``common.pcm_sha256``), in either form."""
    from .common import pcm_sha256 as wav_pcm_sha256
    path = Path(path)

    def run(located: Path) -> str:
        if located == path:
            return wav_pcm_sha256(located, format_error=format_error)
        with open_wave(path) as stream:
            if format_error is not None and (stream.getnchannels() != 1 or stream.getsampwidth() != 2):
                raise format_error
            result = hashlib.sha256()
            while data := stream.readframes(65536):
                result.update(data)
            return result.hexdigest()
    return _retry_missing(path, run)


@contextmanager
def _restored_stream(stored: Path):
    spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_BYTES, dir=stored.parent)
    try:
        _restore(stored, spool)
        spool.seek(0)
        yield spool
    finally:
        spool.close()


@contextmanager
def open_wave(path: str | Path):
    """``wave.open(path, "rb")`` for the logical WAV in either form."""
    path = Path(path)
    located = _retry_missing(path, lambda item: (item.stat(), item)[1])
    if located == path:
        with wave.open(str(path), "rb") as stream:
            yield stream
        return
    with _restored_stream(located) as spool, wave.open(spool, "rb") as stream:
        yield stream


def read_bytes(path: str | Path) -> bytes:
    """The logical WAV's exact bytes."""
    path = Path(path)

    def run(located: Path) -> bytes:
        if located == path:
            return located.read_bytes()
        buffer = io.BytesIO()
        _restore(located, buffer)
        return buffer.getvalue()
    return _retry_missing(path, run)


@contextmanager
def materialize(path: str | Path, *, directory: str | Path | None = None):
    """Yield a real WAV path holding the logical WAV's exact bytes; a restored copy is deleted after.

    A WAV that exists (or audio that is missing in both forms) is yielded
    unchanged. A restored copy goes to ``directory``, by default the system
    temporary directory, so an interruption never leaves it in a book.
    """
    from .audio_naming import temporary_audio_prefix
    path = Path(path)
    located = locate(path)
    if located == path:
        yield path
        return
    folder = Path(directory) if directory is not None else Path(tempfile.gettempdir())
    descriptor, name = tempfile.mkstemp(prefix=temporary_audio_prefix(path, "restored", suffix=LOGICAL_SUFFIX),
                                        suffix=LOGICAL_SUFFIX, dir=folder)
    restored = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _restore(located, stream)
        yield restored
    finally:
        restored.unlink(missing_ok=True)


def link_audio(source: str | Path, target: str | Path) -> Path:
    """Hard-link the logical WAV ``source`` to the logical WAV ``target`` in the form it is kept in."""
    source, target = Path(source), Path(target)
    located = locate(source)
    if located.is_symlink() or not located.is_file():
        raise ValueError("Expected a regular, nonsymlink file")
    destination = target if located == source else stored_path(target)
    destination.hardlink_to(located)
    return destination


def rename_audio(source: str | Path, target: str | Path) -> list[Path]:
    """Rename the logical WAV ``source`` to the logical WAV ``target``, in every form it is kept in."""
    source, target = Path(source), Path(target)
    moved = []
    for item in audio_files(source):
        destination = target if item == source else stored_path(target)
        item.rename(destination)
        moved.append(destination)
    return moved


def copy_audio(source: str | Path, target: str | Path, *, expected_sha256: str, staging: str | Path) -> Path:
    """Copy the logical WAV ``source`` to the logical WAV ``target`` in its kept form, verified.

    The copy is staged in ``staging`` (on the target's filesystem), checked
    against ``expected_sha256`` (the logical WAV's checksum) and moved into
    place with ``os.replace``. Returns the file written.
    """
    import shutil
    source, target = Path(source), Path(target)
    located = locate(source)
    if located.is_symlink() or not located.is_file():
        raise ValueError("Expected a regular, nonsymlink file")
    destination = target if located == source else stored_path(target)
    staged = Path(staging) / destination.name
    shutil.copyfile(located, staged)
    checksum = file_hash(staged) if located == source else _restore(staged, None)[0]
    if checksum != expected_sha256:
        staged.unlink(missing_ok=True)
        raise StoredAudioError("Copied audio does not match its recorded checksum")
    os.replace(staged, destination)
    return destination


def tool_path(path: str | Path) -> Path:
    """A path an FFmpeg-based tool can decode: the WAV, else its stored FLAC (same samples)."""
    return locate(path)


# --- storing -----------------------------------------------------------------------------------------

def _encode(wav: Path, layout: dict) -> bytes:
    import numpy
    import soundfile
    channels = layout["channels"]
    output = io.BytesIO()
    with wav.open("rb") as source, soundfile.SoundFile(
            output, "w", samplerate=layout["rate"], channels=channels, format="FLAC", subtype="PCM_16") as target:
        source.seek(layout["start"])
        remaining = layout["end"] - layout["start"]
        step = _READ_FRAMES * layout["align"]
        while remaining:
            data = source.read(min(step, remaining))
            if not data:
                raise StoredAudioError("WAV ended inside its data chunk")
            remaining -= len(data)
            samples = numpy.frombuffer(data, dtype="<i2").reshape(-1, channels)
            target.write(samples)
    return output.getvalue()


def _fsync_directory(folder: Path) -> None:
    try:
        descriptor = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def store(path: str | Path, *, expected_sha256: str | None = None) -> Path | None:
    """Replace the WAV ``path`` with its stored FLAC; return the FLAC, or None when it must stay a WAV.

    The WAV is removed only after the finished FLAC has been decoded back to
    the WAV's exact bytes (and, when given, ``expected_sha256``). Already
    stored audio returns its FLAC unchanged. A WAV that FLAC cannot hold
    exactly, or a foreign ``.flac`` of the same name, keeps the WAV.
    """
    from .audio_naming import temporary_audio_prefix
    path = Path(path)
    if not _logical(path):
        raise ValueError("Only .wav audio can be stored as FLAC")
    target = stored_path(path)
    if not (path.exists() or path.is_symlink()):
        if _regular(target):
            _restore(target, None)
            return target
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, nonsymlink file")
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(_MAX_WRAPPER_BYTES)
    try:
        layout = _wav_layout(head, size)
    except StoredAudioError:
        return None
    if not _storable(layout):
        return None
    checksum = file_hash(path)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise StoredAudioError(f"WAV does not match its recorded checksum: {path.name}")
    if target.exists() or target.is_symlink():
        if not _regular(target) or not is_stored_flac(target):
            return None
        try:
            if _restore(target, None)[0] == checksum:
                path.unlink()
                _fsync_directory(path.parent)
                return target
        except (StoredAudioError, OSError, RuntimeError):
            pass
    with path.open("rb") as stream:
        stream.seek(layout["end"])
        trailer = stream.read()
    header = head[:layout["start"]]
    if len(trailer) > _MAX_WRAPPER_BYTES:
        return None
    wrapper = json.dumps({"format": FORMAT, "version": VERSION, "wav_sha256": checksum, "wav_bytes": size,
                          "header": base64.b64encode(header).decode("ascii"),
                          "trailer": base64.b64encode(trailer).decode("ascii")},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    data = _with_application_block(_encode(path, layout), wrapper)
    descriptor, name = tempfile.mkstemp(prefix=temporary_audio_prefix(target, "storing", suffix=STORED_SUFFIX),
                                        suffix=STORED_SUFFIX, dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if _restore(staged, None)[0] != checksum:
            raise StoredAudioError(f"stored FLAC does not restore its WAV exactly: {path.name}")
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    if file_hash(path) != checksum:
        # The WAV changed while it was being stored; it stays authoritative.
        target.unlink(missing_ok=True)
        raise StoredAudioError(f"WAV changed while being stored: {path.name}")
    path.unlink()
    _fsync_directory(path.parent)
    return target


def store_quietly(path: str | Path, *, expected_sha256: str | None = None) -> Path | None:
    """``store`` for a writer that just finished a WAV: a failure keeps the WAV and is not an error."""
    try:
        return store(path, expected_sha256=expected_sha256)
    except (StoredAudioError, OSError, ValueError, RuntimeError, ImportError):
        return None
