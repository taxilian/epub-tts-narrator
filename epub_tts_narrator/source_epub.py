"""Local EPUB location hints, separate from immutable analysis/source identities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from .epub import MAX_ARCHIVE_BYTES


SOURCE_EPUB_HINT = "source-epub.json"


def source_epub_hint(directory, book_id) -> Path | None:
    """Read a book-bound local hint without opening the EPUB itself.

    A moved or unavailable original remains an actionable path, not a reason
    to silently omit its cover. Artwork extraction checks the actual SHA-256.
    """
    path = Path(directory) / SOURCE_EPUB_HINT
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
        raise ValueError("Source EPUB hint must be a small regular JSON file")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError):
        raise ValueError("Invalid source EPUB hint; supply --epub with the original book") from None
    if (not isinstance(record, dict)
            or set(record) != {"format", "version", "book_id", "path"}
            or record.get("format") != "epub-tts-narrator-source-epub"
            or type(record.get("version")) is not int or record["version"] != 1
            or record.get("book_id") != book_id
            or not isinstance(record.get("path"), str) or not record["path"].strip()
            or "\x00" in record["path"] or not Path(record["path"]).is_absolute()):
        raise ValueError("Source EPUB hint does not match this book; supply --epub with the original book")
    return Path(record["path"])


def remember_source_epub(directory, epub, book_id) -> None:
    """Atomically remember an explicitly supplied, checksum-matching EPUB.

    The hint is local metadata, deliberately excluded from the portable source
    fingerprint, analyzer caches, and chapter/voice generation identities.
    """
    epub = Path(epub).expanduser().resolve()
    info = epub.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Source EPUB must be a regular file within the archive size limit")
    with epub.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != book_id:
        raise ValueError("Source EPUB does not match the analyzed book ID")
    _write_source_epub_hint(directory, epub, book_id)


def copy_source_epub_hint(source_directory, destination_directory, book_id) -> bool:
    """Carry local location metadata through export even if the EPUB moved.

    This preserves a previously recorded hint, not a claim that the file still
    exists or matches. Publication must validate the actual EPUB before use.
    """
    epub = source_epub_hint(source_directory, book_id)
    if epub is None:
        return False
    _write_source_epub_hint(destination_directory, epub, book_id)
    return True


def _write_source_epub_hint(directory, epub, book_id):
    directory = Path(directory)
    target = directory / SOURCE_EPUB_HINT
    if target.is_symlink():
        raise ValueError("Source EPUB hint must not be a symlink")
    record = {"format": "epub-tts-narrator-source-epub", "version": 1,
              "book_id": book_id, "path": str(epub)}
    encoded = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if target.is_file() and target.read_text(encoding="utf-8") == encoded:
        return
    descriptor, name = tempfile.mkstemp(prefix=".source-epub-", suffix=".json", dir=directory)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
