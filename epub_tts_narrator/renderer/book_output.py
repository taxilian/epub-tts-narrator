"""Resolve fresh and persisted audiobook basenames without changing cache identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
import unicodedata
from typing import Final, TypeAlias

from .common import file_hash


JsonValue: TypeAlias = str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
LEGACY_AUDIOBOOK_FILENAME: Final = "book.m4b"
_SUFFIX: Final = ".m4b"
_MAX_STEM_BYTES: Final = 180
_MAX_FILENAME_BYTES: Final = 240
_FORBIDDEN: Final = frozenset('<>:"/\\|?*')
_GENERIC_STEMS: Final = frozenset({"audio", "audiobook", "book", "output"})
_RESERVED_STEMS: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{number}" for number in range(1, 10)),
     *(f"LPT{number}" for number in range(1, 10))}
)


def _filename_error() -> ValueError:
    return ValueError("audio.audiobook.path must be one nonempty .m4b basename of at most 240 UTF-8 bytes; path separators, absolute/relative paths, hidden/control characters, reserved device names, and leading/trailing spaces or dots are not allowed.")


def _reserved_stem(stem: str) -> bool:
    return stem.split(".", 1)[0].rstrip(" .").upper() in _RESERVED_STEMS


def validate_audiobook_filename(value: JsonValue) -> str:
    """Return an exact safe recorded basename or reject it without rewriting it."""
    if not isinstance(value, str) or not value.endswith(_SUFFIX):
        raise _filename_error()
    stem = value[:-len(_SUFFIX)]
    if (not stem or stem != stem.strip(" .") or len(value.encode("utf-8")) > _MAX_FILENAME_BYTES
            or _reserved_stem(stem)
            or any(character in _FORBIDDEN or unicodedata.category(character).startswith("C")
                   for character in value)):
        raise _filename_error()
    return value


def default_audiobook_filename(book: Mapping[str, JsonValue]) -> str:
    """Derive a bounded, filesystem-safe fresh basename from the actual book title."""
    title = book.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("A nonempty book title is required to create a descriptive audiobook filename")
    normalized = unicodedata.normalize("NFC", title)
    safe = "".join(
        " " if character in _FORBIDDEN or unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    stem = " ".join(safe.split()).strip(" .")
    if not stem:
        stem = "audiobook-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    if len(stem.encode("utf-8")) > _MAX_STEM_BYTES:
        retained: list[str] = []
        size = 0
        for character in stem:
            width = len(character.encode("utf-8"))
            if size + width > _MAX_STEM_BYTES:
                break
            retained.append(character)
            size += width
        stem = "".join(retained).rstrip(" .")
    if stem.casefold() in _GENERIC_STEMS or _reserved_stem(stem):
        stem += " audiobook"
    return validate_audiobook_filename(stem + _SUFFIX)


def recorded_audiobook_filename(previous_audio: Mapping[str, JsonValue] | None) -> str | None:
    """Return the exact basename a previous publication recorded, without touching the disk."""
    if not isinstance(previous_audio, Mapping):
        raise ValueError("The previous audio manifest must be an object with audio.audiobook.path")
    audiobook = previous_audio.get("audiobook", {})
    if not isinstance(audiobook, Mapping):
        raise _filename_error()
    recorded = audiobook.get("path")
    if recorded is None:
        return None
    return validate_audiobook_filename(recorded)


def adopted_audiobook_filename(output_dir: str | Path,
                               previous_audio: Mapping[str, JsonValue] | None) -> str | None:
    """Return the on-disk basename of the recorded audiobook when it was renamed.

    A published audiobook can be renamed outside the renderer. When the
    recorded file is gone and exactly one other audiobook file carries the
    recorded bytes, that file is the same publication under a new name, so the
    publication follows the file instead of renaming it back.
    """
    if not isinstance(previous_audio, Mapping):
        return None
    audiobook = previous_audio.get("audiobook")
    if not isinstance(audiobook, Mapping):
        return None
    recorded = audiobook.get("path")
    checksum = audiobook.get("sha256")
    if not isinstance(recorded, str) or not isinstance(checksum, str):
        return None
    directory = Path(output_dir)
    recorded_path = directory / recorded
    if recorded_path.is_file() and not recorded_path.is_symlink():
        return None
    matches = [path for path in sorted(directory.glob("*.m4b"))
               if path.is_file() and not path.is_symlink() and file_hash(path) == checksum]
    if len(matches) != 1:
        return None
    try:
        return validate_audiobook_filename(matches[0].name)
    except ValueError:
        return None


def resolve_audiobook_filename(book: Mapping[str, JsonValue] | None = None,
                               previous_audio: Mapping[str, JsonValue] | None = None,
                               output_dir: str | Path | None = None) -> str:
    """Honor a recorded name, recognize an intact legacy book, or derive a fresh name."""
    if previous_audio is None:
        if book is None:
            raise ValueError("Book metadata is required when no previous audio manifest exists")
        return default_audiobook_filename(book)
    recorded = recorded_audiobook_filename(previous_audio)
    if recorded is not None:
        if output_dir is not None:
            adopted = adopted_audiobook_filename(output_dir, previous_audio)
            if adopted is not None:
                return adopted
        return recorded
    if output_dir is not None:
        legacy = Path(output_dir) / LEGACY_AUDIOBOOK_FILENAME
        if legacy.is_symlink():
            raise ValueError("Legacy audiobook must be a regular, nonsymlink file: book.m4b")
        if legacy.is_file():
            return LEGACY_AUDIOBOOK_FILENAME
        # A deleted legacy book is recreated under a descriptive name.
        book_metadata = previous_audio.get("book")
        if isinstance(book_metadata, Mapping):
            return default_audiobook_filename(book_metadata)
    raise ValueError("audio.audiobook.path is missing and no intact legacy book.m4b exists; restore the recorded output name or the legacy file before updating this book.")
