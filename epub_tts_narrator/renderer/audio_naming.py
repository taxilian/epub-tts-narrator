"""Safe descriptive audio basenames and exact persisted-name validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
import unicodedata
from typing import Final, TypeAlias


JsonValue: TypeAlias = str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
_MAX_FILENAME_BYTES: Final = 240
_FORBIDDEN: Final = frozenset('<>:"/\\|?*')
_RESERVED_STEMS: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{number}" for number in range(1, 10)),
     *(f"LPT{number}" for number in range(1, 10))}
)
_AUDIO_SUFFIXES: Final = frozenset({".aac", ".flac", ".m4a", ".m4b", ".mp3", ".ogg", ".wav"})


def _filename_error(suffix: str) -> ValueError:
    return ValueError(
        f"Recorded audio filename must be one nonempty {suffix} basename of at most 240 UTF-8 bytes; "
        + "path separators, absolute/relative paths, hidden/control characters, reserved device names, "
        + "and leading/trailing spaces or dots are not allowed."
    )


def _reserved_stem(stem: str) -> bool:
    return stem.split(".", 1)[0].rstrip(" .").upper() in _RESERVED_STEMS


def validate_audio_filename(value: JsonValue, *, suffix: str) -> str:
    """Return an exact safe recorded basename, including explicit legacy names."""
    if suffix not in _AUDIO_SUFFIXES:
        raise ValueError("Unsupported audio filename suffix")
    if not isinstance(value, str) or not value.endswith(suffix):
        raise _filename_error(suffix)
    stem = value[:-len(suffix)]
    if (not stem or stem != stem.strip(" .") or len(value.encode("utf-8")) > _MAX_FILENAME_BYTES
            or _reserved_stem(stem) or Path(value).name != value
            or any(character in _FORBIDDEN or unicodedata.category(character).startswith("C")
                   for character in value)):
        raise _filename_error(suffix)
    return value


def _safe_part(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    safe = "".join(
        " " if character in _FORBIDDEN or unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    return " ".join(safe.replace("..", " ").split()).strip(" .")


def _truncate(value: str, maximum: int) -> str:
    retained: list[str] = []
    size = 0
    for character in value:
        width = len(character.encode("utf-8"))
        if size + width > maximum:
            break
        retained.append(character)
        size += width
    return "".join(retained).rstrip(" .")


def descriptive_audio_filename(purpose: str, *context: str, suffix: str) -> str:
    """Derive one bounded basename from an artifact purpose and its owning context."""
    role = _safe_part(purpose)
    owners = [_safe_part(item) for item in context]
    owners = [item for item in owners if item]
    if not role:
        raise ValueError("Audio filename purpose must contain safe visible characters")
    if not owners:
        source = "\0".join((purpose, *context)).encode("utf-8")
        owners = [hashlib.sha256(source).hexdigest()[:16]]
    stem_budget = _MAX_FILENAME_BYTES - len(suffix.encode("utf-8"))
    tail = "-" + role
    owner = _truncate("-".join(owners), stem_budget - len(tail.encode("utf-8")))
    if not owner:
        owner = hashlib.sha256("\0".join(owners).encode("utf-8")).hexdigest()[:16]
        owner = _truncate(owner, stem_budget - len(tail.encode("utf-8")))
    return validate_audio_filename(owner + tail + suffix, suffix=suffix)
