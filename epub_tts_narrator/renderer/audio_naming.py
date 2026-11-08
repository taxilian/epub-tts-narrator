"""Safe descriptive audio basenames and exact persisted-name validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
import unicodedata
from typing import Final, TypeAlias

from .common import digest


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


def _take_context(request: Mapping[str, JsonValue]) -> list[str]:
    kind = request.get("kind")
    if kind == "create_voice":
        context = [str(request.get("voice_id", "voice"))]
        profile = request.get("profile")
        if isinstance(profile, Mapping):
            name = profile.get("name") or profile.get("display_name")
            if isinstance(name, str):
                context.append(name)
        return context
    context: list[str] = []
    chunk = request.get("chunk")
    if isinstance(chunk, Mapping):
        chunk_id = chunk.get("chunk_id")
        if isinstance(chunk_id, str):
            context.append(chunk_id)
        segment = chunk.get("segment")
        if isinstance(segment, Mapping):
            voice_id = segment.get("voice_id")
            if isinstance(voice_id, str):
                context.append(voice_id)
            simultaneous = segment.get("simultaneous_speakers")
            if isinstance(simultaneous, Sequence) and not isinstance(simultaneous, (str, bytes)):
                simultaneous_ids = []
                for speaker in simultaneous:
                    if isinstance(speaker, Mapping):
                        simultaneous_voice = speaker.get("voice_id")
                        if isinstance(simultaneous_voice, str):
                            simultaneous_ids.append(simultaneous_voice)
                context.extend(sorted(simultaneous_ids))
    voices = request.get("voice_assets")
    if isinstance(voices, Mapping):
        voice_ids = [key for key in voices if isinstance(key, str)]
        context.extend(sorted(voice_ids))
    performer = request.get("performer")
    if isinstance(performer, Mapping):
        performer_voice = performer.get("voice_id")
        if isinstance(performer_voice, str):
            context.append(performer_voice)
    nested = request.get("synthesis_request")
    if isinstance(nested, Mapping):
        context.extend(_take_context(nested))
    identified = bool(context)
    part = request.get("part")
    if isinstance(part, Mapping) and type(part.get("start")) is int and type(part.get("end")) is int:
        context.append(f"part-{part['start']:03d}-{part['end']:03d}")
    if not identified:
        for key in ("recovery_request_hash", "parent_request_hash"):
            identifier = request.get(key)
            if isinstance(identifier, str):
                context.insert(0, identifier[:16])
                break
    if not context:
        context = [str(kind or "speech"), digest(request)[:16]]
    return list(dict.fromkeys(context))


def speech_take_filename(request: Mapping[str, JsonValue], attempt: int, *, role: str = "") -> str:
    """Name a take by its stable request context and public attempt role."""
    if type(attempt) is not int or attempt < 1:
        raise ValueError("Speech attempt number must be a positive integer")
    return descriptive_audio_filename(role or "take", f"attempt-{attempt:03d}", *_take_context(request), suffix=".wav")


_TEMP_RANDOM_CHARACTERS: Final = 8  # NamedTemporaryFile/mkstemp random suffix
_FILESYSTEM_NAME_BYTES: Final = 255


def temporary_audio_prefix(target: str | Path, stage: str, *, suffix: str = "") -> str:
    """Return one hidden staging prefix naming its real destination and stage.

    NamedTemporaryFile/mkstemp append eight random characters plus the
    caller's suffix, so the destination name is truncated to keep the whole
    staged basename within the filesystem name limit.
    """
    stage_name = _safe_part(stage)
    if not stage_name:
        raise ValueError("Staging stage name must contain safe visible characters")
    name = Path(target).name
    stem = _safe_part(name)
    if suffix and stem.endswith(suffix):
        stem = stem[:-len(suffix)]
    if not stem:
        stem = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    budget = max(_FILESYSTEM_NAME_BYTES - 11 - len(stage_name.encode("utf-8"))
                 - len(suffix.encode("utf-8")), 16)
    stem = _truncate(stem, budget) or hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f".{stem}-{stage_name}-"
