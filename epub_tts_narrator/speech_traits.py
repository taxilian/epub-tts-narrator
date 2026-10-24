from __future__ import annotations

import unicodedata


MAX_SPEECH_TRAITS_CHARACTERS = 400


def validate_speech_traits(value):
    """Return valid speech-trait text unchanged."""
    if not isinstance(value, str):
        raise ValueError("speechTraits must be a string")
    if not value.strip():
        raise ValueError("speechTraits must not be empty or whitespace")
    if len(value) > MAX_SPEECH_TRAITS_CHARACTERS:
        raise ValueError(f"speechTraits must be at most {MAX_SPEECH_TRAITS_CHARACTERS} characters")
    if (len(value.splitlines()) != 1
            or any(unicodedata.category(character) in {"Zl", "Zp"} for character in value)):
        raise ValueError("speechTraits must be a single line")
    if any(character in "[]<>" for character in value):
        raise ValueError("speechTraits must be plain text without brackets or markup")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("speechTraits must not contain control or invisible characters")
    return value


def speech_traits(profile):
    """Return a validated optional speechTraits value."""
    if "speechTraits" not in profile:
        return ""
    return validate_speech_traits(profile["speechTraits"])
