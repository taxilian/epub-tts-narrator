from __future__ import annotations

import re
import unicodedata


MAX_SPEECH_TRAITS_CHARACTERS = 400

# Words that name or describe a way of speaking tied to a place or people.
_ACCENT_WORDS = re.compile(r"\b(?:accent(?:s|ed)?|dialect(?:s|al)?|brogue|received pronunciation|general american)\b",
                           re.IGNORECASE)

# Said to a model whose description named an accent.
ACCENT_FREE = ("must not name or describe an accent or dialect ({words}): each voice's accent is chosen separately "
               "and added when the voice is made")


def accent_words(text):
    """The words in ``text`` that name or describe an accent, in order, once each.

    A voice description a model writes never carries an accent: the accent tool
    (``renderer/accent_assignment.py``) chooses each voice's accent, and the
    application adds it when the voice is designed (``renderer/speech_traits.with_accent``).
    """
    return list(dict.fromkeys(match.group().casefold() for match in _ACCENT_WORDS.finditer(text or "")))


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
