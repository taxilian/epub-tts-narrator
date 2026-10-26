"""Deterministic defaults for a new character voice, without source quotations."""


MAX_PROMPT_CHARACTERS = 4000
AUDIOBOOK_VOICE_CONTEXT = (
    "Adult voice mimicking a character that sounds like"
)
_CONTEXT_PREFIX = AUDIOBOOK_VOICE_CONTEXT + " "
_CONTEXT_PREFIXES = (
    _CONTEXT_PREFIX,
    "Adult audiobook narrator portraying a character in a book. "
    "Create an adult performer's imitation of the following vocal qualities: ",
    "audiobook narrator portraying a character in a book: ",
)


def voice_prompt_description(prompt):
    """Remove only known application wrappers for legacy provenance comparisons."""
    # One released wrapper accidentally supplied two separator spaces. Treat
    # that exact boundary as legacy, but do not lstrip arbitrary description
    # whitespace (three or more spaces can include deliberate custom text).
    legacy_double = _CONTEXT_PREFIX + " "
    if prompt.startswith(legacy_double) and not prompt[len(legacy_double):].startswith(" "):
        return prompt[len(legacy_double):]
    for prefix in _CONTEXT_PREFIXES:
        if prompt.startswith(prefix):
            return prompt[len(prefix):]
    return prompt


def audiobook_voice_prompt(prompt):
    """Apply current adult-performer framing without altering the description."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Voice prompt must be nonempty text")
    description = voice_prompt_description(prompt)
    if not description.strip():
        raise ValueError("Voice prompt must include a nonempty description")
    result = _CONTEXT_PREFIX + description
    if len(result) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"Voice prompt including audiobook context must be at most {MAX_PROMPT_CHARACTERS} characters; "
            f"shorten the description by {len(result) - MAX_PROMPT_CHARACTERS} characters")
    return result


def default_voice_prompt(profile):
    """Use the complete prose design; trait records remain evidence metadata."""
    description = " ".join(voice_prompt_description(profile["description"]).split())
    # Never silently trim a distinctive quality out of an overlong default.
    return audiobook_voice_prompt(description)
