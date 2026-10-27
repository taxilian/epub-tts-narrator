"""Validate authored voice descriptions without adding performance context."""


MAX_PROMPT_CHARACTERS = 4000
# Historical application text, retained only to recognize saved prompts.
# Context for new voices belongs in the authored description, not a wrapper.
AUDIOBOOK_VOICE_CONTEXT = (
    "Adult voice mimicking a character that sounds like"
)
_LEGACY_CONTEXT_PREFIX = AUDIOBOOK_VOICE_CONTEXT + " "
_LEGACY_CONTEXT_PREFIXES = (
    _LEGACY_CONTEXT_PREFIX,
    "Adult audiobook narrator portraying a character in a book. "
    "Create an adult performer's imitation of the following vocal qualities: ",
    "audiobook narrator portraying a character in a book: ",
)


def voice_prompt_description(prompt):
    """Remove only known application wrappers for legacy provenance comparisons."""
    # One released wrapper accidentally supplied two separator spaces. Treat
    # that exact boundary as legacy, but do not lstrip arbitrary description
    # whitespace (three or more spaces can include deliberate custom text).
    legacy_double = _LEGACY_CONTEXT_PREFIX + " "
    while True:
        if prompt.startswith(legacy_double) and not prompt[len(legacy_double):].startswith(" "):
            prompt = prompt[len(legacy_double):]
            continue
        for prefix in _LEGACY_CONTEXT_PREFIXES:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):]
                break
        else:
            return prompt


def audiobook_voice_prompt(prompt):
    """Pass through the complete description, removing historical wrappers only."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Voice prompt must be nonempty text")
    description = voice_prompt_description(prompt)
    if not description.strip():
        raise ValueError("Voice prompt must include a nonempty description")
    if len(description) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"Voice prompt must be at most {MAX_PROMPT_CHARACTERS} characters; "
            f"shorten the description by {len(description) - MAX_PROMPT_CHARACTERS} characters")
    return description


def default_voice_prompt(profile):
    """Use the complete prose design; trait records remain evidence metadata."""
    # Never silently trim a distinctive quality out of an overlong default.
    return audiobook_voice_prompt(profile["description"])
