"""Deterministic defaults for a new character voice, without source quotations."""


MAX_PROMPT_CHARACTERS = 4000
AUDIOBOOK_VOICE_CONTEXT = "audiobook narrator portraying a character in a book:"
_CONTEXT_PREFIX = AUDIOBOOK_VOICE_CONTEXT + " "


def voice_prompt_description(prompt):
    """Remove only our exact context prefix for legacy provenance comparisons."""
    return prompt[len(_CONTEXT_PREFIX):] if prompt.startswith(_CONTEXT_PREFIX) else prompt


def audiobook_voice_prompt(prompt):
    """Include the intended use once without altering a user's description."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Voice prompt must be nonempty text")
    result = prompt if prompt.startswith(_CONTEXT_PREFIX) else _CONTEXT_PREFIX + prompt
    if len(result) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"Voice prompt including audiobook context must be at most {MAX_PROMPT_CHARACTERS} characters; "
            f"shorten the description by {len(result) - MAX_PROMPT_CHARACTERS} characters")
    return result


def default_voice_prompt(profile):
    """Use the complete prose design; trait records remain evidence metadata."""
    description = voice_prompt_description(" ".join(profile["description"].split()))
    # Never silently trim a distinctive quality out of an overlong default.
    return audiobook_voice_prompt(description)
