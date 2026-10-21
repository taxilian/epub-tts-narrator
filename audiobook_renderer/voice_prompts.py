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
    """Include audible trait fields, never their book-evidence quotations."""
    description = voice_prompt_description(" ".join(profile["description"].split()))
    traits, seen = [], set()
    for trait in profile.get("traits", []):
        name = " ".join(trait["name"].split())
        value = " ".join(trait["value"].split())
        key = (name.casefold(), value.casefold())
        if name and value and key not in seen:
            seen.add(key)
            traits.append(f"{name}: {value}")
    prompt = _CONTEXT_PREFIX + description
    if traits:
        prompt += "\nVoice traits: " + "; ".join(traits)
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        prefix = prompt[:MAX_PROMPT_CHARACTERS - 1]
        boundary = max(prefix.rfind(" "), prefix.rfind("\n"))
        if boundary > 0:
            prefix = prefix[:boundary]
        prompt = prefix.rstrip() + "…"
    return prompt
