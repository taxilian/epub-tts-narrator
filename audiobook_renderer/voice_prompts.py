"""Deterministic defaults for a new character voice, without source quotations."""


MAX_PROMPT_CHARACTERS = 4000


def default_voice_prompt(profile):
    """Include audible trait fields, never their book-evidence quotations."""
    description = " ".join(profile["description"].split())
    traits, seen = [], set()
    for trait in profile.get("traits", []):
        name = " ".join(trait["name"].split())
        value = " ".join(trait["value"].split())
        key = (name.casefold(), value.casefold())
        if name and value and key not in seen:
            seen.add(key)
            traits.append(f"{name}: {value}")
    prompt = description
    if traits:
        prompt += "\nVoice traits: " + "; ".join(traits)
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        prefix = prompt[:MAX_PROMPT_CHARACTERS - 1]
        boundary = max(prefix.rfind(" "), prefix.rfind("\n"))
        if boundary > 0:
            prefix = prefix[:boundary]
        prompt = prefix.rstrip() + "…"
    return prompt
