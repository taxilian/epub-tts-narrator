"""Share reference identities by explicit character IDs, never by similar names."""

from __future__ import annotations


def validate_aliases(aliases):
    if not isinstance(aliases, dict) or any(
        not isinstance(key, str) or not key.strip()
        or not isinstance(value, str) or not value.strip() or key == value
        for key, value in aliases.items()
    ):
        raise ValueError("Voice aliases must map distinct, nonempty voice IDs")
    for voice_id in aliases:
        seen = set()
        current = voice_id
        while current in aliases:
            if current in seen:
                raise ValueError("Voice aliases must not contain cycles")
            seen.add(current)
            current = aliases[current]
    return dict(aliases)


def canonical_aliases(cast, explicit=None):
    characters = {item["id"]: item for item in cast["characters"]}
    owners = {}
    aliases = {}
    for character in characters.values():
        owners.setdefault(character["voice_id"], set()).add(character["id"])
    for role in cast["roles"]:
        character_id = role.get("character_id")
        if character_id is None:
            continue
        character = characters.get(character_id)
        if character is None:
            raise ValueError("Voice role refers to an unknown character")
        voice_id, canonical = role["voice_id"], character["voice_id"]
        owners.setdefault(voice_id, set()).add(character_id)
        if voice_id != canonical:
            if voice_id in aliases and aliases[voice_id] != canonical:
                raise ValueError("Voice role has conflicting character identities")
            aliases[voice_id] = canonical
    if any(len(values) > 1 for values in owners.values()):
        raise ValueError("A voice ID belongs to multiple distinct characters")
    aliases.update(validate_aliases(explicit or {}))
    return validate_aliases(aliases)


def cast_voice_owners(cast):
    """Record exact identities for primary voices and alternate role voices."""
    owners = {}
    for character in cast["characters"]:
        owners.setdefault(character["voice_id"], set()).add(character["id"])
    for role in cast["roles"]:
        if role.get("character_id") is not None:
            owners.setdefault(role["voice_id"], set()).add(role["character_id"])
    if any(len(values) > 1 for values in owners.values()):
        raise ValueError("A voice ID belongs to multiple distinct characters")
    return {voice_id: next(iter(characters)) for voice_id, characters in owners.items()}


def validate_alias_ownership(aliases, explicit_aliases, owners):
    """Only explicitly configured edges may connect distinct character IDs."""
    links = {}
    for source, target in aliases.items():
        if source in explicit_aliases:
            continue
        links.setdefault(source, set()).add(target)
        links.setdefault(target, set()).add(source)
    visited = set()
    for source in links:
        if source in visited:
            continue
        pending, characters = [source], set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in owners:
                characters.add(owners[current])
            pending.extend(links.get(current, set()) - visited)
        if len(characters) > 1:
            raise ValueError("Persisted voice aliases would merge distinct character identities")


def require_designer(engine, designer=None):
    """Do not silently turn a requested description into random voice sampling."""
    source = designer if designer is not None else engine
    if not engine.capabilities.get("reference_voice"):
        raise ValueError("This engine needs a compatible preset-voice binding")
    if not source.capabilities.get("voice_design"):
        raise ValueError(
            "This engine cannot apply character voice descriptions. Configure "
            "[voice_design] with a voice designer or import a reference recording."
        )
    return source


def design_voice(designer, prompt, text, path, *, seed):
    metadata = designer.create_voice(prompt, text, path, seed=seed)
    if not isinstance(metadata, dict) or metadata.get("description_applied") is not True:
        raise ValueError("Voice designer did not confirm applying the character description")
    return metadata


def designer_identity(designer):
    """Fingerprint casting separately when an engine also renders saved voices."""
    from copy import deepcopy
    identity = getattr(designer, "voice_design_identity", None)
    return deepcopy(designer.identity if identity is None else identity)
