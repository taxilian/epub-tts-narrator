"""A voice's accent must be decided before the voice is designed.

Voices are designed with the accent their lines will be cued with
(``speech_traits.with_accent``), so a voice designed before its accent was
decided would be recorded with whatever accent its description happened to
name. Casting and automatic voice creation therefore stop until every voice
they would design has an accent decision: a cue a person chose
(``voice-accents.json``, an empty choice included), an accent leading vocal
traits a person set, or a saved book-wide assignment
(``voice-accent-assignments.json``, inherited ones and the model's "no
accent" included). A person makes those decisions with ``audiobook-render
accents BOOK``.

A voice with no lines in this book needs no accent, and a derived voice (a
character's later state) follows its main voice. The rule applies only when
accents can be assigned at all: ``casting.assign_accents`` is on and a text
model is configured (``provider.model``).
"""

from __future__ import annotations


def applies(config):
    return bool((config.get("casting") or {}).get("assign_accents", True)
                and (config.get("provider") or {}).get("model"))


def undecided(bundle, config, voice_ids=None):
    """The voices among ``voice_ids`` (default: the whole cast) whose accent nobody has decided yet."""
    if not (isinstance(bundle, dict) and {"directory", "cast", "source"} <= set(bundle)):
        return []
    from .book_context import resolve_book_config
    config = resolve_book_config(bundle, config)
    if not applies(config):
        return []
    from .accent_assignment import assigned_accents
    from .speech_traits import _book_terms, accent_overrides, speech_trait_overrides, trait_accent
    speaking = {row.get("voice_id") for chapter in bundle.get("chapters", []) for row in chapter.get("rows", [])}
    decided = set(accent_overrides(bundle, config, include_cleared=True)) | set(assigned_accents(bundle, config))
    decided |= {voice_id for voice_id, traits in speech_trait_overrides(bundle, config).items()
                if traits and trait_accent(traits, _book_terms(bundle))}
    profiles = {profile["id"]: profile for profile in bundle["cast"]["voices"]}
    result = []
    for voice_id in (voice_ids if voice_ids is not None else list(profiles)):
        profile = profiles.get(voice_id, {})
        base = profile["derived_from"].get("voice_id") if isinstance(profile.get("derived_from"), dict) else None
        target = base or voice_id
        if target in decided or voice_id in decided or not ({voice_id, target} & speaking):
            continue
        result.append(voice_id)
    return result


def message(bundle, voice_ids):
    names = {character["voice_id"]: character["name"] for character in bundle["cast"]["characters"]}
    shown = ", ".join(names.get(voice_id, voice_id) for voice_id in voice_ids[:6])
    more = f" and {len(voice_ids) - 6} more" if len(voice_ids) > 6 else ""
    return (f"Set this book's accents before designing voices: {shown}{more} "
            f"{'has' if len(voice_ids) == 1 else 'have'} no accent yet, and each voice is designed with its accent. "
            f"Run: audiobook-render accents {bundle['directory']} (just accents BOOK), review the plan and save it. "
            "Set casting.assign_accents = false to design voices without accents.")


def require(bundle, config, voice_ids=None, *, error=ValueError):
    """Raise ``error`` naming the voices that need an accent decision before they are designed."""
    missing = undecided(bundle, config, voice_ids)
    if missing:
        raise error(message(bundle, missing))


def require_before_render(bundle, config, *, error=ValueError):
    """Before a render assigns accents itself: every voice it would still design must be decided.

    Render's automatic assignment then only fills in cues for voices that
    already have recordings; it never decides the accent a new voice is
    designed with.
    """
    if not (isinstance(bundle, dict) and {"directory", "cast", "source"} <= set(bundle)):
        return
    from .book_context import resolve_book_config, voice_store
    config = resolve_book_config(bundle, config)
    if not applies(config):
        return
    store = voice_store(bundle, config)
    unrecorded = [voice["id"] for voice in bundle["cast"]["voices"] if store.load(voice["id"]) is None]
    require(bundle, config, unrecorded, error=error)
