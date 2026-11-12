from copy import deepcopy

from ..spoken_text import MAX_DELIVERY_CHARACTERS, effective_delivery
from .bundle import chunk_speakers


TRAITS_FILENAME = "voice-traits.json"


def _validate(value):
    # An explicit empty value disables traits and shadows ancestor choices.
    if value != "":
        from ..speech_traits import validate_speech_traits
        validate_speech_traits(value)


ACCENTS_FILENAME = "voice-accents.json"
MAX_ACCENT_CHARACTERS = 80


def validate_accent(value):
    """A short accent cue Drama 3 hears before every line in that voice, e.g. "heavy Brooklyn accent"."""
    from ..speech_traits import validate_speech_traits
    try:
        validate_speech_traits(value)
    except ValueError as exc:
        raise ValueError(str(exc).replace("speechTraits", "Accent cue")) from None
    if len(value) > MAX_ACCENT_CHARACTERS:
        raise ValueError(f"Accent cue must be at most {MAX_ACCENT_CHARACTERS} characters; "
                         "name the accent briefly (e.g. heavy Brooklyn accent)")
    return value


def _validate_accent_choice(value):
    # An explicit empty value clears an inherited cue.
    if value != "":
        validate_accent(value)


# Book-local per-voice choices: field -> (sidecar file, record format, validation).
_SIDECARS = {
    "speech_traits": (TRAITS_FILENAME, "audiobook-book-voice-traits", _validate),
    "accents": (ACCENTS_FILENAME, "audiobook-book-voice-accents", _validate_accent_choice),
}


def _setting(field="speech_traits"):
    from .book_voice_settings import VoiceSetting
    filename, record_format, validate = _SIDECARS[field]
    label = field.replace("_", " ")
    return VoiceSetting(
        filename=filename, record_format=record_format, field=field, validate=validate,
        invalid_message=f"Invalid book voice {label} or {label} belong to another book/series",
        unknown_voice_message=f"Book voice {label} must use voice IDs from this book",
        busy_message=f"Book voice {label} are being edited; retry after the current edit")


def speech_trait_overrides(bundle, config):
    """Resolve deliberate book-local choices nearest first, independent of audio aliases."""
    from .book_voice_settings import nearest_book_settings
    return nearest_book_settings(_setting("speech_traits"), bundle, config)


def accent_overrides(bundle, config, *, include_cleared=False):
    """Each voice's accent cue, nearest book first; an empty value clears an inherited cue."""
    from .book_voice_settings import nearest_book_settings
    values = nearest_book_settings(_setting("accents"), bundle, config)
    return values if include_cleared else {voice_id: value for voice_id, value in values.items() if value}


def save_speech_traits(bundle, config, voice_id, value):
    """Save an explicit performance choice without rewriting analyzed profiles or audio."""
    from .book_voice_settings import save_book_setting
    save_book_setting(_setting("speech_traits"), bundle, config, voice_id, value)


def save_accent(bundle, config, voice_id, value):
    """Save a voice's accent cue for this book (and books continuing it); empty text clears it here."""
    from .book_voice_settings import save_book_setting
    save_book_setting(_setting("accents"), bundle, config, voice_id, value)



def suggest_accent(description):
    """The last "<one to five words> accent" phrase in a voice description, as a starting cue, else ""."""
    import re
    matches = list(re.finditer(r"((?:[^\W\d_][\w'’-]*\s+){1,5})accent\b", description or "", re.IGNORECASE))
    if not matches:
        return ""
    words = matches[-1].group(1).split()
    while words and words[0].casefold() in {"a", "an", "the", "with", "and", "in", "of", "its", "his", "her"}:
        words.pop(0)
    phrase = " ".join([*words, "accent"])
    return phrase if words and len(phrase) <= MAX_ACCENT_CHARACTERS else ""


def performance_direction(chunk, voice_id, *, scene_direction=None):
    if scene_direction is None:
        scene_direction = effective_delivery(chunk["segment"])
    traits_by_voice = chunk.get("speech_traits")
    if traits_by_voice is None:
        return scene_direction
    if not isinstance(traits_by_voice, dict):
        raise ValueError("Speech traits bound to a passage must be a voice mapping")
    traits = traits_by_voice.get(voice_id, "")
    if traits:
        from ..speech_traits import validate_speech_traits

        traits = validate_speech_traits(traits)
    direction = "; ".join(value for value in (traits, scene_direction) if value)
    if len(direction) > MAX_DELIVERY_CHARACTERS:
        raise ValueError(
            f"Combined speech traits and delivery exceeds {MAX_DELIVERY_CHARACTERS} characters; "
            "shorten the stable traits or passage direction"
        )
    return direction


def bind_plan_speech_traits(plan, cast, *, overrides=None, profile_traits=True):
    """Bind each speaker's per-line prefix into every chunk (``speech_traits`` voice mapping).

    With ``profile_traits=False`` only ``overrides`` bind (renders bind accent
    cues this way); otherwise a voice without an override uses its profile's
    ``speechTraits``.
    """
    from ..speech_traits import speech_traits

    result = deepcopy(plan)
    profiles = {profile["id"]: profile for profile in cast["voices"]}

    def bind(chunk):
        chunk.pop("speech_traits", None)
        turns = chunk.get("paragraph_turns")
        if turns is not None:
            for turn in turns:
                bind(turn)
            return
        bound = {}
        for speaker in chunk_speakers(chunk):
            voice_id = speaker["voice_id"]
            profile = profiles.get(voice_id)
            if profile is None:
                raise ValueError(f"Voice ID is not in the bundle cast: {voice_id}")
            traits = (overrides[voice_id] if overrides is not None and voice_id in overrides
                      else speech_traits(profile) if profile_traits else "")
            if traits:
                from ..speech_traits import validate_speech_traits
                validate_speech_traits(traits)
            if traits:
                bound[voice_id] = traits
        if bound:
            chunk["speech_traits"] = bound

    for chapter in result["chapters"]:
        for chunk in chapter["chunks"]:
            bind(chunk)
    return result
