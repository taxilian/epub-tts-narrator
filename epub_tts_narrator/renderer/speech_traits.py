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



# Words that join an accent phrase to the rest of a description ("voice with
# a thick Welsh accent"); the cue starts after the last of them.
_JOINING = frozenset("""a an the with and in of its his her their has having into to from for or but like by on at
as any some voice speaks speaking spoken""".split())
_NEGATING = frozenset({"no", "not", "without", "never", "lacking", "free", "none"})
# Real-world words that do not name a place or people on their own ("neutral accent").
_NOT_PLACES = frozenset({"general", "standard", "neutral", "east", "west", "north", "south", "central"})


def _sentence_start(text, position):
    before = text[:position].rstrip()
    return not before or before[-1] in ".!?:;"


def suggest_accent(description, terms=frozenset()):
    """The cue a voice description names, e.g. "heavy Brooklyn accent", else "".

    The last "<words> accent" phrase, from after its last joining word. It must
    name a real place or people: a capitalized word not starting a sentence, a
    known real-world accent word, or a lowercase -ian/-ese word ("italian").
    A negated phrase ("no accent"), a bare manner ("heavy accent"), and one
    naming a book's own term (``terms``, e.g. "Nyissan accent", which a speech
    model cannot know) give no cue.
    """
    import re
    from ..narrator_read import _REAL_WORLD
    text = description or ""
    found = list(re.finditer(r"\baccent\b", text, re.IGNORECASE))
    if not found:
        return ""
    end = found[-1].start()
    words = []
    for match in reversed(list(re.finditer(r"[^\W\d_][\w'’-]*", text[:end]))):
        if text[match.end():end].strip():
            break  # Punctuation ends the phrase.
        words.insert(0, match)
        end = match.start()
    if any(word.group().casefold() in _NEGATING for word in words):
        return ""
    joins = [index for index, word in enumerate(words) if word.group().casefold() in _JOINING]
    words = words[joins[-1] + 1 if joins else 0:][-5:]
    places = _REAL_WORLD - _NOT_PLACES
    named = False
    for word in words:
        for part in re.split(r"[-'’]", word.group()):
            if part.casefold() in terms:
                return ""
            named = named or (part[:1].isupper() and not _sentence_start(text, word.start())
                              or part.casefold() in places or part.islower() and part.endswith(("ian", "ese")))
    phrase = " ".join([*(word.group() for word in words), "accent"])
    return phrase if named and len(phrase) <= MAX_ACCENT_CHARACTERS else ""


_TERMS = {}


def _book_terms(bundle):
    """The book's own proper nouns (``narrator_read.book_terms``), once per bundle."""
    key = bundle.get("bundle_hash") or id(bundle)
    if key not in _TERMS:
        from ..narrator_read import book_terms
        characters = [{"name": character.get("name", ""), "aliases": character.get("aliases", [])}
                      for character in bundle["cast"]["characters"]]
        _TERMS.clear()  # One book at a time is enough; a queue renders books in turn.
        _TERMS[key] = frozenset(book_terms(bundle.get("source") or {"blocks": []}, {"characters": characters}))
    return _TERMS[key]


def designed_description(store, profile):
    """The description that designed a voice's saved recording, else the one that will."""
    saved = store.load(profile["id"]) if store is not None else None
    requested = ((saved or {}).get("creation") or {}).get("requested_description")
    return requested if isinstance(requested, str) and requested else profile.get("description", "")


def accent_cues(bundle, config, *, store=None):
    """Every voice's accent cue for a render.

    A person's choice (``voice-accents.json``, nearest book first; an empty
    choice means no cue) wins. Any other voice uses the accent its description
    names (``suggest_accent``): the description that designed its saved
    recording, or, before it is made, the one that will.
    """
    chosen = accent_overrides(bundle, config, include_cleared=True)
    if store is None:
        from .book_context import voice_store
        store = voice_store(bundle, config)
    cues = {}
    for profile in bundle["cast"]["voices"]:
        voice_id = profile["id"]
        if voice_id in chosen:
            cue = chosen[voice_id]
        else:
            cue = suggest_accent(designed_description(store, profile))
            if cue:
                cue = suggest_accent(designed_description(store, profile), _book_terms(bundle))
        if cue:
            cues[voice_id] = cue
    return cues


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
