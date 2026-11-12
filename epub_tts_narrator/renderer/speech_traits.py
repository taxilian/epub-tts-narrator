from copy import deepcopy
from pathlib import Path

from ..spoken_text import MAX_DELIVERY_CHARACTERS, effective_delivery
from .bundle import chunk_speakers


TRAITS_FILENAME = "voice-traits.json"


def _parent_context(directory):
    """Read only checked identity/cast metadata, never parse chapter scripts on redraw."""
    from ..voice_profile_guard import _require_ready
    from .common import file_hash, read_json, regular_file

    directory = Path(directory)
    # resolve_book_config already validated the continuation chain. Reread its
    # live pending markers without reloading the entire analysis and scripts.
    _require_ready(directory)
    manifest = read_json(regular_file(directory / "manifest.json"))
    if (not isinstance(manifest, dict) or manifest.get("format") != "epub-audiobook-bundle" or manifest.get("version") != 2
            or manifest.get("files", {}).get("cast") != "cast.json"
            or file_hash(regular_file(directory / "cast.json")) != manifest.get("sha256", {}).get("cast.json")):
        raise ValueError("Parent cast identity changed while reading voice traits")
    return {"directory": str(directory), "source": {"book": manifest["book"]},
            "cast": read_json(directory / "cast.json")}


ACCENTS_FILENAME = "voice-accents.json"
MAX_ACCENT_CHARACTERS = 80
# Book-local per-voice choices: field -> (sidecar file, record format, edit lock directory).
_SIDECARS = {
    "speech_traits": (TRAITS_FILENAME, "audiobook-book-voice-traits", ".voice-traits.lock"),
    "accents": (ACCENTS_FILENAME, "audiobook-book-voice-accents", ".voice-accents.lock"),
}


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


def _validator(field):
    if field == "accents":
        return validate_accent
    from ..speech_traits import validate_speech_traits
    return validate_speech_traits


def _read_overrides(bundle, series_id, field="speech_traits"):
    from .common import read_json, regular_file

    filename, record_format, _lock = _SIDECARS[field]
    path = Path(bundle["directory"]) / filename
    if not path.exists() and not path.is_symlink():
        return {}
    record = read_json(regular_file(path))
    fields = {"format", "version", "series_id", "book_id", field}
    label = field.replace("_", " ")
    if (not isinstance(record, dict) or set(record) != fields
            or record.get("format") != record_format
            or type(record.get("version")) is not int or record["version"] != 1
            or record.get("series_id") != series_id
            or record.get("book_id") != bundle["source"]["book"]["id"]
            or not isinstance(record.get(field), dict)):
        raise ValueError(f"Invalid book voice {label} or {label} belong to another book/series")
    values = record[field]
    if set(values) - {voice["id"] for voice in bundle["cast"]["voices"]}:
        raise ValueError(f"Book voice {label} must use voice IDs from this book")
    validate = _validator(field)
    for value in values.values():
        # An explicit empty value disables the choice and shadows ancestor choices.
        if value != "":
            validate(value)
    return dict(values)


def _overrides(bundle, config, field):
    from .book_context import resolve_book_config

    config = resolve_book_config(bundle, config)
    series_id = config["voices"]["series_id"]
    result = _read_overrides(bundle, series_id, field)
    for directory in config.get("_voice_parents", []):
        path = Path(directory) / _SIDECARS[field][0]
        if not path.exists() and not path.is_symlink():
            continue
        for voice_id, value in _read_overrides(_parent_context(directory), series_id, field).items():
            result.setdefault(voice_id, value)
    allowed = {voice["id"] for voice in bundle["cast"]["voices"]}
    return {voice_id: value for voice_id, value in result.items() if voice_id in allowed}


def speech_trait_overrides(bundle, config):
    """Resolve deliberate book-local choices nearest first, independent of audio aliases."""
    return _overrides(bundle, config, "speech_traits")


def accent_overrides(bundle, config, *, include_cleared=False):
    """Each voice's accent cue, nearest book first; an empty value clears an inherited cue."""
    values = _overrides(bundle, config, "accents")
    return values if include_cleared else {voice_id: value for voice_id, value in values.items() if value}


def _save(bundle, config, voice_id, value, field):
    from .common import write_json

    if voice_id not in {voice["id"] for voice in bundle["cast"]["voices"]}:
        raise ValueError("Voice ID is not in the bundle cast")
    if value != "":
        _validator(field)(value)
    filename, record_format, lock_name = _SIDECARS[field]
    label = field.replace("_", " ")
    directory = Path(bundle["directory"])
    lock = directory / lock_name
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f"Book voice {label} are being edited; retry after the current edit") from None
    try:
        series_id = config["voices"]["series_id"]
        overrides = _read_overrides(bundle, series_id, field)
        overrides[voice_id] = value
        write_json(directory / filename, {
            "format": record_format, "version": 1,
            "series_id": series_id, "book_id": bundle["source"]["book"]["id"],
            field: overrides,
        })
    finally:
        lock.rmdir()


def save_speech_traits(bundle, config, voice_id, value):
    """Save an explicit performance choice without rewriting analyzed profiles or audio."""
    _save(bundle, config, voice_id, value, "speech_traits")


def save_accent(bundle, config, voice_id, value):
    """Save a voice's accent cue for this book (and books continuing it); empty text clears it here."""
    _save(bundle, config, voice_id, value, "accents")


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
