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


def _read_overrides(bundle, series_id):
    from ..speech_traits import validate_speech_traits
    from .common import read_json, regular_file

    path = Path(bundle["directory"]) / TRAITS_FILENAME
    if not path.exists() and not path.is_symlink():
        return {}
    record = read_json(regular_file(path))
    fields = {"format", "version", "series_id", "book_id", "speech_traits"}
    if (not isinstance(record, dict) or set(record) != fields
            or record.get("format") != "audiobook-book-voice-traits"
            or type(record.get("version")) is not int or record["version"] != 1
            or record.get("series_id") != series_id
            or record.get("book_id") != bundle["source"]["book"]["id"]
            or not isinstance(record.get("speech_traits"), dict)):
        raise ValueError("Invalid book voice traits or traits belong to another book/series")
    values = record["speech_traits"]
    if set(values) - {voice["id"] for voice in bundle["cast"]["voices"]}:
        raise ValueError("Book voice traits must use voice IDs from this book")
    for value in values.values():
        # An explicit empty value disables traits and shadows ancestor choices.
        if value != "":
            validate_speech_traits(value)
    return dict(values)


def speech_trait_overrides(bundle, config):
    """Resolve deliberate book-local choices nearest first, independent of audio aliases."""
    from .book_context import resolve_book_config

    config = resolve_book_config(bundle, config)
    series_id = config["voices"]["series_id"]
    result = _read_overrides(bundle, series_id)
    for directory in config.get("_voice_parents", []):
        path = Path(directory) / TRAITS_FILENAME
        if not path.exists() and not path.is_symlink():
            continue
        for voice_id, value in _read_overrides(_parent_context(directory), series_id).items():
            result.setdefault(voice_id, value)
    allowed = {voice["id"] for voice in bundle["cast"]["voices"]}
    return {voice_id: value for voice_id, value in result.items() if voice_id in allowed}


def save_speech_traits(bundle, config, voice_id, value):
    """Save an explicit performance choice without rewriting analyzed profiles or audio."""
    from ..speech_traits import validate_speech_traits
    from .common import write_json

    if voice_id not in {voice["id"] for voice in bundle["cast"]["voices"]}:
        raise ValueError("Voice ID is not in the bundle cast")
    if value != "":
        validate_speech_traits(value)
    directory = Path(bundle["directory"])
    lock = directory / ".voice-traits.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Book voice traits are being edited; retry after the current edit") from None
    try:
        series_id = config["voices"]["series_id"]
        overrides = _read_overrides(bundle, series_id)
        overrides[voice_id] = value
        write_json(directory / TRAITS_FILENAME, {
            "format": "audiobook-book-voice-traits", "version": 1,
            "series_id": series_id, "book_id": bundle["source"]["book"]["id"],
            "speech_traits": overrides,
        })
    finally:
        lock.rmdir()


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


def bind_plan_speech_traits(plan, cast, *, overrides=None):
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
                      else speech_traits(profile))
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
