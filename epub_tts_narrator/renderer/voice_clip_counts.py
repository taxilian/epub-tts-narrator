"""Per-voice reference clip counts chosen in casting (book-local ``voice-clips.json``).

``casting.reference_clips`` (1-3) is the default number of clips a new voice
records and joins into its reference. A person can choose a different count
for one voice; the choice belongs to the voice's recording, so it is looked up
by the shared (canonical) voice first, then by the role's own voice ID, nearest
book first. It applies the next time that voice is generated and never changes
an existing recording. Only designers that keep one voice across several
scripts (Gemini) record more than one clip.
"""

from .book_voice_settings import VoiceSetting, any_book_settings, nearest_book_settings, save_book_setting


CLIPS_FILENAME = "voice-clips.json"
MIN_CLIPS, MAX_CLIPS = 1, 3


def _validate(value):
    if type(value) is not int or not MIN_CLIPS <= value <= MAX_CLIPS:
        raise ValueError(f"Reference clip count must be an integer from {MIN_CLIPS} through {MAX_CLIPS}")


SETTING = VoiceSetting(
    filename=CLIPS_FILENAME, record_format="audiobook-book-voice-clips", field="reference_clips",
    validate=_validate,
    invalid_message="Invalid book voice clip counts or counts belong to another book/series",
    unknown_voice_message="Book voice clip counts must use voice IDs from this book",
    busy_message="Book voice clip counts are being edited; retry after the current edit")


def default_clips(config):
    return (config.get("casting") or {}).get("reference_clips", 3)


def clip_overrides(bundle, config):
    """Saved per-voice counts, nearest book first; empty when no book saved any."""
    if not isinstance(bundle, dict) or not {"directory", "manifest", "cast", "source"} <= set(bundle):
        return {}
    # Checked before resolving: an unresolved configuration knows only this
    # book, and a book chain without any sidecar needs no configuration at all.
    if not any_book_settings(SETTING, bundle, config):
        return {}
    from .book_context import resolve_book_config
    return nearest_book_settings(SETTING, bundle, resolve_book_config(bundle, config))


def clip_choice(bundle, config, voice_id, *, canonical_id=lambda value: value, overrides=None):
    """``(count, source)`` for one voice; source is ``"voice"`` or ``"default"``."""
    overrides = clip_overrides(bundle, config) if overrides is None else overrides
    for key in (canonical_id(voice_id), voice_id):
        if key in overrides:
            return overrides[key], "voice"
    return default_clips(config), "default"


def save_clip_count(bundle, config, voice_id, count):
    """Save this book's count for one voice; ``None`` returns it to the default."""
    from .book_context import resolve_book_config
    save_book_setting(SETTING, bundle, resolve_book_config(bundle, config), voice_id, count)


def extra_clip_plan(bundle, config, *, canonical_id=lambda value: value):
    """Extra scripts to prepare per voice: the global int, or a per-voice mapping.

    Without saved per-voice counts this is exactly ``extra_reference_clips``,
    so books that never chose a count prepare the same scripts as before.
    """
    from .config import designer_records_several_clips, extra_reference_clips
    overrides = clip_overrides(bundle, config)
    if not overrides or not designer_records_several_clips(config):
        return extra_reference_clips(config)
    return {voice["id"]: clip_choice(bundle, config, voice["id"], canonical_id=canonical_id,
                                     overrides=overrides)[0] - 1
            for voice in bundle["cast"]["voices"]}
