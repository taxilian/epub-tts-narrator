"""Book-local, per-voice casting choices stored beside a book, nearest book first.

Each kind of choice (vocal traits, reference clip count, ...) lives in its own
small identity-checked JSON sidecar in the book directory:

    {"format": ..., "version": 1, "series_id": ..., "book_id": ..., FIELD: {voice_id: value}}

A continued book inherits an earlier book's choice unless it saves its own.
Saving never rewrites analysis files, cast metadata, or voice recordings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class VoiceSetting:
    """One sidecar kind: its file, record format, value field, and validation."""

    filename: str
    record_format: str
    field: str
    validate: Callable[[Any], None]
    invalid_message: str
    unknown_voice_message: str
    busy_message: str

    @property
    def lock_name(self):
        return "." + self.filename.removesuffix(".json") + ".lock"


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


def _present(path):
    return path.exists() or path.is_symlink()


def read_book_setting(setting, bundle, series_id):
    """This book's own saved values, validated against its identity and cast."""
    from .common import read_json, regular_file

    path = Path(bundle["directory"]) / setting.filename
    if not _present(path):
        return {}
    record = read_json(regular_file(path))
    fields = {"format", "version", "series_id", "book_id", setting.field}
    if (not isinstance(record, dict) or set(record) != fields
            or record.get("format") != setting.record_format
            or type(record.get("version")) is not int or record["version"] != 1
            or record.get("series_id") != series_id
            or record.get("book_id") != bundle["source"]["book"]["id"]
            or not isinstance(record.get(setting.field), dict)):
        raise ValueError(setting.invalid_message)
    values = record[setting.field]
    if set(values) - {voice["id"] for voice in bundle["cast"]["voices"]}:
        raise ValueError(setting.unknown_voice_message)
    for value in values.values():
        setting.validate(value)
    return dict(values)


def nearest_book_settings(setting, bundle, config):
    """Resolve saved values nearest book first; earlier books fill the rest."""
    from .book_context import resolve_book_config

    config = resolve_book_config(bundle, config)
    series_id = config["voices"]["series_id"]
    result = read_book_setting(setting, bundle, series_id)
    for directory in config.get("_voice_parents", []):
        if not _present(Path(directory) / setting.filename):
            continue
        for voice_id, value in read_book_setting(setting, _parent_context(directory), series_id).items():
            result.setdefault(voice_id, value)
    allowed = {voice["id"] for voice in bundle["cast"]["voices"]}
    return {voice_id: value for voice_id, value in result.items() if voice_id in allowed}


def any_book_settings(setting, bundle, config):
    """Whether this book or an earlier book in its chain has this sidecar at all."""
    directories = [Path(bundle["directory"])]
    directories += [Path(path) for path in (config.get("_voice_parents") or [])]
    return any(_present(directory / setting.filename) for directory in directories)


def save_book_setting(setting, bundle, config, voice_id, value):
    """Save (or, with ``value=None``, remove) this book's own value for one voice.

    Removing a value makes the voice inherit an earlier book's choice or the
    global default again.
    """
    save_book_settings(setting, bundle, config, {voice_id: value})


def save_book_settings(setting, bundle, config, changes, *, replace=False):
    """Save several voices' values for this book in one write (``None`` removes one).

    With ``replace`` this book's own values become exactly ``changes``.
    """
    from .common import write_json

    allowed = {voice["id"] for voice in bundle["cast"]["voices"]}
    if set(changes) - allowed:
        raise ValueError("Voice ID is not in the bundle cast")
    for value in changes.values():
        if value is not None:
            setting.validate(value)
    directory = Path(bundle["directory"])
    lock = directory / setting.lock_name
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(setting.busy_message) from None
    try:
        series_id = config["voices"]["series_id"]
        values = {} if replace else read_book_setting(setting, bundle, series_id)
        for voice_id, value in changes.items():
            if value is None:
                values.pop(voice_id, None)
            else:
                values[voice_id] = value
        write_json(directory / setting.filename, {
            "format": setting.record_format, "version": 1,
            "series_id": series_id, "book_id": bundle["source"]["book"]["id"],
            setting.field: values,
        })
    finally:
        lock.rmdir()
