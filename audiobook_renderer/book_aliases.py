"""Editable voice-sharing choices belong to a book, never global settings."""

from pathlib import Path

from .common import read_json
from .voice_bindings import validate_aliases


def read_book_aliases(path, series_id, book_id, allowed_ids):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return {}
    if path.is_symlink() or not path.is_file():
        raise ValueError("Book voice-aliases.json must be a regular, nonsymlink file")
    record = read_json(path)
    fields = {"format", "version", "series_id", "book_id", "aliases"}
    if (not isinstance(record, dict) or set(record) != fields
            or record.get("format") != "audiobook-book-voice-aliases"
            or type(record.get("version")) is not int or record["version"] != 1
            or record.get("series_id") != series_id or record.get("book_id") != book_id):
        raise ValueError("Invalid book voice aliases or aliases belong to another book/series")
    aliases = validate_aliases(record["aliases"])
    if (set(aliases) | set(aliases.values())) - set(allowed_ids):
        raise ValueError("Book voice aliases must use exact voice IDs from this book or its ancestors")
    return aliases


def alias_record(bundle, config, aliases):
    return {"format": "audiobook-book-voice-aliases", "version": 1,
            "series_id": config["voices"]["series_id"], "book_id": bundle["source"]["book"]["id"],
            "aliases": validate_aliases(aliases)}


def edit_book_aliases(bundle, config, *, assignments=(), removals=()):
    """Validate a proposed local policy against the chain before publishing it."""
    from .book_context import voice_store
    from .common import write_json
    path = Path(bundle["directory"]) / "voice-aliases.json"
    store = voice_store(bundle, config)
    if not assignments and not removals:
        return {"path": str(path), "local": store.alias_loader(), "effective": store._alias_mapping()}
    # Serialize policy commands in this book; no ancestor file is changed.
    lock = path.with_name(".voice-aliases.lock")
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError("Book voice aliases are being edited; retry after the current edit") from None
    voice_lock = None
    try:
        voice_lock = store.lock()
        original = store.alias_loader()
        aliases = dict(original)
        for source in removals:
            aliases.pop(source, None)
        for source, target in assignments:
            aliases[source] = target
        record = alias_record(bundle, config, aliases)
        # Build a fresh local store with the candidate policy without writing it.
        proposed = voice_store(bundle, config, local_aliases=record["aliases"])
        effective = proposed._alias_mapping()
        # Reject a symlink, malformed identity, or concurrent manual file change
        # by reading the current policy again before replacing its regular file.
        if original != read_book_aliases(path, config["voices"]["series_id"],
                                       bundle["source"]["book"]["id"], proposed.book_voice_ids):
            raise ValueError("Book voice aliases changed during editing; retry")
        write_json(path, record)
        return {"path": str(path), "local": record["aliases"], "effective": effective}
    finally:
        if voice_lock is not None:
            voice_lock.rmdir()
        lock.rmdir()
