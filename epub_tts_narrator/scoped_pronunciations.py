"""Entity/alias choices, separate from untouched legacy global registries."""

from copy import deepcopy
from pathlib import Path

from .name_script import alias_words
from .name_script_state import load_name_script
from .pipeline import fingerprint, write_json
from .pronunciations import (_directory, _document_source, _load_local, _regular_json, _source, validate_pronunciation,
                             validate_entry, validate_proposal, validate_word, word_id, WORD_PATTERN)
from .pronunciation_groups import case_key, family_words, find_pronunciation, group_pronunciations, named_families
from .renderer.run_locks import acquire_directory_lock


FILENAME = "scoped-pronunciations.json"


def legacy_named_families(directory):
    from .analysis_directory import continuation_directories
    ancestors = continuation_directories(directory) if (directory / "manifest.json").exists() else []
    entries = []
    for owner in [directory, *ancestors]:
        record = _load_local(owner)
        if record is not None:
            entries.extend(record["entries"])
    return named_families(entries)


def scope_for(document, name_id=None):
    if name_id is None:
        return {"class": "special"}
    snapshot = load_name_script(document["directory"], document)
    key = str(name_id)
    if key.isdecimal():
        key = "N" + key
    if key not in snapshot["names"]:
        raise ValueError("Unknown --name-id; list prepare-name-script BOOK --action list")
    return identity_scope(document, snapshot["names"][key]["character_id"])


def identity_scope(document, character_id):
    provenance = document.get("manifest", {}).get("provenance", document.get("provenance", {}))
    series = provenance.get("series", {}).get("series_id")
    return {"class": "name", "character_id": character_id,
            "namespace": ["series", series] if series else ["book", document["source"]["book"]["id"]]}


def validate_scope(scope):
    if scope == {"class": "special"}:
        return scope
    if (not isinstance(scope, dict) or set(scope) != {"class", "character_id", "namespace"}
            or scope["class"] != "name" or not isinstance(scope["character_id"], str)
            or not isinstance(scope["namespace"], list) or len(scope["namespace"]) != 2
            or not isinstance(scope["namespace"][0], str) or scope["namespace"][0] not in {"series", "book"}
            or not isinstance(scope["namespace"][1], str) or not scope["namespace"][1]):
        raise ValueError("Invalid pronunciation scope")
    return scope


def local(directory, *, source=None):
    path = Path(directory) / FILENAME
    if not path.exists() and not path.is_symlink():
        record = {"format": "epub-tts-scoped-pronunciations", "version": 1, "entries": [], "migrated": []}
        if source is not None:
            record.update(book_id=source["book"]["id"], source_fingerprint=fingerprint(source))
        return record
    value = _regular_json(path)
    if (not isinstance(value, dict) or set(value) != {"format", "version", "book_id", "source_fingerprint", "entries", "migrated"}
            or value["format"] != "epub-tts-scoped-pronunciations" or type(value["version"]) is not int or value["version"] != 1
            or not isinstance(value["entries"], list) or not isinstance(value["migrated"], list)):
        raise ValueError("Invalid scoped pronunciation registry or missing book/source ownership")
    source = _source(Path(directory)) if source is None else source
    if value["book_id"] != source["book"]["id"] or value["source_fingerprint"] != fingerprint(source):
        raise ValueError("Scoped pronunciation registry belongs to a different book or source")
    for migration in value["migrated"]:
        if not isinstance(migration, dict) or set(migration) != {"word", "family"}:
            raise ValueError("Scoped pronunciation migration requires a legacy spelling family")
        word = validate_word(migration["word"])
        family = migration["family"]
        if (not isinstance(family, list) or not all(isinstance(part, str) for part in family)
                or tuple(family) not in {case_key(word), case_key(word, {word.lower()})}):
            raise ValueError("Invalid legacy pronunciation migration family")
    seen = set()
    for entry in value["entries"]:
        required = {"id", "scope", "word", "status", "kind", "reason", "evidence"}
        if not isinstance(entry, dict) or not required <= set(entry) or set(entry) - required - {"proposal", "pronunciation", "review_requested"}:
            raise ValueError("Invalid scoped pronunciation entry")
        validate_scope(entry["scope"])
        validate_word(entry["word"])
        if entry["id"] != fingerprint([entry["scope"], entry["word"]]) or entry["id"] in seen:
            raise ValueError("Invalid or duplicate scoped pronunciation identity")
        seen.add(entry["id"])
        legacy_fields = {key: item for key, item in entry.items() if key != "scope"}
        legacy_fields["id"] = word_id(entry["word"])
        validate_entry(legacy_fields, source)
    return value


def load(directory, scope):
    validate_scope(scope)
    directory = _directory(directory)
    from .analysis_directory import continuation_directories
    ancestors = continuation_directories(directory) if (directory / "manifest.json").exists() else []
    entries = [dict(deepcopy(entry), origin_directory=str(owner), inherited=owner != directory)
               for owner in [directory, *ancestors] for entry in local(owner)["entries"] if entry["scope"] == scope]
    if not entries:
        return []
    words = {token.group() for block in _source(directory)["blocks"] for token in WORD_PATTERN.finditer(block["text"])}
    return group_pronunciations(entries, words, contextual=scope["class"] == "name")


def set_choice(directory, document, word, pronunciation=None, status="confirmed", *, scope, proposal=None,
               migrate=False):
    validate_scope(scope)
    validate_word(word)
    directory = _directory(directory)
    source = _document_source(directory, document)
    if status not in {"pending", "confirmed", "ignored"}:
        raise ValueError("Invalid pronunciation status")
    if status == "confirmed":
        pronunciation = validate_pronunciation(pronunciation)
    elif pronunciation is not None:
        raise ValueError("Only confirmed choices contain pronunciation")
    blocks = source["blocks"]
    if scope["class"] == "name":
        snapshot = load_name_script(directory, document)
        name_id = next((key for key, value in snapshot["names"].items()
                        if value["character_id"] == scope["character_id"]), None)
        if name_id is None or scope_for(dict(document, directory=directory), name_id) != scope:
            raise ValueError("Name scope does not belong to this book identity")
        if word.lower() not in alias_words(snapshot["names"][name_id]):
            raise ValueError("Name pronunciation must select an alias word for this entity")
        if status == "confirmed" and any(name_id in item["names"] and item["word"].lower() == word.lower()
                                         for item in snapshot["unresolved"]):
            raise ValueError("Resolve affected name-script mentions before confirming this name")
    citation = next(({"block_id": block["id"], "quote": word} for block in blocks
                     if any(token.group() == word for token in WORD_PATTERN.finditer(block["text"]))), None)
    if citation is None:
        raise ValueError("Pronunciation word must occur exactly in this book")
    lock = acquire_directory_lock(directory / ".scoped-pronunciations.lock")
    try:
        record = local(directory, source=source)
        if proposal is not None:
            current = find_pronunciation(load(directory, scope), word)
            if current and (current["status"] != "pending" or "proposal" in current):
                return current
        words = {token.group() for block in source["blocks"] for token in WORD_PATTERN.finditer(block["text"])}
        family = family_words(word, load(directory, scope), words, contextual=scope["class"] == "name")
        if scope["class"] == "special" and status == "confirmed":
            from .name_script import FILENAME as NAME_SCRIPT
            from .name_script_state import ordinary_span
            path = directory / NAME_SCRIPT
            if path.exists() or path.is_symlink():
                snapshot = load_name_script(directory, document)
                if any(item["word"] in family for item in snapshot["unresolved"]):
                    raise ValueError("Resolve affected name-script mentions before confirming this Special rule")
                if not any(token.group() in family and ordinary_span(snapshot, block["id"], token.start(), token.end())
                           for block in blocks for token in WORD_PATTERN.finditer(block["text"])):
                    raise ValueError("Special pronunciation requires a resolved ordinary occurrence")
        record["entries"] = [entry for entry in record["entries"]
                             if not (entry["scope"] == scope and entry["word"] in family)]
        entry = {"id": fingerprint([scope, word]), "scope": deepcopy(scope), "word": word, "status": status,
                 "kind": "name" if scope["class"] == "name" else "invented",
                 "reason": "Explicit scoped pronunciation review.", "evidence": [citation], "review_requested": True}
        if pronunciation is not None:
            entry["pronunciation"] = pronunciation
        if proposal is not None:
            if status != "pending":
                raise ValueError("Only pending choices can contain proposals")
            entry["proposal"] = validate_proposal(proposal)
        record["entries"].append(entry)
        if migrate or scope["class"] == "special":
            migrations = {tuple(item["family"]): item for item in record["migrated"]}
            key = case_key(word, legacy_named_families(directory))
            migrations[key] = {"word": word, "family": list(key)}
            record["migrated"] = [migrations[key] for key in sorted(migrations)]
        write_json(directory / FILENAME, record, overwrite=True)
    finally:
        lock.rmdir()
    return find_pronunciation(load(directory, scope), word)


def reset(directory, word, scope):
    directory = _directory(directory)
    validate_scope(scope)
    validate_word(word)
    lock = acquire_directory_lock(directory / ".scoped-pronunciations.lock")
    try:
        record = local(directory, source=_source(directory))
        words = {entry["word"] for entry in record["entries"] if entry["scope"] == scope}
        family = family_words(word, load(directory, scope), words, contextual=scope["class"] == "name")
        record["entries"] = [entry for entry in record["entries"]
                             if not (entry["scope"] == scope and entry["word"] in family)]
        write_json(directory / FILENAME, record, overwrite=True)
    finally:
        lock.rmdir()
    return load(directory, scope)
