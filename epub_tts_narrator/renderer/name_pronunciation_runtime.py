from pathlib import Path

from ..name_script import FILENAME, alias_words, identity_table
from ..name_script_state import load_name_script
from ..pronunciations import WORD_PATTERN, load_pronunciations
from ..pronunciation_groups import case_key
from ..scoped_pronunciations import identity_scope, legacy_named_families, load, local


def name_pronunciation_snapshot(bundle, legacy):
    directory = Path(bundle["directory"])
    active = (directory / FILENAME).exists() or (directory / FILENAME).is_symlink()
    special = [entry for entry in load(directory, {"class": "special"}) if entry["status"] == "confirmed"]
    from ..analysis_directory import continuation_directories
    owners = [directory, *continuation_directories(directory)] if (directory / "manifest.json").exists() else [directory]
    records = [local(owner) for owner in owners]
    scoped = [entry for record in records for entry in record["entries"] if entry["scope"]["class"] == "name"]
    migrations = {}
    for rank, record in enumerate(records):
        for migration in record["migrated"]:
            migrations.setdefault(tuple(migration["family"]), rank)
    if not active and not scoped and not special and not migrations:
        return None
    ranks = {str(owner): rank for rank, owner in enumerate(owners)}
    named = legacy_named_families(directory)
    newer_defaults = any(entry["status"] == "ignored" and case_key(entry["word"], named) in migrations
                         and migrations[case_key(entry["word"], named)] > ranks[entry["origin_directory"]]
                         for entry in load_pronunciations(directory)) if migrations else False
    migrated = {word for word, entry in legacy.items() if case_key(word, named) in migrations
                and migrations[case_key(word, named)] <= ranks.get(entry.get("origin_directory", str(directory)), 0)}
    table = identity_table(bundle)
    aliases = {word for name in table.values() for word in alias_words(name)}
    alias_families = {case_key(token.group(), named) for name in table.values() for alias in name["aliases"]
                      for token in WORD_PATTERN.finditer(alias)}
    if newer_defaults or any(word not in migrated and (case_key(word, named) in migrations or
           ((active or scoped) and case_key(word, named) in alias_families)) for word in legacy):
        raise ValueError("Legacy global pronunciation collides with a name; use pronunciations --action migrate "
                         "--word WORD --class name --name-id N1, or --class special to retain global intent")
    identities = {entry["scope"]["character_id"] for entry in scoped
                  if entry["scope"]["character_id"] in table
                  and entry["scope"] == identity_scope(bundle, entry["scope"]["character_id"])}
    effective = {key: [entry for entry in load(directory, identity_scope(bundle, key))
                       if entry["word"].lower() in alias_words(table[key])] for key in identities}
    applicable = any(entry["status"] == "confirmed" for rows in effective.values() for entry in rows)
    needs_script = (applicable or any(entry["word"].lower() in aliases for entry in special)
                    or (active and any(word.lower() in aliases for word in legacy if word not in migrated)))
    snapshot = load_name_script(directory, bundle) if needs_script else None
    names = {}
    if snapshot is not None:
        for key, name in snapshot["names"].items():
            entries = effective.get(name["character_id"], [])
            names[key] = {word: entry for entry in entries if entry["status"] == "confirmed"
                          for word in entry.get("variants", [entry["word"]])}
        for item in snapshot["unresolved"]:
            if (item["word"] in legacy and item["word"] not in migrated) or any(
                    item["word"] in names[key] for key in item["names"]) or any(
                    item["word"] in entry.get("variants", [entry["word"]]) and entry["status"] == "confirmed"
                    for entry in special):
                raise ValueError("Unresolved saved name mention affects a pronunciation rule; automatic name preparation is retired")
    return {"annotations": snapshot, "names": names, "migrated": sorted(migrated),
            "special": {word: entry for entry in special if entry["status"] == "confirmed"
                        for word in entry.get("variants", [entry["word"]])}}


def row_matches(row, legacy, snapshot):
    annotations = snapshot["annotations"]
    spans = annotations["mentions"].get(row["block_id"], []) if annotations else []
    for item in spans:
        if (item["start"] < row["end"] and row["start"] < item["end"]
                and not row["start"] <= item["start"] < item["end"] <= row["end"]
                and item["word"] in snapshot["names"][item["name_id"]]):
            raise ValueError("Confirmed name pronunciation crosses a script segment boundary; resolve the segmentation")
    result = []
    for match in WORD_PATTERN.finditer(row["text"]):
        start, end = row["start"] + match.start(), row["start"] + match.end()
        marked = next((item for item in spans if item["start"] == start and item["end"] == end), None)
        word = match.group()
        if marked:
            entry = snapshot["names"][marked["name_id"]].get(word)
        else:
            entry = snapshot["special"].get(word)
            if entry is None and word not in snapshot["migrated"]:
                entry = legacy.get(word)
        if entry is not None:
            result.append((match, entry))
    return result
