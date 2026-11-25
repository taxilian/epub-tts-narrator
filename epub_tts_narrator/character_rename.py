"""Renaming a cast character: a misspelled name, or the name a person wants the character called.

A rename changes only words: the character's ``name``, its aliases, and its
old name where its own description and roles' descriptions use it. Every ID,
voice, line and recording stays, so nothing is rendered again. The old name
becomes an alias when the book's text uses it (an epithet or an earlier name
the book really says), and is dropped otherwise (a misspelling); a new name
that was an alias stops being one. A new name another entry already has is
refused, because two entries for one person are merged with
``merge-characters`` instead. Each rename is recorded in
``provenance.character_renames``; the ``character-rename`` revision scope
(``analysis_revision``) accepts exactly this change.
"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata

from .character_merge import name_key


def clean_name(name):
    if not isinstance(name, str):
        raise ValueError("A character's new name must be text")
    name = " ".join(unicodedata.normalize("NFC", name).split())
    if not name or len(name) > 80 or any(unicodedata.category(char).startswith("C") for char in name):
        raise ValueError("A character's new name must be 1-80 printable characters")
    return name


def _replace(text, old, new):
    if not isinstance(text, str) or not old.strip():
        return text
    return re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", lambda _: new, text)


def used_in_text(document, name):
    pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I)
    return any(pattern.search(block["text"]) for block in document["source"]["blocks"])


def rename_document(document, character_id, new_name, *, old_name=None, basis=None):
    """``document`` with ``character_id`` called ``new_name`` (``old_name``: the name to replace, default its current one).

    ``basis="analysis"`` records a rename the analysis made (``name_review``: the name the book gives); a
    record without a basis is a person's, which the analysis never renames again.
    """
    new_name = clean_name(new_name)
    revised = deepcopy(document)
    cast = revised["cast"]
    character = next((row for row in cast["characters"] if row["id"] == character_id), None)
    if character is None:
        raise ValueError(f"No cast character {character_id}")
    old = character["name"] if old_name is None else old_name
    if name_key(new_name) == name_key(old) and new_name == old:
        raise ValueError(f"{old} is already called {new_name}")
    for row in cast["characters"]:
        if row["id"] != character_id and name_key(new_name) in {
                name_key(name) for name in [row["name"], *row.get("aliases", [])] if name.strip()}:
            raise ValueError(f"{new_name!r} is already the name of another cast entry ({row['name']}, {row['id']}); "
                             "if they are one person, merge them with merge-characters")
    keep_old = name_key(old) != name_key(new_name) and used_in_text(document, old)
    aliases = [alias for alias in character.get("aliases", []) if name_key(alias) != name_key(new_name)]
    if keep_old and name_key(old) not in {name_key(alias) for alias in aliases}:
        aliases.insert(0, old)
    character.update(name=new_name, aliases=aliases, description=_replace(character.get("description"), old, new_name))
    for role in cast["roles"]:
        if role.get("character_id") == character_id:
            role["description"] = _replace(role.get("description"), old, new_name)
    samples = revised["provenance"].get("voice_samples")
    if isinstance(samples, dict) and "cast_fingerprint" in samples:
        from .pipeline import fingerprint
        samples["cast_fingerprint"] = fingerprint(cast)
    record = {"version": 1, "character_id": character_id, "from": old, "to": new_name,
              "old_name_kept_as_alias": keep_old}
    if basis is not None:
        if basis != "analysis":
            raise ValueError("A rename's basis is a person's (none) or \"analysis\"")
        record["basis"] = basis
    revised["provenance"].setdefault("character_renames", []).append(record)
    return revised


def validate_character_rename(document, candidate):
    """Refuse anything but the renames ``candidate`` newly records over ``document``."""
    before = document["provenance"].get("character_renames", [])
    after = candidate["provenance"].get("character_renames")
    if not isinstance(after, list) or after[:len(before)] != before or len(after) == len(before):
        raise ValueError("A character rename must record each rename in provenance.character_renames")
    expected = document
    for record in after[len(before):]:
        if not isinstance(record, dict) or not isinstance(record.get("character_id"), str):
            raise ValueError("Invalid provenance.character_renames record")
        expected = rename_document(expected, record["character_id"], record.get("to"), old_name=record.get("from"),
                                   basis=record.get("basis"))
        if expected["provenance"]["character_renames"][-1] != record:
            raise ValueError("A character rename record does not match the rename it describes")
    if candidate != expected:
        raise ValueError("A character rename may change only the character's name, aliases and the descriptions "
                         "that use its old name")
