"""Merge two cast entries of a published book into one character.

A person says two cast entries are the same character. The surviving entry
keeps its character ID, voice ID (and so its recorded voice), description and
per-voice settings; the absorbed entry's names join its aliases and its
evidence joins its evidence, every line it spoke becomes the survivor's line,
and the absorbed character and its main voice leave the cast. Roles that only
duplicate one of the survivor's roles merge into it; the absorbed entry's other
roles (a changed-voice state, a separate narration voice) move to the survivor.
A simultaneous passage the two entries spoke together becomes the survivor's
line alone.

The change is deterministic from a small receipt recorded in
``provenance.character_merges``, so publication can recompute and compare the
candidate exactly (``validate_character_merge``). Segments whose speaker
changed are listed there as ``revoiced_segments``; the renderer folds their
current speakers into their chunk identity so they are rendered again under a
new chunk ID.
"""

from copy import deepcopy
import unicodedata

from .contracts import readiness, validate_document
from .pipeline import fingerprint


VERSION = 1
SPOKEN_KINDS = {"dialogue", "thought", "letter"}
_RECEIPT_FIELDS = {"version", "kept_character_id", "absorbed_character_id", "name"}
_RECORD_FIELDS = _RECEIPT_FIELDS | {"kept_records", "kept_voice_id", "absorbed_voice_id", "absorbed_names",
                                    "role_map", "revoiced_segments", "source_fingerprint",
                                    "cast_fingerprint_before", "cast_fingerprint_after", "basis"}


def name_key(name):
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def _names(character):
    return [name for name in [character["name"], *character["aliases"]] if name.strip()]


def character_roles(document, character_id):
    return [role for role in document["cast"]["roles"] if role.get("character_id") == character_id]


def segment_speakers(segment):
    """``[(role_id, voice_id)]`` for one segment: its performer, every simultaneous one, or none."""
    if "simultaneous_speakers" in segment:
        return [(speaker["role_id"], speaker["voice_id"]) for speaker in segment["simultaneous_speakers"]]
    if segment.get("role_id"):
        return [(segment["role_id"], segment["voice_id"])]
    return []


def character_lines(document, character_id):
    """How many segments this character speaks or narrates (simultaneous passages included)."""
    roles = {role["id"] for role in character_roles(document, character_id)}
    return sum(1 for segment in document["segments"]
               if any(role in roles for role, _ in segment_speakers(segment)))


def resolve_character(document, text):
    """The one cast character a person means by an ID, its voice ID, or a name/alias."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Name a character by its name, an alias, its character ID or its voice ID")
    characters = document["cast"]["characters"]
    exact = [row for row in characters if text in (row["id"], row["voice_id"])]
    if exact:
        return exact[0]
    key = name_key(text)
    named = [row for row in characters if row["name"].strip() and name_key(row["name"]) == key]
    if not named:
        named = [row for row in characters if key in {name_key(name) for name in _names(row)}]
    if len(named) == 1:
        return named[0]
    if named:
        raise ValueError(f"{text!r} names several characters; use one of their IDs: "
                         + ", ".join(f"{row['name']} ({row['id']})" for row in named))
    raise ValueError(f"No character in this book is called {text!r}; use a name, alias, character ID or voice ID "
                     "from the cast")


def combined_names(kept, absorbed):
    names = []
    seen = set()
    for name in [*_names(kept), *_names(absorbed)]:
        if name_key(name) not in seen:
            seen.add(name_key(name))
            names.append(name)
    return names


def receipt(kept_character_id, absorbed_character_id, name, *, kept_records=None):
    record = {"version": VERSION, "kept_character_id": kept_character_id,
              "absorbed_character_id": absorbed_character_id, "name": name}
    if kept_records is not None:
        record["kept_records"] = deepcopy(kept_records)
    return record


def carried_voice(voice, *, series_id, source_fingerprint):
    """The survivor's voice record as a later book inherits it: casting-only, citing no text of its own."""
    voice = deepcopy(voice)
    if "series_origin" in voice:
        return voice
    origin = {"kind": "series_casting_reuse", "series_id": series_id,
              "source_fingerprint": source_fingerprint, "voice_id": voice["id"]}
    voice["series_origin"] = origin
    for index, trait in enumerate(voice["traits"]):
        trait["series_origin"] = dict(origin, trait_index=index, original_basis=trait["basis"])
        trait["basis"], trait["evidence"] = "casting", []
    return voice


def kept_records_for(merged, character_id):
    """Records a later book that lacks the survivor needs to take it in, from a merged book."""
    character = next(row for row in merged["cast"]["characters"] if row["id"] == character_id)
    voice = next(row for row in merged["cast"]["voices"] if row["id"] == character["voice_id"])
    carried = {key: deepcopy(value) for key, value in character.items()
               if key not in {"evidence", "series_identity"}}
    carried["evidence"] = []
    series_id = merged["provenance"].get("series", {}).get("series_id")
    return {"character": carried,
            "voice": carried_voice(voice, series_id=series_id, source_fingerprint=fingerprint(merged["source"]))}


def _require(condition, message):
    if not condition:
        raise ValueError("Cannot merge these characters: " + message)


def _dedupe(values):
    return list(dict.fromkeys(values))


def merge_document(document, merge):
    """Return ``document`` with ``merge["absorbed_character_id"]`` merged into the kept character.

    ``merge`` is a receipt from :func:`receipt`. ``kept_records`` supplies the
    kept character and voice records when this book (a later book in the
    series) has only the absorbed entry.
    """
    _require(isinstance(merge, dict) and set(merge) <= _RECEIPT_FIELDS | {"kept_records"}
             and merge.get("version") == VERSION, "the merge receipt is not version 1")
    result = deepcopy(document)
    cast = result["cast"]
    kept_id, absorbed_id = merge.get("kept_character_id"), merge.get("absorbed_character_id")
    _require(isinstance(kept_id, str) and isinstance(absorbed_id, str) and kept_id != absorbed_id,
             "choose two different characters")
    characters = {row["id"]: row for row in cast["characters"]}
    _require(absorbed_id in characters, f"{absorbed_id} is not in this book's cast")
    cast_before = fingerprint(document["cast"])
    records = merge.get("kept_records")
    if kept_id not in characters:
        _require(isinstance(records, dict) and set(records) == {"character", "voice"}
                 and isinstance(records["character"], dict) and records["character"].get("id") == kept_id
                 and isinstance(records["voice"], dict)
                 and records["voice"].get("id") == records["character"].get("voice_id"),
                 f"{kept_id} is not in this book's cast")
        voices = {row["id"] for row in cast["voices"]}
        _require(records["voice"]["id"] not in voices
                 and not any(row["voice_id"] == records["voice"]["id"] for row in cast["characters"]),
                 "the kept voice already belongs to another record in this book")
        cast["characters"].append(deepcopy(records["character"]))
        cast["voices"].append(deepcopy(records["voice"]))
        characters[kept_id] = cast["characters"][-1]
    else:
        _require(records is None, "kept records are only for a book without the kept character")
    kept, absorbed = characters[kept_id], characters[absorbed_id]
    kept_voice, absorbed_voice = kept["voice_id"], absorbed["voice_id"]
    voices = {row["id"]: row for row in cast["voices"]}
    name = merge.get("name")
    _require(isinstance(name, str) and name.strip() == name and bool(name)
             and name_key(name) in {name_key(item) for item in combined_names(kept, absorbed)},
             "the merged name must be one of the two entries' names or aliases")

    def ends_changed(voice_id):
        return any(isinstance(voice.get("derived_from"), dict) and voice["derived_from"].get("voice_id") == voice_id
                   and voice["derived_from"].get("through_end") is True for voice in voices.values())
    _require(not (ends_changed(kept_voice) and ends_changed(absorbed_voice)),
             "both end the book in a changed voice; keep only one of those states first")

    before = {segment["id"]: segment_speakers(segment) for segment in result["segments"]}
    # Roles: a role of the absorbed entry on its main voice duplicates one of
    # the survivor's roles of the same kind when the survivor has one.
    role_map, rebased = {}, set()
    for role in list(cast["roles"]):
        if role.get("character_id") == absorbed_id and role["voice_id"] == absorbed_voice:
            target = next((other for other in cast["roles"] if other.get("character_id") == kept_id
                           and other["kind"] == role["kind"] and other["voice_id"] == kept_voice), None)
            if target is not None:
                role_map[role["id"]] = target["id"]
                continue
        if role.get("character_id") == absorbed_id:
            role["character_id"] = kept_id
        if role["voice_id"] == absorbed_voice:
            role["voice_id"] = kept_voice
    cast["roles"] = [role for role in cast["roles"] if role["id"] not in role_map]
    for voice in cast["voices"]:
        derived = voice.get("derived_from")
        if isinstance(derived, dict) and derived.get("voice_id") == absorbed_voice:
            derived["voice_id"] = kept_voice
            rebased.add(voice["id"])
    cast["voices"] = [voice for voice in cast["voices"] if voice["id"] != absorbed_voice]
    kept_name = kept["name"]
    kept["aliases"] = [alias for alias in _dedupe([kept_name, *kept["aliases"], absorbed["name"], *absorbed["aliases"]])
                       if alias.strip() and name_key(alias) != name_key(name)]
    seen, aliases = set(), []
    for alias in kept["aliases"]:
        if name_key(alias) not in seen:
            seen.add(name_key(alias))
            aliases.append(alias)
    kept["aliases"] = aliases
    kept["name"] = name
    kept["evidence"] = [*kept["evidence"], *[item for item in absorbed["evidence"] if item not in kept["evidence"]]]
    cast["characters"] = [row for row in cast["characters"] if row["id"] != absorbed_id]

    roles = {role["id"]: role for role in cast["roles"]}

    def mapped(role_id):
        return role_map.get(role_id, role_id)

    for segment in result["segments"]:
        attribution = segment["attribution"]
        attribution["candidates"] = _dedupe(mapped(role) for role in attribution["candidates"])
        if "simultaneous_speakers" in segment:
            members = _dedupe(mapped(speaker["role_id"]) for speaker in segment["simultaneous_speakers"])
            if len(members) == 1:
                segment.pop("simultaneous_speakers")
                segment.update(role_id=members[0], voice_id=roles[members[0]]["voice_id"])
            else:
                segment["simultaneous_speakers"] = [{"role_id": role, "voice_id": roles[role]["voice_id"]}
                                                    for role in members]
        elif segment.get("role_id"):
            segment["role_id"] = mapped(segment["role_id"])
            segment["voice_id"] = roles[segment["role_id"]]["voice_id"]
    revoiced = [segment["id"] for segment in result["segments"]
                if segment_speakers(segment) != before[segment["id"]]
                or any(voice in rebased for _, voice in segment_speakers(segment))]

    identities = {absorbed_id: kept_id, absorbed_voice: kept_voice, **role_map}
    for issue in result["issues"]:
        issue["target"] = identities.get(issue["target"], issue["target"])
    provenance = result["provenance"]
    series = provenance.get("series")
    if isinstance(series, dict):
        for field in ("character_id_map", "voice_id_map"):
            if isinstance(series.get(field), dict):
                series[field] = {local: identities.get(value, value) for local, value in series[field].items()}
    samples = provenance.get("voice_samples")
    if isinstance(samples, dict):
        for field in ("scripts", "samples", "pending"):
            if isinstance(samples.get(field), dict):
                samples[field].pop(absorbed_voice, None)
        if "pending" in samples and not samples["pending"]:
            samples.pop("pending")
        if "cast_fingerprint" in samples:
            samples["cast_fingerprint"] = fingerprint(cast)
    record = {**{key: deepcopy(value) for key, value in merge.items()},
              "kept_voice_id": kept_voice, "absorbed_voice_id": absorbed_voice,
              "absorbed_names": _names(absorbed), "role_map": role_map, "revoiced_segments": revoiced,
              "source_fingerprint": fingerprint(result["source"]), "cast_fingerprint_before": cast_before,
              "cast_fingerprint_after": fingerprint(cast), "basis": "user"}
    provenance["character_merges"] = [*deepcopy(provenance.get("character_merges", [])), record]
    errors = validate_document(result)
    _require(not errors, "the merged analysis would be invalid: " + "; ".join(errors[:6]))
    result["readiness"] = readiness(result)
    return result


def validate_character_merge(original, candidate):
    """Require ``candidate`` to be exactly ``original`` with its one new recorded merge applied."""
    before = original["provenance"].get("character_merges", [])
    after = candidate.get("provenance", {}).get("character_merges")
    if (not isinstance(after, list) or len(after) != len(before) + 1 or after[:-1] != before
            or not isinstance(after[-1], dict) or set(after[-1]) - _RECORD_FIELDS):
        raise ValueError("Invalid character-merge revision: it must add exactly one merge record")
    expected = merge_document(original, {key: after[-1][key] for key in (_RECEIPT_FIELDS | {"kept_records"})
                                         if key in after[-1]})
    if candidate != expected:
        differences = sorted(key for key in set(candidate) | set(expected) if candidate.get(key) != expected.get(key))
        raise ValueError("Invalid character-merge revision: candidate differs from its recorded merge: "
                         + ", ".join(differences))


def revoiced_segment_ids(provenance):
    """Segment IDs whose speaker a character merge changed (the renderer rebinds their chunk identity)."""
    merges = provenance.get("character_merges", []) if isinstance(provenance, dict) else []
    if not isinstance(merges, list):
        raise ValueError("Invalid provenance.character_merges")
    result = set()
    for merge in merges:
        segments = merge.get("revoiced_segments") if isinstance(merge, dict) else None
        if not isinstance(segments, list) or not all(isinstance(item, str) for item in segments):
            raise ValueError("Invalid provenance.character_merges revoiced_segments")
        result.update(segments)
    return result


def remap_decisions(decisions, merged):
    """A pending speaker-decision journal's choices restated for the merged cast, or None if one no longer fits."""
    role_map = merged["provenance"]["character_merges"][-1]["role_map"]
    roles = {role["id"]: role for role in merged["cast"]["roles"]}
    result = []
    for decision in decisions:
        if not isinstance(decision, dict) or not isinstance(decision.get("selected_roles"), list):
            return None
        entry = deepcopy(decision)
        previous = entry.get("previous_attribution")
        if isinstance(previous, dict) and isinstance(previous.get("candidates"), list):
            previous["candidates"] = _dedupe(role_map.get(role, role) for role in previous["candidates"])
        if entry.get("defined_voice") is not None:
            # A voice the person defined selects only its own new role, replayed as it was.
            result.append(entry)
            continue
        selected = _dedupe(role_map.get(role, role) for role in entry["selected_roles"])
        if not selected or not all(role in roles for role in selected):
            return None
        entry["selected_roles"] = selected
        entry["selected_voices"] = [roles[role]["voice_id"] for role in selected]
        entry["simultaneous"] = len(selected) > 1
        result.append(entry)
    return result
