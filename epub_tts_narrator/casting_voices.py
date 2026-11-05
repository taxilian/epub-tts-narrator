"""Consolidate voice facts after approved candidate identity merges."""

from copy import deepcopy

from .pipeline import fingerprint


def consolidate_merged_voices(cast, prior_casts, approved_merges):
    """Archive superseded local voices without losing their source-backed traits."""
    updated = deepcopy(cast)
    ownership = {character["id"]: character["voice_id"]
                 for prior in prior_casts for character in prior["characters"]}
    parents = {merge["from_character_id"]: merge["into_character_id"] for merge in approved_merges}
    characters = {character["id"]: character for character in updated["characters"]}
    voices = {voice["id"]: voice for voice in updated["voices"]}
    referenced = {row["voice_id"] for row in [*updated["characters"], *updated["roles"]]}
    superseded = {}
    for origin in parents:
        target = parents[origin]
        while target in parents:
            target = parents[target]
        if target in characters and origin in ownership:
            old_voice = ownership[origin]
            new_voice = characters[target]["voice_id"]
            if old_voice != new_voice:
                superseded[old_voice] = new_voice
    archive, issues = [], []
    for old_id, target_id in superseded.items():
        if old_id in referenced or old_id not in voices:
            continue
        original, target = voices[old_id], voices[target_id]
        if original.get("series_origin") or target.get("series_origin"):
            continue
        known = {fingerprint({key: value for key, value in trait.items() if key != "evidence"}): trait
                 for trait in target["traits"] if trait["basis"] == "text"}
        for trait in original["traits"]:
            if trait["basis"] != "text":
                continue
            key = fingerprint({field: value for field, value in trait.items() if field != "evidence"})
            if key not in known:
                carried = deepcopy(trait)
                target["traits"].append(carried)
                known[key] = carried
            else:
                evidence_keys = {fingerprint(item) for item in known[key]["evidence"]}
                for item in trait["evidence"]:
                    evidence_key = fingerprint(item)
                    if evidence_key not in evidence_keys:
                        known[key]["evidence"].append(deepcopy(item))
                        evidence_keys.add(evidence_key)
        values = {}
        for trait in target["traits"]:
            if trait["basis"] == "text":
                values.setdefault(trait["name"].casefold(), set()).add(trait["value"])
        if any(len(options) > 1 for options in values.values()):
            issues.append({"kind": "casting_traits", "target": target_id,
                           "reason": "Merged candidates contain different source-supported values for a vocal trait; "
                                     "review the retained evidence before casting."})
        archive.append({"voice": deepcopy(original), "canonical_voice_id": target_id,
                        "reason": "approved_identity_merge"})
    archived_ids = {row["voice"]["id"] for row in archive}
    updated["voices"] = [voice for voice in updated["voices"] if voice["id"] not in archived_ids]
    return updated, archive, issues
