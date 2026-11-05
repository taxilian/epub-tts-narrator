"""Reconstruct and validate manual speaker decisions against a trusted pre-UI snapshot."""

from copy import deepcopy

from .contracts import readiness
from .pipeline import fingerprint
from .speaker_completion import ATTRIBUTION_ISSUE_KINDS, unassigned_segments
from .speaker_decisions import _create_voice_records, _role_labels


_DECISION_FIELDS = {"segment_id", "block_id", "start", "end", "text_fingerprint",
                    "selected_roles", "selected_voices", "previous_attribution", "previous_kind",
                    "kind", "simultaneous"}
_DEFINED_FIELDS = {"character_id", "voice_id", "name", "created"}


def _require(condition, message):
    if not condition:
        raise ValueError("Invalid speaker-decisions revision: " + message)


def _assign(segment, roles, labels):
    before = deepcopy(segment)
    if len(roles) == 1:
        role = roles[0]
        segment.update(role_id=role["id"], voice_id=role["voice_id"])
        segment.pop("simultaneous_speakers", None)
        if role["kind"] == "narration" and segment["kind"] in {"dialogue", "thought", "letter"}:
            segment["kind"] = "narration"
        elif role["kind"] == "dialogue" and segment["kind"] not in {"dialogue", "thought", "letter"}:
            segment["kind"] = "dialogue"
    else:
        segment.update(role_id=None, voice_id=None,
                       simultaneous_speakers=[{"role_id": role["id"], "voice_id": role["voice_id"]}
                                              for role in roles])
        if segment["kind"] not in {"dialogue", "thought", "letter"}:
            segment["kind"] = "dialogue"
    selected = "; ".join(labels[role["id"]] for role in roles)
    segment["attribution"] = {"basis": "user", "reason": "User selected " + selected + ".",
                              "evidence": deepcopy(before["attribution"]["evidence"]),
                              "candidates": [], "review_required": False}


def _apply_decisions(expected, speaker_input, receipt):
    decisions = receipt.get("decisions")
    _require(type(receipt.get("version")) is int and receipt["version"] == 1
             and isinstance(decisions, list) and bool(decisions),
             "a new receipt must contain version-1 decisions")
    _require(receipt.get("source_fingerprint") == fingerprint(speaker_input["source"])
             and receipt.get("input_script_fingerprint") == fingerprint(speaker_input["segments"]),
             "receipt source or input script does not match the trusted pre-UI document")
    positions = {segment["id"]: index for index, segment in enumerate(speaker_input["segments"])}
    pending = {segment["id"] for segment in unassigned_segments(speaker_input)}
    expected["cast"] = deepcopy(speaker_input["cast"])
    roles = expected["cast"]["roles"]
    characters = {row["id"]: row for row in expected["cast"]["characters"]}
    available = {role["id"]: role for role in roles}
    labels = _role_labels(roles, characters, expected["source"]["book"])
    known_ids = {row["id"] for collection in expected["cast"].values() if isinstance(collection, list)
                 for row in collection if isinstance(row, dict) and isinstance(row.get("id"), str)}
    seen = set()
    for decision in decisions:
        _require(isinstance(decision, dict)
                 and set(decision) in (_DECISION_FIELDS, _DECISION_FIELDS | {"defined_voice"}),
                 "decision fields do not match the terminal receipt schema")
        segment_id = decision.get("segment_id")
        _require(segment_id in pending and segment_id not in seen, "decision target is not uniquely pending")
        seen.add(segment_id)
        before = speaker_input["segments"][positions[segment_id]]
        _require(decision.get("block_id") == before["block_id"]
                 and decision.get("start") == before["start"] and decision.get("end") == before["end"]
                 and decision.get("text_fingerprint") == fingerprint(before["text"])
                 and decision.get("previous_attribution") == before["attribution"]
                 and decision.get("previous_kind") == before["kind"],
                 "decision target, text, or previous attribution does not match the trusted segment")
        selected_ids = decision.get("selected_roles")
        selected_voices = decision.get("selected_voices")
        if (not isinstance(selected_ids, list) or not selected_ids
                or not isinstance(selected_voices, list) or len(selected_voices) != len(selected_ids)):
            raise ValueError("Invalid speaker-decisions revision: selected roles and voices must be nonempty paired lists")
        defined = decision.get("defined_voice")
        if defined is not None:
            _require(isinstance(defined, dict) and set(defined) == _DEFINED_FIELDS
                     and isinstance(defined.get("created"), dict)
                     and set(defined["created"]) == {"character", "voice", "role"}
                     and all(isinstance(defined["created"].get(key), dict)
                             for key in ("character", "voice", "role")),
                 "defined voice does not match the terminal receipt schema")
            name = defined.get("name")
            description = defined["created"]["voice"].get("description")
            _require(isinstance(name, str) and bool(name.strip()) and name == name.strip()
                     and isinstance(description, str) and bool(description.strip()),
                 "defined voice needs the entered name and description")
            character, voice, role = _create_voice_records(expected, name, description)
            created = {"character": character, "voice": voice, "role": role}
            _require(created == defined["created"]
                     and defined.get("character_id") == character["id"]
                     and defined.get("voice_id") == voice["id"],
                     "defined voice is not the deterministic append-only created triple")
            created_ids = (character["id"], voice["id"], role["id"])
            _require(all(isinstance(value, str) and value not in known_ids for value in created_ids)
                     and created_ids[0] not in created_ids[1:] and created_ids[1] != created_ids[2],
                     "defined voice IDs collide with an existing or earlier created record")
            known_ids.update(created_ids)
            characters[character["id"]] = character
            available[role["id"]] = role
            labels[role["id"]] = name
            _require(selected_ids == [role["id"]] and selected_voices == [voice["id"]],
                     "a defining decision must select exactly its new role and voice")
        _require(all(role_id in available for role_id in selected_ids), "selected role is not available")
        selected = [available[role_id] for role_id in selected_ids]
        _require(selected_voices == [role["voice_id"] for role in selected]
                 and len(set(selected_ids)) == len(selected_ids),
                 "selected voices do not match distinct selected roles")
        _require(len(selected) == 1 or (all(role["kind"] == "dialogue" for role in selected)
                                       and len(set(selected_voices)) == len(selected_voices)),
                 "simultaneous choices require distinct dialogue roles and voices")
        segment = expected["segments"][positions[segment_id]]
        _assign(segment, selected, labels)
        _require(decision.get("kind") == segment["kind"]
                 and decision.get("simultaneous") is (len(selected) > 1),
                 "decision kind or simultaneous flag does not match its assignment")
    remaining = unassigned_segments(expected)
    covered = {row["segment_id"] for row in decisions}
    covered.update(row["block_id"] for row in decisions
                   if not any(item["block_id"] == row["block_id"] for item in remaining))
    for issue in expected["issues"]:
        if not issue["resolved"] and issue["kind"] in ATTRIBUTION_ISSUE_KINDS and issue["target"] in covered:
            issue.update(resolved=True, resolution={"method": "speaker_decision", "basis": "user"})
    provenance = expected["provenance"]
    if "speaker_decisions" in speaker_input["provenance"]:
        history = speaker_input["provenance"].get("speaker_decision_history", [])
        _require(isinstance(history, list), "prior decision history must be a list")
        provenance["speaker_decision_history"] = deepcopy(history) + [
            deepcopy(speaker_input["provenance"]["speaker_decisions"])]
    provenance["speaker_decisions"] = {
        "version": 1, "source_fingerprint": fingerprint(expected["source"]),
        "cast_fingerprint": fingerprint(expected["cast"]),
        "input_script_fingerprint": fingerprint(speaker_input["segments"]),
        "output_script_fingerprint": fingerprint(expected["segments"]),
        "decisions": deepcopy(decisions), "remaining_unassigned_segments": len(remaining)}


def _validate_samples(expected, speaker_input, candidate):
    before = speaker_input["provenance"].get("voice_samples")
    after = candidate["provenance"].get("voice_samples")
    if before is None:
        _require(after is None, "audition metadata cannot be introduced by speaker decisions")
        return
    _require(isinstance(before, dict) and isinstance(after, dict)
             and after.get("source_fingerprint") == fingerprint(expected["source"])
             and after.get("cast_fingerprint") == fingerprint(expected["cast"]),
             "voice_samples must retain source and bind the post-decision cast")
    if after == before:
        return
    old_scripts, new_scripts = before.get("scripts", {}), after.get("scripts", {})
    _require(isinstance(old_scripts, dict) and isinstance(new_scripts, dict)
             and all(new_scripts.get(voice) == script for voice, script in old_scripts.items()),
             "existing voice_samples scripts must be preserved")
    expected["provenance"]["voice_samples"] = deepcopy(after)
    from .audition_scripts import prepare_document_samples
    prepared = prepare_document_samples(expected, None, allow_incomplete=True)
    _require(prepared["provenance"]["voice_samples"] == after,
             "voice_samples is not a stable product of audition preparation")


def validate_manual_speaker_revision(original, speaker_input, candidate):
    """Require candidate to equal the deterministic result of its manual receipt."""
    _require(isinstance(speaker_input, dict) and isinstance(candidate, dict)
             and isinstance(speaker_input.get("provenance"), dict)
             and isinstance(candidate.get("provenance"), dict)
             and speaker_input.get("source") == original.get("source")
             and speaker_input.get("cast") == original.get("cast")
             and speaker_input.get("provenance", {}).get("series")
             == original.get("provenance", {}).get("series"),
             "trusted pre-UI input must preserve original source, cast, and series identities")
    expected = deepcopy(speaker_input)
    receipt = candidate.get("provenance", {}).get("speaker_decisions")
    if receipt != speaker_input.get("provenance", {}).get("speaker_decisions"):
        _require(isinstance(receipt, dict), "changed receipt must be an object")
        _apply_decisions(expected, speaker_input, receipt)
    _validate_samples(expected, speaker_input, candidate)
    expected["readiness"] = readiness(expected)
    if candidate != expected:
        keys = sorted(set(candidate) | set(expected))
        differences = [key for key in keys if candidate.get(key) != expected.get(key)]
        _require(False, "candidate differs from recorded decisions and permitted updates: "
                 + ", ".join(differences))
    return candidate["cast"] != original["cast"]
