"""Pure, cast-preserving manual attribution segment edits."""

from __future__ import annotations

from copy import deepcopy


_SIMULTANEOUS_KINDS = {"dialogue", "thought", "letter"}


def edit_speaker(segment, role_ids, roles_by_id, labels, candidates=None):
    """Return a segment with an existing-cast speaker choice; never mutate input.

    New voices are deliberately outside this UI: attribution revisions must
    preserve the published cast. Simultaneous speech needs distinct dialogue
    roles and voices and an already simultaneous-eligible passage kind.
    """
    result = deepcopy(segment)
    selected_ids = list(role_ids or [])
    if not selected_ids:
        result.update(role_id=None, voice_id=None)
        result.pop("simultaneous_speakers", None)
        attribution = deepcopy(segment["attribution"])
        attribution.update(basis="unresolved", review_required=True)
        if candidates is not None:
            attribution["candidates"] = deepcopy(list(candidates))
        result["attribution"] = attribution
        return result
    try:
        roles = [roles_by_id[role_id] for role_id in selected_ids]
    except KeyError as exc:
        raise ValueError(f"Unknown speaker role id: {exc.args[0]}") from None
    if len(roles) == 1:
        role = roles[0]
        result.update(role_id=role["id"], voice_id=role["voice_id"])
        result.pop("simultaneous_speakers", None)
        if role["kind"] == "narration" and result["kind"] in _SIMULTANEOUS_KINDS:
            result["kind"] = "narration"
        elif role["kind"] == "dialogue" and result["kind"] not in _SIMULTANEOUS_KINDS:
            result["kind"] = "dialogue"
    else:
        if result["kind"] not in _SIMULTANEOUS_KINDS:
            raise ValueError("Simultaneous speech requires dialogue, thought, or letter text")
        if len({role["id"] for role in roles}) != len(roles):
            raise ValueError("Simultaneous speaker roles must be distinct")
        if any(role["kind"] != "dialogue" for role in roles):
            raise ValueError("Simultaneous speech requires dialogue roles; narration cannot be a group member")
        if len({role["voice_id"] for role in roles}) != len(roles):
            raise ValueError("Simultaneous speaker voices must be distinct")
        result.update(role_id=None, voice_id=None,
                      simultaneous_speakers=[{"role_id": role["id"], "voice_id": role["voice_id"]}
                                             for role in roles])
    selected = "; ".join(labels[role["id"]] for role in roles)
    result["attribution"] = {
        "basis": "user", "reason": "User selected " + selected + ".",
        "evidence": deepcopy(segment["attribution"]["evidence"]),
        "candidates": [], "review_required": False,
    }
    return result


def edit_direction(segment, direction):
    """Return a segment with bounded, whitespace-trimmed delivery direction.

    The direction is marked as typed by a person (``delivery.basis = "user"``),
    so direction refreshes keep it (``direction_policy.typed``).
    """
    if not isinstance(direction, str):
        raise ValueError("Delivery direction must be a string")
    value = direction.strip()
    if len(value) > 240:
        raise ValueError(f"Delivery direction exceeds 240 characters (received {len(value)})")
    result = deepcopy(segment)
    result["delivery"]["direction"] = value
    result["delivery"]["basis"] = "user"
    return result


def edit_review(segment, review_required):
    """Return a segment with its review flag changed, preserving unresolved policy."""
    if segment["attribution"]["basis"] == "unresolved" and not review_required:
        raise ValueError("Unresolved attribution requires review_required=True; assign a speaker first")
    result = deepcopy(segment)
    result["attribution"]["review_required"] = bool(review_required)
    return result


def apply_edits(document, edits):
    """Return a deep-copied document with the supplied segment IDs replaced."""
    result = deepcopy(document)
    positions = {segment["id"]: index for index, segment in enumerate(result["segments"])}
    unknown = [segment_id for segment_id in edits if segment_id not in positions]
    if unknown:
        raise ValueError("Unknown segment id(s): " + ", ".join(sorted(unknown)))
    for segment_id, segment in edits.items():
        if segment.get("id") != segment_id:
            raise ValueError(f"Replacement segment id does not match edit key: {segment_id}")
        result["segments"][positions[segment_id]] = deepcopy(segment)
    return result
