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
    # A line's group override and unison mode belong to its old speakers.
    if selected_ids != ([segment["role_id"]] if segment.get("role_id") else
                        [speaker["role_id"] for speaker in segment.get("simultaneous_speakers", [])]):
        result.pop("ensemble", None)
        result.pop("simultaneous_mode", None)
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


def edit_ensemble(segment, voices_by_id, *, mode=None, members=None, lanes=None, slot=None):
    """Return a group line with a person's override (``COLLECTIVE_VOICES.md``); all None clears it.

    ``members`` is a subset of the group's ensemble voices (the first one pins
    a pool or alternate line's voice); ``lanes`` how many voices say a line
    together. Only a line whose group a person cast as an ensemble takes one.
    """
    from .ensembles import MAX_VOICES, MODES, ensemble, lanes_of
    lead = voices_by_id.get(segment.get("voice_id"))
    if ensemble(lead) is None:
        raise ValueError("This line's group has no voices cast yet; set them up in the cast screen (C) first")
    result = deepcopy(segment)
    previous = deepcopy(segment.get("ensemble") or {})
    if mode is None and members is None and lanes is None and slot is None:
        result.pop("ensemble", None)
        return result
    value = {}
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}")
        value["mode"] = mode
    if members is not None:
        allowed = lanes_of(lead)
        members = list(dict.fromkeys(members))
        if not members or not set(members) <= set(allowed):
            raise ValueError("Choose voices of this group's ensemble")
        value["members"] = members
    if lanes is not None:
        if type(lanes) is not int or not 1 <= lanes <= MAX_VOICES:
            raise ValueError(f"A line takes 1 to {MAX_VOICES} voices")
        value["lanes"] = lanes
    slot = previous.get("slot") if slot is None else slot
    if slot is not None:
        value["slot"] = slot
    result["ensemble"] = dict(value, basis="user")
    return result


def edit_murmur(segment, *, group=None, until=None, shape=None, level=None, remove=False):
    """Return a segment with a person's crowd murmur starting here (or removed).

    A new murmur needs ``group`` and ``until``; changing one keeps what is not
    given. The span's other checks (a later segment of the same chapter, a cast
    group) are the document contract's.
    """
    from .ensembles import MURMUR_LEVELS, MURMUR_SHAPES
    result = deepcopy(segment)
    if remove:
        result.pop("murmur", None)
        return result
    current = deepcopy(segment.get("murmur") or {})
    value = {key: current[key] for key in ("group", "until", "shape", "level", "from_line") if key in current}
    for key, given in (("group", group), ("until", until), ("shape", shape), ("level", level)):
        if given is not None:
            value[key] = given
    if not value.get("group") or not value.get("until"):
        raise ValueError("A murmur needs a group and the passage where it ends")
    if value.get("shape", "steady") not in MURMUR_SHAPES or value.get("level", "low") not in MURMUR_LEVELS:
        raise ValueError("Unknown murmur shape or level")
    result["murmur"] = dict(value, basis="user")
    return result


def edit_simultaneous_mode(segment, mode):
    """Return a line said by named characters together, entering ``mode`` (unison clears it)."""
    from .ensembles import SIMULTANEOUS_MODES
    if not segment.get("simultaneous_speakers"):
        raise ValueError("Only a line several named characters say together has this choice")
    if mode not in SIMULTANEOUS_MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {', '.join(SIMULTANEOUS_MODES)}")
    result = deepcopy(segment)
    result.pop("simultaneous_mode", None)
    if mode != "unison":
        result["simultaneous_mode"] = mode
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
