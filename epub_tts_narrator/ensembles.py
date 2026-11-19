"""Collective voices: one cast role heard as several voices (``COLLECTIVE_VOICES.md``).

Identity and sound are kept apart:

* ``cast.characters[i].group`` is analysis's (or a person's) statement that a
  character is a group, with the source's words for its size. It is a
  proposal only: nothing is rendered differently because of it.
* ``cast.voices[i].ensemble`` is a person's casting choice: the group's own
  voice (its recording is the *lead*, the first lane) plus further member
  voices, a default mode and a mix recipe. Only a person writes it.
* ``cast.voices[i].member_of`` marks a *pool voice*: a voice designed for one
  group, belonging to no character.
* ``segments[i].ensemble`` is a per-line override backed by the line's own tag
  (or a person): a mode, a member subset, a lane count or a pool slot.

This module holds the vocabulary and the format rules; the renderer resolves
lanes from it. Stdlib only.
"""

from __future__ import annotations

from typing import Any

MODES = ("single", "unison", "alternate", "pool", "staggered", "layered")
# Modes heard as several takes at once; the others speak one voice per line.
LANE_MODES = frozenset({"unison", "staggered", "layered"})
SCALES = ("pair", "few", "several", "many", "crowd")
SCALE_VOICES = {"pair": 2, "few": 3, "several": 4, "many": 6, "crowd": 8}
MAX_VOICES = 8
# The renderer's closed recipe set (``renderer.lane_mix.RECIPES``).
RECIPES = ("tight", "loose", "crowd", "swell", "babble", "echo", "lead")
DEFAULT_RECIPES = {"unison": "loose", "staggered": "babble", "layered": "lead"}
# ``simultaneous_speakers`` of named characters: a small fixed choice.
SIMULTANEOUS_MODES = {"unison": "tight", "staggered": "babble", "layered": "lead"}
GROUP_BASES = ("analysis", "user")
LINE_BASES = ("explicit", "context", "user")


def proposed_voices(size: int | None, scale: str | None) -> int | None:
    """How many voices to cast for a group, from the source's own number or words."""
    if type(size) is int and size >= 2:
        return min(size, MAX_VOICES)
    return SCALE_VOICES.get(scale) if isinstance(scale, str) else None


def recipe_for(mode: str, recipe: str | None = None) -> str:
    """The mix recipe a lane mode uses: the chosen one, else the mode's default."""
    return recipe or DEFAULT_RECIPES[mode]


def ensemble(voice: dict[str, Any] | None) -> dict[str, Any] | None:
    value = voice.get("ensemble") if isinstance(voice, dict) else None
    return value if isinstance(value, dict) else None


def lanes_of(voice: dict[str, Any]) -> list[str]:
    """The ensemble's voices in lane order: the lead (the voice itself), then its members."""
    value = ensemble(voice) or {}
    return [voice["id"], *value.get("members", [])]


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def validate_group(check, character: dict[str, Any], path: str, characters: dict[str, Any]) -> None:
    """``cast.characters[i].group``. ``check`` is the contract validator."""
    group = check.object(character.get("group"), path)
    size = group.get("size")
    if size is not None and (type(size) is not int or size < 2):
        check.error(f"{path}.size", "must be an integer of at least 2, or null")
    scale = group.get("scale")
    if scale is not None:
        check.choice(scale, set(SCALES), f"{path}.scale")
    members = check.strings(group.get("members", []), f"{path}.members")
    for j, member in enumerate(members):
        at = f"{path}.members[{j}]"
        if check.reference(member, characters, at):
            if member == character.get("id"):
                check.error(at, "a group cannot be its own member")
            elif "group" in characters[member]:
                check.error(at, "a member must be an individual character, not another group")
    if len(set(members)) != len(members):
        check.error(f"{path}.members", "must not repeat a character")
    voices = group.get("proposed_voices")
    if voices is not None and (type(voices) is not int or not 2 <= voices <= MAX_VOICES):
        check.error(f"{path}.proposed_voices", f"must be an integer from 2 to {MAX_VOICES}, or null")
    if "proposed_mode" in group:
        check.choice(group["proposed_mode"], set(MODES), f"{path}.proposed_mode")
    check.choice(group.get("basis"), set(GROUP_BASES), f"{path}.basis")
    # A group stated by analysis needs the source's own words; names alone never make one.
    if group.get("basis") != "user" or "evidence" in group:
        check.evidence(group.get("evidence"), f"{path}.evidence", required=group.get("basis") == "analysis")


def validate_voices(check, voices: list[dict[str, Any]], voice_index: dict[str, Any],
                    character_voices: dict[str, str], narration_voices: set[str]) -> None:
    """``cast.voices[i].ensemble`` and ``member_of``."""
    for i, voice in enumerate(voices):
        path = f"cast.voices[{i}]"
        if "ensemble" in voice:
            at = f"{path}.ensemble"
            value = check.object(voice["ensemble"], at)
            mode_ok = check.choice(value.get("mode"), set(MODES), f"{at}.mode")
            members = check.strings(value.get("members"), f"{at}.members")
            if voice.get("id") not in character_voices:
                check.error(at, "only a character's voice can lead an ensemble")
            if voice.get("id") in narration_voices:
                check.error(at, "the narration voice cannot be an ensemble")
            if "derived_from" in voice:
                check.error(at, "a derived voice cannot lead an ensemble")
            if len(set(members)) != len(members) or voice.get("id") in members:
                check.error(f"{at}.members", "must be distinct voices other than the lead")
            if len(members) + 1 > MAX_VOICES:
                check.error(f"{at}.members", f"an ensemble has at most {MAX_VOICES} voices, its lead included")
            if mode_ok and value["mode"] != "single" and not members:
                check.error(f"{at}.members", "only a declined ensemble (mode single) has no further members")
            for j, member in enumerate(members):
                if check.reference(member, voice_index, f"{at}.members[{j}]") and "ensemble" in voice_index[member]:
                    check.error(f"{at}.members[{j}]", "an ensemble member cannot itself be an ensemble")
            if "mix" in value:
                check.choice(value["mix"], set(RECIPES), f"{at}.mix")
            if value.get("basis") != "user":
                check.error(f"{at}.basis", "must be user: only a person sets up an ensemble")
        if "member_of" in voice:
            at = f"{path}.member_of"
            owner = voice["member_of"]
            if check.reference(owner, voice_index, at):
                if voice.get("id") not in (ensemble(voice_index[owner]) or {}).get("members", []):
                    check.error(at, "must name an ensemble voice that lists this voice as a member")
            if voice.get("id") in character_voices:
                check.error(at, "a pool voice belongs to no character")
            if "ensemble" in voice:
                check.error(at, "a pool voice cannot lead an ensemble")


def validate_segment(check, segment: dict[str, Any], path: str, voice_index: dict[str, Any],
                     group_voices: set[str]) -> None:
    """``segments[i].ensemble`` and ``simultaneous_mode``."""
    if "simultaneous_mode" in segment:
        check.choice(segment["simultaneous_mode"], set(SIMULTANEOUS_MODES), f"{path}.simultaneous_mode")
        if "simultaneous_speakers" not in segment:
            check.error(f"{path}.simultaneous_mode", "applies only to simultaneous_speakers rows")
    if "ensemble" not in segment:
        return
    at = f"{path}.ensemble"
    value = check.object(segment["ensemble"], at)
    voice_id = segment.get("voice_id")
    lead = voice_index.get(voice_id) if isinstance(voice_id, str) else None
    chosen = ensemble(lead)
    if voice_id not in group_voices and chosen is None:
        check.error(at, "applies only to a group's lines")
    if segment.get("kind") not in {"dialogue", "thought", "letter"}:
        check.error(f"{at}", "applies only to spoken lines")
    if not any(key in value for key in ("mode", "members", "lanes", "slot")):
        check.error(at, "must set at least one of mode, members, lanes or slot")
    if "mode" in value:
        check.choice(value["mode"], set(MODES), f"{at}.mode")
    if "members" in value:
        members = check.strings(value["members"], f"{at}.members")
        allowed = set(lanes_of(lead)) if chosen is not None else set()
        if chosen is None:
            check.error(f"{at}.members", "needs the group's ensemble: members are voices a person cast")
        elif not members or len(set(members)) != len(members) or not set(members) <= allowed:
            check.error(f"{at}.members", "must be distinct voices of this group's ensemble")
    if "lanes" in value and (type(value["lanes"]) is not int or not 1 <= value["lanes"] <= MAX_VOICES):
        check.error(f"{at}.lanes", f"must be an integer from 1 to {MAX_VOICES}")
    if "slot" in value and not _positive(value["slot"]):
        check.error(f"{at}.slot", "must be a positive integer")
    check.choice(value.get("basis"), set(LINE_BASES), f"{at}.basis")
    if value.get("basis") != "user" or "evidence" in value:
        check.evidence(value.get("evidence"), f"{at}.evidence", required=value.get("basis") != "user")
