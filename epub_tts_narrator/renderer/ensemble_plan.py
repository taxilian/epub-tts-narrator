"""Which voices speak each line of a collective voice (``COLLECTIVE_VOICES.md``).

A group's line names the group's role and voice, as any line does. When a
person has cast that voice as an ensemble (``cast.voices[i].ensemble``), the
plan resolves the line here into its performers:

* ``single``: the lead alone (today's behaviour);
* ``unison`` / ``staggered`` / ``layered``: several lanes, each a verified take
  of the whole line, mixed by a lane recipe (``lane_mix``);
* ``pool``: one member per anonymous individual. A line's ``slot`` names the
  individual within the chapter; a slot keeps its voice there, and a new slot
  takes the member with the highest rendezvous score not already speaking for
  another slot, so adding a member moves only the slots that now prefer it.
  A line with no slot is a new individual, never the previous line's voice;
* ``alternate``: members take turns through a run of the group's consecutive
  lines (another speaker's line ends the run).

Every resolution is deterministic from the book, the cast and the chapter, so
the same plan always names the same voices, and a changed membership moves
exactly the lines whose performers changed. Stdlib only.
"""

from __future__ import annotations

from copy import deepcopy

from ..ensembles import LANE_MODES, MAX_VOICES, ensemble, lanes_of, recipe_for
from .common import digest

VERSION = 1
SPOKEN = frozenset({"dialogue", "thought", "letter"})


def _score(*parts):
    return digest(list(parts))


def _ranked(voice_ids, *key):
    """Members in a stable order of preference for this key (rendezvous hashing)."""
    return sorted(voice_ids, key=lambda voice_id: _score(*key, voice_id), reverse=True)


def _lane_subset(lanes, count, *key):
    """``count`` lanes, the lead (first) always kept, the rest by stable preference."""
    lead, others = lanes[0], lanes[1:]
    return [lead, *_ranked(others, *key)[:max(0, count - 1)]]


def resolve_chapter(rows, voices, chapter_id):
    """``{segment_id: resolution}`` for the ensemble lines of one chapter's rows.

    ``voices`` maps voice ID to cast voice. A resolution is
    ``{"version", "voice_id", "mode", "recipe", "performers": [{role_id, voice_id}], "slot"?}``;
    ``recipe`` is set only for modes heard as several lanes at once.
    """
    resolved = {}
    slots = {}            # ensemble voice -> {slot: voice_id}
    previous = {}         # ensemble voice -> voice_id of its previous pool/alternate line
    turns = {}            # ensemble voice -> position in the current alternate run
    last_speaker = None
    automatic = {}        # ensemble voice -> next automatic slot number
    for row in rows:
        if row.get("kind") not in SPOKEN or "simultaneous_speakers" in row:
            if row.get("kind") in SPOKEN:
                last_speaker = None
            continue
        voice_id = row.get("voice_id")
        lead = voices.get(voice_id) if isinstance(voice_id, str) else None
        chosen = ensemble(lead)
        if chosen is None:
            last_speaker = voice_id
            continue
        override = row.get("ensemble") if isinstance(row.get("ensemble"), dict) else {}
        mode = override.get("mode", chosen["mode"])
        lanes = list(override.get("members") or lanes_of(lead))[:MAX_VOICES]
        role_id = row["role_id"]
        resolution = {"version": VERSION, "voice_id": voice_id, "mode": mode, "recipe": None}
        if mode == "single":
            performers = [voice_id]
        elif mode in LANE_MODES:
            count = override.get("lanes")
            performers = _lane_subset(lanes, count, voice_id, row["id"]) if count else lanes
            if len(performers) > 1:
                mix = chosen.get("mix") if mode == chosen["mode"] else None
                resolution["recipe"] = recipe_for(mode, mix)
        elif mode == "pool":
            assigned = slots.setdefault(voice_id, {})
            slot = override.get("slot")
            if slot is None:
                # No slot from the source: this line is a new individual.
                automatic[voice_id] = automatic.get(voice_id, 0) + 1
                slot = f"line-{automatic[voice_id]}"
            if len(lanes) == 1:
                assigned[slot] = lanes[0]
            elif slot not in assigned or assigned[slot] not in lanes:
                taken = set(assigned.values())
                ranked = _ranked(lanes, voice_id, chapter_id, str(slot))
                free = [member for member in ranked if member not in taken]
                choice = free[0] if free else next(
                    (member for member in ranked if member != previous.get(voice_id)), ranked[0])
                if isinstance(slot, str) and choice == previous.get(voice_id) and len(ranked) > 1:
                    choice = next(member for member in ranked if member != previous[voice_id])
                assigned[slot] = choice
            performers = [assigned[slot]]
            if not isinstance(slot, str):
                resolution["slot"] = slot
        elif mode == "alternate":
            if last_speaker != voice_id:
                turns[voice_id] = 0
            if override.get("members"):
                # A person pinned this line's member; the turns continue from it.
                performers = [override["members"][0]]
                order = lanes_of(lead)
                turns[voice_id] = order.index(performers[0]) if performers[0] in order else 0
            else:
                order = lanes
                performers = [order[turns.get(voice_id, 0) % len(order)]]
            turns[voice_id] = turns.get(voice_id, 0) + 1
        else:
            raise ValueError(f"Unknown ensemble mode {mode!r} for segment {row['id']}")
        previous[voice_id] = performers[0] if len(performers) == 1 else previous.get(voice_id)
        last_speaker = voice_id
        resolution["performers"] = [{"role_id": role_id, "voice_id": member} for member in performers]
        resolved[row["id"]] = resolution
    return resolved


def lane_recipe(chunk):
    """The mix recipe of a line heard as several lanes (group or ``simultaneous_speakers``)."""
    from ..ensembles import SIMULTANEOUS_MODES
    if isinstance(chunk.get("ensemble"), dict):
        return chunk["ensemble"].get("recipe")
    row = chunk.get("segment") or {}
    if "simultaneous_speakers" in row:
        return SIMULTANEOUS_MODES[row.get("simultaneous_mode", "unison")]
    return None


def validate_resolution(chunk):
    """Shape checks a chapter service makes without the cast (it trusts no client plan)."""
    value = chunk.get("ensemble")
    row = chunk.get("segment") or {}
    from ..ensembles import MODES, RECIPES
    if (not isinstance(value, dict) or value.get("version") != VERSION
            or value.get("voice_id") != row.get("voice_id") or value.get("mode") not in MODES
            or "simultaneous_speakers" in row):
        raise ValueError("Invalid collective voice resolution")
    performers = value.get("performers")
    if (not isinstance(performers, list) or not 1 <= len(performers) <= MAX_VOICES
            or any(not isinstance(item, dict) or set(item) != {"role_id", "voice_id"}
                   or item["role_id"] != row.get("role_id")
                   or not isinstance(item["voice_id"], str) or not item["voice_id"] for item in performers)
            or len({item["voice_id"] for item in performers}) != len(performers)):
        raise ValueError("Invalid collective voice performers")
    several = len(performers) > 1
    if several != (value.get("recipe") is not None) or several and (
            value["mode"] not in LANE_MODES or value["recipe"] not in RECIPES):
        raise ValueError("Invalid collective voice mix recipe")
    return deepcopy(performers)
