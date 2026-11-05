"""Explicit complete-record cast scoping for large identity requests.

``scoped_cast`` selects the cast records relevant to supplied source units and
requested characters, copying every supplied record whole. The paired scope
object discloses exactly what the request supplies and omits, so bounded
requests never imply a trimmed or complete cast silently.
"""

from __future__ import annotations

from copy import deepcopy
import re


CAST_SCOPE_GUIDANCE = """
CAST SCOPE DISCLOSURE
The cast_scope object states exactly which cast records this request supplies.
complete=true means original_cast is the complete current cast. complete=false
means original_cast is a complete-record subset selected from a larger cast:
every supplied character, role, and voice record is entire and untrimmed, and
full_cast_fingerprint identifies the complete cast it was selected from.
omitted_counts reports how many records of each kind exist in that complete
cast but are not supplied here. Never invent, rename, or cite omitted records,
and restrict every finding to supplied character IDs. Other source regions are
screened separately and their findings are combined afterward; the
bounded name-collision screen separately nominates disjoint same-name and
similar-name pairs across the complete cast; it does not certify every pair.
"""


def scoped_cast(source, cast, units, *, character_ids=(), identity_case=False):
    """Return ``(view, scope)``: complete relevant cast records plus disclosure.

    The view never trims a record. It contains, copied whole: characters with
    stored evidence inside the supplied units, characters whose name or alias
    occurs in unit text, explicitly requested characters, mandatory narration
    roles and the characters that own them, and every role and voice record
    referenced by the included characters and roles. Identity-case views omit
    unrelated narration roles and their owners. ``scope`` reports
    ``complete``, ``full_cast_fingerprint``, ``included_ids``,
    ``omitted_counts``, and the ``selection`` method.
    """
    from .pipeline import fingerprint
    known = {row["id"] for row in cast["characters"]}
    requested = set(character_ids)
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError("cast_scope requested unknown character ID(s)")
    intervals = {}
    for unit in units:
        intervals.setdefault(unit["block_id"], []).append((unit["start"], unit["start"] + len(unit["text"])))
    visible = {}
    for block_id, spans in intervals.items():
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        visible[block_id] = merged
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    unit_ids = {unit["unit_id"] for unit in units}
    text = "\n".join(unit["text"] for unit in units)
    folded_text = text.casefold()
    included = set(requested)
    narration_ids = set()
    for role in cast["roles"]:
        if role["kind"] != "narration":
            continue
        owner = role["character_id"]
        scope = role.get("source_scope")
        applies = owner in requested
        if not identity_case:
            if role["id"] == "narrator_main" or not isinstance(scope, dict):
                applies = True
            else:
                scoped_ids, scoped_intervals = scope.get("unit_ids", []), scope.get("intervals", [])
                if (not isinstance(scoped_ids, list) or any(not isinstance(value, str) for value in scoped_ids)
                        or not isinstance(scoped_intervals, list)):
                    applies = True
                elif not scoped_ids and not scoped_intervals:
                    applies = True
                elif isinstance(scoped_ids, list) and unit_ids.intersection(scoped_ids):
                    applies = True
                elif isinstance(scoped_intervals, list):
                    applies = applies or any(
                        isinstance(span, dict) and type(span.get("start")) is int and type(span.get("end")) is int
                        and any(start < span["end"] and span["start"] < end
                                for start, end in visible.get(span.get("block_id"), []))
                        for span in scoped_intervals)
        if applies:
            narration_ids.add(role["id"])
            if owner is not None:
                included.add(owner)
    for row in cast["characters"]:
        if row["id"] in included:
            continue
        if any(any(blocks[item["block_id"]].find(item["quote"], start, end) >= 0
                   for start, end in visible.get(item["block_id"], [])) for item in row["evidence"]):
            included.add(row["id"])
            continue
        names = [value.strip() for value in [row.get("name"), *row.get("aliases", [])]
                 if isinstance(value, str) and value.strip()]
        if any((value.casefold() in folded_text if not value.isascii() else
                re.search(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE))
               for value in names):
            included.add(row["id"])
    characters = [deepcopy(row) for row in cast["characters"] if row["id"] in included]
    included_ids = {row["id"] for row in characters}
    roles = [deepcopy(role) for role in cast["roles"]
             if (role["kind"] == "narration" and role["id"] in narration_ids)
             or (role["kind"] != "narration" and role["character_id"] in included_ids)]
    referenced_voices = {row.get("voice_id") for row in characters} | {role.get("voice_id") for role in roles}
    voices = [deepcopy(voice) for voice in cast["voices"] if voice.get("id") in referenced_voices]
    view = {"characters": characters, "roles": roles, "voices": voices}
    scope = {
        "complete": all(len(view[key]) == len(cast[key]) for key in view),
        "full_cast_fingerprint": fingerprint(cast),
        "included_ids": sorted(included_ids),
        "omitted_counts": {"characters": len(cast["characters"]) - len(characters),
                           "roles": len(cast["roles"]) - len(roles),
                           "voices": len(cast["voices"]) - len(voices)},
        "selection": ("explicit identity-case participants and their complete associated records" if identity_case else
                      "source-visible evidence and names, requested characters, and applicable narration roles; "
                      "every supplied character, role, and voice record is copied whole"),
    }
    return view, scope
