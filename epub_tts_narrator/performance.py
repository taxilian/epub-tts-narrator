"""Source-derived delivery cues, independent of speakers and TTS providers."""

from copy import deepcopy


BRACKET_DIRECTION = (
    "Give this bracketed text a slight, audible change in delivery while keeping "
    "the surrounding sentence natural. Keep the selected speaker's voice."
)


def bracket_ranges(source):
    """Find balanced outer square brackets, including pairs across paragraphs.

    Nested pairs form one delivery group. An unmatched bracket is reported, never
    allowed to consume the rest of a chapter or cross a chapter boundary.
    """
    ranges = {block["id"]: [] for block in source["blocks"]}
    warnings, chapter, pending, stack = [], None, [], []

    def finish():
        for block_id, offset in stack:
            warnings.append({"block_id": block_id, "offset": offset,
                             "reason": "Unmatched opening square bracket; left literal."})

    for block in source["blocks"]:
        if chapter != block["chapter_id"]:
            finish()
            chapter, pending, stack = block["chapter_id"], [], []
        for offset, char in enumerate(block["text"]):
            if char == "[":
                if not stack:
                    pending = []
                stack.append((block["id"], offset))
            if stack:
                if not pending or pending[-1]["block_id"] != block["id"]:
                    pending.append({"block_id": block["id"], "start": offset, "end": offset + 1})
                else:
                    pending[-1]["end"] = offset + 1
            if char == "]":
                if not stack:
                    warnings.append({"block_id": block["id"], "offset": offset,
                                     "reason": "Unmatched closing square bracket; left literal."})
                    continue
                opening = stack.pop()
                if not stack:
                    group_id = f"bracket:{opening[0]}:{opening[1]}"
                    for index, span in enumerate(pending):
                        ranges[span["block_id"]].append(dict(
                            span, group_id=group_id, first=index == 0,
                            last=index == len(pending) - 1))
                    pending = []
    finish()
    return ranges, warnings


def apply_bracket_delivery(document, direction=BRACKET_DIRECTION):
    """Split exact source segments at brackets without changing voice attribution."""
    if not isinstance(direction, str) or not direction.strip():
        raise ValueError("Bracket delivery direction must be a nonempty string")
    result = deepcopy(document)
    ranges, warnings = bracket_ranges(result["source"])
    segments = []
    for original in result["segments"]:
        spans = ranges[original["block_id"]]
        boundaries = {original["start"], original["end"]}
        for span in spans:
            boundaries.update(value for value in (span["start"], span["end"])
                              if original["start"] < value < original["end"])
        points = sorted(boundaries)
        for start, end in zip(points, points[1:]):
            piece = deepcopy(original)
            piece.update(id=f"{piece['block_id']}:{start}:{end}", start=start, end=end,
                         text=original["text"][start-original["start"]:end-original["start"]])
            piece.pop("performance", None)
            piece.pop("spoken_text", None)
            if start != original["start"] or end != original["end"]:
                # Edits are relative to the original segment; a new boundary
                # needs a fresh source-bound normalization pass.
                piece.pop("speech_normalization", None)
            enclosing = next((span for span in spans if span["start"] <= start < end <= span["end"]), None)
            if enclosing:
                piece["performance"] = {
                    "kind": "bracketed", "group_id": enclosing["group_id"],
                    "direction": direction, "delimiters": "omit",
                    "enter": enclosing["first"] and start == enclosing["start"],
                    "exit": enclosing["last"] and end == enclosing["end"],
                }
                piece["spoken_text"] = piece["text"].translate(str.maketrans("", "", "[]"))
            if "speech_normalization" in piece:
                from .spoken_text import build_spoken
                piece.pop("spoken_text", None)
                piece["spoken_text"] = build_spoken(piece)[0]
            segments.append(piece)
    result["segments"] = segments
    result.setdefault("provenance", {})["bracket_delivery"] = {
        "version": 1, "basis": "source_delimiters", "direction": direction,
        "groups": len({span["group_id"] for values in ranges.values() for span in values}),
        "warnings": warnings,
    }
    # Idempotent even when run after a source-bound human correction.
    previous_warnings = {issue["id"]: issue for issue in result.get("issues", [])
                         if isinstance(issue.get("id"), str)
                         and issue["id"].startswith("bracket_unmatched:")}
    result["issues"] = [issue for issue in result.get("issues", [])
                        if issue.get("id") not in previous_warnings]
    for warning in warnings:
        issue_id = f"bracket_unmatched:{warning['block_id']}:{warning['offset']}"
        previous = previous_warnings.get(issue_id, {})
        # Reapplying source-derived cues must not reopen an explicit resolution
        # of the same warning. Source edits are separately fingerprint-bound.
        resolved = (previous.get("resolved") is True
                    and previous.get("target") == warning["block_id"]
                    and previous.get("reason") == warning["reason"])
        result["issues"].append({
            "id": issue_id,
            "kind": "delivery", "target": warning["block_id"],
            "reason": warning["reason"], "resolved": resolved,
        })
    return result
