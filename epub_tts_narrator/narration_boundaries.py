"""Conservative, source-bound repair of speech tags absorbed into dialogue.

Only explicit speech tags outside balanced, unnested double quotations qualify.
This is deliberately not a general quotation-to-speaker parser: quoted thoughts,
letters, user assignments, and ambiguous typography retain their assignments.
"""

from collections import defaultdict
from copy import deepcopy
import re

from .contracts import readiness, validate_document
from .performance import apply_bracket_delivery


VERSION = 1
_VERBS = ("said", "asked", "replied", "answered", "whispered", "shouted", "yelled",
          "murmured", "muttered", "added", "continued", "called", "cried", "exclaimed",
          "demanded", "hissed", "growled", "snapped", "remarked", "admitted", "explained",
          "insisted", "suggested", "protested", "warned", "agreed", "grumbled", "sputtered",
          "stammered", "offered", "corrected", "responded")
_PRONOUNS = r"(?:he|she|they|it|I|we|you)"
_DESCRIPTIVE_SUBJECT = (r"(?:the|a|an|my|his|her|our|their)\s+"
                        r"(?:[\w’-]+\s+){0,4}"
                        r"(?:voice|man|woman|girl|boy|child|stranger|speaker)")


def _quote_layout(text):
    """Return complete outer pairs, or None; also flag an unfinished quotation."""
    curly = "“" in text or "”" in text
    straight = '"' in text
    if curly and straight:
        return None, text.count("“") > text.count("”")
    # Single quotation marks can delimit nested dialogue. Apostrophes within
    # words (don't, Mara's) do not. Skip instead of guessing their nesting scope.
    if "‘" in text or re.search(r"(?<!\w)'(?=\w)[^\n]*?'(?!\w)", text):
        return None, (text.count("“") > text.count("”") if curly else text.count('"') % 2 == 1)
    pairs, opening, invalid = [], None, False
    for index, character in enumerate(text):
        if character not in ('"', "“", "”"):
            continue
        if index and text[index - 1] == "\\":
            invalid = True
        if character == "“" or character == '"' and opening is None:
            if opening is not None:
                invalid = True
            opening = index
        elif opening is None:
            invalid = True
        else:
            pairs.append((opening, index + 1))
            opening = None
    return (None if invalid or opening is not None else pairs), opening is not None


def narration_tag_ranges(text, *, names=()):
    """Find exact leading, middle, and trailing narration-tag source ranges.

    Callers handling a whole book must also exclude blocks continuing an
    unfinished quotation from an earlier paragraph, and establish that the
    outer text is narration rather than unquoted reported speech. Ranges include
    boundary punctuation and whitespace, preserving every source character.
    """
    quotations, _ = _quote_layout(text)
    if not quotations:
        return []
    known_names = [re.escape(name) for name in names if isinstance(name, str) and name.strip()]
    subject = "(?:" + "|".join([_PRONOUNS, _DESCRIPTIVE_SUBJECT, *known_names]) + ")"
    verb = "(?:" + "|".join(_VERBS) + ")"
    adverb = r"(?:(?:[\w-]+ly|then|finally)\s+){0,2}"
    tag = re.compile(r"^[\s,;:.—–-]*(?:" + subject + r"\s+" + adverb + verb
                     + r"\b|" + verb + r"\s+" + subject + r"\b)", re.IGNORECASE)
    gaps, cursor = [], 0
    for start, end in quotations:
        gaps.append((cursor, start))
        cursor = end
    gaps.append((cursor, len(text)))
    return [(start, end) for start, end in gaps
            if 0 < len(text[start:end].strip()) <= 600 and tag.match(text[start:end])]


def _narrator_choices(block, originals, chapter_narrators, roles):
    local = {item["role_id"] for item in originals
             if item["kind"] == "narration" and roles.get(item["role_id"], {}).get("kind") == "narration"}
    choices = local or chapter_narrators[block["chapter_id"]]
    if not choices:
        choices = {key for key, role in roles.items() if role["kind"] == "narration"}
    return sorted(choices)


def repair_narration_boundaries(document, *, progress=lambda message: None):
    """Split clear speech tags from character dialogue without changing the cast.

The returned document is detached even on a no-op. A unique narrator already
used in the block/chapter is reused; conflicting scopes leave only the tag
unassigned with narration-role candidates. Explicit user decisions always win.
    """
    errors = validate_document(document)
    if errors:
        raise ValueError("Invalid input analysis: " + "; ".join(errors[:12]))
    result = deepcopy(document)
    roles = {item["id"]: item for item in document["cast"]["roles"]}
    names = [name for character in document["cast"]["characters"]
             for name in [character["name"], *character["aliases"]]]
    originals, chapter_narrators = defaultdict(list), defaultdict(set)
    blocks = {item["id"]: item for item in document["source"]["blocks"]}
    for item in document["segments"]:
        originals[item["block_id"]].append(item)
        if item["kind"] == "narration" and roles.get(item["role_id"], {}).get("kind") == "narration":
            chapter_narrators[blocks[item["block_id"]]["chapter_id"]].add(item["role_id"])
    replacements, repairs = {}, []
    chapter, continuing = None, False
    for block in document["source"]["blocks"]:
        if chapter != block["chapter_id"]:
            chapter, continuing = block["chapter_id"], False
        quotations, unfinished = _quote_layout(block["text"])
        was_continuing = continuing
        if any(character in block["text"] for character in ('"', "“", "”")):
            continuing = unfinished
        if was_continuing or quotations is None:
            continue
        first_quote = quotations[0][0] if quotations else len(block["text"])
        if block["text"][:first_quote].strip():
            # A quote inside unquoted dialogue may be a character recounting
            # another conversation: I said "Go," she said "Wait." Require an
            # independently assigned narration frame before repairing its tags.
            prefix = [item for item in originals[block["id"]] if item["start"] < first_quote
                      and block["text"][item["start"]:min(first_quote, item["end"])].strip()]
            if not prefix or any(item["kind"] != "narration"
                                 or roles.get(item["role_id"], {}).get("kind") != "narration"
                                 for item in prefix):
                continue
        ranges = narration_tag_ranges(block["text"], names=names)
        for start, end in ranges:
            overlapping = [item for item in originals[block["id"]]
                           if item["start"] < end and start < item["end"]]
            # Even partial explicit decisions and directly rendered writings or
            # thoughts protect the complete tag; never erase those policies.
            if any(item["attribution"]["basis"] == "user" or item["kind"] in {"thought", "letter"}
                   or "source_restoration" in item for item in overlapping):
                continue
            affected = [item for item in overlapping if item["kind"] == "dialogue"
                        and (roles.get(item["role_id"], {}).get("kind") == "dialogue"
                             or "simultaneous_speakers" in item)
                        and any(character.isalnum() for character in
                                block["text"][max(start, item["start"]):min(end, item["end"])])]
            if not affected:
                continue
            candidates = _narrator_choices(block, originals[block["id"]], chapter_narrators, roles)
            narrator = roles[candidates[0]] if len(candidates) == 1 else None
            repair = {"block_id": block["id"], "start": start, "end": end,
                      "original_segment_ids": [item["id"] for item in affected],
                      "role_id": narrator["id"] if narrator else None,
                      "candidates": [] if narrator else candidates}
            repairs.append(repair)
            for item in affected:
                replacements.setdefault(item["id"], []).append((start, end, narrator, candidates))
    if not repairs:
        return result
    segments = []
    for item in result["segments"]:
        changes = replacements.get(item["id"])
        if not changes:
            segments.append(item)
            continue
        points = {item["start"], item["end"]}
        for start, end, _, _ in changes:
            points.update((max(start, item["start"]), min(end, item["end"])))
        for start, end in zip(sorted(points), sorted(points)[1:]):
            piece = deepcopy(item)
            piece.update(id=f"{piece['block_id']}:{start}:{end}", start=start, end=end,
                         text=blocks[piece["block_id"]]["text"][start:end])
            piece.pop("performance", None)
            piece.pop("spoken_text", None)
            change = next((row for row in changes if row[0] <= start < end <= row[1]), None)
            if change:
                left, right, narrator, candidates = change
                piece.update(kind="narration", role_id=narrator["id"] if narrator else None,
                             voice_id=narrator["voice_id"] if narrator else None,
                             delivery={"direction": "", "evidence": []})
                piece.pop("simultaneous_speakers", None)
                piece["attribution"] = {
                    "basis": "context" if narrator else "unresolved",
                    "reason": ("Speech tag outside top-level quotations belongs to the narrator. "
                               + ("Using the narration role established in this block/chapter."
                                  if narrator else "The applicable narrator is ambiguous; choose a narration role.")),
                    "evidence": [{"block_id": piece["block_id"],
                                  "quote": blocks[piece["block_id"]]["text"][left:right].strip()}],
                    "candidates": [] if narrator else candidates, "review_required": narrator is None,
                }
            segments.append(piece)
    result["segments"] = segments
    previous = result["provenance"].get("narration_boundaries", {}).get("repairs", [])
    result["provenance"]["narration_boundaries"] = {"version": VERSION, "repairs": [*previous, *repairs]}
    if "bracket_delivery" in result["provenance"]:
        result = apply_bracket_delivery(result, result["provenance"]["bracket_delivery"]["direction"])
    result["readiness"] = readiness(result)
    errors = validate_document(result)
    if errors:
        raise ValueError("Invalid narration boundary repair: " + "; ".join(errors[:12]))
    progress(f"Separated {len(repairs)} narrative speech tag(s) from character dialogue.")
    return result
