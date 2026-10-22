"""Build an approved spoken form while retaining the untouched source text.

Replacement spans map to whole original source spans. They do not pretend that
different-length spellings have a character-for-character correspondence.
All offsets use Python/Unicode code points, as the analysis contract does.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping


MAX_DELIVERY_CHARACTERS = 1200


def _plain(value, label, *, maximum=None):
    if (not isinstance(value, str) or not value.strip()
            or any(char in value for char in "[]<>")
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for char in value)
            or maximum is not None and len(value) > maximum):
        raise ValueError(f"{label} must be nonempty plain text without control syntax")


def validate_normalization(row):
    """Validate explicit edits without requiring a previously derived spoken form.

    The analysis stage owns semantic approval and its audit. This shared validator
    enforces the source/offset contract, including at remote worker boundaries.
    """
    if not isinstance(row, dict):
        raise ValueError("A spoken passage requires a source segment")
    text, start, end = row.get("text"), row.get("start"), row.get("end")
    if (not isinstance(text, str) or type(start) is not int or type(end) is not int
            or start < 0 or end != start + len(text)):
        raise ValueError("Invalid source text or offsets for spoken normalization")
    if "speech_normalization" not in row:
        return
    record = row["speech_normalization"]
    if (not isinstance(record, dict) or type(record.get("version")) is not int
            or record["version"] != 1 or not isinstance(record.get("edits"), list)):
        raise ValueError("Invalid speech normalization version or edits")
    cursor = 0
    for edit in record["edits"]:
        if not isinstance(edit, dict) or set(edit) != {
                "start", "end", "original", "replacement", "direction", "reason"}:
            raise ValueError("Speech edits need source bounds, original, replacement, direction and reason")
        a, b = edit["start"], edit["end"]
        if (type(a) is not int or type(b) is not int or not cursor <= a < b <= len(text)
                or not isinstance(edit["original"], str) or edit["original"] != text[a:b]):
            raise ValueError("Speech edits must be ordered, nonoverlapping exact source substrings")
        _plain(edit["replacement"], "Speech replacement")
        _plain(edit["direction"], "Speech delivery direction", maximum=240)
        _plain(edit["reason"], "Speech normalization reason", maximum=1000)
        if edit["replacement"] == edit["original"]:
            raise ValueError("A speech normalization edit must change its source spelling")
        cursor = b


def effective_delivery(row):
    """Combine existing delivery and approved spelling styles without dropping any.

    The source row is never mutated, so a remote worker can derive the same
    direction again without appending duplicate instructions. Overlong composed
    instructions fail explicitly instead of silently losing a requested style.
    """
    if not isinstance(row, Mapping):
        raise ValueError("A delivery direction requires a source segment")
    delivery = row.get("delivery", {})
    if not isinstance(delivery, Mapping) or not isinstance(delivery.get("direction", ""), str):
        raise ValueError("Speech delivery must contain a direction string")
    candidates = [delivery.get("direction", "")]
    if "speech_normalization" in row:
        validate_normalization(row)
        candidates.extend(edit["direction"] for edit in row["speech_normalization"]["edits"])
    directions, seen = [], set()
    for direction in candidates:
        value = direction.strip()
        if not value:
            continue
        _plain(value, "Speech delivery direction")
        identity = " ".join(value.casefold().split())
        if identity not in seen:
            directions.append(value)
            seen.add(identity)
    result = "; ".join(directions)
    if len(result) > MAX_DELIVERY_CHARACTERS:
        raise ValueError(f"Combined speech delivery exceeds {MAX_DELIVERY_CHARACTERS} characters; "
                         "shorten the saved directions without discarding the intended styles")
    return result


def build_spoken(row):
    """Return approved speech and spans linking it to canonical source offsets.

    If ``spoken_text`` is present it must equal the deterministic result. Producers
    may remove a stale spoken_text first, derive the new text, and then save it.
    Legacy bracket-only rows retain their existing behavior.
    """
    validate_normalization(row)
    text = row["text"]
    edits = row.get("speech_normalization", {}).get("edits", [])
    performance = row.get("performance", {})
    if not isinstance(performance, dict):
        raise ValueError("Invalid speech performance metadata")
    bracketed = (performance.get("kind") == "bracketed"
                 or ("spoken_text" in row and row["spoken_text"] != text
                     and row["spoken_text"] == text.translate(str.maketrans("", "", "[]"))))
    pieces, spans, spoken_cursor = [], [], 0

    def append(value, start, end, kind):
        nonlocal spoken_cursor
        if not value:
            return
        pieces.append(value)
        spans.append({"spoken_start": spoken_cursor, "spoken_end": spoken_cursor + len(value),
                      "source_start": row["start"] + start, "source_end": row["start"] + end,
                      "kind": kind})
        spoken_cursor += len(value)

    def original(start, end):
        cursor = start
        for index in range(start, end):
            if bracketed and text[index] in "[]":
                append(text[cursor:index], cursor, index, "source")
                cursor = index + 1
        append(text[cursor:end], cursor, end, "source")

    cursor = 0
    for edit in edits:
        original(cursor, edit["start"])
        append(edit["replacement"], edit["start"], edit["end"], "replacement")
        cursor = edit["end"]
    original(cursor, len(text))
    spoken = "".join(pieces)
    if "spoken_text" in row and row["spoken_text"] != spoken:
        raise ValueError("spoken_text differs from its approved source transformations")
    return spoken, spans


def spoken_source_ranges(row, start, end):
    """Project a spoken slice onto source ranges without inventing fine alignment.

    An overlapping replacement always refers to its full original substring, even
    when a rendering chunk cuts inside the replacement. Unchanged text maps exactly.
    """
    spoken, spans = build_spoken(row)
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(spoken):
        raise ValueError("Invalid spoken slice for source mapping")
    result = []
    for span in spans:
        a, b = max(start, span["spoken_start"]), min(end, span["spoken_end"])
        if a >= b:
            continue
        if span["kind"] == "source":
            source_start = span["source_start"] + a - span["spoken_start"]
            source_end = span["source_start"] + b - span["spoken_start"]
        else:
            source_start, source_end = span["source_start"], span["source_end"]
        if result and source_start <= result[-1]["end"]:
            result[-1]["end"] = max(result[-1]["end"], source_end)
        else:
            result.append({"start": source_start, "end": source_end})
    return result
