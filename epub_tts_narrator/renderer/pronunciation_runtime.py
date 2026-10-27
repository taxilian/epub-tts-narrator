"""Bind confirmed book/series pronunciations without altering canonical speech."""

from copy import deepcopy

from ..performance_plan import text_sha256, validate_performance
from ..pronunciations import WORD_PATTERN as _WORD
from ..spoken_text import build_spoken


def confirmed_pronunciations(directory):
    """Snapshot approved choices for every exact source spelling in each group.

    Grouping/inheritance decides which case variants share a choice. Binding
    below still matches those complete source spellings exactly, preserving
    offsets and leaving similar names and possessives independent.
    """
    from ..pronunciations import load_pronunciations
    result = {}
    for entry in load_pronunciations(directory):
        if entry["status"] != "confirmed":
            continue
        for word in entry.get("variants", [entry["word"]]):
            if word in result and result[word]["pronunciation"] != entry["pronunciation"]:
                raise ValueError("Conflicting confirmed pronunciation choices for one exact source spelling")
            result[word] = entry
    return result


def _mapped_word(row, match, mapping):
    start, end = row["start"] + match.start(), row["start"] + match.end()
    pieces = []
    for span in mapping:
        left, right = max(start, span["source_start"]), min(end, span["source_end"])
        if left >= right:
            continue
        if span["kind"] == "source":
            pieces.append((span["spoken_start"] + left - span["source_start"],
                           span["spoken_start"] + right - span["source_start"]))
        elif start <= span["source_start"] < span["source_end"] <= end:
            pieces.append((span["spoken_start"], span["spoken_end"]))
        else:
            raise ValueError("Confirmed pronunciation overlaps a multiword speech-normalization edit; "
                             "resolve that saved edit before rendering")
    if not pieces or any(a[1] != b[0] for a, b in zip(pieces, pieces[1:])):
        raise ValueError("Confirmed pronunciation cannot map to one contiguous approved spoken word")
    return pieces[0][0], pieces[-1][1]


def merge_row_pronunciations(row, record, entries):
    """Merge source-word corrections with an optional exact spoken-text plan.

    An approved spelling edit may change the spelling at a target, but the
    correction still refers to its original source word. Ambiguous multiword
    edits fail explicitly; guessed offsets could change unrelated dialogue.
    """
    from .performance_runtime import _with_normalization_directions
    if not entries:
        return _with_normalization_directions(row, record) if record is not None else None
    matches = [(match, entries[match.group()]) for match in _WORD.finditer(row["text"])
               if match.group() in entries]
    if not matches:
        return _with_normalization_directions(row, record) if record is not None else None
    spoken, mapping = build_spoken(row)
    result = (deepcopy(record) if record is not None else
              {"version": 1, "text_sha256": text_sha256(spoken), "controls": []})
    result = validate_performance(result, spoken)
    suppressed = set()
    for match, entry in matches:
        start, end = _mapped_word(row, match, mapping)
        text = spoken[start:end]
        if not _WORD.fullmatch(text):
            raise ValueError("Confirmed pronunciation maps to multiple approved spoken words; "
                             "resolve that speech-normalization edit before rendering")
        if (start and spoken[start - 1].isalnum()
                or end < len(spoken) and spoken[end].isalnum()):
            raise ValueError("Confirmed pronunciation maps to only part of an approved spoken word; "
                             "resolve that speech-normalization edit before rendering")
        suppressed.update((edit["start"], edit["end"])
                          for edit in row.get("speech_normalization", {}).get("edits", [])
                          if match.start() <= edit["start"] < edit["end"] <= match.end())
        pronunciation = entry["pronunciation"]
        control = {"kind": "pronunciation", "start": start, "end": end, "text": text,
                   "alphabet": pronunciation["alphabet"],
                   "replacement" if pronunciation["alphabet"] == "respelling" else "phonemes": pronunciation["value"]}
        overlaps = [item for item in result["controls"] if item["kind"] == "pronunciation"
                    and start < item["end"] and item["start"] < end]
        if overlaps:
            if overlaps == [control]:
                continue
            raise ValueError("Confirmed pronunciation conflicts with a pronunciation in the performance plan; "
                             "make the saved correction and performance plan agree")
        result["controls"].append(control)
    # Confirmed corrections replace older normalization-derived lexical hints
    # on that exact source word. Explicit scene/plan delivery stays intact.
    return _with_normalization_directions(row, result, suppressed_normalizations=suppressed)
