"""Speech the book sets in italics instead of quotation marks (thoughts, telepathy, words that appear).

No model calls. Some books print unquoted speech in italics and its tag in
roman type: *Awaken,* the words said as they appeared before my eyes. *For you
are the hope of the night sky.* The model marks unquoted speech by its first
and last words, so one span swallows the tag between the two halves and the
character reads the narration. The italics show exactly where the speech is.

``split_span`` cuts such a span at the roman text between its italic runs when
that text reads as narration (a speech or thought verb, or three or more
words), so the narrator reads the tag. A short roman word inside italic
speech is the book's stress ("I told you *never* to come", set the other way
round) and stays in the line. A span whose first words are not italic is not
italic-set speech and is left whole.

``italic_misreads`` finds saved segments the same rules contradict: a line that
includes a tag between its italic runs, or italic words the narrator reads
right beside a speech tag ("*Don't-know,* he said to my mind").
"""

from __future__ import annotations

import re

from .quote_spans import SPEECH_VERBS, dialogue_style, find_tag, quote_spans, spans_by_block

_WORD = re.compile(r"[^\W_](?:[\w'’-]*[^\W_])?")
# Speech-tag verbs for unquoted speech: the quoted-speech verbs plus thinking, writing and speaking.
_TAG_VERB = re.compile(rf"\b(?:{SPEECH_VERBS}|thought|thinks|wrote|writes|spoke|speaks|accused|accuses|"
                       r"begged|begs|pleaded|pleads|commanded|commands|projected|projects|sent|sends)\b",
                       re.IGNORECASE)
_SPEECH_TAG = re.compile(rf"\b(?:{SPEECH_VERBS}|spoke|speaks)\b", re.IGNORECASE)
NARRATION_WORDS = 3
_QUOTES = "\"“”‘’'"


def italic_runs(block, start=0, end=None):
    """The book's italic runs (``inline_spans`` emphasis) inside ``block["text"][start:end]``, in order."""
    end = len(block["text"]) if end is None else end
    runs = sorted((max(start, span["start"]), min(end, span["end"])) for span in block.get("inline_spans", [])
                  if span.get("kind") == "emphasis" and span["start"] < end and span["end"] > start)
    merged = []
    for low, high in runs:
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def is_narration(text):
    """Roman text between italic speech that reads as narration: a speech or thought verb, or a few words."""
    return bool(_TAG_VERB.search(text)) or len(_WORD.findall(text)) >= NARRATION_WORDS


def _only_marks(text):
    return not _WORD.search(text) and not any(mark in text for mark in _QUOTES)


def split_span(block, start, end):
    """``[(start, end)]``: the speech pieces of an unquoted span, cut at the narration between italic runs."""
    text = block["text"]
    runs = italic_runs(block, start, end)
    if not runs or not _only_marks(text[start:runs[0][0]]):
        return [(start, end)]
    pieces = [list(runs[0])]
    for low, high in runs[1:]:
        if is_narration(text[pieces[-1][1]:low]):
            pieces.append([low, high])
        else:
            pieces[-1][1] = high
    tail = text[pieces[-1][1]:end]
    if not is_narration(tail):
        pieces[-1][1] = end
    pieces[0][0] = start
    result = []
    for low, high in pieces:
        # Keep the speech's own closing punctuation; never start or end a piece on a space.
        while low < high and text[low].isspace():
            low += 1
        while high > low and text[high - 1].isspace():
            high -= 1
        if low < high:
            result.append((low, high))
    return result or [(start, end)]


def italic_misreads(document, *, skip=lambda rows: False):
    """``{block_id: reason}`` for paragraphs whose italic speech the saved split contradicts."""
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    rows = {}
    for row in document["segments"]:
        rows.setdefault(row["block_id"], []).append(row)
    # Quoted speech is the quotation marks' business (``split_check``): italics there are stress.
    quoted = spans_by_block(quote_spans(document["source"], dialogue_style(document["source"])))
    result = {}
    for block in document["source"]["blocks"]:
        saved = rows.get(block["id"], [])
        if not saved or not any(span.get("kind") == "emphasis" for span in block.get("inline_spans", [])):
            continue
        if skip(saved):
            continue
        text = block["text"]
        quotes = [(span["start"], span["end"]) for span in quoted.get(block["id"], [])]
        for row in saved:
            role = roles.get(row.get("role_id"), {})
            if any(low < row["end"] and high > row["start"] for low, high in quotes):
                continue
            if row["kind"] in {"dialogue", "thought", "letter"} and (role.get("kind") == "dialogue"
                                                                     or "simultaneous_speakers" in row):
                if split_span(block, row["start"], row["end"]) != [(row["start"], row["end"])] and any(
                        is_narration(text[low:high]) for low, high in _gaps(block, row["start"], row["end"])):
                    result[block["id"]] = "a character's italic speech includes the narration between its words"
                    break
            elif row["kind"] == "narration" and role.get("kind") == "narration":
                spoken = [(low, high) for low, high in italic_runs(block, row["start"], row["end"])
                          if len(_WORD.findall(text[low:high])) >= 1 and _beside_tag(text, low, high)]
                if spoken:
                    result[block["id"]] = "the narrator reads italic speech beside its speech tag"
                    break
    return result


def _gaps(block, start, end):
    runs = italic_runs(block, start, end)
    return [(runs[index][1], runs[index + 1][0]) for index in range(len(runs) - 1)] + (
        [(runs[-1][1], end)] if runs and runs[-1][1] < end else [])


def _beside_tag(text, low, high):
    """Italic words a speech tag right after or before them attributes ("*Hurry,* he said")."""
    return find_tag(text, low, high) is not None and bool(
        _SPEECH_TAG.search(text[high:high + 40]) or _SPEECH_TAG.search(text[max(0, low - 40):low]))
