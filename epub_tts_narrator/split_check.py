"""Find saved paragraphs whose speech split contradicts their quotation marks.

No model calls. A saved book keeps the split it was analyzed with, so a fix to
how quotation marks are read (``quote_spans``: mixed curly and straight marks,
a paragraph that opens inside speech) would never reach it. Rerunning
``analyze BOOK`` compares every paragraph's saved segments with the current
quotation spans: character dialogue with words outside every quotation (a
speech tag or action beat read in the character's voice), or narration with
words inside one (speech read by the narrator), marks the paragraph as
misread, and the book update reads it again with its neighbours
(``book_update.update_book(reread=...)``).

Only paragraphs that contain quotation marks in the book's dialogue style are
compared; unquoted speech is the model's call. Paragraphs with a person's
speaker choice, a quoted thought or letter, or a restored source span are
left alone. A paragraph read again at this ``VERSION`` is recorded in
``provenance.split_rereads`` with its text hash and is not flagged again
for the same text, whatever the second read decided.
"""

from __future__ import annotations

import hashlib

from .quote_spans import dialogue_style, quote_spans, spans_by_block

VERSION = 1


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_words(text, start, end, ranges, *, covered):
    """Whether text[start:end] has a letter or digit inside (``covered``) or outside the quotation ranges."""
    for index in range(start, end):
        if text[index].isalnum() and any(low <= index < high for low, high in ranges) is covered:
            return True
    return False


def misread_blocks(document):
    """``{block_id: reason}`` for paragraphs whose saved split contradicts their quotation marks."""
    source = document["source"]
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    rows = {}
    for row in document["segments"]:
        rows.setdefault(row["block_id"], []).append(row)
    done = (document["provenance"].get("split_rereads") or {})
    done = done.get("blocks", {}) if done.get("version") == VERSION else {}
    quoted = spans_by_block(quote_spans(source, dialogue_style(source)))
    result = {}
    for block in source["blocks"]:
        spans = quoted.get(block["id"])
        saved = rows.get(block["id"], [])
        if not spans or not saved or done.get(block["id"]) == text_hash(block["text"]):
            continue
        if any(row["attribution"]["basis"] == "user" or row["kind"] in {"thought", "letter"}
               or "source_restoration" in row for row in saved):
            continue
        text = block["text"]
        ranges = [(span["start"], span["end"]) for span in spans]
        # A short quotation inside a running sentence (`an "honest" offer`, `by “essay” I mean`)
        # names a word, not a line of speech: the narrator may read it.
        speech_ranges = [(start, end) for start, end in ranges
                         if not (text[:start].rstrip()[-1:].isalnum() and len(text[start:end].split()) <= 4)]
        for row in saved:
            role = roles.get(row.get("role_id"), {})
            speech = row["kind"] == "dialogue" and (role.get("kind") == "dialogue" or "simultaneous_speakers" in row)
            narration = row["kind"] == "narration" and role.get("kind") == "narration"
            if speech and _has_words(text, row["start"], row["end"], ranges, covered=False):
                result[block["id"]] = "character speech includes words outside the quotation marks"
                break
            if narration and _has_words(text, row["start"], row["end"], speech_ranges, covered=True):
                result[block["id"]] = "the narrator reads words inside quotation marks"
                break
    return result


def record_rereads(provenance, document, block_ids):
    """Remember that ``block_ids`` were read again at this version, by their current text."""
    texts = {block["id"]: block["text"] for block in document["source"]["blocks"]}
    marker = provenance.get("split_rereads")
    blocks = dict(marker.get("blocks", {})) if isinstance(marker, dict) and marker.get("version") == VERSION else {}
    blocks.update({block_id: text_hash(texts[block_id]) for block_id in block_ids if block_id in texts})
    provenance["split_rereads"] = {"version": VERSION, "blocks": dict(sorted(blocks.items()))}
