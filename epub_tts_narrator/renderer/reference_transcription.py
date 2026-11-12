"""Reference-only nomination of bounded, non-aligned transcription replacements."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math
from typing import Callable, TypedDict


class _ReferenceNominationPolicy(TypedDict):
    version: int
    alignment: str
    operations: list[str]
    max_span_words: int
    max_pairs: int
    max_fraction: float
    minimum_allowed_pairs: int
    max_words: int
    max_characters: int
    context_words: int
    counter_permutation: str
    shared_changed_words: str
    participants: str
    protected: str
    source_spans: str
    anchor_expansion: str


class _TokenizedPassage(TypedDict):
    text: str
    words: list[str]
    originals: list[tuple[str, int, int]]


REFERENCE_NOMINATION: _ReferenceNominationPolicy = {
    "version": 1,
    "alignment": "SequenceMatcher(normalize_words, autojunk=False)",
    "operations": ["equal", "replace"],
    "max_span_words": 3,
    "max_pairs": 4,
    "max_fraction": 0.2,
    "minimum_allowed_pairs": 1,
    "max_words": 2000,
    "max_characters": 100000,
    "context_words": 3,
    "counter_permutation": "reject",
    "shared_changed_words": "reject",
    "participants": "alphabetic_only",
    "protected": "numeric_and_negation",
    "source_spans": "complete_original",
    "anchor_expansion": "common_equal_tokens",
}


def _outside_source(originals, start, end):
    left = start > 0 and originals[start - 1][1] == originals[start][1]
    right = end < len(originals) and originals[end - 1][2] == originals[end][2]
    return left, right


def _context(text, originals, span):
    start, end = span
    left = max(0, start - REFERENCE_NOMINATION["context_words"])
    right = min(len(originals), end + REFERENCE_NOMINATION["context_words"])
    return text[originals[left][1]:originals[right - 1][2]]


def nominate_replacements(expected_passage: _TokenizedPassage, heard_passage: _TokenizedPassage,
                          eligible_participant: Callable[[str], bool]):
    """Nominate bounded replacements from already tokenized, source-bound passages."""
    expected_text, transcript = expected_passage["text"], heard_passage["text"]
    expected, heard = expected_passage["words"], heard_passage["words"]
    if Counter(expected) == Counter(heard):
        return None
    opcodes = SequenceMatcher(None, expected, heard, autojunk=False).get_opcodes()
    if any(tag not in {"equal", "replace"} for tag, *_ in opcodes):
        return None
    blocks = []
    for position, (tag, expected_start, expected_end, actual_start, actual_end) in enumerate(opcodes):
        if tag == "equal":
            continue
        if (not expected_start < expected_end or not actual_start < actual_end
                or expected_end - expected_start > REFERENCE_NOMINATION["max_span_words"]
                or actual_end - actual_start > REFERENCE_NOMINATION["max_span_words"]):
            return None
        blocks.append({"position": position, "expected_start": expected_start,
                       "expected_end": expected_end, "actual_start": actual_start,
                       "actual_end": actual_end})
    if not blocks:
        return None
    expected_changed = {word for block in blocks
                        for word in expected[block["expected_start"]:block["expected_end"]]}
    heard_changed = {word for block in blocks
                     for word in heard[block["actual_start"]:block["actual_end"]]}
    if expected_changed & heard_changed:
        return None
    expected_original, heard_original = expected_passage["originals"], heard_passage["originals"]

    pairs = []
    for block in blocks:
        position = block["position"]
        previous = opcodes[position - 1] if position else None
        following = opcodes[position + 1] if position + 1 < len(opcodes) else None
        lower_expected = previous[1] if previous else 0
        lower_actual = previous[3] if previous else 0
        upper_expected = following[2] if following else len(expected)
        upper_actual = following[4] if following else len(heard)
        start, end = block["expected_start"], block["expected_end"]
        heard_start, heard_end = block["actual_start"], block["actual_end"]
        while True:
            left, right = _outside_source(expected_original, start, end)
            heard_left, heard_right = _outside_source(heard_original, heard_start, heard_end)
            if not (left or right or heard_left or heard_right):
                break
            if left or heard_left:
                if (start <= lower_expected or heard_start <= lower_actual
                        or expected[start - 1] != heard[heard_start - 1]):
                    return None
                start, heard_start = start - 1, heard_start - 1
            else:
                if (end >= upper_expected or heard_end >= upper_actual
                        or expected[end] != heard[heard_end]):
                    return None
                end, heard_end = end + 1, heard_end + 1
            if (end - start > REFERENCE_NOMINATION["max_span_words"]
                    or heard_end - heard_start > REFERENCE_NOMINATION["max_span_words"]):
                return None
        participants = expected[start:end] + heard[heard_start:heard_end]
        if not all(eligible_participant(word) for word in participants):
            return None
        source_start, source_end = expected_original[start][1], expected_original[end - 1][2]
        transcript_start = heard_original[heard_start][1]
        transcript_end = heard_original[heard_end - 1][2]
        pairs.append({
            "id": f"pair_{len(pairs) + 1}", "word_index": start,
            "expected_start": start, "expected_end": end,
            "actual_start": heard_start, "actual_end": heard_end,
            "expected_source_start": source_start, "expected_source_end": source_end,
            "recognized_source_start": transcript_start, "recognized_source_end": transcript_end,
            "expected_word": expected_text[source_start:source_end],
            "recognized_word": transcript[transcript_start:transcript_end],
            "coverage_operation": "replace", "coverage_rule": "reference_balanced_replace",
            "requested_context": _context(expected_text, expected_original, (start, end)),
            "recognized_context": _context(transcript, heard_original, (heard_start, heard_end)),
        })
    allowed = max(REFERENCE_NOMINATION["minimum_allowed_pairs"],
                  math.floor(REFERENCE_NOMINATION["max_fraction"] * len(expected)))
    return pairs if len(pairs) <= min(allowed, REFERENCE_NOMINATION["max_pairs"]) else None
