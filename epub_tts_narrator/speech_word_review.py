"""Ordered word coverage and explicit contextual evidence for unfamiliar spellings."""

import difflib
from functools import lru_cache
import json
import re

from .speech_normalization import _WORDS, _fold, _word_forms_accounted_for


def _fused_token(word):
    """A long single token that may fuse several ordinary spoken words."""
    return len(re.sub(r"[^A-Za-z]", "", word)) >= 12


def _fused_plausible(source, words):
    """Letters of the fused token and the expansion substantially correspond."""
    letters = lambda value: _fold(re.sub(r"[^A-Za-z']+", "", value).casefold().replace("'", ""))
    before, after = letters(source), letters("".join(words))
    if not before or not after:
        return False
    return difflib.SequenceMatcher(None, before, after).ratio() >= 0.6


def word_alignment(before, after):
    """Preserve ordered anchors and word slots; unfamiliar pairs require a jury.

    Known joins/splits retain the existing spelling rules. Unknown edges are
    one-to-one only and cannot consume a word present unchanged on both sides.
    Minimize unknown pairs, breaking ties by the fixed edge order.
    """
    original, replacement = _WORDS.findall(before), _WORDS.findall(after)
    anchors = {_fold(word) for word in original} & {_fold(word) for word in replacement}

    @lru_cache(None)
    def aligned(i, j):
        if i == len(original) or j == len(replacement):
            return (0, ()) if i == len(original) and j == len(replacement) else None
        best = None
        sizes = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1)]
        if i < len(original) and _fused_token(original[i]) and j + 1 < len(replacement):
            sizes += [(1, size) for size in range(4, 9) if j + size <= len(replacement)]
        for left_size, right_size in sizes:
            left, right = original[i:i + left_size], replacement[j:j + right_size]
            if len(left) != left_size or len(right) != right_size:
                continue
            known = _word_forms_accounted_for(" ".join(left), " ".join(right))
            if not known and not (left_size == right_size == 1
                                  and all(word.replace("’", "'").replace("'", "").isalpha()
                                          for word in (left[0], right[0]))
                                  and _fold(left[0]) not in anchors and _fold(right[0]) not in anchors) \
                    and not (left_size == 1 and right_size > 1
                             and _fused_token(left[0]) and _fused_plausible(left[0], right)):
                continue
            rest = aligned(i + left_size, j + right_size)
            if rest is not None:
                candidate = (rest[0] + (0 if known else 1), ((tuple(left), tuple(right)),) + rest[1])
                if best is None or candidate[0] < best[0]:
                    best = candidate
        return best

    result = aligned(0, 0)
    return None if result is None else result[1]


def review_requirement(edit):
    if _word_forms_accounted_for(edit["original"], edit["replacement"]):
        return None
    alignment = word_alignment(edit["original"], edit["replacement"])
    if alignment is None:
        raise ValueError("Contextual word review requires complete ordered source/replacement coverage")
    return {"method": "contextual", "pairs": [
        {"word_id": f"w{index}", "original": list(left), "replacement": list(right)}
        for index, (left, right) in enumerate(alignment)]}


def validate_word_verdicts(vote, proposal):
    if "word_preservation" in proposal:
        expected = [pair["word_id"] for pair in proposal["word_preservation"]["pairs"]]
    else:
        if not _word_forms_accounted_for(proposal["original"], proposal["replacement"]):
            raise ValueError("Unfamiliar edits require supplied word_preservation pairs")
        expected = None
    words = vote.get("word_verdicts")
    if not isinstance(words, list):
        raise ValueError("word_verdicts must be an array")
    if expected is not None and len(words) != len(expected):
        raise ValueError(f"Expected {len(expected)} ordered word verdict(s) for complete coverage; received {len(words)}")
    if expected is None and len(words) > 16:
        raise ValueError("Optional word annotations exceed the bounded eight-word-per-side edit scope")
    for index, word in enumerate(words):
        if (not isinstance(word, dict) or set(word) != {"word_id", "verdict"}
                or not isinstance(word.get("word_id"), str) or not word["word_id"].strip()
                or len(word["word_id"]) > 128 or not word["word_id"].isprintable()
                or not isinstance(word.get("verdict"), str)
                or word["verdict"] not in {"approve", "reject", "uncertain"}):
            raise ValueError(f"word_verdicts[{index}] requires a nonempty bounded word_id and approve/reject/uncertain verdict")
    identifiers = [word["word_id"] for word in words]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("word_verdicts contains duplicate word_id values; each ordered word verdict needs a unique ID")
    if expected is not None and identifiers != expected:
        raise ValueError(f"Expected provided word_ids {json.dumps(expected)}; received {json.dumps(identifiers)}. "
                         "Copy the IDs from this edit's word_preservation pairs; do not number across proposals.")


def approves(vote, proposal):
    if vote["verdict"] != "approve":
        return False
    return (all(word["verdict"] == "approve" for word in vote["word_verdicts"])
            if "word_verdicts" in vote else "word_preservation" not in proposal)
