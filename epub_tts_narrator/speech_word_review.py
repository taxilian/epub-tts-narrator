"""Ordered word coverage and explicit contextual evidence for unfamiliar spellings."""

from functools import lru_cache

from .speech_normalization import _WORDS, _fold, _word_forms_accounted_for


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
        for left_size, right_size in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
            left, right = original[i:i + left_size], replacement[j:j + right_size]
            if len(left) != left_size or len(right) != right_size:
                continue
            known = _word_forms_accounted_for(" ".join(left), " ".join(right))
            if not known and not (left_size == right_size == 1
                                  and all(word.replace("’", "'").replace("'", "").isalpha()
                                          for word in (left[0], right[0]))
                                  and _fold(left[0]) not in anchors and _fold(right[0]) not in anchors):
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
    expected = [pair["word_id"] for pair in proposal["word_preservation"]["pairs"]]
    words = vote.get("word_verdicts")
    if (not isinstance(words, list) or len(words) != len(expected)
            or any(not isinstance(word, dict) or set(word) != {"word_id", "verdict"}
                   or word.get("word_id") != identifier
                   or not isinstance(word.get("verdict"), str)
                   or word.get("verdict") not in {"approve", "reject", "uncertain"}
                   for word, identifier in zip(words, expected))):
        raise ValueError("Contextual word review must return word_verdicts for every word_id exactly once in pair order")


def approves(vote, proposal):
    if vote["verdict"] != "approve":
        return False
    return ("word_preservation" not in proposal
            or all(word["verdict"] == "approve" for word in vote["word_verdicts"]))
