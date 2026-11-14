"""Accept a chapter take whose transcript is close to its passage.

Most nonexact transcripts are the recognizer mishearing accented speech or
invented names, not the engine saying something else. This check accepts a
take locally when every passage word is present in order, exactly or as a
near sound or spelling (including a name heard across new word breaks), and
the remaining differences are few and scattered. It still rejects what a
listener hears as wrong: a dropped word, a run of more than
``MAX_DIFFERENCE_RUN`` consecutive wrong or added words (a skipped or
replaced phrase, or speech from another passage), too many differences
overall, a changed, added or dropped negation, or a changed or added number
(numbers and negation stay protected, as everywhere in verification). Very
short passages, where one word is the whole line, are left to the stricter
checks.

Vocal sounds (``oh``, ``ah``) and symbols without a fixed spoken form never
count either way. This is a chapter-speech rule only: a voice reference must
match its words exactly because its transcript conditions cloning.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
import re

CLOSE_MATCH_VERSION = 1
MAX_DIFFERENCE_RUN = 2
MIN_PASSAGE_WORDS = 2
_MAX_WORDS = 400
_JOINS = ((2, 1), (3, 1), (1, 2), (1, 3))
_JOIN_COST = 0.1
_NEGATIONS = frozenset("no not never nor neither none nobody nothing nowhere cannot".split())
_SYMBOL_NAMES = {"/": frozenset({"slash", "stroke"})}


def _negation(word):
    return word in _NEGATIONS or word.endswith("n't")


def _number(word):
    from .phonetic_match import _PROTECTED_NUMERIC
    return word in _PROTECTED_NUMERIC or word[:1].isdigit()


def _protected_change(change):
    """A difference that changes what is said, not how it was heard."""
    expected, heard = change["expected"], change["actual"]
    if any(_negation(word) for word in expected + heard):
        return True
    if change["operation"] == "added":
        # A recognizer writes a joining hyphen ("F42NW-43NW") as a sign.
        return any(_number(word) and word not in {"minus", "plus"} for word in heard)
    return (any(_number(word) for word in expected) and any(_number(word) for word in heard))


def _ignorable(word):
    from .speech_coverage import _VOCALIZATIONS
    return word in _VOCALIZATIONS or not word[:1].isalnum()


@lru_cache(maxsize=65536)
def _distance(left, right):
    """0 for the same word, below 1 for a near sound or spelling, 1 otherwise."""
    if left == right:
        return 0.0
    if right in _SYMBOL_NAMES.get(left, ()) or left in _SYMBOL_NAMES.get(right, ()):
        return 0.0
    if (_negation(left) or _negation(right) or _number(left) or _number(right)
            or not (left[:1].isalnum() and right[:1].isalnum())):
        return 1.0  # Only an exact negation or number matches one.
    from .phonetic_match import _load_double_metaphone
    metaphone = _load_double_metaphone()
    left, right = left.replace("'", ""), right.replace("'", "")
    left_keys = {key for key in metaphone(left) if key}
    right_keys = {key for key in metaphone(right) if key}
    if left_keys & right_keys:
        return 0.2
    if SequenceMatcher(None, left, right, autojunk=False).ratio() >= 0.6:
        return 0.4
    if any(len(first) >= 2 and len(second) >= 2
           and SequenceMatcher(None, first, second, autojunk=False).ratio() >= 0.66
           for first in left_keys for second in right_keys):
        return 0.5
    return 1.0


def _group_distance(expected, heard):
    """A name or compound heard across different word breaks, at similar length."""
    if len(expected) > 1 and any(len(word) < 2 for word in expected):
        # A spelled letter is a word of its own: it joins only an exact
        # spelling ("e mailed" / "emailed"), never a near one.
        return 0.0 if "".join(expected) == "".join(heard) else 1.0
    if any(_negation(word) or _number(word) for word in expected + heard):
        return 1.0
    left, right = "".join(expected), "".join(heard)
    if not re.fullmatch(r"[a-z']+", left) or not re.fullmatch(r"[a-z']+", right):
        return 1.0
    if min(len(left), len(right)) < 0.75 * max(len(left), len(right)):
        return 1.0  # The shorter side would swallow a word.
    return _distance(left, right)


def _alignment(expected, heard):
    rows, columns = len(expected), len(heard)
    cost = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    step = [[None] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0], step[i][0] = float(i), ("deleted", 1, 0)
    for j in range(1, columns + 1):
        cost[0][j], step[0][j] = float(j), ("added", 0, 1)
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            distance = _distance(expected[i - 1], heard[j - 1])
            options = [(cost[i - 1][j - 1] + distance,
                        ("equal" if distance == 0 else "near" if distance < 1 else "different", 1, 1)),
                       (cost[i - 1][j] + 1, ("deleted", 1, 0)), (cost[i][j - 1] + 1, ("added", 0, 1))]
            for left, right in _JOINS:
                if i >= left and j >= right:
                    distance = _group_distance(expected[i - left:i], heard[j - right:j])
                    if distance < 1:
                        options.append((cost[i - left][j - right] + distance + _JOIN_COST, ("near", left, right)))
            cost[i][j], step[i][j] = min(options, key=lambda option: option[0])
    changes, i, j = [], rows, columns
    while i or j:
        operation, left, right = step[i][j]
        changes.append({"operation": operation, "expected_start": i - left, "expected_end": i,
                        "actual_start": j - right, "actual_end": j,
                        "expected": expected[i - left:i], "actual": heard[j - right:j]})
        i, j = i - left, j - right
    return changes[::-1]


def close_transcript(expected_text, transcript, max_fraction):
    """Return the audited close match, or ``None`` when the take must be checked strictly."""
    from .quality import normalize_words
    if not isinstance(max_fraction, (int, float)) or max_fraction <= 0:
        return None
    expected, heard = normalize_words(expected_text), normalize_words(transcript)
    spoken = [word for word in expected if not _ignorable(word)]
    if (len(spoken) < MIN_PASSAGE_WORDS or not heard
            or len(expected) > _MAX_WORDS or len(heard) > _MAX_WORDS):
        return None
    changes = _alignment(expected, heard)
    differences, run, longest = [], 0, 0
    for change in changes:
        if change["operation"] in {"equal", "near"}:
            run = 0
            if change["operation"] == "near":
                differences.append(dict(change, counted=False))
            continue
        if all(_ignorable(word) for word in change["expected"] + change["actual"]):
            differences.append(dict(change, counted=False))
            continue
        if change["operation"] == "deleted" or _protected_change(change):
            return None  # A passage word never heard, or a changed number or negation.
        run += 1
        longest = max(longest, run)
        differences.append(dict(change, counted=True))
    counted = sum(change["counted"] for change in differences)
    allowed = max(1, round(max_fraction * len(spoken)))
    if longest > MAX_DIFFERENCE_RUN or counted > allowed:
        return None
    return {"method": f"close_match_v{CLOSE_MATCH_VERSION}", "expected_words": len(spoken),
            "counted_differences": counted, "allowed_differences": allowed,
            "longest_difference_run": longest, "differences": differences}
