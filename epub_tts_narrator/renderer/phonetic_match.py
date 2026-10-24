"""Bounded phonetic comparison of complete, ordered ASR transcripts.

This is a pronunciation tolerance, not a statement about a character's intended
pronunciation. Every word must participate in an exact or full phonetic match;
there are no insertion, deletion, or prefix-only matching operations.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
import re
import shlex
import sys
import unicodedata

from .letter_names import make_letter_name_matcher


PHONETIC_MATCH_VERSION = 4
_MAX_ALIGNMENT_STATES = 20_000
_ALPHABETIC_WORD = re.compile(r"[a-z]+(?:'[a-z]+)*\Z")
_PROTECTED_NUMERIC = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand million billion trillion quadrillion
first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth
thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth
twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth
hundredth thousandth millionth billionth trillionth zeroth once twice thrice
half halves quarter quarters dozen dozens minus plus percent percentage
""".split())
_PROTECTED = _PROTECTED_NUMERIC | frozenset("""
no not never neither nor none nobody nothing nowhere cannot cant dont doesnt
didnt wont wouldnt shouldnt couldnt isnt arent wasnt werent hasnt havent hadnt
mustnt neednt shant aint without
""".split())

# A word boundary may have been inserted inside a name. It must not absorb a
# real grammatical word into that name instead. Single-word comparisons still
# tolerate pronunciation changes unless the word is numerically/negatively
# meaningful, but all participants in a multiword comparison exclude this set.
_FUNCTION_WORDS = frozenset("""
a an the and or but if as at by for from in into of off on onto out over per
than that then to up upon via with within about after before between through
under until while during without is am are was were be been being do does did
done have has had having can could may might must shall should will would ought
it its it's he him his himself she her hers herself they them their theirs
themselves we us our ours ourselves you your yours yourself yourselves i me my
mine myself this these those who whom whose which what where when why how
there here so such only also even still yet just too very all any both each
either every few many more most much other same some several enough another
""".split())

# Coalesced speech can have different phonetic codes when a recognizer spells
# out the words. These are exact, bounded expansions, not permission to drop
# function words or invent implied auxiliaries (e.g. "what are you"). Keep
# them inside the alignment so surrounding names and coverage are checked too.
_COLLOQUIAL_EXPANSIONS = {
    "ya": ("you",),
    "what'cha": ("what", "you"),
    "whatcha": ("what", "you"),
    "whatchu": ("what", "you"),
}


def _load_double_metaphone():
    try:
        from metaphone import doublemetaphone
    except ImportError as exc:
        raise RuntimeError(
            "Phonetic speech verification requires Metaphone 0.6. "
            "Install the renderer dependencies, or run "
            f"uv pip install --python {shlex.quote(sys.executable)} 'Metaphone==0.6'."
        ) from exc
    return doublemetaphone


def require_phonetic_matcher() -> None:
    """Check the renderer dependency before expensive speech generation starts."""
    _load_double_metaphone()


def phonetic_equivalence(expected_text: str, actual_text: str) -> dict | None:
    """Return an audited full-transcript match, or leave it for other checks.

    Match a word to one word or to two/three whitespace-separated words using
    complete Double Metaphone primary/alternate keys, exact letter-name
    sequences, or an explicit colloquial expansion. Multiword keys concatenate
    each word's own encoding, preserving every boundary sound. General phonetic
    joins exclude short/non-Latin words and punctuation boundaries; exact letter
    sequences have their own spelling-separator rules. Numbers, negations and
    unmatched words cannot be excused. Exact transcripts return ``None``.
    """
    # Imported at call time so quality can use this helper without a module
    # initialization cycle; all verification paths share lexical normalization.
    from .quality import _lexeme_matches, normalize_words

    def tokens(text):
        source = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
        words, spans = [], []
        for match in _lexeme_matches(source):
            normalized = normalize_words(match.group())
            words.extend(normalized)
            spans.extend([(match.start(), match.end(), match.group().casefold())] * len(normalized))
        return source, words, spans

    expected_source, expected, expected_spans = tokens(expected_text)
    actual_source, actual, actual_spans = tokens(actual_text)
    if len(expected) > 2000 or len(actual) > 4000:
        raise ValueError("text exceeds per-chunk verification token limit")
    if not expected or not actual or expected == actual:
        return None
    if Counter(expected) == Counter(actual):
        # Homophonic words can have identical keys; that cannot justify moving
        # the actual words around when their exact identities are available.
        return None
    double_metaphone = _load_double_metaphone()
    letter_matches = make_letter_name_matcher(
        expected_source, expected, expected_spans, actual_source, actual, actual_spans,
        protected=_PROTECTED)

    @lru_cache(maxsize=None)
    def word_keys(word):
        bare = word.replace("'", "")
        if (not _ALPHABETIC_WORD.fullmatch(word) or len(bare) < 2
                or bare in _PROTECTED or "n't" in word):
            return ()
        return tuple(dict.fromkeys(key for key in double_metaphone(bare) if key))

    @lru_cache(maxsize=None)
    def group_keys(side, start, size, multiword_match):
        source, words, spans = ((expected_source, expected, expected_spans)
                                if side == "expected" else
                                (actual_source, actual, actual_spans))
        end = start + size
        if end > len(words):
            return ()
        selected = words[start:end]
        if multiword_match and any(word.replace("'", "") in _FUNCTION_WORDS
                                   for word in selected):
            return ()
        if any(not _ALPHABETIC_WORD.fullmatch(span[2]) for span in spans[start:end]):
            return ()
        if any(not source[spans[index - 1][1]:spans[index][0]].isspace()
               for index in range(start + 1, end)):
            return ()
        keys = [word_keys(word) for word in selected]
        if not all(keys):
            return ()
        # At most 2**3 combinations. Do not encode a joined spelling: that can
        # remove an added vowel-initial word or collapse consonants at a join.
        return tuple(sorted({"".join(parts) for parts in product(*keys)
                             if len("".join(parts)) >= 2}))

    def next_matches(i, j):
        if expected[i] == actual[j]:
            # Preserve exact anchors rather than swallowing a matching word
            # into a neighboring name to make an otherwise bad passage fit.
            yield i + 1, j + 1, None
            yield from letter_matches(i, j)
            return
        yield from letter_matches(i, j)
        for reverse in (False, True):
            form = actual[j] if reverse else expected[i]
            expansion = _COLLOQUIAL_EXPANSIONS.get(form)
            if expansion is None:
                continue
            source, words, spans, start = ((expected_source, expected, expected_spans, i)
                                          if reverse else
                                          (actual_source, actual, actual_spans, j))
            size = len(expansion)
            if tuple(words[start:start + size]) != expansion:
                continue
            if any(not source[spans[index - 1][1]:spans[index][0]].isspace()
                   for index in range(start + 1, start + size)):
                continue
            end_i, end_j = (i + size, j + 1) if reverse else (i + 1, j + size)
            yield end_i, end_j, {
                "operation": "colloquial", "rule": "_".join(expansion),
                "expected_start": i, "expected_end": end_i,
                "actual_start": j, "actual_end": end_j,
                "expected": expected[i:end_i], "actual": actual[j:end_j],
                "expansion": list(expansion),
            }
        for left_size, right_size in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
            if letter_matches.blocks_fuzzy(i, j, left_size, right_size):
                continue
            multi = left_size != right_size
            if multi:
                single = expected[i] if left_size == 1 else actual[j]
                several = actual[j:j + right_size] if left_size == 1 else expected[i:i + left_size]
                if single in several:
                    # An unchanged word plus other words is added/omitted
                    # content, not an uncertain boundary inside that word.
                    continue
            left = group_keys("expected", i, left_size, multi)
            right = group_keys("actual", j, right_size, multi)
            common = sorted(set(left).intersection(right))
            if not common:
                continue
            end_i, end_j = i + left_size, j + right_size
            operation = "substitute" if not multi else "split" if left_size == 1 else "join"
            yield end_i, end_j, {
                "operation": operation,
                "expected_start": i, "expected_end": end_i,
                "actual_start": j, "actual_end": end_j,
                "expected": expected[i:end_i], "actual": actual[j:end_j],
                "matched_key": common[0],
                "expected_keys": list(left), "actual_keys": list(right),
            }

    # A sparse, acyclic search permits bounded split/join ambiguity without an
    # unbounded edit-distance matrix. Every edge consumes words on BOTH sides.
    start = (0, 0)
    pending = [start]
    parents = {start: None}
    finish = (len(expected), len(actual))
    while pending:
        state = pending.pop()
        if state == finish:
            changes = []
            while parents[state] is not None:
                previous, change = parents[state]
                if change is not None:
                    changes.append(change)
                state = previous
            changes.reverse()
            if not changes:
                return None
            # Explicit expansions already preserve the same lexical content;
            # only uncertain sound matches can disguise moved exact words.
            phonetic_changes = [change for change in changes
                                if change["operation"] not in {"colloquial", "letter_names"}]
            changed_expected = {word for change in phonetic_changes for word in change["expected"]}
            changed_actual = {word for change in phonetic_changes for word in change["actual"]}
            if changed_expected & changed_actual:
                # An exact word displaced into another changed span is a
                # reorder, even if all those words happen to sound alike.
                return None
            return {
                "method": f"double_metaphone_v{PHONETIC_MATCH_VERSION}",
                "implementation": "Metaphone", "version": "0.6",
                "alignment": changes,
            }
        i, j = state
        if i == len(expected) or j == len(actual):
            continue
        for end_i, end_j, change in reversed(list(next_matches(i, j))):
            following = (end_i, end_j)
            if following not in parents:
                if len(parents) >= _MAX_ALIGNMENT_STATES:
                    return None
                parents[following] = (state, change)
                pending.append(following)
    return None
