"""Conservative, complete word coverage required before a text-only judgment.

Passing this check means a transcript is eligible for pronunciation review,
not that it has been approved. Every alignment edge consumes words on both
sides. There is deliberately no percentage allowance for missing speech.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import re
import unicodedata

from .letter_names import make_letter_name_matcher
from .phonetic_match import (
    _ALPHABETIC_WORD, _COLLOQUIAL_EXPANSIONS, _FUNCTION_WORDS, _PROTECTED,
    _load_double_metaphone, phonetic_equivalence,
)


SPEECH_COVERAGE_VERSION = 7
_MAX_ALIGNMENT_STATES = 20_000
_SPELLING_VARIANTS = {
    frozenset(("qi", "chi")): "standard_transliteration",
    frozenset(("judgement", "judgment")): "regional_spelling",
}
_CONTRACTIONS = {
    "i'm": (("i", "am"),), "let's": (("let", "us"),),
    "can't": (("can", "not"),), "cannot": (("can", "not"),),
    "won't": (("will", "not"),), "shan't": (("shall", "not"),),
    **{f"{word}n't": ((word, "not"),) for word in (
        "are", "is", "was", "were", "do", "does", "did", "have", "has", "had",
        "could", "would", "should", "must", "need", "dare",
    )},
    **{f"{word}'re": ((word, "are"),) for word in ("you", "we", "they", "who")},
    **{f"{word}'ve": ((word, "have"),) for word in ("i", "you", "we", "they", "could", "would", "should", "must")},
    **{f"{word}'ll": ((word, "will"),) for word in ("i", "you", "he", "she", "it", "we", "they", "who", "that", "there")},
    **{f"{word}'d": ((word, "had"), (word, "would")) for word in ("i", "you", "he", "she", "it", "we", "they", "who", "that", "there")},
    **{f"{word}'s": ((word, "is"), (word, "has")) for word in ("he", "she", "it", "who", "what", "that", "there", "here")},
}


def stretched_word_equivalent(expected, actual):
    """Recognize shortening a visibly elongated word, never a missing word.

    One source run must contain at least three repeated letters. Once that
    identifies deliberate elongation, other doubled runs may shorten too:
    ``Wheerrre`` becomes ``Where``. Ordinary ``dessert``/``desert`` does not
    qualify. The target preserves every distinct letter and apostrophe in
    order, cannot lengthen any run, and cannot add or remove a word boundary.
    This permits sound review; it does not approve a take by itself.
    """
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    expected, actual = (value.casefold().replace("’", "'") for value in (expected, actual))
    if (not _ALPHABETIC_WORD.fullmatch(expected) or not _ALPHABETIC_WORD.fullmatch(actual)
            or len(actual.replace("'", "")) < 2 or expected == actual):
        return False
    left, right = ([match.group() for match in re.finditer(r"(.)\1*", value)]
                   for value in (expected, actual))
    return (len(left) == len(right)
            and any(run[0].isalpha() and len(run) >= 3 for run in left)
            and all(a[0] == b[0] and len(b) <= len(a) for a, b in zip(left, right)))


def _near_phonetic_word(left, right, left_keys, right_keys):
    """Admit a small within-word sound change for review, never acceptance.

    Metaphone keys are rough sound encodings, not evidence that speech is
    correct. Permit one insertion, deletion, or substitution within complete
    keys only when at least three encoded sounds remain and the edit accounts
    for at most a quarter of the longer key. Whole-word alignment still has
    no insertion/deletion edges. Grammar words, quantities, and negations do
    not gain this tolerance, and the sound judge must approve the candidate.
    """
    if any(len(word.replace("'", "")) < 4 or word in _FUNCTION_WORDS
           or word.replace("'", "") in _PROTECTED or "n't" in word
           for word in (left, right)):
        return False
    for first in left_keys:
        for second in right_keys:
            if min(len(first), len(second)) < 3 or max(len(first), len(second)) < 4:
                continue
            if len(first) == len(second):
                if sum(a != b for a, b in zip(first, second)) == 1:
                    return True
            else:
                shorter, longer = sorted((first, second), key=len)
                if len(longer) == len(shorter) + 1 and any(
                        longer[:index] + longer[index + 1:] == shorter
                        for index in range(len(longer))):
                    return True
    return False


def speech_coverage(expected_text, transcript):
    """Return an auditable eligibility result without making any API calls.

    In addition to established complete phonetic matches, permit exact
    contractions and spelling boundaries, and bounded pronunciation candidates
    such as ``admitted`` / ``admit it`` or ``libary`` / ``library``. These extra
    edges still require an independent sound judgment; they cannot approve audio.
    """
    from .quality import _lexeme_matches, normalize_words, word_spacing_equivalence

    result = {"passed": False, "version": SPEECH_COVERAGE_VERSION,
              "reason": "unmatched_spoken_content", "alignment": []}
    if (not isinstance(expected_text, str) or not isinstance(transcript, str)
            or len(expected_text) > 100_000 or len(transcript) > 100_000):
        return dict(result, reason="invalid_text")

    def tokens(text):
        source = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
        words, spans = [], []
        for match in _lexeme_matches(source):
            normalized = normalize_words(match.group())
            words.extend(normalized)
            spans.extend([(match.start(), match.end(), match.group().casefold())] * len(normalized))
        return source, words, spans

    expected_source, expected, expected_spans = tokens(expected_text)
    actual_source, actual, actual_spans = tokens(transcript)
    if not expected or not actual:
        return dict(result, reason="empty_spoken_content")
    if len(expected) > 2000 or len(actual) > 4000:
        return dict(result, reason="comparison_limit")
    if expected == actual:
        return dict(result, passed=True, reason="exact_spoken_content")
    if Counter(expected) == Counter(actual):
        return dict(result, reason="reordered_spoken_content")
    spacing = word_spacing_equivalence(expected_text, transcript)
    if spacing is not None:
        return dict(result, passed=True, reason="exact_spelling_boundaries",
                    alignment=spacing["alignment"])

    complete = phonetic_equivalence(expected_text, transcript)
    if complete is not None:
        return dict(result, passed=True, reason="complete_phonetic_coverage",
                    alignment=complete["alignment"])
    metaphone = _load_double_metaphone()
    letter_matches = make_letter_name_matcher(
        expected_source, expected, expected_spans, actual_source, actual, actual_spans,
        protected=_PROTECTED)
    from .address_speech import make_address_matcher
    address_matches = make_address_matcher(
        expected_source, expected, expected_spans, actual_source, actual, actual_spans)

    @lru_cache(maxsize=None)
    def keys(spelling):
        return tuple(dict.fromkeys(key for key in metaphone(spelling) if len(key) >= 2))

    def eligible_word(word):
        bare = word.replace("'", "")
        return (bool(_ALPHABETIC_WORD.fullmatch(word)) and len(bare) >= 2
                and bare not in _PROTECTED and "n't" not in word)

    def contiguous(side, start, size, *, hyphen=False):
        source, words, spans = ((expected_source, expected, expected_spans)
                                if side == "expected" else
                                (actual_source, actual, actual_spans))
        if start + size > len(words):
            return False
        return all((bool(re.fullmatch(r"[\s\-‐‑]+", source[spans[index - 1][1]:spans[index][0]]))
                    if hyphen else source[spans[index - 1][1]:spans[index][0]].isspace())
                   for index in range(start + 1, start + size))

    def exact_compound(i, j, left_size, right_size):
        left, right = expected[i:i + left_size], actual[j:j + right_size]
        if len(left) != left_size or len(right) != right_size:
            return False
        # Use the same exact boundary rules as direct acceptance. A component
        # such as "half" inside "halfblade" is not a changed numeric value.
        # Numeric tokens and missing letters still cannot use this route.
        left_text = expected_source[expected_spans[i][0]:expected_spans[i + left_size - 1][1]]
        right_text = actual_source[actual_spans[j][0]:actual_spans[j + right_size - 1][1]]
        return word_spacing_equivalence(left_text, right_text) is not None

    def span_words(side, start, size):
        source, words, spans = ((expected_source, expected, expected_spans)
                                if side == "expected" else
                                (actual_source, actual, actual_spans))
        end = start + size
        if end > len(words):
            return None
        selected = words[start:end]
        if (not all(eligible_word(word) for word in selected)
                or any(not _ALPHABETIC_WORD.fullmatch(span[2]) for span in spans[start:end])
                or any(not source[spans[index - 1][1]:spans[index][0]].isspace()
                       for index in range(start + 1, end))):
            return None
        return selected

    @lru_cache(maxsize=None)
    def group_keys(side, start, size):
        words = span_words(side, start, size)
        if words is None:
            return ()
        full = set(keys("".join(words).replace("'", "")))
        if size > 1:
            # A word must contribute to the sound of the whole group. Without
            # this constraint a vowel-initial extra word may disappear when
            # the encoder processes the joined spelling.
            for index in range(size):
                remainder = "".join(words[:index] + words[index + 1:]).replace("'", "")
                full.difference_update(keys(remainder))
        return tuple(sorted(full))

    def one_repeated_consonant(left, right):
        shorter, longer = sorted((left, right), key=len)
        if len(shorter) < 3 or len(longer) != len(shorter) + 1 or shorter[0] != longer[0]:
            return False
        return any(longer[index] == longer[index - 1]
                   and longer[:index] + longer[index + 1:] == shorter
                   for index in range(1, len(longer)))

    def one_spelling_letter(left, right):
        shorter, longer = sorted((left.replace("'", ""), right.replace("'", "")), key=len)
        return len(longer) == len(shorter) + 1 and any(
            longer[:index] + longer[index + 1:] == shorter for index in range(len(longer)))

    def spelling_variant(left, right):
        folded = ["".join(char for char in unicodedata.normalize("NFD", word)
                          if not unicodedata.combining(char)) for word in (left, right)]
        if not all(eligible_word(word) for word in folded):
            return None
        if folded[0] == folded[1]:
            return "accent_spelling"
        return _SPELLING_VARIANTS.get(frozenset(folded))

    def bounded_pronunciation(left, right, left_keys, right_keys, i, j):
        # All rules cover a whole token. They confer eligibility for the model
        # to judge the sound, never permission to remove a neighboring word.
        if all(re.fullmatch(r"h?m{2,}", word) for word in (left, right)):
            return "nasal_hum"
        runs = [re.findall(r"(.)\1*", word) for word in (left, right)]
        if runs[0] == runs[1]:
            groups = [re.findall(r"((.)\2*)", word) for word in (left, right)]
            lengths = [[len(run) for run, _ in parts] for parts in groups]
            if (any(a != b for a, b in zip(*lengths))
                    and all(a == b or max(a, b) >= 3 for a, b in zip(*lengths))):
                return "stretched_spelling"
        for longer, shorter in ((left, right), (right, left)):
            if longer.endswith("ing") and len(longer) >= 5 and shorter == longer[:-1]:
                return "colloquial_ing"
        if any(a[0] in {"J", "X"} and b[0] in {"J", "X"} and a[1:] == b[1:]
               for a in left_keys for b in right_keys):
            return "affricate_voicing"
        if one_spelling_letter(left, right) and any(
                one_repeated_consonant(a, b) for a in left_keys for b in right_keys):
            return "repeated_phonetic_consonant"
        if (not any(source[span[0]:span[1]].isupper()
                    for source, span in ((expected_source, expected_spans[i]),
                                         (actual_source, actual_spans[j])))
                and _near_phonetic_word(left, right, left_keys, right_keys)):
            return "near_phonetic_word"
        return None

    def terminal_s_variation(left, right, i, j):
        # A weak final /s/ or /z/ can disappear in connected speech or its
        # transcription. Retain BOTH complete words and require a sound judge;
        # this is not stemming, grammatical correction, or local acceptance.
        if (any(word in _CONTRACTIONS or word.replace("'", "") in _PROTECTED
                or word in _FUNCTION_WORDS for word in (left, right))):
            return None
        base, inflected = sorted((left, right), key=len)
        if (not re.fullmatch(r"[a-z]{3,}", base) or not re.search(r"[aeiouy]", base)
                or inflected not in {base + "s", base + "'s"}):
            return None
        for source, span in ((expected_source, expected_spans[i]),
                             (actual_source, actual_spans[j])):
            if source[span[0]:span[1]].isupper():
                # Check both sides: ASR can lowercase a shortened initialism.
                # A final spoken letter is not a weak grammatical ending.
                return None
        return "terminal_s_variation"

    def next_matches(i, j):
        yield from address_matches(i, j)
        if expected[i] == actual[j]:
            yield i + 1, j + 1, None
            yield from letter_matches(i, j)
            return
        yield from letter_matches(i, j)
        if stretched_word_equivalent(expected[i], actual[j]):
            yield i + 1, j + 1, {"operation": "pronunciation_candidate", "rule": "stretched_spelling",
                "expected_start": i, "expected_end": i + 1, "actual_start": j,
                "actual_end": j + 1, "expected": expected[i:i + 1], "actual": actual[j:j + 1]}
        variant = spelling_variant(expected[i], actual[j])
        if variant is not None:
            yield i + 1, j + 1, {"operation": "pronunciation_candidate", "rule": variant,
                "expected_start": i, "expected_end": i + 1, "actual_start": j,
                "actual_end": j + 1, "expected": expected[i:i + 1], "actual": actual[j:j + 1]}
        left_expansions, right_expansions = _CONTRACTIONS.get(expected[i], ()), _CONTRACTIONS.get(actual[j], ())
        if set(left_expansions).intersection(right_expansions):
            yield i + 1, j + 1, {"operation": "contraction", "expected_start": i,
                "expected_end": i + 1, "actual_start": j, "actual_end": j + 1,
                "expected": expected[i:i + 1], "actual": actual[j:j + 1]}
        for reverse in (False, True):
            expansions = right_expansions if reverse else left_expansions
            words, start, side = (expected, i, "expected") if reverse else (actual, j, "actual")
            for expansion in expansions:
                size = len(expansion)
                if tuple(words[start:start + size]) != expansion or not contiguous(side, start, size):
                    continue
                end_i, end_j = (i + size, j + 1) if reverse else (i + 1, j + size)
                yield end_i, end_j, {"operation": "contraction", "expected_start": i,
                    "expected_end": end_i, "actual_start": j, "actual_end": end_j,
                    "expected": expected[i:end_i], "actual": actual[j:end_j]}
        for reverse in (False, True):
            form = actual[j] if reverse else expected[i]
            expansion = _COLLOQUIAL_EXPANSIONS.get(form)
            if expansion is None:
                continue
            source, words, spans, start = ((expected_source, expected, expected_spans, i)
                                          if reverse else
                                          (actual_source, actual, actual_spans, j))
            size = len(expansion)
            if (tuple(words[start:start + size]) != expansion
                    or any(not source[spans[index - 1][1]:spans[index][0]].isspace()
                           for index in range(start + 1, start + size))):
                continue
            end_i, end_j = (i + size, j + 1) if reverse else (i + 1, j + size)
            yield end_i, end_j, {"operation": "colloquial", "expected_start": i,
                "expected_end": end_i, "actual_start": j, "actual_end": end_j,
                "expected": expected[i:end_i], "actual": actual[j:end_j]}
        for left_size, right_size in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
            if letter_matches.blocks_fuzzy(i, j, left_size, right_size):
                continue
            if left_size != right_size and exact_compound(i, j, left_size, right_size):
                end_i, end_j = i + left_size, j + right_size
                yield end_i, end_j, {"operation": "spelling_boundary", "expected_start": i,
                    "expected_end": end_i, "actual_start": j, "actual_end": end_j,
                    "expected": expected[i:end_i], "actual": actual[j:end_j]}
            left = span_words("expected", i, left_size)
            right = span_words("actual", j, right_size)
            if left is None or right is None:
                continue
            multi = left_size != right_size
            if multi and set(left).intersection(right):
                continue
            left_keys, right_keys = group_keys("expected", i, left_size), group_keys("actual", j, right_size)
            common = sorted(set(left_keys).intersection(right_keys))
            candidate = ((terminal_s_variation(left[0], right[0], i, j)
                          or bounded_pronunciation(left[0], right[0], left_keys, right_keys, i, j))
                         if not multi and not common else None)
            if not common and candidate is None:
                continue
            end_i, end_j = i + left_size, j + right_size
            yield end_i, end_j, {
                "operation": "pronunciation_candidate" if candidate else "phonetic_group",
                "rule": candidate or "complete_group_sound",
                "expected_start": i, "expected_end": end_i,
                "actual_start": j, "actual_end": end_j,
                "expected": left, "actual": right,
                "expected_keys": list(left_keys), "actual_keys": list(right_keys),
            }

    pending, parents = [(0, 0)], {(0, 0): None}
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
            uncertain = [change for change in changes if change["operation"] not in {
                "colloquial", "contraction", "spelling_boundary", "letter_names", "address_pronunciation"}]
            if ({word for change in uncertain for word in change["expected"]}
                    & {word for change in uncertain for word in change["actual"]}):
                return dict(result, reason="reordered_spoken_content")
            return dict(result, passed=True, reason="complete_reviewable_coverage", alignment=changes)
        i, j = state
        if i == len(expected) or j == len(actual):
            continue
        for end_i, end_j, change in reversed(list(next_matches(i, j))):
            following = (end_i, end_j)
            if following not in parents:
                if len(parents) >= _MAX_ALIGNMENT_STATES:
                    return dict(result, reason="comparison_limit")
                parents[following] = (state, change)
                pending.append(following)
    return result
