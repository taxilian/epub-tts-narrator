"""Exact spoken English letter names, with complete ordered coverage.

ASR can write two spoken occurrences of ``see`` as ``CC``. That is a
spelling convention, rather than a fuzzy pronunciation match. Keep the
individual letters so an extra or missing repetition cannot disappear.
"""

from functools import lru_cache
import re


_MAX_LETTERS = 6
_MAX_STATES = 512
_UPPER_RUN = re.compile(r"[A-Z]{2,6}\Z")
# Y can act as a vowel in ordinary words such as why, dry, and rhythm.
# Uppercase runs still support it, as do separated individual letters.
_LOWER_RUN = re.compile(r"[bcdfghjklmnpqrstvwxz]{2,6}\Z")
_CONSONANT_RUN = re.compile(r"[bcdfghjklmnpqrstvwxz]{2,}\Z")
_SINGLE_LETTER = re.compile(r"[A-Za-z]\Z")
_EXPRESSIVE_SOUND = re.compile(r"(?:h?m{2,}|sh+|br+|ps+t|ts+k|s{2,})\Z")
_NAME_TO_LETTER = {
    name: letter
    for letter, names in {
        "A": ("ay", "aye"), "B": ("bee", "be"),
        "C": ("see", "sea", "cee"), "D": ("dee",), "E": ("ee",),
        "F": ("ef", "eff"), "G": ("gee",), "H": ("aitch", "haitch"),
        "I": ("eye",), "J": ("jay",), "K": ("kay",),
        "L": ("el", "ell"), "M": ("em",), "N": ("en",), "O": ("oh",),
        "P": ("pee", "pea"), "Q": ("cue", "queue"), "R": ("are", "ar"),
        "S": ("ess", "es"), "T": ("tee", "tea"),
        "U": ("you", "ewe", "yew"), "V": ("vee",),
        "X": ("ex",), "Y": ("why", "wye"), "Z": ("zee", "zed"),
    }.items()
    for name in names
}
_REWRITE_NAME_TO_LETTER = {
    re.sub(r"([a-z])\1+", r"\1", name): letter for name, letter in _NAME_TO_LETTER.items()
}


def letter_name_rewrite_equivalent(source, replacement):
    """Recognize one exact letter-name respelling for reviewed synthesis only."""
    if (not isinstance(source, str) or not _SINGLE_LETTER.fullmatch(source)
            or not isinstance(replacement, str)):
        return False
    name = re.sub(r"([a-z])\1+", r"\1", replacement.casefold())
    letter = "W" if name in {"double you", "double u"} else _REWRITE_NAME_TO_LETTER.get(name)
    return letter == source.upper()


def make_letter_name_matcher(expected_source, expected_words, expected_spans,
                             actual_source, actual_words, actual_spans, *, protected):
    """Return exact alignment edges for two through six spoken letters.

    Sources retain case; spans contain (start, end, casefolded lexeme). At
    least one side must actually contain letter glyphs. Arbitrary lowercase
    words are never expanded into their spelling. Every edge consumes both
    complete spans, including each repeated letter, in its original order.
    """
    protected = frozenset(protected)
    sides = ((expected_source, expected_words, expected_spans),
             (actual_source, actual_words, actual_spans))

    def allowed_word(word):
        return word.replace("'", "") not in protected and "n't" not in word

    def gap_allowed(source, spans, left, right, *, dotted=False):
        gap = source[spans[left][1]:spans[right][0]]
        # Periods are spelling separators only between actual letter glyphs;
        # do not cross sentence boundaries between ordinary spoken names.
        return bool(re.fullmatch(r"[\s.\-‐‑]+" if dotted else r"[\s\-‐‑]+", gap))

    @lru_cache(maxsize=None)
    def representations(side, start):
        source, words, spans = sides[side]
        if start < 0 or start >= len(words) or len(spans) != len(words):
            return ()
        pending = [(start, "", False, False)]
        seen = set(pending)
        results = set()
        while pending:
            position, letters, has_glyph, previous_glyph = pending.pop()
            if len(letters) >= 2:
                results.add((position, letters, has_glyph))
            if position >= len(words) or len(letters) >= _MAX_LETTERS:
                continue
            word = words[position]
            if not allowed_word(word):
                continue
            begin, end, lexeme = spans[position]
            raw = source[begin:end]
            # Number/symbol normalization may produce words sharing a span.
            # They must never be interpreted as original spoken letter names.
            if word != lexeme:
                continue
            options = []
            if (_SINGLE_LETTER.fullmatch(raw) or _UPPER_RUN.fullmatch(raw)
                    or _LOWER_RUN.fullmatch(raw.casefold())):
                options.append((position + 1, raw.upper(), True))
            if word in _NAME_TO_LETTER:
                options.append((position + 1, _NAME_TO_LETTER[word], False))
            if (word == "double" and position + 1 < len(words)
                    and words[position + 1] in {"you", "u"}
                    and words[position + 1] == spans[position + 1][2]
                    and allowed_word(words[position + 1])
                    and gap_allowed(source, spans, position, position + 1)):
                options.append((position + 2, "W", False))
            for following, added, glyph in options:
                if len(letters) + len(added) > _MAX_LETTERS:
                    continue
                if position != start and not gap_allowed(
                        source, spans, position - 1, position,
                        dotted=previous_glyph and glyph):
                    continue
                state = (following, letters + added, has_glyph or glyph, glyph)
                if state not in seen:
                    if len(seen) >= _MAX_STATES:
                        return ()
                    seen.add(state)
                    pending.append(state)
        return tuple(sorted(results))

    def matches(i, j):
        right = {}
        for end, letters, glyph in representations(1, j):
            right.setdefault(letters, []).append((end, glyph))
        emitted = set()
        for end_i, letters, left_glyph in representations(0, i):
            for end_j, right_glyph in right.get(letters, ()):
                if not (left_glyph or right_glyph) or (end_i, end_j, letters) in emitted:
                    continue
                emitted.add((end_i, end_j, letters))
                yield end_i, end_j, {
                    "operation": "letter_names", "rule": "exact_english_letter_names",
                    "expected_start": i, "expected_end": end_i,
                    "actual_start": j, "actual_end": end_j,
                    "expected": list(expected_words[i:end_i]),
                    "actual": list(actual_words[j:end_j]),
                    "matched_letters": list(letters),
                }

    def has_strong_run(side, start, end, *, oversized=False):
        source, words, spans = sides[side]
        if start < 0 or end > len(words):
            return False
        return any(
            _CONSONANT_RUN.fullmatch(source[spans[index][0]:spans[index][1]].casefold())
            and not _EXPRESSIVE_SOUND.fullmatch(words[index])
            and (not oversized or len(words[index]) > _MAX_LETTERS)
            for index in range(start, end)
        )

    class LetterNameMatcher:
        def __call__(self, i, j):
            return matches(i, j)

        def blocks_fuzzy(self, i, j, left_size, right_size):
            """Stop a fuzzy edge from erasing explicit spoken letter counts.

            Both complete spans must already be interpretable as letters and
            one must contain a strong compact initialism. Oversized consonant
            runs are also guarded, without becoming eligible for acceptance.
            Shouted ordinary names and expressive sounds remain lexical.
            """
            end_i, end_j = i + left_size, j + right_size
            left = {letters for end, letters, _ in representations(0, i) if end == end_i}
            right = {letters for end, letters, _ in representations(1, j) if end == end_j}
            if left.intersection(right):
                return False
            if (expected_words[i:end_i] == actual_words[j:end_j]
                    and i >= 0 and j >= 0):
                return False
            strong_left, strong_right = (has_strong_run(0, i, end_i),
                                         has_strong_run(1, j, end_j))
            # Longer runs are deliberately ineligible for acceptance, but an
            # unsupported length must not reopen a fuzzy shortcut that drops
            # their explicit repeated letters.
            if ((has_strong_run(0, i, end_i, oversized=True) and (right or strong_right))
                    or (has_strong_run(1, j, end_j, oversized=True) and (left or strong_left))):
                return True
            return bool(left and right and (strong_left or strong_right))

    return LetterNameMatcher()
