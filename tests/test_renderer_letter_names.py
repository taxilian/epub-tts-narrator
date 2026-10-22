"""Letter spelling conventions preserve every spoken letter and its order."""

import unittest
import unicodedata
from unittest.mock import patch

from epub_tts_narrator.renderer.letter_names import make_letter_name_matcher
from epub_tts_narrator.renderer.phonetic_match import _PROTECTED
from epub_tts_narrator.renderer.quality import _lexeme_matches, normalize_words


def tokens(text):
    source = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    words, spans = [], []
    for match in _lexeme_matches(source):
        normalized = normalize_words(match.group())
        words.extend(normalized)
        spans.extend([(match.start(), match.end(), match.group().casefold())] * len(normalized))
    return source, words, spans


def full_match(expected, actual):
    left, right = tokens(expected), tokens(actual)
    edges = make_letter_name_matcher(*left, *right, protected=_PROTECTED)(0, 0)
    return next((change for i, j, change in edges
                 if i == len(left[1]) and j == len(right[1])), None)


class LetterNameTests(unittest.TestCase):
    def fuzzy_blocked(self, expected, actual):
        left, right = tokens(expected), tokens(actual)
        return make_letter_name_matcher(*left, *right, protected=_PROTECTED).blocks_fuzzy(
            0, 0, len(left[1]), len(right[1]))

    def test_letter_spelling_matches_exact_names_in_both_directions(self):
        pairs = (
            ("See-see?", " CC?"), ("See-see?", "Cc?"),
            ("see sea", "cc"), ("cee see", "C C"),
            ("see see", "C.C."), ("see see", "c-c"), ("sea tea", "CT"),
            ("bee be", "bb"), ("pea see", "pc"), ("cue are", "qr"),
            ("queue are", "Q R"), ("eye see you", "ICU"),
            ("double you double you", "WW"), ("double u why", "WY"),
            ("ay bee see dee ee ef", "ABCDEF"), ("zee zed", "ZZ"),
            ("see sea", "c see"), ("see sea", "sea C"), ("C see", "see C"),
        )
        for first, second in pairs:
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    audit = full_match(expected, actual)
                    self.assertIsNotNone(audit)
                    self.assertEqual(audit["operation"], "letter_names")
                    self.assertEqual(audit["rule"], "exact_english_letter_names")
                    self.assertGreaterEqual(len(audit["matched_letters"]), 2)

    def test_audit_retains_both_repeated_words_and_source_token_positions(self):
        left, right = tokens("Ask See-see? Now."), tokens("Ask CC? Now.")
        edges = list(make_letter_name_matcher(*left, *right, protected=_PROTECTED)(1, 1))
        self.assertEqual(edges, [(3, 2, {
            "operation": "letter_names", "rule": "exact_english_letter_names",
            "expected_start": 1, "expected_end": 3, "actual_start": 1, "actual_end": 2,
            "expected": ["see", "see"], "actual": ["cc"], "matched_letters": ["C", "C"],
        })])

    def test_missing_extra_reordered_or_repeated_letters_are_rejected(self):
        for expected, actual in (
            ("see see", "C"), ("see see", "CCC"), ("see see", "CC C"),
            ("see see", "see"), ("see see", "see see see"), ("see bee", "BC"),
            ("double you double you", "W"), ("double you double you", "you you"),
            ("see see", "CC thank you"), ("see see now", "CC"),
            ("see see now", "CC now now"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(full_match(expected, actual))

    def test_protected_numbers_and_negations_never_get_consumed(self):
        for expected, actual in (
            ("see see two", "CC too"), ("see see no", "CC know"),
            ("see see not", "CC"), ("see see 2", "CC two"),
            ("oh oh", "00"), ("one one", "II"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(full_match(expected, actual))
        left, right = tokens("see see"), tokens("CC")
        self.assertEqual(list(make_letter_name_matcher(*left, *right, protected={"see"})(0, 0)), [])

    def test_arbitrary_lowercase_words_and_singleton_homophones_are_not_letters(self):
        for expected, actual in (
            ("cat", "see ay tee"), ("arm", "ay are em"), ("in", "eye en"),
            ("why", "double you aitch why"), ("by", "bee why"),
            ("dry", "dee are why"), ("I", "eye"), ("a", "ay"),
            ("see", "C"), ("see sea", "sea see"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(full_match(expected, actual))

    def test_only_explicit_letter_spellings_have_dotted_boundaries(self):
        for expected, actual in (
            ("See. See.", "CC"), ("see, see", "CC"), ("see—see", "CC"),
            ("C, C", "see see"), ("C—C", "see see"), ("see.see", "CC"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(full_match(expected, actual))
        self.assertIsNotNone(full_match("see\nsee", "C. C."))
        self.assertIsNotNone(full_match("see‐see", "C‑C"))

    def test_runs_search_and_empty_indices_are_bounded(self):
        self.assertIsNone(full_match("see " * 7, "CCCCCCC"))
        self.assertIsNone(full_match("see " * 7, "C C C C C C C"))
        left, right = tokens("See-see"), tokens("CC")
        matcher = make_letter_name_matcher(*left, *right, protected=_PROTECTED)
        self.assertEqual(list(matcher(2, 1)), [])
        self.assertEqual(list(matcher(-1, 0)), [])
        with patch("epub_tts_narrator.renderer.letter_names._MAX_STATES", 1):
            self.assertIsNone(full_match("see see", "CC"))

    def test_fuzzy_guards_preserve_compact_letter_order_and_counts(self):
        pairs = (
            ("PC", "PCC"), ("pc", "pcc"), ("CC", "CCC"), ("cc", "cccc"),
            ("CC bee", "CB"), ("CC bee", "CBB"), ("CC bee", "CCBB"),
            ("PC", "CP"), ("BB", "BBB"), ("CC", "see see see"),
            ("PC", "PCCCCCC"), ("PCCCCCC", "PCCCCCCC"),
            ("pee see", "PCCCCCC"),
        )
        for first, second in pairs:
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertTrue(self.fuzzy_blocked(expected, actual))

    def test_fuzzy_guards_do_not_reclassify_ordinary_names_or_hums(self):
        for expected, actual in (
            ("MOTA", "MODA"), ("Derrins", "Darins"), ("hmm", "hmmm"),
            ("HMM", "hmmm"), ("mm", "mmm"), ("AA", "AAA"),
            ("shh", "shhhh"), ("brr", "brrrr"), ("psst", "pssst"),
            ("tsk", "tsssk"), ("ss", "sssss"), ("HMM", "hmmmmmmmm"),
            ("PCCCCCC", "PCCCCCC"),
            ("CC", "see see"), ("CC bee", "CCB"), ("CC", "C C"),
            ("See-see", "CC"), ("unknown word", "CC"), ("CC", ""),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertFalse(self.fuzzy_blocked(expected, actual))

    def test_fuzzy_guard_checks_only_the_proposed_complete_span(self):
        left, right = tokens("Ask CC bee now"), tokens("Ask CB now")
        matcher = make_letter_name_matcher(*left, *right, protected=_PROTECTED)
        self.assertTrue(matcher.blocks_fuzzy(1, 1, 2, 1))
        self.assertFalse(matcher.blocks_fuzzy(1, 1, 3, 2))
        self.assertFalse(matcher.blocks_fuzzy(-1, 0, 2, 2))
        self.assertFalse(matcher.blocks_fuzzy(4, 3, 1, 1))


if __name__ == "__main__":
    unittest.main()
