"""Soundalike names may pass; lost, added or reordered words must not."""

import builtins
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.phonetic_match import phonetic_equivalence, require_phonetic_matcher


class PhoneticMatchTests(unittest.TestCase):
    def test_names_and_spacing_match_in_both_directions_without_known_name_list(self):
        pairs = (
            ("Vandire", "van dyer"), ("Vandire", "van deer"),
            ("Derrins", "Darins"), ("Derrins", "Darren's"),
            ("Derrins", "Darin’s"), ("Qarm", "Khorm"),
            ("Qarm", "quorum"), ("Qarm", "Karim"),
            ("Mota", "Moda"), ("starforge", "star forge"),
            ("moonlightflower", "moon light flower"),
            ("flower", "flour"), ("knight", "night"),
        )
        for first, second in pairs:
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    report = phonetic_equivalence(f"Ask {expected} to wait.", f"Ask {actual} to wait.")
                    self.assertIsNotNone(report)
                    self.assertEqual(report["method"], "double_metaphone_v2")
                    self.assertEqual(report["implementation"], "Metaphone")
                    self.assertEqual(report["version"], "0.6")
                    self.assertEqual(len(report["alignment"]), 1)
                    edit = report["alignment"][0]
                    self.assertIn(edit["matched_key"], edit["expected_keys"])
                    self.assertIn(edit["matched_key"], edit["actual_keys"])
                    self.assertGreaterEqual(len(edit["matched_key"]), 2)
                    self.assertEqual(edit["expected_start"], 1)
                    self.assertEqual(edit["actual_start"], 1)

    def test_complete_ordered_passage_can_have_several_phonetic_variants(self):
        report = phonetic_equivalence(
            "Derrins asked Vandire to meet Qarm and Mota beside the flower.",
            "Darin's asked van dyer to meet Khorm and Moda beside the flour.")
        self.assertIsNotNone(report)
        self.assertEqual(len(report["alignment"]), 5)
        self.assertEqual([item["operation"] for item in report["alignment"]],
                         ["substitute", "split", "substitute", "substitute", "substitute"])

    def test_missing_added_repeated_or_reordered_content_is_not_ignored(self):
        pairs = (
            ("Vandire asked us to wait.", "Van dyer asked us."),
            ("Vandire asked us to wait.", "Van deer asked us to wait. We left."),
            ("Vandire asked us to wait. We left.", "Van deer asked us to wait."),
            ("Derrins greeted Mota.", "Darin's warmly greeted Moda."),
            ("Derrins warmly greeted Mota.", "Darin's greeted Moda."),
            ("Vandire waited.", "Van deer deer waited."),
            ("Mota asked Qarm.", "Khorm asked Moda."),
            ("The knight saw night.", "The night saw knight."),
            ("Derrins saw the knight before night.", "Darins saw the night before knight."),
            ("Vandire waits.", "Van uh dyer waits."),
            ("Vandire waits.", "Van oh dyer waits."),
            ("Vandire waits.", "Van I dyer waits."),
            ("Vandire waits.", "Van dyer yes waits."),
        )
        for expected, actual in pairs:
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_multiword_keys_retain_boundary_consonants_and_vowel_words(self):
        # Encoding the whole joined string would erase the real extra sound.
        for expected, actual in (("redder", "red dear"), ("Vandire", "van a dyer"),
                                 ("Vandire", "van ee dyer"), ("Vandire", "van w dyer")):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_function_words_cannot_be_absorbed_inside_a_phonetic_join(self):
        for expected, actual in (("into", "in to"), ("band", "be and"),
                                 ("overtaking", "over taking"), ("Theron", "the ron"),
                                 ("Hereward", "here ward"), ("Ismael", "is male")):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_unchanged_anchor_cannot_absorb_an_added_or_missing_word(self):
        # Mocked equal keys isolate the anchor invariant from this package's
        # current treatment of any particular added word's pronunciation.
        def keys(word):
            return ("AB", "B") if word == "name" else ("A", "")

        with patch("epub_tts_narrator.renderer.phonetic_match._load_double_metaphone", return_value=keys):
            for expected, actual in (("name", "word name"), ("word name", "name"),
                                     ("name", "name word"), ("name word", "name")):
                with self.subTest(expected=expected, actual=actual):
                    self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_numbers_and_negations_require_exact_lexical_agreement(self):
        pairs = (
            ("Vandire has four ships.", "Van deer has for ships."),
            ("Vandire has fifteen ships.", "Van deer has fifty ships."),
            ("Vandire won second place.", "Van deer won seckond place."),
            ("Vandire has two ships.", "Van deer has too ships."),
            ("Vandire has one ship.", "Van deer has won ship."),
            ("Vandire is not there.", "Van deer is knot there."),
            ("Vandire is not there.", "Van deer is there."),
            ("Vandire can go.", "Van deer can't go."),
            ("Vandire won't go.", "Van deer went go."),
            ("Vandire isn't there.", "Van deer isnt there."),
            ("Vandire has 12 ships.", "Van deer has 13 ships."),
        )
        for expected, actual in pairs:
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(phonetic_equivalence(expected, actual))
        self.assertIsNotNone(phonetic_equivalence("Vandire has 12 ships.", "Van deer has twelve ships."))

    def test_punctuation_non_ascii_or_unbounded_spans_are_not_joined(self):
        for expected, actual in (("Vandire", "van-dyer"), ("Vandire", "van, dyer"),
                                 ("Vandire", "van. Dyer"), ("Vandire", "van—dyer"),
                                 ("café", "cafe"), ("東京", "東亰"),
                                 ("sunmoonlightflower", "sun moon light flower"),
                                 ("aa", "ee"), ("I", "eye"), ("a", "ah")):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(phonetic_equivalence(expected, actual))
        self.assertIsNotNone(phonetic_equivalence("Vandire", "van\n\tdyer"))

    def test_complete_key_must_match_not_a_soundex_length_prefix(self):
        self.assertIsNone(phonetic_equivalence("Vandire", "Vandirestone"))
        self.assertIsNone(phonetic_equivalence("Christopher", "Christopherson"))

    def test_exact_empty_and_punctuation_only_inputs_do_not_claim_a_match(self):
        for expected, actual in (("", ""), ("...", "!"), ("Vandire", "VANDIRE!"),
                                 ("Vandire", ""), ("", "Vandire")):
            self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_input_and_ambiguous_search_are_bounded(self):
        for expected, actual in (("name " * 2001, "neme"), ("name", "neme " * 4001)):
            with self.assertRaisesRegex(ValueError, "per-chunk verification token limit"):
                phonetic_equivalence(expected, actual)
        with patch("epub_tts_narrator.renderer.phonetic_match._MAX_ALIGNMENT_STATES", 1):
            self.assertIsNone(phonetic_equivalence("Vandire", "Van dyer"))

    def test_missing_dependency_provides_actionable_error(self):
        original = builtins.__import__

        def importing(name, *args, **kwargs):
            if name == "metaphone":
                raise ModuleNotFoundError("metaphone")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, "uv pip install --python.*Metaphone==0.6"):
                phonetic_equivalence("Vandire", "Van dyer")
            with self.assertRaisesRegex(RuntimeError, "uv pip install --python.*Metaphone==0.6"):
                require_phonetic_matcher()


if __name__ == "__main__":
    unittest.main()
