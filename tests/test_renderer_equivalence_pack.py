"""Historical near-miss mismatches reach review without weakening coverage."""

import unittest

from epub_tts_narrator.renderer.phonetic_match import phonetic_equivalence
from epub_tts_narrator.renderer.phonetic_review import substitution_pairs
from epub_tts_narrator.renderer.quality import word_spacing_equivalence
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
import test_renderer_reference_consensus as fixtures
from test_renderer_jev_verification import Judge


class EquivalencePackCoverageTests(unittest.TestCase):
    def assert_review(self, expected, actual, rule):
        self.assertIsNone(phonetic_equivalence(expected, actual))
        coverage = speech_coverage(expected, actual)
        self.assertTrue(coverage["passed"], coverage)
        self.assertEqual(coverage["reason"], "complete_reviewable_coverage")
        self.assertEqual(len(coverage["alignment"]), 1, coverage)
        change = coverage["alignment"][0]
        self.assertEqual(change["operation"], "pronunciation_candidate")
        self.assertEqual(change["rule"], rule)
        pairs = substitution_pairs(expected, actual)
        self.assertIsNotNone(pairs, coverage)
        self.assertEqual([pair["coverage_rule"] for pair in pairs], [rule])

    def assert_rejected(self, expected, actual):
        self.assertIsNone(phonetic_equivalence(expected, actual))
        self.assertFalse(speech_coverage(expected, actual)["passed"])
        self.assertIsNone(substitution_pairs(expected, actual))

    def test_title_abbreviations_reach_sound_review_in_both_directions(self):
        pairs = (
            ("MISTER Wolf", "mr Wolf"),
            ("Doctor", "Dr."),
            ("Professor", "Prof."),
            ("Saint", "St."),
            ("Captain", "Capt."),
            ("Sergeant", "Sgt."),
        )
        for full, abbreviation in pairs:
            for expected, actual in ((full, abbreviation), (abbreviation, full)):
                with self.subTest(expected=expected, actual=actual):
                    self.assert_review(expected, actual, "title_abbreviation")

    def test_title_abbreviations_reject_other_missing_and_extra_words(self):
        for expected, actual in (
            ("Mister Wolf", "Dr. Wolf"),
            ("Mister Wolf", "Mr."),
            ("Mister", "Mr. Wolf"),
            ("Mister Wolf", "Mr. Wolf arrives"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assert_rejected(expected, actual)

    def test_single_letter_homoglyph_reaches_sound_review_in_both_directions(self):
        for expected, actual in (("I", "l"), ("l", "I")):
            with self.subTest(expected=expected, actual=actual):
                self.assert_review(expected, actual, "single_letter_homoglyph")

    def test_single_letter_homoglyph_rejects_longer_different_missing_and_extra_words(self):
        for expected, actual in (
            ("ill", "i"),
            ("l", "it"),
            ("l", "t"),
            ("I left", "l"),
            ("I", "l left"),
            ("I know l", "l know i"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assert_rejected(expected, actual)

    def test_colloquial_expansions_reach_sound_review_in_both_directions(self):
        pairs = (
            ("kinda", "kind of"),
            ("gonna", "going to"),
            ("wanna", "want to"),
            ("d'you", "do you"),
            ("'cause", "because"),
            ("lemme", "let me"),
            ("gimme", "give me"),
            ("gotcha", "got you"),
        )
        for colloquial, expansion in pairs:
            for expected, actual in ((colloquial, expansion), (expansion, colloquial)):
                with self.subTest(expected=expected, actual=actual):
                    self.assert_review(expected, actual, "colloquial_expansion")

    def test_colloquial_expansions_reject_partial_different_missing_and_extra_words(self):
        for expected, actual in (
            ("gonna", "going"),
            ("kinda", "sort of"),
            ("cause", "because"),
            ("kinda left", "kind of"),
            ("kinda", "kind of left"),
            ("gotcha", "got you got you"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assert_rejected(expected, actual)

    def test_leading_zero_digit_strings_are_canonically_equal_in_both_directions(self):
        pairs = (
            ("0800", "800"),
            ("0930", "930"),
            ("At 0800 hours.", "At 800 hours."),
            ("000", "0"),
            ("$0800", "$800"),
        )
        for first, second in pairs:
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertIsNotNone(word_spacing_equivalence(expected, actual))
                    coverage = speech_coverage(expected, actual)
                    self.assertTrue(coverage["passed"], coverage)
                    self.assertEqual(coverage["reason"], "exact_spelling_boundaries")

    def test_leading_zero_rule_preserves_all_other_numeric_and_coverage_guards(self):
        for expected, actual in (
            ("0800", "900"),
            ("0800", "0801"),
            ("0800 800", "800 0800"),
            ("-0800", "-800"),
            ("+0800", "+800"),
            ("0800.0", "800.0"),
            ("$0800", "800"),
            ("$0800", "€800"),
            ("At 0800", "800"),
            ("0800", "800 hours"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(word_spacing_equivalence(expected, actual))
                self.assertFalse(speech_coverage(expected, actual)["passed"])

    def test_compound_spacing_remains_canonically_equal_in_both_directions(self):
        for first, second in (("Moonbase", "Moon Base"), ("half-blade", "halfblade")):
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    comparison = word_spacing_equivalence(expected, actual)
                    self.assertIsNotNone(comparison)
                    self.assertTrue(comparison["alignment"])
                    coverage = speech_coverage(expected, actual)
                    self.assertTrue(coverage["passed"], coverage)
                    self.assertEqual(coverage["reason"], "exact_spelling_boundaries")
        self.assertIsNone(word_spacing_equivalence("Moonbase", "Moon Bases"))
        self.assertFalse(speech_coverage("Moonbase", "Moon")["passed"])


class EquivalencePackAudioTests(unittest.TestCase):
    setUp = fixtures.ReferenceConsensusTests.setUp
    write_audio = fixtures.ReferenceConsensusTests.write_audio
    check = fixtures.ReferenceConsensusTests.check

    def test_new_nominations_require_an_affirmative_judgment(self):
        for expected, heard in (("Mister Wolf", "Mr. Wolf"), ("I", "l"), ("kinda", "kind of")):
            with self.subTest(expected=expected, heard=heard, judge="absent"):
                report = self.check(fixtures.PrimaryVerifier(heard), expected=expected, purpose="speech")
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
            for decision, approved in (("different", False), ("equivalent", True)):
                with self.subTest(expected=expected, heard=heard, decision=decision):
                    report = self.check(
                        ReferenceTextJudgingVerifier(fixtures.PrimaryVerifier(heard), Judge(decision)),
                        expected=expected, purpose="speech")
                    self.assertEqual(report["passed"], approved, report)
                    if approved:
                        self.assertEqual(report["text_confirmation_method"], "jev_primary")

    def test_leading_zero_equality_uses_the_existing_written_confirmation_path(self):
        judge = Judge("unavailable")
        report = self.check(
            ReferenceTextJudgingVerifier(fixtures.PrimaryVerifier("800"), judge),
            expected="0800", purpose="speech")
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["text_confirmation_method"], "word_spacing")
        self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()
