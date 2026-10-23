"""Compound-word reference comparisons share fixtures from test_renderer_reference_consensus."""

import unittest

from epub_tts_narrator.renderer.quality import compare_reference_words, compare_words
import test_renderer_reference_consensus as consensus


IndependentVerifier = consensus.IndependentVerifier


SCRIPT = (
    "We will remain hidden until we are strong enough to defend ourselves. "
    "To do otherwise will risk both Vessa and the pack-mate we have back on Earth. "
    "Do you want Vessa to be alone?"
)


class ReferenceCompoundComparisonTests(unittest.TestCase):
    def test_explicit_source_compound_accepts_only_its_own_word_boundaries(self):
        for separator in ("-", "\u2010", "\u2011"):
            for actual in ("packmate", "pack mate", "PACK-MATE"):
                with self.subTest(separator=separator, actual=actual):
                    comparison = compare_reference_words("pack" + separator + "mate", actual)
                    self.assertEqual(comparison["wer"], 0)
                    self.assertEqual(comparison["source_compounds"], [["pack", "mate"]])
        self.assertGreater(compare_words("pack-mate", "packmate")["wer"], 0)
        self.assertGreater(
            compare_reference_words("My pack-mate likes ice cream.", "My packmate likes icecream.")["wer"],
            0,
        )

    def test_numeric_words_retain_existing_normalization(self):
        comparison = compare_reference_words(
            "Twenty-one pack-mates met 15 people.",
            "21 packmates met fifteen people.")
        self.assertEqual(comparison["wer"], 0)
        self.assertEqual(comparison["source_compounds"], [["pack", "mates"]])

    def test_other_separators_and_partial_identifier_matches_do_not_authorize_joining(self):
        for source, actual in (
            ("cannot", "can not"),
            ("can not", "cannot"),
            ("ice cream", "icecream"),
            ("pack—mate", "packmate"),
            ("pack–mate", "packmate"),
            ("pack−mate", "packmate"),
            ("pack\u00admate", "packmate"),
            ("pack－mate", "packmate"),
            ("pack--mate", "packmate"),
            ("x-y", "xy"),
            ("3-2", "32"),
            ("abc2-pack-mate", "abc2packmate"),
            ("pack-mate9", "packmate9"),
            ("pack-mate's", "packmates"),
        ):
            with self.subTest(source=source, actual=actual):
                comparison = compare_reference_words(source, actual)
                self.assertGreater(comparison["wer"], 0)
                self.assertEqual(comparison["source_compounds"], [])

    def test_secondary_comparison_cannot_invent_compound_rules(self):
        self.assertGreater(compare_reference_words(
            "ice-cream", "icecream", source_text="I ate ice cream.")["wer"], 0)
        self.assertEqual(compare_reference_words(
            "packmate", "pack mate", source_text="My pack-mate arrived.")["wer"], 0)

    def test_compounds_do_not_erase_real_name_contraction_or_missing_word_errors(self):
        for actual in (
            SCRIPT.replace("Vessa", "Vesa"),
            SCRIPT.replace("otherwise will", "otherwise we'll"),
            SCRIPT.replace("strong enough", "strong"),
        ):
            with self.subTest(actual=actual):
                self.assertGreater(compare_reference_words(SCRIPT, actual)["wer"], 0)


class ReferenceCompoundValidationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_compound_join_matching_source_boundaries_is_accepted_like_speech(self):
        actual = SCRIPT.replace("pack-mate", "packmate")
        verifier = IndependentVerifier(primary=actual, secondary=actual)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertGreater(report["metrics"]["wer"], 0)
        self.assertEqual(report["reference_comparison"]["wer"], 0)
        self.assertEqual(report["text_confirmation_method"], "word_spacing")
        self.assertEqual(report["reference_text"], actual)
        self.assertTrue(report["reference_text_adjusted"])
        self.assertNotIn("corroboration", report)
        self.assertEqual(verifier.calls, [("reference", self.audio)])

    def test_spacing_matched_primary_needs_no_secondary_agreement(self):
        actual = SCRIPT.replace("pack-mate", "packmate")
        verifier = IndependentVerifier(primary=actual, secondary=actual.replace("Earth", "Mars"))
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], actual)
        self.assertNotIn("corroboration", report)
        self.assertEqual(verifier.calls, [("reference", self.audio)])

    def test_speech_applies_the_same_spacing_equivalence_as_references(self):
        actual = SCRIPT.replace("pack-mate", "packmate")
        verifier = IndependentVerifier(primary=actual, secondary=actual)
        report = self.check(verifier, expected=SCRIPT, purpose="speech")
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["text_confirmation_method"], "word_spacing")
        self.assertNotIn("reference_comparison", report)
        self.assertNotIn("reference_text", report)
        self.assertNotIn("corroboration", report)
        self.assertEqual(verifier.calls, [("speech", self.audio)])

    def test_three_real_jasper_transcripts_only_accept_fully_corroborated_third_take(self):
        first_primary = SCRIPT.replace("otherwise will", "otherwise we'll").replace("pack-mate", "packmate")
        later_primary = first_primary.replace("Vessa", "Vesa")
        cases = [
            (first_primary, first_primary.replace("packmate", "pac-mate"), False),
            (later_primary, later_primary.replace("otherwise we'll", "otherwise will"), False),
            (later_primary, later_primary, True),
        ]
        for primary, secondary, passed in cases:
            with self.subTest(primary=primary, secondary=secondary):
                verifier = IndependentVerifier(primary=primary, secondary=secondary)
                report = self.check(verifier, expected=SCRIPT)
                self.assertEqual(report["passed"], passed, report)
                self.assertEqual(verifier.calls, [
                    ("reference", self.audio), ("corroboration", self.audio)])
                self.assertGreater(report["metrics"]["deletions"], 0)
                self.assertEqual(report["reference_comparison"]["deletions"], 0)
                if passed:
                    self.assertEqual(report["reference_text"], primary)
                    self.assertEqual(report["reference_comparison"]["substitutions"], 3)
                    self.assertEqual(report["corroboration"]["decision"], "agreed_reference_transcript")
                else:
                    self.assertEqual(report["corroboration"]["decision"], "disagreed")
                    self.assertNotIn("reference_text", report)

    def test_four_substitutions_at_ten_percent_can_be_corroborated_but_five_cannot(self):
        expected = "one two three four five six seven eight nine ten " * 5
        for count in (4, 5):
            actual = expected.replace("one", "eleven", count)
            verifier = IndependentVerifier(primary=actual, secondary=actual)
            report = self.check(verifier, expected=expected)
            self.assertEqual(report["passed"], count == 4, report)
            self.assertEqual(
                len(verifier.calls), 2 if count == 4 else 1)


if __name__ == "__main__":
    unittest.main()
