import unittest
from unittest.mock import patch

from audiobook_renderer import quality
from audiobook_renderer.common import digest
from audiobook_renderer.quality import compare_reference_words, compare_words, validation_key
import test_renderer_reference_consensus as consensus


SCRIPT = (
    "I am Master Talim of the Sparkling Sky Sect, and judging by your reaction "
    "I suspect you are familiar with us. May I please apologize for this "
    "incident, and speak with the master of your Sourcepalace?"
)
ACTUAL = SCRIPT.replace("Sourcepalace", "Source Palace")


class ReferenceBoundaryTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_talim_split_requires_independent_agreement_and_preserves_raw_diagnostics(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], ACTUAL)
        self.assertTrue(report["reference_text_adjusted"])
        original = compare_words(SCRIPT, ACTUAL)
        self.assertEqual(original["expected_word_count"], 36)
        self.assertEqual(original["insertions"], 1)
        self.assertEqual(original["substitutions"], 1)
        for key, value in original.items():
            self.assertEqual(report["metrics"][key], value)
        self.assertEqual(report["reference_comparison"], compare_reference_words(SCRIPT, ACTUAL))
        self.assertEqual(report["corroboration"]["decision"], "agreed_reference_transcript")
        eligibility = report["corroboration"]["eligibility"]
        self.assertEqual(eligibility["substitutions"], 0)
        self.assertEqual(eligibility["boundary_changes"], 1)
        self.assertEqual(eligibility["variation_count"], 1)
        self.assertTrue(eligibility["alignment"])
        self.assertEqual(verifier.calls, [
            ("reference", self.audio), ("corroboration", self.audio)])

    def test_bounded_two_or_three_part_split_and_join_can_be_corroborated(self):
        for source, actual in (
            ("moonstone", "moon stone"),
            ("moon stone", "moonstone"),
            ("moonlightstone", "moon light stone"),
            ("moon light stone", "moonlightstone"),
        ):
            with self.subTest(source=source, actual=actual):
                expected = SCRIPT.replace("Sourcepalace", source)
                spoken = SCRIPT.replace("Sourcepalace", actual)
                verifier = consensus.IndependentVerifier(primary=spoken, secondary=spoken)
                report = self.check(verifier, expected=expected)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["reference_text"], spoken)
                self.assertEqual(report["corroboration"]["eligibility"]["boundary_changes"], 1)
                self.assertEqual(len(verifier.calls), 2)

    def test_missing_or_failed_secondary_never_accepts_boundary_variation(self):
        primary_only = consensus.PrimaryVerifier(ACTUAL)
        report = self.check(primary_only, expected=SCRIPT)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "unavailable")
        self.assertNotIn("reference_text", report)
        for result in (RuntimeError("decoder unavailable"), None,
                       {"text": ACTUAL, "segments": []}):
            with self.subTest(result=result):
                verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
                verifier.secondary = result
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["corroboration"]["decision"], "verification_failed")
                self.assertNotIn("reference_text", report)

    def test_disagreeing_secondary_does_not_accept_boundary_variation(self):
        verifier = consensus.IndependentVerifier(
            primary=ACTUAL, secondary=ACTUAL.replace("familiar", "unfamiliar"))
        report = self.check(verifier, expected=SCRIPT)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "disagreed")
        self.assertNotIn("reference_text", report)

    def test_exact_secondary_requested_script_can_resolve_boundary_variation(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=SCRIPT)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "secondary_matches_requested")
        self.assertEqual(report["reference_text"], SCRIPT)
        self.assertFalse(report["reference_text_adjusted"])
        self.assertEqual(report["transcript"], ACTUAL)

    def test_boundary_eligibility_does_not_define_global_spelling_equivalence(self):
        for expected, actual in (("Sourcepalace", "Source Palace"),
                                 ("nowhere", "now here"),
                                 ("cannot", "can not"),
                                 ("ice cream", "icecream")):
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_reference_words(expected, actual)["wer"], 0)
        expected = SCRIPT + " We found nowhere to rest."
        primary = expected.replace("Sourcepalace", "Source Palace")
        secondary = primary.replace("nowhere", "now here")
        verifier = consensus.IndependentVerifier(primary=primary, secondary=secondary)
        report = self.check(verifier, expected=expected)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "disagreed")
        self.assertGreater(report["corroboration"]["primary_comparison"]["wer"], 0)
        self.assertGreater(report["corroboration"]["requested_comparison"]["wer"], 0)

    def test_real_missing_or_added_words_still_skip_secondary(self):
        for actual in (ACTUAL + " Again.", ACTUAL.replace("familiar with", "familiar"),
                       SCRIPT.replace("Sourcepalace", "Source Palaces")):
            with self.subTest(actual=actual):
                verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["corroboration"]["decision"], "mismatch_outside_reference_limits")
                self.assertEqual(verifier.calls, [("reference", self.audio)])

    def test_short_parts_numbers_apostrophes_symbols_and_four_parts_are_not_boundaries(self):
        for source, actual in (
            ("around", "a round"),
            ("twentyone", "twenty one"),
            ("onehundred", "one hundred"),
            ("can'not", "can not"),
            ("moon$stone", "moonstone"),
            ("moonlightstonepath", "moon light stone path"),
        ):
            with self.subTest(source=source, actual=actual):
                expected = SCRIPT.replace("Sourcepalace", source)
                spoken = SCRIPT.replace("Sourcepalace", actual)
                verifier = consensus.IndependentVerifier(primary=spoken, secondary=spoken)
                report = self.check(verifier, expected=expected)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["corroboration"]["decision"], "mismatch_outside_reference_limits")
                self.assertEqual(verifier.calls, [("reference", self.audio)])

    def test_raw_word_error_does_not_double_count_a_boundary_only_change(self):
        for word_count in (20, 19):
            with self.subTest(word_count=word_count):
                expected = " ".join(["steady"] * (word_count - 1) + ["Sourcepalace"])
                actual = expected.replace("Sourcepalace", "Source Palace")
                verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=expected)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["metrics"]["wer"], 2 / word_count)
                self.assertEqual(len(verifier.calls), 2)

    def test_two_compound_splits_above_ten_percent_require_full_independent_agreement(self):
        expected = (
            "The bloodbeast waited quietly just outside the station while the captain checked every locked "
            "door and window. A sourcevow bound the traveler to remain nearby until the night train "
            "arrived with fresh supplies for the village."
        )
        actual = expected.replace("bloodbeast", "blood beast").replace("sourcevow", "source vow")
        for secondary, accepted in ((actual, True), (expected, True),
                                    (actual.replace("quietly", "quickly"), False)):
            with self.subTest(secondary=secondary):
                verifier = consensus.IndependentVerifier(primary=actual, secondary=secondary)
                report = self.check(verifier, expected=expected)
                self.assertEqual(report["passed"], accepted, report)
                self.assertEqual(report["metrics"]["expected_word_count"], 36)
                self.assertEqual(report["metrics"]["wer"], 4 / 36)
                self.assertEqual(report["corroboration"]["eligibility"]["boundary_changes"], 2)
                self.assertEqual(report["corroboration"]["eligibility"]["substitutions"], 0)
                self.assertEqual(verifier.calls, [("reference", self.audio), ("corroboration", self.audio)])
                if not accepted:
                    self.assertNotIn("reference_text", report)
        primary_only = consensus.PrimaryVerifier(actual)
        report = self.check(primary_only, expected=expected)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "unavailable")

    def test_short_direct_substrings_and_unrelated_thank_you_never_gain_boundary_eligibility(self):
        for expected, actual in (
            ("The bloodbeast waits.", "The bloodbeast"),
            ("A bloodbeast waits.", "bloodbeast waits."),
            ("The sourcevow remains binding.", "The sourcevow remains"),
            ("The bloodbeast waits.", "The blood beast"),
            ("The only fellow who could hear the sourcevow was the bloodbeast.", "Thank you."),
            ("The only fellow who could hear the sourcevow was the bloodbeast.",
             "Thank you who could hear the source vow was the blood beast."),
        ):
            with self.subTest(expected=expected, actual=actual):
                verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=expected)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["corroboration"]["decision"], "mismatch_outside_reference_limits")
                self.assertEqual(verifier.calls, [("reference", self.audio)])

    def test_boundary_only_exception_never_relaxes_book_speech(self):
        expected = "The bloodbeast heard the sourcevow."
        actual = "The blood beast heard the source vow."
        verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
        report = self.check(verifier, expected=expected, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertNotIn("corroboration", report)
        self.assertEqual(verifier.calls, [("speech", self.audio)])

    def test_high_wer_boundary_only_changes_still_obey_four_change_limit(self):
        for count in (4, 5):
            with self.subTest(count=count):
                expected = " ".join(["Sourcepalace"] * count)
                actual = expected.replace("Sourcepalace", "Source Palace")
                verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=expected)
                self.assertEqual(report["passed"], count == 4, report)
                self.assertEqual(len(verifier.calls), 2 if count == 4 else 1)

    def test_four_boundary_changes_can_be_corroborated_but_five_cannot(self):
        for count in (4, 5):
            with self.subTest(count=count):
                expected = " ".join(["steady"] * 100 + ["Sourcepalace"] * count)
                actual = expected.replace("Sourcepalace", "Source Palace")
                verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=expected)
                self.assertLessEqual(report["metrics"]["wer"], 0.1)
                self.assertEqual(report["passed"], count == 4, report)
                if count == 4:
                    self.assertEqual(report["corroboration"]["eligibility"]["variation_count"], count)
                else:
                    self.assertEqual(report["corroboration"]["decision"], "mismatch_outside_reference_limits")
                self.assertEqual(len(verifier.calls), 2 if count == 4 else 1)

    def test_substitutions_and_boundaries_share_the_four_variation_limit(self):
        expected = " ".join(["steady"] * 100 + ["Sourcepalace", "quiet", "bright", "warm", "calm"])
        for count in (3, 4):
            with self.subTest(substitutions=count):
                actual = expected.replace("Sourcepalace", "Source Palace")
                for source, replacement in list(zip(
                        ("quiet", "bright", "warm", "calm"),
                        ("silent", "clear", "cool", "still")))[:count]:
                    actual = actual.replace(source, replacement)
                verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=expected)
                self.assertEqual(report["passed"], count == 3, report)
                if count == 3:
                    eligibility = report["corroboration"]["eligibility"]
                    self.assertEqual(eligibility["substitutions"], count)
                    self.assertEqual(eligibility["boundary_changes"], 1)
                    self.assertEqual(eligibility["variation_count"], count + 1)
                else:
                    self.assertEqual(report["corroboration"]["decision"], "mismatch_outside_reference_limits")
                self.assertEqual(len(verifier.calls), 2 if count == 3 else 1)

    def test_book_speech_remains_strict_for_the_same_talim_recording(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
        report = self.check(verifier, expected=SCRIPT, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertNotIn("corroboration", report)
        self.assertNotIn("reference_comparison", report)
        self.assertEqual(verifier.calls, [("speech", self.audio)])

    def test_new_reference_policy_invalidates_reference_cache_but_keeps_speech_cache(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
        legacy = {
            "policy_version": quality.QUALITY_VERSION,
            "quality": quality.validate_policy(None),
            "verifier": verifier.identity,
            "expected_text": SCRIPT,
        }
        speech_key = validation_key(SCRIPT, verifier, purpose="speech")
        reference_key = validation_key(SCRIPT, verifier, purpose="reference")
        self.assertEqual(speech_key, digest({**legacy, "purpose": "speech"}))
        self.assertNotEqual(reference_key, digest({**legacy, "purpose": "reference"}))
        with patch.object(quality, "REFERENCE_VARIATION_VERSION", quality.REFERENCE_VARIATION_VERSION + 1):
            self.assertEqual(validation_key(SCRIPT, verifier, purpose="speech"), speech_key)
            self.assertNotEqual(validation_key(SCRIPT, verifier, purpose="reference"), reference_key)


if __name__ == "__main__":
    unittest.main()
