import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import quality
from epub_tts_narrator.renderer.quality import compare_words, validation_key
import test_renderer_reference_consensus as consensus


SCRIPT = (
    "You sshould reconssider. I know sshe iss ssweet. I know sshe iss tassty. "
    "I know you do not wissh to sshare."
)
ACTUAL = (
    "You should reconsider. I know she is sweet. I know she is tasty. "
    "I know you do not wish to share."
)


class ReferenceStylizationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def assert_skips_secondary(self, expected, actual):
        verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
        report = self.check(verifier, expected=expected)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "mismatch_outside_reference_limits")
        self.assertEqual(verifier.calls, [("reference", self.audio)])
        self.assertNotIn("reference_text", report)
        return report

    def test_actual_hissing_script_requires_independent_agreement_and_keeps_evidence(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["expected_text"], SCRIPT)
        self.assertEqual(report["raw_transcript"], ACTUAL)
        self.assertEqual(report["transcript"], ACTUAL)
        self.assertEqual(report["reference_text"], ACTUAL)
        self.assertTrue(report["reference_text_adjusted"])
        raw = compare_words(SCRIPT, ACTUAL)
        self.assertGreater(raw["wer"], .1)
        for name, value in raw.items():
            self.assertEqual(report["metrics"][name], value)
        eligibility = report["corroboration"]["eligibility"]
        self.assertEqual(eligibility["method"], "repeated_consonant_stylization_v1")
        self.assertEqual(eligibility["letter"], "s")
        self.assertEqual(eligibility["stylized_word_count"], 10)
        self.assertEqual(set(eligibility["distinct_stylized_words"]), {
            "sshould", "reconssider", "sshe", "iss", "ssweet", "tassty", "wissh", "sshare",
        })
        self.assertEqual(len(eligibility["alignment"]), 10)
        self.assertEqual(report["corroboration"]["decision"], "agreed_reference_transcript")
        self.assertEqual(report["corroboration"]["primary_comparison"]["wer"], 0)
        self.assertEqual(verifier.calls, [
            ("reference", self.audio), ("corroboration", self.audio)])

    def test_no_secondary_cannot_accept_stylized_reference(self):
        verifier = consensus.PrimaryVerifier(ACTUAL)
        report = self.check(verifier, expected=SCRIPT)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "unavailable")
        self.assertNotIn("reference_text", report)

    def test_failed_or_malformed_secondary_cannot_accept_stylized_reference(self):
        for secondary in (RuntimeError("decoder unavailable"), None,
                          {"text": ACTUAL, "segments": []}):
            with self.subTest(secondary=secondary):
                verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
                verifier.secondary = secondary
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["corroboration"]["decision"], "verification_failed")
                self.assertNotIn("reference_text", report)

    def test_secondary_must_agree_on_every_word_without_stylization_equivalence(self):
        for secondary in (ACTUAL.replace("sweet", "kind"),
                          ACTUAL.replace("sweet", "ssweet"),
                          ACTUAL + " Again.",
                          ACTUAL.replace("not ", "")):
            with self.subTest(secondary=secondary):
                verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=secondary)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["corroboration"]["decision"], "disagreed")
                self.assertNotIn("reference_text", report)
                self.assertEqual(len(verifier.calls), 2)

    def test_real_substitution_cannot_hide_among_patterned_changes(self):
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("sweet", "kind"))
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("not", "now"))

    def test_added_or_deleted_words_cannot_hide_among_patterned_changes(self):
        self.assert_skips_secondary(SCRIPT, ACTUAL + " Again.")
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("not ", ""))
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("not wish", "wish not"))

    def test_missing_nonrepeated_letters_or_whole_runs_are_not_style(self):
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("should", "shold"))
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("share", "hare"))
        self.assert_skips_secondary(SCRIPT, ACTUAL.replace("sweet", "wet"))

    def test_mixed_consonant_shortening_does_not_establish_a_single_pattern(self):
        self.assert_skips_secondary(SCRIPT.replace("know", "knnow"), ACTUAL)

    def test_repeating_one_word_is_not_enough_evidence_of_style(self):
        self.assert_skips_secondary("Sshe sshe sshe sshe.", "She she she she.")
        self.assert_skips_secondary("Sshe iss sshe iss.", "She is she is.")

    def test_varied_stretching_of_one_word_is_not_three_distinct_words(self):
        expected, actual = "Sshe ssshe sssshe.", "She she she."
        self.assertGreater(compare_words(expected, actual)["wer"], .1)
        self.assert_skips_secondary(expected, actual)

    def test_natural_doubled_words_do_not_authorize_blanket_normalization(self):
        for expected, actual in (
            ("The dessert is delicious.", "The desert is delicious."),
            ("The cannon is ready.", "The canon is ready."),
            ("Dessert assess mess.", "Desert ases mes."),
            ("Cannon manner dinner.", "Canon maner diner."),
            ("Llama ill still.", "Lama il stil."),
        ):
            with self.subTest(expected=expected):
                self.assert_skips_secondary(expected, actual)

    def test_vowels_are_not_consonant_stylization(self):
        self.assert_skips_secondary("Oopen dooor soooon.", "Open dor son.")

    def test_triple_consonant_anchor_can_preserve_natural_double_letters(self):
        expected = "Bbbold rabbbits stabbbed."
        actual = "Bold rabbits stabbed."
        verifier = consensus.IndependentVerifier(primary=actual, secondary=actual)
        report = self.check(verifier, expected=expected)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], actual)
        self.assertEqual(report["corroboration"]["eligibility"]["letter"], "b")
        self.assertEqual(report["corroboration"]["decision"], "agreed_reference_transcript")

    def test_number_words_cannot_be_reinterpreted_as_stylization(self):
        self.assert_skips_secondary("Oonne tthhree ffoour.", "One three four.")
        self.assert_skips_secondary("Nnine nninety nnineteen.", "Nine ninety nineteen.")
        self.assert_skips_secondary("Tttwo tttwenty tttwelve.", "Two twenty twelve.")

    def test_punctuation_within_words_and_unicode_do_not_expand_eligibility(self):
        self.assert_skips_secondary(SCRIPT.replace("ssweet", "s'sweet"), ACTUAL)
        self.assert_skips_secondary("Šše išš ššweet.", "Še iš šweet.")

    def test_shortening_is_one_way_and_does_not_authorize_elongated_actual_words(self):
        self.assert_skips_secondary(ACTUAL, SCRIPT)

    def test_book_speech_retains_exact_lexical_gate_for_same_recording(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
        report = self.check(verifier, expected=SCRIPT, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertNotIn("corroboration", report)
        self.assertNotIn("reference_text", report)
        self.assertEqual(verifier.calls, [("speech", self.audio)])

    def test_style_policy_version_invalidates_only_reference_verification(self):
        verifier = consensus.IndependentVerifier(primary=ACTUAL, secondary=ACTUAL)
        speech_key = validation_key(SCRIPT, verifier, purpose="speech")
        reference_key = validation_key(SCRIPT, verifier, purpose="reference")
        with patch.object(quality, "REFERENCE_VARIATION_VERSION", 2):
            self.assertEqual(validation_key(SCRIPT, verifier, purpose="speech"), speech_key)
            self.assertNotEqual(validation_key(SCRIPT, verifier, purpose="reference"), reference_key)


if __name__ == "__main__":
    unittest.main()
