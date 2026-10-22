"""Spacing-only ASR differences must not waste good speech takes or JEV calls."""

import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.quality import validation_key, word_spacing_equivalence
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, SpeechVerificationFailed
import test_renderer_reference_consensus as consensus
from test_renderer_jev_verification import Judge
import test_renderer_speech_retry as retry
from test_renderer_workflow import FFMPEG, FFPROBE


class WordSpacingComparisonTests(unittest.TestCase):
    def test_split_and_join_preserve_every_normalized_letter_and_record_boundaries(self):
        pairs = (
            ("We were over taking the cart.", "We were overtaking the cart."),
            ("Enter the sourcepalace.", "Enter the source palace."),
            ("The moon\nlight snow flake drifted.", "The moonlight snowflake drifted."),
            ("A sunlightflower.", "A sun light flower."),
            ("The caféhouse opens.", "The café house opens."),
        )
        for first, second in pairs:
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    result = word_spacing_equivalence(expected, actual)
                    self.assertIsNotNone(result)
                    self.assertGreater(result["boundary_changes"], 0)
                    for edit in result["alignment"]:
                        self.assertEqual("".join(edit["expected"]), "".join(edit["actual"]))
                        self.assertIn(edit["operation"], {"split", "join"})

    def test_cannot_hide_missing_repeated_changed_reordered_or_numeric_content(self):
        pairs = (
            ("We were over taking the cart.", "We were overtaking."),
            ("We were over taking the cart.", "We were overtaking the cart cart."),
            ("We were over taking the cart.", "We were overtaking the cat."),
            ("We were not over taking the cart.", "We were overtaking the cart."),
            ("The sun flower grows slowly.", "Slowly the sunflower grows."),
            ("Twenty one children came.", "Twentyone children came."),
            ("12 34", "1234"),
            ("1.23 4", "1.234"),
            ("A 001 002 code", "A 001002 code"),
            ("No, thank you.", "No thankyou, goodbye."),
            ("We were over taking.", "Thank you."),
        )
        for expected, actual in pairs:
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(word_spacing_equivalence(expected, actual))

    def test_ambiguous_punctuation_and_short_or_unbounded_pieces_stay_for_judgment(self):
        for expected, actual in (
            ("re-sign", "resign"), ("re—sign", "resign"),
            ("sun, flower", "sunflower"), ("sun. Flower", "sunflower"),
            ("can't", "cant"), ("a part", "apart"),
            ("some where over there", "somewhereoverthere"),
            ("over taking", "overtak ing"),
            ("sunflower", "s u n f l o w e r"),
            ("sun&flower", "sunandflower"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(word_spacing_equivalence(expected, actual))

    def test_exact_and_empty_text_do_not_claim_a_boundary_change(self):
        for expected, actual in (("", ""), ("...", "!"), ("Over taking!", "OVER TAKING.")):
            self.assertIsNone(word_spacing_equivalence(expected, actual))


class WordSpacingVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_both_purposes_accept_spacing_before_jev_and_preserve_raw_diagnostics(self):
        expected = "We were over taking the cart beside the sourcepalace."
        actual = "We were overtaking the cart beside the source palace."
        for purpose in ("reference", "speech"):
            for enabled in (False, True):
                with self.subTest(purpose=purpose, enabled=enabled):
                    recognizer = consensus.IndependentVerifier(actual, "Unrelated secondary.")
                    judge = Judge("unavailable")
                    verifier = ReferenceTextJudgingVerifier(recognizer, judge) if enabled else recognizer
                    report = self.check(verifier, expected=expected, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["text_confirmation_method"], "word_spacing")
                    self.assertEqual(report["word_boundary_comparison"]["boundary_changes"], 2)
                    self.assertEqual(report["expected_text"], expected)
                    self.assertEqual(report["transcript"], actual)
                    self.assertEqual(report["verified_transcript"], actual)
                    self.assertGreater(report["metrics"]["wer"], 0)
                    self.assertTrue(report["metrics"]["alignment"])
                    self.assertEqual(judge.calls, [])
                    self.assertNotIn("text_judge", report)
                    self.assertEqual(recognizer.calls, [(purpose, self.audio)])
                    if purpose == "reference":
                        self.assertEqual(report["reference_text"], actual)
                        self.assertTrue(report["reference_text_adjusted"])
                    else:
                        self.assertTrue(report["words"])

    def test_other_differences_still_reach_judge_even_with_a_valid_compound_change(self):
        expected = "We were not over taking the cart."
        actual = "We were overtaking the cart."
        judge = Judge("different")
        verifier = ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(actual), judge)
        report = self.check(verifier, expected=expected, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(judge.calls, [(expected, actual, None)])
        self.assertNotIn("word_boundary_comparison", report)

    def test_acoustic_failure_cannot_be_excused_by_perfect_spacing_equivalence(self):
        self.write_audio(amplitude=0)
        verifier = consensus.PrimaryVerifier("We were overtaking.")
        report = self.check(verifier, expected="We were over taking.", purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertIn("silent_audio", report["reasons"])
        self.assertEqual(verifier.calls, [])

    def test_secondary_spacing_match_recovers_rejected_primary_without_another_judgment(self):
        expected = "We were over taking the cart."
        primary = "We were taking the cat."
        secondary = "We were overtaking the cart."
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                judge = Judge("different")
                verifier = ReferenceTextJudgingVerifier(
                    consensus.IndependentVerifier(primary, secondary), judge)
                report = self.check(verifier, expected=expected, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "secondary_word_spacing")
                self.assertEqual(report["verified_transcript"], secondary)
                self.assertEqual(report["transcript"], primary)
                self.assertEqual(judge.calls, [(expected, primary, None)])
                self.assertEqual(report["text_judge_history"][0]["judgment"]["decision"], "different")
                if purpose == "speech":
                    self.assertEqual(report["words"], [])
                    self.assertEqual(report["word_timing_status"], "unavailable")

    def test_policy_change_invalidates_failed_verification_for_both_purposes(self):
        verifier = consensus.PrimaryVerifier("Overtaking.")
        for purpose in ("speech", "reference"):
            current = validation_key("Over taking.", verifier, purpose=purpose)
            with patch("epub_tts_narrator.renderer.quality.QUALITY_VERSION", 6):
                previous = validation_key("Over taking.", verifier, purpose=purpose)
            self.assertNotEqual(current, previous)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for retry integration")
class WordSpacingResumeTests(unittest.TestCase):
    setUp = retry.SpeechRetryTests.setUp
    generate = retry.SpeechRetryTests.generate
    take = retry.SpeechRetryTests.take

    def test_rejected_spacing_takes_are_rechecked_and_reused_after_policy_update(self):
        self.text = "We were over taking the cart."
        self.request["text"] = self.text
        judge = Judge("uncertain")
        verifier = ReferenceTextJudgingVerifier(consensus.IndependentVerifier(
            "We were overtaking the cart.", "We were overtaking the cart."), judge)
        with patch("epub_tts_narrator.renderer.quality.QUALITY_VERSION", 6), \
                patch("epub_tts_narrator.renderer.quality.word_spacing_equivalence", return_value=None):
            with self.assertRaises(SpeechVerificationFailed):
                self.take(SpeechRetryBudget(), verifier=verifier)
        self.assertEqual(len(self.calls), 2)
        old_judge_calls = len(judge.calls)
        first = self.root / "takes" / "attempt-001.wav"
        saved_audio = first.read_bytes()
        with patch.object(self.engine, "_write", side_effect=AssertionError("Good audio was regenerated")):
            wav, accepted = self.take(SpeechRetryBudget(), verifier=verifier)
        self.assertEqual(wav, first)
        self.assertEqual(wav.read_bytes(), saved_audio)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["text_confirmation_method"], "word_spacing")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(judge.calls), old_judge_calls)


if __name__ == "__main__":
    unittest.main()
