"""Local phonetic coverage accepts intact speech without remote judgment."""

import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.quality import validation_key
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier, create_verifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget
import test_renderer_reference_consensus as consensus
from test_renderer_jev_verification import Judge
import test_renderer_speech_retry as retry
from test_renderer_workflow import FFMPEG, FFPROBE


class LocalPhoneticVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_missing_dependency_stops_both_factories_before_model_initialization(self):
        from epub_tts_narrator.renderer.chapter_verification import create_chapter_verifier
        with patch("epub_tts_narrator.renderer.phonetic_match.require_phonetic_matcher",
                   side_effect=RuntimeError("Install Metaphone")), \
             patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier") as mlx, \
             patch("epub_tts_narrator.renderer.chapter_verification.FasterWhisperVerifier") as faster:
            for factory, backend in ((create_verifier, "mlx_whisper"),
                                     (create_chapter_verifier, "faster_whisper")):
                with self.subTest(backend=backend), self.assertRaisesRegex(RuntimeError, "Metaphone"):
                    factory({"type": backend, "model": "/unused", "language": "en"})
            mlx.assert_not_called()
            faster.assert_not_called()

    def test_primary_sound_matches_bypass_jev_for_both_purposes_with_original_evidence(self):
        cases = (("Vandire", "van dyer"), ("Vandire", "van deer"),
                 ("Derrins", "Darins"), ("Derrins", "Darren's"),
                 ("Qarm", "Khorm"), ("Mota", "Moda"))
        for expected_name, heard_name in cases:
            expected = expected_name + " waited beside the gate while the rain fell."
            actual = heard_name + " waited beside the gate while the rain fell."
            for purpose in ("speech", "reference"):
                for enabled in (False, True):
                    with self.subTest(expected_name=expected_name, heard_name=heard_name,
                                      purpose=purpose, jev_enabled=enabled):
                        recognizer = consensus.IndependentVerifier(actual, "Unrelated secondary speech.")
                        judge = Judge("different")
                        verifier = ReferenceTextJudgingVerifier(recognizer, judge) if enabled else recognizer
                        report = self.check(verifier, expected=expected, purpose=purpose)
                        self.assertTrue(report["passed"], report)
                        self.assertEqual(report["text_confirmation_method"], "double_metaphone")
                        self.assertIsInstance(report["phonetic_comparison"], dict)
                        self.assertTrue(report["phonetic_comparison"])
                        self.assertEqual(report["expected_text"], expected)
                        self.assertEqual(report["transcript"], actual)
                        self.assertEqual(report["raw_transcript"], actual)
                        self.assertEqual(report["verified_transcript"], actual)
                        self.assertGreater(report["metrics"]["wer"], 0)
                        self.assertTrue(report["metrics"]["alignment"])
                        self.assertEqual(judge.calls, [])
                        self.assertEqual(recognizer.calls, [(purpose, self.audio)])
                        self.assertNotIn("text_judge", report)
                        self.assertNotIn("phonetic_reviews", report)
                        if purpose == "reference":
                            self.assertEqual(report["reference_text"], actual)
                            self.assertTrue(report["reference_text_adjusted"])
                        else:
                            self.assertNotIn("reference_text", report)
                            self.assertTrue(report["words"])

    def test_missing_added_and_repeated_speech_cannot_hide_behind_a_name_match(self):
        cases = (
            ("Vandire did not open the gate.", "Van Dyer did open the gate."),
            ("Derrins waited beside the door.", "Darins waited beside the door. Thank you."),
            ("Vandire stood beside the gate.", "Van Deer stood beside."),
            ("Derrins waited by the gate.", "Darins Darins waited by the gate."),
            ("Vandire saw Mota by the gate.", "Van Deer saw by the gate."),
        )
        for expected, actual in cases:
            for purpose in ("speech", "reference"):
                for enabled in (False, True):
                    with self.subTest(expected=expected, actual=actual, purpose=purpose, jev_enabled=enabled):
                        recognizer = consensus.PrimaryVerifier(actual)
                        judge = Judge("different")
                        verifier = ReferenceTextJudgingVerifier(recognizer, judge) if enabled else recognizer
                        report = self.check(verifier, expected=expected, purpose=purpose)
                        self.assertFalse(report["passed"], report)
                        self.assertIn("text_mismatch", report["reasons"])
                        self.assertNotIn("phonetic_comparison", report)
                        self.assertNotIn("verified_transcript", report)
                        self.assertEqual(judge.calls, [(expected, actual, None)] if enabled else [])

    def test_silence_stops_before_recognition_or_local_phonetic_check(self):
        self.write_audio(amplitude=0)
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                recognizer = consensus.PrimaryVerifier("Van Dyer waited by the gate.")
                judge = Judge()
                verifier = ReferenceTextJudgingVerifier(recognizer, judge)
                with patch("epub_tts_narrator.renderer.quality.phonetic_equivalence",
                           side_effect=AssertionError("Silent audio reached text approval")):
                    report = self.check(verifier, expected="Vandire waited by the gate.", purpose=purpose)
                self.assertFalse(report["passed"], report)
                self.assertIn("silent_audio", report["reasons"])
                self.assertEqual(recognizer.calls, [])
                self.assertEqual(judge.calls, [])
                self.assertNotIn("phonetic_comparison", report)

    def test_secondary_sound_match_keeps_original_transcript_and_avoids_second_judgment(self):
        expected = "Vandire waited beside the gate."
        primary = "Someone wandered away from the building."
        secondary = "Van Dyer waited beside the gate."
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                recognizer = consensus.IndependentVerifier(primary, secondary)
                judge = Judge("different")
                verifier = ReferenceTextJudgingVerifier(recognizer, judge)
                report = self.check(verifier, expected=expected, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "secondary_double_metaphone")
                self.assertEqual(report["transcript"], primary)
                self.assertEqual(report["verified_transcript"], secondary)
                self.assertGreater(report["metrics"]["wer"], 0)
                self.assertTrue(report["phonetic_comparison"])
                self.assertEqual(judge.calls, [(expected, primary, None)])
                self.assertEqual(recognizer.calls, [(purpose, self.audio), ("corroboration", self.audio)])
                self.assertEqual(report["text_judge_history"][0]["judgment"]["decision"], "different")
                if purpose == "reference":
                    self.assertEqual(report["reference_text"], secondary)
                else:
                    self.assertEqual(report["words"], [])
                    self.assertEqual(report["word_timing_status"], "unavailable")

    def test_local_policy_revision_changes_failed_verification_identity(self):
        verifier = consensus.PrimaryVerifier("Darins waited.")
        for purpose in ("speech", "reference"):
            current = validation_key("Derrins waited.", verifier, purpose=purpose)
            with patch("epub_tts_narrator.renderer.quality.QUALITY_VERSION", 0):
                previous = validation_key("Derrins waited.", verifier, purpose=purpose)
            self.assertNotEqual(current, previous)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for saved-take integration")
class LocalPhoneticResumeTests(unittest.TestCase):
    setUp = retry.SpeechRetryTests.setUp
    generate = retry.SpeechRetryTests.generate
    take = retry.SpeechRetryTests.take

    def test_accepted_local_match_is_reused_without_synthesis_or_remote_judgment(self):
        self.text = "Vandire waited beside the gate."
        self.request["text"] = self.text
        actual = "Van Dyer waited beside the gate."
        recognizer = consensus.PrimaryVerifier(actual)
        judge = Judge("different")
        verifier = ReferenceTextJudgingVerifier(recognizer, judge)
        first, original = self.take(SpeechRetryBudget(), verifier=verifier)
        self.assertEqual(len(self.calls), 1)
        saved_bytes = first.read_bytes()
        saved_report = first.with_suffix(".json").read_bytes()
        previous_judge_calls = len(judge.calls)
        previous_asr_calls = len(recognizer.calls)
        wav, accepted = self.take(SpeechRetryBudget(), verifier=verifier)
        self.assertEqual(wav, first)
        self.assertEqual(accepted, original)
        self.assertEqual(first.read_bytes(), saved_bytes)
        self.assertEqual(first.with_suffix(".json").read_bytes(), saved_report)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(judge.calls), previous_judge_calls)
        self.assertEqual(len(recognizer.calls), previous_asr_calls)
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["text_confirmation_method"], "double_metaphone")
        self.assertEqual(accepted["report"]["transcript"], actual)
        self.assertGreater(accepted["report"]["metrics"]["wer"], 0)


if __name__ == "__main__":
    unittest.main()
