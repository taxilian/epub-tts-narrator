"""Spoken initialisms preserve every repeated letter without a remote verdict."""

import unittest

from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer
from epub_tts_narrator.renderer.quality import compare_words
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget
from test_renderer_jev_verification import Judge
from test_renderer_phonetic_fallback import PairClient
import test_renderer_reference_consensus as consensus
import test_renderer_speech_retry as retry
from test_renderer_workflow import FFMPEG, FFPROBE


class LetterNamesVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_repeated_letters_pass_locally_with_original_word_error_evidence(self):
        cases = (
            ("See-see?", "CC?", "CC"),
            ("CC?", "See-see?", "CC"),
            ("See see see, she repeated.", "CCC, she repeated.", "CCC"),
            ("Say bee bee before leaving.", "Say BB before leaving.", "BB"),
        )
        for expected, actual, letters in cases:
            for purpose in ("speech", "reference"):
                for enabled in (False, True):
                    with self.subTest(expected=expected, actual=actual,
                                      purpose=purpose, judge_enabled=enabled):
                        recognizer = consensus.IndependentVerifier(
                            actual, "Unrelated secondary transcript.")
                        judge = Judge("different")
                        verifier = (ReferenceTextJudgingVerifier(recognizer, judge)
                                    if enabled else recognizer)
                        report = self.check(verifier, expected=expected, purpose=purpose)
                        self.assertTrue(report["passed"], report)
                        self.assertEqual(report["text_confirmation_method"], "double_metaphone")
                        self.assertEqual(report["expected_text"], expected)
                        self.assertEqual(report["transcript"], actual)
                        self.assertEqual(report["raw_transcript"], actual)
                        self.assertEqual(report["verified_transcript"], actual)
                        original = compare_words(expected, actual)
                        self.assertGreater(original["wer"], 0)
                        for key, value in original.items():
                            self.assertEqual(report["metrics"][key], value)
                        letter_operations = [row for row in report["phonetic_comparison"]["alignment"]
                                             if row["operation"] == "letter_names"]
                        self.assertEqual(len(letter_operations), 1, report)
                        operation = letter_operations[0]
                        self.assertEqual(operation["matched_letters"], list(letters))
                        self.assertEqual(operation["rule"], "exact_english_letter_names")
                        self.assertEqual(operation["expected_end"] - operation["expected_start"],
                                         len(operation["expected"]))
                        self.assertEqual(operation["actual_end"] - operation["actual_start"],
                                         len(operation["actual"]))
                        self.assertEqual(judge.calls, [])
                        self.assertEqual(recognizer.calls, [(purpose, self.audio)])
                        self.assertNotIn("text_judge", report)
                        self.assertNotIn("phonetic_reviews", report)
                        if purpose == "reference":
                            self.assertEqual(report["reference_text"], actual)
                            self.assertTrue(report["reference_text_adjusted"])
                        else:
                            self.assertNotIn("reference_text", report)

    def test_letter_expansion_does_not_interfere_with_soundalike_name_coverage(self):
        expected = "Vandire said see see while Derrins waited."
        actual = "Van Dyer said CC while Darren's waited."
        judge = Judge("different")
        recognizer = consensus.PrimaryVerifier(actual)
        report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                            expected=expected, purpose="speech")
        self.assertTrue(report["passed"], report)
        operations = report["phonetic_comparison"]["alignment"]
        self.assertIn("letter_names", [row["operation"] for row in operations])
        self.assertGreater(len(operations), 1)
        self.assertEqual(report["transcript"], actual)
        self.assertEqual(judge.calls, [])

    def test_exact_letter_expansion_is_excluded_from_pronunciation_jury(self):
        expected = "See-see, the library is open."
        actual = "CC, the libary is open."
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                recognizer = consensus.PrimaryVerifier(actual)
                judge = Judge("different")
                client = PairClient()
                verifier = ReferenceTextJudgingVerifier(
                    recognizer, judge, PronunciationReviewer(client))
                report = self.check(verifier, expected=expected, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "phonetic_primary")
                self.assertEqual(len(judge.calls), 1)
                self.assertEqual(len(client.calls), 3)
                for _, payload in client.calls:
                    self.assertEqual(len(payload["pairs"]), 1)
                    self.assertEqual(payload["pairs"][0]["expected_word"], "library")
                    self.assertEqual(payload["pairs"][0]["recognized_word"], "libary")
                self.assertEqual(report["expected_text"], expected)
                self.assertEqual(report["transcript"], actual)

    def test_missing_extra_and_changed_content_cannot_use_optimistic_judge(self):
        cases = (
            ("See-see?", "C?"),
            ("See-see?", "CCC?"),
            ("CC?", "See?"),
            ("CC?", "See see see?"),
            ("PC?", "PCC?"),
            ("PC?", "PCCCCCC?"),
            ("CC?", "CCC?"),
            ("CC bee", "CB"),
            ("CC bee", "CBB"),
            ("Say see see before leaving.", "CC before leaving."),
            ("Say see see before leaving.", "Say CC before leaving now."),
            ("Do not say see see.", "Do say CC."),
            ("Repeat see see fifteen times.", "Repeat CC fifty times."),
            ("Vandire said see see while Derrins waited.", "Van Dyer said CC while waited."),
        )
        for expected, actual in cases:
            for purpose in ("speech", "reference"):
                with self.subTest(expected=expected, actual=actual, purpose=purpose):
                    judge = Judge("equivalent")
                    recognizer = consensus.PrimaryVerifier(actual)
                    report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                                        expected=expected, purpose=purpose)
                    self.assertFalse(report["passed"], report)
                    self.assertIn("text_mismatch", report["reasons"])
                    self.assertEqual(judge.calls, [])
                    self.assertNotIn("phonetic_comparison", report)
                    self.assertNotIn("verified_transcript", report)
                    self.assertNotIn("reference_text", report)

    def test_expressive_consonants_keep_their_pronunciation_review_path(self):
        for expected_sound, actual_sound in (
                ("hmm", "hmmm"), ("shh", "shhhh"), ("brr", "brrrr"),
                ("psst", "pssst"), ("tsk", "tsssk"), ("sss", "sssss")):
            with self.subTest(expected_sound=expected_sound, actual_sound=actual_sound):
                expected = f"He said {expected_sound} softly."
                actual = f"He said {actual_sound} softly."
                judge = Judge("equivalent")
                recognizer = consensus.PrimaryVerifier(actual)
                report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                                    expected=expected, purpose="speech")
                self.assertTrue(report["passed"], report)

    def test_secondary_letter_match_preserves_primary_evidence_and_discards_its_timings(self):
        expected = "Say see see before leaving."
        primary = "CC before leaving."
        secondary = "Say CC before leaving."
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                recognizer = consensus.IndependentVerifier(primary, secondary)
                judge = Judge("different")
                verifier = ReferenceTextJudgingVerifier(recognizer, judge)
                report = self.check(verifier, expected=expected, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "secondary_double_metaphone")
                self.assertEqual(report["expected_text"], expected)
                self.assertEqual(report["transcript"], primary)
                self.assertEqual(report["raw_transcript"], primary)
                self.assertEqual(report["verified_transcript"], secondary)
                self.assertGreater(report["metrics"]["wer"], 0)
                self.assertEqual(report["phonetic_comparison"]["alignment"][0]["operation"],
                                 "letter_names")
                self.assertEqual(judge.calls, [])
                self.assertEqual(recognizer.calls,
                                 [(purpose, self.audio), ("corroboration", self.audio)])
                if purpose == "speech":
                    self.assertEqual(report["words"], [])
                    self.assertEqual(report["word_timing_status"], "unavailable")
                else:
                    self.assertEqual(report["reference_text"], secondary)

    def test_two_decoders_cannot_collectively_supply_missing_repeated_letters(self):
        expected = "Say see see before leaving."
        recognizer = consensus.IndependentVerifier("Say C before leaving.", "Say C before leaving.")
        judge = Judge("equivalent")
        report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                            expected=expected, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(judge.calls, [])
        self.assertNotIn("verified_transcript", report)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required for saved-take integration")
class LetterNamesResumeTests(unittest.TestCase):
    setUp = retry.SpeechRetryTests.setUp
    generate = retry.SpeechRetryTests.generate
    take = retry.SpeechRetryTests.take

    def test_local_letter_acceptance_is_cached_without_another_take(self):
        self.text = "See-see?"
        self.request["text"] = self.text
        recognizer = consensus.PrimaryVerifier("CC?")
        judge = Judge("different")
        verifier = ReferenceTextJudgingVerifier(recognizer, judge)
        wav, accepted = self.take(SpeechRetryBudget(), verifier=verifier)
        waveform = wav.read_bytes()
        saved_report = wav.with_suffix(".json").read_bytes()
        recognitions = list(recognizer.calls)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(accepted["report"]["passed"], accepted)
        self.assertEqual(accepted["report"]["transcript"], "CC?")
        self.assertEqual(accepted["report"]["phonetic_comparison"]["alignment"][0]["operation"],
                         "letter_names")
        again = self.take(SpeechRetryBudget(), verifier=verifier)
        self.assertEqual(again, (wav, accepted))
        self.assertEqual(wav.read_bytes(), waveform)
        self.assertEqual(wav.with_suffix(".json").read_bytes(), saved_report)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(recognizer.calls, recognitions)
        self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()
