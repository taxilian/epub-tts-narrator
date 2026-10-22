"""Explicit colloquial expansions preserve ordered speech coverage and evidence."""

import unittest

from epub_tts_narrator.renderer.quality import normalize_words, phonetic_equivalence
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from test_renderer_jev_verification import Judge
import test_renderer_reference_consensus as consensus


class ColloquialAlignmentTests(unittest.TestCase):
    def assert_complete_alignment(self, expected, actual, report):
        expected_words, actual_words = normalize_words(expected), normalize_words(actual)
        left_cursor = right_cursor = 0
        for change in report["alignment"]:
            self.assertEqual(expected_words[left_cursor:change["expected_start"]],
                             actual_words[right_cursor:change["actual_start"]])
            self.assertEqual(expected_words[change["expected_start"]:change["expected_end"]],
                             change["expected"])
            self.assertEqual(actual_words[change["actual_start"]:change["actual_end"]],
                             change["actual"])
            self.assertGreater(change["expected_end"], change["expected_start"])
            self.assertGreater(change["actual_end"], change["actual_start"])
            left_cursor, right_cursor = change["expected_end"], change["actual_end"]
        self.assertEqual(expected_words[left_cursor:], actual_words[right_cursor:])

    def test_explicit_variants_match_both_directions_with_whitespace_only_expansion(self):
        for contraction in ("what'cha", "whatcha", "whatchu", "What’cha"):
            for expansion in ("what you", "what\n\tyou"):
                for first, second in ((contraction, expansion), (expansion, contraction)):
                    expected, actual = f"Tell me {first} need.", f"Tell me {second} need."
                    with self.subTest(expected=expected, actual=actual):
                        report = phonetic_equivalence(expected, actual)
                        self.assertIsNotNone(report)
                        self.assertEqual(report["method"], "double_metaphone_v2")
                        self.assertEqual(len(report["alignment"]), 1)
                        change = report["alignment"][0]
                        self.assertEqual(change["operation"], "colloquial")
                        self.assertEqual(change["rule"], "what_you")
                        self.assertEqual(change["expected_start"], 2)
                        self.assertEqual(change["actual_start"], 2)
                        self.assert_complete_alignment(expected, actual, report)

    def test_colloquial_and_name_changes_are_both_audited(self):
        expected, actual = "What'cha doing, Dalley?", "What you doing, Dally?"
        for first, second in ((expected, actual), (actual, expected)):
            with self.subTest(expected=first):
                report = phonetic_equivalence(first, second)
                self.assertIsNotNone(report)
                self.assertEqual(len(report["alignment"]), 2)
                self.assertEqual([change["operation"] for change in report["alignment"]],
                                 ["colloquial", "substitute"])
                self.assertEqual(report["alignment"][0]["rule"], "what_you")
                self.assert_complete_alignment(first, second, report)

    def test_nearby_expansions_are_not_implied_by_the_explicit_rule(self):
        alternatives = ("what", "you", "what did you", "what are you", "what have you",
                        "what didyou", "what areyou", "what haveyou", "what they",
                        "what your", "what youyou", "what you you")
        for contraction in ("what'cha", "whatcha", "whatchu"):
            for alternative in alternatives:
                for expected, actual in ((contraction, alternative), (alternative, contraction)):
                    with self.subTest(expected=expected, actual=actual):
                        self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_punctuation_cannot_connect_the_expanded_words(self):
        for separated in ("what, you", "what. You", "what—you", "what-you", "what/you",
                          "what: you", "what (you)", "what 'you'"):
            for expected, actual in (("whatcha", separated), (separated, "whatcha")):
                with self.subTest(expected=expected, actual=actual):
                    self.assertIsNone(phonetic_equivalence(expected, actual))

    def test_colloquial_match_cannot_hide_other_missing_or_extra_content(self):
        cases = (
            ("Whatcha doing today, Dalley?", "What you doing, Dally?"),
            ("Whatcha doing, Dalley?", "What you doing, Dally? Goodbye."),
            ("Whatcha not doing, Dalley?", "What you doing, Dally?"),
            ("Whatcha doing with fifteen coins, Dalley?", "What you doing with fifty coins, Dally?"),
            ("What does it mean, Insang?", "What is it mean, Insang?"),
            ("Whatcha doing, Insang? What does it mean?", "What you doing, Insang? What is it mean?"),
            ("Dalley asked whatcha doing.", "What you doing Dalley asked."),
        )
        for expected, actual in cases:
            for first, second in ((expected, actual), (actual, expected)):
                with self.subTest(expected=first, actual=second):
                    self.assertIsNone(phonetic_equivalence(first, second))


class ColloquialAudioVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_both_purposes_keep_original_text_and_raw_wer_without_jev(self):
        for expected in ("What'cha doing, Dalley?", "Whatcha doing, Dalley?", "Whatchu doing, Dalley?"):
            actual = "What you doing, Dally?"
            for purpose in ("speech", "reference"):
                with self.subTest(expected=expected, purpose=purpose):
                    recognizer = consensus.IndependentVerifier(actual, "Unrelated fallback speech.")
                    judge = Judge("different")
                    judge.error = AssertionError("Local colloquial match must precede JEV")
                    verifier = ReferenceTextJudgingVerifier(recognizer, judge)
                    report = self.check(verifier, expected=expected, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["expected_text"], expected)
                    self.assertEqual(report["transcript"], actual)
                    self.assertEqual(report["raw_transcript"], actual)
                    self.assertEqual(report["verified_transcript"], actual)
                    self.assertGreater(report["metrics"]["wer"], 0)
                    self.assertTrue(report["metrics"]["alignment"])
                    self.assertEqual(report["text_confirmation_method"], "double_metaphone")
                    self.assertEqual(report["phonetic_comparison"]["method"], "double_metaphone_v2")
                    self.assertEqual([change["operation"] for change in report["phonetic_comparison"]["alignment"]],
                                     ["colloquial", "substitute"])
                    self.assertEqual(judge.calls, [])
                    self.assertEqual(recognizer.calls, [(purpose, self.audio)])
                    self.assertNotIn("text_judge", report)
                    if purpose == "reference":
                        self.assertEqual(report["reference_text"], actual)
                        self.assertTrue(report["reference_text_adjusted"])
                    else:
                        self.assertNotIn("reference_text", report)

    def test_missing_nearby_words_reject_before_jev(self):
        expected, actual = "Whatcha not doing, Dalley?", "What you doing, Dally?"
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                judge = Judge("different")
                recognizer = consensus.PrimaryVerifier(actual)
                report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                                    expected=expected, purpose=purpose)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertEqual(judge.calls, [])
                self.assertNotIn("phonetic_comparison", report)
                self.assertEqual(report["expected_text"], expected)
                self.assertEqual(report["transcript"], actual)

    def test_silence_is_not_approved_by_a_colloquial_text_match(self):
        self.write_audio(amplitude=0)
        recognizer = consensus.PrimaryVerifier("What you doing, Dally?")
        judge = Judge()
        verifier = ReferenceTextJudgingVerifier(recognizer, judge)
        report = self.check(verifier, expected="What'cha doing, Dalley?", purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertIn("silent_audio", report["reasons"])
        self.assertEqual(recognizer.calls, [])
        self.assertEqual(judge.calls, [])
        self.assertNotIn("phonetic_comparison", report)


if __name__ == "__main__":
    unittest.main()
