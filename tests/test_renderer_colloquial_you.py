"""The exact ya/you pronunciation variant never permits missing spoken words."""

import unittest

from epub_tts_narrator.renderer.phonetic_match import phonetic_equivalence
from epub_tts_narrator.renderer.quality import compare_words
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from test_renderer_jev_verification import Judge
import test_renderer_reference_consensus as consensus


class ColloquialYouAlignmentTests(unittest.TestCase):
    def test_exact_one_word_variant_is_audited_in_both_directions(self):
        for first, second in (("Ya", "you"), ("you", "ya")):
            expected, actual = f"I told {first} to wait.", f"I told {second} to wait."
            with self.subTest(expected=expected):
                result = phonetic_equivalence(expected, actual)
                self.assertIsNotNone(result)
                self.assertEqual(result["alignment"], [{
                    "operation": "colloquial", "rule": "you",
                    "expected_start": 2, "expected_end": 3,
                    "actual_start": 2, "actual_end": 3,
                    "expected": [first.lower()], "actual": [second],
                    "expansion": ["you"],
                }])
                self.assertTrue(speech_coverage(expected, actual)["passed"])

    def test_variant_composes_with_an_independent_name_sound_match(self):
        expected, actual = "Ya should ask Derrins.", "You should ask Darins."
        for first, second in ((expected, actual), (actual, expected)):
            with self.subTest(expected=first):
                result = phonetic_equivalence(first, second)
                self.assertIsNotNone(result)
                self.assertEqual([edit["operation"] for edit in result["alignment"]],
                                 ["colloquial", "substitute"])

    def test_local_colloquial_match_does_not_approve_an_uncertain_suffix(self):
        expected, actual = "Ya should put the weapons down.", "You should put the weapon down."
        self.assertIsNone(phonetic_equivalence(expected, actual))
        coverage = speech_coverage(expected, actual)
        self.assertTrue(coverage["passed"], coverage)
        self.assertEqual([row["operation"] for row in coverage["alignment"]],
                         ["colloquial", "pronunciation_candidate"])
        self.assertEqual(coverage["alignment"][1]["rule"], "terminal_s_variation")

    def test_no_missing_extra_reordered_negated_or_implied_words(self):
        cases = (
            ("I told ya to wait.", "I told you."),
            ("I told ya to wait.", "I told you to wait here."),
            ("Ya did.", "You did not."),
            ("Ya did not.", "You did."),
            ("Ya did.", "You did did."),
            ("Ya ya did.", "You did."),
            ("Ya did.", "You you did."),
            ("Ya did.", "Did you."),
            ("Ya saw me.", "Me saw you."),
            ("Ya saw you.", "You saw ya."),
            ("Ya came.", "You are came."),
            ("Ya came.", "You have came."),
            ("Ya need two ships.", "You need too ships."),
        )
        for expected, actual in cases:
            for first, second in ((expected, actual), (actual, expected)):
                with self.subTest(expected=first, actual=second):
                    self.assertIsNone(phonetic_equivalence(first, second))
                    self.assertFalse(speech_coverage(first, second)["passed"])


class ColloquialYouAudioTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def test_speech_and_reference_accept_locally_and_preserve_raw_evidence(self):
        pair = ("I told ya to wait.", "I told you to wait.")
        for expected, actual in (pair, pair[::-1]):
            for purpose in ("speech", "reference"):
                with self.subTest(expected=expected, purpose=purpose):
                    recognizer = consensus.IndependentVerifier(actual, "No fallback should run.")
                    judge = Judge("different")
                    judge.error = AssertionError("Ya/you needs no text judge")
                    report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                                        expected=expected, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["expected_text"], expected)
                    self.assertEqual(report["transcript"], actual)
                    self.assertEqual(report["raw_transcript"], actual)
                    self.assertEqual(report["verified_transcript"], actual)
                    for key, value in compare_words(expected, actual).items():
                        self.assertEqual(report["metrics"][key], value)
                    self.assertGreater(report["metrics"]["wer"], 0)
                    self.assertEqual(report["phonetic_comparison"]["alignment"][0]["rule"], "you")
                    self.assertNotIn("text_judge", report)
                    self.assertEqual(judge.calls, [])
                    self.assertEqual(recognizer.calls, [(purpose, self.audio)])
                    if purpose == "reference":
                        self.assertEqual(report["reference_text"], actual)
                        self.assertTrue(report["reference_text_adjusted"])
                    else:
                        self.assertNotIn("reference_text", report)

    def test_negation_cannot_be_approved_by_a_permissive_judge(self):
        for purpose in ("speech", "reference"):
            with self.subTest(purpose=purpose):
                judge = Judge("equivalent")
                recognizer = consensus.PrimaryVerifier("You did not.")
                report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                                    expected="Ya did.", purpose=purpose)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertEqual(judge.calls, [])

    def test_mixed_suffix_still_needs_its_own_full_passage_sound_judgment(self):
        expected, actual = "Ya should put the weapons down.", "You should put the weapon down."
        for decision, accepted in (("equivalent", True), ("different", False)):
            with self.subTest(decision=decision):
                judge = Judge(decision)
                report = self.check(ReferenceTextJudgingVerifier(consensus.PrimaryVerifier(actual), judge),
                                    expected=expected, purpose="speech")
                self.assertEqual(report["passed"], accepted, report)
                self.assertEqual(judge.calls, [(expected, actual, None)])
                self.assertNotIn("phonetic_comparison", report)
                self.assertEqual(report["expected_text"], expected)
                self.assertEqual(report["transcript"], actual)

    def test_silence_still_fails_before_transcription(self):
        self.write_audio(amplitude=0)
        recognizer = consensus.PrimaryVerifier("You did.")
        judge = Judge()
        report = self.check(ReferenceTextJudgingVerifier(recognizer, judge),
                            expected="Ya did.", purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertIn("silent_audio", report["reasons"])
        self.assertEqual(recognizer.calls, [])
        self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()
