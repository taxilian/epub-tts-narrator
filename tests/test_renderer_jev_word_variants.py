"""Reference wording judgments keep complete, independently heard transcripts."""

from unittest.mock import patch
import unittest

from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from epub_tts_narrator.jev import JevTranscriptJudge, QUESTIONS
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import PrimaryVerifier


CONTEXT = (
    " We waited beside the window while the captain checked the doors. "
    "Outside, the rain fell steadily across the empty courtyard. "
    "There was still time to finish our work before the other travelers "
    "returned from the village."
)


class JevWordVariantTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual):
        recognizer = PrimaryVerifier(actual)
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        return ReferenceTextJudgingVerifier(recognizer, judge), recognizer, judge

    @staticmethod
    def response(score):
        return {
            "model": "jev-fixture",
            "answers": {"roughly_same": {"type": "noul", "noul": score}},
            "usage": {"input_tokens": 100, "output_tokens": 3},
        }

    def test_confirmed_spelling_and_pronunciation_variants_keep_actual_reference_words(self):
        cases = (
            ("She put the doll on the shelf.", "She put the dull on the shelf."),
            ("Givve me the book.", "Give me the book."),
            ("I will recieve the book.", "I will receive the book."),
            ("Meet me at the libary.", "Meet me at the library."),
            ("Meet me at the library.", "Meet me at the libary."),
        )
        for source, heard in cases:
            with self.subTest(source=source, heard=heard):
                expected, actual = source + CONTEXT, heard + CONTEXT
                verifier, recognizer, judge = self.wrapped(actual)
                with patch.object(judge._client, "evaluate", return_value=self.response(.98)) as evaluate:
                    report = self.check(verifier, expected=expected)

                self.assertTrue(report["passed"], report)
                self.assertEqual(report["expected_text"], expected)
                self.assertEqual(report["reference_text"], actual)
                self.assertTrue(report["reference_text_adjusted"])
                self.assertGreater(report["reference_comparison"]["wer"], 0)
                self.assertEqual(recognizer.calls, [("reference", self.audio)])
                evaluate.assert_called_once()
                state = evaluate.call_args.args[0]
                self.assertEqual(set(state), {"requested_passage", "recognized_passage"})
                self.assertEqual(state["requested_passage"], expected)
                self.assertEqual(state["recognized_passage"], actual)
                audit = report["text_judge"]
                self.assertEqual(audit["decision"], "equivalent")
                self.assertEqual(audit["probabilities"]["roughly_same"], .98)
                self.assertEqual(audit["method"], "jev_spoken_passage")
                self.assertEqual(audit["questions"], QUESTIONS)
                self.assertEqual(audit["comparisons"][0]["transcripts"], ["primary"])
                self.assertEqual(audit["comparisons"][0]["requested_passage"], expected)
                self.assertEqual(audit["comparisons"][0]["recognized_passage"], actual)

    def test_negative_passage_judgments_are_respected(self):
        cases = (
            ("She bought a doll.", "She purchased a toy."),
            ("Give me the book.", "Take me the book."),
            ("Jasper carried the book.", "Jasmine carried the book."),
            ("Seven travelers carried books.", "Eleven travelers carried books."),
            ("I will not leave.", "I will now leave."),
        )
        for source, heard in cases:
            with self.subTest(source=source, heard=heard):
                verifier, recognizer, judge = self.wrapped(heard + CONTEXT)
                with patch.object(judge._client, "evaluate", return_value=self.response(.02)) as evaluate:
                    report = self.check(verifier, expected=source + CONTEXT)

                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertNotIn("reference_text", report)
                self.assertEqual(report["text_judge"]["decision"], "different")
                self.assertEqual(recognizer.calls, [("reference", self.audio)])
                evaluate.assert_called_once()

    def test_entire_passage_including_missing_or_added_words_reaches_judge(self):
        expected = "Givve me the book." + CONTEXT
        corrected = expected.replace("Givve", "Give")
        for actual in (corrected.replace("the captain checked", "checked"),
                       corrected + " Goodbye."):
            with self.subTest(actual=actual):
                verifier, recognizer, judge = self.wrapped(actual)
                with patch.object(judge._client, "evaluate", return_value=self.response(.01)) as evaluate:
                    report = self.check(verifier, expected=expected)

                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertEqual(recognizer.calls, [("reference", self.audio)])
                self.assertEqual(report["text_judge"]["decision"], "different")
                self.assertNotIn("reference_text", report)
                evaluate.assert_called_once()
                self.assertEqual(evaluate.call_args.args[0],
                                 {"requested_passage": expected, "recognized_passage": actual})

    def test_uncertain_word_variant_remains_rejected(self):
        expected = "Givve me the book." + CONTEXT
        verifier, _, judge = self.wrapped(expected.replace("Givve", "Give"))
        with patch.object(judge._client, "evaluate", return_value=self.response(.60)) as evaluate:
            report = self.check(verifier, expected=expected)

        self.assertFalse(report["passed"], report)
        self.assertEqual(report["text_judge"]["decision"], "uncertain")
        self.assertNotIn("reference_text", report)
        evaluate.assert_called_once()

    def test_short_names_are_judged_without_character_distance_or_substring_gates(self):
        cases = (("Doll!", "Dull!"), ("Jas!", "Chase!"), ("Jas!", "Jazz!"), ("Jas!", "Jace!"), ("Jas, wait.", "Jas"))
        for purpose in ("reference", "speech"):
            for expected, heard in cases:
                with self.subTest(purpose=purpose, expected=expected, heard=heard):
                    verifier, recognizer, judge = self.wrapped(heard)
                    with patch.object(judge._client, "evaluate", return_value=self.response(.95)) as evaluate:
                        report = self.check(verifier, expected=expected, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    evaluate.assert_called_once()
                    self.assertEqual(evaluate.call_args.args[0],
                                     {"requested_passage": expected, "recognized_passage": heard})

    def test_jev_confirmed_word_variant_is_also_accepted_for_audiobook_speech(self):
        expected = "Givve me the book." + CONTEXT
        verifier, recognizer, judge = self.wrapped(expected.replace("Givve", "Give"))
        with patch.object(judge._client, "evaluate", return_value=self.response(.99)) as evaluate:
            report = self.check(verifier, expected=expected, purpose="speech")

        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reasons"], [])
        self.assertEqual(recognizer.calls, [("speech", self.audio)])
        self.assertNotIn("reference_text", report)
        self.assertEqual(report["text_judge"]["decision"], "equivalent")
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
