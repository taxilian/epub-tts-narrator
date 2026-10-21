"""Reference wording judgments keep complete, independently heard transcripts."""

from unittest.mock import patch
import unittest

from audiobook_renderer.verification import ReferenceTextJudgingVerifier
from epub_audiobook.jev import JevTranscriptJudge
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import IndependentVerifier


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
        recognizer = IndependentVerifier(primary=actual, secondary=actual)
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        return ReferenceTextJudgingVerifier(recognizer, judge), recognizer, judge

    @staticmethod
    def response(score):
        return {
            "model": "jev-fixture",
            "answers": {"harmless_variation": {"type": "noul", "noul": score}},
            "usage": {"input_tokens": 100, "output_tokens": 3},
        }

    def test_confirmed_spelling_and_pronunciation_variants_keep_actual_reference_words(self):
        cases = (
            ("Givve me the book.", "Give me the book.", "givve", "give"),
            ("I will recieve the book.", "I will receive the book.", "recieve", "receive"),
            ("Meet me at the libary.", "Meet me at the library.", "libary", "library"),
            ("Meet me at the library.", "Meet me at the libary.", "library", "libary"),
        )
        for source, heard, expected_word, heard_word in cases:
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
                self.assertEqual(recognizer.calls, [("reference", self.audio), ("corroboration", self.audio)])
                evaluate.assert_called_once()
                state = evaluate.call_args.args[0]
                self.assertEqual(state["expected_text"], expected_word)
                self.assertEqual(state["primary_transcript"], heard_word)
                audit = report["text_judge"]
                self.assertEqual(audit["decision"], "equivalent")
                self.assertEqual(audit["probabilities"]["harmless_variation"], .98)
                self.assertEqual({row["transcript"] for row in audit["differences"][0]["occurrences"]},
                                 {"primary", "secondary"})

    def test_changed_words_names_quantities_and_negation_are_rejected(self):
        cases = (
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
                self.assertEqual(recognizer.calls, [("reference", self.audio), ("corroboration", self.audio)])
                evaluate.assert_called_once()

    def test_missing_and_added_words_cannot_be_hidden_by_harmless_spelling(self):
        expected = "Givve me the book." + CONTEXT
        corrected = expected.replace("Givve", "Give")
        for actual in (corrected.replace("checked the doors", "checked doors"),
                       corrected + " Goodbye."):
            with self.subTest(actual=actual):
                verifier, recognizer, judge = self.wrapped(actual)
                with patch.object(judge._client, "evaluate", return_value=self.response(.99)) as evaluate:
                    report = self.check(verifier, expected=expected)

                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertEqual(recognizer.calls, [("reference", self.audio)])
                self.assertNotIn("text_judge", report)
                self.assertNotIn("reference_text", report)
                evaluate.assert_not_called()

    def test_uncertain_word_variant_remains_rejected(self):
        expected = "Givve me the book." + CONTEXT
        verifier, _, judge = self.wrapped(expected.replace("Givve", "Give"))
        with patch.object(judge._client, "evaluate", return_value=self.response(.60)) as evaluate:
            report = self.check(verifier, expected=expected)

        self.assertFalse(report["passed"], report)
        self.assertEqual(report["text_judge"]["decision"], "uncertain")
        self.assertNotIn("reference_text", report)
        evaluate.assert_called_once()

    def test_reference_word_variant_policy_does_not_relax_audiobook_speech(self):
        expected = "Givve me the book." + CONTEXT
        verifier, recognizer, judge = self.wrapped(expected.replace("Givve", "Give"))
        with patch.object(judge._client, "evaluate", return_value=self.response(.99)) as evaluate:
            report = self.check(verifier, expected=expected, purpose="speech")

        self.assertFalse(report["passed"], report)
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertEqual(recognizer.calls, [("speech", self.audio)])
        self.assertNotIn("reference_text", report)
        self.assertNotIn("text_judge", report)
        evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
