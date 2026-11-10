"""Real CLI publication regressions for manual post-audit attribution edits."""

import unittest

from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.performance_plan import text_sha256
from test_attribution_verification_cli import AttributionCLIHarness


class AttributionManualReviewCLITests(AttributionCLIHarness):
    def test_manual_edit_uses_a_third_cast_speaker_and_preserves_source_cast_and_controls(self):
        def add_controls(document):
            segment = document["segments"][1]
            segment["performance_controls"] = {
                "version": 1, "text_sha256": text_sha256(segment["text"]),
                "controls": [{"kind": "delivery", "tag": "whisper", "direction": "",
                              "start": 0, "end": 1, "text": segment["text"][0]}],
            }

        original = self.make_book(prepare=add_controls)
        code, output, errors, _ = self.invoke(input_text="e\ne\n1\n\ny\ny\nn\n", tty=True)

        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        edited = revised["segments"][1]
        self.assertEqual(edited["role_id"], "narrator_main")
        self.assertEqual(edited["attribution"]["basis"], "user")
        self.assertTrue(edited["attribution"]["review_required"])
        self.assertEqual(edited["text"], original["segments"][1]["text"])
        self.assertEqual(edited["performance_controls"], original["segments"][1]["performance_controls"])
        self.assertEqual(revised["source"], original["source"])
        self.assertEqual(revised["cast"], original["cast"])
        self.assertIn("1 applied / 1 left unchanged.", output)
        self.assertIn("0 accepted proposal(s); 1 manual edit(s).", output)
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))

    def test_manual_delivery_edit_reprompts_over_240_characters_and_saves_the_bounded_value(self):
        original = self.make_book()
        answers = "e\ne\n\n" + "x" * 241 + "\n  " + "x" * 240 + "  \n\ny\nn\n"
        code, output, errors, _ = self.invoke(input_text=answers, tty=True)

        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        self.assertEqual(revised["segments"][1]["delivery"]["direction"], "x" * 240)
        self.assertEqual(revised["segments"][2]["role_id"], original["segments"][2]["role_id"])
        self.assertIn("240 characters", output)
        self.assertIn("0 accepted proposal(s); 1 manual edit(s).", output)

    def test_cancelling_a_staged_manual_edit_discards_every_staged_field(self):
        self.make_book()
        before = (self.book / "analysis.json").read_bytes()
        answers = "e\ne\n1\nquietly\n/cancel\nn\nn\n"
        code, output, errors, _ = self.invoke(input_text=answers, tty=True)

        self.assertEqual(code, 0, errors)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())
        self.assertIn("Manual edit discarded.", output)
        self.assertIn("0 accepted proposal(s); 0 manual edit(s).", output)

    def test_eof_discards_the_staged_edit_but_publishes_previously_accepted_findings(self):
        original = self.make_book()
        answers = "e\ny\ne\n1\nquietly\n"
        code, output, errors, _ = self.invoke(input_text=answers, tty=True)

        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        self.assertEqual(revised["segments"][1]["role_id"], "tom")
        self.assertEqual(revised["segments"][2]["role_id"], original["segments"][2]["role_id"])
        self.assertEqual(revised["segments"][2]["attribution"], original["segments"][2]["attribution"])
        self.assertIn("discarding the unconfirmed manual edit", output)
        self.assertIn("1 applied / 1 left unchanged.", output)
        self.assertIn("1 accepted proposal(s); 0 manual edit(s).", output)


if __name__ == "__main__":
    unittest.main()
