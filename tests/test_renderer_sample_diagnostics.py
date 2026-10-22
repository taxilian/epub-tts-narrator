"""Audition validation names the defect without changing selection rules."""

import unittest

from epub_tts_narrator.renderer.samples import (
    _safe_text, sample_text_problems, sample_word_count, validate_casting_script,
)


SCRIPT = (
    "Yesss, the morning train is nearly here, and I have saved a seat beside the window. "
    "Bring your coat and the small blue bag. We can watch the hills together while I "
    "tell you what happened at the quiet little station yesterday evening."
)


class SampleDiagnosticsTests(unittest.TestCase):
    def test_sufficient_length_script_names_stretched_word_and_its_repair(self):
        self.assertEqual(sample_word_count(SCRIPT), 43)
        problems = sample_text_problems(SCRIPT)
        self.assertEqual(len(problems), 1)
        self.assertIn("Yesss", problems[0])
        self.assertIn("ordinary spelling", problems[0])
        self.assertIn("voice profile", problems[0])
        self.assertLessEqual(len(problems[0]), 240)
        self.assertFalse(_safe_text(SCRIPT))
        with self.assertRaises(ValueError):
            validate_casting_script({"text": SCRIPT, "basis": "casting", "reason": "Audition."})

    def test_ordinary_spelling_is_usable_without_changing_the_rest_of_the_script(self):
        normal = SCRIPT.replace("Yesss", "Yes")
        self.assertEqual(sample_text_problems(normal), [])
        self.assertTrue(_safe_text(normal))
        script = {"text": normal, "basis": "casting", "reason": "Audition."}
        self.assertEqual(validate_casting_script(script), script)

    def test_multiple_stretched_words_are_identified_without_duplicate_warnings(self):
        problems = sample_text_problems("Yesss, thisss is fine. Yesss, it is.")
        self.assertEqual(len(problems), 1)
        self.assertIn("Yesss", problems[0])
        self.assertIn("thisss", problems[0])
        self.assertEqual(problems[0].count("Yesss"), 1)

    def test_forbidden_controls_are_printed_as_codepoints_not_executed_characters(self):
        problems = sample_text_problems("Read\x1bthis\u200b carefully\ud800.")
        self.assertEqual(len(problems), 1)
        for value in ("U+001B", "U+200B", "U+D800"):
            self.assertIn(value, problems[0])
        for value in ("\x1b", "\u200b", "\ud800"):
            self.assertNotIn(value, problems[0])

    def test_ordinary_line_breaks_and_tabs_are_allowed(self):
        self.assertEqual(sample_text_problems("Read this.\nThen\tthis.\rAnd this."), [])

    def test_independent_problems_are_reported_together(self):
        problems = sample_text_problems("[whisper] Yesss, visit https://example.invalid or a@b.invalid—")
        self.assertEqual(len(problems), 4)
        self.assertTrue(any("markup" in message for message in problems))
        self.assertTrue(any("URL" in message and "email" in message for message in problems))
        self.assertTrue(any("Yesss" in message for message in problems))
        self.assertTrue(any("unfinished" in message for message in problems))

    def test_uppercase_limit_keeps_its_existing_strict_boundary(self):
        self.assertEqual(sample_text_problems("ABCDEFGHIJK abcdefghi"), [])
        problems = sample_text_problems("ABCDEFGHIJKL abcdefghi")
        self.assertEqual(len(problems), 1)
        self.assertIn("uppercase", problems[0])
        self.assertIn("55%", problems[0])

    def test_existing_case_sensitive_repetition_and_trailing_dash_rules_are_unchanged(self):
        for allowed in ("YEssS, all is well.", "ééé", "The sentence was unfinished— "):
            with self.subTest(allowed=allowed):
                self.assertEqual(sample_text_problems(allowed), [])
                self.assertTrue(_safe_text(allowed))
        for rejected in ("The sentence was unfinished—”", "The sentence was unfinished–", "unfinished-\""):
            with self.subTest(rejected=rejected):
                self.assertTrue(any("unfinished" in message for message in sample_text_problems(rejected)))
                self.assertFalse(_safe_text(rejected))


if __name__ == "__main__":
    unittest.main()
