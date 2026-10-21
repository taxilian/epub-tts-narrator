"""Rejected takes explain bounded lexical differences without terminal controls."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.quality import compare_words
from audiobook_renderer.workflow import (
    RenderBlocked, _diagnostic_text, _verification_failure_detail, verified_take,
)


class FailureOutputTests(unittest.TestCase):
    def detail(self, expected, actual, **extra):
        return _verification_failure_detail({
            "metrics": compare_words(expected, actual), **extra,
        })

    def test_adjacent_word_edits_are_grouped(self):
        detail = self.detail("My pack mate knows Vessa.", "My packmate knows Vesa.")
        self.assertEqual(detail, "expected → heard: pack mate → packmate; vessa → vesa")

    def test_insertions_and_deletions_show_the_empty_side(self):
        self.assertEqual(self.detail("", "Invented words."),
                         "expected → heard: (nothing) → invented words")
        self.assertEqual(self.detail("Missing words.", ""),
                         "expected → heard: missing words → (nothing)")

    def test_no_differences_or_alignment_produces_no_detail(self):
        self.assertEqual(self.detail("Hello there", "Hello there"), "")
        self.assertEqual(_verification_failure_detail({"reasons": ["silent_audio"]}), "")
        self.assertEqual(_verification_failure_detail({"metrics": {"alignment": None}}), "")

    def test_displays_at_most_three_differences(self):
        detail = self.detail("one a two b three c four d", "first a second b third c fourth d")
        self.assertEqual(
            detail, "expected → heard: one → first; two → second; three → third; +1 more")

    def test_long_spans_are_truncated(self):
        detail = self.detail(" ".join(["expected"] * 100), " ".join(["actual"] * 100))
        self.assertLess(len(detail), 150)
        self.assertEqual(detail.count("…"), 2)

    def test_terminal_controls_and_newlines_are_inert(self):
        detail = _verification_failure_detail({"metrics": {"alignment": [{
            "operation": "substitute",
            "expected": ["\x1b[31mexpected\x1b[0m\nwords\t"],
            "actual": ["\x1b]0;fake title\x07heard\rwords\x08\u202e"],
        }]}})
        self.assertEqual(detail, "expected → heard: expected words → heard words")
        self.assertEqual(_diagnostic_text("\x00\ufeffvalue"), "value")

    def test_secondary_check_explains_why_it_did_not_resolve_the_failure(self):
        decisions = {
            "mismatch_outside_reference_limits": "secondary check skipped: discrepancy exceeds reference limits",
            "unavailable": "secondary recognizer unavailable",
            "verification_failed": "secondary recognition failed",
            "disagreed": "recognizers disagreed",
        }
        for decision, explanation in decisions.items():
            with self.subTest(decision=decision):
                self.assertEqual(_verification_failure_detail({
                    "corroboration": {"decision": decision}}), explanation)

    def test_verified_take_emits_diagnostic_and_still_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            messages = []
            report = {
                "passed": False, "reasons": ["text_mismatch"],
                "metrics": compare_words("My pack mate.", "My packmate."),
                "corroboration": {"decision": "disagreed"},
            }

            def generate(path, seed):
                Path(path).write_bytes(b"fixture")
                return {"seed": seed}

            def normalize(raw, wav, **kwargs):
                Path(wav).write_bytes(Path(raw).read_bytes())

            config = {"quality": {}, "render": {
                "max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
            }}
            with patch("audiobook_renderer.workflow.make_validation_key", return_value="fixture"), \
                    patch("audiobook_renderer.workflow.validate_audio", return_value=report), \
                    patch("audiobook_renderer.workflow.normalize_clip", side_effect=normalize):
                with self.assertRaises(RenderBlocked):
                    verified_take("My pack mate.", generate, temp, {"test": True},
                                  object(), config, notify=messages.append)
            self.assertEqual(messages, [
                "  Attempt 1/1 rejected: text_mismatch "
                "(expected → heard: pack mate → packmate; recognizers disagreed)",
            ])
            self.assertFalse((Path(temp) / "accepted.json").exists())

    def test_safe_gemini_error_is_visible_without_exposing_arbitrary_errors(self):
        from audiobook_renderer.gemini_voice_design import GeminiVoiceError
        safe = "Gemini API request failed (HTTP 403). Check Gemini API access and project permissions."
        for error, visible in ((GeminiVoiceError(safe), True), (RuntimeError("private detail"), False)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                messages = []
                config = {"quality": {}, "render": {
                    "max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
                }}
                def generate(path, seed):
                    raise error
                with patch("audiobook_renderer.workflow.make_validation_key", return_value="fixture"):
                    with self.assertRaises(RenderBlocked):
                        verified_take("Sample text.", generate, temp, {"test": True},
                                      object(), config, notify=messages.append)
                self.assertEqual(safe in " ".join(messages), visible)
                self.assertNotIn("private detail", " ".join(messages))


if __name__ == "__main__":
    unittest.main()
