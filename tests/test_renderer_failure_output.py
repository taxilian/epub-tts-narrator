"""Rejected takes explain bounded lexical differences without terminal controls."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.quality import compare_words
from epub_tts_narrator.renderer.workflow import (
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
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"), \
                    patch("epub_tts_narrator.renderer.workflow.validate_audio", return_value=report), \
                    patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize):
                with self.assertRaises(RenderBlocked):
                    verified_take("My pack mate.", generate, temp, {"test": True},
                                  object(), config, notify=messages.append)
            self.assertEqual(messages, [
                "  Attempt 1/1 rejected: text_mismatch "
                "(expected → heard: pack mate → packmate; recognizers disagreed)",
            ])
            self.assertFalse((Path(temp) / "accepted.json").exists())

    def test_jev_failure_is_explained_without_echoing_remote_details(self):
        for decision, expected in (
            ("different", "JEV judged the transcript unlikely for the requested speech"),
            ("uncertain", "JEV could not confirm a plausible transcription"),
            ("unavailable", "JEV transcription check unavailable"),
            ("equivalent", ""),
            ("unexpected private response", ""),
        ):
            with self.subTest(decision=decision):
                detail = _verification_failure_detail({"text_judge": {
                    "decision": decision, "error": "private remote detail",
                }})
                self.assertEqual(detail, expected)
                self.assertNotIn("equivalent wording", detail)
                self.assertNotIn("substantive text difference", detail)

    def test_accepted_jev_take_reports_transcription_plausibility(self):
        with tempfile.TemporaryDirectory() as temp:
            messages = []
            report = {"passed": True, "reasons": [],
                      "text_judge": {"decision": "equivalent"}}

            def generate(path, seed):
                Path(path).write_bytes(b"saved audio fixture")
                return {"seed": seed}

            def normalize(raw, wav, **kwargs):
                Path(wav).write_bytes(Path(raw).read_bytes())

            config = {"quality": {}, "render": {
                "max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
            }}
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"), \
                    patch("epub_tts_narrator.renderer.workflow.validate_audio", return_value=report), \
                    patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize):
                verified_take("She carried the doll.", generate, temp, {"test": True},
                              object(), config, notify=messages.append, purpose="reference")
            self.assertEqual(messages, [
                "  JEV accepted the transcript as plausible recognition of the requested speech.",
            ])
            self.assertNotIn("equivalent", " ".join(messages))

    def test_jev_outage_retains_audio_and_same_command_retries_verification_only(self):
        with tempfile.TemporaryDirectory() as temp:
            generated = []
            def generate(path, seed):
                generated.append(path)
                Path(path).write_bytes(b"saved audio fixture")
                return {"seed": seed}
            def normalize(raw, wav, **kwargs):
                Path(wav).write_bytes(Path(raw).read_bytes())
            unavailable = {"passed": False, "reasons": ["text_mismatch"],
                           "text_judge": {"decision": "unavailable"}}
            passed = {"passed": True, "reasons": [], "text_judge": {"decision": "equivalent"}}
            config = {"quality": {}, "render": {
                "max_attempts": 3, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
            }}
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"), \
                    patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize), \
                    patch("epub_tts_narrator.renderer.workflow.validate_audio", side_effect=[unavailable, passed]) as validate:
                with self.assertRaisesRegex(RenderBlocked, "rerun the same command"):
                    verified_take("A reference.", generate, temp, {"test": True}, object(),
                                  config, purpose="reference")
                self.assertEqual(len(generated), 1)
                self.assertFalse((Path(temp) / "accepted.json").exists())
                audio, record = verified_take("A reference.", generate, temp, {"test": True}, object(),
                                              config, purpose="reference")
            self.assertEqual(len(generated), 1)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(audio.read_bytes(), b"saved audio fixture")
            self.assertTrue(record["report"]["passed"])

    def test_generation_errors_are_visible_for_gemini_and_local_engines(self):
        from epub_tts_narrator.renderer.gemini_voice_design import GeminiVoiceError
        safe = "Gemini API request failed (HTTP 403). Check Gemini API access and project permissions."
        for error in (GeminiVoiceError(safe), RuntimeError("Local model could not allocate memory")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                messages = []
                config = {"quality": {}, "render": {
                    "max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
                }}
                def generate(path, seed):
                    raise error
                with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"):
                    with self.assertRaises(RenderBlocked):
                        verified_take("Sample text.", generate, temp, {"test": True},
                                      object(), config, notify=messages.append)
                self.assertIn(str(error), " ".join(messages))


if __name__ == "__main__":
    unittest.main()
