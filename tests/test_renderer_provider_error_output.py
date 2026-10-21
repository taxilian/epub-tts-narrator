"""Provider failures retain actionable details in casting output and saved attempts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote, quote_plus

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.diagnostics import configured_secrets, safe_error_text
from epub_tts_narrator.renderer.gemini_voice_design import GeminiVoiceError
from epub_tts_narrator.renderer.workflow import RenderBlocked, _verification_failure_detail, verified_take
from audition_fixtures import prepare_bundle


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


class ProviderErrorOutputTests(unittest.TestCase):
    def test_only_exact_configured_credentials_and_escaped_forms_are_masked(self):
        key = 'fixture/+ api"key-ñ'
        variants = [key, quote(key, safe=""), quote(key), quote_plus(key),
                    json.dumps(key, ensure_ascii=True)[1:-1],
                    json.dumps(key, ensure_ascii=False)[1:-1]]
        diagnostic = "INVALID_ARGUMENT for voice_test123\n" + "\n".join(variants) + "\nDetails are complete."
        self.assertEqual(
            safe_error_text(diagnostic, secrets=(key,)),
            "INVALID_ARGUMENT for voice_test123\n" + "\n".join(["[REDACTED]"] * len(variants)) +
            "\nDetails are complete.",
        )

    def test_configured_secrets_are_found_recursively_without_masking_model_settings(self):
        config = {"gemini": {"api_key": "fixture-gemini"},
                  "nested": [{"password": "fixture-password", "access_token": "fixture-token"}],
                  "model": "gemini-3.8-flash-tts", "max_tokens": 100, "tokenizer": "keep-visible"}
        self.assertEqual(configured_secrets(config),
                         {"fixture-gemini", "fixture-password", "fixture-token"})

    def test_provider_details_are_complete_and_keep_line_breaks(self):
        diagnostic = (
            "Gemini API request failed (HTTP 400) during sample synthesis.\n"
            "INVALID_ARGUMENT: " + "A detailed provider explanation. " * 80 +
            "\nfieldViolations: generation_config.speech_config[0].voice\n"
            "Request ID: request-end-marker"
        )
        self.assertEqual(_verification_failure_detail({"provider_error": diagnostic}), diagnostic)

    def test_provider_controls_are_removed_without_losing_the_error_message(self):
        diagnostic = (
            "\x1b[31mINVALID_ARGUMENT\x1b[0m: failed\n"
            "\x1b]0;replaced terminal title\x07Details: \x00voice missing\u202e\n"
            "\tfield: speech_config"
        )
        self.assertEqual(
            _verification_failure_detail({"provider_error": diagnostic}),
            "INVALID_ARGUMENT: failed\nDetails: voice missing\n\tfield: speech_config",
        )

    def test_failed_generation_keeps_complete_provider_details_in_output_and_attempt(self):
        diagnostic = "Gemini HTTP 400\n" + "diagnostic detail " * 100 + "\nfinal provider detail"
        with tempfile.TemporaryDirectory() as temp:
            messages = []
            config = {"quality": {}, "render": {
                "max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
            }}
            generate = Mock(side_effect=GeminiVoiceError(diagnostic))
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"):
                with self.assertRaises(RenderBlocked) as stopped:
                    verified_take("Sample text.", generate, temp, {"fixture": True},
                                  object(), config, notify=messages.append, purpose="reference")
            self.assertIn(diagnostic, "\n".join(messages))
            self.assertIn(diagnostic, str(stopped.exception))
            report = read_json(Path(temp) / "attempt-001.json")["report"]
            self.assertEqual(report["provider_error"], diagnostic)
            self.assertEqual(report["error"], diagnostic)
            self.assertFalse((Path(temp) / "accepted.json").exists())

    def test_local_generation_error_masks_credentials_in_saved_report_and_all_output(self):
        key = "fixture-local-secret"
        diagnostic = f"Local adapter failed\nUpstream said invalid argument; key={key}\nEnd of detail."
        expected = diagnostic.replace(key, "[REDACTED]")
        with tempfile.TemporaryDirectory() as temp:
            messages = []
            config = {"quality": {}, "gemini": {"api_key": key}, "render": {
                "max_attempts": 1, "seed": 42, "sample_rate": 44100, "ffmpeg": "unused",
            }}
            generate = Mock(side_effect=RuntimeError(diagnostic))
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"):
                with self.assertRaises(RenderBlocked) as stopped:
                    verified_take("Sample text.", generate, temp, {"fixture": True},
                                  object(), config, notify=messages.append)
                with self.assertRaises(RenderBlocked) as resumed:
                    verified_take("Sample text.", generate, temp, {"fixture": True},
                                  object(), config, notify=messages.append)
            self.assertIn(expected, "\n".join(messages))
            self.assertIn(expected, str(stopped.exception))
            self.assertIn(expected, str(resumed.exception))
            report = read_json(Path(temp) / "attempt-001.json")["report"]
            self.assertEqual(report["error"], expected)
            self.assertNotIn(key, "\n".join(messages) + str(stopped.exception) + str(resumed.exception)
                             + json.dumps(report))
            generate.assert_called_once()

    def test_casting_regeneration_reports_full_error_and_does_not_select_failed_voice(self):
        diagnostic = "Gemini HTTP 400\n" + "provider explanation " * 100 + "\nvoice field rejected"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = deepcopy(DEFAULTS)
            config["voices"].update(directory=str(root / "voices"), series_id="error-output-fixture")
            config["render"].update(work_dir=str(root / "work"), output_dir=str(root / "output"),
                                    max_attempts=1)
            config["voice_design"]["type"] = "gemini"
            config["voice_design"]["api_key"] = "fixture-key-never-used"
            bundle = prepare_bundle(load_bundle(EXAMPLE))
            engine = Mock(identity={"engine": "fixture", "model": "fixture-v1"},
                          capabilities={"reference_voice": True})
            designer = Mock(identity={"engine": "gemini_voice_design", "model": "fixture-gemini"},
                            capabilities={"voice_design": True})
            designer.voice_design_identity = designer.identity
            designer.create_voice.side_effect = GeminiVoiceError(diagnostic)
            messages = []
            session = CastingSession(bundle, config, engine_factory=lambda settings: engine,
                                     designer_factory=lambda settings: designer,
                                     verifier_factory=lambda *args, **kwargs: Mock(identity={"backend": "fixture"}),
                                     notify=messages.append)
            self.addCleanup(session.close)
            voice_id = bundle["cast"]["voices"][0]["id"]
            with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="fixture"):
                with self.assertRaises(RenderBlocked):
                    session.regenerate(voice_id)
            self.assertIn(diagnostic, "\n".join(messages))
            attempts = list((root / "work" / "casting-candidates").glob("*/attempt-001.json"))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(read_json(attempts[0])["report"]["provider_error"], diagnostic)
            self.assertIsNone(session.store.load(voice_id))
            engine.synthesize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
