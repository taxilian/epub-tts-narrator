"""ElevenLabs is an explicit cast provider using the same global config file."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.cli import main, parser
from audiobook_renderer.config import DEFAULTS, load_config, resolve_voice_design, select_voice_designer
from audiobook_renderer.engines import EngineError
from audiobook_renderer.voice_design import create_designer
from epub_audiobook.config import load_config as load_analyzer_config


SECRET = "fixture-elevenlabs-key-do-not-echo"


class ElevenLabsSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "epub-audiobook.toml"
        self.base = '[engine]\ncheckpoint="./tts-model"\n[verification]\nmodel="./asr-model"\n'
        self.path.write_text(self.base)

    def write_provider(self, fields=None):
        self.path.write_text(self.base + "[elevenlabs]\n" + (
            fields if fields is not None else f'api_key="{SECRET}"\n'))

    def test_global_settings_are_read_without_environment_and_override_is_temporary(self):
        self.write_provider(f'api_key="{SECRET}"\nmodel="eleven_multilingual_ttv_v2"\n'
                            'guidance_scale=7.0\nloudness=-0.1\ntimeout_seconds=150\n')
        original = self.path.read_bytes()
        with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "environment-key-is-ignored"}):
            config = load_config(self.path)
            selected = select_voice_designer(config, "elevenlabs")
        self.assertEqual(selected["voice_design"], {"type": "elevenlabs", **config["elevenlabs"]})
        self.assertEqual(resolve_voice_design(selected), selected["voice_design"])
        self.assertEqual(selected["voice_design"]["api_key"], SECRET)
        self.assertEqual(config["voice_design"]["type"], "auto")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(selected["engine"], config["engine"])

    def test_auto_preference_does_not_change_when_elevenlabs_key_is_added(self):
        self.write_provider()
        config = load_config(self.path)
        self.assertEqual(resolve_voice_design(config)["type"], "engine")
        config["gemini"]["api_key"] = "fixture-gemini-key"
        self.assertEqual(resolve_voice_design(config)["type"], "gemini")
        config["voice_design"]["type"] = "elevenlabs"
        self.assertEqual(resolve_voice_design(config), {"type": "elevenlabs", **config["elevenlabs"]})

    def test_missing_key_fails_before_casting_even_if_environment_has_key(self):
        config = deepcopy(DEFAULTS)
        for key in (None, "", "   "):
            with self.subTest(key=key), patch.dict("os.environ", {"ELEVENLABS_API_KEY": SECRET}):
                config["elevenlabs"]["api_key"] = key
                with self.assertRaisesRegex(ValueError, r"elevenlabs\.api_key") as raised:
                    select_voice_designer(config, "elevenlabs")
                self.assertNotIn(SECRET, str(raised.exception))
                config["voice_design"]["type"] = "elevenlabs"
                with self.assertRaisesRegex(ValueError, r"elevenlabs\.api_key"):
                    resolve_voice_design(config)

    def test_combined_config_remains_valid_for_analyzer(self):
        self.write_provider()
        with self.path.open("a") as stream:
            stream.write('[voice_design]\ntype="elevenlabs"\n'
                         '[provider]\nmodel="fixture-analysis-model"\napi_key="fixture-analysis-key"\n')
        renderer = load_config(self.path)
        analyzer = load_analyzer_config(self.path)
        self.assertEqual(resolve_voice_design(renderer)["type"], "elevenlabs")
        self.assertEqual(analyzer["provider"]["model"], "fixture-analysis-model")
        self.assertNotIn("elevenlabs", analyzer)

    def test_invalid_provider_settings_are_rejected_without_echoing_secrets(self):
        for field in ('model="unknown-model"', 'model=42', 'timeout_seconds=0',
                      'timeout_seconds=601', 'timeout_seconds=nan',
                      'guidance_scale=-1', 'guidance_scale=101', 'guidance_scale=true',
                      'guidance_scale=inf', 'loudness=-1.1', 'loudness=1.1',
                      'loudness=false', 'unknown_setting=1'):
            with self.subTest(field=field):
                self.write_provider(f'api_key="{SECRET}"\n{field}\n')
                with self.assertRaises(ValueError) as raised:
                    load_config(self.path)
                self.assertNotIn(SECRET, str(raised.exception))
        for field in ('api_key=123', 'api_key="bad\\tkey"'):
            with self.subTest(field=field):
                self.write_provider(field)
                with self.assertRaisesRegex(ValueError, r"elevenlabs\.api_key"):
                    load_config(self.path)

    def test_parser_exposes_designer_on_cast(self):
        for choice in ("local", "gemini", "elevenlabs"):
            args = parser().parse_args(["cast", "book", "--voice-designer", choice])
            self.assertEqual(args.voice_designer, choice)

    def test_cli_passes_choice_without_modifying_global_config(self):
        self.write_provider()
        original = self.path.read_bytes()
        config = load_config(self.path)
        with patch("audiobook_renderer.cli.load_bundle", return_value={}), \
                patch("audiobook_renderer.cli.resolve_book_config", return_value=config), \
                patch("audiobook_renderer.audition.casting_command", return_value=0) as casting:
            status = main(["cast", "fixture-book", "--config", str(self.path),
                           "--voice-designer", "elevenlabs"])
        self.assertEqual(status, 0)
        self.assertEqual(casting.call_args.args[2]["voice_design"]["type"], "elevenlabs")
        self.assertEqual(self.path.read_bytes(), original)

    def test_cli_missing_key_stops_before_any_casting_action(self):
        config = load_config(self.path)
        stderr = io.StringIO()
        with patch("audiobook_renderer.cli.load_bundle", return_value={}), \
                patch("audiobook_renderer.cli.resolve_book_config", return_value=config), \
                patch("audiobook_renderer.audition.casting_command") as casting, \
                contextlib.redirect_stderr(stderr):
            status = main(["cast", "fixture-book", "--config", str(self.path),
                           "--voice-designer", "elevenlabs", "--action", "regenerate-all"])
        self.assertEqual(status, 2)
        self.assertIn("elevenlabs.api_key", stderr.getvalue())
        casting.assert_not_called()

    def test_factory_passes_only_provider_settings_and_rejects_unknown_fields(self):
        settings = {"type": "elevenlabs", **DEFAULTS["elevenlabs"], "api_key": SECRET}
        with patch("audiobook_renderer.elevenlabs_voice_design.ElevenLabsVoiceDesigner") as factory:
            self.assertIs(create_designer(settings), factory.return_value)
        factory.assert_called_once_with(**{key: value for key, value in settings.items() if key != "type"})
        with self.assertRaisesRegex(EngineError, "Unknown ElevenLabs"):
            create_designer({**settings, "extra": SECRET})


if __name__ == "__main__":
    unittest.main()
