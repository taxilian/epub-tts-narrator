"""Hugging Face download credentials come only from the shared config file."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.config import load_config as load_analyzer_config
from epub_tts_narrator.renderer.config import load_config as load_renderer_config
from epub_tts_narrator.renderer.diagnostics import configured_secrets, safe_error_text


SECRET = "hf_fixture-file-only-download-key"


class HuggingFaceConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "epub-tts-narrator.toml"
        self.base = '[engine]\ncheckpoint="./drama-3"\n[verification]\nmodel="./whisper"\n'

    def write(self, section=""):
        self.path.write_text(self.base + section, encoding="utf-8")

    def test_default_is_anonymous_despite_environment_tokens(self):
        self.write()
        with patch.dict("os.environ", {"HF_TOKEN": "environment-token-is-ignored",
                                       "HUGGING_FACE_HUB_TOKEN": "other-ignored-token"}):
            for loader in (load_analyzer_config, load_renderer_config):
                with self.subTest(loader=loader.__module__):
                    config = loader(self.path)
                    self.assertEqual(config["huggingface"], {"api_key": ""})
                    config["huggingface"]["api_key"] = "changed"
                    self.assertEqual(loader(self.path)["huggingface"], {"api_key": ""})

    def test_shared_config_preserves_literal_file_token_for_both_commands(self):
        self.write(f'[huggingface]\napi_key="{SECRET}"\n')
        original = self.path.read_bytes()
        with patch.dict("os.environ", {"HF_TOKEN": "environment-token-is-ignored"}):
            for loader in (load_analyzer_config, load_renderer_config):
                with self.subTest(loader=loader.__module__):
                    self.assertEqual(loader(self.path)["huggingface"], {"api_key": SECRET})
        self.assertEqual(self.path.read_bytes(), original)

    def test_empty_table_and_empty_key_are_valid(self):
        for section in ("[huggingface]\n", '[huggingface]\napi_key=""\n'):
            self.write(section)
            for loader in (load_analyzer_config, load_renderer_config):
                with self.subTest(section=section, loader=loader.__module__):
                    self.assertEqual(loader(self.path)["huggingface"], {"api_key": ""})

    def test_invalid_credentials_rejected_without_exposing_values(self):
        for value in ('123', 'true', '[]', '{}', '"hf_fixture secret"',
                      '"hf_fixture\\tsecret"', '"hf_fixture\\nsecret"',
                      '"hf_fixture\\u007fsecret"'):
            self.write(f"[huggingface]\napi_key={value}\n")
            for loader in (load_analyzer_config, load_renderer_config):
                with self.subTest(value=value, loader=loader.__module__):
                    with self.assertRaisesRegex(ValueError, r"huggingface\.api_key") as caught:
                        loader(self.path)
                    self.assertNotIn("hf_fixture", str(caught.exception))

    def test_unknown_key_warns_without_exposing_token(self):
        self.write(f'[huggingface]\napi_key="{SECRET}"\nsecret_unknown_key="{SECRET}"\n')
        for loader in (load_analyzer_config, load_renderer_config):
            with self.subTest(loader=loader.__module__):
                diagnostic = io.StringIO()
                with redirect_stderr(diagnostic):
                    config = loader(self.path)
                self.assertEqual(config["huggingface"], {"api_key": SECRET})
                self.assertIn("huggingface.secret_unknown_key", diagnostic.getvalue())
                self.assertNotIn(SECRET, diagnostic.getvalue())

    def test_huggingface_token_is_masked_without_hiding_download_error(self):
        self.write(f'[huggingface]\napi_key="{SECRET}"\n')
        config = load_renderer_config(self.path)
        secrets = configured_secrets(config)
        self.assertIn(SECRET, secrets)
        diagnostic = f"HTTP 403: access denied for fishaudio/drama-3; Bearer {SECRET}"
        self.assertEqual(safe_error_text(diagnostic, secrets),
                         "HTTP 403: access denied for fishaudio/drama-3; Bearer [REDACTED]")


if __name__ == "__main__":
    unittest.main()
