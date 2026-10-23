"""Configuration warnings name unknown keys without echoing their values or mutating settings."""

from contextlib import redirect_stderr
import io
import tempfile
from pathlib import Path
import unittest

from epub_tts_narrator.config_warnings import known_settings
from epub_tts_narrator.renderer.config import load_config


class ConfigurationWarningTests(unittest.TestCase):
    def test_filters_fields_without_mutation_or_values_and_retains_plugin_options(self):
        settings = {"model": "secret-value", "options": {"plugin_setting": 3}}
        diagnostic = io.StringIO()
        with redirect_stderr(diagnostic):
            result = known_settings(settings, {"options"}, section="engine")
        self.assertEqual(result, {"options": {"plugin_setting": 3}})
        self.assertIn("model", settings)
        self.assertEqual(diagnostic.getvalue(),
                         "Warning: ignoring unknown configuration key engine.model.\n")
        self.assertNotIn("secret-value", diagnostic.getvalue())

    def test_unusual_or_credential_shaped_keys_cannot_inject_terminal_output(self):
        for key in ("api\nkey", "\x1b[2J", "a" * 200, "hf_fixture-secret", "apikey_fixture"):
            diagnostic = io.StringIO()
            with self.subTest(key=key), redirect_stderr(diagnostic):
                self.assertEqual(known_settings({key: "sensitive"}, {}, section="engine"), {})
            self.assertEqual(diagnostic.getvalue(),
                             "Warning: ignoring unknown configuration key engine.<redacted-key>.\n")

    def test_worker_and_server_unknown_fields_are_removed_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text('''[verification]
model="./whisper"
[[render_pool.workers]]
name="remote"
type="fish_http"
url="https://render.example/v1/tts"
obsolete_setting="private-value"
[server]
port=8090
obsolete_setting="private-value"
''')
            diagnostic = io.StringIO()
            with redirect_stderr(diagnostic):
                config = load_config(path)
            self.assertNotIn("obsolete_setting", config["render_pool"]["workers"][0])
            self.assertNotIn("obsolete_setting", config["server"])
            self.assertEqual(config["server"]["port"], 8090)
            self.assertIn("render_pool.workers[1].obsolete_setting", diagnostic.getvalue())
            self.assertIn("server.obsolete_setting", diagnostic.getvalue())
            self.assertNotIn("private-value", diagnostic.getvalue())


if __name__ == "__main__":
    unittest.main()
