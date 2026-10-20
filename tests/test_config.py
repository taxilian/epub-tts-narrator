"""File configuration is strict, deterministic, and keeps secrets out of errors."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_audiobook.config import load_config
from epub_audiobook.pipeline import DEFAULT_POLICY


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "settings.toml"

    def load_text(self, text):
        self.path.write_text(text, encoding="utf-8")
        return load_config(self.path)

    def test_missing_default_file_returns_all_defaults(self):
        with patch("epub_audiobook.config.Path.cwd", return_value=self.root):
            configuration = load_config()
        self.assertEqual(configuration, {
            "provider": {"model": None, "base_url": "https://api.openai.com/v1", "api_key": None,
                         "timeout": 120.0, "json_mode": True, "max_output_tokens": None,
                         "token_limit_field": "max_completion_tokens", "reasoning_effort": None,
                         "max_retries": 0, "retry_initial_delay": 5.0, "retry_max_delay": 60.0},
            "analysis": {"chunk_chars": 12000, "max_request_chars": 200000,
                         "repair_attempts": 1, "cache_dir": None, "whole_book_discovery": False,
                         "identity_consensus": True,
                         "attribution_review": False, "attribution_review_chunk_chars": 60000,
                         "attribution_review_notes": "",
                         "cast_reconciliation_rounds": 1,
                         "cast_reconciliation_notes": [],
                         "attribution_workers": 1, "cast_review_notes": "", "series_registry": None},
            "extraction": {"include_auxiliary": False},
            "performance_policy": DEFAULT_POLICY,
        })

    def test_default_file_is_loaded_from_current_directory(self):
        (self.root / "epub-audiobook.toml").write_text('[provider]\nmodel = "local-model"\n')
        with patch("epub_audiobook.config.Path.cwd", return_value=self.root):
            configuration = load_config()
        self.assertEqual(configuration["provider"]["model"], "local-model")

    def test_empty_config_returns_fresh_independent_defaults(self):
        first = self.load_text("")
        first["provider"]["base_url"] = "changed"
        first["performance_policy"]["thoughts"] = "changed"
        first["analysis"]["cast_reconciliation_notes"].append("changed")
        second = load_config(str(self.path))
        self.assertEqual(second["provider"]["base_url"], "https://api.openai.com/v1")
        self.assertEqual(second["performance_policy"]["thoughts"], DEFAULT_POLICY["thoughts"])
        self.assertEqual(second["analysis"]["cast_reconciliation_notes"], [])

    def test_all_documented_fields_can_be_configured_in_file(self):
        configuration = self.load_text('''
[provider]
model = "fixture-model"
base_url = "http://localhost:1234/v1"
api_key = "literal-secret-key"
timeout = 31.5
json_mode = false
max_output_tokens = 4096
token_limit_field = "max_tokens"
[analysis]
chunk_chars = 512
max_request_chars = 8192
repair_attempts = 2
cast_reconciliation_rounds = 3
cast_reconciliation_notes = ["", "", "Verify the source's shared utterance."]
cache_dir = "stored/cache"
identity_consensus = false
attribution_review = true
attribution_review_chunk_chars = 1024
attribution_review_notes = "Check the continuing speaker at b9."
[extraction]
include_auxiliary = true
[performance_policy]
thoughts = "The named thinker speaks interior thoughts."
letters = "Read letters using the named author's voice."
nested_quotations = "Keep the outer speaker's voice."
headings_and_nonstory_text = "The narrator reads headings."
''')
        self.assertEqual(configuration["provider"], {
            "model": "fixture-model", "base_url": "http://localhost:1234/v1",
            "api_key": "literal-secret-key", "timeout": 31.5, "json_mode": False,
            "max_output_tokens": 4096, "token_limit_field": "max_tokens", "reasoning_effort": None,
            "max_retries": 0, "retry_initial_delay": 5.0, "retry_max_delay": 60.0,
        })
        self.assertEqual(configuration["analysis"], {
            "chunk_chars": 512, "max_request_chars": 8192, "repair_attempts": 2,
            "cache_dir": str((self.root / "stored/cache").resolve()),
            "cast_reconciliation_rounds": 3,
            "cast_reconciliation_notes": ["", "", "Verify the source's shared utterance."],
            "identity_consensus": False,
            "attribution_review": True, "attribution_review_chunk_chars": 1024,
            "attribution_review_notes": "Check the continuing speaker at b9.",
            "whole_book_discovery": False, "attribution_workers": 1, "cast_review_notes": "", "series_registry": None,
        })
        self.assertTrue(configuration["extraction"]["include_auxiliary"])
        self.assertEqual(configuration["performance_policy"]["thoughts"], "The named thinker speaks interior thoughts.")
        self.assertEqual(len(configuration["performance_policy"]), 5)

    def test_partial_policy_override_preserves_other_defaults(self):
        configuration = self.load_text('[performance_policy]\nthoughts = "Use narration for thoughts."\n')
        self.assertEqual(configuration["performance_policy"]["thoughts"], "Use narration for thoughts.")
        self.assertEqual(configuration["performance_policy"]["letters"], DEFAULT_POLICY["letters"])

    def test_large_context_and_reasoning_settings(self):
        configuration = self.load_text('[provider]\nreasoning_effort="high"\n[analysis]\nwhole_book_discovery=true\nattribution_workers=2\n')
        self.assertEqual(configuration["provider"]["reasoning_effort"], "high")
        self.assertTrue(configuration["analysis"]["whole_book_discovery"])
        self.assertEqual(configuration["analysis"]["attribution_workers"], 2)
        for text in ('[provider]\nreasoning_effort="invalid"', '[provider]\nreasoning_effort=[]',
                     '[analysis]\nwhole_book_discovery=1', '[analysis]\nattribution_workers=true',
                     '[analysis]\nattribution_workers=0', '[analysis]\nattribution_workers=9'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load_text(text)

    def test_cast_review_observations_are_literal_file_text(self):
        notes = "Check the named speaker in b000010; this is an observation, not approval."
        self.assertEqual(self.load_text('[analysis]\ncast_review_notes = "' + notes + '"\n')["analysis"]["cast_review_notes"], notes)
        self.assertEqual(self.load_text('[analysis]\ncast_review_notes = ""')["analysis"]["cast_review_notes"], "")
        with self.assertRaises(ValueError):
            self.load_text('[analysis]\ncast_review_notes = true')

    def test_cast_reconciliation_rounds_are_bounded_file_integers(self):
        for value in (1, 2, 3):
            self.assertEqual(self.load_text(f"[analysis]\ncast_reconciliation_rounds = {value}")["analysis"]
                             ["cast_reconciliation_rounds"], value)
        for value in ("0", "4", "-1", "true", "1.0", '"2"', "[]"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cast_reconciliation_rounds"):
                self.load_text(f"[analysis]\ncast_reconciliation_rounds = {value}")

    def test_reconciliation_notes_are_bounded_per_pass_string_lists(self):
        settings = self.load_text('[analysis]\ncast_reconciliation_notes = ["", "", "Final source lead."]\n'
                                  'cast_reconciliation_rounds = 3\n')["analysis"]
        self.assertEqual(settings["cast_reconciliation_notes"], ["", "", "Final source lead."])
        for value in ('"private-fixture-value"', "true", "3", "[1]", "[true]", '[["nested"]]',
                      '{ first = "lead" }', '["", "", "", ""]'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cast_reconciliation_notes") as error:
                self.load_text(f"[analysis]\ncast_reconciliation_rounds = 3\ncast_reconciliation_notes = {value}\n")
            self.assertNotIn("private-fixture-value", str(error.exception))
        with self.assertRaisesRegex(ValueError, "at most"):
            self.load_text('[analysis]\ncast_reconciliation_notes = ["", ""]\n')

    def test_explicit_missing_or_unreadable_path_is_an_error(self):
        missing = self.root / "missing-secret-name.toml"
        with self.assertRaises(ValueError) as captured:
            load_config(missing)
        self.assertNotIn("missing-secret-name", str(captured.exception))
        with self.assertRaisesRegex(ValueError, "Cannot read configuration"):
            load_config(self.root)

    def test_relative_cache_directory_uses_config_directory(self):
        folder = self.root / "config"
        folder.mkdir()
        path = folder / "settings.toml"
        path.write_text('[analysis]\ncache_dir = "../cache/results"\n')
        configuration = load_config(path)
        self.assertEqual(configuration["analysis"]["cache_dir"], str((self.root / "cache/results").resolve()))
        self.assertFalse((self.root / "cache").exists(), "Loading configuration must not create cache directories")

    def test_absolute_cache_directory_remains_absolute(self):
        destination = str((self.root / "absolute-cache").resolve())
        configuration = self.load_text(f'[analysis]\ncache_dir = "{destination}"\n')
        self.assertEqual(configuration["analysis"]["cache_dir"], destination)

    def test_environment_variables_cannot_supply_or_override_settings(self):
        environment = {"OPENAI_API_KEY": "environment-secret", "OPENAI_MODEL": "environment-model",
                       "OPENAI_BASE_URL": "https://environment.invalid/v1", "EPUB_AUDIOBOOK_MODEL": "also-wrong",
                       "EPUB_AUDIOBOOK_CONFIG": str(self.root / "other.toml"),
                       "EPUB_AUDIOBOOK_CACHE_DIR": "/environment-cache"}
        with patch.dict(os.environ, environment, clear=True), patch("epub_audiobook.config.Path.cwd", return_value=self.root):
            defaults = load_config()
            configured = self.load_text('[provider]\nmodel = "file-model"\napi_key = "file-secret"\n')
        self.assertIsNone(defaults["provider"]["model"])
        self.assertIsNone(defaults["provider"]["api_key"])
        self.assertEqual(defaults["provider"]["base_url"], "https://api.openai.com/v1")
        self.assertIsNone(defaults["analysis"]["cache_dir"])
        self.assertEqual(configured["provider"]["model"], "file-model")
        self.assertEqual(configured["provider"]["api_key"], "file-secret")

    def test_values_are_literal_without_environment_or_home_interpolation(self):
        with patch.dict(os.environ, {"API_TOKEN": "secret-from-env", "CACHE_ROOT": "/elsewhere", "HOME": "/other-home"}):
            configuration = self.load_text('''
[provider]
api_key = "${API_TOKEN}"
model = "$MODEL"
[analysis]
cache_dir = "${CACHE_ROOT}/~/cache"
''')
        self.assertEqual(configuration["provider"]["api_key"], "${API_TOKEN}")
        self.assertEqual(configuration["provider"]["model"], "$MODEL")
        self.assertEqual(configuration["analysis"]["cache_dir"], str((self.root / "${CACHE_ROOT}/~/cache").resolve()))

    def test_syntax_errors_never_echo_secret_values_or_lines(self):
        for text in ('[provider]\napi_key = "sk-secret-syntax\n',
                     '[provider]\napi_key = "sk-secret-syntax" trailing-secret\n',
                     '[provider]\napi_key = "sk-secret-syntax"\napi_key = "other-secret"\n'):
            with self.subTest(text=text), self.assertRaises(ValueError) as captured:
                self.load_text(text)
            self.assertEqual(str(captured.exception), "Invalid TOML configuration.")
            self.assertIsNone(captured.exception.__cause__)
            self.assertTrue(captured.exception.__suppress_context__)

    def test_invalid_utf8_is_reported_without_contents(self):
        self.path.write_bytes(b'[provider]\napi_key="sk-secret-\xff"')
        with self.assertRaisesRegex(ValueError, "Invalid TOML configuration") as captured:
            load_config(self.path)
        self.assertNotIn("sk-secret", str(captured.exception))

    def test_unknown_sections_keys_and_non_tables_are_rejected_without_echoing_keys(self):
        for text, message in [
            ('[secret_unknown_section]\nvalue="secret-value"', "Unknown configuration section"),
            ('[provider]\nsecret_unknown_key="secret-value"', "Unknown key in configuration provider"),
            ('[analysis]\nsecret_unknown_key="secret-value"', "Unknown key in configuration analysis"),
            ('[extraction]\nsecret_unknown_key="secret-value"', "Unknown key in configuration extraction"),
            ('[performance_policy]\nsecret_unknown_key="secret-value"', "Unknown key in configuration performance_policy"),
            ('provider="secret-value"', "provider must be a table"),
            ('analysis=[]', "analysis must be a table"),
            ('extraction=true', "extraction must be a table"),
            ('performance_policy=5', "performance_policy must be a table"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message) as captured:
                self.load_text(text)
            self.assertNotIn("secret", str(captured.exception))

    def test_known_fields_reject_wrong_types_empty_strings_and_invalid_limits(self):
        cases = {
            "provider": {
                "model": ['""', '"   "', "true", "[]"],
                "base_url": ['""', "false"],
                "api_key": ['""', "123", "{}"],
                "timeout": ["0", "-1", "true", '"30"', "inf", "-inf", "nan"],
                "json_mode": ["1", '"true"'],
                "max_output_tokens": ["0", "-1", "true", "1.5"],
                "token_limit_field": ['"unsupported-secret"', "true", "[]"],
                "max_retries": ["-1", "11", "true", "1.5"],
                "retry_initial_delay": ["0", "61", "true", '"5"', "nan", "inf"],
                "retry_max_delay": ["0", "61", "true", '"60"', "nan", "inf"],
            },
            "analysis": {
                "chunk_chars": ["255", "true", "256.0"],
                "max_request_chars": ["0", "-1", "true", "1.5"],
                "repair_attempts": ["-1", "6", "true", "1.0"],
                "identity_consensus": ["0", "1", '"true"', "[]"],
                "attribution_review": ["0", "1", '"true"', "[]"],
                "attribution_review_chunk_chars": ["255", "true", "60000.0"],
                "attribution_review_notes": ["false", "1", "[]"],
                "cache_dir": ['""', '"   "', "false", "[]"],
            },
            "extraction": {"include_auxiliary": ["1", '"false"']},
            "performance_policy": {"thoughts": ['""', "false", "[]"]},
        }
        for section, fields in cases.items():
            for key, values in fields.items():
                for value in values:
                    with self.subTest(section=section, key=key, value=value), self.assertRaisesRegex(ValueError, f"{section}\\.{key}") as captured:
                        self.load_text(f"[{section}]\n{key} = {value}\n")
                    self.assertNotIn("unsupported-secret", str(captured.exception))

    def test_timeout_overflow_is_a_safe_validation_error(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            self.load_text("[provider]\ntimeout = " + "9" * 1000)

    def test_retry_settings_are_file_configured_and_bounds_are_consistent(self):
        configuration = self.load_text("[provider]\nmax_retries=5\nretry_initial_delay=3\nretry_max_delay=45\n")
        self.assertEqual(configuration["provider"]["max_retries"], 5)
        self.assertEqual(configuration["provider"]["retry_initial_delay"], 3.0)
        self.assertEqual(configuration["provider"]["retry_max_delay"], 45.0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self.load_text("[provider]\nretry_initial_delay=30\nretry_max_delay=10\n")

    def test_valid_integer_limits_and_numeric_timeout_are_accepted(self):
        configuration = self.load_text('''
[provider]
timeout = 1
max_output_tokens = 1
[analysis]
chunk_chars = 256
max_request_chars = 1
repair_attempts = 0
''')
        self.assertEqual(configuration["provider"]["timeout"], 1.0)
        self.assertIs(type(configuration["provider"]["timeout"]), float)
        self.assertEqual(configuration["analysis"]["repair_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
