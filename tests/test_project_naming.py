"""The renamed command preserves existing settings and protects both filenames."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.config import load_config as load_renderer_config
from epub_tts_narrator.cli import main, parser
from epub_tts_narrator.config import (
    DEFAULT_FILENAME, LEGACY_FILENAME, load_config, resolve_config_path,
)
from epub_tts_narrator.pipeline import write_json
from test_contracts import valid_document


class ProjectNamingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.enterContext(contextlib.chdir(self.root))

    def settings(self, filename, label):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'[provider]\nmodel = "{label}"\n'
            f'[analysis]\ncache_dir = "{label}-cache"\n'
            f'[engine]\ncheckpoint = "{label}-engine"\n'
            f'[verification]\nmodel = "{label}-whisper"\n', encoding="utf-8")
        return path

    def assert_selected(self, label, path=None):
        analyzer = load_config(path)
        renderer = load_renderer_config(path)
        base = Path(path).parent if path is not None else self.root
        self.assertEqual(analyzer["provider"]["model"], label)
        self.assertEqual(analyzer["analysis"]["cache_dir"], str((base / f"{label}-cache").resolve()))
        self.assertEqual(renderer["verification"]["model"], str((base / f"{label}-whisper").resolve()))

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main([str(arg) for arg in args])
        return status, stderr.getvalue()

    def test_both_tools_read_canonical_config(self):
        path = self.settings(DEFAULT_FILENAME, "canonical")
        self.assertEqual(resolve_config_path(), path)
        self.assert_selected("canonical")

    def test_both_tools_fall_back_to_legacy_config(self):
        path = self.settings(LEGACY_FILENAME, "legacy")
        self.assertEqual(resolve_config_path(), path)
        self.assert_selected("legacy")

    def test_canonical_wins_without_merging_legacy(self):
        self.settings(LEGACY_FILENAME, "legacy")
        self.settings(DEFAULT_FILENAME, "canonical")
        self.assert_selected("canonical")

    def test_explicit_path_overrides_both_and_retains_its_relative_paths(self):
        self.settings(LEGACY_FILENAME, "legacy")
        self.settings(DEFAULT_FILENAME, "canonical")
        explicit = self.settings("elsewhere/custom.toml", "explicit")
        self.assert_selected("explicit", explicit)
        self.assert_selected("legacy", self.root / LEGACY_FILENAME)

    def test_missing_explicit_path_does_not_fall_back(self):
        self.settings(DEFAULT_FILENAME, "canonical")
        for loader in (load_config, load_renderer_config):
            with self.subTest(loader=loader.__module__), self.assertRaises((ValueError, FileNotFoundError)):
                loader(self.root / "missing.toml")

    def test_invalid_canonical_file_does_not_fall_back(self):
        self.settings(LEGACY_FILENAME, "legacy")
        (self.root / DEFAULT_FILENAME).write_text("[broken TOML", encoding="utf-8")
        for loader in (load_config, load_renderer_config):
            with self.subTest(loader=loader.__module__), self.assertRaisesRegex(ValueError, "Invalid TOML"):
                loader()

    def test_broken_canonical_symlink_does_not_fall_back(self):
        self.settings(LEGACY_FILENAME, "legacy")
        (self.root / DEFAULT_FILENAME).symlink_to(self.root / "missing.toml")
        for loader in (load_config, load_renderer_config):
            with self.subTest(loader=loader.__module__), self.assertRaises((ValueError, FileNotFoundError)):
                loader()

    def test_outputs_cannot_replace_selected_settings_even_with_force(self):
        source = self.root / "source.json"
        analysis = self.root / "analysis.json"
        document = valid_document()
        write_json(source, {"format": "epub-audiobook-source", "version": 1, "source": document["source"]})
        write_json(analysis, document)
        for name in (LEGACY_FILENAME, DEFAULT_FILENAME, "explicit.json"):
            with self.subTest(name=name):
                path = self.settings(name, "protected")
                original = path.read_bytes()
                arguments = ["--config", path] if name == "explicit.json" else []
                for command, input_path in (("extract", source), ("analyze", source),
                                            ("review-attribution", analysis)):
                    with self.subTest(command=command), patch("epub_tts_narrator.cli.OpenAICompatibleClient") as client:
                        status, error = self.run_cli(command, input_path, "-o", path, "--force", *arguments)
                        self.assertEqual(status, 1)
                        # analyze rejects --force for a non-JSON directory target
                        # before it needs to inspect any source or settings.
                        self.assertTrue("separate path from configuration" in error or "--force applies only" in error, error)
                        self.assertEqual(path.read_bytes(), original)
                        client.assert_not_called()

    def test_series_registry_cannot_replace_selected_settings(self):
        source = self.root / "source.json"
        write_json(source, {"format": "epub-audiobook-source", "version": 1,
                            "source": valid_document()["source"]})
        for name in (LEGACY_FILENAME, DEFAULT_FILENAME, "explicit.toml"):
            with self.subTest(name=name):
                path = self.settings(name, "protected")
                original = path.read_bytes()
                arguments = ["--config", path] if name == "explicit.toml" else []
                status, error = self.run_cli("plan", source, "--series", path, *arguments)
                self.assertEqual(status, 1)
                self.assertIn("separate path from input, output, and configuration", error)
                self.assertEqual(path.read_bytes(), original)

    def test_help_identifies_the_renamed_analyzer(self):
        self.assertEqual(parser().prog, "epub-tts-narrator")
        self.assertIn("epub-tts-narrator", parser().format_help())


if __name__ == "__main__":
    unittest.main()
