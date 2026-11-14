"""Render entry points freeze chapter choices before model readiness work."""

import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator import chapter_ignores
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.render_queue import _chapter_outline, _prepare_book
from epub_tts_narrator.renderer.speech_traits import save_accent
from audition_fixtures import prepare_document
from test_chapter_ignores import exported_book, multi_chapter_document


class ChapterIgnoreEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        document = prepare_document(multi_chapter_document())
        self.directory = exported_book(document, self.root)
        self.bundle = load_bundle(self.directory)
        self.config = self.root / "renderer.toml"
        self.config.write_text('[provider]\nmodel="offline-fixture"\n'
                               '[engine]\ncheckpoint="./tts-model"\n'
                               '[verification]\nmodel="./asr-model"\n'
                               '[voices]\nseries_id="test-series"\n')
        settings = resolve_book_config(self.bundle, load_config(self.config))
        save_accent(self.bundle, settings, "vn", "")
        self.document = chapter_ignores.bundle_document(self.bundle)
        chapter_ignores.set_ignored(self.directory, self.document, 1, True,
                                    expected_manifest_hash=self.bundle["bundle_hash"])

    def test_queued_outline_omits_ignored_chapters_with_original_numbers(self):
        outline = _chapter_outline(self.directory)

        self.assertEqual([chapter["number"] for chapter in outline], [1, 3])

    def test_queue_preparation_needs_no_accent_or_model_for_an_ignored_unique_voice(self):
        args = argparse.Namespace(config=self.config, chapters_only=True)
        with patch("epub_tts_narrator.provider.OpenAICompatibleClient",
                   side_effect=AssertionError("No model request is needed")):
            _, settings, plan = _prepare_book(self.directory, args, notify=lambda _: None)

        self.assertEqual(settings[chapter_ignores.SNAPSHOT_KEY], frozenset({1}))
        self.assertEqual(plan["voice_ids"], ["vn"])
        self.assertTrue(plan["chapters"][1]["ignored"])
        chapter_ignores.set_ignored(self.directory, self.document, 1, False,
                                    expected_manifest_hash=self.bundle["bundle_hash"])
        with self.assertRaisesRegex(ValueError, "no accent yet"):
            _prepare_book(self.directory, args, notify=lambda _: None)

    def test_single_book_render_skips_ignored_voice_readiness_before_synthesis(self):
        output, error = io.StringIO(), io.StringIO()
        with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(error),
              patch("epub_tts_narrator.provider.OpenAICompatibleClient",
                    side_effect=AssertionError("No model request is needed")),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", return_value={"chapters": []}) as render):
            status = main(["render", str(self.directory), "--config", str(self.config), "--chapters-only"])

        self.assertEqual(status, 0, error.getvalue())
        render.assert_called_once()
        self.assertEqual(render.call_args.args[1][chapter_ignores.SNAPSHOT_KEY], frozenset({1}))
