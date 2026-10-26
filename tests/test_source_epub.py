"""Original EPUB discovery must not invalidate source, voices, or audio caches."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
from epub_tts_narrator.analysis_revision import revise_directories
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.series import link_analysis
from epub_tts_narrator.source_epub import SOURCE_EPUB_HINT, remember_source_epub, source_epub_hint
from test_series import fixture


class SourceEpubTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.epub = self.root / "original.epub"
        self.epub.write_bytes(b"checksum-bound EPUB fixture")
        self.book_id = hashlib.sha256(self.epub.read_bytes()).hexdigest()
        self.directory = self.root / "book"
        self.document = fixture()
        self.document["source"]["book"]["id"] = self.book_id

    def publish(self, hint=True):
        registry = fresh_registry(self.directory)
        self.document = link_analysis(self.document, registry)
        publish_directory(self.document, registry, self.directory, export_bundle,
                          source_epub=self.epub if hint else None)
        return load_bundle(self.directory)

    def test_publication_remembers_source_without_changing_analysis_identity(self):
        source = deepcopy(self.document["source"])
        bundle = self.publish()
        self.assertEqual(bundle["source"], source)
        self.assertEqual(source_epub_hint(self.directory, self.book_id), self.epub)
        self.assertNotIn(SOURCE_EPUB_HINT, bundle["manifest"]["sha256"])
        before = (self.directory / "manifest.json").read_bytes()
        moved = self.root / "relocated.epub"
        moved.write_bytes(self.epub.read_bytes())
        remember_source_epub(self.directory, moved, self.book_id)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), before)
        self.assertEqual(load_bundle(self.directory)["bundle_hash"], bundle["bundle_hash"])

    def test_resolver_uses_saved_hint_and_retains_missing_original_for_actionable_error(self):
        bundle = self.publish()
        self.epub.unlink()
        config = resolve_book_config(bundle, deepcopy(DEFAULTS))
        self.assertEqual(config["render"]["epub"], str(self.epub))

    def test_explicit_epub_overrides_wrong_hint_without_reading_it(self):
        bundle = self.publish()
        hint = self.directory / SOURCE_EPUB_HINT
        record = json.loads(hint.read_text())
        record["book_id"] = "wrong-book"
        hint.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_book_config(bundle, deepcopy(DEFAULTS))
        config = resolve_book_config(bundle, deepcopy(DEFAULTS), epub=self.epub)
        self.assertEqual(config["render"]["epub"], str(self.epub))

    def test_legacy_bundle_without_hint_keeps_optional_epub(self):
        bundle = self.publish(hint=False)
        self.assertIsNone(resolve_book_config(bundle, deepcopy(DEFAULTS))["render"]["epub"])

    def test_wrong_source_cannot_publish_a_hint_or_completed_bundle(self):
        self.epub.write_bytes(b"different book")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.publish()
        self.assertFalse(self.directory.exists())

    def test_hint_symlink_and_nonregular_file_are_rejected(self):
        self.publish(hint=False)
        hint = self.directory / SOURCE_EPUB_HINT
        hint.symlink_to(self.root / "outside.json")
        with self.assertRaisesRegex(ValueError, "regular"):
            source_epub_hint(self.directory, self.book_id)
        with self.assertRaisesRegex(ValueError, "symlink"):
            remember_source_epub(self.directory, self.epub, self.book_id)
        self.assertFalse((self.root / "outside.json").exists())

    def test_cli_reexport_carries_hint_and_revisions_leave_it_unchanged(self):
        self.publish()
        saved = (self.directory / SOURCE_EPUB_HINT).read_bytes()
        output = self.root / "reexported"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main(["export", str(self.directory / "analysis.json"), "-o", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(source_epub_hint(output, self.book_id), self.epub)
        revised = deepcopy(self.document)
        revised["provenance"]["fixture_revision"] = True
        revise_directories({self.directory: revised}, progress=lambda _: None)
        self.assertEqual((self.directory / SOURCE_EPUB_HINT).read_bytes(), saved)

    def test_cli_reexport_preserves_a_stale_hint_without_reopening_original(self):
        bundle = self.publish()
        moved = self.root / "moved-original.epub"
        self.epub.rename(moved)
        output = self.root / "reexported"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = main(["export", str(self.directory / "analysis.json"), "-o", str(output)])
        self.assertEqual(result, 0)
        self.assertEqual(source_epub_hint(output, self.book_id), self.epub)
        self.assertEqual(load_bundle(output)["source"], bundle["source"])
        resolved = resolve_book_config(load_bundle(output), deepcopy(DEFAULTS))
        self.assertEqual(resolved["render"]["epub"], str(self.epub))
        with self.assertRaises(FileNotFoundError):
            remember_source_epub(output, self.epub, self.book_id)

    def test_resuming_old_bundle_with_original_epub_remembers_location(self):
        self.publish(hint=False)
        from epub_tts_narrator.pronunciations import save_candidates
        save_candidates(self.directory, self.document, [])
        config = self.root / "config.toml"
        config.write_text("")
        with (patch("epub_tts_narrator.cli.source_from", return_value=deepcopy(self.document["source"])),
              patch("epub_tts_narrator.narration_boundaries.repair_narration_boundaries", side_effect=lambda document, **_: document),
              patch("epub_tts_narrator.speech_normalization.normalization_pending", return_value=False),
              patch("epub_tts_narrator.audition_scripts.audition_preparation_pending", return_value=False),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=AssertionError("No model calls")),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            result = main(["analyze", str(self.epub), "-o", str(self.directory), "--config", str(config)])
        self.assertEqual(result, 0)
        self.assertEqual(source_epub_hint(self.directory, self.book_id), self.epub)
