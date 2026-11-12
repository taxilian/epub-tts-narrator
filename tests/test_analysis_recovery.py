"""Published-book recovery restores only verified baselines and preserves drift."""

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (_completed_directory, continuation_directories,
    fresh_registry, publish_directory)
from epub_tts_narrator.analysis_recovery import JsonValue, RECOVERY_DIRNAME, recover_analysis_directory
from epub_tts_narrator.analysis_revision import _install, revise_directories
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.series import link_analysis
from epub_tts_narrator.source_epub import source_epub_hint
from tests.shared_fixtures import make_serial_epub, snapshot
from tests.test_book_update import CHAPTER_ONE, CHAPTER_THREE, SerialClient
from tests.test_series import fixture


class AnalysisRecoveryTests(unittest.TestCase):
    root: Path = Path()
    book: Path = Path()
    baseline: dict[str, bytes] = {}
    baseline_source: dict[str, JsonValue] = {}
    config: Path = Path()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.book = self.root / "book-one"
        document = link_analysis(fixture(), fresh_registry(self.book))
        publish_directory(document, fresh_registry(self.book), self.book, export_bundle)
        self.make_snapshot(self.book)
        self.baseline = self.managed(self.book)
        self.baseline_source = deepcopy(_completed_directory(self.book)[0]["source"])
        self.config = self.root / "epub-tts-narrator.toml"
        self.config.write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nidentity_consensus = false\nattribution_review = false\n'
            'repair_attempts = 0\nattribution_workers = 1\n', encoding="utf-8")

    def recovered(self, directory, source):
        result = recover_analysis_directory(directory, source)
        if result is None:
            self.fail("Expected the inconsistent book to be recovered")
        return result

    def make_snapshot(self, book):
        document = deepcopy(_completed_directory(book)[0])
        result = revise_directories({book: document}, directories=[book], scope="source-edit", progress=lambda _: None)
        return Path(result["backup_directory"])

    def managed(self, book):
        manifest = json.loads((book / "manifest.json").read_text())
        names = ["analysis.json", "series-cast.json", "source.json", "cast.json", "review.json",
                 *[row["path"] for row in manifest["files"]["scripts"]], "manifest.json"]
        return {name: (book / name).read_bytes() for name in names}

    def drift(self, book=None):
        book = self.book if book is None else book
        original_source = (book / "source.json").read_bytes()
        document = deepcopy(_completed_directory(book)[0])
        block = next(block for block in document["source"]["blocks"] if block["text"] == "They left.")
        block["text"] = "They leIt."
        for segment in document["segments"]:
            if segment["block_id"] == block["id"]:
                segment["text"] = block["text"][segment["start"]:segment["end"]]
        self.assertEqual(validate_document(document), [])
        exported = self.root / f"{book.name}-drift-export"
        export_bundle(document, exported, allow_review=True)
        write_json(exported / "analysis.json", document)
        names = ["source.json", "analysis.json",
                 *[f"scripts/{path.name}" for path in (book / "scripts").glob("*.jsonl")]]
        for name in names:
            shutil.copyfile(exported / name, book / name)
        self.assertNotEqual((book / "source.json").read_bytes(), original_source)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            _completed_directory(book)
        return document

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                 patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=SerialClient):
                try:
                    status = main([str(arg) for arg in args])
                except SystemExit as error:
                    status = error.code
        finally:
            os.chdir(previous)
        return status, stdout.getvalue(), stderr.getvalue()

    def epub(self, name, chapters, identifier):
        return make_serial_epub(self.root / f"{name}.epub", "The Serial", identifier, chapters)

    def test_valid_book_is_a_read_only_no_op(self):
        before = snapshot(self.book)
        self.assertIsNone(recover_analysis_directory(self.book, _completed_directory(self.book)[0]["source"]))
        self.assertEqual(snapshot(self.book), before)

    def test_recovery_restores_a_verified_snapshot_and_preserves_every_drifted_byte(self):
        older = sorted((self.book / ".analysis-revisions").iterdir())[-1]
        newer = self.make_snapshot(self.book)
        self.baseline = self.managed(self.book)
        drifted = self.drift()
        drift_bytes = self.managed(self.book)
        (older / "book-0001/after/source.json").write_text("tampered")

        result = self.recovered(self.book, _completed_directory(newer / "book-0001/after")[0]["source"])

        self.assertEqual(result.baseline_revision, newer)
        self.assertFalse(result.source_matches_preserved_edit)
        self.assertEqual(set(result.preserved_files), {"analysis.json", "source.json", "scripts/chapter-0001.jsonl"})
        self.assertEqual(self.managed(self.book), self.baseline)
        for name, data in drift_bytes.items():
            self.assertEqual((result.backup_directory / "before" / name).read_bytes(), data)
        self.assertTrue((result.backup_directory / "recovery.json").is_file())
        self.assertEqual(json.loads((result.backup_directory / "recovery.json").read_text())["status"], "complete")
        _completed_directory(self.book)
        self.assertEqual(drifted["source"], json.loads((result.backup_directory / "before/source.json").read_text()))

    def test_matching_authoritative_source_is_disclosed_but_never_manifest_patched(self):
        drifted = self.drift()
        result = self.recovered(self.book, drifted["source"])
        self.assertTrue(result.source_matches_preserved_edit)
        self.assertEqual(self.managed(self.book), self.baseline)
        self.assertNotEqual((result.backup_directory / "before/analysis.json").read_bytes(), self.baseline["analysis.json"])

    def test_missing_or_tampered_baseline_is_refused_without_replacing_drift(self):
        for tamper in ("missing", "tampered"):
            with self.subTest(tamper=tamper):
                book = self.root / tamper
                document = link_analysis(fixture(), fresh_registry(book))
                publish_directory(document, fresh_registry(book), book, export_bundle)
                revision = self.make_snapshot(book)
                if tamper == "missing":
                    (revision / "book-0001/after/analysis.json").unlink()
                else:
                    (revision / "book-0001/after/analysis.json").write_text("tampered")
                self.drift(book)
                before = self.managed(book)
                with self.assertRaisesRegex(ValueError, "No verified publication snapshot"):
                    recover_analysis_directory(book, document["source"])
                self.assertEqual(self.managed(book), before)
                self.assertFalse((book / RECOVERY_DIRNAME).exists())

    def test_unrelated_cast_corruption_is_not_recoverable(self):
        self.drift()
        cast = json.loads((self.book / "cast.json").read_text())
        cast["voices"][0]["description"] = "Unrelated cast corruption"
        write_json(self.book / "cast.json", cast, overwrite=True)
        before = self.managed(self.book)
        with self.assertRaisesRegex(ValueError, "only coherent"):
            recover_analysis_directory(self.book, self.baseline_source)
        self.assertEqual(self.managed(self.book), before)
        self.assertFalse((self.book / RECOVERY_DIRNAME).exists())

    def test_incompatible_authoritative_source_is_refused_before_backup_or_replacement(self):
        drifted = self.drift()
        incompatible = deepcopy(drifted["source"])
        incompatible["chapters"].pop()
        before = self.managed(self.book)
        with self.assertRaisesRegex(ValueError, "missing from the new EPUB"):
            recover_analysis_directory(self.book, incompatible)
        self.assertEqual(self.managed(self.book), before)
        self.assertFalse((self.book / RECOVERY_DIRNAME).exists())

    def test_active_render_and_update_ownership_block_recovery(self):
        self.drift()
        before = self.managed(self.book)
        (self.book / ".audio-render.lock").mkdir()
        with self.assertRaisesRegex(ValueError, "Rendering is active"):
            recover_analysis_directory(self.book, self.baseline_source)
        (self.book / ".audio-render.lock").rmdir()
        from epub_tts_narrator.renderer.run_locks import acquire_directory_lock
        lock = acquire_directory_lock(self.book / ".analysis-update.lock")
        try:
            with self.assertRaisesRegex(ValueError, "revision is active"):
                recover_analysis_directory(self.book, self.baseline_source)
        finally:
            lock.rmdir()
        self.assertEqual(self.managed(self.book), before)
        self.assertFalse((self.book / RECOVERY_DIRNAME).exists())

    def test_interrupted_install_resumes_from_its_durable_backup(self):
        self.drift()
        drift_bytes = self.managed(self.book)
        destination_book = self.book

        def replace_then_interrupt(source, destination):
            _install(source, destination)
            if Path(destination) == destination_book / "source.json":
                raise KeyboardInterrupt()

        with patch("epub_tts_narrator.analysis_recovery._install", side_effect=replace_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                recover_analysis_directory(self.book, self.baseline_source)
        transaction = next((self.book / RECOVERY_DIRNAME).iterdir())
        self.assertEqual(json.loads((transaction / "recovery.json").read_text())["status"], "applying")
        for name, data in drift_bytes.items():
            self.assertEqual((transaction / "before" / name).read_bytes(), data)
        with self.assertRaises(ValueError):
            _completed_directory(self.book)

        result = self.recovered(self.book, self.baseline_source)

        self.assertEqual(result.backup_directory, transaction)
        self.assertEqual(self.managed(self.book), self.baseline)
        self.assertEqual(json.loads((transaction / "recovery.json").read_text())["status"], "complete")
        _completed_directory(self.book)

    def test_analyze_backs_up_stale_drift_and_reconciles_the_baseline_epub(self):
        parent = self.root / "baseline-parent"
        original_epub = self.epub("baseline-original", [CHAPTER_ONE, CHAPTER_THREE], "baseline-book")
        status, _, stderr = self.run_cli("analyze", "--strategy", "consensus", original_epub, "-o", parent)
        self.assertEqual(status, 0, stderr)
        self.make_snapshot(parent)
        baseline = self.managed(parent)
        self.drift(parent)
        drift_bytes = self.managed(parent)

        status, stdout, stderr = self.run_cli("analyze", "--strategy", "consensus", original_epub, "-o", parent)

        self.assertEqual(status, 0, stderr + stdout)
        transaction = next((parent / RECOVERY_DIRNAME).iterdir())
        self.assertIn(str(transaction), stdout)
        self.assertIn("does not contain every preserved source edit", stdout)
        self.assertEqual(self.managed(parent), baseline)
        for name, data in drift_bytes.items():
            self.assertEqual((transaction / "before" / name).read_bytes(), data)
        _completed_directory(parent)

    def test_directory_only_recovery_failure_requests_an_authoritative_epub(self):
        self.drift()

        status, _, stderr = self.run_cli("analyze", "--strategy", "consensus", self.book, "-o", self.book)

        self.assertEqual(status, 1)
        self.assertIn("supply the authoritative EPUB", stderr)
        self.assertIn(f"source.json in {self.book}", stderr)
        self.assertFalse((self.book / RECOVERY_DIRNAME).exists())

    def test_recovered_parent_accepts_epub_edit_and_rebinds_descendant_through_normal_update(self):
        parent = self.root / "epub-parent"
        child = self.root / "epub-child"
        original_epub = self.epub("original", [CHAPTER_ONE, CHAPTER_THREE], "book-one")
        status, _, stderr = self.run_cli("analyze", "--strategy", "consensus", original_epub, "-o", parent)
        self.assertEqual(status, 0, stderr)
        self.make_snapshot(parent)
        status, _, stderr = self.run_cli(
            "analyze", self.epub("continuation", [("Four", ["Mara said, “Morning.”"])], "book-two"),
            "-o", child, "--continue-from", parent, "--strategy", "consensus")
        self.assertEqual(status, 0, stderr)
        original = _completed_directory(parent)[0]
        child_before = {name: (child / name).read_bytes() for name in
                        ["analysis.json", *[f"scripts/{path.name}" for path in (child / "scripts").glob("*.jsonl")]]}
        self.drift(parent)
        drift_bytes = self.managed(parent)
        fixed_epub = self.epub("fixed", [("One", [CHAPTER_ONE[1][0], "They leIt."]), CHAPTER_THREE], "book-one-fixed")

        status, stdout, stderr = self.run_cli("analyze", "--strategy", "consensus", fixed_epub, "-o", parent)

        self.assertEqual(status, 0, stderr + stdout)
        transaction = next((parent / RECOVERY_DIRNAME).iterdir())
        self.assertIn(str(transaction), stdout)
        self.assertIn("contains the preserved source edit", stdout)
        for name, data in drift_bytes.items():
            self.assertEqual((transaction / "before" / name).read_bytes(), data)
        updated, _, _ = _completed_directory(parent)
        _completed_directory(child)
        self.assertEqual(continuation_directories(child), [parent.resolve()])
        self.assertIn("They leIt.", [block["text"] for block in updated["source"]["blocks"]])
        self.assertEqual(updated["source"]["book"], original["source"]["book"])
        self.assertEqual(updated["cast"], original["cast"])
        for name, data in child_before.items():
            self.assertEqual((child / name).read_bytes(), data)
        self.assertEqual(source_epub_hint(parent, original["source"]["book"]["id"]), fixed_epub.resolve())


if __name__ == "__main__":
    unittest.main()
