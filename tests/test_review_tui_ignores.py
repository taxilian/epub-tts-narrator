"""Chapter ignore/restore controls in the published-book review UI."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator import chapter_ignores
from epub_tts_narrator.analysis_revision import revise_directories
from epub_tts_narrator.review_tui import ReviewApp
from epub_tts_narrator.review_tui_model import ignored_empty_chapters
from tests.test_review_tui import (
    FakeScreen, RevisionStub, add_failure_status, publish_book,
)


class ChapterIgnoreTests(unittest.TestCase):
    root: Path = Path()
    _app_counter = 0

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self._app_counter = 0

    def app(self, *, revise=None):
        """Publish a book whose partitions are Cover(0, empty), One(1), Two(2)."""
        self._app_counter += 1
        root = self.root / f"book-{self._app_counter}"
        root.mkdir(parents=True, exist_ok=True)
        directory, document = publish_book(root, two_chapters=True, empty_chapters=((0, "Cover"),))
        revision = RevisionStub(directory) if revise is None else revise
        app = ReviewApp(directory, document, revise_directories=revision,
                        related_directories=lambda directories: [directory])
        return app, revision

    def ignore_selected(self, app):
        app.handle_key("i")
        app.handle_key("y")

    def make_pending_edit(self, app):
        app.handle_key("ENTER")
        app.handle_key("DOWN")
        app.handle_key("ENTER")
        app.handle_key("r")
        self.assertIn("s2", app.pending)
        app.handle_key("ESC")
        app.handle_key("ESC")
        self.assertEqual(app.view, "toc")

    def test_ignore_first_chapter_confirms_saves_immediately_and_hides_it(self):
        app, revision = self.app()
        app.handle_key("i")
        screen = FakeScreen()
        app.draw(screen)
        self.assertEqual(app.view, "confirm_ignore")
        self.assertIn("Ignore chapter 2?", screen.text)

        app.handle_key("y")

        sidecar = app.directory / chapter_ignores.FILENAME
        self.assertTrue(sidecar.is_file())
        self.assertEqual(chapter_ignores.ignored_indices(app.directory, app.document), frozenset({1}))
        self.assertEqual(app.ignored_chapters, frozenset({1}))
        self.assertEqual([row["id"] for row in app._toc_rows()], [2])
        self.assertEqual(app.view, "toc")
        self.assertEqual(revision.calls, [])
        self.assertEqual(app.pending, {})
        self.assertIn("next render and assemble", app.notice)
        drawn = FakeScreen()
        app.draw(drawn)
        self.assertIn("1/1", drawn.text)
        self.assertNotIn("Chapter One", drawn.text)

        reloaded = ReviewApp(app.directory, app.document, revise_directories=RevisionStub(app.directory),
                             related_directories=lambda directories: [app.directory])
        self.assertEqual(reloaded.ignored_chapters, frozenset({1}))
        self.assertEqual([row["id"] for row in reloaded._toc_rows()], [2])

    def test_escape_or_n_cancels_the_confirmation_without_saving_anything(self):
        for key in ("ESC", "n", "q"):
            with self.subTest(key=key):
                app, _ = self.app()
                app.handle_key("i")
                app.handle_key(key)
                self.assertEqual(app.view, "toc")
                self.assertFalse((app.directory / chapter_ignores.FILENAME).exists())
                self.assertEqual(app.ignored_chapters, frozenset())
                screen = FakeScreen()
                app.draw(screen)
                self.assertIn("Chapter One", screen.text)
                self.assertIn("Chapter Two", screen.text)

    def test_ignore_middle_chapter_keeps_original_ids_for_failure_markers(self):
        app, _ = self.app()
        add_failure_status(app.directory, 3, error_type="SpeechVerificationFailed",
                           error="Verification exhausted")
        app.handle_key("DOWN")
        self.assertEqual(app.chapter_index, 2)

        self.ignore_selected(app)

        self.assertEqual(chapter_ignores.ignored_indices(app.directory, app.document), frozenset({2}))
        self.assertEqual([row["id"] for row in app._toc_rows()], [1])
        screen = FakeScreen()
        app.draw(screen)
        self.assertIn("Chapter One", screen.text)
        self.assertNotIn("FAILED", screen.text)

        app.handle_key("h")
        shown = FakeScreen()
        app.draw(shown)
        self.assertIn("IGNORED", shown.text)
        self.assertIn("FAILED SpeechVerificationFailed", shown.text)

    def test_show_mode_labels_ignored_and_restore_returns_the_original_row(self):
        app, _ = self.app()
        self.ignore_selected(app)
        self.assertEqual(app.ignored_chapters, frozenset({1}))

        app.handle_key("h")
        screen = FakeScreen()
        app.draw(screen)
        self.assertIn("IGNORED", screen.text)
        self.assertIn("2/2", screen.text)
        ignored_row = next(row for row in app._toc_rows() if row["ignored"])
        self.assertEqual(ignored_row["id"], 1)

        app.handle_key("UP")
        app.handle_key("i")
        confirm = FakeScreen()
        app.draw(confirm)
        self.assertIn("Restore chapter 2?", confirm.text)
        app.handle_key("y")

        self.assertEqual(chapter_ignores.ignored_indices(app.directory, app.document), frozenset())
        self.assertEqual(app.ignored_chapters, frozenset())
        self.assertEqual([row["id"] for row in app._toc_rows()], [1, 2])
        self.assertIn("restored", app.notice)

    def test_stale_manifest_hash_surfaces_the_error_and_changes_nothing(self):
        app, _ = self.app()
        self.ignore_selected(app)
        sidecar = app.directory / chapter_ignores.FILENAME
        original_sidecar = sidecar.read_bytes()
        manifest_path = app.directory / "manifest.json"
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["revision"] = "changed-after-load"
        manifest_path.write_text(json.dumps(record), encoding="utf-8")

        self.ignore_selected(app)

        self.assertEqual(sidecar.read_bytes(), original_sidecar)
        self.assertEqual(chapter_ignores.ignored_indices(app.directory, app.document), frozenset({1}))
        self.assertEqual(app.ignored_chapters, frozenset({1}))
        self.assertIn("Ignore save failed:", app.notice)
        self.assertIn("changed after it was loaded", app.notice)
        self.assertEqual([row["id"] for row in app._toc_rows()], [2])
        app.handle_key("ENTER")
        self.assertEqual(app.view, "passages")

    def test_pending_attribution_edits_survive_an_immediate_ignore_save(self):
        app, revision = self.app()
        self.make_pending_edit(app)

        self.ignore_selected(app)

        self.assertEqual(chapter_ignores.ignored_indices(app.directory, app.document), frozenset({1}))
        self.assertIn("s2", app.pending)
        self.assertEqual(revision.calls, [])
        self.assertIn("unsaved attribution edits are unchanged", app.notice)

    def test_ignore_save_keeps_original_receipts_so_a_manifest_race_blocks_attribution(self):
        app, _ = self.app(revise=revise_directories)
        self.make_pending_edit(app)
        original_hashes = dict(app.expected_manifest_hashes)
        original_canonical_hash = app.canonical_manifest_hash
        manifest_path = app.directory / "manifest.json"
        real_set_ignored = chapter_ignores.set_ignored

        def change_manifest_after_save(*args, **kwargs):
            updated = real_set_ignored(*args, **kwargs)
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["revision"] = "changed-after-ignore"
            manifest_path.write_text(json.dumps(record), encoding="utf-8")
            return updated

        with patch("epub_tts_narrator.chapter_ignores.set_ignored", side_effect=change_manifest_after_save):
            self.ignore_selected(app)
        changed_manifest = manifest_path.read_bytes()

        self.assertEqual(app.expected_manifest_hashes, original_hashes)
        self.assertEqual(app.canonical_manifest_hash, original_canonical_hash)
        self.assertEqual(chapter_ignores.ignored_indices(app.directory, app.document), frozenset({1}))

        self.assertFalse(app.save())

        self.assertIn("Save failed:", app.notice)
        self.assertIn("changed after this change was prepared", app.notice)
        self.assertIn("s2", app.pending)
        self.assertEqual(manifest_path.read_bytes(), changed_manifest)

    def test_all_chapters_ignored_draws_a_safe_screen_and_h_restores_the_list(self):
        app, _ = self.app()
        self.ignore_selected(app)
        self.ignore_selected(app)
        self.assertEqual(app.ignored_chapters, frozenset({1, 2}))
        self.assertEqual(app._toc_rows(), [])

        screen = FakeScreen()
        app.draw(screen)
        self.assertIn("0/0", screen.text)
        self.assertIn("All chapters are ignored", screen.text)
        self.assertIn("Press H", screen.text)

        app.handle_key("i")
        self.assertEqual(app.view, "toc")
        app.handle_key("h")
        shown = FakeScreen()
        app.draw(shown)
        self.assertEqual(shown.text.count("IGNORED"), 2)
        app.handle_key("h")
        hidden = FakeScreen()
        app.draw(hidden)
        self.assertIn("All chapters are ignored", hidden.text)

    def test_selection_follows_the_original_chapter_across_hide_show_cycles(self):
        app, _ = self.app()
        app.handle_key("DOWN")
        self.assertEqual(app.chapter_index, 2)

        app.handle_key("h")
        app.handle_key("h")

        self.assertEqual(app.chapter_index, 2)
        rows = app._toc_rows()
        self.assertEqual(rows[app.toc_index]["id"], 2)

    def test_an_ignored_chapter_that_lost_its_passages_stays_restorable(self):
        app, _ = self.app()
        self.ignore_selected(app)
        kept = [segment for segment in app.document["segments"]
                if segment["block_id"] not in {"b1", "b2"}]
        app.document["segments"] = kept

        hidden = FakeScreen()
        app.draw(hidden)
        self.assertNotIn("Chapter One", hidden.text)

        app.handle_key("h")
        shown = FakeScreen()
        app.draw(shown)
        self.assertIn("IGNORED", shown.text)
        self.assertIn("Chapter One", shown.text)
        rows = [row for row in app._toc_rows() if row["id"] == 1]
        self.assertEqual(rows[0]["passage_count"], 0)

        app.handle_key("UP")
        app.handle_key("i")
        app.handle_key("y")
        self.assertEqual(app.ignored_chapters, frozenset())
        self.assertEqual(ignored_empty_chapters(app.document, frozenset()), [])
        cleared = FakeScreen()
        app.draw(cleared)
        self.assertNotIn("Chapter One", cleared.text)

    def test_ignore_save_error_leaves_state_usable_and_data_unchanged(self):
        app, _ = self.app()
        with patch("epub_tts_narrator.chapter_ignores.set_ignored", side_effect=OSError("disk full")):
            self.ignore_selected(app)

        self.assertIn("Ignore save failed: disk full", app.notice)
        self.assertFalse((app.directory / chapter_ignores.FILENAME).exists())
        self.assertEqual(app.ignored_chapters, frozenset())
        self.assertEqual(app.view, "toc")
        app.handle_key("ENTER")
        self.assertEqual(app.view, "passages")

    def test_toc_footer_lists_the_ignore_and_show_keys(self):
        app, _ = self.app()
        screen = FakeScreen()
        app.draw(screen)
        footer = next(row["text"] for row in screen.current if "ignore/restore" in row["text"])
        self.assertIn("I ignore/restore", footer)
        self.assertIn("H ignored", footer)


if __name__ == "__main__":
    unittest.main()
