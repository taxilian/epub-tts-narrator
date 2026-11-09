"""Drive the review state machine through a fake screen, never a fake curses module."""
# noqa: SIZE_OK — one FakeScreen contract covers the complete ReviewApp state machine.

import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.review_tui import ReviewApp, run_review
from epub_tts_narrator.series import link_analysis
from tests.test_contracts import valid_document


Document = dict[str, Any]


def review_document(*, two_chapters=False, unresolved=False) -> Document:
    document: Document = valid_document()
    if unresolved:
        document["segments"][1].update(role_id=None, voice_id=None)
        document["segments"][1]["attribution"].update(
            basis="unresolved", review_required=True, reason="No consensus.", candidates=["mara"])
    if two_chapters:
        document["source"]["chapters"].append({
            "id": "ch2", "title": "Chapter Two", "source_href": "text/ch2.xhtml", "block_ids": ["b3"],
        })
        document["source"]["blocks"].append({
            "id": "b3", "chapter_id": "ch2", "text": "Later, Lena waited.",
            "source": {"href": "text/ch2.xhtml", "element_id": None, "tag": "p"},
        })
        document["segments"].append({
            "id": "s4", "block_id": "b3", "start": 0, "end": 19, "text": "Later, Lena waited.",
            "kind": "narration", "role_id": "narrator", "voice_id": "vn",
            "delivery": {"direction": "", "evidence": []},
            "attribution": {"basis": "explicit", "reason": "Narrative.", "evidence": [],
                            "candidates": [], "review_required": False},
        })
    return document


def publish_book(root, *, two_chapters=False, unresolved=False):
    """Publish a self-contained temporary book through the real exporter/loader contract."""
    directory = Path(root) / "review-book"
    document = review_document(two_chapters=two_chapters, unresolved=unresolved)
    registry = fresh_registry(directory)
    linked = link_analysis(document, registry)
    publish_directory(linked, registry, directory, export_bundle)
    return directory, linked


class FakeScreen:
    def __init__(self, keys=(), *, height=24, width=110):
        self.keys = list(keys)
        self.height = height
        self.width = width
        self.frames = []
        self.current = []

    def size(self):
        return self.height, self.width

    def clear(self):
        self.current = []
        self.frames.append(self.current)

    def write(self, y, x, text, attributes):
        self.current.append({"y": y, "x": x, "text": str(text), "attributes": attributes})

    def refresh(self):
        return None

    def read_key(self):
        return self.keys.pop(0) if self.keys else None

    @property
    def text(self):
        return "\n".join(row["text"] for row in self.current)


class RevisionStub:
    def __init__(self, directory, *, error=None, rewrite_manifest=False):
        self.directory = Path(directory)
        self.error = error
        self.rewrite_manifest = rewrite_manifest
        self.calls = []

    def __call__(self, revisions, **kwargs):
        self.calls.append({"revisions": revisions, **kwargs})
        if self.error is not None:
            raise self.error
        kwargs["progress"]("Updated fixture revision")
        if self.rewrite_manifest:
            manifest = self.directory / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
        return {"backup_directory": str(self.directory / ".analysis-revisions" / "fixture")}


class ReviewAppTests(unittest.TestCase):
    root: Path = Path()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root: Path = Path(temporary.name).resolve()

    def app(self, *, two_chapters=False, unresolved=False, error=None, rewrite_manifest=False):
        directory, document = publish_book(self.root, two_chapters=two_chapters, unresolved=unresolved)
        revision = RevisionStub(directory, error=error, rewrite_manifest=rewrite_manifest)
        related_calls = []

        def related(directories):
            related_calls.append(list(directories))
            return [directory]

        app = ReviewApp(directory, document, revise_directories=revision, related_directories=related)
        return app, revision, related_calls

    def open_second_passage(self, app):
        app.handle_key("ENTER")
        app.handle_key("DOWN")
        app.handle_key("ENTER")

    def test_toc_navigation_opens_the_selected_chapter_passages(self):
        app, _, _ = self.app(two_chapters=True)
        first = FakeScreen()
        app.draw(first)

        app.handle_key("DOWN")
        second = FakeScreen()
        app.draw(second)
        app.handle_key("ENTER")
        third = FakeScreen()
        app.draw(third)

        self.assertIn("Chapter One", first.text)
        self.assertIn("Chapter Two", first.text)
        self.assertIn("> Chapter Two", second.text)
        self.assertEqual(app.view, "passages")
        self.assertIn("Later, Lena waited.", third.text)

    def test_passage_rows_use_cast_role_colors_and_review_badges(self):
        app, _, _ = self.app()
        app.document["segments"][1]["attribution"]["review_required"] = True
        app.handle_key("ENTER")
        screen = FakeScreen()

        app.draw(screen)

        narrator = next(row for row in screen.current if row["text"] == "Narrator")
        mara = next(row for row in screen.current if row["text"] == "Mara")
        self.assertEqual(narrator["attributes"]["role"], 0)
        self.assertEqual(mara["attributes"]["role"], 1)
        self.assertNotEqual(narrator["attributes"]["role"], mara["attributes"]["role"])
        self.assertTrue(any("D!" in row["text"] for row in screen.current))
        self.assertTrue(any(row["attributes"]["reverse"] for row in screen.current if row["y"] == 3))

    def test_info_dialog_displays_passage_attribution_and_delivery_metadata(self):
        app, _, _ = self.app()
        self.open_second_passage(app)
        screen = FakeScreen()

        app.draw(screen)

        self.assertEqual(app.view, "info")
        self.assertIn("Speaker(s): Mara", screen.text)
        self.assertIn("Direction: (none)", screen.text)
        self.assertIn("Attribution basis: explicit", screen.text)
        self.assertIn("Role IDs: mara", screen.text)
        self.assertIn("Voice IDs: " + app.document["segments"][1]["voice_id"], screen.text)
        self.assertIn("Voice 1 description: Proposed light, distinct voice.", screen.text)

    def test_assigned_passage_picker_offers_all_roles_and_records_user_choice(self):
        app, _, _ = self.app()
        self.open_second_passage(app)
        app.handle_key("F2")
        picker = FakeScreen()
        app.draw(picker)

        app.handle_key("UP")
        app.handle_key("ENTER")

        self.assertIn("Narrator", picker.text)
        self.assertIn("Mara", picker.text)
        edited = app.pending["s2"]
        self.assertEqual(edited["role_id"], "narrator")
        self.assertEqual(edited["kind"], "narration")
        self.assertEqual(edited["attribution"]["basis"], "user")
        self.assertEqual(app.document["segments"][1]["role_id"], "mara")
        self.assertEqual(app.view, "info")

    def test_direction_editor_applies_printable_text_and_preserves_pending_state(self):
        app, _, _ = self.app()
        self.open_second_passage(app)
        app.handle_key("F3")
        for character in "Quietly":
            app.handle_key(character)

        app.handle_key("ENTER")

        self.assertEqual(app.pending["s2"]["delivery"]["direction"], "Quietly")
        self.assertEqual(app.view, "info")
        self.assertEqual(app.document["segments"][1]["delivery"]["direction"], "")

    def test_direction_editor_escape_cancels_without_creating_an_edit(self):
        app, _, _ = self.app()
        self.open_second_passage(app)
        app.handle_key("F3")
        app.handle_key("x")

        app.handle_key("ESC")

        self.assertEqual(app.pending, {})
        self.assertEqual(app.view, "info")

    def test_review_key_toggles_an_assigned_passage(self):
        app, _, _ = self.app()
        self.open_second_passage(app)

        app.handle_key("r")

        self.assertTrue(app.pending["s2"]["attribution"]["review_required"])

    def test_review_guard_surfaces_unresolved_error_without_an_edit(self):
        app, _, _ = self.app(unresolved=True)
        self.open_second_passage(app)

        app.handle_key("r")

        self.assertEqual(app.pending, {})
        self.assertIn("Unresolved attribution", app.notice)
        self.assertEqual(app.view, "info")

    def test_save_publishes_revised_document_and_clears_pending_edits(self):
        app, revision, related_calls = self.app()
        original_hash = hashlib.sha256((app.directory / "manifest.json").read_bytes()).hexdigest()
        self.open_second_passage(app)
        app.handle_key("r")

        app.handle_key("S")

        self.assertEqual(related_calls, [[app.directory]])
        self.assertEqual(len(revision.calls), 1)
        call = revision.calls[0]
        self.assertEqual(call["directories"], [app.directory])
        self.assertEqual(call["expected_manifest_hashes"], {str(app.directory): original_hash})
        self.assertEqual(call["scope"], "attribution")
        self.assertTrue(call["revisions"][app.directory]["segments"][1]["attribution"]["review_required"])
        self.assertEqual(app.pending, {})
        self.assertTrue(app.document["segments"][1]["attribution"]["review_required"])
        self.assertEqual(app.notice, "Saved revision")
        self.assertIn("Updated fixture revision", app.progress)

    def test_second_save_uses_refreshed_manifest_hash(self):
        app, revision, _ = self.app(rewrite_manifest=True)
        self.open_second_passage(app)
        app.handle_key("r")
        app.handle_key("S")
        refreshed = hashlib.sha256((app.directory / "manifest.json").read_bytes()).hexdigest()
        app.handle_key("r")

        app.handle_key("S")

        self.assertEqual(len(revision.calls), 2)
        first = revision.calls[0]["expected_manifest_hashes"][str(app.directory)]
        second = revision.calls[1]["expected_manifest_hashes"][str(app.directory)]
        self.assertNotEqual(first, second)
        self.assertEqual(second, refreshed)

    def test_save_error_keeps_pending_edits_and_shows_the_message(self):
        app, revision, _ = self.app(error=ValueError("revision rejected"))
        self.open_second_passage(app)
        app.handle_key("r")

        app.handle_key("S")

        self.assertIn("s2", app.pending)
        self.assertFalse(app.document["segments"][1]["attribution"]["review_required"])
        self.assertIn("Save failed: revision rejected", app.notice)
        self.assertEqual(len(revision.calls), 1)

    def test_quit_with_unsaved_changes_warns_before_discarding(self):
        app, _, _ = self.app()
        self.open_second_passage(app)
        app.handle_key("r")

        app.handle_key("q")
        screen = FakeScreen()
        app.draw(screen)

        self.assertEqual(app.view, "confirm_quit")
        self.assertTrue(app.running)
        self.assertIn("Unsaved changes", screen.text)
        app.handle_key("n")
        self.assertFalse(app.running)

    def test_injected_screen_runs_the_real_loader_without_a_tty(self):
        directory, _ = publish_book(self.root)
        screen = FakeScreen(["q"])

        app = run_review(directory, screen=screen)

        self.assertFalse(app.running)
        self.assertIn("Chapter One", screen.text)

    def test_cli_dispatches_review_ui_with_the_resolved_directory(self):
        with patch("epub_tts_narrator.review_tui.run_review") as review:
            status = main(["review-ui", str(self.root / "book"), "--config", str(self.root / "config.toml")])

        self.assertEqual(status, 0)
        review.assert_called_once_with((self.root / "book").resolve())

    def test_non_tty_cli_returns_the_interactive_terminal_error_before_loading(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch.dict(os.environ, {"TERM": "xterm-256color"}),
              patch("sys.stdin") as stdin,
              contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
            stdin.isatty.return_value = False
            status = main(["review-ui", str(self.root / "missing-book")])

        self.assertEqual(status, 1)
        self.assertIn("Error: review-ui requires an interactive terminal", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
