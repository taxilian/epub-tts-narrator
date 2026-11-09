"""Drive the review state machine through a fake screen, never a fake curses module."""
# noqa: SIZE_OK — one FakeScreen contract covers the complete ReviewApp state machine.

import contextlib
from collections.abc import Callable
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.review_tui import ReviewApp, run_review
from epub_tts_narrator.series import link_analysis
from tests.test_contracts import valid_document


Document = dict[str, Any]


def review_document(*, two_chapters=False, unresolved=False, long_passage=False) -> Document:
    document: Document = valid_document()
    if long_passage:
        text = " ".join(f"word{index}" for index in range(30)) + "."
        document["source"]["blocks"][1]["text"] = text
        document["segments"][2].update(text=text, end=len(text))
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


def publish_book(root, *, two_chapters=False, unresolved=False, long_passage=False):
    """Publish a self-contained temporary book through the real exporter/loader contract."""
    directory = Path(root) / "review-book"
    document = review_document(two_chapters=two_chapters, unresolved=unresolved, long_passage=long_passage)
    registry = fresh_registry(directory)
    linked = link_analysis(document, registry)
    publish_directory(linked, registry, directory, export_bundle)
    return directory, linked


def write_wave(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        output.writeframes(b"\x01\x00" * 80)


def add_render_work(directory, *, drift=False) -> dict[str, Any]:
    """Build chapter-local rendered state from the real one-chapter source plan."""
    from epub_tts_narrator.renderer.book_context import resolve_book_config
    from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
    from epub_tts_narrator.renderer.config import DEFAULTS

    bundle = load_bundle(directory)
    settings: dict[str, Any] = deepcopy(DEFAULTS)
    settings["render"].update(chapter_numbers=[1], max_chunk_chars=40)
    settings = resolve_book_config(bundle, settings)
    plan: dict[str, Any] = make_plan(bundle, settings)
    chunks: list[dict[str, Any]] = plan["chapters"][0]["chunks"]
    work = Path(settings["render"]["work_dir"])
    acceptance, clips = [], {}
    for chunk in chunks:
        if chunk["silent_control"]:
            continue
        segment_id = chunk["segment"]["id"]
        request_hash = hashlib.sha256(("rendered:" + chunk["chunk_id"]).encode()).hexdigest()
        clip = work / "clips" / request_hash
        audio_paths = []
        for attempt in (1, 2):
            audio = clip / f"passage-{segment_id}-attempt-{attempt:03d}.wav"
            write_wave(audio)
            audio_paths.append(audio)
            record = {"attempt": attempt, "report": {"passed": attempt == 2,
                      "reasons": [] if attempt == 2 else ["text_mismatch"]}}
            (clip / f"attempt-{attempt:03d}.json").write_text(json.dumps(record), encoding="utf-8")
        accepted = {"request_hash": request_hash, "request": {"kind": "speech", "chunk": chunk},
                    "audio_file": audio_paths[-1].name,
                    "audio_sha256": hashlib.sha256(audio_paths[-1].read_bytes()).hexdigest(),
                    "validation_key": "2" * 64, "report": {"passed": True}}
        (clip / "accepted.json").write_text(json.dumps(accepted), encoding="utf-8")
        checkpoint_chunk_id = ("drifted-" if drift else "") + chunk["chunk_id"]
        acceptance.append({"chunk_id": checkpoint_chunk_id, "request_hash": request_hash,
                           "audio_sha256": accepted["audio_sha256"], "validation_key": "2" * 64,
                           "report": {"passed": True}})
        clips[request_hash] = {"directory": clip, "audio_paths": audio_paths,
                               "chunk_id": checkpoint_chunk_id, "segment_id": segment_id}
    checkpoint = work / "completed-chapters" / "chapter-0001.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(json.dumps({"format": "audiobook-chapter-checkpoint", "version": 1,
                                      "chapter_number": 1, "acceptance": acceptance}), encoding="utf-8")
    unrelated = work / "clips" / ("f" * 64)
    unrelated.mkdir(parents=True)
    (unrelated / "accepted.json").write_text("{not json", encoding="utf-8")
    return {"work": work, "checkpoint": checkpoint, "acceptance": acceptance, "clips": clips}


class TakeSelectionStub:
    def __init__(self, fixture: dict[str, Any]):
        self.fixture = fixture
        self.rows_by_hash = {}
        self.list_calls = []
        self.select_calls = []
        self.error: ValueError | None = None
        for request_hash, clip in fixture["clips"].items():
            self.rows_by_hash[request_hash] = [
                {"attempt": 1, "audio_path": str(clip["audio_paths"][0]), "passed": False,
                 "reasons": ["text_mismatch"], "text_confirmation_method": None,
                 "duration_seconds": 1.25, "transcript": "DO NOT DISPLAY", "echo": False,
                 "eligible": True, "ineligible_reason": None, "accepted": False},
                {"attempt": 2, "audio_path": str(clip["audio_paths"][1]), "passed": True,
                 "reasons": [], "text_confirmation_method": "verbatim", "duration_seconds": 2.5,
                 "transcript": "DO NOT DISPLAY", "echo": False, "eligible": True,
                 "ineligible_reason": None, "accepted": True},
            ]

    def list(self, clips_directory, request_hash):
        self.list_calls.append((Path(clips_directory), request_hash))
        return deepcopy(self.rows_by_hash[request_hash])

    def select(self, clips_directory, request_hash, attempt, *, note=""):
        self.select_calls.append((Path(clips_directory), request_hash, attempt, note))
        if self.error is not None:
            raise self.error
        for row in self.rows_by_hash[request_hash]:
            row["accepted"] = row["attempt"] == attempt
        return {"request_hash": request_hash, "attempt": attempt}


class PlaybackStub:
    def __init__(self, outcome: bool | Exception = True):
        self.outcome = outcome
        self.calls = []
        self.before: Callable[[], None] | None = None

    def __call__(self, player, path):
        if self.before is not None:
            self.before()
        self.calls.append((player, path))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


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

    def rendered_app(self, *, root=None, long_passage=False, drift=False, playback=None):
        root = Path(root) if root is not None else self.root
        root.mkdir(parents=True, exist_ok=True)
        directory, document = publish_book(root, long_passage=long_passage)
        fixture = add_render_work(directory, drift=drift)
        revision = RevisionStub(directory)
        takes = TakeSelectionStub(fixture)
        playback = playback if playback is not None else PlaybackStub()
        app = ReviewApp(directory, document, revise_directories=revision,
                        related_directories=lambda directories: [directory], player=object(),
                        play_sample=playback, list_takes=takes.list, select_take=takes.select,
                        render_config={"render": {"max_chunk_chars": 40}})
        return app, fixture, takes, playback

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

    def test_takes_action_lists_every_attempt_for_the_passage_clip(self):
        app, fixture, takes, _ = self.rendered_app()
        self.open_second_passage(app)

        app.handle_key("t")
        screen = FakeScreen()
        app.draw(screen)

        clip = next(clip for clip in fixture["clips"].values() if clip["segment_id"] == "s2")
        request_hash = clip["directory"].name
        self.assertEqual(app.view, "takes")
        self.assertEqual(takes.list_calls, [(fixture["work"] / "clips", request_hash)])
        self.assertEqual([row["attempt"] for row in app.takes_rows], [1, 2])
        self.assertIn("Take 1", screen.text)
        self.assertIn("* Take 2", screen.text)
        self.assertIn("text_mismatch", screen.text)
        self.assertNotIn("DO NOT DISPLAY", screen.text)

    def test_takes_playback_draws_status_first_and_maps_completion_interruption_and_failure(self):
        cases = ((True, "Playback finished"), (False, "Playback interrupted"),
                 (ValueError("player failed\n\x1b[31mraw"), "Playback failed:"))
        for index, (outcome, notice) in enumerate(cases):
            with self.subTest(outcome=outcome):
                playback = PlaybackStub(outcome)
                app, _, _, _ = self.rendered_app(root=self.root / str(index), playback=playback)
                self.open_second_passage(app)
                app.handle_key("t")
                screen = FakeScreen()
                playback.before = lambda: self.assertIn("Playing take 1 — any key stops", screen.text)

                app.handle_key("ENTER", screen=screen)

                self.assertEqual(len(playback.calls), 1)
                self.assertTrue(playback.calls[0][1].endswith("attempt-001.wav"))
                self.assertIn(notice, app.notice)
                self.assertNotIn("\x1b", app.notice)
                self.assertNotIn("\n", app.notice)
                self.assertTrue(app.running)

    def test_info_play_key_plays_the_accepted_take(self):
        app, _, _, playback = self.rendered_app()
        self.open_second_passage(app)

        app.handle_key("p", screen=FakeScreen())

        self.assertEqual(len(playback.calls), 1)
        self.assertTrue(playback.calls[0][1].endswith("attempt-002.wav"))
        self.assertEqual(app.view, "info")
        self.assertEqual(app.notice, "Playback finished")

    def test_multi_chunk_passage_can_change_the_selected_chunk(self):
        app, _, takes, _ = self.rendered_app(long_passage=True)
        app.handle_key("ENTER")
        app.handle_key("DOWN")
        app.handle_key("DOWN")
        app.handle_key("ENTER")
        app.handle_key("t")
        first_hash = takes.list_calls[-1][1]

        app.handle_key("RIGHT")
        screen = FakeScreen()
        app.draw(screen)

        self.assertGreater(len(app.takes_clips), 1)
        self.assertNotEqual(takes.list_calls[-1][1], first_hash)
        self.assertIn(f"Chunk 2/{len(app.takes_clips)}", screen.text)

    def test_accepting_a_failed_eligible_take_confirms_archives_and_refreshes(self):
        app, fixture, takes, _ = self.rendered_app()
        self.open_second_passage(app)
        app.handle_key("t")

        app.handle_key("a")
        confirmation = FakeScreen()
        app.draw(confirmation)
        app.handle_key("y")

        self.assertIn("It did not pass ordinary verification.", confirmation.text)
        self.assertEqual(len(takes.select_calls), 1)
        self.assertEqual(takes.select_calls[0][2:], (1, ""))
        self.assertFalse(fixture["checkpoint"].exists())
        archived = list(fixture["checkpoint"].parent.glob("chapter-0001.json.superseded-*"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(len(archived[0].name.rsplit("-", 1)[1]), 12)
        self.assertTrue(app.takes_rows[0]["accepted"])
        self.assertFalse(app.takes_rows[1]["accepted"])
        self.assertIn("--chapters-only 1", app.notice)
        self.assertIn("audiobook-render assemble", app.notice)

    def test_take_selection_refusal_changes_no_acceptance_or_checkpoint(self):
        app, fixture, takes, _ = self.rendered_app()
        takes.error = ValueError("selection refused")
        self.open_second_passage(app)
        app.handle_key("t")
        app.handle_key("a")

        app.handle_key("y")

        self.assertEqual(app.notice, "selection refused")
        self.assertTrue(fixture["checkpoint"].exists())
        self.assertEqual(list(fixture["checkpoint"].parent.glob("*.superseded-*")), [])
        self.assertFalse(app.takes_rows[0]["accepted"])
        self.assertTrue(app.takes_rows[1]["accepted"])

    def test_ineligible_and_echo_takes_refuse_before_calling_selection(self):
        cases = (("echo", {"echo": True, "eligible": False, "ineligible_reason": "history_echo"}),
                 ("ineligible", {"echo": False, "eligible": False, "ineligible_reason": "corrupt_audio"}))
        for name, changes in cases:
            with self.subTest(name=name):
                app, fixture, takes, _ = self.rendered_app(root=self.root / name)
                request_hash = next(key for key, clip in fixture["clips"].items() if clip["segment_id"] == "s2")
                takes.rows_by_hash[request_hash][0].update(changes)
                self.open_second_passage(app)
                app.handle_key("t")

                app.handle_key("a")

                self.assertEqual(app.view, "takes")
                self.assertEqual(takes.select_calls, [])
                self.assertIn(changes["ineligible_reason"], app.notice)
                self.assertTrue(fixture["checkpoint"].exists())

    def test_takes_view_reports_a_book_without_render_state(self):
        app, _, _ = self.app()
        self.open_second_passage(app)

        app.handle_key("t")
        screen = FakeScreen()
        app.draw(screen)

        self.assertEqual(app.view, "takes")
        self.assertIn("Not rendered: this book has no render-work directory.", app.notice)
        self.assertIn("No rendered clips for this passage.", screen.text)

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
        config_path = self.root / "config.toml"
        with (patch("epub_tts_narrator.review_tui.run_review") as review,
              patch("epub_tts_narrator.cli.load_config", return_value={"casting": {"player": ["fixture-player"]}}) as config):
            status = main(["review-ui", str(self.root / "book"), "--config", str(config_path)])

        self.assertEqual(status, 0)
        config.assert_called_once_with(config_path)
        review.assert_called_once_with((self.root / "book").resolve(), player_command=["fixture-player"])

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
