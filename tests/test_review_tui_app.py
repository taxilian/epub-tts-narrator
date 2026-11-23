"""The full-screen review (Textual): the same ReviewApp decisions, driven headlessly through real keys."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.review_tui import ReviewApp
from tests.test_review_tui import (
    RevisionStub, TakeSelectionStub, add_failure_status, add_render_work, publish_book,
)


class PlayerStub:
    """``AudioPlayer``'s start/poll/stop; plays for ``polls`` polls."""

    def __init__(self, polls=2):
        self.started, self.stopped, self.polls, self._left = [], 0, polls, 0

    def start(self, path):
        self.started.append(str(path))
        self._left = self.polls

    def poll(self):
        self._left -= 1
        return self._left > 0

    def stop(self):
        self.stopped += 1
        self._left = 0


class ReviewTuiAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def make(self, *, rendered=False, two_chapters=False, empty_chapters=()):
        from epub_tts_narrator.review_tui_app import ReviewTuiApp
        directory, document = publish_book(self.root, two_chapters=two_chapters, empty_chapters=empty_chapters)
        self.revision = RevisionStub(directory)
        self.player = PlayerStub()
        options = {}
        if rendered:
            self.fixture = add_render_work(directory)
            self.takes = TakeSelectionStub(self.fixture)
            options = {"list_takes": self.takes.list, "select_take": self.takes.select,
                       "render_config": {"render": {"max_chunk_chars": 40}}}
        self.controller = ReviewApp(directory, document, revise_directories=self.revision,
                                    related_directories=lambda directories: [directory], player=self.player,
                                    **options)
        self.app = ReviewTuiApp(self.controller)
        return self.app

    async def wait_for(self, pilot, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Timed out waiting for the review screen")
            await pilot.pause(0.05)

    def logged(self):
        return "\n".join(text for _, _, text in self.app.messages)

    async def open_mara(self, pilot):
        """Chapter One → its passages → the second passage (Mara's line)."""
        await pilot.press("enter", "down")
        await pilot.pause()
        self.assertEqual(self.controller.info_segment_id, "s2")

    async def test_chapters_passages_and_metadata_show_with_original_numbers_and_colors(self):
        app = self.make(two_chapters=True, empty_chapters=((0, "Cover"),))
        add_failure_status(app.controller.directory, 3, segment_ids=["s4"])
        async with app.run_test(size=(150, 45)) as pilot:
            chapters = app.query_one("#chapters")
            # The empty cover is hidden, yet the chapters keep their book numbers.
            self.assertEqual(app._chapter_keys, ["1", "2"])
            self.assertEqual(str(chapters.get_row_at(1)[0]), "3")
            self.assertIn("FAILED", str(chapters.get_row_at(1)[3]))
            await self.open_mara(pilot)
            passages = app.query_one("#passages")
            speaker = passages.get_row_at(1)[1]
            narrator = passages.get_row_at(0)[1]
            self.assertEqual(speaker.plain, "Mara")
            self.assertNotEqual(speaker.spans[0].style, narrator.spans[0].style)
            detail = str(app.query_one("#detail").render())
            self.assertIn("Speaker(s): Mara", detail)
            self.assertIn("Role IDs: mara", detail)
            # F goes to the failed passage of the failed chapter.
            await pilot.press("escape", "down", "F")
            await pilot.pause()
            self.assertEqual(self.controller.info_segment_id, "s4")
            self.assertIn("FAILED", passages.get_row_at(0)[0].plain)

    async def test_speaker_direction_and_review_edits_stay_pending_until_saved(self):
        from epub_tts_narrator.tui_widgets import MultiChoiceScreen, TextEditScreen
        app = self.make()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_mara(pilot)
            await pilot.press("e")
            self.assertIsInstance(app.screen, MultiChoiceScreen)
            await pilot.press(*"narr", "enter")
            await pilot.pause()
            self.assertEqual(self.controller.pending["s2"]["role_id"], "narrator")
            self.assertEqual(self.controller.pending["s2"]["attribution"]["basis"], "user")

            await pilot.press("d")
            self.assertIsInstance(app.screen, TextEditScreen)
            await pilot.press(*"x" * 241, "enter")
            self.assertIsInstance(app.screen, TextEditScreen)  # Too long: the limit is shown and it stays open.
            self.assertIn("240", str(app.screen.query_one("#error").render()))
            await pilot.press("ctrl+a", "ctrl+k", *"Quietly", "enter")
            await pilot.pause()
            self.assertEqual(self.controller.pending["s2"]["delivery"]["direction"], "Quietly")

            await pilot.press("r")
            self.assertTrue(self.controller.pending["s2"]["attribution"]["review_required"])
            self.assertIn("1 unsaved", app.sub_title)

            await pilot.press("S")
            await self.wait_for(pilot, lambda: not self.controller.pending)
            call = self.revision.calls[0]
            self.assertEqual(call["scope"], "attribution")
            self.assertEqual(set(call["expected_manifest_hashes"]), {str(self.controller.directory)})
            self.assertEqual(call["revisions"][self.controller.directory]["segments"][1]["delivery"]["direction"],
                             "Quietly")
            self.assertIn("Saved revision", self.logged())

    async def test_two_marked_speakers_are_offered_as_one_simultaneous_line(self):
        app = self.make()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_mara(pilot)
            await pilot.press("e")
            screen = app.screen
            await pilot.press("down")  # Into the list; Mara is marked already.
            choices = screen.query_one("#choices")
            target = [value for value, _ in screen.options].index("narrator")
            current = int(choices.get_option_at_index(choices.highlighted).id)
            await pilot.press(*(["down"] * (target - current) if target > current else ["up"] * (current - target)))
            await pilot.press("space")
            self.assertEqual(screen.marked, ["mara", "narrator"])
            await pilot.press("enter")
            await pilot.pause()
        # The group reached the shared edit, which keeps narration out of simultaneous speech.
        self.assertEqual(self.controller.pending, {})
        self.assertIn("Simultaneous speech requires dialogue roles", self.logged())

    async def test_quitting_with_unsaved_edits_asks_and_can_discard(self):
        from epub_tts_narrator.tui_widgets import ChoiceScreen
        app = self.make()
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_mara(pilot)
            await pilot.press("r", "q")
            self.assertIsInstance(app.screen, ChoiceScreen)
            await pilot.press("escape")
            self.assertTrue(app.is_running)
            await pilot.press("q", "down", "enter")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(self.revision.calls, [])

    async def test_ignoring_a_chapter_saves_at_once_and_hidden_chapters_can_be_restored(self):
        app = self.make(two_chapters=True)
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("down", "I", "y")
            await pilot.pause()
            self.assertEqual(self.controller.ignored_chapters, frozenset({1}))
            self.assertEqual(app._chapter_keys, ["0"])
            sidecar = json.loads((self.controller.directory / "chapter-ignores.json").read_text())
            self.assertEqual(len(sidecar["ignored"]), 1)
            await pilot.press("H")
            await pilot.pause()
            self.assertEqual(app._chapter_keys, ["0", "1"])
            self.assertIn("IGNORED", str(app.query_one("#chapters").get_row_at(1)[3]))
            await pilot.press("down", "I", "y")
            await pilot.pause()
            self.assertEqual(self.controller.ignored_chapters, frozenset())

    async def test_play_uses_the_accepted_take_and_space_stops_it(self):
        app = self.make(rendered=True)
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_mara(pilot)
            await pilot.press("p")
            self.assertTrue(self.player.started[-1].endswith("attempt-002.wav"))
            await self.wait_for(pilot, lambda: app.playing is None)
            self.assertIn("Playback finished", self.logged())
            self.player.polls = 1000
            await pilot.press("p", "space")
            self.assertIsNone(app.playing)
            self.assertIn("Playback stopped", self.logged())

    async def test_greater_than_switches_playback_speed_and_restarts_the_take_playing(self):
        from epub_tts_narrator.renderer.playback_speed import PlaybackSpeed
        app = self.make(rendered=True)
        self.player.speed, restarts = PlaybackSpeed(), []
        self.player.restart = lambda: restarts.append(self.player.speed.label)
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_mara(pilot)
            self.assertIn("> speed (1x)", app.sub_title)
            self.player.polls = 1000
            await pilot.press("p", "greater_than_sign")
            self.assertEqual(restarts, ["2x"])  # The take playing starts again at 2x.
            self.assertIn("> speed (2x)", app.sub_title)
            self.assertIn("Playback speed is now 2x", self.logged())
            self.assertIsNotNone(app.playing)
            await pilot.press("question_mark")
            self.assertIn("Speed", " ".join(description for _, description in app.screen.rows))

    async def test_takes_open_the_shared_take_dialog_and_using_one_archives_the_checkpoint(self):
        from epub_tts_narrator.run_shell_screens import TakesScreen
        app = self.make(rendered=True)
        async with app.run_test(size=(150, 45)) as pilot:
            await self.open_mara(pilot)
            await pilot.press("t")
            self.assertIsInstance(app.screen, TakesScreen)
            table = app.screen.query_one("#takes-table")
            self.assertEqual(table.row_count, 2)
            await pilot.press("u")  # Take 1 (rejected) → confirm the use.
            await pilot.press("enter")
            await self.wait_for(pilot, lambda: self.takes.select_calls)
            await self.wait_for(pilot, lambda: "Accepted take 1" in self.logged())
        self.assertEqual(self.takes.select_calls[0][2:], (1, ""))
        self.assertFalse(self.fixture["checkpoint"].exists())
        self.assertIn("--chapters-only 1", self.logged())
        feedback = json.loads((self.fixture["work"] / "take-feedback.json").read_text())
        self.assertEqual([entry["verdict"] for entry in feedback["takes"].values()], ["good"])

    async def test_keys_dialog_lists_every_key(self):
        from epub_tts_narrator.tui_widgets import KeysScreen
        app = self.make()
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("question_mark")
            self.assertIsInstance(app.screen, KeysScreen)
            keys = " ".join(keys for keys, _ in app.screen.rows)
            for key in ("S", "e", "d", "r", "p", "t", "F", "I", "H"):
                self.assertIn(key, keys)


    async def test_quitting_during_a_save_waits_in_a_dialog_that_can_quit_now(self):
        import threading
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        root = self.root
        for key, forced in (("none", None), ("Q", 130)):
            self.root = root / key  # each pass publishes its own book
            self.root.mkdir()
            app = self.make()
            app.exit_process = []
            app.exit_process = app.exit_process.append
            release = threading.Event()
            self.addCleanup(release.set)
            async with app.run_test(size=(150, 45)) as pilot:
                app.run_worker(lambda: release.wait(10), thread=True, group="save")  # a save still writing
                await pilot.pause(0.2)
                await pilot.press("q")
                await self.wait_for(pilot, lambda: isinstance(app.screen, QuitWaitScreen))
                self.assertIn("Saving your edits", app.screen.describe())
                if key == "none":
                    release.set()
                else:
                    await pilot.press(key)
                await self.wait_for(pilot, lambda: not app.is_running)
            release.set()
            self.assertEqual(app.forced_exit, forced)


class ReviewUiCliTests(unittest.TestCase):
    def test_a_terminal_opens_the_full_screen_review_and_plain_keeps_curses(self):
        book = Path("/books/fixture")
        with (patch("epub_tts_narrator.cli.load_config", return_value={"casting": {"player": ["fixture-player"]}}),
              patch("epub_tts_narrator.tui_support.wants_tui", return_value=True),
              patch("epub_tts_narrator.review_tui_app.run_review_tui") as full,
              patch("epub_tts_narrator.review_tui.run_review") as curses):
            self.assertEqual(main(["review-ui", str(book)]), 0)
        full.assert_called_once_with(book.resolve(), player_command=["fixture-player"])
        curses.assert_not_called()
        with (patch("epub_tts_narrator.cli.load_config", return_value={}),
              patch("epub_tts_narrator.review_tui_app.run_review_tui") as full,
              patch("epub_tts_narrator.review_tui.run_review") as curses):
            self.assertEqual(main(["review-ui", str(book), "--plain"]), 0)
        full.assert_not_called()
        curses.assert_called_once_with(book.resolve(), player_command=["/usr/bin/afplay"])


if __name__ == "__main__":
    unittest.main()
