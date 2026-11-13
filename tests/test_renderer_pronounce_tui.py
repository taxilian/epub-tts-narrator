"""The full-screen ``pronounce``: the same listening loop, answered by keys."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from epub_tts_narrator.name_pronunciations import effective_definitions
from epub_tts_narrator.pipeline import CachedRequests
import test_name_pronunciations as line_tool
from test_name_pronunciations import HAS_FFMPEG, FakeModel


@unittest.skipUnless(HAS_FFMPEG, "FFmpeg and FFprobe are required for listening previews")
class PronounceAppTests(unittest.IsolatedAsyncioTestCase):
    setUp = line_tool.InteractiveToolTests.setUp  # The line tool's fixtures: book, fake engine, saved voices.

    def command(self, model, **arguments):
        from epub_tts_narrator.renderer.name_pronunciation_tool import _run_tool
        args = SimpleNamespace(name="Mara", sound=None, phonemes=None, voice_id=None, tag_only=False,
                               review_tags=False, list=False)
        for key, value in arguments.items():
            setattr(args, key, value)
        self.player = Mock()
        self.player.poll.return_value = False
        self.states = []

        def command(input_fn, output, playback_fn, observer):
            def watch(state):
                self.states.append(state)
                observer(state)
            return _run_tool(self.bundle, self.config, args, name=args.name, sound=None, parts=None,
                             analysis_config=lambda: {"analysis": {"attribution_workers": 1}},
                             input_fn=input_fn, output=output, playback_fn=playback_fn, interactive=True,
                             engine_factory=self.factory, store=self.store, player=self.player,
                             requests=CachedRequests(model, self.directory / ".analysis-cache", repair_attempts=0),
                             observer=watch)
        return command

    async def drive(self, command, *steps):
        from epub_tts_narrator.renderer.pronounce_tui import PronounceApp
        app = PronounceApp(command, title="pronounce — book")
        async with app.run_test(size=(140, 40)) as pilot:
            for step in steps:
                deadline = time.monotonic() + 20
                while not app.waiting and app.is_running:
                    if time.monotonic() > deadline:
                        self.fail("The pronounce screen never asked")
                    await pilot.pause(0.05)
                if not app.is_running:
                    break
                await pilot.press(*step)
                await pilot.pause(0.1)
            deadline = time.monotonic() + 20
            while app.is_running and time.monotonic() < deadline:
                await pilot.pause(0.05)
        if app.raised is not None:
            raise app.raised
        return app

    async def test_describe_edit_and_accept_by_keys(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0", "respelling": "MAR-uh"},
                                   {"phonemes": "M AA1 R AH0", "respelling": "MAH-ruh"}])
        app = await self.drive(self.command(model),
                               ["i", *"more like MAH-ruh", "enter"],
                               ["e", "ctrl+a", "ctrl+k", *"M AA1 R AH0", "enter"],
                               ["a"])
        self.assertEqual(app.result, 0)
        defined = [state for state in self.states if state["kind"] == "define"]
        self.assertEqual([state["phonemes"] for state in defined[:3]], ["M AE1 R AH0", "M AA1 R AH0", "M AA1 R AH0"])
        self.assertTrue(defined[0]["spans"])  # the uses in the test sentence are marked
        self.assertEqual(effective_definitions(self.directory)["mara"]["phonemes"], "M AA1 R AH0")
        self.assertTrue(self.player.start.called)

    async def test_choosing_parts_of_speech_and_quitting_keeps_the_draft(self):
        model = FakeModel(guesses=[{"phonemes": "M AE1 R AH0", "respelling": "MAR-uh"}])
        app = await self.drive(self.command(model), ["s", "down", "space", "enter"], ["q"])
        self.assertEqual(app.result, 0)
        defined = [state for state in self.states if state["kind"] == "define"]
        self.assertNotEqual(defined[0]["parts"], defined[-1]["parts"])
        self.assertNotIn("mara", effective_definitions(self.directory))


if __name__ == "__main__":
    unittest.main()


import test_renderer_pronunciation_review_queue as queue_tests


@unittest.skipUnless(queue_tests.HAS_FFMPEG, "ffmpeg/ffprobe required")
class PronunciationQueueAppTests(unittest.IsolatedAsyncioTestCase):
    setUp = queue_tests.PronunciationReviewQueueTests.setUp  # Real registry, preview cache and fake engine.
    row = queue_tests.PronunciationReviewQueueTests.row
    rows = queue_tests.PronunciationReviewQueueTests.rows

    def command(self):
        from epub_tts_narrator.renderer.pronunciation_review_queue import run_pronunciation_queue
        player = Mock()
        player.poll.return_value = False

        def command(input_fn, output, playback_fn, observer, wait_fn, redraw):
            return run_pronunciation_queue(self.session, player, playback_fn, input_fn=input_fn, output=output,
                                           interactive=True, redraw=redraw, wait_fn=wait_fn, observer=observer,
                                           gap_seconds=0)
        return command

    async def drive(self, *steps):
        from epub_tts_narrator.renderer.pronounce_tui import PronunciationQueueApp
        app = PronunciationQueueApp(self.command(), title="pronunciations — book")
        async with app.run_test(size=(150, 42)) as pilot:
            for step in steps:
                deadline = time.monotonic() + 60
                while not app.waiting and app.is_running:
                    if time.monotonic() > deadline:
                        self.fail("The queue screen never asked")
                    await pilot.pause(0.05)
                if callable(step):
                    step(app)
                else:
                    await pilot.press(*step)
                await pilot.pause(0.1)
            deadline = time.monotonic() + 60
            while app.is_running and time.monotonic() < deadline:
                await pilot.pause(0.05)
        if app.raised is not None:
            raise app.raised
        return app

    def select(self, word):
        def choose(app):
            from textual.widgets import OptionList
            number = next(row["number"] for row in app.state["rows"] if row["word"] == word)
            app.query_one("#queue-entries", OptionList).highlighted = number - 1
        return choose

    async def test_edit_phonemes_then_confirm_by_keys(self):
        app = await self.drive(self.select("Qarm"), ["p", *"K W AO1 R M", "enter"], ["c"], ["q"])
        self.assertEqual(app.result, 0)
        self.assertEqual(self.row("Qarm")["status"], "confirmed")
        self.assertEqual(self.row("Qarm")["pronunciation"], queue_tests.CMU_QARM)

    async def test_listening_through_the_queue_accepts_every_word_heard_twice(self):
        app = await self.drive(["s"], ["q"])
        self.assertEqual(app.result, 0)
        self.assertTrue(all(row["status"] != "pending" for row in self.rows()))
        self.assertEqual(app.state["queued"], 0)
