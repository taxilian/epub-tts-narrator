"""Quitting a full-screen UI while work runs: a dialog says what runs, and Quit now ends at once."""

import asyncio
import threading
import time
import unittest

from cast_tui_fixtures import FakePlayer, FakeSession


async def wait_for(test, pilot, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            test.fail("Timed out waiting for the screen")
        await pilot.pause(0.05)


class QuitWaitScreenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    def host(self, describe, **options):
        from textual.app import App
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        results = []

        class Host(App):
            def on_mount(self):
                self.push_screen(QuitWaitScreen(describe, **options), results.append)
        return Host(), results

    async def test_it_shows_what_runs_and_closes_by_itself_when_the_work_ends(self):
        from textual.widgets import Static
        work = ["Generate a voice for Mara — 4 s"]
        app, results = self.host(lambda: list(work))
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.2)
            self.assertIn("Generate a voice for Mara", str(app.screen.query_one("#quit-wait-work", Static).render()))
            work.clear()
            await wait_for(self, pilot, lambda: results)
        self.assertEqual(results, ["done"])

    async def test_shift_q_quits_now_and_escape_keeps_working(self):
        for key, expected in (("Q", "now"), ("escape", "cancel")):
            app, results = self.host(lambda: ["A model request"])
            async with app.run_test(size=(120, 30)) as pilot:
                await pilot.pause(0.2)
                await pilot.press(key)
                await wait_for(self, pilot, lambda: results)
            self.assertEqual(results, [expected])

    async def test_without_cancelling_escape_does_nothing(self):
        app, results = self.host(lambda: ["Stopping"], can_cancel=False)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.2)
            await pilot.press("escape")
            await pilot.pause(0.2)
            self.assertEqual(results, [])
            await pilot.click("#now")
            await wait_for(self, pilot, lambda: results)
        self.assertEqual(results, ["now"])


class QuitNowTests(unittest.TestCase):
    class FakeApp:
        def __init__(self):
            self.exited, self.exits, self.forced_exit = 0, [], None
            self.exit_process = self.exits.append

        def call_from_thread(self, callback):
            raise RuntimeError("on the screen's own thread")

        def exit(self):
            self.exited += 1

    def test_cleanup_runs_even_when_a_step_fails_then_the_screen_closes(self):
        from epub_tts_narrator.tui_support import end_forced_quit, quit_now
        app, steps = self.FakeApp(), []

        def broken():
            raise OSError("the player is gone")
        quit_now(app, cleanup=[lambda: steps.append("audio"), broken, lambda: steps.append("jobs")], grace=60)
        self.assertEqual((steps, app.forced_exit, app.exited, app.exits), (["audio", "jobs"], 130, 1, []))
        end_forced_quit(app)
        self.assertEqual(app.exits, [130])

    def test_a_screen_that_never_closes_ends_the_process_after_the_grace(self):
        from epub_tts_narrator.tui_support import quit_now
        app = self.FakeApp()
        quit_now(app, grace=0.05)
        time.sleep(0.3)
        self.assertEqual(app.exits, [130])

    def test_an_ordinary_quit_ends_nothing(self):
        from epub_tts_narrator.tui_support import end_forced_quit
        app = self.FakeApp()
        end_forced_quit(app)
        self.assertEqual(app.exits, [])


class CastQuitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        self.player = FakePlayer()
        self.controller = CastController(FakeSession(), self.player, describe_error=str)
        self.addCleanup(self.controller.close)
        self.app = CastApp(self.controller, title="Fixture Book", startup=False)
        self.exits = []
        self.app.exit_process = self.exits.append
        self.release = threading.Event()
        self.addCleanup(self.release.set)

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    def start_work(self):
        started = threading.Event()

        def generate():
            started.set()
            self.release.wait(10)
        self.controller.jobs.submit("Generate a voice for Mara", generate, scope={"voice": "v-mara"})
        self.queued = self.controller.jobs.submit("Generate a voice for Ava", lambda: None, scope={"voice": "v-ava"})
        started.wait(5)

    async def test_quitting_while_a_job_runs_waits_for_it_in_a_dialog(self):
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        async with self.app.run_test(size=(160, 45)) as pilot:
            self.start_work()
            await pilot.press("q")
            await pilot.pause(0.2)
            self.assertIsInstance(self.app.screen, QuitWaitScreen)
            self.assertEqual(self.queued.state, "cancelled")
            self.release.set()
            await wait_for(self, pilot, lambda: not self.app.is_running)
        self.assertIsNone(self.app.forced_exit)

    async def test_quit_now_closes_at_once_without_waiting_for_the_job(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            self.start_work()
            await pilot.press("ctrl+q")
            await pilot.pause(0.2)
            await pilot.press("Q")
            await wait_for(self, pilot, lambda: not self.app.is_running)
        self.assertEqual(self.app.forced_exit, 130)
        self.assertEqual(self.controller.jobs.running().label, "Generate a voice for Mara")  # never waited for

    async def test_dont_quit_keeps_the_screen_and_the_running_job(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            self.start_work()
            await pilot.press("q")
            await pilot.pause(0.2)
            await pilot.press("escape")
            await pilot.pause(0.2)
            self.assertTrue(self.app.is_running)
            self.assertIsNotNone(self.controller.jobs.running())
            self.release.set()
            self.controller.jobs.wait_idle(5)
            await pilot.pause(0.2)
            self.assertTrue(self.app.is_running)


class RunShellQuitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    def make_app(self, command):
        from epub_tts_narrator.renderer.progress import shell_board
        from epub_tts_narrator.run_events import EventBus
        from epub_tts_narrator.run_shell import TaskApp
        from epub_tts_narrator.run_shell_model import RunModel
        bus = EventBus()
        self.model = RunModel("analyze — fixture")
        bus.subscribe(self.model.on_event)
        app = TaskApp(self.model, bus, command, title="analyze — fixture")
        board = shell_board(bus)
        board.__enter__()
        self.addCleanup(board.__exit__, None, None, None)
        self.exits = []
        app.exit_process = self.exits.append
        return app

    async def test_a_command_that_does_not_stop_can_be_left_at_once(self):
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def command():
            started.set()
            release.wait(10)  # a blocking call the stop request cannot interrupt
            return 0
        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await wait_for(self, pilot, started.is_set)
            await pilot.press("q", "y")
            await wait_for(self, pilot, lambda: isinstance(app.screen, QuitWaitScreen))
            self.assertTrue(self.model.running)
            await pilot.press("Q")
            await wait_for(self, pilot, lambda: not app.is_running)
        self.assertEqual(app.forced_exit, 130)

    async def test_the_dialog_closes_the_shell_once_the_command_has_stopped(self):
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        started = threading.Event()

        def command():
            started.set()
            while True:
                time.sleep(0.01)  # stops at the next instruction, as Ctrl-C would
        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await wait_for(self, pilot, started.is_set)
            await pilot.press("q", "y")
            await wait_for(self, pilot, lambda: not app.is_running)
        self.assertIsNone(app.forced_exit)
        self.assertIsInstance(app.raised, KeyboardInterrupt)


class PronounceQuitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    def setUp(self):
        self.release, self.answers = threading.Event(), []
        self.addCleanup(self.release.set)

    def command(self, input_fn, output, playback_fn, observer, *extra):
        output("Making the audition for Mara")
        self.release.wait(10)  # synthesizing: the screen is not waiting for a choice
        self.answers.append(input_fn("Choice: "))
        return 0

    async def test_q_while_working_waits_for_the_step_then_quits_with_q(self):
        from epub_tts_narrator.renderer.pronounce_tui import PronounceApp, PronunciationQueueApp
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        for app_class in (PronounceApp, PronunciationQueueApp):
            self.release.clear()
            self.answers.clear()
            app = app_class(self.command, title="pronounce — book")
            async with app.run_test(size=(140, 40)) as pilot:
                await pilot.pause(0.3)
                await pilot.press("q")
                await wait_for(self, pilot, lambda: isinstance(app.screen, QuitWaitScreen))
                self.assertIn("Making the audition", str(app.screen.describe()))
                self.release.set()
                await wait_for(self, pilot, lambda: not app.is_running)
            self.assertEqual((self.answers, app.result), (["q"], 0), app_class.__name__)

    async def test_quit_now_does_not_wait_for_the_step(self):
        from epub_tts_narrator.renderer.pronounce_tui import PronounceApp
        from epub_tts_narrator.tui_widgets import QuitWaitScreen
        app = PronounceApp(self.command, title="pronounce — book")
        app.exit_process = []
        app.exit_process = app.exit_process.append
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause(0.3)
            await pilot.press("ctrl+q")
            await wait_for(self, pilot, lambda: isinstance(app.screen, QuitWaitScreen))
            await pilot.press("Q")
            await wait_for(self, pilot, lambda: not app.is_running)
        self.assertEqual((app.forced_exit, self.answers), (130, []))


if __name__ == "__main__":
    unittest.main()
