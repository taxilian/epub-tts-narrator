"""The run shell: events from render boards, the run model, and the headless Textual screen."""

import asyncio
import io
import threading
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import progress
from epub_tts_narrator.renderer.progress import (
    BusBoard, ChapterProgress, PrefixedBoardNotify, live_board, report_book_outcome, report_chapter_failure,
    shell_board,
)
from epub_tts_narrator.run_events import EventBus, classify, parse_scope
from epub_tts_narrator.run_shell_model import RunModel, next_severity


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def chunks(count):
    return [{"spoken_text": "word " * 10} for _ in range(count)]


class PromptChoiceTests(unittest.TestCase):
    def test_lettered_choices_are_read_from_the_question(self):
        from epub_tts_narrator.tui_widgets import prompt_choices
        self.assertEqual(prompt_choices("[A] apply the new directions / [L] leave the book unchanged: "),
                         [("A", "apply the new directions"), ("L", "leave the book unchanged")])
        self.assertEqual([key for key, _ in prompt_choices(
            "[A] accept all / [E] edit-review individual corrections / [L] leave unchanged (quit): ")], ["A", "E", "L"])
        self.assertEqual(prompt_choices("Speaker number, U for unresolved, Enter to keep: "), [])
        self.assertEqual(prompt_choices("Press [Enter] to continue"), [])


class EventParsingTests(unittest.TestCase):
    def test_existing_prefixes_become_scope(self):
        scope, text = parse_scope("[book-2] Ch 3: === Failed chapter 3 / 9: verification; continuing ===")
        self.assertEqual(scope, {"book": "book-2", "chapter": 3})
        self.assertTrue(text.startswith("=== Failed"))
        scope, _ = parse_scope("  Chunk 12 (worker gpu-a): generating")
        self.assertEqual(scope, {"chunk": 12, "worker": "gpu-a"})

    def test_severity_comes_from_the_markers_lines_already_use(self):
        self.assertEqual(classify("=== Failed chapter 2 / 9: x ==="), "error")
        self.assertEqual(classify("Renderer stopped: disk full"), "error")
        self.assertEqual(classify("Could not prepare Ava: quota"), "warning")
        self.assertEqual(classify("Books: 2 succeeded | 0 failed"), "info")
        self.assertEqual(classify("Reusing completed chapter 1 / 9"), "info")

    def test_multi_line_messages_become_one_event_per_line(self):
        bus = EventBus()
        events = bus.log("Rendering incomplete: 2 chapters failed\nChapter 1 (RenderBlocked): x\n")
        self.assertEqual([event.text for event in events],
                         ["Rendering incomplete: 2 chapters failed", "Chapter 1 (RenderBlocked): x"])


class ShellBoardTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.events = []
        self.bus.subscribe(self.events.append)

    def kinds(self, kind):
        return [event for event in self.events if event.kind == kind]

    def test_print_renders_reach_the_shell_board_and_plain_output_is_untouched(self):
        with shell_board(self.bus):
            with live_board(print) as notify:
                self.assertIsInstance(notify.__self__, BusBoard)
                notify("Processing chapter 1 / 2: One (3 blocks)")
            custom = []
            with live_board(custom.append) as notify:
                self.assertEqual(notify, custom.append)
        self.assertEqual(self.kinds("log")[0].text, "Processing chapter 1 / 2: One (3 blocks)")
        stream = io.StringIO()
        with live_board(print, stream=stream) as notify:
            self.assertIs(notify, print)

    def test_chapter_progress_publishes_numbers_and_removes_its_row(self):
        clock = Clock()
        with shell_board(self.bus) as board:
            with ChapterProgress(2, 5, chunks(3), notify=board.log, clock=clock, heartbeat=False) as chapter:
                with chapter.task(0) as task:
                    task.phase("generating speech, attempt 1/3")
                    task.worker_activity("started", "gpu-a")
                    clock.now += 4
                    chapter.refresh()
                    snapshot = self.kinds("status")[-1].data["snapshot"]
        self.assertEqual((snapshot["chapter"], snapshot["total_chapters"], snapshot["blocks"]), (2, 5, 3))
        self.assertEqual(snapshot["active"][0]["worker"], "gpu-a")
        self.assertEqual(snapshot["active"][0]["block"], 1)
        self.assertTrue(self.kinds("status_removed"))
        self.assertIn("speech stopped: 1/3 blocks", self.kinds("log")[-1].text)
        self.assertEqual(self.kinds("log")[-1].scope["chapter"], 2)

    def test_queue_prefixes_carry_the_book_into_status_failures_and_outcomes(self):
        with shell_board(self.bus) as board:
            notify = PrefixedBoardNotify(board, "[book-1] ")
            with ChapterProgress(1, 2, chunks(1), notify=notify, heartbeat=False):
                pass
            report_chapter_failure(notify, {"number": 1, "error_type": "SpeechVerificationFailed", "error": "text"})
            report_book_outcome(notify, "/books/book-1", False)
            notify("done")
        self.assertEqual(self.kinds("status")[0].scope["book"], "book-1")
        failure = self.kinds("failure")[0]
        self.assertEqual((failure.scope, failure.data["record"]["error_type"]),
                         ({"chapter": 1, "book": "book-1"}, "SpeechVerificationFailed"))
        self.assertEqual(self.kinds("outcome")[0].scope, {"book": "book-1"})
        self.assertEqual(self.kinds("log")[-1].scope, {"book": "book-1"})

    def test_failures_and_status_are_ignored_by_terminal_and_plain_boards(self):
        report_chapter_failure(print, {"number": 1, "error": "x"})
        stream = io.StringIO()
        board = progress.TerminalBoard(stream)
        notify = PrefixedBoardNotify(board, "[a] ")
        self.assertFalse(notify.structured_status)
        report_chapter_failure(board.log, {"number": 1, "error": "x"})
        self.assertEqual(stream.getvalue(), "")


class ChapterRetriesTests(unittest.TestCase):
    def test_requests_queue_once_and_a_closed_lane_refuses_them(self):
        from epub_tts_narrator.renderer.render_chapter_flow import ChapterRetries
        retries = ChapterRetries()
        self.assertTrue(retries.request(3))
        self.assertTrue(retries.request(3))
        self.assertEqual(retries.take(), 3)
        self.assertFalse(retries.idle())
        retries.done()
        self.assertTrue(retries.idle())
        retries.close()
        self.assertFalse(retries.request(4))
        self.assertIsNone(retries.take())

    def test_a_hold_ends_on_a_request_a_release_or_a_stop(self):
        from epub_tts_narrator.renderer.render_chapter_flow import ChapterRetries
        retries = ChapterRetries()
        threading.Timer(0.05, retries.request, args=(1,)).start()
        self.assertTrue(retries.wait_for_request(poll=0.01))
        retries = ChapterRetries()
        threading.Timer(0.05, retries.release).start()
        self.assertFalse(retries.wait_for_request(poll=0.01))

        class Stop(Exception):
            pass

        def stop():
            raise Stop()
        with self.assertRaises(Stop):
            ChapterRetries().wait_for_request(stop_check=stop, poll=0.01)


class RunModelTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.bus = EventBus(clock=self.clock)
        self.model = RunModel("render", clock=self.clock)
        self.bus.subscribe(self.model.on_event)

    def test_rows_failures_and_outcomes_build_the_screen_state(self):
        board = BusBoard(self.bus)
        spoken = [{"spoken_text": "word " * 10, "segment": {"role_id": "r", "voice_id": "v-mara"}}] * 2
        with ChapterProgress(4, 9, spoken, notify=PrefixedBoardNotify(board, "[b1] "), clock=self.clock,
                             heartbeat=False) as chapter:
            with chapter.task(0) as task:
                task.worker_activity("started", "gpu-a")
                chapter.refresh()
                self.assertEqual(self.model.workers()["gpu-a"][0][:3], ("b1", 4, 1))
                self.assertEqual(len(self.model.visible_rows()), 1)
                self.model.filter.voice = "v-other"
                self.assertEqual(self.model.visible_rows(), [])
                self.model.filter.voice = "v-mara"
                self.assertEqual(len(self.model.visible_rows()), 1)
                self.model.filter.voice = None
        self.assertEqual(self.model.completed, {"b1": 1})
        self.assertEqual(self.model.choices("voice"), ["v-mara"])
        self.bus.log("[b2] Renderer stopped: provider quota")
        board.book_outcome("/x/b2", False)
        board.chapter_failure({"number": 3, "error_type": "RenderBlocked", "error": "bad clip"}, book="b1")
        kinds = [(item["kind"], item["scope"].get("book"), item["error"]) for item in self.model.current_failures()]
        self.assertEqual(kinds, [("book", "b2", "Renderer stopped: provider quota"), ("chapter", "b1", "bad clip")])
        self.assertEqual(self.model.choices("book"), ["b1", "b2"])
        self.assertIn("2 failure(s)", self.model.summary())

    def test_filters_narrow_logs_and_failures_but_keep_run_wide_lines(self):
        for line in ("[b1] Ch 1: generating", "[b2] Ch 1: generating", "Rendering with gpu-a (2 slot(s))",
                     "[b1] Ch 2: Could not verify", "[b1] Ch 2: === Failed chapter 2 ==="):
            self.bus.log(line)
        self.model.filter.book = "b1"
        self.assertEqual([event.text for _, event in self.model.visible_logs()],
                         ["generating", "Rendering with gpu-a (2 slot(s))", "Could not verify", "=== Failed chapter 2 ==="])
        self.model.filter.severity = "warning"
        self.assertEqual(len(self.model.visible_logs()), 2)
        self.model.filter.severity = next_severity(self.model.filter.severity)
        self.assertEqual([event.text for _, event in self.model.visible_logs()], ["=== Failed chapter 2 ==="])
        self.model.filter.severity, self.model.filter.chapter = "debug", 1
        self.assertEqual([event.text for _, event in self.model.visible_logs()], ["generating"])
        self.assertEqual([seq for seq, _ in self.model.visible_logs(after=1)], [])

    def test_a_run_holds_open_after_failures_or_interruption_and_resume_starts_a_new_attempt(self):
        self.model.finish(0)
        self.assertFalse(self.model.should_hold())
        self.bus.publish("failure", "x", scope={"chapter": 1}, data={"number": 1})
        self.model.finish(1)
        self.assertTrue(self.model.should_hold())
        self.model.restart()
        self.assertEqual((self.model.attempt, self.model.current_failures()), (2, []))
        self.assertEqual(len(self.model.failures), 1)
        self.model.finish(130)
        self.assertEqual(self.model.state, "interrupted")
        self.assertTrue(self.model.should_hold())


class RunShellScreenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    def make_app(self, command):
        from epub_tts_narrator.run_shell import RunApp
        self.bus = EventBus()
        self.model = RunModel("render — fixture")
        self.bus.subscribe(self.model.on_event)
        self.app = RunApp(self.model, self.bus, command, title="render — fixture")
        self.board = shell_board(self.bus)
        self.board.__enter__()
        self.addCleanup(self.board.__exit__, None, None, None)
        return self.app

    async def wait_for(self, pilot, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Timed out waiting for the run shell")
            await pilot.pause(0.05)

    async def test_a_failed_chapter_keeps_the_screen_open_with_its_detail_and_resume_reruns(self):
        attempts = []

        def command():
            attempts.append(1)
            with live_board(print) as notify:
                notify("Processing chapter 1 / 2: One (1 blocks)")
                if len(attempts) == 1:
                    report_chapter_failure(notify, {"number": 2, "error_type": "SpeechVerificationFailed",
                                                    "error": "take 3 skipped words", "chunk_id": "c-7"})
                    print("Renderer stopped: Rendering incomplete: 1 chapter failed")
                    return 2
            print("Completed audiobook: /books/fixture/audiobook")
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: not self.model.running)
            await pilot.pause()
            self.assertTrue(app.is_running)
            self.assertEqual(app.query_one("#tabs").active, "errors-tab")
            errors = app.query_one("#errors")
            self.assertEqual(errors.row_count, 1)
            errors.focus()
            await pilot.press("enter")
            from epub_tts_narrator.run_shell_screens import FailureScreen, failure_text
            self.assertIsInstance(app.screen, FailureScreen)
            self.assertIn('"chunk_id": "c-7"', failure_text(app.screen.item, app.screen.hint)[1])
            # After the run, the repair on offer is resuming the book; Escape just closes.
            self.assertEqual(app.screen.actions, ["close"])
            await pilot.press("escape")
            await pilot.press("R")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual((len(attempts), app.result, self.model.state), (2, 0, "finished"))
        self.assertIn("Completed audiobook: /books/fixture/audiobook",
                      [event.text for _, event in self.model.logs])

    async def test_ctrl_c_interrupts_the_command_and_the_screen_reports_it(self):
        started = threading.Event()

        def command():
            started.set()
            while True:
                time.sleep(0.01)

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, started.is_set)
            from epub_tts_narrator.tui_widgets import ConfirmScreen
            # Escape (however often) and Enter on the stop question never stop the run.
            await pilot.press("escape", "escape", "ctrl+c")
            self.assertIsInstance(app.screen, ConfirmScreen)
            await pilot.press("enter")
            await pilot.pause()
            self.assertTrue(self.model.running)
            await pilot.press("ctrl+c")
            self.assertIsInstance(app.screen, ConfirmScreen)
            await pilot.press("y")
            await self.wait_for(pilot, lambda: not self.model.running)
            self.assertEqual(self.model.state, "interrupted")
            self.assertTrue(app.is_running)
            await pilot.press("q")
            await pilot.pause()
        self.assertIsInstance(app.raised, KeyboardInterrupt)

    async def test_a_second_stop_closes_the_screen_before_the_process_exits(self):
        # Exiting at once would cut off Textual's terminal restore (kitty keys, alt screen, mouse).
        from epub_tts_narrator.run_shell import _render_interrupts
        started, release = threading.Event(), threading.Event()
        exits = []

        def command():
            started.set()
            release.wait(10)  # A blocked render worker that ignores the stop.
            return 0

        self.addCleanup(release.set)
        app = self.make_app(command)
        app.interrupt_scope = _render_interrupts(app, exit_process=exits.append, grace=0.3)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, started.is_set)
            await pilot.press("q", "y")
            await self.wait_for(pilot, lambda: self.model.stopping)
            self.assertTrue(app.is_running)
            await pilot.press("ctrl+c", "y")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual((app.forced_exit, exits), (130, []))  # run_in_shell exits once the screen is closed.
        await asyncio.sleep(0.5)
        self.assertEqual(exits, [130])  # The fallback when a screen never closes.

    async def test_printed_lines_keep_their_order_with_board_messages(self):
        def command():
            with live_board(print) as notify:
                print("Rendering with gpu-a (4 slot(s))")
                notify("Processing chapter 1 / 1: One (1 blocks)")
                print("Completed audiobook: /books/fixture/audiobook", file=__import__("sys").stderr)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual([event.text for _, event in self.model.logs],
                         ["Rendering with gpu-a (4 slot(s))", "Processing chapter 1 / 1: One (1 blocks)",
                          "Completed audiobook: /books/fixture/audiobook"])
        self.assertEqual(self.model.logs[-1][1].severity, "info")

    async def test_a_command_s_own_terminal_question_becomes_a_dialog(self):
        import sys
        from epub_tts_narrator.tui_widgets import PromptBridge, PromptScreen
        answers = []

        def command():
            answers.append(sys.stdout.isatty())
            print("3 attribution finding(s).")
            print("[A] accept all / [E] edit-review / [L] leave unchanged (quit): ", end="", flush=True)
            answers.append(sys.stdin.readline())
            return 0

        bridge = PromptBridge(None, on_answer=lambda answer: self.bus.log("> " + answer))
        app = self.make_app(command)
        app.interactive_input, app.bridge, bridge.app = True, bridge, app
        previous = sys.stdin
        sys.stdin = bridge.input
        self.addCleanup(setattr, sys, "stdin", previous)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: isinstance(app.screen, PromptScreen))
            self.assertTrue(app.screen.question.startswith("[A] accept all"))
            self.assertEqual([key for key, _ in app.screen.choices], ["A", "E", "L"])
            self.assertIn("3 attribution finding(s).", app.screen.text)
            await pilot.press("e")  # a choice's letter answers at once
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(answers, [True, "E\n"])  # the commands accept either case
        self.assertIn("> E", [event.text for _, event in self.model.logs])

    async def test_resume_can_run_a_different_command_and_non_resumable_runs_refuse(self):
        calls = []

        def command():
            calls.append("first")
            print("Fresh verification run: run-42")
            return 1

        app = self.make_app(command)
        app.resume_command = lambda model, books=None: (lambda: calls.append("resume") or 0)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: not self.model.running)
            app.resumable = False
            await pilot.press("R")
            await pilot.pause()
            self.assertEqual(calls, ["first"])
            app.resumable = True
            await pilot.press("R")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(calls, ["first", "resume"])

    def held_command(self, outcome):
        from epub_tts_narrator.renderer.progress import report_retry_hold
        from epub_tts_narrator.renderer.render_chapter_flow import ChapterRetries

        def command():
            retries = ChapterRetries()
            with live_board(print) as notify:
                report_chapter_failure(notify, {"number": 2, "error_type": "RenderBlocked", "error": "fixture"},
                                       retry=lambda: retries.request(1))
                report_retry_hold(notify, retries.release, 1)
                deadline = time.monotonic() + 10

                def give_up():  # a failing test must not leave this thread waiting forever
                    if time.monotonic() > deadline:
                        raise TimeoutError("no retry decision")
                outcome.append(retries.wait_for_request(stop_check=give_up, poll=0.05))
                report_retry_hold(notify, None, 0)
            return 0 if outcome[-1] else 2
        return command

    async def test_r_retries_a_held_chapter_and_f_finishes_the_wait(self):
        for key, expected, still_open in (("r", True, False), ("F", False, True)):
            with self.subTest(key=key):
                outcome = []
                app = self.make_app(self.held_command(outcome))
                async with app.run_test(size=(150, 40)) as pilot:
                    await self.wait_for(pilot, lambda: bool(self.model.holds))
                    self.assertIn("waiting for retries", self.model.summary())
                    await pilot.press("5")
                    app.query_one("#errors").focus()
                    await pilot.press(key)
                    if key == "F":
                        await pilot.press("y")  # finishing the wait is confirmed
                    await self.wait_for(pilot, lambda: not self.model.running)
                    await pilot.pause()
                    self.assertEqual(app.is_running, still_open)
                self.assertEqual(outcome, [expected])
                self.board.__exit__(None, None, None)

    async def test_a_free_question_takes_a_typed_answer_and_long_context_scrolls(self):
        import sys
        from epub_tts_narrator.tui_widgets import PromptBridge, PromptScreen
        answers = []

        def command():
            for number in range(200):
                print(f"q{number}: Mara -> 'urgent, low' (was 'quiet')")
            print("Speaker number, U for unresolved, Enter to keep: ", end="", flush=True)
            answers.append(sys.stdin.readline())
            return 0

        bridge = PromptBridge(None)
        app = self.make_app(command)
        app.interactive_input, app.bridge, bridge.app = True, bridge, app
        previous = sys.stdin
        sys.stdin = bridge.input
        self.addCleanup(setattr, sys, "stdin", previous)
        async with app.run_test(size=(120, 30)) as pilot:
            await self.wait_for(pilot, lambda: isinstance(app.screen, PromptScreen))
            await pilot.pause()
            self.assertEqual(app.screen.choices, [])
            self.assertEqual(app.screen.question, "Speaker number, U for unresolved, Enter to keep:")
            self.assertIn("q199:", app.screen.text)
            self.assertTrue(app.screen.query_one("#answer").region.height > 0)  # still on screen
            await pilot.press("1", "2", "enter")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(answers, ["12\n"])

    async def test_escape_in_a_question_ends_input_like_end_of_file(self):
        import sys
        from epub_tts_narrator.tui_widgets import PromptBridge, PromptScreen
        answers = []

        def command():
            print("[A] apply the new directions / [L] leave the book unchanged: ", end="", flush=True)
            answers.append(sys.stdin.readline())
            return 0

        bridge = PromptBridge(None, on_answer=lambda answer: self.bus.log("> " + answer))
        app = self.make_app(command)
        app.interactive_input, app.bridge, bridge.app = True, bridge, app
        previous = sys.stdin
        sys.stdin = bridge.input
        self.addCleanup(setattr, sys, "stdin", previous)
        async with app.run_test(size=(120, 30)) as pilot:
            await self.wait_for(pilot, lambda: isinstance(app.screen, PromptScreen))
            await pilot.press("enter")  # Enter on the focused first choice answers with it...
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(answers, ["A\n"])
        answers.clear()
        app = self.make_app(command)
        bridge = PromptBridge(None, on_answer=lambda answer: self.bus.log("> " + answer))
        app.interactive_input, app.bridge, bridge.app = True, bridge, app
        sys.stdin = bridge.input
        async with app.run_test(size=(120, 30)) as pilot:
            await self.wait_for(pilot, lambda: isinstance(app.screen, PromptScreen))
            await pilot.press("escape")  # ...and Escape ends input, which the commands read as "leave it"
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(answers, [""])
        self.assertIn("> (input ended)", [event.text for _, event in self.model.logs])

    async def test_commands_that_are_not_renders_get_the_task_layout(self):
        from epub_tts_narrator.run_shell import TaskApp
        from epub_tts_narrator.tui_widgets import ConfirmScreen
        release = threading.Event()

        def command():
            print("Narrator read: markup 3/12 complete")
            print("Could not prepare an extra script; the voice keeps fewer clips")
            self.bus.publish("progress", data={"overall": 0.6, "stage": "attribution", "step": 2, "steps": 4,
                                               "step_name": "Attribution", "step_fraction": 0.25, "streams": 3,
                                               "received": 4096, "activity": "Thinking: …", "totals": "LLM 3 active"})
            release.wait(10)
            return 0

        self.bus = EventBus()
        self.model = RunModel("refresh-directions — book")
        self.bus.subscribe(self.model.on_event)
        app = TaskApp(self.model, self.bus, command, title="refresh-directions — book")
        board = shell_board(self.bus)
        board.__enter__()
        self.addCleanup(board.__exit__, None, None, None)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: self.model.progress is not None)
            await pilot.pause(0.6)
            tabs = [pane.id for pane in app.query("TabPane")]
            self.assertEqual(tabs, ["summary", "log-tab", "errors-tab"])
            self.assertIn("Attribution 25%", str(app.query_one("#tile-step").render()))
            self.assertIn("4.0 KiB received", str(app.query_one("#tile-model-requests").render()))
            self.assertIn("1", str(app.query_one("#tile-warnings").render()))
            self.assertEqual(len(app.query_one("#recent").lines), 2)
            await pilot.press("2")
            self.assertEqual(app.query_one("#tabs").active, "log-tab")
            await pilot.press("q")
            self.assertIsInstance(app.screen, ConfirmScreen)
            await pilot.press("escape")
            release.set()
            await self.wait_for(pilot, lambda: not app.is_running)

    async def test_filters_and_tabs(self):
        release = threading.Event()

        def command():
            for line in ("[b1] Ch 1: generating", "[b2] Ch 1: generating", "[b1] Ch 1: Could not verify"):
                print(line)
            release.wait(5)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: len(self.model.logs) >= 3)
            await pilot.press("4")
            self.assertEqual(app.query_one("#tabs").active, "log-tab")
            await pilot.press("b")
            await pilot.press("down", "enter")
            self.assertEqual(self.model.filter.book, "b1")
            await pilot.press("v")
            self.assertEqual([event.text for _, event in self.model.visible_logs()], ["Could not verify"])
            await pilot.press("x")
            self.assertIsNone(self.model.filter.book)
            release.set()
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(app.result, 0)


class RendererCliShellTests(unittest.TestCase):
    def test_long_commands_open_the_shell_only_when_wanted(self):
        from epub_tts_narrator.renderer import cli
        with (patch("epub_tts_narrator.tui_support.wants_tui", return_value=True),
              patch("epub_tts_narrator.run_shell.run_in_shell", return_value=0) as shell):
            self.assertEqual(cli.main(["render", "book-a", "book-b"]), 0)
            kwargs = shell.call_args.kwargs
            self.assertEqual((kwargs["title"], kwargs["graceful_render_stop"], kwargs["resumable"]),
                             ("render — book-a, book-b", True, True))
            model = RunModel("render")
            model.book_states = {"book-a": {"state": "done"}, "book-b": {"state": "failed"}}
            with patch.object(cli, "main", return_value=0) as rerun:
                kwargs["resume_command"](model)()
                rerun.assert_called_once_with(["render", "book-b"], in_shell=True)
            self.assertIsNone(kwargs["resume_command"](RunModel("x")))
            cli.main(["assemble", "book-a"])
            self.assertFalse(shell.call_args.kwargs["graceful_render_stop"])
            with patch.object(cli, "load_config", side_effect=RuntimeError("stop here")):
                # Inside the shell the same main runs, never opening a second shell.
                self.assertEqual(shell.call_args.args[0](), 2)
            self.assertEqual(shell.call_count, 2)
        with (patch("epub_tts_narrator.tui_support.interactive_terminal", return_value=True),
              patch("epub_tts_narrator.tui_support.textual_available", return_value=True),
              patch("epub_tts_narrator.run_shell.run_in_shell") as shell,
              patch.object(cli, "load_config", side_effect=RuntimeError("stop here")),
              patch("sys.stderr", new_callable=io.StringIO)):
            self.assertEqual(cli.main(["render", "book-a", "--plain"]), 2)
            shell.assert_not_called()

class AnalyzerShellTests(unittest.TestCase):
    def test_model_commands_open_the_shell_and_fresh_runs_resume_by_their_printed_id(self):
        from epub_tts_narrator import cli
        with (patch("epub_tts_narrator.tui_support.wants_tui", return_value=True),
              patch("epub_tts_narrator.run_shell.run_in_shell", return_value=0) as shell):
            self.assertEqual(cli.main(["verify-attribution", "/books/one", "--chapter", "3"]), 0)
            kwargs = shell.call_args.kwargs
            self.assertEqual((kwargs["title"], kwargs["interactive_input"]), ("verify-attribution — one", True))
            model = RunModel("x")
            bus = EventBus()
            bus.subscribe(model.on_event)
            self.assertIsNone(kwargs["resume_command"](model))
            bus.log("Fresh verification run: 20261112-abc")
            with patch.object(cli, "main", return_value=0) as rerun:
                kwargs["resume_command"](model)()
            rerun.assert_called_once_with(["verify-attribution", "/books/one", "--chapter", "3", "--resume",
                                           "20261112-abc"], in_shell=True)
            cli.main(["analyze", "book.epub"])
            self.assertIsNone(shell.call_args.kwargs["resume_command"])
        with (patch("epub_tts_narrator.tui_support.wants_tui", return_value=False),
              patch("epub_tts_narrator.run_shell.run_in_shell") as shell,
              patch("sys.stderr", new_callable=io.StringIO)):
            self.assertEqual(cli.main(["analyze", "/missing/book.epub", "--plain"]), 1)
            shell.assert_not_called()

    def test_bulk_casting_opens_the_shell_and_only_safe_reruns_resume(self):
        from epub_tts_narrator.renderer import cli
        with (patch("epub_tts_narrator.tui_support.wants_tui", return_value=True),
              patch("epub_tts_narrator.run_shell.run_in_shell", return_value=0) as shell):
            cli.main(["cast", "book-a", "--action", "generate-missing"])
            self.assertEqual((shell.call_args.kwargs["title"], shell.call_args.kwargs["resumable"]),
                             ("cast generate-missing — book-a", True))
            cli.main(["cast", "book-a", "--action", "regenerate-all"])
            self.assertFalse(shell.call_args.kwargs["resumable"])
            self.assertEqual(shell.call_count, 2)

    def test_the_bus_stream_console_reports_stage_and_model_streams(self):
        from epub_tts_narrator.stream_progress import BusStreamConsole
        bus = EventBus()
        model = RunModel("analyze")
        bus.subscribe(model.on_event)
        console = BusStreamConsole(bus, draw_interval=0)
        console.progress("Attributing voices 1/4 complete")
        console.set_progress(0.6, "attribution")
        self.assertEqual((model.progress["step_name"], model.progress["streams"]), ("Attribution", 0))
        console.observe({"request_id": "r1", "phase": "started"})
        console.observe({"request_id": "r1", "phase": "streaming", "received": 2048, "events": 3})
        self.assertEqual(model.progress["streams"], 1)
        # The console's once-a-second heartbeat may draw this update a moment later.
        deadline = time.monotonic() + 3
        while "2.0 KiB received" not in model.progress["totals"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIn("2.0 KiB received", model.progress["totals"])
        self.assertIn("step 2/4 Attribution", model.summary())
        console.observe({"request_id": "r1", "phase": "completed"})
        console.close()
        self.assertEqual([event.text for _, event in model.logs], ["Attributing voices 1/4 complete"])

    def test_captured_lines_show_what_a_terminal_would(self):
        from epub_tts_narrator.tui_widgets import _visible
        self.assertEqual(_visible("\r\x1b[2Kspeech 3/9\rspeech 4/9"), "speech 4/9")
        self.assertEqual(_visible("\x1b[1;33mtarget\x1b[0m line\r"), "target line")


if __name__ == "__main__":
    unittest.main()
