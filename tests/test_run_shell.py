"""The run shell: events from render boards, the run model, and the headless Textual screen."""

import asyncio
from pathlib import Path
import io
import threading
import time
import unittest
from types import SimpleNamespace
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
        # A yes/no question gets both answers, its default first.
        self.assertEqual(prompt_choices("Download this model now? [y/N] "), [("n", "No"), ("y", "Yes")])
        self.assertEqual(prompt_choices("Continue? [Y/n]"), [("y", "Yes"), ("n", "No")])


class DialogClosesOnceTests(unittest.IsolatedAsyncioTestCase):
    """Two selections in one tick (a held Enter, a double click) once crashed casting with ScreenStackError."""

    async def test_a_second_selection_before_the_dialog_closes_is_ignored(self):
        import asyncio
        from textual.app import App
        from textual.widgets import OptionList
        from epub_tts_narrator.tui_widgets import ChoiceScreen
        asyncio.get_running_loop().slow_callback_duration = 10
        answers = []

        class Host(App):
            def on_mount(self):
                self.push_screen(ChoiceScreen("Generate", [("gemini", "Gemini"), ("local", "Local")],
                                              filterable=False), answers.append)
        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            choices = app.screen.query_one("#choices", OptionList)
            choices.highlighted = 0
            choices.action_select()
            choices.action_select()
            await pilot.pause()
        self.assertIsNone(app._exception)
        self.assertEqual(answers, ["gemini"])  # the callback ran once, so the voice is generated once


class SyncTableTests(unittest.IsolatedAsyncioTestCase):
    """Refreshing a table must not move the highlighted row or the scroll position."""

    async def test_updates_keep_the_cursor_and_scroll_and_a_reorder_keeps_the_row_on_its_line(self):
        import asyncio
        from rich.text import Text
        from textual.app import App
        from textual.widgets import DataTable
        from epub_tts_narrator.tui_widgets import highlighted_key, sync_table
        asyncio.get_running_loop().slow_callback_duration = 10
        keys = [f"v{n:02d}" for n in range(60)]

        class Host(App):
            CSS = "DataTable { height: 1fr; }"

            def compose(self):
                yield DataTable(cursor_type="row")

            def on_mount(self):
                table = self.query_one(DataTable)
                table.add_columns("Voice", "Status")
                sync_table(table, [(key, (key, "pending")) for key in keys])
        app = Host()
        async with app.run_test(size=(60, 12)) as pilot:
            table = app.query_one(DataTable)
            table.focus()
            await pilot.pause()
            table.move_cursor(row=30)
            await pilot.pause()
            table.scroll_to(y=27, animate=False)  # the highlighted row mid-view, not at an edge
            await pilot.pause()
            await pilot.pause()
            line = table.cursor_row - round(table.scroll_y)
            scroll = table.scroll_y
            self.assertGreater(scroll, 0)
            # A status changes in place: nothing moves, and the new style shows.
            sync_table(table, [(key, (key, Text("generating", style="cyan") if key == "v30" else "pending"))
                               for key in keys])
            await pilot.pause()
            self.assertEqual((table.scroll_y, highlighted_key(table)), (scroll, "v30"))
            self.assertEqual(str(table.get_row("v30")[1].style), "cyan")
            # Rows reorder (sorting by status): the voice stays highlighted on the same screen line.
            reordered = [key for key in keys if key != "v30"]
            reordered.insert(40, "v30")
            sync_table(table, [(key, (key, "pending")) for key in reordered])
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(highlighted_key(table), "v30")
            self.assertEqual(table.cursor_row - round(table.scroll_y), line)

    async def test_a_value_that_grows_in_place_widens_its_column(self):
        # "1421/5" for "1421/5830" misleads; a growing count must never be clipped.
        from textual.app import App
        from textual.widgets import DataTable
        from epub_tts_narrator.tui_widgets import sync_table

        class Host(App):
            def compose(self):
                yield DataTable()

            def on_mount(self):
                table = self.query_one(DataTable)
                table.add_columns("Book", "Blocks")
                sync_table(table, [("a", ("crystalsinger", "0/5"))])
        app = Host()
        async with app.run_test(size=(80, 10)) as pilot:
            table = app.query_one(DataTable)
            sync_table(table, [("a", ("crystalsinger", "1421/5830"))])
            await pilot.pause()
            blocks = table.ordered_columns[1]
            self.assertGreaterEqual(blocks.get_render_width(table) - 2 * table.cell_padding, len("1421/5830"))


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

    def test_a_failed_book_keeps_the_error_it_reported_even_when_its_log_line_comes_later(self):
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        board.chapter_failure({"number": 2, "error_type": "SpeechVerificationFailed", "error": "take 3 skipped words"},
                              book="b1")
        # The queue reports the failed state, then the outcome, then its "[b1] failed: ..." line.
        board.book_state("/q/b1", "failed", retry=lambda: True, error="Voices are missing: Mara")
        board.book_outcome("/q/b1", False)
        self.bus.log("[b1] failed: Voices are missing: Mara")
        self.bus.log("[b2] Renderer stopped: unrelated", severity="error")
        failure, chapters = self.model.book_failure("b1")
        self.assertEqual((failure["kind"], failure["error"], failure["state"]), ("book", "Voices are missing: Mara", "failed"))
        self.assertEqual([item["error"] for item in chapters], ["take 3 skipped words"])
        self.assertEqual(self.model.book_failure("b2"), (None, []))

    def test_a_chapter_that_failed_again_retries_from_any_of_its_rows_while_its_book_renders(self):
        # The case a person hit: retry, the chapter fails again, then r on its older row was refused.
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        requests = []
        record = {"number": 2, "error_type": "SpeechVerificationFailed", "error": "first"}
        board.chapter_failure(record, book="b1", retry=lambda: requests.append(1) or True)
        first = self.model.current_failures()[0]
        self.assertEqual(self.model.retry(first), "chapter")
        board.chapter_failure(dict(record, error="second"), book="b1", retry=lambda: requests.append(2) or True)
        # The older row is replaced, not left behind as "retrying".
        self.assertEqual([item["error"] for item in self.model.visible_failures()], ["second"])
        self.assertEqual(self.model.retry(first), "chapter")  # the old row retries the chapter's newest failure
        self.assertEqual(requests, [1, 2])
        board.chapter_state("/q/b1", 2, "done")
        self.assertEqual([item["state"] for item in self.model.visible_failures()], ["retried"])

    def test_a_retried_chapter_that_fails_again_says_so_once_with_the_new_takes_it_made(self):
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        record = {"number": 2, "error_type": "RenderBlocked", "error": "Every usable retained take misread a name"}
        board.chapter_failure(record, book="b1", retry=lambda: True)
        # A take rejected before the retry is not the retry's.
        board.rejected_take({"request_hash": "a" * 64, "attempt": 1, "reasons": ["pronunciation_mismatch"]},
                            None, book="b1")
        self.assertEqual(self.model.pop_notices(), [])
        self.assertEqual(self.model.retry(self.model.current_failures()[0]), "chapter")
        for attempt in (2, 3):
            self.bus.publish("rejected", "pronunciation_mismatch", scope={"book": "b1", "chapter": 2},
                             data={"request_hash": "a" * 64, "attempt": attempt,
                                   "reasons": ["pronunciation_mismatch"]})
        board.chapter_failure(record, book="b1", retry=lambda: True)
        [notice] = self.model.pop_notices()
        self.assertIn("Chapter 2 of b1 failed again after your retry: its 2 new takes were rejected.", notice)
        self.assertIn("misread a name", notice)
        self.assertEqual(self.model.pop_notices(), [])
        self.assertEqual(self.model.current_failures()[0]["after_retry"], 1)

    def test_a_chapter_whose_render_ended_retries_through_its_book_or_as_soon_as_one_can_take_it(self):
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        record = {"number": 3, "error_type": "SpeechVerificationFailed", "error": "silence"}
        board.chapter_failure(record, book="b1", retry=lambda: False)  # its render no longer takes retries
        chapter = self.model.current_failures()[0]
        self.assertEqual(self.model.retry(chapter), "later")
        self.assertEqual(self.model.wanted_books(), ["b1"])
        admitted = []
        board.book_state("/q/b1", "failed", retry=lambda: admitted.append("b1") or True, error="1 chapter failed")
        # The book's failed state re-admits it at once for the chapter that was waiting.
        self.assertEqual(admitted, ["b1"])
        self.assertEqual((chapter["state"], self.model.wanted_books()), ("retrying", []))
        # A chapter of a book that already failed goes back through the book.
        board.book_state("/q/b2", "rendering")
        board.chapter_failure(dict(record, number=1), book="b2", retry=lambda: False)
        board.book_state("/q/b2", "failed", retry=lambda: admitted.append("b2") or True, error="failed")
        self.assertEqual(self.model.retry(self.model.chapter_failure("b2", 1)), "book")
        self.assertEqual(admitted, ["b1", "b2"])

    def test_attempts_are_summed_up_in_plain_words(self):
        from epub_tts_narrator.run_shell_model import plain_failure
        silence = {"passed": False, "reasons": ["generation_or_verification_error"], "echo": False,
                   "error": "EngineError: Drama 3 did not finish the text: 'silence'"}
        self.assertEqual(plain_failure([silence] * 3),
                         "Drama 3 went silent partway through this passage and stopped, on all 3 attempts; "
                         "no audio was kept.")
        heard = {"passed": False, "reasons": ["text_mismatch"], "echo": False, "error": ""}
        self.assertEqual(plain_failure([silence, silence, heard]),
                         "No take passed in 3 attempts: 2 Drama 3 went silent partway through and stopped; "
                         "1 was heard saying different words.")
        self.assertEqual(plain_failure([heard]), "No take passed: its one attempt was heard saying different words.")
        self.assertEqual(plain_failure([]), "")

    def test_every_planned_chapter_is_listed_in_order_with_its_state_and_a_breakdown(self):
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        board.book_state("/q/b2", "queued")
        board.chapter_plan("/q/b2", [{"number": 1, "title": "Dawn", "blocks": None, "saved": False},
                                     {"number": 2, "title": "Dusk", "blocks": None, "saved": False}], outline=True)
        board.chapter_plan("/q/b1", [{"number": number, "title": f"Part {number}", "blocks": 3, "saved": number == 1}
                                     for number in (1, 2, 3, 4)])
        self.assertEqual([chapter["state"] for _, chapter in self.model.visible_chapters()][:1], ["saved"])
        with ChapterProgress(2, 4, chunks(3), notify=PrefixedBoardNotify(board, "[b1] "), clock=self.clock,
                             heartbeat=False) as chapter:
            with chapter.task(0):
                pass
            chapter.refresh()
            live = dict(self.model.visible_chapters())["b1\x1f2"]
            self.assertEqual((live["state"], live["done"], live["blocks"]), ("rendering", 1, 3))
            self.assertIsNotNone(live["snapshot"])
        board.chapter_state("/q/b1", 2, "assembling")
        board.chapter_state("/q/b1", 2, "done")
        # Done once its assembly is saved, and again when the run publishes it in order.
        board.chapter_state("/q/b1", 2, "done")
        board.chapter_state("/q/b1", 1, "done", reused=True)
        board.chapter_failure({"number": 3, "error_type": "SpeechVerificationFailed", "error": "take 3 skipped words"},
                              book="b1", retry=lambda: True)
        board.chapter_failure({"number": 4, "error_type": "RenderBlocked", "error": "bad clip"}, book="b1")
        self.bus.publish("rejected", "skipped words", scope={"book": "b1", "chapter": 3},
                         data={"request_hash": "h", "attempt": 1})
        rows = self.model.visible_chapters()
        # Queue order, then chapter order; a queued book's chapters are listed before it starts.
        self.assertEqual([key for key, _ in rows], ["b1\x1f1", "b1\x1f2", "b1\x1f3", "b1\x1f4", "b2\x1f1", "b2\x1f2"])
        self.assertEqual([(item["state"], item["reused"]) for _, item in rows],
                         [("done", True), ("done", False), ("failed", False), ("blocked", False),
                          ("not started", False), ("not started", False)])
        self.assertEqual((rows[1][1]["done"], rows[2][1]["rejected"], rows[3][1]["error"]), (3, 1, "bad clip"))
        self.assertEqual(self.model.chapter_breakdown(),
                         [("b1", 4, {"done": 2, "failed": 1, "blocked": 1}, 1), ("b2", 2, {"not started": 2}, 0)])
        self.assertEqual(self.model.chapter_states(), ["not started", "done", "failed", "blocked"])
        self.model.filter.state = "failed"
        self.assertEqual([key for key, _ in self.model.visible_chapters()], ["b1\x1f3"])
        self.model.filter.state, self.model.filter.book = None, "b2"
        self.assertEqual(len(self.model.visible_chapters()), 2)
        self.assertEqual([book for book, *_ in self.model.chapter_breakdown()], ["b2"])
        self.model.filter.book = None
        # A late queue outline never replaces a book's plan; the book's plan replaces its outline.
        board.chapter_plan("/q/b1", [{"number": 1, "title": "", "blocks": None, "saved": False}], outline=True)
        board.chapter_plan("/q/b2", [{"number": number, "title": "", "blocks": 2, "saved": False}
                                     for number in (1, 2, 3)])
        board.book_state("/q/b2", "rendering")
        self.assertEqual([item["state"] for key, item in self.model.visible_chapters() if key.startswith("b2")],
                         ["queued"] * 3)
        self.assertEqual(len([key for key, _ in self.model.visible_chapters() if key.startswith("b1")]), 4)
        failure = next(item for item in self.model.current_failures() if item["scope"].get("chapter") == 3)
        self.assertTrue(self.model.retry(failure))
        self.assertEqual(dict(self.model.visible_chapters())["b1\x1f3"]["state"], "retrying")
        self.model.finish(130)
        self.assertEqual(dict(self.model.visible_chapters())["b1\x1f3"]["state"], "stopped")
        self.model.restart()
        self.assertEqual([item["state"] for key, item in self.model.visible_chapters() if key.startswith("b1")],
                         ["done", "done", "not started", "not started"])

    def test_book_progress_never_drops_when_a_chapter_stops_rendering(self):
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        board.chapter_plan("/q/b1", [{"number": n, "title": f"C{n}", "blocks": 10, "saved": False}
                                     for n in (1, 2, 3, 4)])
        def status(key, chapter, done):
            self.bus.publish("status", scope={"book": "b1", "chapter": chapter}, data={"key": key, "snapshot": {
                "chapter": chapter, "total_chapters": 4, "done": done, "blocks": 10, "elapsed": 1,
                "active": []}})
        status("k1", 1, 10)
        status("k2", 2, 5)
        percent = lambda: self.model.book_rows()[0][1]
        self.assertEqual(percent(), 37)  # (10 + 5) of 40 blocks
        # Chapter 1 leaves the board to be assembled; chapter 2 is blocked with half its takes saved.
        self.bus.publish("status_removed", data={"key": "k1"})
        board.chapter_state("/q/b1", 1, "assembling")
        self.assertEqual(percent(), 37)
        self.bus.publish("status_removed", data={"key": "k2"})
        board.chapter_state("/q/b1", 2, "failed")
        self.assertEqual(percent(), 37)
        board.chapter_state("/q/b1", 1, "done")
        self.assertEqual(percent(), 37)
        status("k3", 3, 4)
        self.assertEqual(percent(), 47)

    def test_chapters_with_no_planned_speech_work_stay_hidden_from_every_view(self):
        board = BusBoard(self.bus)
        board.book_state("/q/b1", "rendering")
        board.chapter_plan("/q/b1", [{"number": 1, "title": "Spoken", "blocks": 2, "saved": False},
                                     {"number": 2, "title": "Truly empty", "blocks": 0, "saved": False},
                                     {"number": 4, "title": "Silence controls only", "blocks": 1, "saved": False}])
        # State events for the empty chapter cannot bring it back; silence-control chapters stay.
        board.chapter_state("/q/b1", 2, "done")
        board.chapter_state("/q/b1", 1, "done")
        self.assertEqual([key for key, _ in self.model.visible_chapters()], ["b1\x1f1", "b1\x1f4"])
        self.assertEqual(self.model.chapter_breakdown(), [("b1", 2, {"done": 1, "queued": 1}, 0)])
        book, percent, state, ready, total, *_ = self.model.book_rows()[0]
        self.assertEqual((book, percent, state, ready, total), ("b1", 50, "rendering", 1, 2))
        # A live row and a completion for the hidden chapter do not surface anywhere.
        with ChapterProgress(2, 4, chunks(1), notify=PrefixedBoardNotify(board, "[b1] "), clock=self.clock,
                             heartbeat=False) as chapter:
            chapter.refresh()
            self.assertEqual(self.model.visible_rows(), [])
            self.assertNotIn("chapter(s) active", self.model.summary())
        board.chapter_state("/q/b1", 4, "done")
        self.model.finish(0)
        self.assertNotIn("chapter(s) finished", self.model.summary())
        # Source numbering of visible chapters keeps its gaps: 1 and 4, never renumbered.
        self.assertEqual(self.model.chapter_states(), ["done"])

    def test_live_rows_of_a_command_without_a_plan_still_show(self):
        board = BusBoard(self.bus)
        with ChapterProgress(5, 9, chunks(2), notify=PrefixedBoardNotify(board, "[b9] "), clock=self.clock,
                             heartbeat=False) as chapter:
            chapter.refresh()
            rows = self.model.visible_chapters()
            self.assertEqual([(item["book"], item["chapter"], item["state"]) for _, item in rows], [("b9", 5, "rendering")])
            self.assertTrue(all(type(key) is str for key, _ in rows))
        self.assertEqual(self.model.visible_chapters(), [])

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


class MoreBooksModelTests(unittest.TestCase):
    """A run shell adds books to the running render queue and clears finished ones from its lists."""

    def setUp(self):
        self.bus = EventBus()
        self.model = RunModel("render")
        self.bus.subscribe(self.model.on_event)
        self.board = BusBoard(self.bus)

    def test_paths_are_read_whole_or_shell_quoted(self):
        from epub_tts_narrator.run_shell_model import book_paths
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            spaced = Path(root) / "The Last Lantern"
            spaced.mkdir()
            self.assertEqual(book_paths(f"  {spaced}  "), [str(spaced)])
            self.assertEqual(book_paths(f"'{spaced}'"), [str(spaced)])
        self.assertEqual(book_paths("/books/one /books/Two\\ Words"), ["/books/one", "/books/Two Words"])
        self.assertEqual(book_paths("   "), [])

    def test_books_go_to_the_queue_and_an_ended_queue_takes_none(self):
        calls = []
        self.board.book_adder(lambda path: (calls.append(path), (path != "/bad", f"answer {path}"))[1])
        self.assertEqual(self.model.add_books(["/books/two", "/bad"]),
                         [("/books/two", True, "answer /books/two"), ("/bad", False, "answer /bad")])
        self.board.book_adder(lambda path: 1 / 0)
        self.assertEqual(self.model.add_books(["/books/three"])[0][1], False)  # an error never escapes
        self.board.book_adder(None)
        self.assertEqual(self.model.add_books(["/books/four"]),
                         [("/books/four", False, "This render takes no more books now")])
        self.assertEqual(calls, ["/books/two", "/bad"])

    def test_a_queue_waiting_for_more_books_is_idle_not_waiting_for_retries(self):
        self.board.retry_hold(lambda: None, 0)
        self.assertTrue(self.model.idle())
        self.assertEqual(self.model.health()[0], "ok")
        self.assertIn("waiting for more books (a adds, q closes)", self.model.summary())
        self.board.retry_hold(None, 0)
        self.board.retry_hold(lambda: None, 1)
        self.assertFalse(self.model.idle())
        self.assertIn("waiting for retries", self.model.summary())

    def test_dismissing_clears_only_finished_books_from_every_list(self):
        for book, state in (("/b/done", "done"), ("/b/failed", "failed"), ("/b/live", "rendering")):
            self.board.book_state(book, state)
            self.board.chapter_plan(book, [{"number": 1, "title": "One", "blocks": 2}])
        self.board.book_outcome("/b/done", True)
        self.bus.publish("rejected", "text_mismatch", scope={"book": "done", "chapter": 1},
                         data={"request_hash": "h", "attempt": 1})
        self.model.filter.book = "done"
        self.assertEqual(self.model.dismiss_done(), ["done"])
        self.assertEqual(list(self.model.book_states), ["failed", "live"])
        self.assertNotIn("done", self.model.chapter_plans)
        self.assertEqual(list(self.model.rejections), [])
        self.assertIsNone(self.model.filter.book)
        self.assertNotIn("done", self.model.choices("book"))
        self.assertIn("1 dismissed", self.model.glance()["Books"][1])
        self.assertEqual(self.model.dismiss_done(), [])
        self.board.book_state("/b/done", "queued")  # added again: listed anew
        self.assertEqual(list(self.model.book_states), ["failed", "live", "done"])


class NodeSwitchModelTests(unittest.TestCase):
    """What the run shell offers and shows for speech nodes a person turns off and on."""

    def nodes(self, **states):
        from epub_tts_narrator.renderer.node_activity import NodeActivity
        activity = NodeActivity()
        owner = object()
        for name, state in states.items():
            activity.register(owner, name, "speech", 1, local=name == "local",
                              switch=lambda on, notify=None: "ok", state=lambda state=state: state)
        activity.register(owner, "local", "verify", 1, local=True)
        return activity.snapshot()

    def test_each_speech_node_offers_the_change_it_can_make(self):
        model = RunModel("render")
        choices = dict(model.node_switch_choices(self.nodes(local="on", far="off", near="turning on")))
        self.assertEqual(set(choices), {("local", False), ("far", True), ("near", None)})
        self.assertIn("shuts down and its memory is freed", choices[("local", False)])
        self.assertIn("sends it speech again", choices[("far", True)])
        local_off = dict(model.node_switch_choices(self.nodes(local="off")))
        self.assertIn("loading the model takes a few minutes", local_off[("local", True)])
        self.assertEqual(model.node_states(self.nodes(local="off", far="on")), {"local": "off"})

    def test_with_every_speech_node_off_the_render_is_waiting_not_stalled(self):
        clock = SimpleNamespace(now=0.0)
        model = RunModel("render", clock=lambda: clock.now)
        model.rows["c"] = {"scope": {"book": "b", "chapter": 1}, "snapshot": {}}
        model.set_nodes(self.nodes(local="off", far="away"))
        clock.now += 10_000
        self.assertEqual(model.health()[0], "warning")
        self.assertIn("Every speech node is off", model.health()[1])
        model.set_nodes(self.nodes(local="on", far="away"))  # turned on: the stall clock starts again
        self.assertEqual(model.health(), ("ok", "Running normally"))


class VerdictCellTests(unittest.TestCase):
    def test_every_verdict_a_person_can_set_is_shown_as_judged(self):
        from epub_tts_narrator.renderer.take_feedback import VERDICTS
        from epub_tts_narrator.run_shell import _verdict_cell
        for verdict in VERDICTS:
            self.assertEqual(str(_verdict_cell(verdict)), verdict)
        self.assertEqual(str(_verdict_cell(None)), "not judged")


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
            # After the run, the repairs on offer are rendering it again and reading its messages.
            self.assertEqual(app.screen.actions, ["resume", "log", "close"])
            await pilot.press("escape")
            await pilot.press("R")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual((len(attempts), app.result, self.model.state), (2, 0, "finished"))
        self.assertIn("Completed audiobook: /books/fixture/audiobook",
                      [event.text for _, event in self.model.logs])

    async def test_books_are_added_while_it_runs_finished_ones_dismissed_and_q_closes_when_all_are_done(self):
        import tempfile
        root = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, root, True)
        extra = root / "Second Book"
        extra.mkdir()
        (extra / "manifest.json").write_text("{}")
        added, closed = [], threading.Event()

        def add(path):
            added.append(path)
            progress.report_book_state(notify_box[0], path, "queued")
            return True, f"Added {Path(path).name}"
        notify_box = []

        def command():
            with live_board(print) as notify:
                notify_box.append(notify)
                progress.report_book_adder(notify, add)
                progress.report_book_state(notify, "/books/first", "rendering")
                report_book_outcome(notify, "/books/first", True)
                progress.report_retry_hold(notify, closed.set, 0)
                closed.wait(10)
                progress.report_retry_hold(notify, None, 0)
                progress.report_book_adder(notify, None)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, self.model.idle)
            await pilot.press("a")
            from epub_tts_narrator.tui_widgets import TextEditScreen
            self.assertIsInstance(app.screen, TextEditScreen)
            app.screen.query_one("#editor").value = str(root / "missing")
            await pilot.press("enter")
            self.assertIsInstance(app.screen, TextEditScreen)  # not a book: the dialog says why and stays
            app.screen.query_one("#editor").value = f"'{extra}'"
            await pilot.press("enter")
            await self.wait_for(pilot, lambda: added)
            self.assertEqual(added, [str(extra)])
            await self.wait_for(pilot, lambda: "Second Book" in self.model.book_states)
            await pilot.press("X")
            self.assertEqual(list(self.model.book_states), ["Second Book"])
            self.model.book_states["Second Book"]["state"] = "done"
            await pilot.press("q")  # every book finished: closing asks nothing
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertTrue(closed.is_set())
        self.assertEqual(app.result, 0)

    async def test_p_pauses_new_requests_until_pressed_again(self):
        from unittest import mock
        from epub_tts_narrator.renderer import render_pause
        gate = render_pause.PauseGate()
        sent, go = [], threading.Event()

        def command():
            go.wait(5)
            gate.call(sent.append, "request")  # what a take or review request does
            return 0

        app = self.make_app(command)
        with mock.patch.object(render_pause, "PAUSE", gate):
            async with app.run_test(size=(150, 40)) as pilot:
                await pilot.press("p")
                self.assertTrue(gate.paused())
                self.assertIn("Paused", self.model.health()[1])
                go.set()
                await pilot.pause(0.3)
                self.assertEqual(sent, [])
                await pilot.press("p")
                await self.wait_for(pilot, lambda: not self.model.running)
        self.assertEqual(sent, ["request"])
        self.assertFalse(self.model.paused)

    async def test_o_turns_a_speech_node_off_and_on_from_a_dialog(self):
        from unittest import mock
        from epub_tts_narrator.renderer import node_activity
        from epub_tts_narrator.tui_widgets import ChoiceScreen
        activity, state, calls, done = node_activity.NodeActivity(), ["on"], [], threading.Event()

        def switch(on, notify=None):
            calls.append(on)
            state[0] = "on" if on else "off"
            notify("Render node local is off. o turns it on again.")
            return "Turning local " + ("on" if on else "off")
        activity.register(object(), "local", "speech", 1, local=True, switch=switch, state=lambda: state[0])

        app = self.make_app(lambda: (done.wait(10), 0)[1])
        with mock.patch.object(node_activity, "ACTIVITY", activity):
            async with app.run_test(size=(150, 40)) as pilot:
                await pilot.press("o")
                self.assertIsInstance(app.screen, ChoiceScreen)
                await pilot.press("enter")
                await self.wait_for(pilot, lambda: calls == [False])
                await self.wait_for(pilot, lambda: self.model.speech_nodes_off)
                self.assertIn("(off)", str(app.query_one("#nodes-line").render()))
                logged = [event.text for _, event in self.model.logs]
                self.assertIn("Turning local off", logged)
                self.assertIn("Render node local is off. o turns it on again.", logged)
                await pilot.press("o", "enter")
                await self.wait_for(pilot, lambda: calls == [False, True])
                done.set()
                await self.wait_for(pilot, lambda: not self.model.running)
                await pilot.press("o")  # the render has ended: nothing to turn on or off
                self.assertNotIsInstance(app.screen, ChoiceScreen)

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

    async def test_a_failed_passage_shows_its_full_error_every_attempt_and_one_retry_with_new_takes(self):
        long_error = "Speech verification failed after 3 attempts in this run. " + "x" * 300 + " END-OF-ERROR"
        calls = []
        retried = threading.Event()

        class Review:
            book = Path("/books/fixture")

            def passage(self):
                return {"text": "Nobody answered the door.", "kind": "speech", "segment_ids": ["s9"],
                        "directions": ["softly"]}

            def attempts(self):
                return [{"attempt": number, "audio": False, "passed": False,
                         "reasons": ["generation_or_verification_error"], "echo": False, "pending": False,
                         "error": "EngineError: Drama 3 did not finish the text: 'silence'"} for number in (1, 2, 3)]

            def takes(self):
                return []

            def reset_for_fresh_takes(self):
                calls.append("reset")
                return ["exhaustion-selection.fresh-retry-abc.json"]

        def command():
            with live_board(print) as notify:
                report_chapter_failure(notify, {"number": 4, "error_type": "SpeechVerificationFailed",
                                                "error": long_error, "request_hash": "h" * 64, "block": 7,
                                                "speakers": ["Mara"], "segment_ids": ["s9"]},
                                       retry=lambda: retried.set() or True, repair=Review())
                if not retried.wait(10):
                    return 1
            return 0

        from epub_tts_narrator.run_shell import _error_cell
        self.assertTrue(_error_cell(long_error).endswith("… (Enter: full error)"))
        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: bool(self.model.current_failures()))
            await pilot.press("5")
            app.query_one("#errors").focus()
            await pilot.press("d")  # the row in full, whatever the column width cut
            from epub_tts_narrator.run_shell_screens import FailureScreen, failure_text
            self.assertIsInstance(app.screen, FailureScreen)
            text = failure_text(app.screen.item, app.screen.hint, app.screen.details, app.screen.summary)[1]
            # Plain words first, then the whole error, never cut short.
            self.assertTrue(text.startswith("Drama 3 went silent partway through this passage and stopped, on all "
                                            "3 attempts"), text[:120])
            self.assertIn("Error: " + long_error, text)
            self.assertIn("Failed passage: Chapter 4, block 7, spoken by Mara", text)
            self.assertIn("Text: Nobody answered the door.", text)
            self.assertEqual(text.count("EngineError: Drama 3 did not finish the text: 'silence'"), 3)
            self.assertIn("No attempt made audio", text)
            # One retry, and it always makes new takes: never a retry that only re-reads the failure.
            self.assertEqual(app.screen.actions, ["retry", "edit_direction", "log", "close"])
            await pilot.press("enter")
            await self.wait_for(pilot, lambda: not self.model.running)
        self.assertEqual((calls, retried.is_set(), app.result), (["reset"], True, 0))

    async def test_a_retry_asked_for_when_no_render_can_take_it_resumes_its_book_when_the_run_ends(self):
        from epub_tts_narrator.renderer.progress import report_book_state
        attempts = []

        def command():
            attempts.append(1)
            with live_board(print) as notify:
                if len(attempts) == 1:
                    report_book_state(notify, "/q/b1", "rendering")
                    report_chapter_failure(PrefixedBoardNotify(notify.__self__, "[b1] "),
                                           {"number": 2, "error_type": "SpeechVerificationFailed", "error": "x"},
                                           retry=lambda: False)
                    deadline = time.monotonic() + 10
                    while not self.model.wanted_books() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    return 2
            return 0

        app = self.make_app(command)
        app.resume_command = lambda model, books: (lambda: (attempts.append(tuple(books or ())), 0)[1])
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: bool(self.model.current_failures()))
            await pilot.press("5")
            app.query_one("#errors").focus()
            await pilot.press("r")
            await self.wait_for(pilot, lambda: not app.is_running)
        # The first run ended with the retry still waiting, so the shell rendered that book again.
        self.assertEqual(attempts, [1, ("b1",)])

    async def test_r_in_chapters_retries_the_highlighted_chapter_and_names_it_by_the_number_shown(self):
        # The case a person hit: the row "10 | Chapter Two" answered about chapter 16, Errors' highlight.
        requests, done = [], threading.Event()

        def command():
            with live_board(print) as notify:
                board = notify.__self__
                board.book_state("/q/b4", "rendering")
                board.chapter_plan("/q/b4", [
                    {"number": 1, "title": "Illustration", "blocks": 0, "saved": False},  # hidden: no speech
                    *({"number": number, "title": title, "blocks": 456, "saved": False}
                      for number, title in ((10, "Chapter Two"), (16, "Chapter Eight")))])
                prefixed = PrefixedBoardNotify(board, "[b4] ")
                report_chapter_failure(prefixed, {"number": 16, "error_type": "SpeechVerificationFailed",
                                                  "error": "silence"}, retry=lambda: requests.append(16) or True)
                report_chapter_failure(prefixed, {"number": 10, "error_type": "RenderBlocked", "error": "blocked"},
                                       retry=lambda: requests.append(10) or True)
                done.wait(10)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(160, 40)) as pilot:
            await self.wait_for(pilot, lambda: len(self.model.current_failures()) == 2)
            await pilot.press("5")
            errors = app.query_one("#errors")
            errors.focus()
            errors.move_cursor(row=0)  # chapter 16, the first failure
            await pilot.press("r")
            self.assertEqual(requests, [16])
            await pilot.press("2")
            rows = app.query_one("#rows")
            await self.wait_for(pilot, lambda: rows.row_count == 2)
            self.assertEqual([str(rows.get_row_at(row)[1]) for row in range(2)], ["10", "16"])
            rows.focus()
            rows.move_cursor(row=0)
            notices = []
            app.notify = lambda message, **kwargs: notices.append(message)
            await pilot.press("r")
            # The highlighted chapter, named by the number its row shows (and its title).
            self.assertEqual(requests, [16, 10])
            self.assertTrue(notices and notices[-1].startswith("Retrying chapter 10 (Chapter Two) first"), notices)
            await pilot.press("r")
            self.assertEqual(notices[-1], "Chapter 10 (Chapter Two) is already being rendered again.")
            done.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_errors_keep_their_highlight_when_an_earlier_row_is_replaced(self):
        board_ready = threading.Event()
        release = threading.Event()
        handles = {}

        def command():
            with live_board(print) as notify:
                handles["notify"] = PrefixedBoardNotify(notify.__self__, "[b1] ")
                for number in (3, 7):
                    report_chapter_failure(handles["notify"], {"number": number, "error_type": "RenderBlocked",
                                                               "error": f"failed {number}"}, retry=lambda: True)
                board_ready.set()
                release.wait(10)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(160, 40)) as pilot:
            await self.wait_for(pilot, board_ready.is_set)
            await pilot.press("5")
            errors = app.query_one("#errors")
            await self.wait_for(pilot, lambda: errors.row_count == 2)
            errors.focus()
            errors.move_cursor(row=1)  # chapter 7
            # Chapter 3 fails again: its older row is replaced by a new one at the end.
            report_chapter_failure(handles["notify"], {"number": 3, "error_type": "RenderBlocked",
                                                       "error": "failed 3 again"}, retry=lambda: True)
            app.refresh_view(force=True)
            await pilot.pause()
            self.assertEqual(app._highlighted_failure()["scope"]["chapter"], 7)
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_enter_on_a_failed_book_in_summary_shows_why_and_resumes_it(self):
        from epub_tts_narrator.renderer.progress import report_book_state
        retried = threading.Event()

        def command():
            with live_board(print) as notify:
                report_book_state(notify, "/q/b1", "queued")
                report_book_state(notify, "/q/b2", "rendering")
                report_chapter_failure(PrefixedBoardNotify(notify.__self__, "[b1] "),
                                       {"number": 3, "error_type": "RenderBlocked", "error": "bad clip"})
                report_book_state(notify, "/q/b1", "failed", retry=lambda: retried.set() or True,
                                  error="Voices are missing: Mara")
                print("[b1] failed: Voices are missing: Mara")
                if not retried.wait(10):
                    return 1
                report_book_state(notify, "/q/b1", "preparing")
                report_book_state(notify, "/q/b1", "done")
                report_book_state(notify, "/q/b2", "done")
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            await self.wait_for(pilot, lambda: self.model.book_failure("b1")[0] is not None)
            await pilot.press("1")
            books = app.query_one("#books")
            await self.wait_for(pilot, lambda: books.row_count == 2)
            books.focus()
            books.move_cursor(row=0)
            await pilot.press("enter")
            from epub_tts_narrator.run_shell_screens import FailureScreen, failure_text
            self.assertIsInstance(app.screen, FailureScreen)
            text = failure_text(app.screen.item, app.screen.hint, app.screen.details)[1]
            self.assertIn("Voices are missing: Mara", text)
            self.assertIn("Chapter 3 (the chapter stopped outside any one passage's takes)\n  RenderBlocked: bad clip",
                          text)
            self.assertEqual(app.screen.actions, ["retry_book", "chapter:3", "chapters", "close"])
            await pilot.press("enter")  # the first repair: resume it now
            await self.wait_for(pilot, lambda: not self.model.running)
        self.assertTrue(retried.is_set())
        self.assertEqual(app.result, 0)

    async def test_enter_on_a_book_that_has_not_failed_drills_into_its_chapters(self):
        from epub_tts_narrator.renderer.progress import report_book_state
        release = threading.Event()

        def command():
            with live_board(print) as notify:
                report_book_state(notify, "/q/b1", "rendering")
                release.wait(10)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(150, 40)) as pilot:
            books = app.query_one("#books")
            await self.wait_for(pilot, lambda: books.row_count == 1)
            books.focus()
            await pilot.press("enter")
            self.assertEqual((self.model.filter.book, app.query_one("#tabs").active), ("b1", "chapters"))
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

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

    async def test_unsettled_speakers_are_chosen_on_their_own_screen_inside_the_shell(self):
        import sys
        from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
        from epub_tts_narrator.speaker_tui import SpeakerScreen
        from epub_tts_narrator.tui_support import screen_host
        from epub_tts_narrator.tui_widgets import PromptBridge
        from test_speaker_decisions import add_lena, missing
        document = missing()
        add_lena(document)
        results = []

        def command():
            results.append(prompt_speaker_decisions(document))
            return 0

        bridge = PromptBridge(None)
        app = self.make_app(command)
        app.interactive_input, app.bridge, bridge.app = True, bridge, app
        previous = sys.stdin
        sys.stdin = bridge.input
        self.addCleanup(setattr, sys, "stdin", previous)
        with screen_host(bridge.show):
            async with app.run_test(size=(140, 40)) as pilot:
                await self.wait_for(pilot, lambda: isinstance(app.screen, SpeakerScreen))
                await pilot.pause()
                await pilot.press("slash", *"lena", "enter")
                await self.wait_for(pilot, lambda: not self.model.running)
        self.assertEqual(results[0]["segments"][1]["role_id"], "lena")

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

    async def test_the_chapters_tab_lists_every_chapter_and_keeps_the_cursor_while_states_change(self):
        from epub_tts_narrator.renderer.progress import report_book_state, report_chapter_plan, report_chapter_state
        step, release = threading.Event(), threading.Event()

        def command():
            with live_board(print) as notify:
                report_book_state(notify, "fixture", "rendering")
                report_chapter_plan(notify, "fixture", [
                    {"number": number, "title": title, "blocks": 4, "saved": number == 1}
                    for number, title in ((1, "Harbor"), (2, "Storm"), (3, "Lantern"))])
                report_chapter_state(notify, "fixture", 1, "done", reused=True)
                report_chapter_failure(notify, {"number": 3, "error_type": "SpeechVerificationFailed",
                                                "error": "take 3 skipped words"})
                step.wait(5)
                report_chapter_state(notify, "fixture", 2, "rendering")
                release.wait(5)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(160, 40)) as pilot:
            await self.wait_for(pilot, lambda: len(self.model.visible_chapters()) == 3
                                and self.model.visible_chapters()[2][1]["state"] == "failed")
            await pilot.press("2")
            app.refresh_view(force=True)
            await pilot.pause()
            table = app.query_one("#rows")
            self.assertEqual([key.value for key in table.rows], ["fixture\x1f1", "fixture\x1f2", "fixture\x1f3"])
            self.assertEqual([str(cell) for cell in table.get_row("fixture\x1f1")[2:5]],
                             ["Harbor", "done · earlier run", "4/4 (100%)"])
            self.assertEqual(str(table.get_row("fixture\x1f2")[3]), "queued")
            breakdown = str(app.query_one("#chapter-breakdown").render())
            self.assertIn("3 chapters · 1 queued · 1 done (1 from an earlier run) · 1 failed", breakdown)
            self.assertEqual(str(app.query_one("#tabs").get_tab("chapters").label), "Chapters (1/3 done)")
            table.focus()
            table.move_cursor(row=1)
            step.set()
            await self.wait_for(pilot, lambda: self.model.visible_chapters()[1][1]["state"] == "rendering")
            app.refresh_view(force=True)
            await pilot.pause()
            self.assertEqual(str(table.get_row("fixture\x1f2")[3]), "rendering")
            self.assertEqual(table.cursor_row, 1)  # updated in place: the highlighted chapter stays put
            await pilot.press("s")
            await pilot.press("down", "down", "down", "enter")  # Any, rendering, done, failed
            self.assertEqual(self.model.filter.state, "failed")
            await pilot.pause()
            self.assertEqual([key.value for key in table.rows], ["fixture\x1f3"])
            self.assertIn("Showing 1 of 3", str(app.query_one("#chapter-breakdown").render()))
            await pilot.press("enter")
            # A failed chapter opens its failure, with its full error and repairs.
            from epub_tts_narrator.run_shell_screens import FailureScreen
            self.assertIsInstance(app.screen, FailureScreen)
            self.assertEqual(app.screen.item["scope"].get("chapter"), 3)
            await pilot.press("escape")
            release.set()
            # A failed chapter keeps the screen open after the command ends.
            await self.wait_for(pilot, lambda: not self.model.running)
            await pilot.press("q")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual((app.result, self.model.state), (0, "failed"))

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
