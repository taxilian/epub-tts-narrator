"""Completed-chapter collapsing in the run shell's Chapters view.

The model groups all finished chapters of one book into one summary row when
there are at least two (on by default; ``collapse_completed`` turns it off), and the
screen gets ``E`` to toggle, Enter on a summary to expand it, and cursor
remapping so the highlight follows its chapter. Raw counts (breakdown,
states) never change with the flag.
"""

import asyncio
import threading
import time
import unittest

from epub_tts_narrator.renderer.progress import (
    BusBoard, live_board, report_book_state, report_chapter_failure, report_chapter_plan, report_chapter_state,
    shell_board,
)
from epub_tts_narrator.run_events import EventBus
from epub_tts_narrator.run_shell_model import RunModel
from test_run_shell import Clock


def planned(board, alpha=(1, 2, 3, 4, 5, 6), beta=(1, 2, 3)):
    board.book_state("/q/alpha", "rendering")
    board.chapter_plan("/q/alpha", [{"number": n, "title": f"C{n}", "blocks": 2, "saved": False} for n in alpha])
    board.book_state("/q/beta", "rendering")
    board.chapter_plan("/q/beta", [{"number": n, "title": f"C{n}", "blocks": 2, "saved": False} for n in beta])


def done(board, book, *numbers, reused=()):
    for number in numbers:
        board.chapter_state(f"/q/{book}", number, "done", reused=number in reused)


class CompletedCollapseModelTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.bus = EventBus(clock=self.clock)
        self.model = RunModel("render", clock=self.clock)
        self.bus.subscribe(self.model.on_event)
        self.board = BusBoard(self.bus)

    def keys(self):
        return [key for key, _ in self.model.visible_chapters()]

    def test_zero_or_one_done_chapter_stays_an_individual_row(self):
        planned(self.board)
        done(self.board, "alpha", 1)
        self.assertEqual(self.keys(), [f"alpha\x1f{n}" for n in (1, 2, 3, 4, 5, 6)]
                         + [f"beta\x1f{n}" for n in (1, 2, 3)])
        # A single finished chapter at the end of a book is not a group either.
        done(self.board, "beta", 3)
        self.assertEqual(self.keys()[-3:], ["beta\x1f1", "beta\x1f2", "beta\x1f3"])

    def test_two_or_more_done_chapters_become_one_summary_row(self):
        planned(self.board)
        done(self.board, "alpha", 1, 2)
        keys = self.keys()
        [summary] = [key for key in keys if "\x1fdone\x1f" in key]
        self.assertEqual(summary, "alpha\x1fdone\x1f1-2")
        self.assertEqual(keys, ["alpha\x1fdone\x1f1-2", "alpha\x1f3", "alpha\x1f4", "alpha\x1f5", "alpha\x1f6",
                                "beta\x1f1", "beta\x1f2", "beta\x1f3"])
        chapter = dict(self.model.visible_chapters())[summary]
        self.assertEqual((chapter["book"], chapter["chapter"], chapter["state"]), ("alpha", None, "done"))
        self.assertEqual(chapter["completed"], [1, 2])
        self.assertEqual(chapter["completed_reused"], 0)

    def test_all_done_chapters_in_a_book_share_one_summary_across_barriers(self):
        planned(self.board)
        done(self.board, "alpha", 1, 2)
        self.board.chapter_failure({"number": 3, "error_type": "SpeechVerificationFailed", "error": "bad"},
                                   book="alpha")
        done(self.board, "alpha", 4)
        self.board.chapter_state("/q/alpha", 5, "saved")
        done(self.board, "alpha", 6)
        self.assertEqual(self.keys()[:3], ["alpha\x1fdone\x1f1-6", "alpha\x1f3", "alpha\x1f5"])
        rows = dict(self.model.visible_chapters())
        self.assertEqual(rows["alpha\x1fdone\x1f1-6"]["completed"], [1, 2, 4, 6])
        self.assertEqual(rows["alpha\x1f5"]["state"], "saved")

    def test_adjacent_books_all_done_stay_one_summary_per_book(self):
        planned(self.board, alpha=(1, 2, 3), beta=(1, 2, 3))
        done(self.board, "alpha", 1, 2, 3)
        done(self.board, "beta", 1, 2, 3)
        self.assertEqual(self.keys(), ["alpha\x1fdone\x1f1-3", "beta\x1fdone\x1f1-3"])
        # Every all-complete visible book still has a row.
        self.model.filter.book = "beta"
        self.assertEqual(self.keys(), ["beta\x1fdone\x1f1-3"])

    def test_reused_done_chapters_count_and_saved_chapters_never_group(self):
        planned(self.board, alpha=(1, 2, 3), beta=())
        done(self.board, "alpha", 1, 2, reused=(1,))
        done(self.board, "alpha", 3)
        self.board.chapter_state("/q/alpha", 3, "saved")  # unverified: leaves the group, stays individual
        self.assertEqual(self.keys(), ["alpha\x1fdone\x1f1-2", "alpha\x1f3"])
        chapter = dict(self.model.visible_chapters())["alpha\x1fdone\x1f1-2"]
        self.assertEqual((chapter["completed"], chapter["completed_reused"]), ([1, 2], 1))

    def test_breakdown_counts_and_states_ignore_the_collapse_flag(self):
        planned(self.board)
        done(self.board, "alpha", 1, 2)
        self.board.chapter_failure({"number": 3, "error_type": "SpeechVerificationFailed", "error": "bad"},
                                   book="alpha", retry=lambda: True)
        collapsed = (self.model.chapter_breakdown(), self.model.chapter_states())
        self.model.collapse_completed = False
        expanded = (self.model.chapter_breakdown(), self.model.chapter_states())
        self.assertEqual(collapsed, expanded)
        self.assertEqual(self.keys(), [f"alpha\x1f{n}" for n in (1, 2, 3, 4, 5, 6)]
                         + [f"beta\x1f{n}" for n in (1, 2, 3)])
        self.model.collapse_completed = True
        self.assertEqual(self.model.chapter_breakdown(), collapsed[0])

    def test_state_and_book_filters_apply_before_grouping(self):
        planned(self.board)
        done(self.board, "alpha", 1, 2)
        done(self.board, "beta", 1, 2, 3)
        self.model.filter.state = "done"
        self.assertEqual(self.keys(), ["alpha\x1fdone\x1f1-2", "beta\x1fdone\x1f1-3"])
        self.model.filter.state, self.model.filter.book = None, "alpha"
        self.assertEqual(self.keys(), ["alpha\x1fdone\x1f1-2", "alpha\x1f3", "alpha\x1f4", "alpha\x1f5",
                                       "alpha\x1f6"])
        self.model.filter.book, self.model.filter.chapter = None, 2
        self.assertEqual(self.keys(), ["alpha\x1f2", "beta\x1f2"])  # the chapter filter is book-agnostic

    def test_a_newly_done_chapter_joins_the_summary_and_groups_track_live_changes(self):
        planned(self.board)
        done(self.board, "alpha", 1)
        self.board.chapter_state("/q/alpha", 2, "rendering")
        self.assertEqual(self.keys()[0], "alpha\x1f1")
        done(self.board, "alpha", 2)
        self.assertEqual(self.keys()[0], "alpha\x1fdone\x1f1-2")  # the run grew; one summary again
        self.board.chapter_state("/q/alpha", 2, "failed")
        self.assertEqual(self.keys()[:2], ["alpha\x1f1", "alpha\x1f2"])  # dissolved: both individual

    def test_summary_keys_never_match_numeric_or_live_row_keys(self):
        planned(self.board)
        done(self.board, "alpha", 1, 2)
        self.board.set_status("live-1", 0, {"chapter": 9, "total_chapters": 9, "done": 0, "blocks": 2,
                                            "elapsed": 1, "active": []}, book="beta")
        raw = {key for key, _ in self.model._chapters()}
        summaries = [key for key in self.keys() if "\x1fdone\x1f" in key]
        self.assertTrue(summaries)
        for key in summaries:
            self.assertNotIn(key, raw)

    def test_gapped_chapter_numbers_keep_the_summary_label_compact(self):
        from epub_tts_narrator.run_shell import _chapter_cells
        numbers = tuple(range(2, 60, 2))
        planned(self.board, alpha=numbers, beta=())
        done(self.board, "alpha", *numbers)

        [(key, summary)] = self.model.visible_chapters()
        cells = _chapter_cells(summary)

        self.assertEqual(summary["completed"], list(numbers))
        self.assertEqual((cells[1], str(cells[3])), ("2, …, 58", "29 completed chapters"))


class CompletedCollapseScreenTests(unittest.IsolatedAsyncioTestCase):
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

    def mixed_command(self, release, extra=None):
        def command():
            with live_board(print) as notify:
                report_book_state(notify, "alpha", "rendering")
                report_chapter_plan(notify, "alpha", [{"number": n, "title": f"C{n}", "blocks": 2, "saved": False}
                                                      for n in (1, 2, 3, 4)])
                report_book_state(notify, "beta", "rendering")
                report_chapter_plan(notify, "beta", [{"number": n, "title": f"C{n}", "blocks": 2, "saved": False}
                                                     for n in (1, 2, 3)])
                report_chapter_state(notify, "alpha", 1, "done", reused=True)
                report_chapter_failure(Prefixed(notify, "alpha"),
                                       {"number": 3, "error_type": "SpeechVerificationFailed",
                                        "error": "take 3 skipped words"}, retry=self.retry_calls)
                report_chapter_state(notify, "beta", 1, "done")
                report_chapter_state(notify, "beta", 2, "done", reused=True)
                report_chapter_state(notify, "beta", 3, "done")
                if extra is not None:
                    extra(notify)
                release.wait(10)
            return 0
        return command

    def retry_calls(self):
        self.retried.append(1)
        return True

    async def open_chapters(self, app, pilot):
        await pilot.press("2")
        app.refresh_view(force=True)
        await pilot.pause()

    async def test_e_collapses_and_expands_all_completed_groups_and_only_in_the_chapters_view(self):
        release = threading.Event()
        app = self.make_app(self.mixed_command(release))
        async with app.run_test(size=(120, 32)) as pilot:
            await self.wait_for(pilot, lambda: [key for key, _ in self.model.visible_chapters()]
                                 == ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4",
                                     "beta\x1fdone\x1f1-3"])
            await self.open_chapters(app, pilot)
            table = app.query_one("#rows")
            self.assertEqual([key.value for key in table.rows],
                             ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4", "beta\x1fdone\x1f1-3"])
            cells = table.get_row("beta\x1fdone\x1f1-3")
            self.assertEqual((str(cells[1]), str(cells[3])), ("1-3", "3 completed chapters"))
            self.assertEqual(str(table.get_row("alpha\x1f1")[3]), "done · earlier run")
            # E in another view changes nothing.
            await pilot.press("1")
            await pilot.press("E")
            self.assertTrue(self.model.collapse_completed)
            await self.open_chapters(app, pilot)
            await pilot.press("E")
            self.assertFalse(self.model.collapse_completed)
            await pilot.pause()
            self.assertEqual([key.value for key in table.rows],
                             [f"alpha\x1f{n}" for n in (1, 2, 3, 4)] + [f"beta\x1f{n}" for n in (1, 2, 3)])
            await pilot.press("E")
            await pilot.pause()
            self.assertEqual([key.value for key in table.rows],
                             ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4", "beta\x1fdone\x1f1-3"])
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_enter_on_a_summary_expands_it_and_highlights_its_first_chapter(self):
        release = threading.Event()
        app = self.make_app(self.mixed_command(release))
        async with app.run_test(size=(120, 32)) as pilot:
            await self.wait_for(pilot, lambda: [key for key, _ in self.model.visible_chapters()]
                                 == ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4",
                                     "beta\x1fdone\x1f1-3"])
            await self.open_chapters(app, pilot)
            table = app.query_one("#rows")
            table.focus()
            table.move_cursor(row=4)  # the beta summary
            await pilot.press("enter")
            self.assertFalse(self.model.collapse_completed)
            await pilot.pause()
            from epub_tts_narrator.tui_widgets import highlighted_key
            self.assertEqual(highlighted_key(table), "beta\x1f1")
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_the_cursor_follows_its_chapter_into_the_summary_and_back(self):
        release = threading.Event()
        app = self.make_app(self.mixed_command(release))
        async with app.run_test(size=(120, 32)) as pilot:
            await self.wait_for(pilot, lambda: [key for key, _ in self.model.visible_chapters()]
                                 == ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4",
                                     "beta\x1fdone\x1f1-3"])
            await self.open_chapters(app, pilot)
            table = app.query_one("#rows")
            table.focus()
            await pilot.press("E")  # expand
            await pilot.pause()
            table.move_cursor(row=5)  # beta chapter 2, inside the beta group
            await pilot.press("E")  # collapse: the highlight lands on the beta summary
            await pilot.pause()
            from epub_tts_narrator.tui_widgets import highlighted_key
            self.assertEqual(highlighted_key(table), "beta\x1fdone\x1f1-3")
            await pilot.press("E")  # expand again: the summary yields its first chapter
            await pilot.pause()
            self.assertEqual(highlighted_key(table), "beta\x1f1")
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_live_completion_of_the_highlighted_chapter_selects_the_summary(self):
        release, rendered = threading.Event(), threading.Event()

        def extra(notify):
            rendered.wait(10)
            report_chapter_state(notify, "alpha", 2, "done")

        app = self.make_app(self.mixed_command(release, extra=extra))
        async with app.run_test(size=(120, 32)) as pilot:
            await self.wait_for(pilot, lambda: [key for key, _ in self.model.visible_chapters()]
                                 == ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4",
                                     "beta\x1fdone\x1f1-3"])
            await self.open_chapters(app, pilot)
            table = app.query_one("#rows")
            table.focus()
            table.move_cursor(row=1)  # alpha chapter 2, queued
            rendered.set()
            await self.wait_for(pilot, lambda: self.model.visible_chapters()[0][0] == "alpha\x1fdone\x1f1-2")
            app.refresh_view(force=True)
            await pilot.pause()
            from epub_tts_narrator.tui_widgets import highlighted_key
            self.assertEqual(highlighted_key(table), "alpha\x1fdone\x1f1-2")
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_selected_summary_stays_selected_when_a_distant_chapter_completes(self):
        release, merged = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.addCleanup(merged.set)

        def command():
            with live_board(print) as notify:
                report_book_state(notify, "alpha", "rendering")
                report_chapter_plan(notify, "alpha", [{"number": n, "title": f"C{n}", "blocks": 2, "saved": False}
                                                      for n in range(1, 7)])
                for number in (1, 2, 4, 5):
                    report_chapter_state(notify, "alpha", number, "done")
                report_chapter_failure(Prefixed(notify, "alpha "),
                                       {"number": 3, "error_type": "SpeechVerificationFailed", "error": "failed"})
                report_chapter_state(notify, "alpha", 6, "saved")
                report_book_state(notify, "beta", "rendering")
                report_chapter_plan(notify, "beta", [{"number": n, "title": f"C{n}", "blocks": 2, "saved": False}
                                                      for n in (1, 2, 3)])
                for number in (1, 2, 3):
                    report_chapter_state(notify, "beta", number, "done")
                merged.wait(10)
                report_chapter_state(notify, "alpha", 6, "done")
                release.wait(10)
            return 0

        app = self.make_app(command)
        async with app.run_test(size=(120, 32)) as pilot:
            await self.wait_for(pilot, lambda: [key for key, _ in self.model.visible_chapters()] == [
                "alpha\x1fdone\x1f1-5", "alpha\x1f3", "alpha\x1f6", "beta\x1fdone\x1f1-3"])
            await self.open_chapters(app, pilot)
            table = app.query_one("#rows")
            table.focus()
            table.move_cursor(row=0)

            merged.set()
            await self.wait_for(pilot, lambda: self.model.visible_chapters()[0][0] == "alpha\x1fdone\x1f1-6")
            app.refresh_view(force=True)
            await pilot.pause()

            from epub_tts_narrator.tui_widgets import highlighted_key
            self.assertEqual(highlighted_key(table), "alpha\x1fdone\x1f1-6")
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)

    async def test_retry_takes_and_detail_on_a_summary_are_safe_and_failed_rows_still_retry(self):
        release = threading.Event()
        self.retried = []
        app = self.make_app(self.mixed_command(release))
        notices = []
        async with app.run_test(size=(120, 32)) as pilot:
            await self.wait_for(pilot, lambda: [key for key, _ in self.model.visible_chapters()]
                                 == ["alpha\x1f1", "alpha\x1f2", "alpha\x1f3", "alpha\x1f4",
                                     "beta\x1fdone\x1f1-3"] and len(self.model.current_failures()) == 1)
            await self.open_chapters(app, pilot)
            app.notify = lambda message, **kwargs: notices.append(message)
            table = app.query_one("#rows")
            table.focus()
            table.move_cursor(row=4)  # the beta summary
            await pilot.press("r")
            self.assertIn("summarizes completed chapters", notices[-1])
            await pilot.press("t")
            self.assertIn("summarizes completed chapters", notices[-1])
            self.assertEqual(self.retried, [])  # no member was retried, no lookup for chapter None
            await pilot.press("d")
            from epub_tts_narrator.tui_widgets import DetailScreen
            self.assertIsInstance(app.screen, DetailScreen)
            self.assertIn("3 completed chapters of beta", str(app.screen.query_one("#detail-text").render()))
            await pilot.press("escape")
            table.move_cursor(row=2)  # the real failed row
            await pilot.press("r")
            self.assertEqual(self.retried, [1])
            release.set()
            await self.wait_for(pilot, lambda: not self.model.running)


def Prefixed(notify, book):
    from epub_tts_narrator.renderer.progress import PrefixedBoardNotify
    return PrefixedBoardNotify(notify.__self__, f"[{book}] ")
