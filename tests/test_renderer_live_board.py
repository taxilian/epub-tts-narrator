"""Live terminal board: one persistent row per active chapter, logs scroll above."""

import io
import re
import types
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.progress import (ChapterProgress, PrefixedBoardNotify,
                                                 TerminalBoard, live_board,
                                                 report_book_outcome)


class TTY(io.StringIO):
    def isatty(self):
        return True


def chunks(count):
    return [{"spoken_text": "Some words here.", "silent_control": False} for _ in range(count)]


def screen(raw):
    """Replay the board's escape sequences into the visible terminal lines."""
    lines, row = [""], 0
    index = 0
    while index < len(raw):
        if raw.startswith("\r\033[2K", index):
            lines[row] = ""
            index += 5
        elif raw.startswith("\033[1A\033[2K", index):
            row -= 1
            lines[row] = ""
            index += 8
        elif raw[index] == "\n":
            row += 1
            if row == len(lines):
                lines.append("")
            index += 1
        else:
            lines[row] += raw[index]
            index += 1
    del lines[row + 1:]
    return lines


class TerminalBoardTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value="sched gen 1q/1a")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stream = TTY()
        self.board = TerminalBoard(self.stream, width=60)

    def test_rows_stay_at_the_bottom_while_logs_scroll_above(self):
        self.board.set_row("b", (40, 2), "Ch 40/92 | gen")
        self.board.set_row("a", (39, 1), "Ch 39/92 | verify")
        self.board.log("Processing chapter 41 / 92: Chapter 40 (109 blocks)")
        self.board.set_row("a", (39, 1), "Ch 39/92 | done")
        self.board.log("Ch 39: Chunk 2 rejected", "text_mismatch")
        self.assertEqual(screen(self.stream.getvalue()), [
            "Processing chapter 41 / 92: Chapter 40 (109 blocks)",
            "Ch 39: Chunk 2 rejected text_mismatch",
            "Ch 39/92 | done", "Ch 40/92 | gen", "sched gen 1q/1a"])

    def test_rows_are_truncated_to_the_terminal_width_and_removed_rows_disappear(self):
        self.board.set_row("a", (1, 1), "x" * 200)
        self.board.set_row("b", (2, 2), "second")
        self.assertEqual(screen(self.stream.getvalue())[0], "x" * 58 + "…")
        self.board.remove_row("a")
        self.board.remove_row("b")
        self.assertEqual(screen(self.stream.getvalue()), [""])
        self.board.log("after")
        self.assertEqual(screen(self.stream.getvalue()), ["after", ""])

    def test_close_erases_rows_and_ignores_later_rows(self):
        self.board.set_row("a", (1, 1), "row")
        self.board.close()
        self.board.set_row("a", (1, 1), "late")
        self.assertEqual(screen(self.stream.getvalue()), [""])

    def test_live_board_only_replaces_print_on_an_interactive_terminal(self):
        with live_board(print, stream=TTY()) as notify:
            self.assertIsInstance(notify.__self__, TerminalBoard)
            with live_board(print, stream=TTY()) as nested:
                self.assertIs(nested, print)
        with live_board(print, stream=io.StringIO()) as notify:
            self.assertIs(notify, print)
        custom = []
        with live_board(custom.append, stream=TTY()) as notify:
            self.assertEqual(notify, custom.append)


class BookOutcomeLineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stream = TTY()
        self.board = TerminalBoard(self.stream, width=80)

    def visible(self):
        return [re.sub(r"\033\[[0-9;]*m", "", line) for line in screen(self.stream.getvalue())]

    def test_no_books_line_appears_before_any_book_finishes(self):
        self.board.set_row("a", (1, 1), "Ch 1/9 | gen")
        self.assertNotIn("Books:", " ".join(self.visible()))

    def test_finished_books_show_colored_counts_and_both_parts_together(self):
        self.board.book_outcome("book-1", True)
        self.board.book_outcome("book-2", False)
        self.assertEqual(self.visible()[-1], "Books: 1 succeeded | 1 failed")
        self.assertIn("Books: \033[32m1\033[0m succeeded | \033[31m1\033[0m failed",
                      self.stream.getvalue())

    def test_counts_survive_completed_chapter_rows_disappearing(self):
        self.board.set_row("a", (1, 1), "Ch 1/9 | gen")
        self.board.book_outcome("book-1", True)
        self.board.remove_row("a")
        self.assertEqual(self.visible(), ["Books: 1 succeeded"])

    def test_only_success_shows_a_green_count_without_a_failed_part(self):
        self.board.book_outcome("book-1", True)
        self.board.book_outcome("book-2", True)
        self.assertEqual(self.visible()[-1], "Books: 2 succeeded")
        self.assertNotIn("\033[31m", self.stream.getvalue())

    def test_only_failure_shows_a_red_count_without_a_succeeded_part(self):
        self.board.book_outcome("book-1", False)
        self.assertEqual(self.visible()[-1], "Books: 1 failed")
        self.assertNotIn("\033[32m", self.stream.getvalue())

    def test_rereporting_the_same_book_never_double_counts(self):
        self.board.book_outcome("book-1", True)
        self.board.book_outcome("book-1", True)
        self.assertEqual(self.visible()[-1], "Books: 1 succeeded")

    def test_a_narrow_width_switches_to_a_compact_tally_keeping_both_counts(self):
        board = TerminalBoard(self.stream, width=24)
        board.book_outcome("book-1", True)
        board.book_outcome("book-2", False)
        self.assertEqual(self.visible()[-1], "OK 1 | FAIL 1")
        self.assertIn("OK \033[32m1\033[0m | FAIL \033[31m1\033[0m", self.stream.getvalue())

    def test_a_minimum_width_keeps_both_counts_in_the_tightest_tally(self):
        board = TerminalBoard(self.stream, width=12)
        board.book_outcome("book-1", True)
        board.book_outcome("book-2", False)
        self.assertEqual(self.visible()[-1], "OK1 FAIL1")
        self.assertIn("OK\033[32m1\033[0m FAIL\033[31m1\033[0m", self.stream.getvalue())

    def test_log_lines_around_the_books_line_stay_plain(self):
        self.board.book_outcome("book-1", True)
        self.board.log("an ordinary log line")
        self.assertEqual(self.visible()[-2:], ["an ordinary log line", "Books: 1 succeeded"])
        self.assertIn("an ordinary log line\nBooks: \033[32m1\033[0m succeeded",
                      self.stream.getvalue())

    def test_outcomes_after_close_write_nothing(self):
        self.board.close()
        before = self.stream.getvalue()
        self.board.book_outcome("book-1", True)
        self.assertEqual(self.stream.getvalue(), before)

    def test_report_book_outcome_without_a_board_is_a_noop(self):
        messages = []
        report_book_outcome(messages.append, "book-1", True)
        report_book_outcome(print, "book-1", True)
        self.assertEqual(messages, [])

    def test_report_book_outcome_reaches_the_board_through_the_queue_prefix(self):
        prefixed = PrefixedBoardNotify(self.board, "[book-1] ")
        report_book_outcome(prefixed, "book-1", True)
        self.assertEqual(self.visible()[-1], "Books: 1 succeeded")


class ChapterBoardRowTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value="")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stream = TTY()
        self.board = TerminalBoard(self.stream, width=200)
        self.now = [100.0]

    def progress(self, number, count=3):
        return ChapterProgress(number, 92, chunks(count), notify=self.board.log,
                               clock=lambda: self.now[0], heartbeat=False)

    def test_each_active_chapter_keeps_one_row_and_messages_are_tagged(self):
        with self.progress(40) as later, self.progress(39) as first:
            with first.task(0) as notify:
                notify.worker_activity("assigned", "second-mac")
                notify.worker_activity("started", "second-mac")
                notify.phase("generating paragraph, attempt 2/6")
                self.now[0] += 37
                first.refresh()
                later.refresh()
                rows = screen(self.stream.getvalue())[-2:]
                self.assertEqual(rows[0], "Ch 39/92 | 0/3 blocks (0%) | 00:37 | #1 second-mac gen 2/6 00:37")
                self.assertEqual(rows[1], "Ch 40/92 | 0/3 blocks (0%) | 00:37 | idle")
                notify.worker_activity("completed", "second-mac", 37)
                notify("  Attempt 1/6 rejected: text_mismatch")
            later.refresh()
        lines = screen(self.stream.getvalue())
        self.assertIn("Ch 39: Chunk 1 (audio from second-mac): Attempt 1/6 rejected: text_mismatch", lines)
        self.assertFalse(any("done in" in line for line in lines), lines)
        self.assertEqual(lines[-3:], ["Ch 39: speech stopped: 1/3 blocks in 00:37",
                                      "Ch 40: speech stopped: 0/3 blocks in 00:37", ""])

    def test_generation_on_a_worker_is_named_in_the_row(self):
        with self.progress(7) as chapter:
            with chapter.task(1) as notify:
                notify.worker_activity("started", "local")
                chapter.refresh()
                self.assertIn("#2 local", screen(self.stream.getvalue())[-1])

    def test_queue_prefixes_keep_distinct_pinned_rows_for_overlapping_books(self):
        from epub_tts_narrator.renderer.render_queue import _prefixed
        first = _prefixed("[book-1] ", self.board.log)
        second = _prefixed("[book-2] ", self.board.log)
        with (ChapterProgress(1, 9, chunks(1), notify=first, heartbeat=False) as one,
              ChapterProgress(1, 8, chunks(1), notify=second, heartbeat=False) as two):
            with one.task(0) as task:
                task.worker_activity("started", "worker-a")
                one.refresh()
                two.refresh()
                visible = screen(self.stream.getvalue())
                self.assertTrue(visible[-2].startswith("[book-1] Ch 1/9 |"), visible)
                self.assertIn("worker-a", visible[-2])
                self.assertTrue(visible[-1].startswith("[book-2] Ch 1/8 |"), visible)
                first("an ordinary log message")
                visible = screen(self.stream.getvalue())
                self.assertTrue(visible[-2].startswith("[book-1] Ch 1/9 |"), visible)
                self.assertTrue(visible[-1].startswith("[book-2] Ch 1/8 |"), visible)

    def test_redirected_output_keeps_plain_status_lines(self):
        stream = io.StringIO()
        with ChapterProgress(3, 9, chunks(1), stream=stream, heartbeat=False, clock=lambda: 0.0) as chapter:
            with chapter.task(0):
                pass
        self.assertNotIn("\033[", stream.getvalue())
        self.assertIn("Chapter 3 / 9", stream.getvalue())


class SchedulingDefaultsTests(unittest.TestCase):
    def test_automatic_chapter_concurrency_is_one_per_slot_from_four_to_sixteen(self):
        from epub_tts_narrator.renderer.render_pipeline_config import effective_scheduling
        def chapters(slots):
            workers = [{"name": f"w{index}", "type": "fish_http", "concurrency": 1} for index in range(slots)]
            return effective_scheduling({"render_pool": {"workers": workers}})["chapter_concurrency"]
        self.assertEqual([chapters(count) for count in (0, 1, 2, 3, 5, 12, 20)], [4, 4, 4, 4, 5, 12, 16])
        explicit = effective_scheduling({"render_pool": {"workers": [], "chapter_concurrency": 1}})
        self.assertEqual(explicit["chapter_concurrency"], 1)


class FirstAvailableSlotTests(unittest.TestCase):
    def test_slots_do_not_balance_or_remember_chapters(self):
        from epub_tts_narrator.renderer.pool import _Slots
        local = types.SimpleNamespace(settings={"name": "local", "concurrency": 1})
        remote = types.SimpleNamespace(settings={"name": "second-mac", "concurrency": 1})
        slots = _Slots([local, remote])
        self.assertIs(slots.try_acquire(), local)
        self.assertIs(slots.try_acquire(), remote)
        self.assertIsNone(slots.try_acquire())
        slots.release(local)
        self.assertIs(slots.try_acquire(), local)
        slots.release(remote)
        slots.release(local)
        self.assertEqual(slots.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
