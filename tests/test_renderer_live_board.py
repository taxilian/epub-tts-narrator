"""Live terminal board: one persistent row per active chapter, logs scroll above."""

import io
import threading
import types
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.progress import ChapterProgress, TerminalBoard, live_board


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

    def test_redirected_output_keeps_plain_status_lines(self):
        stream = io.StringIO()
        with ChapterProgress(3, 9, chunks(1), stream=stream, heartbeat=False, clock=lambda: 0.0) as chapter:
            with chapter.task(0):
                pass
        self.assertNotIn("\033[", stream.getvalue())
        self.assertIn("Chapter 3 / 9", stream.getvalue())


class SchedulingDefaultsTests(unittest.TestCase):
    def test_automatic_chapter_concurrency_is_at_least_two_per_slot(self):
        from epub_tts_narrator.renderer.render_pipeline_config import effective_scheduling
        def chapters(slots):
            workers = [{"name": f"w{index}", "type": "fish_http", "concurrency": 1} for index in range(slots)]
            return effective_scheduling({"render_pool": {"workers": workers}})["chapter_concurrency"]
        self.assertEqual([chapters(count) for count in (0, 1, 2, 3, 5, 12)], [4, 4, 4, 6, 10, 16])
        explicit = effective_scheduling({"render_pool": {"workers": [], "chapter_concurrency": 1}})
        self.assertEqual(explicit["chapter_concurrency"], 1)


class BalancedPinningTests(unittest.TestCase):
    def test_new_chapters_spread_across_workers_by_load_not_by_who_is_free(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity
        from epub_tts_narrator.renderer.pool import _Slots
        local = types.SimpleNamespace(settings={"name": "local", "concurrency": 1})
        remote = types.SimpleNamespace(settings={"name": "second-mac", "concurrency": 1})
        slots = _Slots([local, remote])
        chapters = [ChapterContinuity({"engine": "fixture"}, {}) for _ in range(4)]
        busy = []
        # The first take of chapter 1 occupies local; chapters 2-4 still
        # balance two per machine instead of all landing on the free one.
        with chapters[0].transaction({"kind": "speech", "voice_asset_hash": "a" * 64, "chunk": {"segment": {"role_id": "r", "voice_id": "v"}}}):
            busy.append(slots.acquire())
        for chapter in chapters[1:]:
            chapter_worker = []
            def take(chapter=chapter):
                with chapter.transaction({"kind": "speech", "voice_asset_hash": "a" * 64, "chunk": {"segment": {"role_id": "r", "voice_id": "v"}}}):
                    worker = slots.acquire()
                    chapter_worker.append(worker)
                    slots.release(worker)
            thread = threading.Thread(target=take)
            thread.start()
            thread.join(0.2)
            if thread.is_alive():
                slots.release(busy.pop())
                thread.join(2)
        names = [chapter.worker for chapter in chapters]
        self.assertEqual(names[:2], ["local", "second-mac"])
        self.assertEqual(sorted(names), ["local", "local", "second-mac", "second-mac"])
        del chapters[:], chapter, take
        import gc
        gc.collect()
        self.assertEqual([len(slots._chapters[id(worker)]) for worker in (local, remote)], [0, 0])


if __name__ == "__main__":
    unittest.main()
