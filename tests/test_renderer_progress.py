"""Progress reports finished work, including cache reuse and interruption."""

from copy import deepcopy
import io
import os
import threading
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.progress import ChapterProgress, format_duration, set_phase


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Terminal(io.StringIO):
    def isatty(self):
        return True


def blocks(count, *, length=100):
    return [{"spoken_text": "x" * length, "paragraph_turns": []} for _ in range(count)]


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.messages = []

    def reporter(self, chunks, **kwargs):
        return ChapterProgress(6, 35, chunks, notify=self.messages.append,
                               clock=self.clock, **kwargs)

    def generate(self, progress, index, seconds=10):
        with progress.task(index) as notify:
            set_phase(notify, "generating speech, attempt 1/3", generated=True)
            self.clock.advance(seconds)

    def test_completed_count_uses_finishes_not_source_index(self):
        with self.reporter(blocks(3)) as progress:
            with progress.task(0):
                with progress.task(2):
                    self.clock.advance(5)
                self.assertIn("Blocks 1 / 3 complete", self.messages[-1])
                self.assertIn("#1 preparing paragraph 00:05", self.messages[-1])
            with progress.task(1):
                pass
        self.assertIn("Chapter 6 / 35", self.messages[-1])
        self.assertIn("Blocks 3 / 3 complete", self.messages[-1])
        self.assertIn("speech complete", self.messages[-1])

    def test_interrupt_does_not_claim_failed_block_finished(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.reporter(blocks(2)) as progress:
                self.generate(progress, 0)
                with progress.task(1) as notify:
                    set_phase(notify, "verifying speech, attempt 1/3")
                    raise KeyboardInterrupt
        self.assertIn("Blocks 1 / 2 complete", self.messages[-1])
        self.assertIn("speech stopped", self.messages[-1])
        self.assertNotIn("speech complete", self.messages[-1])

    def test_phase_elapsed_updates_without_claiming_generation_percentage(self):
        with self.reporter(blocks(1)) as progress:
            with progress.task(0) as notify:
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                self.clock.advance(65)
                progress.refresh()
                self.assertIn("Blocks 0 / 1 complete", self.messages[-1])
                self.assertIn("generating speech, attempt 1/3 01:05", self.messages[-1])
                self.assertNotIn("%", self.messages[-1])
                set_phase(notify, "verifying speech, attempt 1/3")
                self.clock.advance(2)
                progress.refresh()
                self.assertIn("elapsed 01:07", self.messages[-1])
                self.assertIn("verifying speech, attempt 1/3 00:02", self.messages[-1])

    def test_cache_hits_never_accelerate_or_slow_generation_estimate(self):
        with self.reporter(blocks(7)) as progress:
            with progress.task(0) as notify:
                set_phase(notify, "checking saved audio")
                self.clock.advance(500)
            self.generate(progress, 1)
            self.generate(progress, 2)
            self.assertNotIn("approx speech left", self.messages[-1])
            self.generate(progress, 3)
            self.assertIn("approx speech left 00:30", self.messages[-1])
            with progress.task(4):
                pass
            self.assertIn("approx speech left 00:20", self.messages[-1])

    def test_estimate_includes_retries_and_verification_but_not_prior_cache_check(self):
        with self.reporter(blocks(4)) as progress:
            self.generate(progress, 0)
            self.generate(progress, 1)
            with progress.task(2) as notify:
                self.clock.advance(500)
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                self.clock.advance(10)
                set_phase(notify, "generating speech, attempt 2/3", generated=True)
                self.clock.advance(10)
                set_phase(notify, "verifying speech, attempt 2/3")
                self.clock.advance(10)
            self.assertIn("approx speech left 00:17", self.messages[-1])

    def test_estimate_weights_text_and_accounts_for_parallelism_and_large_tail(self):
        chunks = blocks(6)
        chunks[3]["spoken_text"] *= 3
        with self.reporter(chunks, parallel=2) as progress:
            for index in range(3):
                self.generate(progress, index)
            # 50 seconds of serial work, but a single 30-second block cannot
            # finish in 25 seconds just because two workers are available.
            self.assertIn("approx speech left 00:30", self.messages[-1])
            with progress.task(3) as notify:
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                self.clock.advance(5)
                progress.refresh()
                self.assertIn("approx speech left 00:25", self.messages[-1])

    def test_overdue_active_block_does_not_show_zero_remaining(self):
        with self.reporter(blocks(4)) as progress:
            for index in range(3):
                self.generate(progress, index)
            with progress.task(3) as notify:
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                self.clock.advance(15)
                progress.refresh()
                self.assertNotIn("approx speech left", self.messages[-1])
                self.assertIn("generating speech, attempt 1/3 00:15", self.messages[-1])

    def test_silent_controls_are_counted_but_do_not_add_speech_time(self):
        chunks = blocks(5)
        chunks[4]["silent_control"] = True
        original = deepcopy(chunks)
        with self.reporter(chunks) as progress:
            for index in range(3):
                self.generate(progress, index)
            self.assertIn("approx speech left 00:10", self.messages[-1])
            with progress.task(4):
                pass
            self.assertIn("Blocks 4 / 5 complete", self.messages[-1])
            self.generate(progress, 3)
        self.assertEqual(original, chunks)

    def test_terminal_clears_live_line_before_logs_and_finishes_on_clean_line(self):
        terminal = Terminal()
        with ChapterProgress(6, 35, blocks(1), stream=terminal,
                             clock=self.clock, heartbeat=False) as progress:
            with progress.task(0) as notify:
                notify("Attempt rejected; retrying")
        output = terminal.getvalue()
        self.assertIn("\r\033[2KAttempt rejected; retrying\n", output)
        self.assertTrue(output.endswith("speech complete\n"))

    def test_narrow_terminal_retains_eta_and_current_generation_phase(self):
        terminal = Terminal()
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with ChapterProgress(6, 35, blocks(5), stream=terminal,
                                 clock=self.clock, heartbeat=False) as progress:
                for index in range(3):
                    self.generate(progress, index)
                with progress.task(3) as notify:
                    set_phase(notify, "generating speech, attempt 1/3", generated=True)
                    self.clock.advance(1)
                    progress.refresh()
                    current = terminal.getvalue().rsplit("\033[2K", 1)[-1]
                    self.assertLessEqual(len(current), 79)
                    self.assertIn("~00:19 left", current)
                    self.assertIn("#4 gen 1/3", current)
                    self.assertIn("00:01", current)

    def test_redirected_output_is_plain_and_periodic(self):
        output = io.StringIO()
        with ChapterProgress(6, 35, blocks(3), stream=output,
                             clock=self.clock, heartbeat=False) as progress:
            self.generate(progress, 0, seconds=1)
            self.generate(progress, 1, seconds=30)
            self.generate(progress, 2, seconds=1)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("Blocks 0 / 3 complete", lines[0])
        self.assertIn("Blocks 2 / 3 complete", lines[1])
        self.assertIn("Blocks 3 / 3 complete", lines[2])
        self.assertNotIn("\033", output.getvalue())
        self.assertNotIn("\r", output.getvalue())

    def test_background_heartbeat_updates_blocking_generation_and_stops_on_exit(self):
        updated = threading.Event()

        class ObservedTerminal(Terminal):
            def write(self, text):
                result = super().write(text)
                if "00:05" in text:
                    updated.set()
                return result

        terminal = ObservedTerminal()
        existing_threads = set(threading.enumerate())
        with ChapterProgress(6, 35, blocks(1), stream=terminal, clock=self.clock,
                             refresh_interval=0.01) as progress:
            with progress.task(0) as notify:
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                self.clock.advance(5)
                self.assertTrue(updated.wait(2), "No progress while generation was blocked")
                new_threads = set(threading.enumerate()) - existing_threads
                self.assertTrue(new_threads)
        self.assertTrue(all(not thread.is_alive() for thread in new_threads))

    def test_closed_output_does_not_mask_synthesis_error(self):
        terminal = Terminal()
        with self.assertRaisesRegex(RuntimeError, "synthesis failed"):
            with ChapterProgress(6, 35, blocks(1), stream=terminal,
                                 heartbeat=False) as progress:
                with progress.task(0):
                    terminal.close()
                    raise RuntimeError("synthesis failed")

    def test_duration_handles_long_runs(self):
        self.assertEqual(format_duration(65.9), "01:05")
        self.assertEqual(format_duration(3723), "1:02:03")
        self.assertEqual(format_duration(-1), "00:00")


if __name__ == "__main__":
    unittest.main()
