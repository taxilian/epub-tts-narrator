"""Live model-stream status is TTY-only, bounded, and non-persistent."""

import io
import os
import unittest
from unittest.mock import patch

from epub_tts_narrator.stream_progress import StreamConsole, activate, step_progress, time_left


class TTY(io.StringIO):
    def isatty(self):
        return True


class StreamConsoleTests(unittest.TestCase):
    def test_status_uses_the_attached_terminal_width(self):
        class SizedTTY(TTY):
            def fileno(self):
                return 1

        output = SizedTTY()
        console = StreamConsole(output, width=200, draw_interval=0)
        with patch("epub_tts_narrator.stream_progress.os.get_terminal_size",
                   return_value=os.terminal_size((48, 24))):
            console.set_progress(0.75, "Attribution 80/100 complete")
            console.observe({"request_id": "req-1", "phase": "started"})
            console.observe({
                "request_id": "req-1", "phase": "streaming", "kind": "reasoning",
                "text": "x" * 300, "received": 1, "events": 1,
                "reasoning_chars": 300, "content_chars": 0,
            })
        for line in output.getvalue().splitlines():
            self.assertLessEqual(len(line.replace("\x1b[2A\x1b[J", "")), 47)
        console.observe({"request_id": "req-1", "phase": "failed"})

    def test_two_line_status_shows_reasoning_and_aggregate_stats(self):
        output = TTY()
        console = StreamConsole(output, width=200, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.observe({
            "request_id": "req-1", "phase": "streaming", "kind": "reasoning",
            "text": "checking\nthe scene", "received": 4096, "events": 3,
            "reasoning_chars": 17, "content_chars": 12,
        })
        text = output.getvalue()
        self.assertIn("Thinking: checking the scene", text)
        self.assertIn("LLM 1 active", text)
        self.assertIn("3 events", text)
        self.assertIn("thought 17 chars", text)
        self.assertIn("answer 12 chars", text)
        console.observe({"request_id": "req-1", "phase": "completed"})
        self.assertNotIn("req-1", console._active)

    def test_progress_messages_precede_redrawn_status_without_overwriting_it(self):
        output = TTY()
        console = StreamConsole(output, width=200, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.progress("Discovering cast candidates")
        text = output.getvalue()
        self.assertIn("Discovering cast candidates\n", text)
        self.assertGreaterEqual(text.count("LLM 1 active"), 2)
        console.observe({"request_id": "req-1", "phase": "failed"})

    def test_status_includes_step_and_step_local_analysis_progress(self):
        output = TTY()
        console = StreamConsole(output, width=240, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.set_progress(0.68, "Attribution 37/100")
        console.set_progress(0.20, "Stale milestone")
        text = output.getvalue()
        self.assertIn("step 2/4 Attribution 45% · Attribution 37/100", text[-600:])
        self.assertNotIn("Casting 20%", text[-600:])
        console.observe({"request_id": "req-1", "phase": "failed"})

    def test_the_saved_book_s_checks_show_which_check_of_how_many(self):
        output = TTY()
        console = StreamConsole(output, width=240, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.set_progress(1.0, "Validation")
        console.set_check(3, 8, "each character's gender")
        self.assertIn("step 3/8 Checking the book 0% · each character's gender", output.getvalue()[-600:])
        previous = activate(console)
        try:
            step_progress(2, 4, "jury votes")
        finally:
            activate(previous)
        self.assertIn("step 3/8 Checking the book 50% · each character's gender · 2/4 jury votes",
                      output.getvalue()[-600:])
        console.end_checks()
        console.set_progress(1.0, "Validation")
        self.assertIn("step 4/4", output.getvalue()[-600:])
        console.observe({"request_id": "req-1", "phase": "failed"})

    def test_time_left_is_estimated_from_the_step_s_own_pace(self):
        self.assertIsNone(time_left(10, 0.5), "too early to tell")
        self.assertIsNone(time_left(60, 0.01), "too little done to tell")
        self.assertEqual(time_left(60, 0.25), 180)
        output = TTY()
        clock = [100.0]
        with patch("epub_tts_narrator.stream_progress.monotonic", side_effect=lambda: clock[0]):
            console = StreamConsole(output, width=240, draw_interval=0)
            console.observe({"request_id": "req-1", "phase": "started"})
            console.set_check(6, 8, "every paragraph's speakers")
            console.set_step_progress(0, 40, "scan batches")
            clock[0] += 120
            console.set_step_progress(10, 40, "scan batches")
        self.assertIn("step 6/8 Checking the book 25% (~6:00 left in this step)", output.getvalue()[-600:])

    def test_ordinary_log_writes_scroll_before_redrawn_status(self):
        output = TTY()
        console = StreamConsole(output, width=200, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.write("ordinary log line\n")
        text = output.getvalue()
        self.assertIn("ordinary log line\n", text)
        self.assertGreater(text.rfind("LLM 1 active"), text.index("ordinary log line"))
        console.observe({"request_id": "req-1", "phase": "failed"})

    def test_non_tty_output_stays_plain(self):
        output = io.StringIO()
        console = StreamConsole(output)
        console.observe({
            "request_id": "req-1", "phase": "streaming", "kind": "reasoning",
            "text": "hidden", "received": 1, "events": 1,
            "reasoning_chars": 6, "content_chars": 0,
        })
        console.progress("plain progress")
        self.assertEqual(output.getvalue(), "plain progress\n")

    def test_retry_and_waiting_states_are_visible(self):
        output = TTY()
        console = StreamConsole(output, width=200, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.observe({"request_id": "req-1", "phase": "retry", "attempt": 2, "delay": 6.5})
        text = output.getvalue()
        self.assertIn("Retrying provider request 2 in 6.5s", text)
        console.observe({"request_id": "req-1", "phase": "failed"})


if __name__ == "__main__":
    unittest.main()
