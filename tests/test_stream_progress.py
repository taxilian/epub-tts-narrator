"""Live model-stream status is TTY-only, bounded, and non-persistent."""

import io
import unittest

from epub_tts_narrator.stream_progress import StreamConsole


class TTY(io.StringIO):
    def isatty(self):
        return True


class StreamConsoleTests(unittest.TestCase):
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

    def test_status_includes_monotonic_overall_analysis_progress(self):
        output = TTY()
        console = StreamConsole(output, width=220, draw_interval=0)
        console.observe({"request_id": "req-1", "phase": "started"})
        console.set_progress(0.37, "Attribution 37/100")
        console.set_progress(0.20, "Stale milestone")
        text = output.getvalue()
        self.assertIn("overall 37% Attribution 37/100", text[-500:])
        self.assertNotIn("overall 20%", text[-500:])
        console.observe({"request_id": "req-1", "phase": "failed"})

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
