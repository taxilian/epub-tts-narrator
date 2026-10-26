"""The ChapterProgress status line surfaces scheduling gauges without payload."""

from io import StringIO
from types import SimpleNamespace
import unittest

from epub_tts_narrator.renderer.progress import ChapterProgress
from epub_tts_narrator.renderer.render_schedule_status import (
    clear_runtime, register_runtime,
)


class StubRuntime:
    def __init__(self, *, broken=False):
        self._broken = broken
        self.max_parallel = 4
        self.max_inflight = 16
        self.logical = SimpleNamespace(
            snapshot=lambda: {"pending": 1, "inflight": 2, "owners": 1,
                              "global_limit": 16, "chapter_share": 4, "closed": False})
        self.resources = SimpleNamespace(backlog=SimpleNamespace(
            snapshot=lambda: {"pending_count": 3, "materialized_bytes": 2048,
                              "max_pending": 8, "max_materialized_bytes": 4096,
                              "low_water_bytes": 2048, "new_generation_blocked": False,
                              "waiters": 1, "closed": False,
                              "peak_pending_count": 5, "peak_materialized_bytes": 3072}))


class BrokenRuntime:
    max_parallel = 2

    @property
    def max_inflight(self):
        raise RuntimeError("boom")

    logical = None
    resources = None


def render_line(runtime=None, effective=None):
    if runtime is not None:
        register_runtime(runtime, effective)
    messages = []
    chunks = [{"spoken_text": "hello world", "silent_control": False}]
    with ChapterProgress(1, 1, chunks, notify=messages.append,
                         stream=StringIO(), heartbeat=False) as progress:
        progress.refresh()
    return "\n".join(messages)


class ProgressSchedulingSegmentTests(unittest.TestCase):
    def tearDown(self):
        clear_runtime(object())

    def test_registered_runtime_appends_a_numeric_scheduling_segment(self):
        runtime = StubRuntime()
        line = render_line(runtime, {"scheduler": "pipeline", "chapter_concurrency": 4,
                                     "max_inflight": 16, "synthesis_backlog": 8,
                                     "pending_audio_mb": 256, "review_concurrency": 2})
        self.addCleanup(clear_runtime, runtime)
        self.assertIn("sched", line)
        self.assertIn("backlog", line)
        self.assertIn("peak", line)

    def test_unregistered_render_line_stays_unchanged(self):
        line = render_line()
        self.assertNotIn("sched", line)

    def test_a_broken_runtime_never_breaks_the_status_line(self):
        runtime = BrokenRuntime()
        line = render_line(runtime)
        self.addCleanup(clear_runtime, runtime)
        self.assertIn("Blocks 0 / 1 complete", line)

    def test_the_segment_carries_no_payload(self):
        runtime = StubRuntime()
        line = render_line(runtime, None)
        self.addCleanup(clear_runtime, runtime)
        self.assertNotIn("hello world", line.split("sched", 1)[-1])


if __name__ == "__main__":
    unittest.main()
