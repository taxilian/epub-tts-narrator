"""Read-only scheduling status: single runtime slot, conservative snapshot shape."""

from threading import Lock
from types import SimpleNamespace
import unittest

from epub_tts_narrator.renderer.render_backlog import BacklogBudget
from epub_tts_narrator.renderer.render_metrics import METRICS, STAGES
from epub_tts_narrator.renderer.render_pipeline import LogicalClipScheduler
from epub_tts_narrator.renderer.render_resource_context import resource_scope
from epub_tts_narrator.renderer.render_schedule_status import (
    clear_runtime, register_runtime, scheduling_status,
)


FIXED_LABELS = {"pipeline", "legacy", "pipeline_global_transactions",
                "legacy_per_chapter_window"}


def pipeline_stub(*, backlog=None, logical=None, max_parallel=8, max_inflight=16):
    return SimpleNamespace(max_parallel=max_parallel, max_inflight=max_inflight,
                           logical=logical, resources=SimpleNamespace(backlog=backlog))


def effective(scheduler="pipeline", chapter_concurrency=4, max_inflight=16):
    return {"scheduler": scheduler, "chapter_concurrency": chapter_concurrency,
            "max_inflight": max_inflight, "synthesis_backlog": 32,
            "pending_audio_mb": 256, "review_concurrency": 2}


class RuntimeSlotTests(unittest.TestCase):
    def tearDown(self):
        clear_runtime(object())  # engine-specific clear; an unknown engine is a no-op

    def test_unregistered_process_reports_no_engine_status(self):
        self.assertIsNone(scheduling_status())

    def test_registration_replaces_a_single_slot_and_clear_is_engine_specific(self):
        first, second = pipeline_stub(), pipeline_stub()
        for engine in (first, second, first):
            register_runtime(engine, None)
            self.assertIsNotNone(scheduling_status())
            clear_runtime(engine)
            self.assertIsNone(scheduling_status())
        register_runtime(first, None)
        register_runtime(second, None)
        clear_runtime(first)  # a stale engine must not evict the live slot
        register_runtime(first, None)
        clear_runtime(first)
        self.assertIsNone(scheduling_status())

    def test_unknown_plugin_engine_stays_conservative_and_secret_free(self):
        class Odd:
            SECRET_TOKEN = "private-credential-value"

            def __getattr__(self, name):
                raise AttributeError(name)

        engine = Odd()
        register_runtime(engine, None)
        self.addCleanup(clear_runtime, engine)
        # No known scheduling attribute: report nothing rather than invent it.
        self.assertIsNone(scheduling_status())
        self.assertNotIn("private-credential-value", repr(scheduling_status()))


class SchedulingStatusTests(unittest.TestCase):
    def setUp(self):
        METRICS.reset()
        self.addCleanup(METRICS.reset)

    def tearDown(self):
        clear_runtime(object())

    def test_pipeline_status_reports_logical_backlog_and_capacity_basis(self):
        backlog = BacklogBudget(max_pending=8, max_materialized_bytes=4096)
        scheduler = LogicalClipScheduler(16, 4)
        self.addCleanup(scheduler.close)
        engine = pipeline_stub(backlog=backlog, logical=scheduler)
        register_runtime(engine, effective())
        self.addCleanup(clear_runtime, engine)
        status = scheduling_status()
        self.assertEqual(status["mode"], "pipeline")
        self.assertEqual(status["capacity"]["tts_slots"], 8)
        self.assertEqual(status["capacity"]["max_inflight"], 16)
        self.assertEqual(status["capacity"]["inflight_basis"], "pipeline_global_transactions")
        self.assertEqual(status["capacity"]["chapter_concurrency"], 4)
        self.assertEqual(status["logical"]["global_limit"], 16)
        self.assertEqual(status["logical"]["chapter_share"], 4)
        self.assertEqual(status["backlog"]["max_pending"], 8)
        self.assertIn("peak_pending_count", status["backlog"])

    def test_legacy_window_is_not_reported_as_the_pipeline_projection(self):
        engine = pipeline_stub(max_parallel=2, max_inflight=3, backlog=None, logical=None)
        register_runtime(engine, effective(scheduler="legacy", chapter_concurrency=2,
                                           max_inflight=16))
        self.addCleanup(clear_runtime, engine)
        status = scheduling_status()
        self.assertEqual(status["mode"], "legacy")
        self.assertEqual(status["capacity"]["max_inflight"], 3)
        self.assertEqual(status["capacity"]["inflight_basis"], "legacy_per_chapter_window")
        # The automatic 16 is the pipeline-only projection ceiling, never the
        # live legacy per-chapter window; both numbers stay distinguishable.
        self.assertEqual(status["capacity"]["pipeline_projection_max_inflight"], 16)

    def test_scope_resources_expose_the_backlog_without_a_registered_engine(self):
        backlog = BacklogBudget(max_pending=4, max_materialized_bytes=2048)
        resources = SimpleNamespace(backlog=backlog)
        with resource_scope(resources, ("run", 1)):
            status = scheduling_status()
        self.assertTrue(status["available"])
        self.assertIsNone(status["mode"])
        self.assertIsNone(status["logical"])
        self.assertEqual(status["backlog"]["max_pending"], 4)

    def test_stage_gauges_are_sparse_over_the_closed_stage_set(self):
        operation = METRICS.operation("synthesis")
        self.addCleanup(operation.cancel)
        status = scheduling_status()
        self.assertTrue(status["available"])
        self.assertEqual(set(status["stages"]), {"synthesis"})
        self.assertEqual(status["stages"]["synthesis"], {"queued_now": 1, "active_now": 0})
        self.assertTrue(set(status["stages"]) <= set(STAGES))

    def test_empty_process_reports_nothing(self):
        self.assertIsNone(scheduling_status())

    def test_broken_runtime_sections_never_break_the_snapshot(self):
        class Exploding:
            max_parallel = 4

            @property
            def max_inflight(self):
                raise RuntimeError("boom")

            logical = None
            resources = None

        engine = Exploding()
        register_runtime(engine, None)
        self.addCleanup(clear_runtime, engine)
        status = scheduling_status()
        self.assertEqual(status["capacity"]["tts_slots"], 4)
        self.assertIsNone(status["capacity"]["max_inflight"])
        self.assertIsNone(status["capacity"]["inflight_basis"])
        self.assertIsNone(status["mode"])

    def test_status_values_stay_numeric_or_fixed_labels(self):
        backlog = BacklogBudget(max_pending=2, max_materialized_bytes=512)
        scheduler = LogicalClipScheduler(8, 2)
        self.addCleanup(scheduler.close)
        engine = pipeline_stub(backlog=backlog, logical=scheduler)
        register_runtime(engine, effective())
        self.addCleanup(clear_runtime, engine)
        status = scheduling_status()

        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertIsInstance(key, str)
                    walk(item)
                return
            self.assertIsInstance(value, (int, bool, str), repr(value))
            if isinstance(value, str):
                self.assertIn(value, FIXED_LABELS)

        walk(status)


class SchedulerSnapshotTests(unittest.TestCase):
    def test_fresh_scheduler_reports_zeroed_gauges(self):
        scheduler = LogicalClipScheduler(8, 2)
        self.addCleanup(scheduler.close)
        self.assertEqual(scheduler.snapshot(),
                         {"pending": 0, "inflight": 0, "owners": 0,
                          "global_limit": 8, "chapter_share": 4, "closed": False})

    def test_snapshot_counts_registered_work_and_stays_consistent_after_drain(self):
        gate = Lock()
        gate.acquire()
        scheduler = LogicalClipScheduler(1, 1, pending_limit=8)
        self.addCleanup(scheduler.close)

        def blocked(index):
            with gate:
                pass
            return index

        try:
            futures = [scheduler.submit(("run", 1), blocked, index) for index in range(3)]
            snapshot = scheduler.snapshot()
        finally:
            gate.release()
        self.assertEqual(snapshot["owners"], 1)
        self.assertEqual(snapshot["pending"] + snapshot["inflight"], 3)
        self.assertFalse(snapshot["closed"])
        for future in futures:
            future.result(timeout=5)
        scheduler.retire_owner(("run", 1))
        drained = scheduler.snapshot()
        self.assertEqual(drained["owners"], 0)
        self.assertEqual(drained["pending"] + drained["inflight"], 0)


if __name__ == "__main__":
    unittest.main()
