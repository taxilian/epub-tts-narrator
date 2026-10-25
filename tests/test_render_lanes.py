"""Bounded, cancellable resource lanes for pooled rendering."""

from __future__ import annotations

import contextvars
import json
import threading
import time
import unittest

from epub_tts_narrator.renderer import render_metrics
from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition
from epub_tts_narrator.renderer.render_lanes import LaneFull, ResourceLane


PAYLOAD_MARKER = "DO NOT ECHO: chapter text and api_key=super-secret"

WORKER_WAIT = 10  # deadlock watchdog for every Event wait, join, and future result


def wait_until(predicate, timeout=WORKER_WAIT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def snapshot_gauge(stage):
    """The (queued_now, active_now) pair for one metrics stage."""
    state = render_metrics.METRICS.snapshot()[stage]
    return (state["queued_now"], state["active_now"])


class Blocker:
    """Callable that signals entry and holds its lane worker until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(WORKER_WAIT):
            raise AssertionError("blocker watchdog timed out")
        return "blocked"


class Recorder:
    """Appends a label so a single busy worker defines the service order."""

    def __init__(self, label, order):
        self.label = label
        self.order = order

    def __call__(self):
        self.order.append(self.label)
        return self.label


class LaneTestCase(unittest.TestCase):
    def setUp(self):
        render_metrics.METRICS.reset()
        self.addCleanup(render_metrics.METRICS.reset)
        self.lanes = []

    def make_lane(self, **kwargs):
        lane = ResourceLane(**kwargs)
        self.lanes.append(lane)
        self.addCleanup(lane.close)
        return lane


class ResourceLaneContractTests(LaneTestCase):
    def test_synchronous_run_adapter_returns_the_value_with_arguments_and_owner(self):
        lane = self.make_lane(name="adapter", workers=1, stage="review",
                              max_pending=4)
        self.assertEqual(lane.run(lambda a, b, scale=1: (a + b) * scale,
                                  2, 3, scale=10, owner=("run-1", 4)), 50)

    def test_worker_count_matches_declared_concurrency_and_all_workers_run_at_once(self):
        lane = self.make_lane(name="pooled", workers=8, stage="synthesis",
                              max_pending=16)
        self.assertEqual(sum(1 for thread in threading.enumerate()
                             if thread.name.startswith("pooled-")), 8)
        barrier = threading.Barrier(8, timeout=WORKER_WAIT)
        futures = [lane.submit(barrier.wait) for _ in range(8)]
        self.assertEqual(sorted(future.result(WORKER_WAIT) for future in futures),
                         list(range(8)))

    def test_submissions_receive_a_fresh_context_copy_taken_at_submit_time(self):
        lane = self.make_lane(name="context", workers=1, stage="review",
                              max_pending=4)
        marker = contextvars.ContextVar("lane_marker", default="unset")
        marker.set("at-submit")
        future = lane.submit(marker.get)
        marker.set("changed-after-submit")
        self.assertEqual(future.result(WORKER_WAIT), "at-submit")

    def test_bounded_pending_queue_raises_lane_full_when_capacity_is_exceeded(self):
        lane = self.make_lane(name="bounded", workers=1, stage="synthesis",
                              max_pending=2)
        blocker = Blocker()
        running = lane.submit(blocker)
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        queued = [lane.submit(lambda index=index: index) for index in range(2)]
        with self.assertRaises(LaneFull):
            lane.submit(lambda: "never queued")
        blocker.release.set()
        self.assertEqual(running.result(WORKER_WAIT), "blocked")
        self.assertEqual([future.result(WORKER_WAIT) for future in queued], [0, 1])
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))

    def test_round_robin_serves_owners_one_item_at_a_time_fifo_within_an_owner(self):
        lane = self.make_lane(name="fair", workers=1, stage="synthesis",
                              max_pending=16)
        order = []
        blocker = Blocker()
        holder = lane.submit(blocker, owner=("run-a", 1))
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        labels = (("a2", "run-a"), ("a3", "run-a"), ("b1", "run-b"), ("b2", "run-b"))
        for label, owner in labels:
            lane.submit(Recorder(label, order), owner=(owner, 1))
        blocker.release.set()
        self.assertEqual(holder.result(WORKER_WAIT), "blocked")
        self.assertTrue(wait_until(lambda: len(order) == 4))
        self.assertEqual(order, ["b1", "a2", "b2", "a3"])
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))

    def test_invalid_construction_is_rejected_before_any_worker_starts(self):
        for workers in (0, -1, "four", None):
            with self.subTest(workers=workers):
                with self.assertRaises(ValueError):
                    ResourceLane(name="bad", workers=workers, stage="synthesis",
                                 max_pending=4)
        for stage in ("unknown_stage", "", None):
            with self.subTest(stage=stage):
                with self.assertRaises(ValueError):
                    ResourceLane(name="bad", workers=1, stage=stage, max_pending=4)
        with self.assertRaises(ValueError):
            ResourceLane(name="bad", workers=1, stage="synthesis", max_pending=0)


class LaneCancellationTests(LaneTestCase):
    def test_cancelled_queued_operations_never_execute_and_release_the_queued_gauge(self):
        lane = self.make_lane(name="cancel", workers=1, stage="synthesis",
                              max_pending=8)
        blocker = Blocker()
        running = lane.submit(blocker)
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        executed = []

        def job(index):
            executed.append(index)
            return index

        queued = [lane.submit(job, index) for index in range(2)]
        for future in queued:
            self.assertTrue(future.cancel())
        blocker.release.set()
        self.assertEqual(running.result(WORKER_WAIT), "blocked")
        self.assertEqual(executed, [])
        for future in queued:
            self.assertTrue(future.cancelled())
        stage_state = render_metrics.METRICS.snapshot()["synthesis"]
        self.assertEqual(stage_state["outcomes"]["cancelled"], 2)
        self.assertEqual(stage_state["active_count"], 1)
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))

    def test_stop_requested_at_dequeue_prevents_execution_and_reports_cancelled(self):
        lane = self.make_lane(name="owner-loss", workers=1, stage="recognition",
                              max_pending=4)
        executed = []
        with scoped_stop_condition(lambda: True, RenderPaused):
            future = lane.submit(executed.append, "never")
        with self.assertRaises(RenderPaused):
            future.result(WORKER_WAIT)
        self.assertEqual(executed, [])
        stage_state = render_metrics.METRICS.snapshot()["recognition"]
        self.assertEqual(stage_state["outcomes"]["cancelled"], 1)
        self.assertEqual(stage_state["outcomes"]["failed"], 0)
        self.assertEqual(snapshot_gauge("recognition"), (0, 0))


class LaneFailureTests(LaneTestCase):
    def test_failing_operations_propagate_the_original_exception_and_balance_gauges(self):
        lane = self.make_lane(name="failure", workers=1, stage="synthesis",
                              max_pending=4)

        def rejected():
            raise ValueError(PAYLOAD_MARKER)

        future = lane.submit(rejected)
        with self.assertRaises(ValueError) as caught:
            future.result(WORKER_WAIT)
        self.assertEqual(str(caught.exception), PAYLOAD_MARKER)
        stage_state = render_metrics.METRICS.snapshot()["synthesis"]
        self.assertEqual(stage_state["outcomes"]["failed"], 1)
        self.assertEqual(stage_state["active_count"], 1)
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))
        self.assertNotIn("DO NOT ECHO", json.dumps(render_metrics.METRICS.snapshot()))


class LaneLifecycleTests(LaneTestCase):
    def test_close_cancels_queued_work_and_joins_its_workers_from_the_coordinator(self):
        lane = self.make_lane(name="closer", workers=1, stage="synthesis",
                              max_pending=8)
        blocker = Blocker()
        running = lane.submit(blocker)
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        executed = []
        queued = [lane.submit(executed.append, index) for index in range(3)]
        closer = threading.Thread(target=lane.close)
        closer.start()
        for future in queued:
            self.assertTrue(wait_until(future.cancelled))
        blocker.release.set()
        self.assertEqual(running.result(WORKER_WAIT), "blocked")
        closer.join(WORKER_WAIT)
        self.assertFalse(closer.is_alive())
        self.assertEqual(executed, [])
        self.assertEqual([thread.name for thread in threading.enumerate()
                          if thread.name.startswith("closer-")], [])
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))

    def test_close_with_drain_completes_queued_work_before_workers_exit(self):
        lane = self.make_lane(name="drain", workers=1, stage="synthesis",
                              max_pending=8)
        blocker = Blocker()
        running = lane.submit(blocker)
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        queued = [lane.submit(lambda index=index: f"job-{index}")
                  for index in range(3)]
        closer = threading.Thread(target=lambda: lane.close(drain=True))
        closer.start()
        blocker.release.set()
        self.assertEqual(running.result(WORKER_WAIT), "blocked")
        closer.join(WORKER_WAIT)
        self.assertFalse(closer.is_alive())
        self.assertEqual([future.result(WORKER_WAIT) for future in queued],
                         ["job-0", "job-1", "job-2"])
        self.assertEqual([thread.name for thread in threading.enumerate()
                          if thread.name.startswith("drain-")], [])

    def test_close_from_one_of_its_own_workers_is_refused(self):
        lane = self.make_lane(name="self-close", workers=1, stage="review",
                              max_pending=4)

        def try_close():
            try:
                lane.close()
            except RuntimeError:
                return "refused"
            return "closed"

        self.assertEqual(lane.run(try_close), "refused")
        self.assertEqual([thread.name for thread in threading.enumerate()
                          if thread.name.startswith("self-close-")], ["self-close-0"])

    def test_repeated_close_leaves_no_worker_threads_behind(self):
        before = threading.active_count()
        created = []
        self.addCleanup(lambda: [lane.close() for lane in created])
        for _ in range(3):
            lane = ResourceLane(name="leak", workers=4, stage="review",
                                max_pending=8)
            created.append(lane)
            self.assertEqual(sum(1 for thread in threading.enumerate()
                                 if thread.name.startswith("leak-")), 4)
            self.assertEqual(lane.run(lambda: "ok"), "ok")
            lane.close()
            lane.close()
        self.assertTrue(wait_until(lambda: threading.active_count() == before))
        self.assertEqual([thread.name for thread in threading.enumerate()
                          if thread.name.startswith("leak-")], [])

    def test_submission_after_close_is_rejected(self):
        lane = self.make_lane(name="closed-submit", workers=1, stage="review",
                              max_pending=4)
        lane.close()
        with self.assertRaises(RuntimeError):
            lane.submit(lambda: "never")


class LaneNestingTests(LaneTestCase):
    def test_same_lane_wait_from_a_worker_is_rejected_by_the_deadlock_guard(self):
        lane = self.make_lane(name="guard", workers=1, stage="review",
                              max_pending=4)

        def nested_run():
            try:
                lane.run(lambda: "never")
            except RuntimeError:
                return "run-guarded"
            return "deadlock"

        def nested_submit():
            try:
                lane.submit(lambda: "never")
            except RuntimeError:
                return "submit-guarded"
            return "deadlock"

        self.assertEqual(lane.run(nested_run), "run-guarded")
        self.assertEqual(lane.run(nested_submit), "submit-guarded")

    def test_cross_lane_delegation_from_a_worker_is_allowed(self):
        outer = self.make_lane(name="coordination", workers=1,
                               stage="review_coordination", max_pending=4)
        inner = self.make_lane(name="http", workers=1, stage="review",
                               max_pending=4)
        self.assertEqual(outer.run(lambda: inner.run(lambda: "delegated")),
                         "delegated")
        snapshot = render_metrics.METRICS.snapshot()
        self.assertEqual(snapshot["review_coordination"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["review"]["outcomes"]["completed"], 1)

    def test_lane_stages_are_part_of_the_closed_metrics_catalog(self):
        for stage in ("preparation", "review_coordination"):
            with self.subTest(stage=stage):
                self.assertIn(stage, render_metrics.STAGES)
        lane = self.make_lane(name="prep", workers=2, stage="preparation",
                              max_pending=4)
        self.assertEqual(lane.run(lambda: "ok"), "ok")
        self.assertEqual(render_metrics.METRICS.snapshot()
                         ["preparation"]["outcomes"]["completed"], 1)


class StageOverlapTests(LaneTestCase):
    def test_eight_tts_slots_saturate_while_recognition_stays_blocked(self):
        tts = self.make_lane(name="tts", workers=8, stage="synthesis",
                             max_pending=32)
        asr = self.make_lane(name="asr", workers=1, stage="recognition",
                             max_pending=8)
        blocker = Blocker()
        asr_future = asr.submit(blocker)
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        done = [threading.Event() for _ in range(8)]
        tts_futures = [tts.submit(event.set) for event in done]
        for event in done:
            self.assertTrue(event.wait(WORKER_WAIT))
        for future in tts_futures:
            self.assertTrue(future.done())
        self.assertFalse(asr_future.done())  # ASR stayed blocked the whole time
        snapshot = render_metrics.METRICS.snapshot()
        self.assertEqual(snapshot["synthesis"]["outcomes"]["completed"], 8)
        self.assertEqual(snapshot["recognition"]["active_now"], 1)
        self.assertEqual(snapshot["recognition"]["outcomes"]["completed"], 0)
        blocker.release.set()
        self.assertEqual(asr_future.result(WORKER_WAIT), "blocked")
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))
        self.assertEqual(snapshot_gauge("recognition"), (0, 0))

    def test_fast_lane_completes_more_work_than_a_blocked_lane(self):
        fast = self.make_lane(name="fast", workers=4, stage="synthesis",
                              max_pending=16)
        slow = self.make_lane(name="slow", workers=1, stage="synthesis",
                              max_pending=16)
        blocker = Blocker()
        slow_future = slow.submit(blocker)
        self.assertTrue(blocker.entered.wait(WORKER_WAIT))
        fast_futures = [fast.submit(lambda index=index: index * 2)
                        for index in range(10)]
        self.assertEqual([future.result(WORKER_WAIT) for future in fast_futures],
                         [index * 2 for index in range(10)])
        self.assertFalse(slow_future.done())
        stage_state = render_metrics.METRICS.snapshot()["synthesis"]
        self.assertEqual(stage_state["outcomes"]["completed"], 10)
        self.assertEqual(stage_state["active_now"], 1)
        blocker.release.set()
        self.assertEqual(slow_future.result(WORKER_WAIT), "blocked")
        self.assertEqual(snapshot_gauge("synthesis"), (0, 0))


if __name__ == "__main__":
    unittest.main()
