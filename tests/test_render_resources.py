"""Pipeline resource runtime: scoped context, drained verification, bounded lanes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread, get_ident
import threading
import unittest

import epub_tts_narrator
from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition
from epub_tts_narrator.renderer.local_runtime import LocalRuntime
from epub_tts_narrator.renderer.render_lanes import ResourceLane
from epub_tts_narrator.renderer.render_resource_context import (
    current_owner, current_resources, resource_scope, run_preparation,
    run_recognition, run_review_leaf, suppress_resource_routing,
)
from epub_tts_narrator.renderer.render_resources import RenderResources

# Import provenance: renderer tests must exercise this worktree's package.
assert Path(epub_tts_narrator.__file__).resolve().is_relative_to(
    Path(__file__).resolve().parents[1])


def resources(**overrides):
    values = {"tts_slots": 2, "synthesis_backlog": 4, "review_concurrency": 1,
              "max_inflight": 8}
    values.update(overrides)
    return RenderResources(**values)


def blocked_leaf(started, gate, record, label="leaf"):
    def run():
        record.append((label, get_ident(), threading.current_thread().name))
        started.set()
        if not gate.wait(5):
            raise AssertionError(f"{label} was never released")
        return label
    return run


def runtime_review_concurrency(runtime):
    return len(runtime.review._threads)


class ResourceContextTests(unittest.TestCase):
    def test_without_scope_leaf_helpers_run_directly_on_the_caller_thread(self):
        self.assertIsNone(current_resources())
        self.assertEqual(current_owner(), (None, None))
        threads = []
        self.assertEqual(run_preparation(lambda: threads.append(get_ident()) or "direct"), "direct")
        self.assertEqual(run_recognition(lambda: threads.append(get_ident()) or "direct"), "direct")
        self.assertEqual(run_review_leaf(lambda: threads.append(get_ident()) or "direct"), "direct")
        self.assertEqual(threads, [get_ident()] * 3)

    def test_resource_scope_publishes_resources_owner_and_resets_nested(self):
        first, second = resources(), resources(review_concurrency=2)
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        seen = []
        with resource_scope(first, ("run-1", 2)):
            seen.append((current_resources() is first, current_owner()))
            with resource_scope(second, ("run-2", 3)):
                seen.append((current_resources() is second, current_owner()))
            seen.append((current_resources() is first, current_owner()))
        self.assertEqual(seen, [(True, ("run-1", 2)), (True, ("run-2", 3)), (True, ("run-1", 2))])
        self.assertIsNone(current_resources())
        self.assertEqual(current_owner(), (None, None))

    def test_scope_reset_survives_leaf_errors(self):
        runtime = resources()
        self.addCleanup(runtime.close)
        with self.assertRaises(ValueError):
            with resource_scope(runtime, ("run", 1)):
                run_preparation(_raise_value_error)
        self.assertIsNone(current_resources())


def _raise_value_error():
    raise ValueError("fixture leaf failure")


class LaneRoutingTests(unittest.TestCase):
    def setUp(self):
        self.runtime = resources(review_concurrency=2)
        self.addCleanup(self.runtime.close)

    def scoped(self, function):
        """Run a helper inside this runtime's scope on the caller-pool thread."""
        def run():
            with resource_scope(self.runtime, ("run", 1)):
                return function()
        return run

    def test_preparation_lane_runs_leaves_on_two_workers_with_lane_metrics(self):
        first_started, second_started, gate = Event(), Event(), Event()
        record = []
        with ThreadPoolExecutor(max_workers=3) as callers:
            first = callers.submit(self.scoped(
                lambda: run_preparation(blocked_leaf(first_started, gate, record, "one"))))
            self.assertTrue(first_started.wait(5))
            second = callers.submit(self.scoped(
                lambda: run_preparation(blocked_leaf(second_started, gate, record, "two"))))
            self.assertTrue(second_started.wait(5))
            # Two declared workers are both blocked, so a third leaf cannot run.
            third_started = Event()
            third = callers.submit(self.scoped(
                lambda: run_preparation(blocked_leaf(third_started, gate, record, "three"))))
            self.assertFalse(third_started.is_set())
            gate.set()
            self.assertEqual([future.result(5) for future in (first, second, third)],
                             ["one", "two", "three"])
        self.assertEqual({thread for _, thread, _ in record} & {get_ident()}, set())
        snapshot = threading.enumerate()
        self.assertEqual(sum(1 for thread in snapshot
                             if thread.name.startswith("render-preparation")), 2)

    def test_recognition_lane_is_a_single_serial_owner(self):
        started, gate, record = Event(), Event(), []
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(self.scoped(
                lambda: run_recognition(blocked_leaf(started, gate, record, "asr1"))))
            self.assertTrue(started.wait(5))
            second_started = Event()
            second = callers.submit(self.scoped(
                lambda: run_recognition(blocked_leaf(second_started, gate, record, "asr2"))))
            self.assertFalse(second_started.is_set())
            gate.set()
            self.assertEqual([first.result(5), second.result(5)], ["asr1", "asr2"])
        self.assertEqual(len({thread for _, thread, _ in record}), 1)

    def test_review_leaf_capacity_equals_declared_review_concurrency(self):
        runtime = resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        started, gate, record = Event(), Event(), []

        def scoped(function):
            def run():
                with resource_scope(runtime, ("run", 1)):
                    return function()
            return run

        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(scoped(
                lambda: run_review_leaf(blocked_leaf(started, gate, record, "http1"))))
            self.assertTrue(started.wait(5))
            second_started = Event()
            second = callers.submit(scoped(
                lambda: run_review_leaf(blocked_leaf(second_started, gate, record, "http2"))))
            self.assertFalse(second_started.is_set())
            gate.set()
            self.assertEqual([first.result(5), second.result(5)], ["http1", "http2"])
        self.assertEqual(len({thread for _, thread, _ in record}), 1)

    def test_review_coordination_runs_two_concurrent_parents_at_default_concurrency(self):
        runtime = resources(review_concurrency=1)
        self.addCleanup(runtime.close)
        first_started, second_started, gate = Event(), Event(), Event()
        record = []
        with ThreadPoolExecutor(max_workers=3) as callers:
            first = callers.submit(runtime.run_review_parent,
                                   blocked_leaf(first_started, gate, record, "vote1"))
            self.assertTrue(first_started.wait(5))
            second = callers.submit(runtime.run_review_parent,
                                    blocked_leaf(second_started, gate, record, "vote2"))
            self.assertTrue(second_started.wait(5))
            third_started = Event()
            third = callers.submit(runtime.run_review_parent,
                                   blocked_leaf(third_started, gate, record, "vote3"))
            self.assertFalse(third_started.is_set())
            gate.set()
            self.assertEqual([future.result(5) for future in (first, second, third)],
                             ["vote1", "vote2", "vote3"])
        self.assertEqual(len(runtime.review_coordination._threads),
                         2 * runtime_review_concurrency(runtime))

    def test_lane_jobs_carry_the_resource_scope_and_owner_for_nested_leaves(self):
        seen = {}
        parent_started, entered, gate = Event(), Event(), Event()

        def parent():
            seen["resources"] = current_resources() is self.runtime
            seen["owner"] = current_owner()
            entered.set()
            seen["leaf_thread"] = run_review_leaf(lambda: (gate.wait(5), get_ident())[1])

        def wrapper():
            parent_started.set()
            return self.runtime.run_review_parent(parent, owner=("run-9", 4))

        worker = Thread(target=wrapper)
        worker.start()
        self.assertTrue(parent_started.wait(5))
        self.assertTrue(entered.wait(5))
        self.assertTrue(seen["resources"])
        self.assertEqual(seen["owner"], ("run-9", 4))
        gate.set()
        worker.join(5)
        self.assertNotEqual(seen["leaf_thread"], get_ident())


class ModelThreadLeafSuppressionTests(unittest.TestCase):
    """Model-affine LocalRuntime execution direct-calls leaves; coordination keeps routing."""

    def setUp(self):
        self.runtime = resources()
        self.addCleanup(self.runtime.close)
        self.model = LocalRuntime()
        self.addCleanup(self.model.close)

    def test_leaves_inside_model_execution_run_inline_while_scope_stays_readable(self):
        threads = []
        seen = {}

        def model_operation():
            seen["resources"] = current_resources() is self.runtime
            seen["owner"] = current_owner()
            return run_recognition(lambda: threads.append(get_ident()) or "inline")

        with resource_scope(self.runtime, ("run", 1)):
            self.assertEqual(self.model.run(model_operation), "inline")
        self.assertTrue(seen["resources"])
        self.assertEqual(seen["owner"], ("run", 1))
        self.assertEqual(threads, [self.model.run(get_ident)],
                         "the suppressed leaf must run on the model thread, not a lane")

    def test_scope_routing_resumes_once_model_execution_ends(self):
        with resource_scope(self.runtime, ("run", 1)):
            self.model.run(lambda: run_recognition(lambda: "inline"))
            routed = run_recognition(lambda: get_ident())
        self.assertNotEqual(routed, get_ident(),
                            "ordinary scope routing must survive the model operation")

    def test_suppress_routing_restores_the_previous_state_and_survives_errors(self):
        with resource_scope(self.runtime, ("run", 1)):
            with suppress_resource_routing():
                self.assertEqual(run_recognition(lambda: "direct"), "direct")
            self.assertEqual(run_recognition(lambda: "routed"), "routed")
            with self.assertRaises(ValueError):
                with suppress_resource_routing():
                    run_recognition(_raise_value_error)
            self.assertEqual(run_review_leaf(lambda: "routed"), "routed")


class DrainBoundaryTests(unittest.TestCase):
    """Started-take verification drains past a lost owner; new synthesis never starts."""

    def setUp(self):
        self.runtime = resources(review_concurrency=1)
        self.addCleanup(self.runtime.close)

    def test_expired_owner_predicate_does_not_stop_verification_leaves(self):
        with scoped_stop_condition(lambda: True, RenderPaused("owner lost")):
            self.assertEqual(run_preparation(lambda: "prepared"), "prepared")
            self.assertEqual(run_recognition(lambda: "recognized"), "recognized")
            self.assertEqual(run_review_leaf(lambda: "reviewed"), "reviewed")
            self.assertEqual(self.runtime.run_review_parent(lambda: "parent"), "parent")

    def test_expired_owner_predicate_still_stops_queued_synthesis_lane_jobs(self):
        started = Event()
        with scoped_stop_condition(lambda: True, RenderPaused("owner lost")):
            future = self.runtime.tts.submit(started.set)
            with self.assertRaises(RenderPaused):
                future.result(5)
        self.assertFalse(started.is_set())

    def test_live_owner_scope_leaves_synthesis_and_verification_unchanged(self):
        record = []
        with scoped_stop_condition(lambda: False, RenderPaused("owner lost")):
            future = self.runtime.tts.submit(lambda: record.append("tts"))
            self.assertEqual(future.result(5), None)
            self.assertEqual(run_preparation(lambda: record.append("prep") or "p"), "p")
        self.assertEqual(record, ["tts", "prep"])


class RenderResourcesCloseTests(unittest.TestCase):
    def queued_and_started(self):
        started, gate = Event(), Event()
        record = []
        return started, gate, record

    def test_close_drains_verification_lanes_and_cancels_queued_synthesis(self):
        runtime = resources(tts_slots=1, synthesis_backlog=2, review_concurrency=1,
                            max_inflight=4)
        self.addCleanup(runtime.close)
        prep_started, prep_gate, prep_record = self.queued_and_started()
        tts_started, tts_gate, tts_record = self.queued_and_started()
        first = runtime.preparation.submit(blocked_leaf(prep_started, prep_gate, prep_record))
        self.assertTrue(prep_started.wait(5))
        queued_prep = runtime.preparation.submit(lambda: prep_record.append(("queued", None, None)))
        running_tts = runtime.tts.submit(blocked_leaf(tts_started, tts_gate, tts_record))
        self.assertTrue(tts_started.wait(5))
        queued_tts = runtime.tts.submit(lambda: tts_record.append(("queued", None, None)))

        before = {thread.name for thread in threading.enumerate()}
        prep_gate.set()
        first.result(5)
        closer = Thread(target=runtime.close)
        closer.start()
        # Close cancels queued synthesis before joining its blocked worker.
        cancelled = Event()
        queued_tts.add_done_callback(lambda future: cancelled.set())
        self.assertTrue(cancelled.wait(10))
        self.assertTrue(queued_tts.cancelled())
        tts_gate.set()
        running_tts.result(5)
        closer.join(10)
        runtime.close()

        self.assertEqual(queued_prep.result(5), None)
        self.assertEqual([name for name, _, _ in prep_record], ["leaf", "queued"])
        self.assertEqual([name for name, _, _ in tts_record], ["leaf"],
                         "the queued synthesis job must never run")
        after = {thread.name for thread in threading.enumerate()}
        self.assertFalse({name for name in after - before if name.startswith("render-")},
                         "resource lane threads leaked past close")
        with self.assertRaises(RuntimeError):
            runtime.preparation.submit(lambda: None)
        with self.assertRaises(RuntimeError):
            runtime.tts.submit(lambda: None)

    def test_close_from_a_lane_worker_is_refused(self):
        runtime = resources()
        self.addCleanup(runtime.close)
        for lane in (runtime.preparation, runtime.tts):
            with self.subTest(lane=lane._name):
                with self.assertRaisesRegex(RuntimeError, "coordinator"):
                    lane.submit(lane.close).result(5)
        self.assertEqual(runtime.preparation.submit(lambda: "open").result(5), "open")

    def test_construction_validates_declared_counts(self):
        for kwargs in ({"tts_slots": 0}, {"review_concurrency": 0}, {"max_inflight": 0},
                       {"synthesis_backlog": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    resources(**kwargs)

    def test_lanes_use_the_delivered_resource_lane_api_with_declared_workers(self):
        runtime = resources(tts_slots=3, synthesis_backlog=6, review_concurrency=2,
                            max_inflight=16)
        self.addCleanup(runtime.close)
        expectations = (
            ("render-tts", runtime.tts, 3, 6),
            ("render-preparation", runtime.preparation, 2, 16),
            ("render-recognition", runtime.recognition, 1, 16),
            ("render-review", runtime.review, 2, 20),
            ("render-review-coordination", runtime.review_coordination, 4, 20),
        )
        for name, lane, workers, pending in expectations:
            with self.subTest(lane=name):
                self.assertIsInstance(lane, ResourceLane)
                self.assertEqual(len(lane._threads), workers)
                self.assertEqual(lane._max_pending, pending)


if __name__ == "__main__":
    unittest.main()
