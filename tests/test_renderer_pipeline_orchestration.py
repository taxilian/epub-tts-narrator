"""Shared fair chapter orchestration on the real RenderPool scheduler surface."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.chapter_executor import MAX_PARALLEL_CHAPTERS, ChapterExecutor
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.render_chapter_flow import (
    ChapterRunAdmission, collect_chapter, retire_owner, run_units,
)
from epub_tts_narrator.renderer.workflow import _render_chapter_chunks


def configuration(scheduler="pipeline", chapter_concurrency=4, max_inflight=2,
                  workers_concurrency=2):
    config = deepcopy(DEFAULTS)
    config["render_pool"].update(
        scheduler=scheduler, chapter_concurrency=chapter_concurrency,
        max_inflight=max_inflight,
        workers=[{"name": "fixture", "type": "fish_http",
                  "url": "http://fixture.invalid/v1/tts",
                  "concurrency": workers_concurrency}])
    return config


def chapter(number, count):
    chunks = [{"chunk_id": f"{number}-{index}", "spoken_text": f"Words {index}."}
              for index in range(count)]
    return {"number": number, "chunks": chunks}


class RecordingChunks:
    """Fake engine I/O boundary: records starts, active peaks, and orders."""

    def __init__(self, test):
        self.test = test
        self.lock = threading.Lock()
        self.order = []
        self.events = {}
        self.active = 0
        self.peak = 0
        self.per_owner = {}

    def record(self, name, owner):
        with self.lock:
            self.order.append(name)
            self.events.setdefault(name, threading.Event()).set()
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.per_owner[owner] = self.per_owner.get(owner, 0) + 1
            return self.events[name]

    def finish(self, name, owner):
        with self.lock:
            self.active -= 1
            self.per_owner[owner] -= 1

    def expect(self, name, timeout=10.0):
        self.events.setdefault(name, threading.Event())
        self.test.assertTrue(self.events[name].wait(timeout),
                             f"{name} never started; {self.order}")

    def expect_absent(self, *names):
        with self.lock:
            present = [name for name in names if name in self.order]
        self.test.assertEqual(present, [], f"started too early; {self.order}")


def chunk_runner(recorder, gates=None, failures=None):
    """Patched workflow._render_chunk with Event-gated fakes."""
    gates = gates or {}
    failures = failures or {}

    def fake(chunk, number, total, voices, engine, engine_identity, verifier,
             config, work, take_locks, take_locks_guard, *, notify,
             retry_budget=None, speech_cache=None, paragraph_split_available=False):
        name = chunk["chunk_id"]
        owner = ("run", number)
        recorder.record(name, owner)
        gate = gates.get(name)
        if gate is not None:
            gate.wait(30)
        failure = failures.get(name)
        if failure is not None:
            recorder.finish(name, owner)
            raise failure
        recorder.finish(name, owner)
        return (dict(chunk, wav_path=f"/tmp/{name}.wav"), {"chunk_id": name})
    return fake


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.recorder = RecordingChunks(self)

    def pool(self, **kwargs):
        pool = RenderPool(configuration(**kwargs))
        self.addCleanup(pool.close)
        return pool

    def render_chapter(self, pool, chap, owner_run="run"):
        with patch("epub_tts_narrator.renderer.workflow._render_chunk",
                   side_effect=chunk_runner(self.recorder, self.gates, self.failures)):
            return _render_chapter_chunks(
                chap, {}, pool, pool.identity, object(), configuration(), self.work,
                notify=lambda _: None, retry_budget=object(),
                take_locks={}, take_locks_guard=threading.Lock(),
                scheduler_owner=(owner_run, chap["number"]))
    def test_pipeline_pool_owns_one_scheduler_and_legacy_builds_none(self):
        legacy = self.pool(scheduler="legacy")
        self.assertIsNone(legacy.logical)
        pipeline = self.pool()
        self.assertIsNotNone(pipeline.logical)
        self.assertEqual(pipeline.logical.global_limit, 2)
        self.assertEqual(pipeline.logical.share, 1)
        order = []
        real_resources_close = pipeline._resources.close
        pipeline._resources.close = lambda: (order.append("resources"),
                                             real_resources_close())[1]
        real_logical_close = pipeline._logical.close
        pipeline._logical.close = lambda *a, **k: (order.append("logical"),
                                                   real_logical_close(*a, **k))[1]
        pipeline.close()
        self.assertLess(order.index("logical"), order.index("resources"))

    def test_chapters_share_global_cap_and_late_chapters_keep_reserved_slots(self):
        self.gates = {"1-0": threading.Event(), "2-0": threading.Event()}
        self.failures = {}
        pool = self.pool()  # global_limit 2, chapter_concurrency 4 -> share 1
        admission = ChapterRunAdmission()

        def unit(index):
            return self.render_chapter(pool, chapter(index + 1, 3))

        outcome = {}

        def run():
            try:
                outcome["result"] = run_units([unit, unit, unit, unit], 4, admission)
            except BaseException as error:  # noqa: BLE001
                outcome["error"] = error

        with patch("epub_tts_narrator.renderer.workflow._render_chunk",
                   side_effect=chunk_runner(self.recorder, self.gates, self.failures)):
            thread = threading.Thread(target=run)
            thread.start()
            self.recorder.expect("1-0")
            self.recorder.expect("2-0")  # late chapter admitted beside blocked first
            # Both global slots are held by gated first clips: siblings and
            # later chapters stay queued at their non-borrowable shares.
            self.recorder.expect_absent("1-1", "2-1", "3-0", "4-0")
            self.gates["1-0"].set()
            wait_until(lambda: len(self.recorder.order) >= 3)
            # The freed slot serves exactly one new transaction, never a
            # sibling of the chapter that still holds its share.
            with self.recorder.lock:
                next_starter = self.recorder.order[2]
            self.assertNotEqual(next_starter.split("-")[0], "2", self.recorder.order)
            self.gates["2-0"].set()
            thread.join(60)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome, outcome)
        self.recorder.expect("4-2")
        self.assertEqual(self.recorder.peak, 2)  # global logical cap
        for owner, active in self.recorder.per_owner.items():
            self.assertLessEqual(active, 1, owner)  # non-borrowable share of 1
        result = outcome["result"]
        for chapter_index, unit_result in enumerate(result, 1):
            clips = unit_result[0]
            self.assertEqual([clip["chunk_id"] for clip in clips],
                             [f"{chapter_index}-{i}" for i in range(3)])

    def test_terminal_failure_cancels_queued_siblings_and_drains_started(self):
        self.gates = {"1-0": threading.Event()}
        self.failures = {"1-1": RuntimeError("terminal clip failure")}
        pool = self.pool(chapter_concurrency=1)
        admission = ChapterRunAdmission()

        def first(index):
            return self.render_chapter(pool, chapter(1, 3))

        def second(index):
            return self.render_chapter(pool, chapter(2, 2))

        errors = []

        def run():
            try:
                run_units([first, second], 2, admission)
            except BaseException as error:
                errors.append(error)

        with patch("epub_tts_narrator.renderer.workflow._render_chunk",
                   side_effect=chunk_runner(self.recorder, self.gates, self.failures)):
            thread = threading.Thread(target=run)
            thread.start()
            self.recorder.expect("2-1")  # the healthy chapter completes fully
            self.recorder.expect_absent("1-2")  # queued sibling cancelled before start
            self.gates["1-0"].set()  # the already-started take drains
            thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], RuntimeError)
        self.recorder.expect_absent("1-2")
        with self.assertRaises(RuntimeError):
            admission.enter()

    def test_results_keep_source_order_when_completions_are_out_of_order(self):
        self.gates = {"1-0": threading.Event(), "1-1": threading.Event()}
        self.failures = {}
        pool = self.pool(chapter_concurrency=2, max_inflight=4, workers_concurrency=4)
        results = []

        def attempt():
            results.append(self.render_chapter(pool, chapter(1, 3)))

        thread = threading.Thread(target=attempt)
        thread.start()
        self.recorder.expect("1-0")
        self.recorder.expect("1-1")
        self.recorder.expect_absent("1-2")  # share bounds outstanding transactions
        self.gates["1-1"].set()
        self.recorder.expect("1-2")  # later chunk starts before the first finishes
        self.gates["1-0"].set()
        thread.join(30)
        self.assertFalse(thread.is_alive())
        clips, _ = results[0]
        self.assertEqual([clip["chunk_id"] for clip in clips], ["1-0", "1-1", "1-2"])
    def test_replan_gets_a_fresh_generation_and_owners_leave_no_residue(self):
        self.gates = {}
        self.failures = {"1-0": RuntimeError("terminal")}
        pool = self.pool(chapter_concurrency=2, max_inflight=4, workers_concurrency=4)
        scheduler = pool.logical
        with patch("epub_tts_narrator.renderer.workflow._render_chunk",
                   side_effect=chunk_runner(self.recorder, self.gates, self.failures)):
            with self.assertRaises(RuntimeError):
                self.render_chapter(pool, chapter(1, 2))
        self.assertFalse(scheduler.retire_owner(("run", 1)))
        self.failures = {}  # paragraph fallback replan: fresh generation, same budget
        clips, _ = self.render_chapter(pool, chapter(1, 2))
        self.assertEqual([clip["chunk_id"] for clip in clips], ["1-0", "1-1"])
        # The chapter flow retires the owner on completion: no scheduling
        # residue accumulates on a warm server.
        self.assertEqual(scheduler._owners, {})

    def test_run_units_fails_fast_and_drains_started_chapters(self):
        admission = ChapterRunAdmission()
        two_started = threading.Event()
        failure_recorded = threading.Event()
        gate = threading.Event()
        third_started = threading.Event()
        errors = []

        def unit_one(index):
            self.assertTrue(two_started.wait(10))
            admission.fail(RuntimeError("book failure"))
            failure_recorded.set()
            raise RuntimeError("book failure")

        def unit_two(index):
            two_started.set()
            self.assertTrue(failure_recorded.wait(10))
            self.assertTrue(gate.wait(30))
            return "two"

        def unit_three(index):
            third_started.set()
            return "three"

        def run():
            try:
                run_units([unit_one, unit_two, unit_three], 2, admission)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(failure_recorded.wait(10))
        gate.set()  # the started unit drains; nothing new may start
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertFalse(third_started.is_set())

    def test_unknown_plugins_stay_serial_without_scheduler_or_scope(self):
        engine = type("UnknownEngine", (), {"identity": {"engine": "unknown"}})()
        calls = []

        def fake(chunk, number, total, *args, **kwargs):
            calls.append(chunk["chunk_id"])
            return dict(chunk, wav_path=None), None

        with patch("epub_tts_narrator.renderer.workflow._render_chunk", side_effect=fake):
            clips, acceptance = _render_chapter_chunks(
                chapter(1, 3), {}, engine, engine.identity, object(), configuration(),
                self.work, notify=lambda _: None)
        self.assertEqual(calls, ["1-0", "1-1", "1-2"])
        self.assertIsNone(getattr(engine, "logical", None))

    def test_chapter_executor_and_jobs_follow_configured_parallel_with_sixteen_ceiling(self):
        legacy_pool = RenderPool(configuration(scheduler="legacy", chapter_concurrency=0))
        self.addCleanup(legacy_pool.close)
        pipeline_pool = RenderPool(configuration(chapter_concurrency=4))
        self.addCleanup(pipeline_pool.close)
        legacy = ChapterExecutor(configuration(scheduler="legacy", chapter_concurrency=0),
                                 engine=legacy_pool, verifier=object())
        fixture_verifier = type("FixtureVerifier", (), {"identity": {"backend": "v"}})()
        pipeline = ChapterExecutor(configuration(chapter_concurrency=4), engine=pipeline_pool,
                                   verifier=fixture_verifier)
        self.assertEqual(legacy.parallel_chapters, 2)
        self.assertEqual(pipeline.parallel_chapters, 4)
        self.assertNotIn("parallel_chapters", pipeline.identity)

        class FixtureExecutor:
            identity = {"engine": "fixture"}
            parallel_chapters = 24

        with tempfile.TemporaryDirectory() as temporary:
            service = ChapterJobs(Path(temporary) / "jobs", FixtureExecutor())
            self.assertEqual(service._parallel_chapters, MAX_PARALLEL_CHAPTERS)
            self.assertEqual(MAX_PARALLEL_CHAPTERS, 16)
            bounded = ChapterJobs(Path(temporary) / "jobs2",
                                  type("Bounded", (), {"identity": {"e": 1},
                                                       "parallel_chapters": 4})())
            self.assertEqual(bounded._parallel_chapters, 4)

    def test_collect_chapter_bounds_outstanding_submissions_to_the_share(self):
        pool = self.pool(chapter_concurrency=4, max_inflight=2, workers_concurrency=2)
        scheduler = pool.logical
        inflight = {"now": 0, "peak": 0}
        lock = threading.Lock()
        release = threading.Event()

        def run_one(index):
            with lock:
                inflight["now"] += 1
                inflight["peak"] = max(inflight["peak"], inflight["now"])
            release.wait(30)
            with lock:
                inflight["now"] -= 1
            return index

        with ThreadPoolExecutor(max_workers=1) as coordinator:
            future = coordinator.submit(collect_chapter, scheduler, ("run", 1), 6, run_one)
            deadline = threading.Event()
            while inflight["peak"] < 1 and not deadline.wait(0.05):
                pass
            self.assertLessEqual(inflight["peak"], scheduler.share)
            release.set()
        self.assertEqual(future.result(30), list(range(6)))
        self.assertEqual(inflight["peak"], scheduler.share)


def wait_until(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class VerifierRegressionTests(unittest.TestCase):
    """The four independent-review counterexamples, on the real scheduler."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def pool(self, **kwargs):
        config = deepcopy(DEFAULTS)
        config["render_pool"].update(
            scheduler="pipeline", chapter_concurrency=kwargs.pop("chapter_concurrency", 4),
            max_inflight=kwargs.pop("max_inflight", 0),
            workers=[{"name": "fixture", "type": kwargs.pop("worker_type", "fish_http"),
                      "url": "http://fixture.invalid/v1/tts",
                      "concurrency": kwargs.pop("workers_concurrency", 1)}],
            **kwargs)
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        return pool

    def test_overlapping_runs_of_one_book_chapter_cancel_only_the_old_run(self):
        pool = self.pool(chapter_concurrency=1, max_inflight=8, workers_concurrency=2)
        scheduler = pool.logical
        old_owner = ("a" * 12 + "aaaa", 3)   # job/epoch token chapter_jobs passes
        new_owner = ("b" * 12 + "bbbb", 1)   # a fresh client run of the same book chapter
        old_gate = threading.Event()
        old_started = threading.Event()
        new_started = threading.Event()
        new_running = threading.Event()
        filler_running = threading.Event()
        filler_gate = threading.Event()
        outcomes = {}

        def old_transaction(index):
            if index == 0:
                old_started.set()
                old_gate.wait(30)
                raise RuntimeError("old run terminal clip failure")
            return index

        def new_transaction(index):
            if index == 0:
                new_started.set()
                new_running.wait(30)
                return index
            return index

        def filler(index):
            if index == 0:
                filler_running.set()
            filler_gate.wait(30)
            return index

        def run_job(tag, owner_key, transaction, count):
            try:
                collect_chapter(scheduler, owner_key, count, transaction)
                outcomes[tag] = "ok"
            except BaseException as error:  # noqa: BLE001
                outcomes[tag] = error

        threads = [threading.Thread(target=run_job,
                                    args=("filler", ("filler", 9), filler, 6)),
                   threading.Thread(target=run_job,
                                    args=("old", old_owner, old_transaction, 3)),
                   threading.Thread(target=run_job,
                                    args=("new", new_owner, new_transaction, 4))]
        threads[0].start()
        self.assertTrue(wait_until(lambda: scheduler._inflight == 6))
        threads[1].start()
        self.assertTrue(old_started.wait(10))
        self.assertTrue(wait_until(lambda: scheduler._inflight == 7))
        threads[2].start()
        self.assertTrue(new_started.wait(10))
        self.assertTrue(wait_until(lambda: scheduler._inflight == 8))
        # The new run's transactions 1..3 are queued, undispatched, right now.
        old_gate.set()
        threads[1].join(30)
        filler_gate.set()
        threads[0].join(30)
        new_running.set()
        threads[2].join(30)
        for thread in threads:
            self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcomes["old"], RuntimeError)
        self.assertEqual(outcomes["new"], "ok")   # the connected run is untouched
        self.assertEqual(outcomes["filler"], "ok")

    def test_all_queued_cancellation_never_hangs_the_coordinator(self):
        pool = self.pool(chapter_concurrency=1, max_inflight=8, workers_concurrency=4)
        scheduler = pool.logical
        outcome = {}

        def failing(index):
            raise RuntimeError("fast terminal failure")

        def run():
            try:
                collect_chapter(scheduler, ("run", 1), 5, failing)
            except BaseException as error:  # noqa: BLE001
                outcome["error"] = error

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(30)
        self.assertFalse(thread.is_alive(), "coordinator hung on all-queued cancel")
        self.assertIsInstance(outcome.get("error"), RuntimeError)
        # collect retired the drained owner before raising: no residue.
        self.assertEqual(scheduler._owners, {})
        self.assertFalse(scheduler.retire_owner(("run", 1)))

    def test_queued_siblings_never_start_after_the_terminal_failure_instant(self):
        pool = self.pool(chapter_concurrency=2, max_inflight=2, workers_concurrency=1)
        scheduler = pool.logical
        failing_started = threading.Event()
        failing_now = threading.Event()
        filler_running = threading.Event()
        filler_gate = threading.Event()
        started = []
        started_lock = threading.Lock()

        def filler(index):
            filler_running.set()
            filler_gate.wait(30)
            return index

        def transaction(index):
            name = f"1-{index}"
            with started_lock:
                started.append(name)
            if index == 0:
                failing_started.set()
                self.assertTrue(failing_now.wait(30))
                raise RuntimeError("terminal")
            return name

        filler_thread = threading.Thread(
            target=collect_chapter,
            args=(scheduler, ("filler", 9), 1, filler))
        filler_thread.start()
        self.assertTrue(filler_running.wait(10))
        outcome = {}

        def run():
            try:
                collect_chapter(scheduler, ("run", 1), 4, transaction)
            except BaseException as error:  # noqa: BLE001
                outcome["error"] = error

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(failing_started.wait(10))
        failing_now.set()  # the failure exists from this instant
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome.get("error"), RuntimeError)
        # The scheduler-side admission gate cancelled every queued sibling at
        # the failure instant: no sibling transaction started afterwards.
        with started_lock:
            self.assertEqual(started, ["1-0"])
        # The failed run's owner left no scheduling residue; the successful
        # filler retires through the same call its coordinator makes.
        self.assertFalse(scheduler.retire_owner(("run", 1)))
        filler_gate.set()
        filler_thread.join(30)
        self.assertTrue(scheduler.retire_owner(("filler", 9)))
        self.assertEqual(scheduler._owners, {})

    def test_auto_pipeline_lane_bounds_match_the_logical_limit_with_backpressure(self):
        pool = self.pool(chapter_concurrency=4, max_inflight=0, workers_concurrency=1)
        effective = pool.logical.global_limit
        self.assertEqual(pool.resources.recognition._max_pending, effective)
        self.assertEqual(pool.resources.preparation._max_pending, effective)
        self.assertEqual(pool.resources.review._max_pending,
                         effective + 2 * pool._effective["review_concurrency"])
        self.assertEqual(pool.max_inflight, effective)  # every view agrees
        legacy_auto = RenderPool(configuration(scheduler="legacy", max_inflight=0,
                                               workers_concurrency=1))
        self.addCleanup(legacy_auto.close)
        self.assertEqual(legacy_auto.max_inflight, 2)  # legacy window: slots+1
        self.assertIsNone(legacy_auto.logical)
        blocked = threading.Event()
        completed = []
        errors = []

        def blocked_asr(path):
            blocked.wait(30)
            return {"text": "x"}

        def submitter(index):
            try:
                pool.resources.run_recognition(blocked_asr, f"p{index}",
                                               owner=(f"run{index}", 1))
                completed.append(index)
            except BaseException as error:  # noqa: BLE001
                errors.append(repr(error))

        threads = [threading.Thread(target=submitter, args=(index,))
                   for index in range(6)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 10
        while len(completed) + len(errors) < 6 and time.monotonic() < deadline:
            time.sleep(0.02)
        blocked.set()
        for thread in threads:
            thread.join(30)
        self.assertEqual(errors, [])  # saturated lanes backpressure, never LaneFull
        self.assertEqual(sorted(completed), list(range(6)))

    def test_saturated_lane_backpressures_without_rerunning_started_work(self):
        from epub_tts_narrator.renderer.render_lanes import ResourceLane
        lane = ResourceLane(name="render-test", workers=1, stage="recognition",
                            max_pending=1)
        self.addCleanup(lane.close)
        release = threading.Event()
        runs = []
        lock = threading.Lock()

        def job(index):
            def run():
                with lock:
                    runs.append(index)
                release.wait(30)
                return index
            return run

        results, errors = [], []

        def caller(index):
            try:
                results.append(lane.run(job(index), owner=("o", 1)))
            except BaseException as error:  # noqa: BLE001
                errors.append(repr(error))

        threads = [threading.Thread(target=caller, args=(index,))
                   for index in range(3)]
        for thread in threads:
            thread.start()
        self.assertTrue(wait_until(lambda: len(runs) >= 1))
        time.sleep(0.1)
        self.assertEqual(runs, [0])  # one running, one queued, one blocked
        release.set()
        for thread in threads:
            thread.join(30)
        self.assertEqual(errors, [])
        self.assertEqual(sorted(results), [0, 1, 2])
        self.assertEqual(sorted(runs), [0, 1, 2])  # each function ran exactly once


if __name__ == "__main__":
    unittest.main()
