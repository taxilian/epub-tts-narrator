"""Pipeline-mode RenderPool: declared TTS lanes, global caps, cancellation, close."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Condition, Event, Thread, get_ident
import threading
import unittest
from unittest.mock import patch

import epub_tts_narrator
from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.interrupts import RenderPaused, scoped_stop_condition
from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.recovery_client import RecoveryModelClient
from epub_tts_narrator.renderer.render_lanes import ResourceLane
from epub_tts_narrator.renderer.render_metrics import METRICS
from epub_tts_narrator.renderer.render_resource_context import resource_scope
from epub_tts_narrator.renderer.verification import (
    ReferenceCorroboratingVerifier, ReferenceTextJudgingVerifier,
)
from types import SimpleNamespace

# Import provenance: renderer tests must exercise this worktree's package.
assert Path(epub_tts_narrator.__file__).resolve().is_relative_to(
    Path(__file__).resolve().parents[1])


def pipeline_config(slots=8, review=None, backlog=None):
    config = deepcopy(DEFAULTS)
    config["render_pool"]["workers"] = [
        {"name": f"worker-{index}", "type": "remote", "multi_speaker": True}
        for index in range(slots)]
    config["render_pool"]["scheduler"] = "pipeline"
    if review is not None:
        config["render_pool"]["review_concurrency"] = review
    if backlog is not None:
        config["render_pool"]["synthesis_backlog"] = backlog
    return config


class GateEngine:
    """Event-controlled fake TTS engine tracking concurrency and calls."""

    identity = {"engine": "fixture-gate"}

    def __init__(self, state):
        self.state = state

    def synthesize(self, text, voice, output_path, *, seed, direction=""):
        return self._run(text)

    def synthesize_dialogue(self, turns, output_path, *, seed):
        return self._run("dialogue")

    def _run(self, text):
        state = self.state
        with state["condition"]:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["calls"].append((text, get_ident()))
            state["condition"].notify_all()
        if not state["gate"].wait(10):
            raise AssertionError("engine was never released")
        with state["condition"]:
            state["active"] -= 1
            state["condition"].notify_all()
        return {}

    def close(self):
        self.state["closed"] += 1


class TextlyBrokenEngine(GateEngine):
    def synthesize(self, text, voice, output_path, *, seed, direction=""):
        if text == "broken":
            raise RuntimeError("fixture engine failure")
        return super().synthesize(text, voice, output_path, seed=seed, direction=direction)


def gate_state():
    return {"condition": Condition(), "active": 0, "peak": 0, "calls": [],
            "gate": Event(), "closed": 0}


def wait_for(condition, predicate, timeout=10):
    with condition:
        if not condition.wait_for(predicate, timeout):
            raise AssertionError("expected pool state was never reached")


def wait_for_queued_lane_job(lane, minimum=1, timeout=10):
    """Deterministically wait for queued (not yet dequeued) lane work."""
    with lane._cond:
        if not lane._cond.wait_for(lambda: lane._pending >= minimum, timeout):
            raise AssertionError("queued lane job was never observed")


class PipelinePoolConstructionTests(unittest.TestCase):
    def test_legacy_default_pool_builds_no_resource_runtime(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local"}]
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        self.assertIsNone(pool.resources)

    def test_pipeline_pool_builds_declared_lanes_from_effective_scheduling(self):
        pool = RenderPool(pipeline_config(slots=8))
        self.addCleanup(pool.close)
        runtime = pool.resources
        self.assertIsNotNone(runtime)
        for lane in (runtime.tts, runtime.preparation, runtime.recognition,
                     runtime.review, runtime.review_coordination):
            self.assertIsInstance(lane, ResourceLane)
        self.assertEqual(len(runtime.tts._threads), 8)
        self.assertEqual(len(runtime.preparation._threads), 2)
        self.assertEqual(len(runtime.recognition._threads), 1)
        self.assertEqual(len(runtime.review._threads), 2)
        self.assertEqual(len(runtime.review_coordination._threads), 4)
        # Automatic synthesis backlog: min(256, 4 * 8 slots).
        self.assertEqual(runtime.tts._max_pending, 32)

    def test_explicit_backlog_and_review_concurrency_are_respected(self):
        pool = RenderPool(pipeline_config(slots=4, review=3, backlog=5))
        self.addCleanup(pool.close)
        runtime = pool.resources
        self.assertEqual(runtime.tts._max_pending, 5)
        self.assertEqual(len(runtime.review._threads), 3)
        self.assertEqual(len(runtime.review_coordination._threads), 6)

    def test_pipeline_scheduling_never_changes_pool_identity(self):
        legacy = RenderPool(pipeline_config(slots=4))
        self.addCleanup(legacy.close)
        legacy_config = pipeline_config(slots=4)
        legacy_config["render_pool"]["scheduler"] = "legacy"
        plain = RenderPool(legacy_config)
        self.addCleanup(plain.close)
        self.assertIsNone(plain.resources)
        self.assertEqual(plain.identity, legacy.identity)


class PipelineSynthesisLaneTests(unittest.TestCase):
    def setUp(self):
        self.state = gate_state()
        self.pool = RenderPool(pipeline_config(slots=4, backlog=8))
        self.addCleanup(self.pool.close)
        for worker in self.pool._workers:
            worker._engine = GateEngine(self.state)

    def test_synthesis_never_exceeds_declared_slots_and_reports_the_lane_stage(self):
        METRICS.reset()
        with ThreadPoolExecutor(max_workers=6) as callers:
            futures = [callers.submit(self.pool.synthesize, f"words {index}", {},
                                      f"unused-{index}", seed=index)
                       for index in range(6)]
            wait_for(self.state["condition"], lambda: self.state["active"] == 4)
            self.assertLessEqual(self.state["peak"], 4)
            self.state["gate"].set()
            for future in futures:
                future.result(10)
        wait_for(self.state["condition"], lambda: self.state["active"] == 0)
        self.assertLessEqual(self.state["peak"], 4)
        snapshot = METRICS.snapshot()
        self.assertEqual(snapshot["synthesis"]["active_count"], 6)
        self.assertEqual(snapshot["synthesis"]["outcomes"]["completed"], 6)
        # The lane owns the synthesis gauges; the callable must not double-count.
        self.assertEqual(snapshot["synthesis"]["active_now"], 0)

    def test_speaker_and_dialogue_paths_use_the_same_lane_contract(self):
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(self.pool.synthesize_dialogue, [], "unused", seed=1)
            second = callers.submit(self.pool.synthesize, "words", {}, "unused", seed=2)
            self.state["gate"].set()
            first.result(10)
            second.result(10)
        self.assertEqual(sorted(text for text, _ in self.state["calls"]),
                         ["dialogue", "words"])

    def test_engine_failures_propagate_and_leave_the_lane_usable(self):
        for worker in self.pool._workers:
            worker._engine = TextlyBrokenEngine(self.state)
        with self.assertRaisesRegex(RuntimeError, "fixture engine failure"):
            self.pool.synthesize("broken", {}, "unused", seed=1)
        self.state["gate"].set()
        repaired = self.pool.synthesize("words", {}, "unused", seed=2)
        self.assertEqual(repaired["worker_engine"], {"engine": "fixture-gate"})

    def test_synthesis_beyond_the_backlog_queues_bounded_without_hidden_work(self):
        # backlog 4: six callers -> four running, two queued, none rejected.
        pool = RenderPool(pipeline_config(slots=4, backlog=4))
        self.addCleanup(pool.close)
        state = gate_state()
        for worker in pool._workers:
            worker._engine = GateEngine(state)
        with ThreadPoolExecutor(max_workers=6) as callers:
            futures = [callers.submit(pool.synthesize, f"words {index}", {},
                                      f"unused-{index}", seed=index)
                       for index in range(6)]
            wait_for(state["condition"], lambda: state["active"] == 4)
            wait_for_queued_lane_job(pool.resources.tts, minimum=2)
            self.assertEqual(state["peak"], 4)
            state["gate"].set()
            for future in futures:
                future.result(10)
        self.assertEqual(len(state["calls"]), 6)


class PipelineCancellationBoundaryTests(unittest.TestCase):
    """Owner loss blocks new GPU starts but drains started-take verification."""

    def setUp(self):
        self.state = gate_state()
        self.pool = RenderPool(pipeline_config(slots=2, backlog=8))
        self.addCleanup(self.pool.close)
        for worker in self.pool._workers:
            worker._engine = GateEngine(self.state)

    def test_queued_synthesis_after_owner_loss_never_starts_the_engine(self):
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(self.pool.synthesize, "first", {}, "unused", seed=1)
            second = callers.submit(self.pool.synthesize, "second", {}, "unused", seed=2)
            wait_for(self.state["condition"], lambda: self.state["active"] == 2)

            expired, third_done = Event(), Event()
            failures = []

            def lost_owner():
                try:
                    with scoped_stop_condition(lambda: expired.is_set(),
                                               RenderPaused("owner lost")):
                        self.pool.synthesize("third", {}, "unused", seed=3)
                except BaseException as error:
                    failures.append(error)
                finally:
                    third_done.set()

            waiter = Thread(target=lost_owner)
            waiter.start()
            wait_for_queued_lane_job(self.pool.resources.tts, minimum=1)
            expired.set()
            self.state["gate"].set()
            first.result(10)
            second.result(10)
            self.assertTrue(third_done.wait(10))
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RenderPaused)
        self.assertEqual([text for text, _ in self.state["calls"]], ["first", "second"])
        waiter.join(5)

    def test_started_take_verification_drains_after_owner_loss(self):
        events = []
        recognizer = SimpleNamespace(
            identity={"backend": "fixture-asr"},
            transcribe=lambda path: events.append(("asr", get_ident())) or {"text": "words"})
        base = ReferenceCorroboratingVerifier(recognizer, recognizer)
        wrapped = self.pool.wrap_verifier(ReferenceTextJudgingVerifier(base, JevTranscriptJudge({})))
        with scoped_stop_condition(lambda: True, RenderPaused("owner lost")):
            self.assertEqual(wrapped.transcribe("unused"), {"text": "words"})
        self.assertEqual(len(events), 1)


class PipelineVerifierRoutingTests(unittest.TestCase):
    def setUp(self):
        self.state = gate_state()
        self.pool = RenderPool(pipeline_config(slots=2))
        self.addCleanup(self.pool.close)
        for worker in self.pool._workers:
            worker._engine = GateEngine(self.state)
        self.events = []
        events = self.events
        self.judge = JevTranscriptJudge({})
        self.reviewer = PronunciationReviewer(RecoveryModelClient({
            "model": "fixture", "base_url": "https://example.invalid/v1"}))

        class Recognizer:
            identity = {"backend": "fixture-asr"}

            def transcribe(self, path):
                events.append(("asr", get_ident(), threading.current_thread().name))
                return {"text": "words"}

        self.verifier = ReferenceTextJudgingVerifier(
            ReferenceCorroboratingVerifier(Recognizer(), Recognizer()),
            self.judge, self.reviewer)
        self.wrapped = self.pool.wrap_verifier(self.verifier)

    def blocker(self, name, started, release):
        def run(*args, **kwargs):
            self.events.append((name, get_ident(), threading.current_thread().name))
            started.set()
            if not release.wait(10):
                raise AssertionError(f"{name} was never released")
            return {"decision": "fixture"}
        return run

    def test_blocked_text_review_does_not_block_synthesis_or_recognition(self):
        started, release = Event(), Event()
        with patch.object(self.judge, "judge",
                          side_effect=self.blocker("review", started, release)):
            with ThreadPoolExecutor(max_workers=3) as callers:
                review = callers.submit(self.wrapped.judge_reference, "expected", "heard")
                self.assertTrue(started.wait(5))
                try:
                    self.state["gate"].set()
                    generated = callers.submit(
                        self.pool.synthesize, "words", {}, "unused", seed=1).result(10)
                    self.assertEqual(generated["worker_engine"], {"engine": "fixture-gate"})
                    self.assertEqual(callers.submit(self.wrapped.transcribe, "unused").result(10),
                                     {"text": "words"})
                    self.assertFalse(review.done())
                finally:
                    release.set()
                self.assertEqual(review.result(10), {"decision": "fixture"})
        asr_threads = {thread for name, thread, _ in self.events if name == "asr"}
        review_threads = {thread for name, thread, _ in self.events if name == "review"}
        self.assertEqual(len(asr_threads), 1)
        self.assertEqual(len(review_threads), 1)
        self.assertTrue(asr_threads.isdisjoint(review_threads))

    def test_pipeline_recognition_metrics_are_lane_owned_once(self):
        METRICS.reset()
        self.state["gate"].set()
        for _ in range(3):
            self.assertEqual(self.wrapped.transcribe("unused"), {"text": "words"})
        snapshot = METRICS.snapshot()
        self.assertEqual(snapshot["recognition"]["active_count"], 3)
        self.assertEqual(snapshot["recognition"]["active_now"], 0)
        self.assertEqual(snapshot["model_thread"]["active_count"], 3)


class PipelinePoolCloseTests(unittest.TestCase):
    def test_close_cancels_queued_synthesis_drains_and_closes_engines_once(self):
        state = gate_state()
        pool = RenderPool(pipeline_config(slots=1, backlog=4))
        self.addCleanup(pool.close)
        for worker in pool._workers:
            worker._engine = GateEngine(state)
        closing, running_done, queued_done = Event(), Event(), Event()
        queued_failures = []
        running_result = []

        def running_caller():
            running_result.append(pool.synthesize("first", {}, "unused", seed=1))
            running_done.set()

        def queued_caller():
            try:
                pool.synthesize("second", {}, "unused", seed=2)
            except BaseException as error:
                queued_failures.append(error)
            finally:
                queued_done.set()

        def run_close():
            closing.wait(10)
            pool.close()

        running = Thread(target=running_caller)
        queued = Thread(target=queued_caller)
        closer = Thread(target=run_close)
        running.start()
        wait_for(state["condition"], lambda: state["active"] == 1)
        queued.start()
        wait_for_queued_lane_job(pool.resources.tts, minimum=1)
        try:
            closer.start()
            closing.set()
            # Close cancels queued work before joining the running synthesis.
            self.assertTrue(queued_done.wait(10))
            state["gate"].set()
            self.assertTrue(running_done.wait(10))
        finally:
            running.join(10)
            queued.join(10)
            closer.join(10)
        self.assertEqual(len(running_result), 1)
        self.assertEqual(len(queued_failures), 1)
        self.assertIsInstance(queued_failures[0], CancelledError)
        self.assertTrue(pool._closed)
        self.assertEqual(state["closed"], 1)
        self.assertEqual([text for text, _ in state["calls"]], ["first"])
        remaining = {thread.name for thread in threading.enumerate()
                     if thread.name.startswith("render-")}
        self.assertEqual(remaining, set(), "resource lane threads leaked past pool close")
        with self.assertRaises(EngineError):
            pool.synthesize("third", {}, "unused", seed=3)


if __name__ == "__main__":
    unittest.main()
