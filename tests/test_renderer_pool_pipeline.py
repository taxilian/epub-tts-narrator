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
    def test_explicit_legacy_pool_builds_no_resource_runtime(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local"}]
        config["render_pool"]["scheduler"] = "legacy"
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        self.assertIsNone(pool.resources)

    def test_default_pool_uses_the_pipeline_resource_runtime(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local"}]
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        self.assertIsNotNone(pool.resources)
        self.assertEqual(len(pool.resources.tts._threads), 1)

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

    def test_chapter_registration_exposes_real_scheduling_snapshot(self):
        from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
        from epub_tts_narrator.renderer.render_schedule_status import clear_runtime, scheduling_status

        config = pipeline_config(slots=2)
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        executor = ChapterExecutor(config, engine=pool, verifier=object())
        self.addCleanup(executor.close)
        self.addCleanup(clear_runtime, pool)

        executor._components()
        status = scheduling_status()

        self.assertEqual(status["mode"], "pipeline")
        self.assertEqual(status["capacity"]["tts_slots"], 2)
        self.assertEqual(status["logical"]["global_limit"], pool.max_inflight)
        self.assertIsInstance(status["backlog"], dict)

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


class PipelineFifoDispatchTests(unittest.TestCase):
    """TTS takes start in submission order on the first available workers."""

    def test_queued_take_uses_the_idle_worker_without_waiting_for_a_named_worker(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity

        pool = RenderPool(pipeline_config(slots=2, backlog=8))
        self.addCleanup(pool.close)
        busy, idle = gate_state(), gate_state()
        pool._workers[0]._engine = GateEngine(busy)
        pool._workers[1]._engine = GateEngine(idle)
        first, second, other = (ChapterContinuity({"engine": "fixture"}, {}) for _ in range(3))
        request = {"kind": "speech", "voice_asset_hash": "a" * 64, "chunk": {"segment": {"role_id": "r", "voice_id": "v"}}}

        def take(chapter, text):
            # Production lane owners are per chapter (run, chapter) keys.
            with resource_scope(pool.resources, ("run", text)), chapter.transaction(request):
                return pool.synthesize(text, {}, "unused", seed=1)

        with ThreadPoolExecutor(max_workers=3) as callers:
            running = callers.submit(take, first, "first")
            wait_for(busy["condition"], lambda: busy["active"] == 1)
            queued = callers.submit(take, second, "second")
            wait_for(idle["condition"], lambda: idle["active"] == 1)
            waiting = callers.submit(take, other, "other")
            wait_for_queued_lane_job(pool.resources.tts, minimum=1)
            self.assertEqual([text for text, _ in busy["calls"]], ["first"])
            self.assertEqual([text for text, _ in idle["calls"]], ["second"])
            idle["gate"].set()
            self.assertEqual(queued.result(10)["render_worker"], "worker-1")
            self.assertEqual(waiting.result(10)["render_worker"], "worker-1")
            busy["gate"].set()
            self.assertEqual(running.result(10)["render_worker"], "worker-0")
            self.assertEqual([text for text, _ in idle["calls"]], ["second", "other"])
        self.assertEqual(pool._slots.qsize(), 2)

    def test_concurrent_chapters_never_share_a_slot_and_every_take_completes(self):
        import random
        import time
        from epub_tts_narrator.renderer.continuity import ChapterContinuity

        pool = RenderPool(pipeline_config(slots=2, backlog=16))
        self.addCleanup(pool.close)
        lock = threading.Lock()
        active = {"worker-0": 0, "worker-1": 0}
        peak = dict(active)
        ran = []

        class Recording:
            identity = {"engine": "fixture-stress"}

            def __init__(self, name):
                self.name = name

            def synthesize(self, text, voice, output_path, *, seed, direction=""):
                with lock:
                    active[self.name] += 1
                    peak[self.name] = max(peak[self.name], active[self.name])
                time.sleep(random.uniform(0, 0.003))
                with lock:
                    active[self.name] -= 1
                    ran.append((text, self.name))
                return {}

        for worker in pool._workers:
            worker._engine = Recording(worker.settings["name"])
        chapters = [ChapterContinuity({"engine": "fixture"}, {}) for _ in range(8)]
        request = {"kind": "speech", "voice_asset_hash": "a" * 64, "chunk": {"segment": {"role_id": "r", "voice_id": "v"}}}

        def chapter_takes(index):
            with resource_scope(pool.resources, ("run", index)):
                for take in range(25):
                    with chapters[index].transaction(request):
                        pool.synthesize(f"{index}:{take}", {}, "unused", seed=take)

        with ThreadPoolExecutor(max_workers=8) as callers:
            list(callers.map(chapter_takes, range(8), timeout=60))
        # No slot is used twice at once, no take is lost, and every slot returns.
        self.assertEqual(peak, {"worker-0": 1, "worker-1": 1})
        self.assertEqual(len(ran), 200)
        self.assertEqual(pool._slots.qsize(), 2)
        self.assertEqual(pool.resources.tts._pending, 0)

    def test_waiting_take_does_not_name_a_preferred_worker(self):
        from epub_tts_narrator.renderer.continuity import ChapterContinuity
        from epub_tts_narrator.renderer.progress import generation_activity

        pool = RenderPool(pipeline_config(slots=2, backlog=8))
        self.addCleanup(pool.close)
        state = gate_state()
        state["gate"].set()
        for worker in pool._workers:
            worker._engine = GateEngine(state)
        chapter = ChapterContinuity({"engine": "fixture"}, {})
        events = []

        class Notify:
            def worker_activity(self, event, worker=None, elapsed=None):
                events.append((event, worker))

        with patch("epub_tts_narrator.renderer.progress._TaskNotifier", Notify), \
             generation_activity(Notify()), \
             chapter.transaction({"kind": "speech", "voice_asset_hash": "a" * 64, "chunk": {"segment": {"role_id": "r", "voice_id": "v"}}}):
            pool.synthesize("words", {}, "unused", seed=1)
        self.assertEqual(events[0], ("waiting", None))


class PipelineRecognitionDispatchTests(unittest.TestCase):
    def test_recognition_calls_start_in_fifo_order_across_chapter_owners(self):
        pool = RenderPool(pipeline_config(slots=2))
        self.addCleanup(pool.close)
        release = Event()
        started = []
        condition = Condition()

        def recognize(label):
            with condition:
                started.append(label)
                condition.notify_all()
            if label == "first" and not release.wait(10):
                raise AssertionError("first recognition was never released")
            return {"text": label}

        def take(owner, label):
            with resource_scope(pool.resources, ("run", owner)):
                return pool._recognize("transcribe", recognize, (label,), {}, initialize_mlx=False)

        with ThreadPoolExecutor(max_workers=6) as callers:
            first = callers.submit(take, "a", "first")
            wait_for(condition, lambda: started == ["first"])
            pending = []
            for index, (owner, label) in enumerate((
                    ("b", "second"), ("c", "third"), ("b", "fourth"),
                    ("b", "fifth"), ("c", "sixth")), 1):
                pending.append(callers.submit(take, owner, label))
                wait_for_queued_lane_job(pool.resources.recognition, minimum=index)
            release.set()
            first.result(10)
            for future in pending:
                future.result(10)
        self.assertEqual(started, ["first", "second", "third", "fourth", "fifth", "sixth"])


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
        # Recognition runs on its own thread, never on the synthesis model thread.
        self.assertEqual(snapshot["model_thread"]["active_count"], 0)

    def test_blocked_reference_sound_review_does_not_block_synthesis_or_recognition(self):
        started, release = Event(), Event()
        with patch.object(self.reviewer, "review_reference",
                          side_effect=self.blocker("reference-review", started, release)):
            with ThreadPoolExecutor(max_workers=3) as callers:
                review = callers.submit(self.wrapped.review_reference_phonetics, "expected", "heard")
                self.assertTrue(started.wait(5))
                try:
                    self.state["gate"].set()
                    generated = callers.submit(self.pool.synthesize, "words", {}, "unused", seed=1).result(10)
                    self.assertEqual(generated["worker_engine"], {"engine": "fixture-gate"})
                    self.assertEqual(callers.submit(self.wrapped.transcribe, "unused").result(10), {"text": "words"})
                    self.assertFalse(review.done())
                finally:
                    release.set()
                self.assertEqual(review.result(10), {"decision": "fixture"})


class PipelinePoolCloseTests(unittest.TestCase):
    def test_close_cancels_queued_synthesis_drains_and_closes_engines_once(self):
        state = gate_state()
        # Only this pool's threads matter; an unrelated pool elsewhere in the
        # process must not mask or fake a leak here.
        before = set(threading.enumerate())
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
        remaining = {thread.name for thread in set(threading.enumerate()) - before
                     if thread.name.startswith("render-")}
        self.assertEqual(remaining, set(), "resource lane threads leaked past pool close")
        with self.assertRaises(EngineError):
            pool.synthesize("third", {}, "unused", seed=3)


if __name__ == "__main__":
    unittest.main()
