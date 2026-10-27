"""Bounded, payload-free renderer resource-stage metrics."""

from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import render_metrics
from epub_tts_narrator.renderer.interrupts import RenderPaused
from epub_tts_narrator.renderer.local_runtime import LocalRuntime
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.progress import ChapterProgress


PAYLOAD_MARKER = "DO NOT ECHO: chapter text and api_key=super-secret"


def pool_config(*workers, max_inflight=0):
    return {"engine": {"type": "drama3_mlx"},
            "render_pool": {"workers": list(workers), "max_inflight": max_inflight}}


def worker(name, concurrency=1, local=False):
    return {"name": name, "type": "local" if local else "remote",
            "concurrency": concurrency}


class FakeEngine:
    """Offline TTS boundary double; identity fields satisfy pool result wrapping."""

    identity = {"engine": "fake", "model": "fake-model"}

    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def synthesize(self, text, voice, output_path, *, seed, direction=""):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"seconds": 1.0, "sample_rate": 16000}


class FakeVerifier:
    identity = {"verifier": "fake"}

    def check(self, value):
        return value


class RenderMetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = render_metrics.RenderMetrics()

    def test_unknown_stage_names_are_rejected_instead_of_growing_unbounded_labels(self):
        for bad in ("clip:chunk-17", "synthesis/worker-gpu-3", "", "user supplied", None, 7):
            with self.subTest(stage=bad):
                with self.assertRaises(ValueError):
                    self.metrics.operation(bad)

    def test_completed_failed_and_cancelled_operations_balance_enqueued_counts(self):
        done = self.metrics.operation("synthesis")
        done.start()
        done.end("completed")
        failed = self.metrics.operation("synthesis")
        failed.start()
        try:
            raise ValueError("synthesis rejected the take")
        except ValueError as error:
            failed.end(render_metrics.classify_outcome(error))
        cancelled = self.metrics.operation("synthesis")
        cancelled.start()
        cancelled.cancel()
        stage = self.metrics.snapshot()["synthesis"]
        self.assertEqual(stage["outcomes"], {"completed": 1, "failed": 1, "cancelled": 1})
        self.assertEqual(stage["queued_now"], 0)
        self.assertEqual(stage["active_now"], 0)
        self.assertEqual(stage["active_count"], 3)
        self.assertEqual(stage["wait_count"], 3)

    def test_cancelled_queued_work_never_counts_active_time(self):
        queued = self.metrics.operation("synthesis_slot")
        queued.cancel()
        stage = self.metrics.snapshot()["synthesis_slot"]
        self.assertEqual(stage["outcomes"], {"completed": 0, "failed": 0, "cancelled": 1})
        self.assertEqual(stage["active_count"], 0)
        self.assertEqual(stage["active_seconds"], 0.0)
        self.assertEqual(stage["active_now"], 0)
        self.assertEqual(stage["queued_now"], 0)
        self.assertEqual(stage["wait_count"], 1)

    def test_operations_cannot_be_ended_twice_or_started_after_end(self):
        operation = self.metrics.operation("synthesis")
        operation.start()
        operation.end("completed")
        with self.assertRaises(RuntimeError):
            operation.end("completed")
        never = self.metrics.operation("synthesis")
        never.start()
        never.end("cancelled")
        with self.assertRaises(RuntimeError):
            never.start()

    def test_context_managers_classify_outcomes_and_restore_gauges(self):
        with self.metrics.active("recognition"):
            pass
        with self.assertRaises(ValueError):
            with self.metrics.active("recognition"):
                raise ValueError("acoustic gate rejected the take")
        try:
            with self.metrics.active("review"):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        try:
            with self.metrics.active("review"):
                raise RenderPaused
        except RenderPaused:
            pass
        with self.metrics.waiting("admission"):
            pass
        try:
            with self.metrics.waiting("synthesis_slot"):
                raise RuntimeError("pool closed while waiting")
        except RuntimeError:
            pass
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["recognition"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["recognition"]["outcomes"]["failed"], 1)
        self.assertEqual(snapshot["recognition"]["active_now"], 0)
        self.assertEqual(snapshot["review"]["outcomes"]["cancelled"], 2)
        self.assertEqual(snapshot["review"]["active_now"], 0)
        self.assertEqual(snapshot["admission"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["synthesis_slot"]["outcomes"]["failed"], 1)
        self.assertEqual(snapshot["synthesis_slot"]["queued_now"], 0)

    def test_wait_and_active_durations_land_in_bounded_histogram_buckets(self):
        with self.metrics.waiting("synthesis_slot"):
            time.sleep(0.02)
        with self.metrics.active("synthesis"):
            time.sleep(0.02)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["synthesis_slot"]["wait_buckets"]["10_100ms"], 1)
        self.assertEqual(snapshot["synthesis"]["active_buckets"]["10_100ms"], 1)
        self.assertEqual(sum(snapshot["synthesis_slot"]["wait_buckets"].values()),
                         snapshot["synthesis_slot"]["wait_count"])
        self.assertEqual(sum(snapshot["synthesis"]["active_buckets"].values()),
                         snapshot["synthesis"]["active_count"])

    def test_repeated_operations_do_not_grow_snapshot_shape(self):
        stage_keys = {"queued_now", "active_now", "wait_seconds", "wait_count",
                      "active_seconds", "active_count", "wait_buckets",
                      "active_buckets", "outcomes"}
        for _ in range(500):
            with self.metrics.active("synthesis"):
                pass
        first = self.metrics.snapshot()
        for _ in range(500):
            with self.metrics.active("synthesis"):
                pass
        second = self.metrics.snapshot()
        self.assertEqual(list(first), list(second))
        self.assertEqual(set(second["synthesis"]), stage_keys)
        self.assertEqual(second["synthesis"]["outcomes"]["completed"], 1000)
        self.assertEqual(set(second["synthesis"]["wait_buckets"]), set(render_metrics.BUCKET_NAMES))

    def test_snapshots_are_json_serializable_counter_reports(self):
        with self.metrics.active("synthesis"):
            pass
        encoded = json.dumps(self.metrics.snapshot(), sort_keys=True)
        self.assertIn('"synthesis"', encoded)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["synthesis"]["outcomes"]["completed"], 1)
        self.assertEqual(decoded["gpu_telemetry"], "unavailable")

    def test_failure_payloads_never_appear_in_snapshots(self):
        try:
            with self.metrics.active("synthesis"):
                raise ValueError(PAYLOAD_MARKER)
        except ValueError:
            pass
        try:
            with self.metrics.waiting("synthesis_slot"):
                raise RuntimeError(PAYLOAD_MARKER)
        except RuntimeError:
            pass
        self.metrics.operation("clip").cancel()
        encoded = json.dumps(self.metrics.snapshot())
        self.assertNotIn("DO NOT ECHO", encoded)
        self.assertNotIn("super-secret", encoded)
        self.assertNotIn("api_key", encoded)

    def test_snapshots_can_be_reset_by_tests_without_restarting_the_process(self):
        with self.metrics.active("synthesis"):
            pass
        self.metrics.reset()
        stage = self.metrics.snapshot()["synthesis"]
        self.assertEqual(stage["outcomes"]["completed"], 0)
        self.assertEqual(stage["active_count"], 0)

    def test_concurrent_operations_keep_thread_safe_balanced_counters(self):
        barrier = threading.Barrier(8, timeout=10)

        def run(index):
            barrier.wait()
            try:
                with self.metrics.active("synthesis"):
                    if index % 2:
                        raise ValueError("half the workers fail")
            except ValueError:
                pass
            with self.metrics.waiting("synthesis_slot"):
                time.sleep(0.001)

        threads = [threading.Thread(target=run, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["synthesis"]["outcomes"]["completed"], 4)
        self.assertEqual(snapshot["synthesis"]["outcomes"]["failed"], 4)
        self.assertEqual(snapshot["synthesis"]["active_now"], 0)
        self.assertEqual(snapshot["synthesis_slot"]["outcomes"]["completed"], 8)


class LocalRuntimeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = render_metrics.METRICS
        self.metrics.reset()
        self.addCleanup(self.metrics.reset)

    def test_model_thread_stage_records_queue_wait_and_active_execution(self):
        runtime = LocalRuntime(metrics_stage="model_thread")
        gate, released, result = threading.Event(), threading.Event(), {}

        def blocker():
            gate.set()
            self.assertTrue(released.wait(10))

        holder = threading.Thread(target=lambda: runtime.run(blocker), daemon=True)
        holder.start()
        self.assertTrue(gate.wait(10))
        self.assertEqual(self.metrics.snapshot()["model_thread"]["active_now"], 1)

        def submit_second():
            result["value"] = runtime.run(lambda: "ok")

        second = threading.Thread(target=submit_second, daemon=True)
        second.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.metrics.snapshot()["model_thread"]["queued_now"] == 1:
                break
            time.sleep(0.005)
        self.assertEqual(self.metrics.snapshot()["model_thread"]["queued_now"], 1)
        released.set()
        second.join(10)
        holder.join(10)
        self.assertEqual(result.get("value"), "ok")
        stage = self.metrics.snapshot()["model_thread"]
        self.assertEqual(stage["outcomes"], {"completed": 2, "failed": 0, "cancelled": 0})
        self.assertEqual(stage["queued_now"], 0)
        self.assertEqual(stage["active_now"], 0)
        self.assertEqual(stage["wait_count"], 2)
        self.assertEqual(stage["active_count"], 2)

    def test_failing_model_thread_operations_count_failed_and_restore_gauges(self):
        runtime = LocalRuntime(metrics_stage="model_thread")
        with self.assertRaises(ValueError):
            runtime.run(self._raise_payload_failure)
        stage = self.metrics.snapshot()["model_thread"]
        self.assertEqual(stage["outcomes"]["failed"], 1)
        self.assertEqual(stage["active_now"], 0)
        self.assertEqual(stage["queued_now"], 0)
        self.assertNotIn("DO NOT ECHO", json.dumps(self.metrics.snapshot()))

    def test_runtime_without_a_metrics_stage_stays_uninstrumented(self):
        runtime = LocalRuntime()
        self.assertEqual(runtime.run(lambda: "ok"), "ok")
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["model_thread"]["outcomes"]["completed"], 0)

    @staticmethod
    def _raise_payload_failure():
        raise ValueError(PAYLOAD_MARKER)


class PoolInstrumentationTests(unittest.TestCase):
    def setUp(self):
        self.metrics = render_metrics.METRICS
        self.metrics.reset()
        self.addCleanup(self.metrics.reset)

    def test_pool_transaction_records_mutually_consistent_stage_counters(self):
        pool = RenderPool(pool_config(worker("local", concurrency=2, local=True)))
        pool._local_runtime = LocalRuntime(metrics_stage="model_thread")
        pool._workers[0]._engine = FakeEngine()
        with pool.admit_chunk():
            with self.metrics.active("clip"):
                metadata = pool._execute("synthesize", "unused", "voice", "unused.wav", seed=1)
        self.assertEqual(metadata["render_worker"], "local")
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["synthesis"]["outcomes"],
                         {"completed": 1, "failed": 0, "cancelled": 0})
        self.assertEqual(snapshot["synthesis_slot"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["synthesis_slot"]["wait_count"], 1)
        self.assertEqual(snapshot["model_thread"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["clip"]["outcomes"]["completed"], 1)
        for stage in ("synthesis", "synthesis_slot", "model_thread", "clip"):
            self.assertEqual(snapshot[stage]["active_now"], 0, stage)
            self.assertEqual(snapshot[stage]["queued_now"], 0, stage)

    def test_rejected_synthesis_counts_failed_and_restores_the_slot(self):
        pool = RenderPool(pool_config(worker("local", local=True)))
        pool._local_runtime = LocalRuntime(metrics_stage="model_thread")
        pool._workers[0]._engine = FakeEngine(error=ValueError(PAYLOAD_MARKER))
        slots_before = pool._slots.qsize()
        with self.assertRaises(ValueError):
            pool._execute("synthesize", "unused", "voice", "unused.wav", seed=1)
        self.assertEqual(pool._slots.qsize(), slots_before)
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["synthesis"]["outcomes"]["failed"], 1)
        self.assertEqual(snapshot["synthesis"]["outcomes"]["completed"], 0)
        self.assertEqual(snapshot["synthesis"]["active_now"], 0)
        self.assertEqual(snapshot["synthesis_slot"]["outcomes"]["completed"], 1)
        self.assertNotIn("DO NOT ECHO", json.dumps(snapshot))

    def test_admission_wait_is_recorded_while_the_window_is_saturated(self):
        pool = RenderPool(pool_config(worker("local", local=True), max_inflight=1))
        pool._local_runtime = LocalRuntime(metrics_stage="model_thread")
        pool._workers[0]._engine = FakeEngine()
        entered, release = threading.Event(), threading.Event()

        def hold():
            with pool.admit_chunk():
                entered.set()
                self.assertTrue(release.wait(10))

        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(entered.wait(10))
        with pool.admit_chunk():
            pass
        release.set()
        holder.join(10)
        admission = self.metrics.snapshot()["admission"]
        self.assertEqual(admission["wait_count"], 2)
        self.assertEqual(admission["outcomes"]["completed"], 2)
        self.assertEqual(admission["queued_now"], 0)

    def test_wrapped_verifier_records_recognition_stage_on_the_inference_lane(self):
        pool = RenderPool(pool_config(worker("local", local=True)))
        pool._local_runtime = LocalRuntime(metrics_stage="model_thread")
        verifier = pool.wrap_verifier(FakeVerifier())
        self.assertEqual(verifier.check("value"), "value")
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["recognition"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["recognition"]["active_now"], 0)
        self.assertEqual(snapshot["model_thread"]["outcomes"]["completed"], 1)

    def test_text_review_lane_records_review_stage_without_the_inference_lane(self):
        pool = RenderPool(pool_config(worker("local", local=True)))
        pool._local_runtime = LocalRuntime(metrics_stage="model_thread")
        verifier = pool.wrap_verifier(FakeVerifier())
        with patch("epub_tts_narrator.renderer.pool._text_review_method", return_value=True):
            self.assertEqual(verifier.check("value"), "value")
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["review"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["recognition"]["outcomes"].get("completed", 0), 0)
        self.assertEqual(snapshot["model_thread"]["outcomes"].get("completed", 0), 0)


class ChapterProgressMetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = render_metrics.METRICS
        self.metrics.reset()
        self.addCleanup(self.metrics.reset)

    @staticmethod
    def chunks(count=3):
        return [{"chunk_id": f"c{index}", "spoken_text": "words", "silent_control": False}
                for index in range(count)]

    def test_chapter_clip_window_and_clip_transactions_balance_counters(self):
        with ChapterProgress(1, 1, self.chunks(), notify=lambda _: None) as progress:
            with progress.task(0):
                pass
            with progress.task(1):
                pass
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["chapter_clips"]["outcomes"]["completed"], 1)
        self.assertEqual(snapshot["clip"]["outcomes"]["completed"], 2)
        for stage in ("chapter_clips", "clip"):
            self.assertEqual(snapshot[stage]["active_now"], 0, stage)
            self.assertEqual(snapshot[stage]["queued_now"], 0, stage)

    def test_interrupted_chapter_counts_cancelled_clips_and_chapter_windows(self):
        try:
            with ChapterProgress(1, 1, self.chunks(), notify=lambda _: None) as progress:
                with progress.task(0):
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        snapshot = self.metrics.snapshot()
        self.assertEqual(snapshot["clip"]["outcomes"]["cancelled"], 1)
        self.assertEqual(snapshot["clip"]["outcomes"]["completed"], 0)
        self.assertEqual(snapshot["clip"]["active_now"], 0)
        self.assertEqual(snapshot["chapter_clips"]["outcomes"]["cancelled"], 1)

    def test_progress_metrics_snapshot_never_carries_block_text(self):
        with ChapterProgress(1, 1, self.chunks(), notify=lambda _: None) as progress:
            with progress.task(0):
                pass
        encoded = json.dumps(self.metrics.snapshot())
        self.assertNotIn("words", encoded)
        self.assertNotIn("chunk", encoded)


class StageCatalogTests(unittest.TestCase):
    def test_stage_catalog_stays_payload_free_and_closed(self):
        for stage in render_metrics.STAGES:
            self.assertIsInstance(stage, str)
            self.assertTrue(stage.isidentifier(), stage)
        for expected in ("synthesis", "synthesis_slot", "recognition", "review",
                         "admission", "model_thread", "chapter", "chapter_clips", "clip"):
            self.assertIn(expected, render_metrics.STAGES)
