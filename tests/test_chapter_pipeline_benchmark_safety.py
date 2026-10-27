"""Fail-closed safety and lane-runtime checks for benchmark runs.

Companion to test_chapter_pipeline_benchmark_run.py: tampered references or
passages, manifest-less outputs, and the pipeline scheduler's real lane
runtime must all behave correctly before any measured run is trusted.
"""

import json
import subprocess
import unittest

from scripts.chapter_benchmark_fakeengine import LEAF_ENGINES
from scripts.chapter_benchmark_corpus import load_corpus
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.render_pipeline import chapter_share
from test_renderer_workflow import FFMPEG, FFPROBE
from test_chapter_pipeline_benchmark_run import BenchmarkFixture


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class RunSafetyTests(BenchmarkFixture):
    def test_tampered_reference_audio_fails_closed_before_any_submission(self):
        flac = next((self.corpus / "remote-voice-assets").glob("*.flac"))
        flac.write_bytes(flac.read_bytes() + b"tampered")
        self.expect_failure("run", str(self.frozen), "--config", str(self.config_path),
                            "--scheduler", "legacy", "--output", str(self.root / "run"), "--fake-engine")
        self.assertFalse((self.root / "run" / "book" / "book.m4b").exists())

    def test_frozen_passage_tampering_fails_the_corpus_checksum(self):
        path = self.frozen
        body = read_json(path)
        body["cases"][0]["chunk"]["spoken_text"] += " altered"
        write_json(path, body)
        self.expect_failure("run", str(path), "--config", str(self.config_path),
                            "--scheduler", "legacy", "--output", str(self.root / "run"), "--fake-engine")

    def test_existing_output_without_its_manifest_is_never_reused(self):
        output = self.root / "run"
        output.mkdir(parents=True)
        (output / "stray.txt").write_text("leftover")
        self.expect_failure("run", str(self.frozen), "--config", str(self.config_path),
                            "--scheduler", "legacy", "--output", str(output), "--fake-engine")

    def test_pipeline_scheduler_mode_drives_lane_runtime_and_stays_ineligible(self):
        observed = self.observe_pool()
        self.fresh_fake_registry()
        self.run_main("run", str(self.frozen), "--config", str(self.config_path),
                      "--scheduler", "pipeline", "--output", str(self.root / "run"), "--fake-engine")
        report = self.assert_valid_m4b(self.root / "run")
        self.assertEqual(report["scheduler"], "pipeline")
        self.assertEqual(report["effective_scheduling"]["scheduler"], "pipeline")
        self.assertIs(report["fake_engine"], True)
        self.assertFalse(report["speed_gate_eligible"])
        self.assertEqual(report["eligibility_reason"], "fake_engine")
        self.assertIsNone(report["accepted_words_per_active_second"])
        # Pipeline mode builds the real lane runtime on the real pool object,
        # and every capacity agrees with the effective configuration: the
        # coordinator count is the configured/auto chapter concurrency (4
        # here, never the 16 safety ceiling) and the logical, admission, and
        # lane-queue bounds are the same effective local-auto window.
        self.assertEqual(len(observed), 1)
        pool = observed[0]
        self.assertIsInstance(pool, RenderPool)
        self.assertIsNotNone(pool.resources)
        self.assertIsNotNone(pool.logical)
        effective = report["effective_scheduling"]
        self.assertEqual(effective["scheduler"], "pipeline")
        self.assertEqual(report["parallel_chapters"], effective["chapter_concurrency"])
        self.assertEqual(report["parallel_chapters"], 4)
        self.assertEqual(pool.max_inflight, effective["max_inflight"])
        self.assertEqual(pool.max_inflight, 16)
        self.assertEqual(pool.logical.global_limit, effective["max_inflight"])
        self.assertEqual(pool.logical.share,
                         chapter_share(pool.logical.global_limit, effective["chapter_concurrency"]))
        self.assertEqual(pool.resources.recognition._max_pending, pool.max_inflight)
        self.assertEqual(len(pool.resources.tts._threads), pool.max_parallel)
        self.assertEqual(len(LEAF_ENGINES), 1)
        self.assert_clean_teardown()


def _stage_delta(report, window, stage, field):
    return report["telemetry"][window][stage][field]


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class TelemetryReportTests(BenchmarkFixture):
    """Active-window stage metrics must be real, separated, and redacted."""

    def complete_pipeline_run_report(self):
        self.fresh_fake_registry()
        self.run_main("run", str(self.frozen), "--config", str(self.config_path),
                      "--scheduler", "pipeline", "--output", str(self.root / "run"), "--fake-engine")
        return self.assert_valid_m4b(self.root / "run")

    def test_active_window_stage_metrics_are_real_separated_and_balanced(self):
        report = self.complete_pipeline_run_report()
        telemetry = report["telemetry"]
        # Real overlap counters, not an empty structural schema: the active
        # window shows the measured 8 clips plus per-clip preparation and
        # recognition work, distinctly from the single warmup clip.
        self.assertGreaterEqual(_stage_delta(report, "active_window", "synthesis", "active_count"), 8)
        self.assertGreaterEqual(_stage_delta(report, "active_window", "synthesis", "outcomes")["completed"], 8)
        self.assertGreaterEqual(_stage_delta(report, "active_window", "recognition", "active_count"), 8)
        self.assertGreaterEqual(_stage_delta(report, "active_window", "preparation", "active_count"), 8)
        warm = telemetry["warmup_window"]["synthesis"]["outcomes"]["completed"]
        self.assertEqual(warm, 1)  # warmup must not contaminate the active window
        self.assertLessEqual(_stage_delta(report, "warmup_window", "synthesis", "active_count"), 1)
        # Queue balance at window close: nothing queued or active remains.
        for stage in ("synthesis", "recognition", "preparation"):
            self.assertEqual(telemetry["active_window"][stage]["queued_now"], 0, stage)
            self.assertEqual(telemetry["active_window"][stage]["active_now"], 0, stage)

    def test_resource_lanes_backlog_and_limits_are_captured_while_pool_alive(self):
        report = self.complete_pipeline_run_report()
        telemetry = report["telemetry"]
        self.assertEqual(telemetry["runtime_limits"]["max_inflight"], 16)
        self.assertEqual(telemetry["runtime_limits"]["max_parallel"], 1)
        self.assertEqual(telemetry["logical"]["global_limit"], 16)
        self.assertEqual(telemetry["resource_lanes"]["tts"]["workers"], 1)
        self.assertEqual(telemetry["resource_lanes"]["recognition"]["max_pending"], 16)
        backlog = telemetry["backlog"]
        self.assertIsInstance(backlog, dict)
        for key in ("pending_count", "materialized_bytes", "max_pending", "max_materialized_bytes"):
            self.assertIsInstance(backlog[key], int)
        # Peak gauges arrive with the core's backlog work; until then the
        # report says pending integration instead of inventing zeros.
        marker = telemetry["backlog_peak_integration"]
        if any(key.startswith("peak_") for key in backlog):
            self.assertEqual(marker, "captured")
        else:
            self.assertEqual(marker, "pending")

    def test_telemetry_is_numeric_bounded_and_secret_free(self):
        report = self.complete_pipeline_run_report()
        dumped = json.dumps(report["telemetry"], sort_keys=True)
        self.assertNotIn("api_key", dumped.lower())
        corpus, _ = load_corpus(self.frozen)
        passage = corpus["cases"][0]["chunk"]["spoken_text"][:24]
        self.assertNotIn(passage, dumped)
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            else:
                self.assertIsInstance(value, (int, float, str))
                if isinstance(value, str):
                    self.assertIn(value, {"unavailable", "captured", "pending"})
        walk(report["telemetry"])

    def test_legacy_mode_reports_lanes_and_backlog_as_unavailable_not_zero(self):
        self.fresh_fake_registry()
        self.run_main("run", str(self.frozen), "--config", str(self.config_path),
                      "--scheduler", "legacy", "--output", str(self.root / "run"), "--fake-engine")
        report = self.assert_valid_m4b(self.root / "run")
        telemetry = report["telemetry"]
        self.assertEqual(telemetry["resource_lanes"], "unavailable")
        self.assertEqual(telemetry["backlog"], "unavailable")
        self.assertEqual(telemetry["logical"], "unavailable")
        # Legacy still records real stage aggregates through the pool paths.
        self.assertGreaterEqual(_stage_delta(report, "active_window", "synthesis", "active_count"), 8)
        self.assertGreaterEqual(_stage_delta(report, "active_window", "recognition", "active_count"), 8)


if __name__ == "__main__":
    unittest.main()
