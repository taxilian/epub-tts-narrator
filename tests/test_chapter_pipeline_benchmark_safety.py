"""Fail-closed safety and lane-runtime checks for benchmark runs.

Companion to test_chapter_pipeline_benchmark_run.py: tampered references or
passages, manifest-less outputs, and the pipeline scheduler's real lane
runtime must all behave correctly before any measured run is trusted.
"""

import subprocess
import unittest

from scripts.chapter_benchmark_fakeengine import LEAF_ENGINES
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


if __name__ == "__main__":
    unittest.main()
