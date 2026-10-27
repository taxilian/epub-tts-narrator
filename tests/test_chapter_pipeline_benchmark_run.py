"""Real HTTP chapter pipeline benchmark runs with the model boundary faked.

The run command starts the production ChapterJobs/ChapterExecutor service over
loopback HTTP, drives it with the real ChapterClient session/ACK lifecycle, and
assembles a final M4B. Only the engine and verifier model factories are faked,
exactly at the boundary the CLI's --fake-engine hook swaps; no network leaves
the machine and no model loads.
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

from scripts.benchmark_chapter_pipeline import main
from scripts.chapter_benchmark_corpus import load_corpus
from scripts.chapter_benchmark_fakeengine import FAKE_ENGINE, LEAF_ENGINES
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.render_pipeline import chapter_share
from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.cli import export_bundle
from audition_fixtures import prepare_document
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone

BOOK_ID = "benchbook"


def valid_document():
    sizes = (("short", 8), ("medium", 24), ("long", 48))
    chapters, blocks, segments = [], [], []
    for number in range(1, 5):
        chapter_id = f"ch{number}"
        chapter_blocks = []
        for kind, count in sizes:
            block_id = f"b{number}-{kind}"
            words = [f"k{number}{kind}w{index}" for index in range(count)]
            if kind == "short":
                text = f"Part {number} short: " + " ".join(words) + "."
            elif kind == "medium":
                text = f"The part {number} medium passage carries these words: " + " ".join(words) + "."
            else:
                text = (f"The part {number} long passage stretches across many more spoken words here: "
                        + " ".join(words) + ".")
            blocks.append({
                "id": block_id, "chapter_id": chapter_id, "text": text,
                "source": {"href": f"text/{chapter_id}.xhtml", "element_id": block_id, "tag": "p"},
            })
            segments.append({
                "id": f"s-{block_id}", "block_id": block_id, "start": 0, "end": len(text),
                "text": text, "kind": "narration", "role_id": "narrator", "voice_id": "vn",
                "delivery": {"direction": "", "evidence": []},
                "attribution": {"basis": "explicit", "reason": "Narration fixture.",
                                "evidence": [], "candidates": [], "review_required": False},
            })
            chapter_blocks.append(block_id)
        chapters.append({"id": chapter_id, "title": f"Part {number}",
                         "source_href": f"text/{chapter_id}.xhtml", "block_ids": chapter_blocks})
    return {
        "format": "epub-audiobook-analysis", "version": 1,
        "source": {"book": {"id": BOOK_ID, "title": "Benchmark Fixture Book", "authors": ["F. Author"], "language": "en"},
                   "chapters": chapters, "blocks": blocks, "exclusions": [], "warnings": [],
                   "extraction_policy": "Fixture text."},
        "cast": {"characters": [],
                 "roles": [{"id": "narrator", "kind": "narration", "character_id": None,
                            "voice_id": "vn", "description": "The fixture narrator."}],
                 "voices": [{"id": "vn", "description": "Fixture measured reading voice.", "traits": []}]},
        "segments": segments, "issues": [], "provenance": {"tool_version": "0.1.0"},
    }


class BenchmarkFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = self.root / "book.audiobook"
        document = valid_document()
        export_bundle(prepare_document(document), self.bundle)
        sample = self.root / "reference.wav"
        write_tone(sample)
        store = VoiceStore(self.bundle / "voices", BOOK_ID)
        lock = store.lock()
        try:
            store.publish("vn", document["cast"]["voices"][0], sample,
                          "Reference narration for offline fixtures.",
                          {"engine": "fixture", "model": "fixture-model"}, {"passed": True},
                          {"description_applied": False})
        finally:
            lock.rmdir()
        self.config_path = self.root / "benchmark.toml"
        lines = ["[engine]", 'type = "drama3_mlx"', f'checkpoint = "{self.root / "unused-engine"}"',
                 "", "[verification]", f'model = "{self.root / "unused-asr"}"', "", "[render]",
                 f'work_dir = "{self.root / "cfg-work"}"', f'output_dir = "{self.root / "cfg-out"}"',
                 f'ffmpeg = "{FFMPEG}"', f'ffprobe = "{FFPROBE}"',
                 "paragraph_synthesis = false", "max_attempts = 1", "normalize_loudness = false",
                 "max_chunk_chars = 1000",
                 "", "[remote_chapters]", "poll_seconds = 0.2", "websocket = false",
                 "", "[chapter_server]", "assembly_workers = 1"]
        self.config_path.write_text("\n".join(lines) + "\n")
        self.corpus = self.root / "corpus"
        _, stderr = self.run_main("prepare", str(self.bundle), "--config", str(self.config_path),
                                  "--chapters", "1,2,3,4", "--cases", "8", "--sample", "smoke",
                                  "--output", str(self.corpus))
        self.assertFalse(stderr, stderr)
        self.frozen = self.corpus / "corpus.json"

    def observe_pool(self):
        """Record the real production engine object without replacing its factory."""
        import epub_tts_narrator.renderer.pool as pool_module
        original = pool_module.create_render_engine
        observed = []

        def observer(config, *, shared_inference=False):
            engine = original(config, shared_inference=shared_inference)
            observed.append(engine)
            return engine

        pool_module.create_render_engine = observer
        self.addCleanup(setattr, pool_module, "create_render_engine", original)
        return observed

    def fresh_fake_registry(self):
        FAKE_ENGINE.calls.clear()
        FAKE_ENGINE.fail_texts.clear()
        LEAF_ENGINES.clear()

    def run_main(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main(list(args))
            except SystemExit as exc:
                self._exit_code = exc.code
                self._stderr = stderr.getvalue()
                raise
            else:
                self._exit_code = 0
                self._stderr = stderr.getvalue()
        return stdout.getvalue(), stderr.getvalue()

    def expect_failure(self, *args):
        with self.assertRaises(SystemExit) as raised:
            self.run_main(*args)
        self.assertNotEqual(raised.exception.code, 0, self._stderr)
        return self._stderr

    def corpus_group_text(self, group):
        corpus, _ = load_corpus(self.frozen)
        return next(case["chunk"]["spoken_text"] for case in corpus["cases"] if case["group"] == group)

    def assert_valid_m4b(self, run_dir):
        book = run_dir / "book" / "book.m4b"
        self.assertTrue(book.is_file(), book)
        subprocess.run([FFMPEG, "-v", "error", "-i", str(book), "-f", "null", "-"],
                       check=True, timeout=60)
        report = read_json(run_dir / "report.json")
        self.assertEqual(report["status"], "complete")
        return report

    def assert_clean_teardown(self):
        names = [thread.name for thread in threading.enumerate()]
        self.assertFalse([name for name in names if "chapter-synthesis" in name or "heartbeat" in name or
                          "uvicorn" in name], names)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class RunLifecycleTests(BenchmarkFixture):
    def test_fake_engine_run_drives_the_real_pool_and_is_never_speed_gate_eligible(self):
        observed = self.observe_pool()
        self.fresh_fake_registry()
        before = len(FAKE_ENGINE.calls)
        self.run_main("run", str(self.frozen), "--config", str(self.config_path),
                      "--scheduler", "legacy", "--output", str(self.root / "run"), "--fake-engine")
        report = self.assert_valid_m4b(self.root / "run")
        # A fake-model run is permanently ineligible, even when fresh complete.
        self.assertIs(report["fake_engine"], True)
        self.assertFalse(report["speed_gate_eligible"])
        self.assertEqual(report["eligibility_reason"], "fake_engine")
        self.assertIsNone(report["accepted_words_per_active_second"])
        self.assertEqual(report["total_source_words"], sum(item["source_words"] for item in report["groups"]))
        self.assertEqual(report["accepted_source_words"], report["total_source_words"])
        # The report carries safe real engine/verifier/quality identities for A/B.
        identity = json.dumps(report["renderer_identity"], sort_keys=True)
        self.assertNotIn("api_key", identity.lower())
        self.assertEqual(report["renderer_identity"]["engine"]["engine"], "fish_tts_pool")
        timing = report["timing"]
        for key in ("cold_setup_seconds", "warmup_seconds", "active_pipeline_seconds",
                    "assembly_seconds", "elapsed_seconds"):
            self.assertIn(key, timing)
        self.assertGreaterEqual(timing["elapsed_seconds"],
                                timing["active_pipeline_seconds"] + timing["assembly_seconds"])
        self.assertGreater(timing["active_pipeline_seconds"], 0.0)
        self.assertGreater(timing["warmup_seconds"], 0.0)
        self.assertFalse(timing.get("warmup_cache_hit"))
        self.assertIs(report["seed_applied"], False)
        self.assertIn("seed", report["seed_note"].lower())
        self.assertEqual(report["gpu_samples"], [])
        self.assertEqual(report["gpu_telemetry"], "unavailable")
        self.assertEqual(report["scheduler"], "legacy")
        self.assertEqual(len(FAKE_ENGINE.calls) - before, 9)  # 1 warmup + 8 measured, no retries
        # The production RenderPool object ran this workload, not a bare engine.
        self.assertEqual(len(observed), 1)
        pool = observed[0]
        self.assertIsInstance(pool, RenderPool)
        self.assertEqual(pool.max_parallel, 1)  # declared slots: one local worker
        effective = report["effective_scheduling"]
        self.assertEqual(effective["scheduler"], "legacy")
        # Coordinators agree with the effective configuration, never the 16
        # safety ceiling. The legacy shipped admission window is the explicit
        # raw render_pool.max_inflight or, automatic here, slots+1; the
        # effective 16 projection is a pipeline-mode expectation only.
        self.assertEqual(report["parallel_chapters"], effective["chapter_concurrency"])
        self.assertEqual(pool.max_inflight, pool.max_parallel + 1)
        self.assertIsNone(pool.resources)  # legacy scheduling builds no lane runtime
        self.assertIsNone(pool.logical)
        self.assertEqual(len(LEAF_ENGINES), 1)  # one bounded leaf engine behind the declared slot
        manifest = read_json(self.root / "run" / "run-manifest.json")
        self.assertEqual([item["state"] for item in manifest["groups"]], ["complete"] * 4)
        self.assertTrue(manifest["fake_engine"])
        self.assertNotIn("session", str(manifest).lower())
        self.assert_clean_teardown()

    def test_rerunning_a_completed_output_requires_a_new_directory(self):
        self.run_main("run", str(self.frozen), "--config", str(self.config_path),
                      "--scheduler", "legacy", "--output", str(self.root / "run"), "--fake-engine")
        self.expect_failure("run", str(self.frozen), "--config", str(self.config_path),
                            "--scheduler", "legacy", "--output", str(self.root / "run"), "--fake-engine")

    def test_failed_group_resumes_from_durable_checkpoints_and_loses_speed_eligibility(self):
        failing = self.corpus_group_text(3)
        FAKE_ENGINE.fail_texts.add(failing)
        self.addCleanup(FAKE_ENGINE.fail_texts.clear)
        output = self.root / "run"
        stderr = self.expect_failure("run", str(self.frozen), "--config", str(self.config_path),
                                     "--scheduler", "legacy", "--output", str(output), "--fake-engine")
        self.assertIn("Remote chapter 5 failed", stderr)
        failed_report = read_json(output / "report.json")
        self.assertEqual(failed_report["status"], "failed")
        self.assertIsNone(failed_report["accepted_words_per_active_second"])
        self.assertGreater(failed_report["timing"]["elapsed_seconds"], 0.0)
        completed = read_json(output / "run-manifest.json")["groups"]
        # Coordinator scheduling with the real pool is concurrent, so exactly
        # which non-failing groups downloaded before the failure raised is
        # timing-dependent; at least one completed group must exist, the
        # failing group must not, and durable checkpoints decide the rest.
        states = [item["state"] for item in completed]
        self.assertEqual(states[3], "pending")
        self.assertGreaterEqual(states.count("complete"), 1)
        self.assertEqual(states.count("pending"), 4 - states.count("complete"))
        # A fake-engine run can never resume into an eligible real-labeled run:
        # the manifest binds the engine policy and rejects the mismatch.
        mismatch = self.expect_failure("run", str(self.frozen), "--config", str(self.config_path),
                                       "--scheduler", "legacy", "--output", str(output))
        self.assertIn("engine policy", mismatch)
        corpus_body, _ = load_corpus(self.frozen)
        pending_texts = {case["chunk"]["spoken_text"] for case in corpus_body["cases"]
                         if states[case["group"]] != "complete"}
        before = len(FAKE_ENGINE.calls)
        FAKE_ENGINE.fail_texts.clear()
        self.run_main("run", str(self.frozen), "--config", str(self.config_path),
                      "--scheduler", "legacy", "--output", str(output), "--fake-engine")
        self.assert_valid_m4b(output)
        resumed = read_json(output / "report.json")
        self.assertFalse(resumed["speed_gate_eligible"])
        self.assertEqual(resumed["eligibility_reason"], "fake_engine")
        self.assertIsNone(resumed["accepted_words_per_active_second"])
        fresh_texts = {entry["text"] for entry in FAKE_ENGINE.calls[before:]}
        # Only pending groups re-render: completed groups come from durable
        # validated downloads and the server's accepted speech cache.
        self.assertLessEqual(fresh_texts, pending_texts)
        self.assertTrue(pending_texts)
        self.assert_clean_teardown()


if __name__ == "__main__":
    unittest.main()
