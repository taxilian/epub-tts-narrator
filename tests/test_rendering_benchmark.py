"""Offline checks for fixed workloads, honest parallel timing, and voice safety."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from threading import Barrier, Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.benchmark_rendering import generate, load_corpus, select_cases, summarize, verify
from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from test_renderer_workflow import FFMPEG, FFPROBE, timed_transcript, write_tone


def row(duration=10, elapsed=2, **changes):
    return {"warmup": False, "audio_seconds": duration, "latency_seconds": elapsed, **changes}


class SummaryTests(unittest.TestCase):
    def test_parallel_makespan_not_sum_of_request_latencies_is_denominator(self):
        rows = [row(elapsed=4, passed=True), row(elapsed=4, passed=True)]
        result = summarize(rows, 5, 3)
        self.assertEqual(result["generated_audio_per_generation_second"], 4)
        self.assertEqual(result["verified_audio_per_total_second"], 2.5)
        self.assertEqual(result["generation_wall_seconds"], 5)

    def test_failures_and_rejections_never_earn_verified_audio(self):
        rows = [row(passed=True), row(passed=False),
                {"warmup": False, "latency_seconds": 20, "error": "timeout", "passed": False},
                row(duration=100, elapsed=500, warmup=True, passed=True)]
        result = summarize(rows, 25, 5)
        self.assertEqual(result["generated_audio_seconds"], 20)
        self.assertEqual(result["verified_audio_seconds"], 10)
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected_or_failed"], 2)
        self.assertEqual(result["verified_audio_per_total_second"], 10 / 30)

    def test_unverified_and_empty_results_cannot_claim_verified_throughput(self):
        for rows in ([], [row()], [row(passed=True), row()]):
            result = summarize(rows, 5)
            self.assertFalse(result["verification_complete"])
            self.assertIsNone(result["verified_audio_per_generation_second"])
            self.assertIsNone(result["verified_audio_per_total_second"])


class WorkloadTests(unittest.TestCase):
    def test_selection_is_fixed_heterogeneous_and_does_not_change_source(self):
        voices = {f"v{i}": {"asset_hash": digest(i)} for i in range(12)}
        chunks = []
        for words in (5, 20, 60):
            for n in range(50):
                chunk = {"chunk_id": f"{words}-{n}", "spoken_text": "word " * words,
                               "silent_control": False,
                               "segment": {"voice_id": f"v{n % 12}", "role_id": "fixture", "delivery": {"direction": ""}}}
                if words == 60 and n < 3:
                    chunk["paragraph_turns"] = [dict(deepcopy(chunk), spoken_text="word " * 30) for _ in range(2)]
                chunks.append(chunk)
        chapters = [{"id": "chapter-0001", "number": 1, "chunks": chunks}]
        original = deepcopy(chapters)
        first, second = select_cases(chapters, voices), select_cases(chapters, voices)
        self.assertEqual(first, second)
        self.assertEqual(chapters, original)
        self.assertEqual(len({c["id"] for c in first}), 24)
        self.assertLessEqual(len({c["voice_asset"] for c in first}), 8)
        self.assertEqual(sum(c["paragraph"] for c in first), 3)
        self.assertEqual({name: sum(c["length_class"] == name for c in first)
                          for name in ("short", "medium", "long")}, {"short": 13, "medium": 7, "long": 4})

    def test_corpus_rejects_changed_reference_bytes_and_escape_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            audio = folder / "reference.flac"
            audio.write_bytes(b"fixture reference")
            body = {"format": "epub-tts-render-benchmark", "version": 1, "cases": [],
                    "voices": {"voice": {"reference_audio": audio.name, "reference_sha256": file_hash(audio)}}}
            path = folder / "corpus.json"
            write_json(path, dict(body, corpus_sha256=digest(body)))
            _, voices = load_corpus(path)
            self.assertEqual(voices["voice"]["reference_audio"], str(audio.resolve()))
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "reference changed"):
                load_corpus(path)
            body["voices"]["voice"]["reference_audio"] = "../outside.flac"
            write_json(path, dict(body, corpus_sha256=digest(body)))
            with self.assertRaisesRegex(ValueError, "relative paths"):
                load_corpus(path)


class FakeEngine:
    identity = {"engine": "offline-fixture", "model": "no-model-loaded"}

    def __init__(self):
        self.barrier, self.lock = Barrier(4), Lock()
        self.active = self.peak = 0
        self.closed = False

    def synthesize(self, text, voice, output, *, seed, direction):
        if seed:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            self.barrier.wait(timeout=5)
            time.sleep(.02)
            with self.lock:
                self.active -= 1
        output.write_bytes(b"audio fixture")
        return {"seed_requested": seed, "seed_applied": False}

    def close(self):
        self.closed = True


class GenerationTests(unittest.TestCase):
    def test_parallel_calls_warmup_separation_and_no_accepted_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            corpus = {"corpus_sha256": digest("fixed corpus"), "cases": [
                {"id": f"case-{i}", "length_class": "short", "paragraph": False,
                 "voice_asset": "voice", "seed_requested": i + 1,
                 "expected_text": f"Test passage {i}.",
                 "turns": [{"text": f"Test passage {i}.", "direction": ""}]} for i in range(4)]}
            config = deepcopy(DEFAULTS)
            config["render_pool"]["workers"] = [{"name": "fixture", "type": "fish_http", "concurrency": 1,
                                                  "protocol": "fish_sglang"}]
            args = SimpleNamespace(corpus=folder / "corpus.json", config=None, worker="fixture",
                                   concurrency=4, output=folder / "results")
            engine = FakeEngine()
            with patch("scripts.benchmark_rendering.load_corpus", return_value=(corpus, {"voice": {}})), \
                    patch("epub_tts_narrator.renderer.config.load_config", return_value=config), \
                    patch("epub_tts_narrator.renderer.pool.create_render_engine", return_value=engine), \
                    patch("epub_tts_narrator.renderer.fish_http.inspect_wav", return_value={"duration_seconds": 2}), \
                    patch("scripts.benchmark_rendering.shutil.which", return_value=None), redirect_stdout(io.StringIO()):
                generate(args)
            from epub_tts_narrator.renderer.common import read_json
            result = read_json(args.output / "manifest.json")
            self.assertEqual(engine.peak, 4)
            self.assertTrue(engine.closed)
            self.assertEqual(sum(r["warmup"] for r in result["results"]), 1)
            self.assertEqual(result["summary"]["generated"], 4)
            self.assertFalse(result["summary"]["verification_complete"])
            self.assertEqual(result["seed_control"], "unsupported_stochastic")
            self.assertFalse(list(args.output.rglob("accepted.json")))
            self.assertTrue(all(r["engine_metadata"]["seed_applied"] is False for r in result["results"]))


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required for real normalization")
class VerificationTests(unittest.TestCase):
    def test_real_audio_checks_preserve_raw_bytes_and_count_bad_text_as_rejected(self):
        class Verifier:
            identity = {"backend": "independent-fixture"}
            def transcribe(self, path):
                return timed_transcript("Hello world.")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            audio = folder / "generated.wav"
            write_tone(audio)
            original = audio.read_bytes()
            rows = [dict(row(duration=1), case_id="warm", warmup=True, expected_text="Hello world."),
                    dict(row(duration=1), case_id="good", expected_text="Hello world."),
                    dict(row(duration=1), case_id="bad", expected_text="We need completely different words.")]
            for item in rows:
                item.update(audio=audio.name, audio_sha256=file_hash(audio))
            manifest = folder / "manifest.json"
            write_json(manifest, {"status": "generated", "results": rows, "generation_wall_seconds": 4})
            config = deepcopy(DEFAULTS)
            config["render"].update(ffmpeg=FFMPEG, ffprobe=FFPROBE)
            with patch("epub_tts_narrator.renderer.config.load_config", return_value=config), \
                    patch("epub_tts_narrator.renderer.chapter_verification.create_chapter_verifier", return_value=Verifier()), \
                    redirect_stdout(io.StringIO()):
                verify(SimpleNamespace(manifest=manifest, config=None))
            from epub_tts_narrator.renderer.common import read_json
            report = read_json(manifest)
            self.assertEqual(report["summary"]["accepted"], 1)
            self.assertEqual(report["summary"]["rejected_or_failed"], 1)
            self.assertEqual(report["summary"]["verified_audio_per_generation_second"], .25)
            self.assertLess(report["summary"]["verified_audio_per_total_second"], .25)
            self.assertEqual(audio.read_bytes(), original)
            self.assertFalse(list(folder.glob("*-verified.wav")))
            self.assertFalse((folder / "verification-warmup.wav").exists())


if __name__ == "__main__":
    unittest.main()
