"""Offline checks for fixed workloads, honest parallel timing, and voice safety."""

from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock, get_ident
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.benchmark_rendering import TimedVerifier, generate, load_corpus, pipeline, select_cases, summarize, summarize_pipeline, verify
from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from test_renderer_workflow import FFMPEG, FFPROBE, timed_transcript, write_tone


def row(duration=10, elapsed=2, **changes):
    return {"warmup": False, "audio_seconds": duration, "latency_seconds": elapsed, **changes}


class SummaryTests(unittest.TestCase):
    def test_pipeline_uses_one_overlapped_makespan_and_excludes_warmup(self):
        rows = [row(duration=10, elapsed=8, passed=True), row(duration=20, elapsed=7, passed=False),
                row(duration=100, elapsed=100, passed=True, warmup=True)]
        result = summarize_pipeline(rows, 9)
        self.assertEqual(result["verified_audio_per_pipeline_second"], 10 / 9)
        self.assertEqual(result["generated_audio_per_pipeline_second"], 30 / 9)
        self.assertIsNone(result["verified_audio_per_generation_second"])
        self.assertIsNone(result["verified_audio_per_total_second"])

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

    def test_stage_totals_include_failed_checks_and_exclude_warmup(self):
        rows = [row(passed=True, verification_stage_seconds={"normalization": 1, "transcribe": 2}),
                row(passed=False, verification_stage_seconds={"transcribe": 3, "judge_reference": 4}),
                row(warmup=True, verification_stage_seconds={"transcribe": 100})]
        self.assertEqual(summarize(rows, 5, 10)["verification_stage_seconds"],
                         {"normalization": 1, "transcribe": 5, "judge_reference": 4})


class TimingTests(unittest.TestCase):
    def test_optional_methods_forward_arguments_results_identity_and_exceptions(self):
        calls, seconds = [], {}
        result = {"text": "fixture"}
        failure = RuntimeError("fixture failure")

        def transcribe(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        def judge(*args, **kwargs):
            raise failure

        target = SimpleNamespace(identity={"backend": "fixture"}, transcribe=transcribe,
                                 judge_reference=judge, speech_resolver=object(), close=lambda: calls.append("close"))
        proxy = TimedVerifier(target, seconds)
        self.assertIs(proxy.identity, target.identity)
        self.assertIs(proxy.speech_resolver, target.speech_resolver)
        self.assertFalse(hasattr(proxy, "corroborate_reference"))
        self.assertIsNone(getattr(proxy, "review_phonetics", None))
        with patch("scripts.benchmark_rendering.time.monotonic", side_effect=[10, 12, 20, 23, 30, 34]):
            self.assertIs(proxy.transcribe(Path("audio.wav"), option="unchanged"), result)
            self.assertIs(proxy.transcribe("again"), result)
            with self.assertRaises(RuntimeError) as raised:
                proxy.judge_reference("expected", "heard")
        self.assertIs(raised.exception, failure)
        self.assertEqual(calls, [((Path("audio.wav"),), {"option": "unchanged"}), (("again",), {})])
        self.assertEqual(seconds, {"transcribe": 5, "judge_reference": 4})
        proxy.close()
        self.assertEqual(calls[-1], "close")


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
        from epub_tts_narrator.renderer.quality import validate_audio
        asr_paths = []
        class Verifier:
            identity = {"backend": "independent-fixture"}
            def transcribe(self, path):
                asr_paths.append(path.name)
                return timed_transcript("Hello world.")
        def acoustic_warmup(path, *args, **kwargs):
            if path.name == "verification-warmup.wav":
                return {"passed": False, "reasons": ["silent_audio"]}
            return validate_audio(path, *args, **kwargs)
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
                    patch("epub_tts_narrator.renderer.quality.validate_audio", side_effect=acoustic_warmup), \
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
            self.assertEqual(report["verification_identity"], Verifier.identity)
            self.assertEqual(asr_paths.count("generated.wav"), 1)
            for item in report["results"][1:]:
                self.assertEqual(set(item["verification_stage_seconds"]), {"normalization", "transcribe"})
                self.assertTrue(all(value >= 0 for value in item["verification_stage_seconds"].values()))
            self.assertEqual(set(report["verification_warmup_stage_seconds"]), {"normalization", "transcribe"})
            for stage, elapsed in report["summary"]["verification_stage_seconds"].items():
                self.assertEqual(elapsed, sum(item["verification_stage_seconds"][stage]
                                              for item in report["results"][1:]))


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required for production normalization")
class PipelineTests(unittest.TestCase):
    def test_real_pool_overlaps_tts_and_one_affine_verifier_and_preserves_failures(self):
        from epub_tts_narrator.renderer.pool import RenderPool
        from epub_tts_narrator.renderer.quality import validate_audio
        asr_entered, generation_overlapped = Event(), Event()
        lock, main_thread = Lock(), get_ident()
        asr_threads, close_order, synth_calls, pools, asr_paths = set(), [], [], [], []
        counts = {"active": 0, "peak": 0, "admitted": 0, "admission_peak": 0}

        class Backend:
            identity = {"engine": "fixture-http"}
            def synthesize(self, text, voice, output, *, seed, direction):
                with lock:
                    synth_calls.append((seed, text, direction))
                    counts["active"] += 1
                    counts["peak"] = max(counts["peak"], counts["active"])
                try:
                    if seed == 4:
                        raise TimeoutError("fixture generation failure")
                    if seed == 2:
                        if not asr_entered.wait(5):
                            raise RuntimeError("TTS failed to overlap verification")
                        generation_overlapped.set()
                    write_tone(output)
                    return {"seed_requested": seed, "seed_applied": False}
                finally:
                    with lock:
                        counts["active"] -= 1
            def synthesize_dialogue(self, turns, output, *, seed):
                self.asserted_turns = turns
                return self.synthesize("".join(t["text"] for t in turns), turns[0]["voice"],
                                       output, seed=seed, direction=turns[0]["direction"])
            def close(self):
                close_order.append("engine")

        class Verifier:
            identity = {"backend": "fixture-asr", "device": "cuda"}
            def transcribe(self, path):
                asr_threads.add(get_ident())
                asr_paths.append(path.name)
                if path.name == "case-1-verified.wav":
                    asr_entered.set()
                    if not generation_overlapped.wait(5):
                        raise RuntimeError("No synthesis progressed while ASR was waiting")
                return timed_transcript("Hello world.")
            def close(self):
                self.assert_drained = counts["active"] == counts["admitted"] == 0
                asr_threads.add(get_ident())
                close_order.append("verifier")

        backend, verifier = Backend(), Verifier()
        def make_pool(config):
            pool = RenderPool(config)
            admission = pool.admit_chunk
            @contextmanager
            def tracked():
                with admission():
                    with lock:
                        counts["admitted"] += 1
                        counts["admission_peak"] = max(counts["admission_peak"], counts["admitted"])
                    try:
                        yield
                    finally:
                        with lock:
                            counts["admitted"] -= 1
            pool.admit_chunk = tracked
            pools.append(pool)
            return pool
        def main_thread_write(*args, **kwargs):
            self.assertEqual(get_ident(), main_thread)
            return write_json(*args, **kwargs)
        def acoustic_warmup(path, *args, **kwargs):
            if "-warmup-" in path.name:
                return {"passed": False, "reasons": ["silent_audio"]}
            return validate_audio(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cases = [{"id": f"case-{i}", "length_class": "short", "paragraph": i == 2,
                      "voice_asset": "voice", "seed_requested": i,
                      "expected_text": "Clearly different words altogether." if i == 3 else "Hello world.",
                      "turns": [{"text": "Clearly different words altogether." if i == 3 else "Hello world.",
                                 "direction": "measured"}]} for i in range(1, 5)]
            corpus = {"corpus_sha256": digest("pipeline fixture"), "cases": cases}
            voices = {"voice": {"kind": "reference", "reference_text": "Reference text.",
                                "reference_audio": str(folder / "voice.flac"), "asset_hash": digest("voice")}}
            config = deepcopy(DEFAULTS)
            config["render"].update(ffmpeg=FFMPEG, ffprobe=FFPROBE)
            config["render_pool"].update(max_inflight=5, workers=[
                {"name": "fixture", "type": "fish_http", "protocol": "fish_sglang_drama3", "concurrency": 4}])
            args = SimpleNamespace(corpus=folder / "corpus.json", config=None, worker="fixture",
                                   concurrency=4, output=folder / "results")
            with patch("scripts.benchmark_rendering.load_corpus", return_value=(corpus, voices)), \
                    patch("epub_tts_narrator.renderer.config.load_config", return_value=config), \
                    patch("epub_tts_narrator.renderer.pool.create_render_engine", side_effect=make_pool), \
                    patch("epub_tts_narrator.renderer.fish_http.FishHTTPEngine", return_value=backend), \
                    patch("epub_tts_narrator.renderer.chapter_verification.create_chapter_verifier", return_value=verifier) as factory, \
                    patch("epub_tts_narrator.renderer.quality.validate_audio", side_effect=acoustic_warmup), \
                    patch("scripts.benchmark_rendering.write_json", side_effect=main_thread_write), \
                    patch("scripts.benchmark_rendering.shutil.which", return_value=None), redirect_stdout(io.StringIO()):
                pipeline(args)
            from epub_tts_narrator.renderer.common import read_json
            report = read_json(args.output / "manifest.json")
            measured = [r for r in report["results"] if not r["warmup"]]
            self.assertTrue(generation_overlapped.is_set())
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(len(asr_threads), 1)
            self.assertEqual(asr_paths.count("case-1-warmup.wav"), 1)
            self.assertNotIn(main_thread, asr_threads)
            self.assertEqual(close_order, ["verifier", "engine"])
            self.assertTrue(verifier.assert_drained)
            self.assertLessEqual(counts["peak"], 4)
            self.assertLessEqual(counts["admission_peak"], 5)
            self.assertCountEqual([call[0] for call in synth_calls], [0, 1, 2, 3, 4])
            self.assertEqual(backend.asserted_turns[0]["direction"], "measured")
            self.assertEqual(report["summary"]["generated"], 3)
            self.assertEqual(report["summary"]["generation_failures"], 1)
            self.assertEqual(report["summary"]["accepted"], 2)
            self.assertEqual(report["summary"]["rejected_or_failed"], 2)
            self.assertEqual(report["summary"]["verified_audio_per_pipeline_second"],
                             report["summary"]["verified_audio_seconds"] / report["pipeline_wall_seconds"])
            self.assertIn("queue waits", report["verification_timing_note"])
            self.assertEqual(report["max_inflight"], 5)
            self.assertFalse(list(args.output.glob("*-verified.wav")))
            self.assertFalse(list(args.output.rglob("accepted.json")))
            for result in measured:
                if "error" not in result:
                    self.assertEqual(file_hash(args.output / result["audio"]), result["audio_sha256"])
                    self.assertEqual(set(result["verification_stage_seconds"]), {"normalization", "transcribe"})
                    self.assertIsNot(result["verification_stage_seconds"], report["results"][0].get("verification_stage_seconds"))


if __name__ == "__main__":
    unittest.main()
