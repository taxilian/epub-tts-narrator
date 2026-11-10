"""Serving batched Drama 3: per-request sampling, concurrency, identity, and config.

Offline: a fake batching engine behind the real HTTP server on loopback; the
CUDA engine is constructed without loading torch.
"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

from test_fish_http import FakeEngine, wav_bytes

from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
from epub_tts_narrator.renderer.fish_speech_engine import FishSpeechCudaEngine
from epub_tts_narrator.renderer.request_sampling import current_sampling
from epub_tts_narrator.renderer.runpod_config import build_remote_config, inference_workers, load_settings, write_private_toml
from epub_tts_narrator.renderer.server import make_server

_path = Path(__file__).resolve().parents[1] / "deploy/runpod/worker.py"
_spec = importlib.util.spec_from_file_location("runpod_worker_batching", _path)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


class BatchingFakeEngine(FakeEngine):
    """Accepts concurrent requests and reads sampling per request, never attributes."""

    def __init__(self):
        super().__init__()
        self.batch_rows = 1
        self.together = threading.Barrier(2, timeout=5)
        self.seen = []

    def set_batch_rows(self, rows):
        self.batch_rows = rows

    @property
    def concurrent_requests(self):
        return self.batch_rows

    def synthesize_fish(self, text, voices, path, *, seed, history=()):
        self.seen.append((seed, current_sampling()))
        self.together.wait()  # both requests are inside the engine at once
        return super().synthesize_fish(text, voices, path, seed=seed, history=history)


class BatchedServingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference.wav"
        self.reference.write_bytes(wav_bytes())
        self.voice = {"kind": "reference", "reference_audio": str(self.reference),
                      "reference_text": "The reference words.", "voice_id": "character-1"}

    def serve(self, engine, **server):
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "", "max_request_bytes": 104857600,
                             "max_pending_requests": 4, **server}, "engine": {"type": "drama3_mlx"}}
        try:
            instance = make_server(config, engine, notify=lambda message: None)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(lambda: (instance.shutdown(), thread.join(), instance.server_close()))
        return f"http://127.0.0.1:{instance.server_port}/v1/tts"

    def test_batching_engine_takes_concurrent_requests_with_their_own_sampling(self):
        engine = BatchingFakeEngine()
        url = self.serve(engine, batch_rows=2)
        self.assertEqual(engine.batch_rows, 2)
        clients = [FishHTTPEngine({"url": url, "name": f"client-{n}", "api_key": ""},
                                  {"temperature": temperature, "top_p": 0.7, "top_k": 30})
                   for n, temperature in enumerate((0.5, 0.9))]
        errors = []

        def speak(client, seed):
            try:
                client.synthesize("A passage.", self.voice, self.root / f"take-{seed}.wav", seed=seed)
            except Exception as exc:  # surfaced below
                errors.append(exc)
        threads = [threading.Thread(target=speak, args=(client, seed)) for client, seed in zip(clients, (11, 22))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(engine.max_active, 2)
        sampling = dict(engine.seen)
        self.assertEqual((sampling[11]["temperature"], sampling[22]["temperature"]), (0.5, 0.9))
        # The shared engine's own attributes were never rewritten per request.
        self.assertEqual(engine.temperature, 1.0)

    def test_health_reports_how_many_requests_the_worker_decodes_at_once(self):
        from urllib.request import urlopen
        batching = self.serve(BatchingFakeEngine(), batch_rows=2)
        plain = self.serve(FakeEngine())
        for url, expected in ((batching, 2), (plain, 1)):
            with self.subTest(expected=expected), urlopen(url.replace("/v1/tts", "/health"), timeout=5) as response:
                self.assertEqual(json.loads(response.read())["concurrent_requests"], expected)

    def test_eight_decode_rows_are_all_admitted_with_a_small_pending_queue(self):
        engine = BatchingFakeEngine()
        engine.together = threading.Barrier(8, timeout=5)
        url = self.serve(engine, batch_rows=8, max_pending_requests=1)
        errors = []

        def speak(index):
            try:
                client = FishHTTPEngine({"url": url, "timeout_seconds": 10})
                client.synthesize("A passage.", self.voice,
                                  self.root / f"concurrent-passage-{index}.wav", seed=index)
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=speak, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(engine.seen), 8)
        self.assertTrue(all((self.root / f"concurrent-passage-{index}.wav").is_file()
                            for index in range(8)))

    def test_serve_logs_how_each_batched_request_shared_the_batch(self):
        from epub_tts_narrator.renderer.server import batch_summary
        stats = {"rows": 8, "joined_active": 5, "waited_seconds": 0.4, "steps": 120, "mean_rows": 6.5,
                 "peak_rows": 8, "prompt_rows": 731, "cached_rows": 512, "prefill_seconds": 0.21}
        self.assertEqual(batch_summary(stats),
                         "; batch: joined 5 running of 8 rows after 0.40s, 6.5 rows on average (peak 8) "
                         "over 120 steps, prompt 731 rows (512 cached), prefill 0.21s")
        self.assertEqual(batch_summary(None), "")
        engine = BatchingFakeEngine()
        engine.together = threading.Barrier(1)
        original = engine.synthesize_fish
        engine.synthesize_fish = lambda *args, **kwargs: {**original(*args, **kwargs), "batch": stats}
        messages = []
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "", "max_request_bytes": 104857600,
                             "max_pending_requests": 4, "batch_rows": 2}, "engine": {"type": "drama3_mlx"}}
        try:
            instance = make_server(config, engine, notify=messages.append)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(lambda: (instance.shutdown(), thread.join(), instance.server_close()))
        FishHTTPEngine({"url": f"http://127.0.0.1:{instance.server_port}/v1/tts", "name": "c", "api_key": ""},
                       {"temperature": 0.7}).synthesize("A passage.", self.voice, self.root / "take.wav", seed=3)
        completed = [line for line in messages if "completed in" in line]
        self.assertTrue(completed and completed[-1].endswith(batch_summary(stats)), messages)

    def test_a_client_that_goes_away_cancels_its_batched_request(self):
        from epub_tts_narrator.renderer.fish_http import FishHTTPError
        from epub_tts_narrator.renderer.request_sampling import current_cancellation
        engine = BatchingFakeEngine()
        cancelled = threading.Event()

        def decode_until_cancelled(text, voices, path, *, seed, history=()):
            signal = current_cancellation()
            if signal is not None and signal.wait(10):
                cancelled.set()
            raise RuntimeError("cancelled")

        engine.synthesize_fish = decode_until_cancelled
        url = self.serve(engine, batch_rows=2)
        client = FishHTTPEngine({"url": url, "name": "impatient", "api_key": "", "timeout_seconds": 1},
                                {"temperature": 0.7})
        with self.assertRaises(FishHTTPError):
            client.synthesize("A passage.", self.voice, self.root / "take.wav", seed=4)
        # The client timed out and closed; serve noticed and stopped the decode.
        self.assertTrue(cancelled.wait(5))

    def test_failed_requests_log_why_without_quoting_their_text(self):
        from epub_tts_narrator.renderer.drama3_batch_scheduler import RequestCancelled
        from epub_tts_narrator.renderer.server import failure_summary

        def caused(error, cause):
            try:
                raise error from cause
            except type(error) as raised:
                return raised

        cases = (
            (EngineError("Drama 3 did not finish the text: 'silence'"),
             "EngineError: the take went silent and the silence watch stopped it"),
            (EngineError("Drama 3 did not finish the text: 'length'"),
             "EngineError: the take hit its frame budget without ending"),
            (EngineError("Drama 3 prompt is too long even without chapter history: Secret passage words"),
             "EngineError: the prompt is too long even without chapter history"),
            (RequestCancelled("x"), "RequestCancelled: the client went away, so its batched decode was stopped"),
            (caused(EngineError("Drama 3 generation failed: Secret passage words"), MemoryError()),
             "EngineError: generation failed inside the model (caused by MemoryError)"),
            (ValueError("Secret passage words: something odd"), "ValueError: details were returned to the client"),
        )
        for error, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(failure_summary(error), expected)
                self.assertNotIn("Secret", failure_summary(error))

    def test_single_request_engines_keep_the_serialized_path(self):
        engine = FakeEngine()
        url = self.serve(engine)
        FishHTTPEngine({"url": url, "name": "client", "api_key": ""}, {"temperature": 0.4}).synthesize(
            "A passage.", self.voice, self.root / "take.wav", seed=5)
        self.assertEqual(engine.calls[0]["temperature"], 0.4)
        self.assertEqual(engine.temperature, 1.0)

    def test_batch_rows_need_a_batching_engine(self):
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "", "max_pending_requests": 4,
                             "batch_rows": 2}, "engine": {"type": "drama3_mlx"}}
        with self.assertRaisesRegex(EngineError, "decodes batches"):
            make_server(config, FakeEngine())

    def test_server_batch_rows_is_validated(self):
        path = self.root / "config.toml"
        for value, ok in ((1, True), (16, True), (32, True), (0, False), (33, False), (True, False)):
            with self.subTest(value=value):
                write_private_toml(path, {"server": {"batch_rows": value}})
                if ok:
                    self.assertEqual(load_config(path, require_verifier=False, require_engine=False)["server"]["batch_rows"], value)
                else:
                    with self.assertRaisesRegex(ValueError, "server.batch_rows"):
                        load_config(path, require_verifier=False, require_engine=False)
        self.assertEqual(DEFAULTS["server"]["batch_rows"], 1)


class CudaEngineBatchIdentityTests(unittest.TestCase):
    def engine(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        checkpoint = Path(temporary.name)
        (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"fixture weights")
        (checkpoint / "config.json").write_text("{}")
        return FishSpeechCudaEngine(checkpoint)

    def test_batched_takes_carry_their_own_engine_identity(self):
        engine = self.engine()
        single = engine.identity
        self.assertNotIn("decoder", single)
        self.assertEqual(engine.concurrent_requests, 1)
        engine.set_batch_rows(8)
        batched = engine.identity
        self.assertEqual(engine.concurrent_requests, 8)
        self.assertEqual({key: batched[key] for key in ("decoder", "decoder_version", "batch_rows", "sampler")},
                         {"decoder": "batched", "decoder_version": 2, "batch_rows": 8, "sampler": "counter-hash-v2"})
        self.assertEqual({k: v for k, v in batched.items() if k not in ("decoder", "decoder_version", "batch_rows", "sampler")},
                         single)

    def test_batch_rows_are_bounded_and_fixed_once_the_model_loads(self):
        engine = self.engine()
        for value in (0, 33, 2.0, True):
            with self.subTest(value=value), self.assertRaisesRegex(EngineError, "batch_rows"):
                engine.set_batch_rows(value)
        engine._model = object()
        with self.assertRaisesRegex(EngineError, "before the model loads"):
            engine.set_batch_rows(4)


class RunpodBatchConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.path = self.directory / "epub-tts-narrator.toml"
        write_private_toml(self.path, {
            "engine": {"checkpoint": "models/local"}, "verification": {"model": "models/whisper"},
            "chapter_server": {"runpod_api_key": "file-runpod-key"},
            "chapter_storage": {"bucket": "books", "endpoint_url": "https://storage.example.test",
                                "region": "default", "access_key_id": "ceph-key", "secret_access_key": "ceph-secret"}})

    def settings(self, **runpod):
        write_private_toml(self.directory / "runpod.local.toml", {"runpod": runpod})
        return load_settings(self.path)

    def test_one_batched_copy_takes_that_many_concurrent_requests(self):
        settings = self.settings(inference_batch_rows=8)
        remote = build_remote_config(settings["renderer_config"], settings["override_raw"], "pod",
                                     "application-key", deployment=settings["deployment"])
        self.assertEqual([(w["url"], w["concurrency"]) for w in remote["render_pool"]["workers"]],
                         [("http://127.0.0.1:8080/v1/tts", 8)])
        self.assertEqual(self.settings()["deployment"]["inference_batch_rows"], 1)

    def test_invalid_or_conflicting_batch_settings_are_rejected(self):
        for values, message in (({"inference_batch_rows": 0}, "inference_batch_rows"),
                                ({"inference_batch_rows": 17}, "inference_batch_rows"),
                                ({"inference_batch_rows": "8"}, "inference_batch_rows"),
                                ({"inference_batch_rows": 4, "inference_replicas": 2, "min_gpu_memory_gb": 96},
                                 "not both")):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                self.settings(**values)

    def test_worker_writes_batch_rows_and_a_matching_queue_for_the_copy(self):
        config = deepcopy(DEFAULTS)
        config["chapter_server"]["data_dir"] = str(self.directory / "data")
        config["engine"]["checkpoint"] = "/workspace/models/drama-3"
        for rows, pending in ((1, 4), (2, 4), (8, 16), (16, 32)):
            with self.subTest(rows=rows):
                config["render_pool"]["workers"] = inference_workers(1, rows)
                text = worker.inference_config(config).read_text()
                self.assertIn(f"batch_rows = {rows}\n", text)
                self.assertIn(f"max_pending_requests = {pending}\n", text)


if __name__ == "__main__":
    unittest.main()
