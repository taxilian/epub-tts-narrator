"""Recognition workers: local recognition thread plus matching remote serve recognizers."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from threading import Event, get_ident
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from epub_tts_narrator.renderer.config import DEFAULTS, validate_distributed_settings
from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.recognition_workers import (
    RemoteRecognizer, comparable_identity, identity_differences, recognition_identity, recognizers,
)
from epub_tts_narrator.renderer.render_pipeline_config import effective_scheduling
from epub_tts_narrator.renderer.verification import (
    RecurringSpeechVerifier, ReferenceCorroboratingVerifier, ReferenceTextJudgingVerifier,
)
from test_fish_http import FakeEngine, wav_bytes
from types import SimpleNamespace

KEY = "private-test-key"


def identity(backend, path, **changes):
    return {"backend": backend, "model_path": path, "model_sha256": {"weights.safetensors": "a" * 64},
            "packages": {"mlx": "0.32.4"}, "options": {"temperature": 0.0}, **changes}


class Recognizer:
    """A path-bound recognizer double recording which thread ran each call."""

    def __init__(self, backend, path, text, **changes):
        self.identity = identity(backend, path, **changes)
        self.text = text
        self.threads = []
        self.inputs = []
        self.gate = None

    def _result(self, path):
        self.inputs.append(Path(path).name)
        self.threads.append(get_ident())
        if self.gate is not None and not self.gate.wait(5):
            raise AssertionError("recognizer was never released")
        return {"text": self.text, "segments": [{"text": self.text, "start": 0.0, "end": 0.25}]}

    def transcribe(self, path):
        return self._result(path)

    def transcribe_reference(self, path):
        return self._result(path)


def remote_worker(name="second-mac", roles=("tts", "recognition"), url="http://127.0.0.1:9/v1/tts"):
    return {"name": name, "type": "fish_http", "url": url, "api_key": KEY, "roles": list(roles),
            "recognition_concurrency": 1, "model": "drama-3", "timeout_seconds": 30.0,
            "protocol": "fish_inline", "multi_speaker": True, "concurrency": 1}


class RecognitionIdentityTests(unittest.TestCase):
    def test_identity_ignores_only_machine_local_model_paths(self):
        mine = recognition_identity(Recognizer("mlx-whisper", "/a/whisper", "x"),
                                    Recognizer("mlx-parakeet", "/a/parakeet", "x"))
        theirs = recognition_identity(Recognizer("mlx-whisper", "/b/elsewhere", "x"),
                                      Recognizer("mlx-parakeet", "/b/parakeet", "x"))
        self.assertEqual(mine, theirs)
        self.assertNotIn("model_path", json.dumps(mine))
        changed = recognition_identity(Recognizer("mlx-whisper", "/b", "x", packages={"mlx": "0.31.0"}),
                                       Recognizer("mlx-parakeet", "/b", "x"))
        self.assertEqual(identity_differences(mine, changed), ["primary.packages"])
        # A controller without a reference model ignores the worker's reference.
        primary_only = recognition_identity(Recognizer("mlx-whisper", "/a", "x"))
        self.assertEqual(identity_differences(primary_only, mine), [])
        self.assertEqual(identity_differences(mine, primary_only), ["reference"])
        self.assertEqual(comparable_identity({"model_path": "/x", "nested": [{"model_path": "/y", "k": 1}]}),
                         {"nested": [{"k": 1}]})

    def test_recognizers_are_found_through_every_verifier_wrapper(self):
        primary, reference = Recognizer("p", "/p", "x"), Recognizer("r", "/r", "x")
        judge = SimpleNamespace(identity={"judge": 1})
        wrapped = RecurringSpeechVerifier(
            ReferenceTextJudgingVerifier(ReferenceCorroboratingVerifier(primary, reference), judge),
            SimpleNamespace(identity={"resolver": 1}))
        self.assertEqual(recognizers(wrapped), (primary, reference))
        self.assertEqual(recognizers(primary), (primary, None))


class RecognitionConfigTests(unittest.TestCase):
    def config(self, *workers):
        config = deepcopy(DEFAULTS)
        config["engine"]["type"] = "drama3_mlx"
        config["render_pool"]["workers"] = [dict(worker) for worker in workers]
        return config

    def test_roles_default_to_tts_and_recognition_only_workers_add_no_tts_slots(self):
        plain = remote_worker("tts-mac")
        del plain["roles"], plain["recognition_concurrency"]
        config = self.config({"name": "local", "type": "local"}, plain,
                             remote_worker("asr-mac", roles=("recognition",)))
        validate_distributed_settings(config)
        workers = config["render_pool"]["workers"]
        self.assertEqual(workers[1]["roles"], ["tts"])
        self.assertEqual(workers[2]["roles"], ["recognition"])
        # Two TTS slots (local + tts-mac): chapters scale with TTS, not recognition.
        self.assertEqual(effective_scheduling(config)["chapter_concurrency"], 4)
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        self.assertEqual([worker.settings["name"] for worker in pool._workers], ["local", "tts-mac"])
        self.assertEqual([item.settings["name"] for item in pool._recognizers], ["local", "asr-mac"])
        self.assertEqual(len(pool.resources.tts._threads), 2)
        self.assertEqual(len(pool.resources.recognition._threads), 2)

    def test_invalid_roles_and_recognition_only_pools_are_rejected(self):
        for roles in ([], ["gpu"], ["tts", "tts"], "tts"):
            with self.subTest(roles=roles):
                worker = remote_worker()
                worker["roles"] = roles
                with self.assertRaisesRegex(ValueError, "roles must be a nonempty list"):
                    validate_distributed_settings(self.config(worker))
        with self.assertRaisesRegex(ValueError, "at least one TTS worker"):
            validate_distributed_settings(self.config(remote_worker(roles=("recognition",))))
        worker = remote_worker()
        worker["recognition_concurrency"] = 0
        with self.assertRaisesRegex(ValueError, "recognition_concurrency"):
            validate_distributed_settings(self.config(worker))


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class RecognitionServerTests(unittest.TestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.server import make_server
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "take.wav"
        self.audio.write_bytes(wav_bytes())
        self.primary = Recognizer("mlx-whisper", "/worker/whisper", "primary words")
        self.reference = Recognizer("mlx-parakeet", "/worker/parakeet", "reference words")
        self.server = self.start(make_server, (self.primary, self.reference))
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.client = RemoteRecognizer(remote_worker(url=self.url + "/v1/tts"))
        # The controller's own recognizers: same models in other directories.
        self.expected = recognition_identity(Recognizer("mlx-whisper", "/controller/whisper", "x"),
                                             Recognizer("mlx-parakeet", "/controller/parakeet", "x"))

    def start(self, make_server, recognizers):
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": KEY,
                             "max_request_bytes": 104857600, "max_pending_requests": 4},
                  "engine": {"type": "drama3_mlx"}}
        try:
            server = make_server(config, FakeEngine(), recognizers=recognizers, notify=lambda message: None)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()

        def stop():
            server.shutdown()
            thread.join()
            server.server_close()
        self.addCleanup(stop)
        return server

    def health(self, url):
        request = Request(url + "/health", headers={"Authorization": f"Bearer {KEY}"})
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def test_health_reports_roles_and_path_free_recognizer_identity(self):
        health = self.health(self.url)
        self.assertEqual(health["roles"], ["tts", "recognition"])
        self.assertEqual(health["recognition"], self.expected)
        self.assertNotIn("/worker/", json.dumps(health))

    def test_matching_worker_transcribes_on_its_own_thread_beside_synthesis(self):
        self.client.verify(self.expected)
        primary = self.client.recognize("transcribe", self.audio)
        reference = self.client.recognize("corroborate_reference", self.audio)
        self.assertEqual(primary["text"], "primary words")
        self.assertEqual(primary["recognized_by"], "second-mac")
        self.assertEqual(primary["segments"], [{"text": "primary words", "start": 0.0, "end": 0.25}])
        self.assertEqual(reference["text"], "reference words")
        self.assertEqual(self.client.recognize("transcribe_reference", self.audio)["text"], "primary words")
        recognition_thread = self.server.recognition_runtime.run(get_ident)
        self.assertEqual(set(self.primary.threads) | set(self.reference.threads), {recognition_thread})
        self.assertNotEqual(recognition_thread, self.server.inference_runtime.run(get_ident))

    def test_recognition_scratch_input_name_carries_request_call_and_purpose(self):
        self.client.verify(self.expected)
        self.client.recognize("transcribe", self.audio)
        self.client.recognize("corroborate_reference", self.audio)

        self.assertRegex(self.primary.inputs[0], r"^request-\d{6}-transcribe-recognition-input\.wav$")
        self.assertRegex(self.reference.inputs[0],
                         r"^request-\d{6}-corroborate_reference-recognition-input\.wav$")

    def test_recognition_does_not_wait_for_synthesis_on_the_worker(self):
        self.client.verify(self.expected)
        release = Event()
        self.server.inference_runtime.run(lambda: None)
        blocked = threading.Thread(target=self.server.inference_runtime.run, args=(lambda: release.wait(5),))
        blocked.start()
        try:
            self.assertEqual(self.client.recognize("transcribe", self.audio)["text"], "primary words")
        finally:
            release.set()
            blocked.join()

    def test_different_recognizers_are_refused_naming_fields_only(self):
        other = recognition_identity(Recognizer("mlx-whisper", "/c", "x", model_sha256={"weights.safetensors": "b" * 64}),
                                     Recognizer("mlx-parakeet", "/c", "x"))
        with self.assertRaisesRegex(EngineError, r"second-mac uses different recognizers .*primary\.model_sha256") as caught:
            self.client.verify(other)
        self.assertNotIn("b" * 64, str(caught.exception))
        self.assertNotIn("a" * 64, str(caught.exception))
        with self.assertRaisesRegex(EngineError, "has not been verified"):
            self.client.recognize("transcribe", self.audio)

    def test_worker_restarted_with_other_models_is_refused(self):
        self.client.verify(self.expected)
        self.server.recognition_digest = "c" * 64
        with self.assertRaisesRegex(EngineError, "changed its recognizers"):
            self.client.recognize("transcribe", self.audio)

    def test_tts_only_server_offers_no_recognition(self):
        from epub_tts_narrator.renderer.server import make_server
        server = self.start(make_server, None)
        url = f"http://127.0.0.1:{server.server_port}"
        self.assertEqual(self.health(url)["roles"], ["tts"])
        client = RemoteRecognizer(remote_worker(url=url + "/v1/tts"))
        with self.assertRaisesRegex(EngineError, "does not offer speech recognition"):
            client.verify(self.expected)
        request = Request(url + "/v1/recognize", data=b"\x80", method="POST",
                          headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/msgpack"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 404)
        caught.exception.close()

    def test_unusable_verification_model_leaves_serve_tts_only_with_a_notice(self):
        from epub_tts_narrator.renderer.server import make_server
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": KEY,
                             "max_request_bytes": 104857600, "max_pending_requests": 4},
                  "engine": {"type": "drama3_mlx"},
                  "verification": {"type": "mlx_whisper", "model": str(self.root / "missing-whisper")}}
        messages = []
        try:
            server = make_server(config, FakeEngine(), notify=messages.append)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        self.addCleanup(server.server_close)
        self.assertIsNone(server.recognizers)
        self.assertEqual(len(messages), 1)
        self.assertIn("Speech recognition is not offered", messages[0])

    def test_invalid_recognition_requests_are_rejected(self):
        import msgpack
        for body in ({"call": "align", "audio": b"x"}, {"call": "transcribe"},
                     {"call": "transcribe", "audio": b"not a wav"}):
            with self.subTest(body=sorted(body)):
                request = Request(self.url + "/v1/recognize", data=msgpack.packb(body, use_bin_type=True),
                                  method="POST", headers={"Authorization": f"Bearer {KEY}",
                                                          "Content-Type": "application/msgpack"})
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request, timeout=5)
                self.assertEqual(caught.exception.code, 400)
                caught.exception.close()
        self.assertEqual(self.primary.threads, [])


class FakeRemote:
    """Stands in for RemoteRecognizer inside the pool: no network."""

    def __init__(self, name):
        self.name = name
        self.verified = []
        self.calls = []

    def verify(self, expected):
        self.verified.append(expected)

    def recognize(self, call, path):
        self.calls.append((call, str(path)))
        return {"text": f"{call} on {self.name}", "recognized_by": self.name}


class PooledRecognitionTests(unittest.TestCase):
    def pool(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local"}, remote_worker()]
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        remote = FakeRemote("second-mac")
        pool._recognizers[1].client = remote
        return pool, remote

    def test_wrap_verifier_requires_the_remote_to_match_this_verifiers_models(self):
        pool, remote = self.pool()
        primary = Recognizer("mlx-whisper", "/controller/whisper", "local words")
        pool.wrap_verifier(ReferenceCorroboratingVerifier(primary, Recognizer("mlx-parakeet", "/p", "x")))
        self.assertEqual(remote.verified, [recognition_identity(primary, Recognizer("mlx-parakeet", "/q", "x"))])
        remote.verify = lambda expected: (_ for _ in ()).throw(EngineError("second-mac uses different recognizers"))
        pool._recognition_verified = None
        with self.assertRaisesRegex(EngineError, "different recognizers"):
            pool.wrap_verifier(primary)

    def test_a_busy_local_recognizer_never_delays_the_remote_one(self):
        pool, remote = self.pool()
        primary = Recognizer("mlx-whisper", "/controller/whisper", "local words")
        wrapped = pool.wrap_verifier(primary)
        primary.gate = Event()
        with ThreadPoolExecutor(max_workers=3) as callers:
            first = callers.submit(wrapped.transcribe, "first.wav")
            deadline = threading.Event()
            for _ in range(500):
                if primary.threads:
                    break
                deadline.wait(0.01)
            self.assertEqual(len(primary.threads), 1)
            # The local recognizer is busy: the next call goes to second-mac at once.
            second = callers.submit(wrapped.transcribe_reference, "second.wav").result(5)
            self.assertEqual(second, {"text": "transcribe_reference on second-mac", "recognized_by": "second-mac"})
            self.assertFalse(first.done())
            primary.gate.set()
            self.assertEqual(first.result(5)["recognized_by"], "local")
        self.assertEqual(remote.calls, [("transcribe_reference", "second.wav")])
        self.assertEqual(primary.threads, [pool._recognition_runtime.run(get_ident)])
        self.assertEqual(pool._recognition_slots.qsize(), 2)
        self.assertEqual(pool.resources.recognition._pending, 0)

    def test_many_concurrent_calls_never_overfill_a_recognizer(self):
        pool, remote = self.pool()
        lock = threading.Lock()
        active = {"local": 0, "second-mac": 0}
        peak = dict(active)

        def tracked(name, result):
            with lock:
                active[name] += 1
                peak[name] = max(peak[name], active[name])
            threading.Event().wait(0.002)
            with lock:
                active[name] -= 1
            return result

        primary = Recognizer("mlx-whisper", "/controller/whisper", "local words")
        wrapped = pool.wrap_verifier(primary)
        original = primary._result
        primary._result = lambda path: tracked("local", original(path))
        remote.recognize = lambda call, path: tracked("second-mac", {"text": "remote", "recognized_by": "second-mac"})
        with ThreadPoolExecutor(max_workers=8) as callers:
            results = list(callers.map(lambda index: wrapped.transcribe(f"{index}.wav"), range(80), timeout=60))
        self.assertEqual(len(results), 80)
        self.assertEqual(peak, {"local": 1, "second-mac": 1})
        self.assertEqual({result["recognized_by"] for result in results} <= {"local", "second-mac"}, True)
        self.assertEqual(pool._recognition_slots.qsize(), 2)

    def test_local_only_pools_keep_transcripts_unchanged(self):
        config = deepcopy(DEFAULTS)
        config["render_pool"]["workers"] = [{"name": "local", "type": "local"}]
        pool = RenderPool(config)
        self.addCleanup(pool.close)
        wrapped = pool.wrap_verifier(Recognizer("mlx-whisper", "/w", "words"))
        self.assertNotIn("recognized_by", wrapped.transcribe("take.wav"))


if __name__ == "__main__":
    unittest.main()
