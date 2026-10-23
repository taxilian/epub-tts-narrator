"""Bounded content-addressed reference reuse between FishHTTPEngine and the serve endpoint."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.fish_http import FishHTTPEngine
from epub_tts_narrator.renderer.reference_cache import (
    UNKNOWN_REFERENCE_MARKER,
    ReferenceCache,
)
from epub_tts_narrator.renderer.server import make_server

REQUIREMENTS = "requires renderer HTTP dependencies"


def wav_bytes(sample=1000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<h", sample) * 4000)
    return output.getvalue()


WAV_SHA = hashlib.sha256(wav_bytes()).hexdigest()
ALT_SHA = hashlib.sha256(wav_bytes(2000)).hexdigest()


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.temperature = 1.0
        self.top_p = 0.9
        self.top_k = 30
        self.max_new_tokens = 2048

    def synthesize_fish(self, text, voices, path, *, seed):
        self.calls.append({"text": text, "seed": seed,
                           "references": [(Path(v["reference_audio"]).read_bytes(), v["reference_text"],
                                           v["speaker"]) for v in voices]})
        path.write_bytes(wav_bytes())
        return {"seed": seed}

    def close(self):
        pass

    def warmup(self):
        pass


class ReferenceCacheTests(unittest.TestCase):
    def test_store_and_load_roundtrip_exact_bytes_with_content_addressed_ids(self):
        cache = ReferenceCache()
        audio = wav_bytes()
        content_id = cache.store(audio)
        self.assertEqual(content_id, WAV_SHA)
        self.assertIs(cache.load(WAV_SHA), audio)
        self.assertIsNone(cache.load("0" * 64))

    def test_claimed_id_must_match_content_hash(self):
        cache = ReferenceCache()
        with self.assertRaisesRegex(ValueError, "does not match"):
            cache.store(wav_bytes(), "0" * 64)

    def test_least_recently_used_entry_is_evicted_within_entry_bound(self):
        cache = ReferenceCache(max_entries=2, max_bytes=1 << 20)
        cache.store(b"first")
        cache.store(b"second")
        cache.store(b"third")
        self.assertIsNone(cache.load(cache.content_id(b"first")))
        self.assertEqual(cache.load(cache.content_id(b"second")), b"second")
        # Touching second makes third the least recently used entry.
        cache.store(b"fourth")
        self.assertIsNone(cache.load(cache.content_id(b"third")))
        self.assertEqual(cache.load(cache.content_id(b"second")), b"second")
        self.assertEqual(cache.load(cache.content_id(b"fourth")), b"fourth")

    def test_byte_budget_evicts_oldest_and_refuses_oversized_audio(self):
        cache = ReferenceCache(max_entries=16, max_bytes=150)
        cache.store(b"a" * 100)
        cache.store(b"b" * 100)
        self.assertIsNone(cache.load(cache.content_id(b"a" * 100)))
        self.assertEqual(cache.load(cache.content_id(b"b" * 100)), b"b" * 100)
        oversized = cache.store(b"c" * 200)
        self.assertEqual(oversized, cache.content_id(b"c" * 200))
        self.assertIsNone(cache.load(oversized))
        self.assertEqual(cache.load(cache.content_id(b"b" * 100)), b"b" * 100)

    def test_restoring_same_id_does_not_double_count_budget(self):
        cache = ReferenceCache(max_entries=16, max_bytes=150)
        first = cache.store(b"a" * 100)
        cache.store(b"a" * 100)
        cache.store(b"b" * 40)
        self.assertEqual(cache.load(first), b"a" * 100)
        self.assertEqual(cache.load(cache.content_id(b"b" * 40)), b"b" * 40)

    def test_cache_bounds_are_validated(self):
        for settings in ({"max_entries": 0}, {"max_entries": True}, {"max_bytes": 0},
                         {"max_bytes": -1}, {"max_bytes": 1.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ReferenceCache(**settings)

    def test_concurrent_store_and_load_keep_exact_bytes_within_bounds(self):
        cache = ReferenceCache(max_entries=8, max_bytes=4096)
        blobs = {cache.content_id(bytes([value]) * 100): bytes([value]) * 100
                 for value in range(16)}
        failures = []

        def worker():
            try:
                for _ in range(50):
                    for content_id, blob in blobs.items():
                        cache.store(blob)
                        loaded = cache.load(content_id)
                        if loaded not in (None, blob):
                            failures.append(f"corrupt entry for {content_id}")
            except BaseException as exc:  # Replayed to the main thread below.
                failures.append(f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(worker) for _ in range(4)]:
                future.result(10)
        self.assertEqual(failures, [])


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     REQUIREMENTS)
class ServeReferenceReuseTests(unittest.TestCase):
    """Raw HTTP surface of the serve endpoint's registered-reference protocol."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = FakeEngine()
        self.bodies = []
        original = _parse_request_target()
        def record(data, content_type, cache=None):
            self.bodies.append(bytes(data))
            return original(data, content_type, cache)
        patcher = patch("epub_tts_narrator.renderer.server._parse_request", side_effect=record)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                                  "max_request_bytes": 104857600, "max_pending_requests": 4},
                       "engine": {"type": "drama3_mlx"}}
        try:
            self.server = make_server(self.config, self.engine, notify=lambda message: None)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/tts"

    def stop_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def post(self, body):
        values = {"Content-Type": "application/json", "Authorization": "Bearer private-test-key"}
        request = Request(self.url, data=json.dumps(body).encode(), headers=values)
        return urlopen(request, timeout=5)

    def inline_body(self, *, reference_id=None, text="Reference.", sample=1000):
        reference = {"audio": base64.b64encode(wav_bytes(sample)).decode(), "text": text}
        if reference_id is not None:
            reference["reference_id"] = reference_id
        return {"text": "Hello.", "format": "wav", "normalize": False, "references": [reference]}

    def id_body(self, content_id=WAV_SHA, text="Reference."):
        return {"text": "Hello.", "format": "wav", "normalize": False,
                "references": [{"reference_id": content_id, "text": text}]}

    def test_health_advertises_reference_reuse(self):
        request = Request(self.url.replace("/v1/tts", "/health"),
                          headers={"Authorization": "Bearer private-test-key"})
        with urlopen(request) as response:
            health = json.loads(response.read())
        self.assertIs(health["reference_reuse"], True)

    def test_inline_then_id_only_second_request_transfers_no_audio_bytes(self):
        with self.post(self.inline_body()) as response:
            self.assertEqual(response.read(), wav_bytes())
        with self.post(self.id_body()) as response:
            self.assertEqual(response.read(), wav_bytes())
        self.assertEqual(len(self.engine.calls), 2)
        expected = [(wav_bytes(), "Reference.", None)]
        self.assertEqual(self.engine.calls[0]["references"], expected)
        self.assertEqual(self.engine.calls[1]["references"], expected)
        self.assertLess(len(self.bodies[1]), len(wav_bytes()))
        self.assertNotIn(wav_bytes(), self.bodies[1])
        self.assertIn(WAV_SHA.encode(), self.bodies[1])

    def test_unknown_reference_id_is_rejected_before_synthesis_with_stable_marker(self):
        with self.assertRaises(HTTPError) as raised:
            self.post(self.id_body("1" * 64))
        self.assertEqual(raised.exception.code, 400)
        detail = json.loads(raised.exception.read())["error"]["message"]
        self.assertIn(UNKNOWN_REFERENCE_MARKER, detail)
        self.assertNotIn("Reference.", detail)
        self.assertEqual(self.engine.calls, [])

    def test_inline_audio_with_mismatched_reference_id_is_rejected(self):
        with self.assertRaises(HTTPError) as raised:
            self.post(self.inline_body(reference_id="2" * 64))
        self.assertEqual(raised.exception.code, 400)
        detail = json.loads(raised.exception.read())["error"]["message"]
        self.assertIn("does not match", detail)
        self.assertEqual(self.engine.calls, [])

    def test_malformed_reference_id_is_rejected(self):
        for bad in ("not-a-hash", WAV_SHA.upper(), WAV_SHA[:-1]):
            with self.subTest(reference_id=bad), self.assertRaises(HTTPError) as raised:
                self.post(self.id_body(bad))
            self.assertEqual(raised.exception.code, 400)
            raised.exception.close()
        self.assertEqual(self.engine.calls, [])

    def test_evicted_entry_reports_the_unknown_reference_marker(self):
        self.server.references = ReferenceCache(max_entries=1)
        with self.post(self.inline_body()):
            pass
        with self.post(self.inline_body(sample=2000, text="Second transcript.")):
            pass
        with self.assertRaises(HTTPError) as raised:
            self.post(self.id_body())
        self.assertEqual(raised.exception.code, 400)
        self.assertIn(UNKNOWN_REFERENCE_MARKER,
                      json.loads(raised.exception.read())["error"]["message"])
        self.assertEqual(len(self.engine.calls), 2)

    def test_server_restart_loses_ids_and_accepts_inline_resend(self):
        with self.post(self.inline_body()):
            pass
        self.server.references = ReferenceCache()  # A restart starts empty.
        with self.assertRaises(HTTPError) as raised:
            self.post(self.id_body())
        self.assertEqual(raised.exception.code, 400)
        raised.exception.close()
        with self.post(self.inline_body(reference_id=WAV_SHA)):
            pass
        with self.post(self.id_body()):
            pass
        self.assertEqual(len(self.engine.calls), 3)
        self.assertEqual(self.engine.calls[0]["references"], self.engine.calls[2]["references"])

    def test_top_level_reference_id_remains_rejected(self):
        with self.assertRaises(HTTPError) as raised:
            self.post({**self.inline_body(), "reference_id": WAV_SHA})
        self.assertEqual(raised.exception.code, 400)
        raised.exception.close()
        self.assertEqual(self.engine.calls, [])

    def test_concurrent_inline_registrations_produce_one_shared_entry(self):
        body = self.inline_body()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.post, body) for _ in range(4)]
            for future in futures:
                with future.result():
                    pass
        with self.post(self.id_body()):
            pass
        self.assertEqual(len(self.engine.calls), 5)
        self.assertEqual(len({call["references"][0][0] for call in self.engine.calls}), 1)


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     REQUIREMENTS)
class EngineReferenceReuseTests(unittest.TestCase):
    """FishHTTPEngine against the real loopback serve endpoint, without a model."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        reference = self.root / "reference.wav"
        reference.write_bytes(wav_bytes())
        self.voice = {"kind": "reference", "reference_audio": str(reference),
                      "reference_text": "The reference words.",
                      "reference_sha256": WAV_SHA}
        self.engine = FakeEngine()
        self.bodies = []
        original = _parse_request_target()
        def record(data, content_type, cache=None):
            self.bodies.append(bytes(data))
            return original(data, content_type, cache)
        patcher = patch("epub_tts_narrator.renderer.server._parse_request", side_effect=record)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                                  "max_request_bytes": 104857600, "max_pending_requests": 4},
                       "engine": {"type": "drama3_mlx"}}
        try:
            self.server = make_server(self.config, self.engine, notify=lambda message: None)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.client = FishHTTPEngine({"url": f"http://127.0.0.1:{self.server.server_port}/v1/tts",
                                      "name": "other-mac", "api_key": "private-test-key"})

    def stop_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def test_second_take_sends_registered_id_with_identical_voice_inputs(self):
        self.client.synthesize("Hello.", self.voice, self.root / "first.wav", seed=1)
        self.client.synthesize("Hello again.", self.voice, self.root / "second.wav", seed=2)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(self.engine.calls[0]["references"], self.engine.calls[1]["references"])
        self.assertEqual(self.engine.calls[1]["text"], "Hello again.")
        self.assertLess(len(self.bodies[1]), len(wav_bytes()))
        self.assertNotIn(wav_bytes(), self.bodies[1])
        self.assertFalse(list(self.root.glob(".fish-*")))

    def test_restart_miss_retries_inline_once_without_repeated_synthesis(self):
        self.client.synthesize("First.", self.voice, self.root / "first.wav", seed=1)
        self.server.references = ReferenceCache()  # Simulated restart.
        sent = len(self.bodies)
        self.client.synthesize("Second.", self.voice, self.root / "second.wav", seed=2)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(self.engine.calls[1]["references"], self.engine.calls[0]["references"])
        attempts = self.bodies[sent:]
        self.assertEqual(len(attempts), 2)
        self.assertLess(len(attempts[0]), len(wav_bytes()))
        self.assertGreater(len(attempts[1]), len(wav_bytes()))
        # The inline retry re-registered the audio; the next take reuses the id.
        self.client.synthesize("Third.", self.voice, self.root / "third.wav", seed=3)
        self.assertEqual(len(self.engine.calls), 3)
        self.assertLess(len(self.bodies[-1]), len(wav_bytes()))

    def test_inference_failure_with_reference_marker_does_not_retry_synthesis(self):
        self.client.synthesize("First.", self.voice, self.root / "first.wav", seed=1)
        sent = len(self.bodies)
        with patch.object(self.engine, "synthesize_fish",
                          side_effect=EngineError(UNKNOWN_REFERENCE_MARKER)) as generate:
            with self.assertRaisesRegex(EngineError, "HTTP 500"):
                self.client.synthesize("Second.", self.voice, self.root / "second.wav", seed=2)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(self.bodies) - sent, 1)

    def test_identity_stays_the_inline_contract_without_reuse_markers(self):
        self.assertEqual(self.client.identity, {
            "engine": "fish_http", "model": "drama-3",
            "protocol": "fish-inline-reference-wav-v1",
            "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_new_tokens": 2048},
            "multi_speaker": True,
        })

    def test_unadvertised_endpoint_keeps_full_inline_audio_on_every_take(self):
        opens = []
        def respond(request, **kwargs):
            opens.append(request)
            body = (json.dumps({"status": "ready"}).encode() if request.get_method() == "GET"
                    else wav_bytes())
            response = io.BytesIO(body)
            response.status = 200
            response.headers = {"Content-Length": str(len(body))}
            return response
        opener = Mock()
        opener.open.side_effect = respond
        with patch("epub_tts_narrator.renderer.fish_http.build_opener", return_value=opener):
            client = FishHTTPEngine({"url": "http://127.0.0.1:8080/v1/tts"})
            client.synthesize("Hello.", self.voice, self.root / "one.wav", seed=1)
            client.synthesize("Again.", self.voice, self.root / "two.wav", seed=2)
        self.assertEqual(len(opens), 2)
        import msgpack
        for request in opens:
            self.assertEqual(request.get_method(), "POST")
            payload = msgpack.unpackb(request.data, raw=False)
            self.assertNotIn("reference_id", payload["references"][0])
            self.assertEqual(payload["references"][0]["audio"], wav_bytes())

    def test_advertised_endpoint_second_payload_sends_id_without_audio_bytes(self):
        opens = []
        def respond(request, **kwargs):
            opens.append(request)
            body = (json.dumps({"status": "ready", "reference_reuse": True}).encode()
                    if request.get_method() == "GET" else wav_bytes())
            response = io.BytesIO(body)
            response.status = 200
            response.headers = {"Content-Length": str(len(body)),
                                "X-Audiobook-Reference-Reuse": "sha256-v1"}
            return response
        opener = Mock()
        opener.open.side_effect = respond
        with patch("epub_tts_narrator.renderer.fish_http.build_opener", return_value=opener):
            client = FishHTTPEngine({"url": "http://127.0.0.1:8080/v1/tts"})
            client.synthesize("Hello.", self.voice, self.root / "one.wav", seed=1)
            client.synthesize("Again.", self.voice, self.root / "two.wav", seed=2)
        import msgpack
        self.assertEqual(len(opens), 2)
        first = msgpack.unpackb(opens[0].data, raw=False)
        second = msgpack.unpackb(opens[1].data, raw=False)
        self.assertEqual(first["references"][0]["audio"], wav_bytes())
        self.assertNotIn("reference_id", first["references"][0])
        self.assertEqual(second["references"][0],
                         {"reference_id": WAV_SHA, "text": "The reference words."})
        self.assertGreater(len(opens[0].data), len(wav_bytes()))
        self.assertLess(len(opens[1].data), len(wav_bytes()) // 4)
        self.assertNotIn(wav_bytes(), opens[1].data)

    def test_dialogue_reuses_each_distinct_reference_after_its_first_take(self):
        other = {**self.voice, "reference_text": "Another voice.", "voice_id": "character-2"}
        turns = [{"text": "First.", "voice": self.voice}, {"text": "Second.", "voice": other}]
        self.client.synthesize_dialogue(turns, self.root / "dialogue.wav", seed=1)
        self.client.synthesize_dialogue(turns, self.root / "dialogue-2.wav", seed=2)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(self.engine.calls[0]["references"], self.engine.calls[1]["references"])
        self.assertLess(len(self.bodies[1]), len(wav_bytes()))


def _parse_request_target():
    import epub_tts_narrator.renderer.server as server_module
    return server_module._parse_request


if __name__ == "__main__":
    unittest.main()
