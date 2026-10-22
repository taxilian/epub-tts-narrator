"""Real loopback HTTP integration with a deterministic, model-free engine."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.fish_http import FishHTTPEngine, FishHTTPError
from epub_tts_narrator.renderer.fish_http import trim_wire_text
from epub_tts_narrator.renderer.speech_text import trim_speech_text
from epub_tts_narrator.renderer.server import make_server, serve


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(struct.pack("<h", 1000) * 4000)
    return output.getvalue()


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.temperature = 1.0
        self.top_p = 0.9
        self.top_k = 30
        self.max_new_tokens = 2048
        self.error = None
        self.closed = False
        self.close_thread = None
        self.warmup_thread = None

    def synthesize_fish(self, text, voices, path, *, seed):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append({"text": text, "seed": seed, "temperature": self.temperature,
                               "thread": threading.get_ident(),
                               "references": [(Path(v["reference_audio"]).read_bytes(), v["reference_text"], v["speaker"])
                                              for v in voices]})
            time.sleep(0.02)
            if self.error:
                raise EngineError(self.error)
            path.write_bytes(wav_bytes())
            return {"seed": seed}
        finally:
            self.active -= 1

    def close(self):
        self.closed = True
        self.close_thread = threading.get_ident()

    def warmup(self):
        self.warmup_thread = threading.get_ident()


class SpeechTargetTests(unittest.TestCase):
    def test_alternating_wrapping_marks_and_unicode_whitespace_leave_interior_exact(self):
        interior = 'Don\'t  call the children’s room "empty"…!'
        for left, right in (("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”"),
                            ("«", "»"), ("‹", "›"), ("„", "“")):
            with self.subTest(left=left, right=right):
                source = f"\u2003{left} \t{left}{interior}{right}\n {right}\u00a0"
                self.assertEqual(trim_speech_text(source), interior)
                self.assertEqual(trim_speech_text(trim_speech_text(source)), interior)
        self.assertEqual(trim_speech_text(' ” \t\n\u2003« '), "")

    def test_each_wire_target_is_trimmed_inside_tags_and_prefix_controls(self):
        source = ('<|speaker:0|> [nervous] “ Hello. ”\n'
                  '<|speaker:1|> “ <whisper> « Stay here. » </whisper> ”')
        expected = ('<|speaker:0|> [nervous] Hello.\n'
                    '<|speaker:1|> <whisper>Stay here.</whisper>')
        self.assertEqual(trim_wire_text(source, {0, 1}), expected)

    def test_wire_parenthesis_fragments_are_cleaned_inside_controls(self):
        source = ("<|speaker:0|> [nervous] (Lie, \n"
                  "<|speaker:1|> <whisper>I thought fiercely). </whisper>\n"
                  "<|speaker:0|> He said (quietly) that he agreed.")
        expected = ("<|speaker:0|> [nervous] Lie,\n"
                    "<|speaker:1|> <whisper>I thought fiercely.</whisper>\n"
                    "<|speaker:0|> He said (quietly) that he agreed.")
        self.assertEqual(trim_wire_text(source, {0, 1}), expected)

    def test_empty_or_invalid_wire_turns_are_rejected_even_with_valid_neighbors(self):
        for invalid in ('<|speaker:1|> “”', '<|speaker:1|> <whisper> “” </whisper>',
                        '<|speaker:1|> [nervous] “”', '<|speaker:1|> [unknown] Hi.'):
            with self.subTest(invalid=invalid), self.assertRaises(EngineError):
                trim_wire_text('<|speaker:0|> Valid words.\n' + invalid, {0, 1})


@unittest.skipUnless(importlib.util.find_spec("msgpack"), "requires msgpack")
class FishHTTPErrorBodyTests(unittest.TestCase):
    def test_broken_error_body_preserves_status_detail_and_retryability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            reference.write_bytes(wav_bytes())
            voice = {"kind": "reference", "reference_audio": str(reference), "reference_text": "Hello."}
            client = FishHTTPEngine({"url": "https://example.invalid/v1/tts", "api_key": "private-test-key"})
            for status in (401, 503):
                for body_error in (
                    http.client.IncompleteRead(b"Detail before truncation private-test-key", 100),
                    TimeoutError("timed out reading error body"),
                ):
                    with self.subTest(status=status, error=type(body_error).__name__):
                        body = Mock()
                        body.read.side_effect = body_error
                        failure = HTTPError(client.url, status, "Provider error", {}, body)
                        opener = Mock()
                        opener.open.side_effect = failure
                        with patch("epub_tts_narrator.renderer.fish_http.build_opener", return_value=opener), self.assertRaises(FishHTTPError) as raised:
                            client.synthesize("Hello.", voice, root / "take.wav", seed=1)
                        error = raised.exception
                        self.assertTrue(error.transport_error)
                        self.assertEqual(error.retryable, status == 503)
                        self.assertIn(str(status), str(error))
                        self.assertIn("could not be read completely", str(error))
                        if isinstance(body_error, http.client.IncompleteRead):
                            self.assertIn("Detail before truncation", str(error))
                        self.assertNotIn("private-test-key", str(error))
                        self.assertFalse((root / "take.wav").exists())

    def test_production_server_finishes_warmup_before_announcing_readiness(self):
        events = []
        server = Mock(server_address=("127.0.0.1", 8080))
        server.warmup.side_effect = lambda: events.append("warmup")
        server.serve_forever.side_effect = lambda: events.append("serve")
        with patch("epub_tts_narrator.renderer.server.make_server", return_value=server):
            serve({}, notify=events.append)
        self.assertIn("Loading", events[0])
        self.assertEqual(events[1], "warmup")
        self.assertIn("listening", events[2])
        self.assertEqual(events[-1], "serve")
        server.server_close.assert_called_once()

    def test_warmup_failure_keeps_original_error_and_closes_server(self):
        events = []
        server = Mock(server_address=("127.0.0.1", 8080))
        server.warmup.side_effect = ImportError("missing MLX dependency")
        with patch("epub_tts_narrator.renderer.server.make_server", return_value=server), self.assertRaisesRegex(ImportError, "missing MLX"):
            serve({}, notify=events.append)
        self.assertFalse(any("listening" in event for event in events))
        server.serve_forever.assert_not_called()
        server.server_close.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class NativeDrama3TransportTests(unittest.TestCase):
    """Exercise the native wire contract without downloading a model or using a GPU."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        reference = self.root / "reference.wav"
        reference.write_bytes(wav_bytes())
        self.voice = {"kind": "reference", "reference_audio": str(reference),
                      "reference_text": "Reference words.", "voice_id": "one"}
        self.requests = []
        opener = Mock()

        def respond(request, **kwargs):
            self.requests.append(request)
            response = io.BytesIO(wav_bytes())
            response.status = 200
            response.headers = {"Content-Length": str(len(wav_bytes()))}
            return response

        opener.open.side_effect = respond
        patcher = patch("epub_tts_narrator.renderer.fish_http.build_opener", return_value=opener)
        patcher.start()
        self.addCleanup(patcher.stop)

    def client(self, **settings):
        return FishHTTPEngine({"url": "http://127.0.0.1:8080/v1/tts", **settings})

    def payload(self):
        import msgpack
        request = self.requests[-1]
        return msgpack.unpackb(request.data, raw=False), dict(request.header_items())

    def test_native_profile_preserves_paragraph_enables_cache_and_sends_seed(self):
        client = self.client(protocol="fish_native_drama3")
        result = client.synthesize("One sentence. Another sentence.", self.voice,
                                   self.root / "native.wav", seed=19)
        payload, headers = self.payload()
        self.assertEqual(payload, {
            "text": "One sentence. Another sentence.",
            "references": [{"audio": wav_bytes(), "text": "Reference words."}],
            "format": "wav", "normalize": False, "temperature": 1.0, "top_p": 0.9,
            "max_new_tokens": 2048, "seed": 19, "single_chunk": True, "use_memory_cache": "on",
        })
        self.assertNotIn("top_k", payload)  # Native schema fixes it at 30.
        self.assertNotIn("X-audiobook-top-k", headers)
        self.assertNotIn("X-audiobook-seed", headers)
        self.assertEqual(result["seed_requested"], 19)
        # Native Fish does not acknowledge the seed; do not claim verification.
        self.assertFalse(result["seed_applied"])
        self.assertEqual((self.root / "native.wav").read_bytes(), wav_bytes())
        self.assertEqual(client.identity["protocol"], "fish-native-drama3-wav-v1")
        self.assertEqual(client.identity["request_options"]["fixed_top_k"], 30)

    def test_http_target_quotes_are_trimmed_before_controls_but_reference_is_unchanged(self):
        voice = {**self.voice, "reference_text": ' “Saved reference.” '}
        self.client().synthesize(' \u2003“ Don\'t call it "easy". ” ', voice,
                                 self.root / "trimmed.wav", seed=3, direction="whisper")
        payload, _ = self.payload()
        self.assertEqual(payload["text"], '<whisper>Don\'t call it "easy".</whisper>')
        self.assertEqual(payload["references"][0]["text"], ' “Saved reference.” ')

    def test_http_parenthesis_fragments_are_cleaned_without_changing_reference(self):
        voice = {**self.voice, "reference_text": "(A saved reference)."}
        original = dict(voice)
        cases = (
            ("(Lie, ", "Lie,"),
            ("I thought fiercely). ", "I thought fiercely."),
            ("(Lie.)", "Lie."),
            ("He said (quietly) that he agreed.", "He said (quietly) that he agreed."),
            ("He said (quietly).", "He said (quietly)."),
        )
        for index, (source, target) in enumerate(cases):
            with self.subTest(source=source):
                self.client().synthesize(
                    source, voice, self.root / f"parenthesis-{index}.wav",
                    seed=3, direction="whisper",
                )
                payload, _ = self.payload()
                self.assertEqual(payload["text"], f"<whisper>{target}</whisper>")
                self.assertEqual(payload["references"][0]["text"], "(A saved reference).")
                self.assertEqual(payload["references"][0]["audio"], wav_bytes())
                self.assertEqual(voice, original)

    def test_native_dialogue_keeps_individual_reference_speaker_bindings(self):
        client = self.client(protocol="fish_native_drama3")
        other = {**self.voice, "voice_id": "two", "reference_text": "Another voice."}
        client.synthesize_dialogue([
            {"text": "First speaker.", "voice": self.voice},
            {"text": "Second speaker.", "voice": other},
            {"text": "First again.", "voice": self.voice},
        ], self.root / "dialogue.wav", seed=7)
        payload, _ = self.payload()
        self.assertEqual([reference["text"] for reference in payload["references"]],
                         ["<|speaker:0|> Reference words.", "<|speaker:1|> Another voice."])
        self.assertEqual(payload["text"],
                         "<|speaker:0|> First speaker.\n<|speaker:1|> Second speaker.\n<|speaker:0|> First again.")
        self.assertTrue(payload["single_chunk"])

    def test_default_profile_retains_existing_wire_and_identity(self):
        client = self.client()
        client.synthesize("Hello.", self.voice, self.root / "default.wav", seed=3)
        payload, headers = self.payload()
        self.assertFalse({"seed", "single_chunk", "use_memory_cache"}.intersection(payload))
        self.assertEqual(headers["X-audiobook-seed"], "3")
        self.assertEqual(headers["X-audiobook-top-k"], "30")
        self.assertEqual(client.identity, {
            "engine": "fish_http", "model": "drama-3", "protocol": "fish-inline-reference-wav-v1",
            "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_new_tokens": 2048},
            "multi_speaker": True,
        })
        self.assertEqual(client.identity, self.client(protocol="fish_inline").identity)
        self.assertNotEqual(client.identity, self.client(protocol="fish_native_drama3").identity)

    def test_native_contract_rejects_unsupported_settings_without_sending(self):
        for settings in ({"protocol": []}, {"protocol": "unknown"},
                         {"concurrency": 2}, {"concurrency": True}, {"top_k": 20},
                         {"top_k": 30.0}, {"temperature": 1.1}, {"temperature": 0},
                         {"top_p": 0.01}, {"top_p": True}, {"max_new_tokens": 2049}):
            with self.subTest(settings=settings), self.assertRaises(EngineError):
                self.client(**{"protocol": "fish_native_drama3", **settings})
        self.assertFalse(self.requests)


@unittest.skipUnless(importlib.util.find_spec("msgpack") and importlib.util.find_spec("soundfile"),
                     "requires renderer HTTP dependencies")
class FishHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference.wav"
        self.reference.write_bytes(wav_bytes())
        self.voice = {"kind": "reference", "reference_audio": str(self.reference),
                      "reference_text": "The reference words.", "voice_id": "character-1",
                      "reference_sha256": hashlib.sha256(wav_bytes()).hexdigest()}
        self.engine = FakeEngine()
        self.config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                                  "max_request_bytes": 104857600, "max_pending_requests": 4},
                       "engine": {"type": "drama3_mlx"}}
        try:
            self.server = make_server(self.config, self.engine)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/tts"
        self.client = FishHTTPEngine({"url": self.url, "name": "other-mac", "api_key": "private-test-key"})

    def stop_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def post(self, body, *, headers=None):
        values = {"Content-Type": "application/json", "Authorization": "Bearer private-test-key"}
        values.update(headers or {})
        return urlopen(Request(self.url, data=json.dumps(body).encode(), headers=values), timeout=5)

    def json_body(self):
        return {"text": "Hello.", "format": "wav", "normalize": False,
                "references": [{"audio": base64.b64encode(wav_bytes()).decode(), "text": "Reference."}]}

    def test_single_voice_roundtrip_uses_bytes_seed_and_direction(self):
        output = self.root / "take.wav"
        metadata = self.client.synthesize("Hello.", self.voice, output, seed=19, direction="whisper")
        self.assertEqual(output.read_bytes(), wav_bytes())
        self.assertEqual(self.engine.calls[0]["text"], "<whisper>Hello.</whisper>")
        self.assertEqual(self.engine.calls[0]["references"], [(wav_bytes(), "The reference words.", None)])
        self.assertEqual(self.engine.calls[0]["seed"], 19)
        self.assertTrue(metadata["seed_applied"])
        self.assertEqual(metadata["worker"], "other-mac")
        self.assertEqual(metadata["sample_count"], 4000)
        self.assertFalse(list(self.root.glob(".fish-*")))

    def test_dialogue_matches_reference_tags_and_reuses_same_voice(self):
        voice2 = {**self.voice, "reference_text": "Another voice.", "voice_id": "character-2"}
        turns = [{"text": "First. ", "voice": self.voice}, {"text": "Second.", "voice": self.voice},
                 {"text": "Quiet.", "voice": voice2, "direction": "whisper"}]
        result = self.client.synthesize_dialogue(turns, self.root / "dialogue.wav", seed=2)
        self.assertEqual(self.engine.calls[0]["text"], "<|speaker:0|> First. Second.\n<|speaker:1|> <whisper>Quiet.</whisper>")
        self.assertEqual([r[2] for r in self.engine.calls[0]["references"]], [0, 1])
        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual(result["rendered_turn_count"], 2)

    def test_concurrent_http_requests_share_only_one_model_inference(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.client.synthesize, f"Sentence {i}.", self.voice,
                                   self.root / f"take-{i}.wav", seed=i) for i in range(4)]
            for future in futures:
                future.result()
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(self.engine.max_active, 1)

    def test_model_warmup_inference_and_close_share_one_persistent_thread(self):
        request_threads = []
        original = self.server.process_request_thread
        def record_request_thread(*args):
            request_threads.append(threading.get_ident())
            return original(*args)
        self.server.process_request_thread = record_request_thread
        self.server.warmup()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.client.synthesize, f"Sentence {i}.", self.voice,
                                   self.root / f"thread-{i}.wav", seed=i) for i in range(3)]
            for future in futures:
                future.result()
        inference_threads = {call["thread"] for call in self.engine.calls}
        self.assertEqual(inference_threads, {self.engine.warmup_thread})
        self.assertFalse(inference_threads.intersection(request_threads))
        self.assertNotIn(threading.get_ident(), inference_threads)
        self.stop_server()
        self.assertEqual(self.engine.close_thread, self.engine.warmup_thread)

    def test_full_server_error_is_preserved_with_secret_redaction(self):
        self.engine.error = "Detailed GPU failure: layer 17 / key=private-test-key"
        with self.assertRaises(FishHTTPError) as raised:
            self.client.synthesize("Hello.", self.voice, self.root / "failed.wav", seed=1)
        self.assertIn("Detailed GPU failure: layer 17", str(raised.exception))
        self.assertNotIn("private-test-key", str(raised.exception))
        self.assertTrue(raised.exception.retryable)
        self.assertTrue(raised.exception.transport_error)
        self.assertFalse((self.root / "failed.wav").exists())

    def test_auth_failure_is_permanent_for_this_run(self):
        self.client.api_key = "wrong"
        with self.assertRaises(FishHTTPError) as raised:
            self.client.synthesize("Hello.", self.voice, self.root / "failed.wav", seed=1)
        self.assertIn("401", str(raised.exception))
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(self.engine.calls, [])

    def test_checksum_mismatch_and_existing_output_never_make_requests(self):
        self.voice["reference_sha256"] = "0" * 64
        with self.assertRaisesRegex(EngineError, "checksum"):
            self.client.synthesize("Hello.", self.voice, self.root / "take.wav", seed=1)
        self.voice.pop("reference_sha256")
        output = self.root / "take.wav"
        output.write_bytes(b"accepted")
        with self.assertRaisesRegex(EngineError, "already exists"):
            self.client.synthesize("Hello.", self.voice, output, seed=1)
        self.assertEqual(output.read_bytes(), b"accepted")
        self.assertEqual(self.engine.calls, [])

    def test_json_base64_and_sampling_overrides_restore_engine_defaults(self):
        body = self.json_body()
        body["temperature"] = 0.2
        with self.post(body) as response:
            self.assertEqual(response.read(), wav_bytes())
        self.assertEqual(self.engine.calls[0]["temperature"], 0.2)
        self.assertEqual(self.engine.temperature, 1.0)

    def test_invalid_options_references_controls_and_models_are_400(self):
        bodies = []
        for key, value in (("reference_id", "remote-id"), ("format", "mp3"), ("normalize", True),
                           ("max_new_tokens", 1.1), ("max_new_tokens", 2049), ("temperature", 0),
                           ("text", "<|im_end|> inject"),
                           ("text", "<|speaker:1|> Missing voice."),
                           ("references", [{"audio_path": "/etc/passwd", "text": "x"}])):
            bodies.append({**self.json_body(), key: value})
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(HTTPError) as raised:
                self.post(body)
            self.assertEqual(raised.exception.code, 400)
            raised.exception.close()
        self.assertEqual(self.engine.calls, [])

    def test_request_size_limit_checked_before_body_read(self):
        self.server.settings["max_request_bytes"] = 10
        with self.assertRaises(HTTPError) as raised:
            self.post(self.json_body())
        self.assertEqual(raised.exception.code, 413)
        raised.exception.close()

    def test_health_is_authenticated_and_excludes_private_config(self):
        request = Request(self.url.replace("/v1/tts", "/health"), headers={"Authorization": "Bearer private-test-key"})
        with urlopen(request) as response:
            body = response.read().decode()
        self.assertIn("ready", body)
        self.assertNotIn("private-test-key", body)
        self.assertNotIn(str(self.root), body)

    def test_identity_is_stable_when_url_key_and_worker_name_change(self):
        other = FishHTTPEngine({"url": "https://example.invalid/v1/tts", "name": "cloud", "api_key": "other"})
        self.assertEqual(self.client.identity, other.identity)
        self.assertNotIn("private-test-key", json.dumps(self.client.identity))
        self.assertEqual(self.client.voice_binding_identity, {"engine": "drama3_mlx", "model": "fishaudio/drama-3"})

    def test_non_loopback_requires_api_key_before_binding(self):
        with self.assertRaisesRegex(EngineError, "api_key"):
            make_server({"server": {"host": "0.0.0.0", "api_key": ""}}, FakeEngine())

    def test_reference_speakers_must_be_unambiguous(self):
        reference = self.json_body()["references"][0]
        cases = [
            [reference, reference],
            [{**reference, "speaker": 0}, {**reference, "speaker": 0}],
            [{**reference, "speaker": 1, "text": "<|speaker:0|> Conflict."}],
        ]
        for references in cases:
            with self.subTest(references=references), self.assertRaises(HTTPError) as raised:
                self.post({**self.json_body(), "text": "<|speaker:0|> Hello.", "references": references})
            self.assertEqual(raised.exception.code, 400)
            raised.exception.close()
        self.assertEqual(self.engine.calls, [])

    def _temporary_endpoint(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        def stop():
            server.shutdown()
            thread.join()
            server.server_close()
        self.addCleanup(stop)
        return f"http://127.0.0.1:{server.server_port}/v1/tts"

    def test_valid_wav_with_incomplete_http_body_is_not_published(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Drain inbound bytes so closing sends EOF rather than a reset.
                self.rfile.read(int(self.headers["Content-Length"]))
                payload = wav_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload) + 100))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass
        client = FishHTTPEngine({"url": self._temporary_endpoint(Handler)})
        with self.assertRaisesRegex(FishHTTPError, "truncated"):
            client.synthesize("Hello.", self.voice, self.root / "short.wav", seed=1)
        self.assertFalse((self.root / "short.wav").exists())
        self.assertFalse(list(self.root.glob(".fish-*")))

    def test_redirect_does_not_forward_reference_or_key(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(self.path)
                self.send_response(307)
                self.send_header("Location", "/unwanted")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass
        client = FishHTTPEngine({"url": self._temporary_endpoint(Handler), "api_key": "private"})
        with self.assertRaises(FishHTTPError) as raised:
            client.synthesize("Hello.", self.voice, self.root / "redirect.wav", seed=1)
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(seen, ["/v1/tts"])

    def test_full_queue_rejects_without_additional_inference(self):
        self.server.admission = threading.BoundedSemaphore(1)
        entered, release = threading.Event(), threading.Event()
        original = self.engine.synthesize_fish
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release generation")
            return original(*args, **kwargs)
        self.engine.synthesize_fish = blocked
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(self.client.synthesize, "First.", self.voice, self.root / "first.wav", seed=1)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(FishHTTPError) as raised:
                    self.client.synthesize("Second.", self.voice, self.root / "second.wav", seed=2)
                self.assertIn("503", str(raised.exception))
                self.assertTrue(raised.exception.retryable)
            finally:
                release.set()
            active.result()
        self.assertEqual(len(self.engine.calls), 1)

    def test_malformed_audio_response_never_published(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"oops")

            def log_message(self, *args):
                pass
        remote = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=remote.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            client = FishHTTPEngine({"url": f"http://127.0.0.1:{remote.server_port}/v1/tts"})
            with self.assertRaisesRegex(FishHTTPError, "invalid WAV"):
                client.synthesize("Hello.", self.voice, self.root / "bad.wav", seed=1)
            self.assertFalse((self.root / "bad.wav").exists())
            self.assertFalse(list(self.root.glob(".fish-*")))
        finally:
            remote.shutdown()
            thread.join()
            remote.server_close()


if __name__ == "__main__":
    unittest.main()
