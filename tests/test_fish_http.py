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

from epub_tts_narrator.spoken_text import MAX_DELIVERY_CHARACTERS
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
                        '<|speaker:1|> [nervous] “”', '<|speaker:1|> [nested [style]] Hi.',
                        '<|speaker:1|> [two\nlines] Hi.',
                        '<|speaker:1|> [' + 'a' * (MAX_DELIVERY_CHARACTERS + 1) + '] Hi.'):
            with self.subTest(invalid=invalid), self.assertRaises(EngineError):
                trim_wire_text('<|speaker:0|> Valid words.\n' + invalid, {0, 1})

    def test_style_prefix_survives_transport_while_quotes_are_trimmed(self):
        source = '<|speaker:0|> [sibilant with prolonged s sounds] “ This is stupid. ”'
        self.assertEqual(trim_wire_text(source, {0}),
                         '<|speaker:0|> [sibilant with prolonged s sounds] This is stupid.')


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
        with patch("epub_tts_narrator.renderer.server.make_server", return_value=server) as factory:
            serve({}, notify=events.append)
        self.assertEqual(factory.call_args.kwargs, {"notify": events.append})
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
        self.response_headers = {}
        opener = Mock()

        def respond(request, **kwargs):
            self.requests.append(request)
            response = io.BytesIO(wav_bytes())
            response.status = 200
            response.headers = {"Content-Length": str(len(wav_bytes())), **self.response_headers}
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
        self.assertEqual(result["seed_effective"], 19)
        # Native Fish does not acknowledge the seed; do not claim verification.
        self.assertFalse(result["seed_applied"])
        self.assertEqual((self.root / "native.wav").read_bytes(), wav_bytes())
        self.assertEqual(client.identity["protocol"], "fish-native-drama3-wav-v1")
        self.assertEqual(client.identity["request_options"]["fixed_top_k"], 30)

    def test_native_upper_half_retries_remain_distinct_after_upstream_seed_clamp(self):
        client = self.client(protocol="fish_native_drama3")
        applied = []
        for attempt in range(12):
            seed = 3_000_000_000 + attempt
            result = client.synthesize("A new take.", self.voice,
                                       self.root / f"retry-{attempt}.wav", seed=seed)
            payload, _ = self.payload()
            wire_seed = payload["seed"]
            # Model-free simulation of pinned native set_seed's upper bound.
            applied.append(min(wire_seed, 1 << 31))
            self.assertEqual(result["seed_requested"], seed)
            self.assertEqual(result["seed_effective"], wire_seed)
            self.assertGreaterEqual(wire_seed, 0)
            self.assertLess(wire_seed, 1 << 31)
            self.assertFalse(result["seed_applied"])
        self.assertEqual(len(set(applied)), 12)

    def test_native_seed_mapping_handles_boundaries_without_changing_cache_identity(self):
        client = self.client(protocol="fish_native_drama3")
        expected_identity = {
            "engine": "fish_http", "model": "drama-3", "protocol": "fish-native-drama3-wav-v1",
            "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_new_tokens": 2048},
            "multi_speaker": True,
            "request_options": {"single_chunk": True, "use_memory_cache": "on",
                                "seed_transport": "body", "fixed_top_k": 30},
        }
        cases = ((0, 0), (1, 1), ((1 << 31) - 1, (1 << 31) - 1),
                 (1 << 31, 0), ((1 << 31) + 1, 1),
                 (0xFFFFFFFE, (1 << 31) - 2), (0xFFFFFFFF, (1 << 31) - 1))
        for seed, effective in cases:
            with self.subTest(seed=seed):
                result = client.synthesize("Hello.", self.voice,
                                           self.root / f"boundary-{seed}.wav", seed=seed)
                payload, headers = self.payload()
                self.assertEqual(payload["seed"], effective)
                self.assertNotIn("X-audiobook-seed", headers)
                self.assertEqual(result["seed_requested"], seed)
                self.assertEqual(result["seed_effective"], effective)
                self.assertEqual(client.identity, expected_identity)

    def test_native_seed_acknowledgement_compares_effective_seed(self):
        client = self.client(protocol="fish_native_drama3")
        seed, effective = (1 << 31) + 7, 7
        for index, (acknowledged, expected) in enumerate(((seed, False), (effective, True))):
            self.response_headers["X-Audiobook-Seed-Applied"] = str(acknowledged)
            result = client.synthesize("Hello.", self.voice,
                                       self.root / f"acknowledged-{index}.wav", seed=seed)
            self.assertEqual(result["seed_requested"], seed)
            self.assertEqual(result["seed_effective"], effective)
            self.assertEqual(result["seed_applied"], expected)

    def test_seed_diagnostics_do_not_invalidate_accepted_speech_identity(self):
        from epub_tts_narrator.renderer.common import digest
        from epub_tts_narrator.renderer.speech_identity import generation_identity

        client = self.client(protocol="fish_native_drama3")
        voice = {**self.voice, "asset_hash": digest("saved-voice"),
                 "reference_sha256": hashlib.sha256(wav_bytes()).hexdigest()}
        request = {"kind": "speech", "engine": client.identity, "base_seed": 42,
                   "sample_rate": 44100, "voice_asset_hash": voice["asset_hash"],
                   "chunk": {"spoken_text": "Hello.", "segment": {"delivery": {"direction": ""}}}}
        request_hash = digest(request)
        current = client.synthesize("Hello.", voice, self.root / "identity.wav", seed=0xFFFFFFFF)
        legacy = {key: value for key, value in current.items() if key != "seed_effective"}
        voices = {voice["asset_hash"]: voice}
        expected = generation_identity(request, voices)
        self.assertEqual(generation_identity(request, voices, engine_metadata=legacy), expected)
        self.assertEqual(generation_identity(request, voices, engine_metadata=current), expected)
        self.assertEqual(digest(request), request_hash)

    def test_inline_protocol_keeps_full_uint32_seed_and_acknowledgement(self):
        seed = 0xFFFFFFFF
        self.response_headers["X-Audiobook-Seed-Applied"] = str(seed)
        result = self.client().synthesize("Hello.", self.voice,
                                          self.root / "inline-seed.wav", seed=seed)
        payload, headers = self.payload()
        self.assertNotIn("seed", payload)
        self.assertEqual(headers["X-audiobook-seed"], str(seed))
        self.assertEqual(result["seed_requested"], seed)
        self.assertTrue(result["seed_applied"])
        self.assertNotIn("seed_effective", result)

    def test_sglang_json_speech_preserves_voice_and_reports_unsupported_seed(self):
        client = self.client(protocol="fish_sglang_drama3", concurrency=4)
        self.response_headers["X-Audiobook-Seed-Applied"] = "99"
        result = client.synthesize("Hello, world.", self.voice,
                                   self.root / "sglang.wav", seed=99, direction="whisper")
        request = self.requests[-1]
        self.assertEqual(request.get_header("Content-type"), "application/json")
        payload = json.loads(request.data)
        self.assertEqual(payload, {
            "input": "<whisper>Hello, world.</whisper>",
            "references": [{"data": base64.b64encode(wav_bytes()).decode("ascii"),
                            "media_type": "audio/wav", "text": "Reference words."}],
            "response_format": "wav", "stream": False, "temperature": 1.0,
            "top_p": .9, "top_k": (1 << 31) - 1, "max_new_tokens": 2048,
        })
        self.assertEqual(result["seed_requested"], 99)
        self.assertIsNone(result["seed_effective"])
        self.assertFalse(result["seed_applied"])
        self.assertEqual(result["seed_mode"], "unsupported")
        self.assertEqual(result["sampling_requested"]["top_k"], 30)
        self.assertEqual(result["sampling_effective"], {
            "temperature": 1.0, "top_p": .9, "max_new_tokens": 2048,
            "main_token_top_k": (1 << 31) - 1, "residual_codebook_top_k": 30})
        self.assertEqual(client.identity["sampling_effective"], result["sampling_effective"])
        self.assertFalse(client.capabilities["multi_speaker"])
        self.assertTrue(client.capabilities["single_speaker_paragraph"])
        self.assertEqual(client.identity["request_options"]["seed_transport"], "unsupported")
        self.assertNotEqual(client.identity, self.client(protocol="fish_native_drama3").identity)

    def test_sglang_uniform_paragraph_preserves_styles_and_rejects_multiple_voices(self):
        client = self.client(protocol="fish_sglang_drama3")
        client.synthesize_dialogue([
            {"text": "Hello. ", "voice": self.voice},
            {"text": "Stay here.", "voice": self.voice, "direction": "whisper"},
        ], self.root / "sglang-paragraph.wav", seed=8)
        payload = json.loads(self.requests[-1].data)
        self.assertEqual(payload["input"], "<|speaker:0|> Hello.\n<|speaker:0|> <whisper>Stay here.</whisper>")
        self.assertEqual(payload["references"][0]["text"], "<|speaker:0|> Reference words.")
        previous = len(self.requests)
        with self.assertRaisesRegex(EngineError, "one reference voice"):
            client.synthesize_dialogue([
                {"text": "Hello.", "voice": self.voice},
                {"text": "Goodbye.", "voice": {**self.voice, "reference_text": "Another voice."}},
            ], self.root / "unsupported.wav", seed=9)
        self.assertEqual(len(self.requests), previous)

    def test_sglang_references_remain_lossless_flac_bytes(self):
        import soundfile as sf
        reference = self.root / "reference.flac"
        sf.write(reference, [0.1] * 100, 16000, format="FLAC")
        voice = {**self.voice, "reference_audio": str(reference)}
        self.client(protocol="fish_sglang_drama3").synthesize(
            "Hello.", voice, self.root / "flac-input.wav", seed=1)
        payload = json.loads(self.requests[-1].data)
        self.assertEqual(payload["references"][0]["media_type"], "audio/flac")
        self.assertEqual(base64.b64decode(payload["references"][0]["data"]), reference.read_bytes())

    def test_sglang_transport_allows_concurrent_requests_without_shared_seed_state(self):
        client = self.client(protocol="fish_sglang_drama3", concurrency=4)
        barrier = threading.Barrier(4, timeout=5)
        seen = []
        opener = Mock()
        def respond(request, **kwargs):
            seen.append(json.loads(request.data))
            barrier.wait()  # All requests must enter before any can finish.
            response = io.BytesIO(wav_bytes())
            response.status, response.headers = 200, {}
            return response
        opener.open.side_effect = respond
        with patch("epub_tts_narrator.renderer.fish_http.build_opener", return_value=opener):
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(client.synthesize, f"Take {number}.", self.voice,
                           self.root / f"parallel-{number}.wav", seed=number) for number in range(4)]
                results = [future.result(10) for future in futures]
        self.assertEqual({item["input"] for item in seen}, {f"Take {n}." for n in range(4)})
        self.assertEqual([result["seed_requested"] for result in results], list(range(4)))
        self.assertTrue(all("seed" not in item for item in seen))

    def test_sglang_unsupported_options_and_oversized_reference_fail_before_request(self):
        for settings in ({"multi_speaker": True}, {"concurrency": 0}, {"concurrency": 17},
                         {"top_k": True}, {"top_k": 20}, {"top_k": 30.0},
                         {"temperature": float("nan")}, {"max_new_tokens": 4096}):
            with self.subTest(settings=settings), self.assertRaises(EngineError):
                self.client(protocol="fish_sglang_drama3", **settings)
        with patch("epub_tts_narrator.renderer.fish_http.MAX_SGLANG_REFERENCE_BYTES", 4):
            with self.assertRaisesRegex(EngineError, "10 MiB"):
                self.client(protocol="fish_sglang_drama3").synthesize(
                    "Hello.", self.voice, self.root / "oversized.wav", seed=1)
        self.assertEqual(self.requests, [])

    def test_http_target_quotes_are_trimmed_before_controls_but_reference_is_unchanged(self):
        voice = {**self.voice, "reference_text": ' “Saved reference.” '}
        self.client().synthesize(' \u2003“ Don\'t call it "easy". ” ', voice,
                                 self.root / "trimmed.wav", seed=3, direction="whisper")
        payload, _ = self.payload()
        self.assertEqual(payload["text"], '<whisper>Don\'t call it "easy".</whisper>')
        self.assertEqual(payload["references"][0]["text"], ' “Saved reference.” ')

    def test_http_style_prefix_retains_reference_voice(self):
        direction = "a strong rural drawl"
        self.client().synthesize("Well, I do not know.", self.voice,
                                 self.root / "styled.wav", seed=3, direction=direction)
        payload, _ = self.payload()
        self.assertEqual(payload["text"], "[a strong rural drawl] Well, I do not know.")
        self.assertEqual(payload["references"], [{"audio": wav_bytes(), "text": "Reference words."}])

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
        self.activity = []
        self.activity_changed = threading.Condition()
        def notify(message):
            with self.activity_changed:
                self.activity.append(message)
                self.activity_changed.notify_all()
        self.config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "private-test-key",
                                  "max_request_bytes": 104857600, "max_pending_requests": 4},
                       "engine": {"type": "drama3_mlx"}}
        try:
            self.server = make_server(self.config, self.engine, notify=notify)
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

    def wait_activity(self, marker, count=1):
        with self.activity_changed:
            ready = self.activity_changed.wait_for(
                lambda: sum(marker in event for event in self.activity) >= count, timeout=5)
            self.assertTrue(ready, self.activity)
            return [event for event in self.activity if marker in event]

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

    def test_activity_distinguishes_queue_wait_from_active_inference(self):
        entered, release = threading.Event(), threading.Event()
        original = self.engine.synthesize_fish
        def blocked(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release generation")
            return original(*args, **kwargs)
        self.engine.synthesize_fish = blocked
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.client.synthesize, "Private first passage.", self.voice,
                                self.root / "first.wav", seed=1)
            try:
                self.assertTrue(entered.wait(5))
                second = pool.submit(self.client.synthesize, "Private second passage.", self.voice,
                                     self.root / "second.wav", seed=2)
                self.wait_activity("queued:", 2)
                self.assertEqual(sum("inference started" in event for event in self.activity), 1)
            finally:
                release.set()
            first.result()
            second.result()
        self.wait_activity("completed", 2)
        for request_id in ("request-000001", "request-000002"):
            events = [event for event in self.activity if request_id in event]
            self.assertEqual(len(events), 3, events)
            self.assertRegex(events[0], r"^\[\d{4}-\d{2}-\d{2}T.+\] request-\d{6}: queued: \d+ characters, 1 reference\(s\)$")
            self.assertIn("inference started (queued ", events[1])
            self.assertIn("completed in ", events[2])
            self.assertIn("0.25s audio, 16000 Hz, 8044 bytes", events[2])
        combined = "\n".join(self.activity)
        for secret in ("Private first passage", "Private second passage", "The reference words.",
                       "private-test-key", base64.b64encode(wav_bytes()).decode()):
            self.assertNotIn(secret, combined)

    def test_disconnect_is_logged_without_claiming_delivery(self):
        with patch("epub_tts_narrator.renderer.server.shutil.copyfileobj", side_effect=BrokenPipeError("private text")):
            with self.assertRaises(FishHTTPError):
                self.client.synthesize("Private passage.", self.voice, self.root / "disconnect.wav", seed=1)
            self.wait_activity("client disconnected")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertFalse(any("completed" in event for event in self.activity))
        self.assertFalse(any("private text" in event or "Private passage" in event for event in self.activity))

    def test_closed_activity_output_does_not_fail_inference_or_leak_queue_slot(self):
        original_notify = self.server.notify
        for index, failure in enumerate((BrokenPipeError("closed pipe"), ValueError("closed stream"))):
            with self.subTest(failure=type(failure).__name__):
                self.server.notify = Mock(side_effect=failure)
                # One occupied slot makes this a queue-rejection logging test.
                self.server.admission = threading.BoundedSemaphore(1)
                self.assertTrue(self.server.admission.acquire(blocking=False))
                try:
                    with self.assertRaises(FishHTTPError) as raised:
                        self.client.synthesize("Queued.", self.voice, self.root / f"queued-{index}.wav", seed=1)
                    self.assertIn("503", str(raised.exception))
                finally:
                    self.server.admission.release()
                # All inference lifecycle logs also hit the failing notifier.
                output = self.root / f"closed-log-{index}.wav"
                self.client.synthesize("Hello.", self.voice, output, seed=1)
                self.assertEqual(output.read_bytes(), wav_bytes())
                # Admission release occurs just after delivery; wait for it.
                self.assertTrue(self.server.admission.acquire(timeout=5))
                self.server.admission.release()
                self.assertGreaterEqual(self.server.notify.call_count, 4)
        self.server.notify = original_notify

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
        self.wait_activity("failed (EngineError)")
        self.assertFalse(any("completed" in event for event in self.activity))
        self.assertNotIn("Detailed GPU failure", "\n".join(self.activity))
        self.assertNotIn("private-test-key", "\n".join(self.activity))

    def test_auth_failure_is_permanent_for_this_run(self):
        self.client.api_key = "wrong"
        with self.assertRaises(FishHTTPError) as raised:
            self.client.synthesize("Hello.", self.voice, self.root / "failed.wav", seed=1)
        self.assertIn("401", str(raised.exception))
        self.assertFalse(raised.exception.retryable)
        self.assertEqual(self.engine.calls, [])
        self.wait_activity("rejected (HTTP 401)")
        self.assertNotIn("wrong", "\n".join(self.activity))

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
        self.wait_activity("queue is full (HTTP 503)")

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
