"""Hosted voice-design contracts without credentials, network, or Metal."""

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.error
import wave

from audiobook_renderer.engines import EngineError
from audiobook_renderer.gemini_voice_design import (
    GEMINI_DESIGN_MODEL, GeminiVoiceDesigner, _NoRedirect, _RequestFailure,
)


def wav_bytes(*, frames=2400, rate=24000, channels=1):
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x01\x00" * frames * channels)
    return target.getvalue()


def audio_response(audio=None, *, status="completed"):
    return {"status": status, "steps": [
        {"type": "model_output", "content": [
            {"type": "audio", "mime_type": "audio/wav",
             "data": base64.b64encode(wav_bytes() if audio is None else audio).decode("ascii")},
        ]},
    ]}


class GeminiVoiceDesignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "candidate" / "attempt-001-raw.wav"
        self.key = "test-credential-not-a-real-key"
        self.designer = GeminiVoiceDesigner(self.key)
        self.prompt = "A warm adult female voice, clear and confident."
        self.text = "The door is open. Please come inside."

    def generate(self, designer=None, output=None, **kwargs):
        return (designer or self.designer).create_voice(
            kwargs.pop("description", self.prompt), kwargs.pop("text", self.text),
            output or self.output, seed=kwargs.pop("seed", 42), **kwargs,
        )

    def mock_posts(self, *responses):
        mocked = patch.object(self.designer, "_post", side_effect=responses)
        result = mocked.start()
        self.addCleanup(mocked.stop)
        return result

    def test_prompt_and_script_are_sent_separately_and_automatic_preview_is_ignored(self):
        posts = self.mock_posts(
            {"id": "voice_test123", "sample_audio": {"mime_type": "audio/wav", "data": "invalid"}},
            audio_response(),
        )
        metadata = self.generate()
        create = posts.call_args_list[0]
        self.assertEqual(create.args[0], "/voices")
        self.assertTrue(create.args[1]["store"])
        voice = create.args[1]["voice"]
        self.assertEqual(voice["prompted"], {"input": self.prompt})
        self.assertEqual(voice["model"], GEMINI_DESIGN_MODEL)
        self.assertNotIn("gender", voice)
        self.assertNotIn(self.text, json.dumps(create.args[1]))
        synthesize = posts.call_args_list[1]
        self.assertEqual(synthesize.args[0], "/interactions")
        payload = synthesize.args[1]
        self.assertEqual(payload["input"], [
            {"type": "user_input", "content": [{"type": "text", "text": self.text}]},
        ])
        self.assertEqual(payload["generation_config"], {
            "seed": 42, "speech_config": [{"voice": "voice_test123"}],
        })
        self.assertNotIn(self.prompt, json.dumps(payload))
        self.assertFalse(payload["store"])
        self.assertFalse(payload["stream"])
        self.assertEqual(self.output.read_bytes(), wav_bytes())
        self.assertEqual(metadata["remote_voice_id"], "voice_test123")
        self.assertEqual(metadata["reference_text"], self.text)
        self.assertEqual(metadata["requested_description"], self.prompt)
        self.assertTrue(metadata["description_applied"])
        self.assertEqual(metadata["sample_rate"], 24000)
        self.assertEqual(metadata["sample_count"], 2400)
        self.assertEqual(metadata["duration_seconds"], 0.1)
        self.assertEqual(metadata["creation_method"], "prompt_conditioned_voice_design")
        self.assertEqual(list(self.output.parent.glob(".gemini-*")), [])

    def test_retry_and_process_restart_reuse_voice_id_before_synthesizing(self):
        self.mock_posts({"id": "voice_persistent"}, audio_response(status="incomplete"))
        with self.assertRaisesRegex(EngineError, "did not complete"):
            self.generate()
        self.assertFalse(self.output.exists())
        state = json.loads((self.output.parent / "gemini-voice.json").read_text())
        self.assertEqual(state["voice_id"], "voice_persistent")
        self.assertFalse((self.output.parent / "gemini-voice-pending.json").exists())
        resumed = GeminiVoiceDesigner(self.key)
        with patch.object(resumed, "_post", return_value=audio_response()) as posts:
            result = self.generate(resumed, self.output.with_name("attempt-002-raw.wav"))
        self.assertEqual(posts.call_count, 1)
        self.assertEqual(posts.call_args.args[0], "/interactions")
        self.assertEqual(result["remote_voice_id"], "voice_persistent")

    def test_new_candidate_directory_creates_a_new_voice(self):
        posts = self.mock_posts({"id": "voice_first"}, audio_response(),
                                {"id": "voice_second"}, audio_response())
        first = self.generate()
        second = self.generate(output=self.root / "regenerated" / "attempt-001-raw.wav")
        self.assertNotEqual(first["remote_voice_id"], second["remote_voice_id"])
        self.assertEqual([call.args[0] for call in posts.call_args_list],
                         ["/voices", "/interactions", "/voices", "/interactions"])

    def test_unsigned_renderer_seeds_are_mapped_to_signed_api_range_and_recorded(self):
        cases = [(0, 0), (0x7FFFFFFF, 0x7FFFFFFF), (0x80000000, 0),
                 (0xFFFFFFFF, 0x7FFFFFFF), (3316078238, 1168594590)]
        for requested, effective in cases:
            with self.subTest(seed=requested), patch.object(
                self.designer, "_post", side_effect=[{"id": "voice_seed"}, audio_response()],
            ) as posts:
                metadata = self.generate(output=self.root / str(requested) / "voice.wav", seed=requested)
                self.assertEqual(posts.call_args.args[1]["generation_config"]["seed"], effective)
                self.assertEqual(metadata["seed"], requested)
                self.assertEqual(metadata["synthesis_seed"], effective)

    def test_changed_prompt_in_existing_candidate_is_rejected_without_api_call(self):
        posts = self.mock_posts({"id": "voice_first"}, audio_response())
        self.generate()
        with self.assertRaisesRegex(EngineError, "mismatched"):
            self.generate(output=self.output.with_name("attempt-002-raw.wav"),
                          description="A different voice.")
        self.assertEqual(posts.call_count, 2)

    def test_unknown_creation_outcome_prevents_duplicate_request(self):
        posts = self.mock_posts(_RequestFailure("Gemini API request failed or timed out"))
        for attempt in (1, 2):
            with self.subTest(attempt=attempt), self.assertRaisesRegex(EngineError, "Check Google AI Studio"):
                self.generate(output=self.output.with_name(f"attempt-{attempt:03d}-raw.wav"))
        self.assertEqual(posts.call_count, 1)
        self.assertFalse(self.output.exists())
        self.assertTrue((self.output.parent / "gemini-voice-pending.json").exists())

    def test_explicit_pending_id_recovery_reuses_voice_without_creation(self):
        posts = self.mock_posts(_RequestFailure("Timed out"), audio_response())
        with self.assertRaises(EngineError):
            self.generate()
        pending = self.output.parent / "gemini-voice-pending.json"
        state = json.loads(pending.read_text())
        state["voice_id"] = "voice_recovered"
        pending.write_text(json.dumps(state))
        result = self.generate()
        self.assertEqual(result["remote_voice_id"], "voice_recovered")
        self.assertEqual([call.args[0] for call in posts.call_args_list], ["/voices", "/interactions"])
        self.assertFalse(pending.exists())

    def test_definite_creation_rejection_allows_retry(self):
        posts = self.mock_posts(_RequestFailure("HTTP 429", definite_rejection=True),
                                {"id": "voice_retry"}, audio_response())
        with self.assertRaisesRegex(EngineError, "HTTP 429"):
            self.generate()
        self.assertFalse((self.output.parent / "gemini-voice-pending.json").exists())
        self.generate()
        self.assertEqual(posts.call_count, 3)

    def test_invalid_create_response_leaves_pending_marker_and_never_publishes_preview(self):
        for index, created in enumerate(({}, {"id": "not-a-voice"}, {"id": []})):
            with self.subTest(created=created), patch.object(self.designer, "_post", return_value=created) as posts:
                path = self.root / str(index) / "voice.wav"
                with self.assertRaisesRegex(EngineError, "valid voice ID"):
                    self.generate(output=path)
                self.assertEqual(posts.call_count, 1)
                self.assertFalse(path.exists())
                self.assertTrue((path.parent / "gemini-voice-pending.json").exists())

    def test_existing_output_and_symlink_are_not_overwritten_or_regenerated(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"previous voice")
        with patch.object(self.designer, "_post") as posts:
            with self.assertRaisesRegex(EngineError, "already exists"):
                self.generate()
            self.assertEqual(self.output.read_bytes(), b"previous voice")
            self.output.unlink()
            self.output.symlink_to(self.output.parent / "missing-target.wav")
            with self.assertRaisesRegex(EngineError, "already exists"):
                self.generate()
        posts.assert_not_called()

    def test_incomplete_failed_and_missing_statuses_never_publish(self):
        for status in ("incomplete", "failed", "cancelled", "in_progress", None):
            with self.subTest(status=status), patch.object(
                self.designer, "_post", side_effect=[{"id": "voice_status"}, audio_response(status=status)],
            ):
                target = self.root / str(status) / "voice.wav"
                with self.assertRaisesRegex(EngineError, "did not complete"):
                    self.generate(output=target)
                self.assertFalse(target.exists())

    def test_invalid_response_or_multiple_audio_segments_are_rejected(self):
        multiple = audio_response()
        multiple["steps"][0]["content"] *= 2
        wrong_mime = audio_response()
        wrong_mime["steps"][0]["content"][0]["mime_type"] = "audio/pcm"
        bad_base64 = audio_response()
        bad_base64["steps"][0]["content"][0]["data"] = "%%nope%%"
        cases = [
            {"status": "completed"}, {"status": "completed", "steps": [None]},
            {"status": "completed", "steps": [{"type": "model_output", "content": "invalid"}]},
            {"status": "completed", "steps": []}, multiple, wrong_mime, bad_base64,
            {**audio_response(), "error": {"message": "blocked"}},
        ]
        for index, response in enumerate(cases):
            with self.subTest(index=index), patch.object(
                self.designer, "_post", side_effect=[{"id": "voice_test"}, response],
            ):
                path = self.root / str(index) / "voice.wav"
                with self.assertRaises(EngineError):
                    self.generate(output=path)
                self.assertFalse(path.exists())

    def test_invalid_or_truncated_wav_audio_never_publishes(self):
        cases = [b"", b"not a WAV", wav_bytes()[:-2], wav_bytes() + b"garbage",
                 wav_bytes(frames=1), wav_bytes(rate=1000), wav_bytes(channels=3)]
        # A data chunk whose declared size does not match its byte count.
        corrupt = bytearray(wav_bytes())
        corrupt[40:44] = (len(corrupt) + 100).to_bytes(4, "little")
        cases.append(bytes(corrupt))
        for index, audio in enumerate(cases):
            with self.subTest(index=index), patch.object(
                self.designer, "_post", side_effect=[{"id": "voice_wav"}, audio_response(audio)],
            ):
                path = self.root / str(index) / "voice.wav"
                with self.assertRaises(EngineError):
                    self.generate(output=path)
                self.assertFalse(path.exists())

    def test_invalid_input_fails_without_network_or_files(self):
        cases = [
            {"description": ""}, {"description": "x" * 4001}, {"text": ""},
            {"text": "x" * 1001}, {"text": "bad\x00text"}, {"text": "<|system|>"},
            {"text": "<cough> Hello."}, {"text": "Wait <short pause> here."},
            {"seed": -1}, {"seed": True},
        ]
        with patch.object(self.designer, "_post") as posts:
            for case in cases:
                with self.subTest(case=case), self.assertRaises(EngineError):
                    self.generate(**case)
        posts.assert_not_called()
        self.assertFalse(self.output.parent.exists())

    def test_invalid_settings_are_rejected_without_echoing_credentials(self):
        cases = [
            {"api_key": ""}, {"api_key": None}, {"api_key": "my-secret\nkey"},
            {"model": "../bad"}, {"model": ""}, {"language_code": "English (US)"},
            {"timeout_seconds": True}, {"timeout_seconds": 0},
            {"timeout_seconds": float("inf")}, {"timeout_seconds": 601},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(EngineError) as caught:
                GeminiVoiceDesigner(**{"api_key": self.key, **case})
            self.assertNotIn(self.key, str(caught.exception))
            self.assertNotIn("my-secret", str(caught.exception))

    def test_metadata_state_and_identity_do_not_contain_credentials(self):
        self.mock_posts({"id": "voice_private"}, audio_response())
        metadata = self.generate()
        self.assertNotIn(self.key, json.dumps(metadata))
        self.assertNotIn(self.key, (self.output.parent / "gemini-voice.json").read_text())
        identity = self.designer.identity
        self.assertEqual(identity["engine"], "gemini_voice_design")
        identity["model"] = "changed"
        self.assertEqual(self.designer.identity["model"], GEMINI_DESIGN_MODEL)
        self.assertEqual(self.designer.identity, GeminiVoiceDesigner("other-key").identity)
        self.designer.close()

    def test_http_request_uses_fixed_https_endpoint_and_header_not_query_or_body(self):
        response = io.BytesIO(json.dumps({"ok": True}).encode())
        self.designer._opener = Mock()
        self.designer._opener.open.return_value = response
        self.assertEqual(self.designer._post("/voices", {"store": True}), {"ok": True})
        call = self.designer._opener.open.call_args
        request = call.args[0]
        self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/voices")
        self.assertEqual(request.get_header("X-goog-api-key"), self.key)
        self.assertNotIn(self.key, request.full_url)
        self.assertNotIn(self.key.encode(), request.data)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(call.kwargs["timeout"], 180)
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "Moved", {}, "https://other.invalid"))

    def test_api_errors_do_not_expose_remote_body_url_or_exception_details(self):
        for code in (400, 401, 403, 404, 408, 429, 500):
            error = urllib.error.HTTPError(
                "https://example.invalid?secret=" + self.key, code, self.key,
                {}, io.BytesIO(self.key.encode()),
            )
            self.designer._opener = Mock()
            self.designer._opener.open.side_effect = error
            with self.subTest(code=code), self.assertRaises(_RequestFailure) as caught:
                self.designer._post("/voices", {})
            self.assertNotIn(self.key, str(caught.exception))
            self.assertIn(f"HTTP {code}", str(caught.exception))
            self.assertEqual(caught.exception.definite_rejection, 400 <= code < 500 and code != 408)
            self.assertTrue(caught.exception.__suppress_context__)
        for error in (urllib.error.URLError(self.key), TimeoutError(self.key), OSError(self.key)):
            self.designer._opener.open.side_effect = error
            with self.assertRaises(_RequestFailure) as caught:
                self.designer._post("/interactions", {})
            self.assertNotIn(self.key, str(caught.exception))
            self.assertFalse(caught.exception.definite_rejection)

    def test_invalid_and_oversized_json_responses_are_rejected(self):
        for data in (b"not json", b"[]", b"\xff"):
            self.designer._opener = Mock()
            self.designer._opener.open.return_value = io.BytesIO(data)
            with self.subTest(data=data), self.assertRaises(_RequestFailure):
                self.designer._post("/voices", {})
        self.designer._opener.open.return_value = io.BytesIO(b"x" * 101)
        with patch("audiobook_renderer.gemini_voice_design._MAX_RESPONSE_BYTES", 100):
            with self.assertRaisesRegex(_RequestFailure, "size limit"):
                self.designer._post("/voices", {})


if __name__ == "__main__":
    unittest.main()
