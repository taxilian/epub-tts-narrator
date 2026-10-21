"""Voice-design API contracts, local preview reuse, and clear provider failures."""

import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import wave

from epub_tts_narrator.renderer.elevenlabs_voice_design import ElevenLabsVoiceDesigner, ElevenLabsVoiceError, _NoRedirect
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import RenderBlocked, verified_take


TEXT = ("The door stood open at the end of the quiet hallway. I paused to listen, "
        "then stepped into the morning light. There was still time to decide what to do next.")
PROMPT = "An adult male voice with warm resonance, clear American English and an easy conversational cadence."
KEY = "test-elevenlabs-credential-never-publish"


def response(count=3):
    return {"text": TEXT, "previews": [
        {"audio_base_64": base64.b64encode(bytes([index + 1, 0]) * 2400).decode(),
         "generated_voice_id": f"generated_test_{index}", "media_type": "audio/pcm",
         "duration_secs": .1, "language": "en"} for index in range(count)]}


class ElevenLabsDesignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "attempt-001-raw.wav"
        self.designer = ElevenLabsVoiceDesigner(KEY)

    def create(self, *, designer=None, output=None, seed=1729, text=TEXT, description=PROMPT):
        return (designer or self.designer).create_voice(description, text, output or self.output, seed=seed)

    def test_api_payload_pcm_and_local_only_identity(self):
        with patch.object(self.designer, "_post", return_value=response()) as post:
            result = self.create(seed=0xFFFFFFFF)
        sent = post.call_args.args[0]
        self.assertEqual(sent["voice_description"], PROMPT)
        self.assertEqual(sent["text"], TEXT)
        self.assertEqual(sent["model_id"], "eleven_ttv_v3")
        self.assertEqual(sent["seed"], 0x7FFFFFFF)
        for key in ("auto_generate_text", "stream_previews", "should_enhance"):
            self.assertIs(sent[key], False)
        self.assertTrue(result["description_applied"])
        self.assertFalse(result["remote_voice_created"])
        self.assertNotIn("remote_voice_id", result)
        self.assertNotIn(KEY, json.dumps(result))
        with wave.open(str(self.output), "rb") as audio:
            self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()), (24000, 1, 2))
            self.assertEqual(audio.readframes(2400), b"\x01\x00" * 2400)
        for file in self.root.glob("*.json"):
            self.assertNotIn(KEY, file.read_text())

    def test_retries_and_restart_consume_saved_alternatives_before_new_api_request(self):
        with patch.object(self.designer, "_post", return_value=response()) as post:
            first = self.create()
            second = self.create(output=self.root / "attempt-002-raw.wav", seed=1730)
        self.assertEqual(post.call_count, 1)
        resumed = ElevenLabsVoiceDesigner(KEY)
        with patch.object(resumed, "_post", side_effect=AssertionError("No new API request")):
            third = self.create(designer=resumed, output=self.root / "attempt-003-raw.wav", seed=1731)
        self.assertEqual([r["preview_index"] for r in (first, second, third)], [0, 1, 2])
        self.assertEqual([r["design_seed"] for r in (first, second, third)], [1729] * 3)
        with patch.object(resumed, "_post", return_value=response()) as post:
            fourth = self.create(designer=resumed, output=self.root / "attempt-004-raw.wav", seed=1732)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(fourth["design_seed"], 1732)

    def test_output_loss_recovers_same_reserved_preview_without_network(self):
        with patch.object(self.designer, "_post", return_value=response()):
            first = self.create()
        original = self.output.read_bytes()
        self.output.unlink()
        with patch.object(self.designer, "_post", side_effect=AssertionError("No network")):
            resumed = self.create()
        self.assertEqual(resumed, first)
        self.assertEqual(self.output.read_bytes(), original)
        self.output.unlink()
        next(self.root.glob("elevenlabs-previews-*.json")).unlink()
        with patch.object(self.designer, "_post", side_effect=AssertionError("No network")), \
             self.assertRaisesRegex(ElevenLabsVoiceError, "batch is missing"):
            self.create()

    def test_wrong_script_invalid_audio_and_cached_corruption_fail_without_output(self):
        invalid = []
        wrong = response(); wrong["text"] = "Unrequested script"; invalid.append(wrong)
        for field, value in (("audio_base_64", "not base64"), ("media_type", "audio/mpeg"),
                             ("media_type", None), ("generated_voice_id", "../../voice"),
                             ("audio_base_64", base64.b64encode(b"a").decode())):
            bad = response(); bad["previews"][0][field] = value; invalid.append(bad)
        for index, bad in enumerate(invalid):
            with self.subTest(index=index), patch.object(self.designer, "_post", return_value=bad), \
                 self.assertRaises(ElevenLabsVoiceError):
                self.create(output=self.root / str(index) / "raw.wav")
            self.assertFalse((self.root / str(index) / "raw.wav").exists())
        with patch.object(self.designer, "_post", return_value=response()):
            self.create()
        cached = next(self.root.glob("elevenlabs-previews-*.json"))
        cached.write_text(cached.read_text().replace('generated_test_0', 'generated_bad_00'))
        with self.assertRaisesRegex(ElevenLabsVoiceError, "preview cache"):
            self.create(output=self.root / "attempt-002-raw.wav")

    def test_validation_happens_before_network_and_existing_files_never_overwrite(self):
        with patch.object(self.designer, "_post", side_effect=AssertionError("No network")):
            for text in ("short", "x" * 1001, TEXT + " [laughs]", TEXT + "<|tag|>"):
                with self.subTest(text=text[:10]), self.assertRaises(ElevenLabsVoiceError):
                    self.create(text=text)
            for prompt in ("short", "x" * 1001):
                with self.assertRaises(ElevenLabsVoiceError):
                    self.create(description=prompt)
            self.output.write_bytes(b"keep")
            with self.assertRaisesRegex(ElevenLabsVoiceError, "already exists"):
                self.create()
            self.assertEqual(self.output.read_bytes(), b"keep")

    def test_credentials_header_redirect_denial_and_403_diagnostics(self):
        response_handle = Mock()
        response_handle.__enter__ = Mock(return_value=response_handle)
        response_handle.__exit__ = Mock(return_value=False)
        response_handle.read.return_value = json.dumps(response()).encode()
        with patch.object(self.designer._opener, "open", return_value=response_handle) as opened:
            self.designer._post({"text": TEXT})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.elevenlabs.io/v1/text-to-voice/design?output_format=pcm_24000")
        self.assertEqual(request.get_header("Xi-api-key"), KEY)
        self.assertIsNone(_NoRedirect().redirect_request(request, None, 302, "Found", {}, "https://example.com"))
        error = urllib.error.HTTPError(request.full_url, 403, "Forbidden", {},
            io.BytesIO(('paid plan required ' + KEY).encode()))
        with patch.object(self.designer._opener, "open", side_effect=error), \
             self.assertRaises(ElevenLabsVoiceError) as raised:
            self.designer._post({})
        self.assertIn("HTTP 403", str(raised.exception))
        self.assertIn("paid plan required", str(raised.exception))
        self.assertNotIn(KEY, str(raised.exception))
        self.assertFalse(raised.exception.retryable)

    def test_verification_stops_after_one_nonretryable_provider_failure(self):
        config = deepcopy(DEFAULTS); config["quality"] = validate_policy({})
        config["elevenlabs"] = {"api_key": KEY}
        verifier = SimpleNamespace(identity={"fixture": True})
        generate = Mock(side_effect=ElevenLabsVoiceError("HTTP 403: paid plan required " + KEY))
        with self.assertRaisesRegex(RenderBlocked, "paid plan required") as raised:
            verified_take(TEXT, generate, self.root / "takes", {"kind": "fixture"}, verifier,
                          config, notify=lambda _: None, purpose="reference")
        self.assertEqual(generate.call_count, 1)
        self.assertNotIn(KEY, str(raised.exception))
        report = json.loads((self.root / "takes" / "attempt-001.json").read_text())["report"]
        self.assertFalse(report["retryable"])
        self.assertFalse((self.root / "takes" / "accepted.json").exists())


if __name__ == "__main__":
    unittest.main()
