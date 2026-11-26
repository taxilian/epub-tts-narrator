"""Voice-design API contracts, local preview reuse, and clear provider failures."""

import base64
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import time
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

    def test_script_returned_with_only_its_spacing_changed_is_accepted(self):
        spaced = " " + TEXT.replace(". I paused", ".  I paused") + "\n"
        with patch.object(self.designer, "_post", return_value=response()) as post:
            result = self.create(text=spaced)
        self.assertEqual(post.call_args.args[0]["text"], spaced)
        self.assertEqual(result["reference_text"], spaced)
        self.assertTrue(self.output.exists())
        for changed in (TEXT.replace("quiet", "silent"), TEXT.replace("hallway.", "hallway"),
                        TEXT.replace("I paused", "Ipaused")):
            other = self.root / changed[-12:].replace(" ", "-")
            reworded = response(); reworded["text"] = changed
            with self.subTest(changed=changed), patch.object(self.designer, "_post", return_value=reworded), \
                 self.assertRaisesRegex(ElevenLabsVoiceError, "changed the words"):
                self.create(output=other / "raw.wav", text=spaced)
            self.assertFalse((other / "raw.wav").exists())

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

    def test_staged_audio_and_cache_publications_name_destination_and_stage(self):
        # Given the real publication path with the HTTP boundary already faked.
        staged = []
        original = tempfile.NamedTemporaryFile

        def record(*args, **kwargs):
            staged.append(kwargs["prefix"])
            return original(*args, **kwargs)

        # When the created audio and its preview cache publish through staging.
        with patch.object(self.designer, "_post", return_value=response()), \
             patch("tempfile.NamedTemporaryFile", side_effect=record):
            self.create()

        # Then every staged file names its destination and stage, publishes the
        # exact bytes, and leaves no hidden temporary behind.
        self.assertTrue(any("attempt-001-raw" in prefix and "elevenlabs-design" in prefix
                            for prefix in staged))
        self.assertTrue(any("elevenlabs-previews-001" in prefix and "elevenlabs-design" in prefix
                            for prefix in staged))
        with wave.open(str(self.output), "rb") as audio:
            self.assertEqual(audio.readframes(2400), b"\x01\x00" * 2400)
        self.assertEqual([path.name for path in self.root.iterdir() if path.name.startswith(".")], [])

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


class ElevenLabsClipVoiceTests(unittest.TestCase):
    """Extra reference clips: the chosen preview saved as a temporary voice, used, then deleted."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.creation = self.root / "creation-a"
        self.designer = ElevenLabsVoiceDesigner(KEY)
        self.sent = []
        self.permission = 404

    def fake_send(self, method, url, payload=None, *, action, audio=False):
        self.sent.append((method, url, deepcopy(payload)))
        if method == "DELETE" and url.endswith("/audiobook-permission-check"):
            raise ElevenLabsVoiceError(f"HTTP {self.permission}: missing the permission voices_write"
                                       if self.permission == 401 else "HTTP 404: voice_not_found",
                                       status=self.permission)
        if method == "POST" and url.endswith("/v1/text-to-voice"):
            return {"voice_id": "hostedVoice1", "name": payload["voice_name"]}
        if method == "POST" and "/v1/text-to-speech/" in url:
            self.assertTrue(audio)
            return bytes([3, 0]) * 4800
        if method == "DELETE":
            return {}
        raise AssertionError((method, url))

    def speak(self, text, index, seed=11):
        with patch.object(self.designer, "_send", self.fake_send):
            return self.designer.speak_clip(PROMPT, text, self.creation / f"clip-{index}-take.wav", seed=seed,
                                            first={"generated_voice_id": "preview_0"})

    def test_clips_share_one_temporary_voice_that_is_deleted_once(self):
        first = self.speak("Second clip words.", 2)
        self.speak("Third clip words.", 3, seed=2 ** 32 - 1)
        creates = [call for call in self.sent if call[1].endswith("/v1/text-to-voice")]
        self.assertEqual(len(creates), 1)
        self.assertEqual(creates[0][2]["generated_voice_id"], "preview_0")
        self.assertEqual(creates[0][2]["voice_description"], PROMPT)
        speech = [call for call in self.sent if "/text-to-speech/" in call[1]]
        self.assertEqual([call[1] for call in speech],
                         ["https://api.elevenlabs.io/v1/text-to-speech/hostedVoice1?output_format=pcm_24000"] * 2)
        self.assertEqual([(call[2]["text"], call[2]["model_id"], call[2]["seed"]) for call in speech],
                         [("Second clip words.", "eleven_v3", 11), ("Third clip words.", "eleven_v3", 2 ** 32 - 1)])
        self.assertEqual((first["description_applied"], first["reference_text"], first["generated_voice_id"]),
                         (True, "Second clip words.", "preview_0"))
        with wave.open(str(self.creation / "clip-2-take.wav"), "rb") as audio:
            self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getnframes()), (24000, 1, 4800))
        with patch.object(self.designer, "_send", self.fake_send):
            self.assertEqual(self.designer.release_clip_voice(self.creation), "hostedVoice1")
            self.assertIsNone(self.designer.release_clip_voice(self.creation))
        deletes = [call[1] for call in self.sent if call[0] == "DELETE" and "permission-check" not in call[1]]
        self.assertEqual(deletes, ["https://api.elevenlabs.io/v1/voices/hostedVoice1"])
        for file in self.creation.glob("*.json"):
            self.assertNotIn(KEY, file.read_text())
        with self.assertRaisesRegex(ElevenLabsVoiceError, "already deleted"):
            self.speak("Fourth clip words.", 4)

    def test_a_key_without_voice_permissions_is_refused_before_any_voice_is_saved(self):
        self.permission = 401
        with self.assertRaisesRegex(ElevenLabsVoiceError, "voices_read and voices_write") as raised:
            self.speak("Second clip words.", 2)
        self.assertNotIn(KEY, str(raised.exception))
        self.assertEqual([call[0] for call in self.sent], ["DELETE"])
        self.assertFalse(list(self.creation.glob("elevenlabs-voice*.json")))

    def test_a_save_whose_answer_was_lost_is_found_by_its_name_on_the_rerun(self):
        lost = ElevenLabsVoiceError("timed out", retryable=True)
        with patch.object(self.designer, "_send", side_effect=[{}, lost]), self.assertRaises(ElevenLabsVoiceError):
            self.designer.speak_clip(PROMPT, "Second clip words.", self.creation / "take.wav", seed=1,
                                     first={"generated_voice_id": "preview_0"})
        self.assertTrue((self.creation / "elevenlabs-voice-pending.json").exists())
        name = ElevenLabsVoiceDesigner._voice_name(self.creation)
        answers = [ElevenLabsVoiceError("HTTP 400: Voice has already been created", status=400),
                   {"voices": [{"voice_id": "foundVoice", "name": name}, {"voice_id": "other", "name": name + "x"}]},
                   bytes([3, 0]) * 4800]
        with patch.object(self.designer, "_send", side_effect=answers):
            self.designer.speak_clip(PROMPT, "Second clip words.", self.creation / "take.wav", seed=1,
                                     first={"generated_voice_id": "preview_0"})
        state = json.loads((self.creation / "elevenlabs-voice.json").read_text())
        self.assertEqual(state["voice_id"], "foundVoice")
        self.assertFalse((self.creation / "elevenlabs-voice-pending.json").exists())

    def test_a_crashed_creations_temporary_voice_is_deleted_by_a_later_creation(self):
        self.speak("Second clip words.", 2)
        state = self.creation / "elevenlabs-voice.json"
        os.utime(state, (time.time() - 7200, time.time() - 7200))
        self.creation = self.root / "creation-b"
        self.sent.clear()
        self.speak("Another voice's clip.", 2)
        self.assertIn(("DELETE", "https://api.elevenlabs.io/v1/voices/hostedVoice1", None), self.sent)
        self.assertTrue((self.root / "creation-a" / "elevenlabs-voice-cleanup.json").exists())

    def test_a_recent_unfinished_creation_keeps_its_voice(self):
        self.speak("Second clip words.", 2)
        self.creation = self.root / "creation-b"
        self.sent.clear()
        self.speak("Another voice's clip.", 2)
        self.assertNotIn(("DELETE", "https://api.elevenlabs.io/v1/voices/hostedVoice1", None), self.sent)

    def test_scripts_with_control_tags_are_refused_before_any_request(self):
        for text in ("He laughs [laughing] at it.", "Say <break time='1s'/> now.", ""):
            with self.subTest(text=text), self.assertRaises(ElevenLabsVoiceError):
                self.speak(text, 2)
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
