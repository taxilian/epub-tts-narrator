"""Remote Gemini voice cleanup preserves local casting assets and attribution."""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from urllib.parse import parse_qs, urlsplit
import wave

from epub_tts_narrator.renderer.gemini_voice_design import (
    GEMINI_DESIGN_MODEL, GeminiVoiceDesigner, GeminiVoiceError,
)


def _response(value):
    return io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())


def _http_error(code, message="Provider error"):
    return urllib.error.HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/voices/voice_saved",
        code, "Provider error", {},
        _response({"error": {"code": code, "message": message}}),
    )


class _QueuedHTTP:
    """Replace only the external HTTP boundary; all adapter logic runs normally."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, **kwargs):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected HTTP request: " + request.get_method() + " " + request.full_url)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return _response(response)


def _audio_response():
    target = io.BytesIO()
    with wave.open(target, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\x01\x00" * 2400)
    return {"status": "completed", "steps": [{
        "type": "model_output", "content": [{
            "type": "audio", "mime_type": "audio/wav",
            "data": base64.b64encode(target.getvalue()).decode(),
        }],
    }]}


class GeminiVoiceCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "candidate"
        self.output = self.directory / "attempt-001-raw.wav"
        self.key = "offline-test-credential"
        self.designer = GeminiVoiceDesigner(self.key)
        self.prompt = "An adult audiobook narrator with a warm, measured voice."
        self.script = "The door is open. Please come inside."

    def http(self, *responses):
        result = _QueuedHTTP(*responses)
        self.designer._opener = result
        return result

    def create(self):
        return self.designer.create_voice(self.prompt, self.script, self.output, seed=17)

    def saved_candidate(self):
        self.http({"id": "voice_saved"}, _audio_response())
        self.create()
        return json.loads((self.directory / "gemini-voice.json").read_text())

    def pending_candidate(self):
        self.http(TimeoutError("Connection lost after request was sent"))
        with self.assertRaises(GeminiVoiceError):
            self.create()
        return json.loads((self.directory / "gemini-voice-pending.json").read_text())

    def test_list_collects_all_pages_with_escaped_continuation_tokens(self):
        http = self.http(
            {"voices": [{"id": "voice_first"}], "next_page_token": "cursor/+?=&"},
            {"voices": [{"id": "voice_second"}]},
        )

        voices = self.designer.list_voices()

        self.assertEqual([voice["id"] for voice in voices], ["voice_first", "voice_second"])
        queries = [parse_qs(urlsplit(request.full_url).query) for request in http.requests]
        self.assertEqual(queries[0]["type"], ["prompted"])
        self.assertEqual(queries[0]["page_size"], ["100"])
        self.assertEqual(queries[1]["page_token"], ["cursor/+?=&"])
        for request in http.requests:
            self.assertEqual(request.get_method(), "GET")
            self.assertIsNone(request.data)
            self.assertNotIn(self.key, request.full_url)
            self.assertEqual(request.get_header("X-goog-api-key"), self.key)

    def test_delete_accepts_empty_success_response_and_missing_voice(self):
        http = self.http(b"", _http_error(404, "Voice no longer exists"))

        self.assertTrue(self.designer.delete_voice("voice_saved"))
        self.assertFalse(self.designer.delete_voice("voice_saved"))

        self.assertEqual([request.get_method() for request in http.requests], ["DELETE", "DELETE"])
        self.assertTrue(all(request.data is None for request in http.requests))

    def test_invalid_voice_ids_cannot_change_delete_endpoint(self):
        http = self.http()
        for voice_id in ("../voices", "voice_saved/other", "voice_saved?force=true",
                         "voice_saved#fragment", "voice_saved%2fother", "voice_saved\n", "", None):
            with self.subTest(voice_id=voice_id), self.assertRaises(GeminiVoiceError):
                self.designer.delete_voice(voice_id)
        self.assertEqual(http.requests, [])

    def test_successful_cleanup_preserves_local_audio_and_state_and_does_not_repeat_delete(self):
        state = self.saved_candidate()
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        http = self.http(b"")

        self.assertEqual(self.designer.release_voice(self.directory), ["voice_saved"])
        self.designer.release_voice(self.directory)

        self.assertEqual(len(http.requests), 1)
        for name, contents in before.items():
            self.assertEqual((self.directory / name).read_bytes(), contents)
        cleanup = json.loads((self.directory / "gemini-voice-cleanup.json").read_text())
        self.assertTrue(cleanup["remote_deleted"])
        self.assertEqual(cleanup["voice_ids"], ["voice_saved"])
        self.assertEqual(cleanup["request_fingerprint"], state["request_fingerprint"])
        self.assertNotIn(self.key, json.dumps(cleanup))

    def test_already_deleted_voice_still_finishes_local_cleanup(self):
        self.saved_candidate()
        self.http(_http_error(404))

        self.designer.release_voice(self.directory)

        self.assertTrue(json.loads((self.directory / "gemini-voice-cleanup.json").read_text())["remote_deleted"])
        self.assertTrue(self.output.is_file())

    def test_failed_delete_is_actionable_and_retryable_without_false_success_marker(self):
        self.saved_candidate()
        http = self.http(_http_error(500, "Backend unavailable; credential=" + self.key), b"")

        with self.assertRaises(GeminiVoiceError) as caught:
            self.designer.release_voice(self.directory)

        self.assertIn("HTTP 500", str(caught.exception))
        self.assertIn("Backend unavailable", str(caught.exception))
        self.assertNotIn(self.key, str(caught.exception))
        self.assertFalse((self.directory / "gemini-voice-cleanup.json").exists())
        self.assertTrue(self.output.is_file())
        self.assertEqual(self.designer.release_voice(self.directory), ["voice_saved"])
        self.assertEqual([request.get_method() for request in http.requests], ["DELETE", "DELETE"])

    def test_cleaned_candidate_cannot_reuse_deleted_remote_id(self):
        self.saved_candidate()
        self.http(b"")
        self.designer.release_voice(self.directory)
        http = self.http()

        with self.assertRaises(GeminiVoiceError):
            self.designer.create_voice(
                self.prompt, self.script, self.directory / "attempt-002-raw.wav", seed=18,
            )

        self.assertEqual(http.requests, [])
        self.assertFalse((self.directory / "attempt-002-raw.wav").exists())

    def test_incomplete_cleanup_cannot_reuse_remote_voice_after_delete(self):
        self.saved_candidate()
        self.http(b"")
        link = os.link

        def fail_final_marker(source, target, *args, **kwargs):
            if Path(target).name == "gemini-voice-cleanup.json":
                raise OSError("Temporary local disk failure")
            return link(source, target, *args, **kwargs)

        with patch("os.link", side_effect=fail_final_marker):
            with self.assertRaises(OSError):
                self.designer.release_voice(self.directory)
        http = self.http()

        with self.assertRaises(GeminiVoiceError):
            self.designer.create_voice(
                self.prompt, self.script, self.directory / "attempt-002-raw.wav", seed=18,
            )

        self.assertEqual(http.requests, [])
        self.assertFalse((self.directory / "attempt-002-raw.wav").exists())

    def test_unknown_creation_is_reconciled_by_exact_name_model_and_type(self):
        pending = self.pending_candidate()
        exact_name = pending["display_name"]
        matching = {"id": "voice_recovered", "display_name": exact_name,
                    "model": GEMINI_DESIGN_MODEL, "type": "prompted"}
        http = self.http({"voices": [
            dict(matching, id="voice_prefix", display_name=exact_name + " extra"),
            dict(matching, id="voice_other_model", model="another-model"),
            dict(matching, id="voice_other_type", type="cloned"),
            matching,
        ]}, b"")

        self.assertEqual(self.designer.release_voice(self.directory), ["voice_recovered"])

        deleted = [request for request in http.requests if request.get_method() == "DELETE"]
        self.assertEqual(len(deleted), 1)
        self.assertEqual(urlsplit(deleted[0].full_url).path, "/v1beta/voices/voice_recovered")
        self.assertTrue((self.directory / "gemini-voice-pending.json").exists())
        cleanup = json.loads((self.directory / "gemini-voice-cleanup.json").read_text())
        self.assertEqual(cleanup["voice_ids"], ["voice_recovered"])
        self.assertEqual(cleanup["request_fingerprint"], pending["request_fingerprint"])

    def test_unresolved_creation_remains_pending_without_claiming_remote_deletion(self):
        self.pending_candidate()
        http = self.http({"voices": []})

        with self.assertRaises(GeminiVoiceError):
            self.designer.release_voice(self.directory)

        self.assertTrue((self.directory / "gemini-voice-pending.json").exists())
        self.assertFalse((self.directory / "gemini-voice-cleanup.json").exists())
        self.assertEqual([request.get_method() for request in http.requests], ["GET"])

    def test_pending_reconciliation_accepts_full_model_resource_but_not_similar_model(self):
        pending = self.pending_candidate()
        matching = {"id": "voice_full_model", "display_name": pending["display_name"],
                    "model": "models/" + GEMINI_DESIGN_MODEL, "type": "prompted"}
        http = self.http({"voices": [
            dict(matching, id="voice_similar_model", model=matching["model"] + "-preview"),
            dict(matching, id="voice_different_model", model="models/another-model"),
            matching,
        ]}, b"")

        self.assertEqual(self.designer.release_voice(self.directory), ["voice_full_model"])

        deleted = [request for request in http.requests if request.get_method() == "DELETE"]
        self.assertEqual(len(deleted), 1)
        self.assertEqual(urlsplit(deleted[0].full_url).path, "/v1beta/voices/voice_full_model")

    def test_pending_cleanup_recovers_when_final_marker_write_failed_after_delete(self):
        pending = self.pending_candidate()
        http = self.http({"voices": [{
            "id": "voice_discovered", "display_name": pending["display_name"],
            "model": GEMINI_DESIGN_MODEL, "type": "prompted",
        }]}, b"", _http_error(404))
        link = os.link

        def fail_final_marker(source, target, *args, **kwargs):
            if Path(target).name == "gemini-voice-cleanup.json":
                raise OSError("Temporary local disk failure")
            return link(source, target, *args, **kwargs)

        with patch("os.link", side_effect=fail_final_marker):
            with self.assertRaises(OSError):
                self.designer.release_voice(self.directory)
        self.assertFalse((self.directory / "gemini-voice-cleanup.json").exists())

        self.assertEqual(self.designer.release_voice(self.directory), ["voice_discovered"])

        self.assertEqual([request.get_method() for request in http.requests], ["GET", "DELETE", "DELETE"])
        cleanup = json.loads((self.directory / "gemini-voice-cleanup.json").read_text())
        self.assertTrue(cleanup["remote_deleted"])
        self.assertEqual(cleanup["voice_ids"], ["voice_discovered"])

    def test_pending_cleanup_uses_original_model_after_configuration_changes(self):
        pending = self.pending_candidate()
        self.designer = GeminiVoiceDesigner(self.key, model="gemini-other-tts")
        matching = {"id": "voice_original_model", "display_name": pending["display_name"],
                    "model": "models/" + GEMINI_DESIGN_MODEL, "type": "prompted"}
        http = self.http({"voices": [
            dict(matching, id="voice_changed_model", model=self.designer.model), matching,
        ]}, b"")

        self.assertEqual(self.designer.release_voice(self.directory), ["voice_original_model"])

        deleted = [request for request in http.requests if request.get_method() == "DELETE"]
        self.assertEqual(len(deleted), 1)
        self.assertEqual(urlsplit(deleted[0].full_url).path, "/v1beta/voices/voice_original_model")

    def test_manually_confirmed_absence_of_unknown_pending_voice_needs_no_delete(self):
        pending = self.pending_candidate()
        (self.directory / "gemini-voice-cleanup.json").write_text(json.dumps({
            "version": 1, "request_fingerprint": pending["request_fingerprint"],
            "voice_ids": [], "remote_deleted": True, "confirmed_absent": True,
        }))
        http = self.http()

        self.assertEqual(self.designer.release_voice(self.directory), [])

        self.assertEqual(http.requests, [])

    def test_empty_cleanup_record_cannot_claim_success_without_explicit_confirmation(self):
        pending = self.pending_candidate()
        (self.directory / "gemini-voice-cleanup.json").write_text(json.dumps({
            "version": 1, "request_fingerprint": pending["request_fingerprint"],
            "voice_ids": [], "remote_deleted": True,
        }))
        http = self.http()

        with self.assertRaises(GeminiVoiceError):
            self.designer.release_voice(self.directory)

        self.assertEqual(http.requests, [])

    def test_no_creation_state_requires_no_remote_requests(self):
        http = self.http()

        self.assertEqual(self.designer.release_voice(self.directory), [])

        self.assertEqual(http.requests, [])
        self.assertFalse(self.directory.exists())

    def test_repeated_page_token_stops_instead_of_hanging_cleanup(self):
        http = self.http(
            {"voices": [], "next_page_token": "same-page"},
            {"voices": [], "next_page_token": "same-page"},
        )

        with self.assertRaisesRegex(GeminiVoiceError, "pagination"):
            self.designer.list_voices()

        self.assertEqual(len(http.requests), 2)

    def test_malformed_pending_inventory_never_deletes_partially_listed_voices(self):
        pending = self.pending_candidate()
        http = self.http({"voices": [
            {"id": "voice_valid", "display_name": pending["display_name"],
             "model": GEMINI_DESIGN_MODEL, "type": "prompted"},
            {"id": "voice_bad/path", "display_name": pending["display_name"],
             "model": GEMINI_DESIGN_MODEL, "type": "prompted"},
        ]})

        with self.assertRaises(GeminiVoiceError):
            self.designer.release_voice(self.directory)

        self.assertEqual([request.get_method() for request in http.requests], ["GET"])
        self.assertFalse((self.directory / "gemini-voice-cleanup.json").exists())

    def test_cleanup_refuses_symlinked_state_without_deleting_remote_voice(self):
        self.saved_candidate()
        state_path = self.directory / "gemini-voice.json"
        original_path = self.directory.parent / "original-state.json"
        state_path.rename(original_path)
        state_path.symlink_to(original_path)
        http = self.http()

        with self.assertRaises(GeminiVoiceError):
            self.designer.release_voice(self.directory)

        self.assertEqual(http.requests, [])
        self.assertTrue(original_path.is_file())
        self.assertFalse((self.directory / "gemini-voice-cleanup.json").exists())

    def test_mismatched_cleanup_record_cannot_skip_a_live_voice(self):
        self.saved_candidate()
        (self.directory / "gemini-voice-cleanup.json").write_text(json.dumps({
            "version": 1, "request_fingerprint": "0" * 64,
            "voice_ids": ["voice_some_other_candidate"], "remote_deleted": True,
        }))
        http = self.http()

        with self.assertRaises(GeminiVoiceError):
            self.designer.release_voice(self.directory)

        self.assertEqual(http.requests, [])


if __name__ == "__main__":
    unittest.main()
