"""Temporary Gemini voices are released once casting stops, not between takes."""

import base64
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import workflow
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.gemini_voice_design import GeminiVoiceDesigner, GeminiVoiceError, _RequestFailure
from epub_tts_narrator.renderer.workflow import RenderBlocked, VerificationUnavailable
import test_renderer_gemini_fresh_regeneration as fixtures
from test_renderer_voice_design_integration import FFMPEG


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting integration")
class CastingGeminiLifecycleTests(unittest.TestCase):
    setUp = fixtures.GeminiFreshRegenerationTests.setUp
    session = fixtures.GeminiFreshRegenerationTests.session

    def remote(self, *, synthesis_errors=(), cleanup_errors=(), creation_error=None):
        remote = GeminiVoiceDesigner("fixture-credential-never-sent")
        self.events, self.hosted = [], {}
        synthesis_errors, cleanup_errors = list(synthesis_errors), list(cleanup_errors)

        def post(endpoint, payload):
            if endpoint == "/voices":
                identifier = f"voice_test_{len([event for event in self.events if event[0] == 'create']) + 1}"
                self.events.append(("create", identifier))
                self.hosted[identifier] = dict(payload["voice"], id=identifier)
                if creation_error is not None:
                    raise creation_error
                return {"id": identifier}
            self.assertEqual(endpoint, "/interactions")
            identifier = payload["generation_config"]["speech_config"][0]["voice"]
            self.assertIn(identifier, self.hosted, "A deleted voice was used for synthesis")
            self.events.append(("synthesize", identifier))
            if synthesis_errors:
                raise synthesis_errors.pop(0)
            text = payload["input"][0]["content"][0]["text"]
            audio = self.root / f"remote-{len(self.events)}.wav"
            self.audio.write_audio(text, audio, payload["generation_config"]["seed"])
            return {"status": "completed", "steps": [{"type": "model_output", "content": [{
                "type": "audio", "mime_type": "audio/wav",
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
            }]}]}

        def delete(identifier):
            self.events.append(("delete", identifier))
            if cleanup_errors:
                raise cleanup_errors.pop(0)
            self.hosted.pop(identifier, None)

        remote._post = post
        remote.delete_voice = delete
        remote.list_voices = lambda **kwargs: list(self.hosted.values())
        return remote

    def test_retries_share_one_remote_voice_then_success_releases_it(self):
        self.config["render"]["max_attempts"] = 3
        remote = self.remote(synthesis_errors=[_RequestFailure("temporary failure"), _RequestFailure("temporary failure")])
        session = self.session(remote)

        selected = session.regenerate("voice_mara")

        self.assertEqual(self.events, [("create", "voice_test_1"),
                                     *[("synthesize", "voice_test_1")] * 3, ("delete", "voice_test_1")])
        self.assertEqual(self.hosted, {})
        self.assertTrue(Path(selected["reference_audio"]).is_file())
        self.assertEqual(session.ensure_ready("voice_mara"), selected)
        self.assertEqual(len(self.events), 5)
        self.assertEqual(len(list(self.root.rglob("gemini-voice-cleanup.json"))), 1)

    def test_rejected_samples_release_voice_only_after_all_attempts(self):
        self.config["render"]["max_attempts"] = 3
        remote = self.remote()
        self.verifier.wrong = True
        session = self.session(remote)

        with self.assertRaisesRegex(RenderBlocked, "after 3 attempts"):
            session.ensure_ready("voice_mara")

        self.assertEqual(self.events, [("create", "voice_test_1"),
                                     *[("synthesize", "voice_test_1")] * 3, ("delete", "voice_test_1")])
        self.assertIsNone(session.store.load("voice_mara"))
        self.assertEqual(self.hosted, {})

    def test_interruption_releases_created_voice_without_masking_interrupt(self):
        interrupted = KeyboardInterrupt("user stopped casting")
        remote = self.remote(synthesis_errors=[interrupted])

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.session(remote).regenerate("voice_mara")

        self.assertIs(caught.exception, interrupted)
        self.assertEqual(self.events[-1], ("delete", "voice_test_1"))
        self.assertEqual(self.hosted, {})

    def test_publication_failure_still_releases_remote_voice(self):
        remote = self.remote()
        session = self.session(remote)
        error = OSError("local publication failed")
        with patch.object(session.store, "publish_candidate", side_effect=error), \
             self.assertRaises(OSError) as caught:
            session.regenerate("voice_mara")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.events[-1], ("delete", "voice_test_1"))
        self.assertEqual(self.hosted, {})

    def test_cleanup_failure_keeps_selected_sample_and_next_resume_retries_without_models(self):
        remote = self.remote(cleanup_errors=[GeminiVoiceError("delete temporarily failed")])
        session = self.session(remote)
        selected = session.regenerate("voice_mara")
        audio = Path(selected["reference_audio"]).read_bytes()
        markers = list(self.root.rglob("gemini-cleanup-pending.json"))
        self.assertEqual(len(markers), 1)
        self.assertIn("cleanup failed", "\n".join(self.notices))
        self.assertEqual(session.store.load("voice_mara"), selected)

        resumed = self.session(remote)
        with patch.object(resumed, "_get_engine", side_effect=AssertionError("No playback model needed")), \
             patch.object(resumed, "_get_verifier", side_effect=AssertionError("No ASR needed")):
            self.assertEqual(resumed.ensure_ready("voice_mara"), selected)

        self.assertFalse(markers[0].exists())
        self.assertEqual(Path(selected["reference_audio"]).read_bytes(), audio)
        self.assertEqual(self.events, [("create", "voice_test_1"), ("synthesize", "voice_test_1"),
                                      ("delete", "voice_test_1"), ("delete", "voice_test_1")])

    def test_crash_after_publication_is_cleaned_on_resume_without_recasting(self):
        remote = self.remote()
        session = self.session(remote)
        with patch.object(session, "_release_gemini_voice", return_value=True):
            selected = session.regenerate("voice_mara")
        self.assertFalse(list(self.root.rglob("gemini-voice-cleanup-request.json")))
        audio = Path(selected["reference_audio"]).read_bytes()

        resumed = self.session(remote)
        with patch.object(resumed, "_get_engine", side_effect=AssertionError("No model needed")):
            self.assertEqual(resumed.ensure_ready("voice_mara"), selected)

        self.assertEqual(self.events, [("create", "voice_test_1"), ("synthesize", "voice_test_1"),
                                      ("delete", "voice_test_1")])
        self.assertEqual(Path(selected["reference_audio"]).read_bytes(), audio)
        self.assertEqual(self.hosted, {})

    def test_startup_sweep_does_not_clean_another_books_remote_voice(self):
        remote = self.remote()
        session = self.session(remote)
        with patch.object(session, "_release_gemini_voice", return_value=True):
            selected = session.regenerate("voice_mara")
        events = deepcopy(self.events)
        resumed = self.session(remote)
        resumed.bundle = deepcopy(resumed.bundle)
        resumed.bundle["manifest"]["source_fingerprint"] = "another-book-fingerprint"

        self.assertEqual(resumed.ensure_ready("voice_mara"), selected)

        self.assertEqual(self.events, events)
        self.assertIn("voice_test_1", self.hosted)

    def test_unmarked_pending_verification_cleans_remote_before_reusing_saved_audio(self):
        remote = self.remote()
        from epub_tts_narrator.renderer.casting import CastingSession
        with patch.object(CastingSession, "_release_gemini_voice", return_value=True):
            path, record = self.make_pending(remote)
        original = (path.parent / record["audio_file"]).read_bytes()
        self.assertFalse(list(self.root.rglob("gemini-voice-cleanup.json")))

        selected = self.session(remote).ensure_ready("voice_mara")

        self.assertEqual(selected["candidate_id"], record["request"]["nonce"])
        self.assertEqual(Path(selected["reference_audio"]).read_bytes(), original)
        self.assertEqual(self.events, [("create", "voice_test_1"), ("synthesize", "voice_test_1"),
                                      ("delete", "voice_test_1")])

    def test_crash_after_cleanup_intent_retries_without_a_failure_marker(self):
        remote = self.remote(cleanup_errors=[GeminiVoiceError("delete temporarily failed")])
        selected = self.session(remote).regenerate("voice_mara")
        marker = next(self.root.rglob("gemini-cleanup-pending.json"))
        journal = marker.parent / "gemini-voice-cleanup-request.json"
        self.assertTrue(journal.is_file())
        marker.unlink()  # A hard crash would not have written this later marker.

        resumed = self.session(remote)
        with patch.object(resumed, "_get_engine", side_effect=AssertionError("No playback model needed")):
            self.assertEqual(resumed.ensure_ready("voice_mara"), selected)

        self.assertEqual(self.hosted, {})
        self.assertEqual(self.events[-1], ("delete", "voice_test_1"))
        self.assertTrue((journal.parent / "gemini-voice-cleanup.json").is_file())
        events = deepcopy(self.events)
        restarted = self.session(remote)
        with patch.object(restarted, "_designer_factory", side_effect=AssertionError("Already cleaned")):
            self.assertEqual(restarted.ensure_ready("voice_mara"), selected)
        self.assertEqual(self.events, events)

    def test_changed_cleanup_journal_is_not_used_for_remote_deletion(self):
        remote = self.remote(cleanup_errors=[GeminiVoiceError("delete temporarily failed")])
        selected = self.session(remote).regenerate("voice_mara")
        marker = next(self.root.rglob("gemini-cleanup-pending.json"))
        journal = marker.parent / "gemini-voice-cleanup-request.json"
        marker.unlink()
        record = read_json(journal)
        record["request_fingerprint"] = "0" * 64
        write_json(journal, record)
        events = deepcopy(self.events)

        self.assertEqual(self.session(remote).ensure_ready("voice_mara"), selected)

        self.assertEqual(self.events, events)
        self.assertIn("Could not retry Gemini voice cleanup", "\n".join(self.notices))
        self.assertTrue(journal.exists())

    def test_cleanup_failure_does_not_replace_primary_failure(self):
        remote = self.remote(cleanup_errors=[GeminiVoiceError("cleanup failed separately")])
        self.verifier.wrong = True
        with self.assertRaisesRegex(RenderBlocked, "Speech verification failed"):
            self.session(remote).ensure_ready("voice_mara")
        self.assertIn("cleanup failed separately", "\n".join(self.notices))
        self.assertEqual(len(list(self.root.rglob("gemini-cleanup-pending.json"))), 1)

    def test_fresh_regeneration_blocks_unknown_creation_if_cleanup_still_fails(self):
        remote = self.remote(creation_error=_RequestFailure("lost creation response"),
                             cleanup_errors=[GeminiVoiceError("cleanup unavailable")] * 2)
        with self.assertRaisesRegex(RenderBlocked, "after 1 attempts"):
            self.session(remote).ensure_ready("voice_mara")

        with self.assertRaises(RenderBlocked) as blocked:
            self.session(remote).regenerate("voice_mara")
        self.assertIn("cleanup is still pending", str(blocked.exception))
        self.assertIn("provider error", str(blocked.exception))
        self.assertIn("gemini-voice-pending.json", str(blocked.exception))

        self.assertEqual(self.events, [("create", "voice_test_1"), ("delete", "voice_test_1"),
                                      ("delete", "voice_test_1")])
        self.assertTrue(list(self.root.rglob("gemini-cleanup-pending.json")))

    def test_fresh_regeneration_can_continue_after_known_voice_cleanup_failure(self):
        remote = self.remote(cleanup_errors=[GeminiVoiceError("cleanup unavailable")] * 2)
        first = self.session(remote).regenerate("voice_mara")

        second = self.session(remote).regenerate("voice_mara")

        self.assertNotEqual(first["candidate_id"], second["candidate_id"])
        self.assertTrue(Path(first["reference_audio"]).exists())
        self.assertTrue(Path(second["reference_audio"]).exists())
        self.assertEqual(self.events, [("create", "voice_test_1"), ("synthesize", "voice_test_1"),
                                      ("delete", "voice_test_1"), ("delete", "voice_test_1"),
                                      ("create", "voice_test_2"), ("synthesize", "voice_test_2"),
                                      ("delete", "voice_test_2")])
        self.assertIn("voice_test_1", self.hosted)
        self.assertNotIn("voice_test_2", self.hosted)

    def test_cleanup_retry_uses_original_model_after_global_model_change(self):
        remote = self.remote(cleanup_errors=[GeminiVoiceError("delete temporarily failed")])
        selected = self.session(remote).regenerate("voice_mara")
        resumed = self.session(remote)
        resumed.config["gemini"]["model"] = "gemini-new-design-model"
        resumed.config["voice_design"]["model"] = "gemini-new-design-model"
        settings_seen = []
        resumed._designer_factory = lambda settings: settings_seen.append(deepcopy(settings)) or remote

        self.assertEqual(resumed.ensure_ready("voice_mara"), selected)

        self.assertEqual(settings_seen[0]["model"], remote.identity["model"])
        self.assertEqual(resumed.config["voice_design"]["model"], "gemini-new-design-model")
        self.assertFalse(list(self.root.rglob("gemini-cleanup-pending.json")))

    def make_pending(self, remote):
        session = self.session(remote)
        def unavailable(path, text, *args, **kwargs):
            return {"passed": False, "purpose": "reference", "expected_text": text,
                    "reasons": ["text_mismatch"], "text_judge": {"decision": "unavailable"}}
        with patch.object(workflow, "validate_audio", side_effect=unavailable), \
             self.assertRaises(VerificationUnavailable):
            session.ensure_ready("voice_mara")
        record_path = next(self.root.rglob("attempt-001.json"))
        return record_path, read_json(record_path)

    def test_deleted_remote_voice_does_not_prevent_saved_sample_verification(self):
        remote = self.remote()
        path, record = self.make_pending(remote)
        original = (path.parent / record["audio_file"]).read_bytes()
        events = deepcopy(self.events)

        selected = self.session(remote).ensure_ready("voice_mara")

        self.assertEqual(selected["candidate_id"], record["request"]["nonce"])
        self.assertEqual(Path(selected["reference_audio"]).read_bytes(), original)
        self.assertEqual(self.events, events)

    def test_rejected_saved_sample_starts_fresh_request_after_remote_cleanup(self):
        self.config["render"]["max_attempts"] = 3
        remote = self.remote()
        _, record = self.make_pending(remote)
        validate = workflow.validate_audio
        count = 0
        def reject_saved_once(path, text, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return {"passed": False, "purpose": "reference", "expected_text": text,
                        "reasons": ["text_mismatch"]}
            return validate(path, text, *args, **kwargs)
        with patch.object(workflow, "validate_audio", side_effect=reject_saved_once):
            selected = self.session(remote).ensure_ready("voice_mara")
        self.assertNotEqual(selected["candidate_id"], record["request"]["nonce"])
        self.assertEqual(self.events, [("create", "voice_test_1"), ("synthesize", "voice_test_1"), ("delete", "voice_test_1"),
                                      ("create", "voice_test_2"), ("synthesize", "voice_test_2"), ("delete", "voice_test_2")])

    def test_preparation_reconciles_only_matching_legacy_pending_creation(self):
        remote = self.remote(creation_error=_RequestFailure("lost creation response"))
        session = self.session(remote)
        with patch.object(session, "_release_gemini_voice", return_value=True), \
             self.assertRaisesRegex(RenderBlocked, "after 1 attempts"):
            session.ensure_ready("voice_mara")
        pending = next(self.root.rglob("gemini-voice-pending.json"))
        self.assertEqual(self.events, [("create", "voice_test_1")])
        # The legacy request's unknown outcome is reconciled before creating a
        # replacement; retain its state as an audit of the deleted remote ID.
        resumed = self.session(remote)
        with self.assertRaisesRegex(RenderBlocked, "after 1 attempts"):
            resumed.ensure_ready("voice_mara")
        self.assertEqual(self.events[:3], [("create", "voice_test_1"), ("delete", "voice_test_1"), ("create", "voice_test_2")])
        self.assertTrue((pending.parent / "gemini-voice-cleanup.json").exists())

    def test_legacy_pending_creation_uses_original_model_during_preparation(self):
        remote = self.remote(creation_error=_RequestFailure("lost creation response"))
        session = self.session(remote)
        with patch.object(session, "_release_gemini_voice", return_value=True), \
             self.assertRaisesRegex(RenderBlocked, "after 1 attempts"):
            session.ensure_ready("voice_mara")
        pending = next(self.root.rglob("gemini-voice-pending.json"))
        state = read_json(pending)
        state.pop("model", None)  # Legacy states need their casting request.
        write_json(pending, state)
        newer = GeminiVoiceDesigner("fixture-credential-never-sent", model="gemini-new-design-model")
        newer._post, newer.delete_voice, newer.list_voices = remote._post, remote.delete_voice, remote.list_voices
        resumed = self.session(newer)
        resumed.config["gemini"]["model"] = newer.model
        resumed.config["voice_design"]["model"] = newer.model
        settings_seen = []
        def factory(settings):
            settings_seen.append(settings["model"])
            return remote if settings["model"] == remote.model else newer
        resumed._designer_factory = factory

        # Exercise preparation itself as well as the normal startup sweep.
        with patch.object(resumed, "_retry_gemini_cleanup"), \
             self.assertRaisesRegex(RenderBlocked, "after 1 attempts"):
            resumed.ensure_ready("voice_mara")

        self.assertEqual(settings_seen, [newer.model, remote.model])
        self.assertEqual(self.events[:3], [("create", "voice_test_1"), ("delete", "voice_test_1"),
                                          ("create", "voice_test_2")])
        self.assertTrue((pending.parent / "gemini-voice-cleanup.json").is_file())


if __name__ == "__main__":
    unittest.main()
