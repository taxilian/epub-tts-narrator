"""Automatic reference creation releases hosted Gemini voices, retaining local audio."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import workflow
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.gemini_voice_design import GeminiVoiceDesigner, GeminiVoiceError
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voices import VoiceStore
from test_renderer_voice_design_integration import (
    FFMPEG, FFPROBE, ReferenceDesigner, ReferencePlaybackEngine, TranscriptVerifier,
)


class FixtureGemini(GeminiVoiceDesigner):
    """Real request-state/cleanup lifecycle with in-memory remote calls and PCM."""

    def __init__(self, transcripts):
        super().__init__("fixture-gemini-secret")
        self.audio = ReferenceDesigner(transcripts)
        self.hosted = set()
        self.events = []
        self.created = 0
        self.cleanup_error = None
        self.generation_error = None
        self.before_delete = None

    def _post(self, endpoint, payload):
        if endpoint != "/voices":
            raise AssertionError("Unexpected remote endpoint")
        self.created += 1
        identifier = "voice_fixture_" + str(self.created)
        self.hosted.add(identifier)
        self.events.append(("create", identifier))
        return {"id": identifier}

    def delete_voice(self, identifier):
        self.events.append(("delete", identifier))
        if self.before_delete:
            self.before_delete(identifier)
        if self.cleanup_error:
            raise self.cleanup_error
        self.hosted.discard(identifier)
        return True

    def list_voices(self, *, search=None):
        raise AssertionError("Known references do not need a remote search")

    def create_voice(self, prompt, text, path, *, seed):
        identifier = self._remote_voice(prompt, Path(path).parent)
        self.events.append(("synthesize", identifier))
        if self.generation_error:
            raise self.generation_error
        self.audio.write_audio(text, path, seed)
        return {"description_applied": True, "remote_voice_id": identifier,
                "requested_description": prompt, "reference_text": text,
                "designer": self.identity}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for reference verification")
class AutomaticGeminiCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="fixture-series")
        self.config["render"].update(max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["gemini"]["api_key"] = "fixture-gemini-secret"
        self.config["quality"] = validate_policy({})
        self.work = self.root / "work"
        self.transcripts = {}
        self.engine = ReferencePlaybackEngine(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.remote = FixtureGemini(self.transcripts)
        self.store = VoiceStore(self.root / "voices", "fixture-series")
        self.voice_id = "voice_fixture"
        self.profile = {"id": self.voice_id, "description": "A warm, clear adult narrator.", "traits": []}
        self.messages = []

    def ensure(self, **kwargs):
        lock = self.store.lock()
        try:
            return workflow.ensure_voice(
                self.store, self.voice_id, self.profile, self.engine, self.verifier,
                self.config, self.work, designer=self.remote, notify=self.messages.append, **kwargs)
        finally:
            lock.rmdir()

    def directories(self):
        return sorted((self.work / "voice-candidates").iterdir())

    def test_success_deletes_after_publication_and_reuses_local_reference(self):
        self.remote.before_delete = lambda _: self.assertIsNotNone(self.store.load(self.voice_id))
        selected = self.ensure()
        directory, = self.directories()
        self.assertTrue(Path(selected["reference_audio"]).is_file())
        self.assertTrue((directory / "accepted.json").is_file())
        self.assertTrue(read_json(directory / "gemini-voice-cleanup.json")["remote_deleted"])
        self.assertEqual(self.remote.events, [("create", "voice_fixture_1"),
                                             ("synthesize", "voice_fixture_1"),
                                             ("delete", "voice_fixture_1")])
        events, calls = list(self.remote.events), list(self.verifier.calls)
        self.assertEqual(self.ensure(), selected)
        self.assertEqual(self.remote.events, events)
        self.assertEqual(self.verifier.calls, calls)

    def test_failed_verification_cleans_once_after_retries_then_uses_new_remote_voice(self):
        self.verifier.wrong = True
        with self.assertRaises(workflow.SpeechVerificationFailed):
            self.ensure()
        first_directory, = self.directories()
        self.assertIsNone(self.store.load(self.voice_id))
        self.assertEqual(self.remote.events, [("create", "voice_fixture_1"),
                                             ("synthesize", "voice_fixture_1"),
                                             ("synthesize", "voice_fixture_1"),
                                             ("delete", "voice_fixture_1")])
        old_audio = {path.name: path.read_bytes() for path in first_directory.glob("*.wav")}
        self.verifier.wrong = False
        selected = self.ensure()
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_fixture_2")
        self.assertEqual(self.remote.created, 2)
        self.assertEqual(len(self.directories()), 2)
        self.assertEqual({path.name: path.read_bytes() for path in first_directory.glob("*.wav")}, old_audio)
        self.assertFalse(self.remote.hosted)

    def test_generation_interrupt_cleans_and_preserves_interrupt(self):
        self.remote.generation_error = KeyboardInterrupt("fixture interrupted")
        with self.assertRaisesRegex(KeyboardInterrupt, "fixture interrupted"):
            self.ensure()
        self.assertFalse(self.remote.hosted)
        self.assertEqual(self.remote.events[-1], ("delete", "voice_fixture_1"))

    def test_failed_cleanup_does_not_mask_failure_and_retry_blocks_more_creation(self):
        self.verifier.wrong = True
        self.remote.cleanup_error = GeminiVoiceError("fixture-gemini-secret cleanup unavailable")
        with self.assertRaises(workflow.SpeechVerificationFailed):
            self.ensure()
        self.assertNotIn("fixture-gemini-secret", "\n".join(self.messages))
        self.assertEqual(len(list(self.work.rglob("gemini-cleanup-pending.json"))), 1)
        with self.assertRaisesRegex(workflow.RenderBlocked, "cleanup is still pending"):
            self.ensure()
        self.assertEqual(self.remote.created, 1)
        self.remote.cleanup_error = None
        self.verifier.wrong = False
        self.ensure()
        self.assertEqual(self.remote.created, 2)
        self.assertFalse(list(self.work.rglob("gemini-cleanup-pending.json")))

    def test_failed_cleanup_returns_valid_reference_and_retries_without_generation(self):
        self.remote.cleanup_error = GeminiVoiceError("Deletion temporarily unavailable")
        selected = self.ensure()
        self.assertTrue(Path(selected["reference_audio"]).is_file())
        self.assertIn("voice_fixture_1", self.remote.hosted)
        self.remote.cleanup_error = None
        calls = list(self.verifier.calls)
        self.assertEqual(self.ensure(), selected)
        self.assertEqual(self.remote.created, 1)
        self.assertEqual(self.verifier.calls, calls)
        self.assertFalse(self.remote.hosted)
        self.assertFalse(list(self.work.rglob("gemini-cleanup-pending.json")))

    def test_hard_crash_before_cleanup_marker_retries_for_saved_reference(self):
        # A completed local publication can outlive an interrupted finally block.
        with patch.object(self.remote, "release_voice", return_value=[]):
            selected = self.ensure()
        directory, = self.directories()
        self.assertFalse((directory / "gemini-cleanup-pending.json").exists())
        self.assertFalse((directory / "gemini-voice-cleanup.json").exists())
        self.assertIn("voice_fixture_1", self.remote.hosted)
        self.assertEqual(self.ensure(), selected)
        self.assertFalse(self.remote.hosted)
        self.assertEqual(self.remote.created, 1)

    def test_legacy_crash_recovers_original_request_from_accepted_record(self):
        with patch.object(self.remote, "release_voice", return_value=[]):
            selected = self.ensure()
        directory, = self.directories()
        (directory / "voice-request.json").unlink()
        self.assertEqual(self.ensure(), selected)
        self.assertFalse(self.remote.hosted)
        self.assertEqual(self.remote.created, 1)

    def test_publication_failure_cleans_and_reuses_accepted_audio_on_retry(self):
        with patch.object(self.store, "publish", side_effect=OSError("fixture disk error")), \
                self.assertRaisesRegex(OSError, "fixture disk error"):
            self.ensure()
        self.assertFalse(self.remote.hosted)
        events = list(self.remote.events)
        selected = self.ensure()
        self.assertTrue(Path(selected["reference_audio"]).is_file())
        self.assertEqual(self.remote.events, events)

    def test_interruption_during_asr_creates_new_bound_reference_on_resume(self):
        with patch.object(workflow, "validate_audio", side_effect=KeyboardInterrupt("fixture ASR interruption")), \
                self.assertRaises(KeyboardInterrupt):
            self.ensure()
        self.assertFalse(self.remote.hosted)
        selected = self.ensure()
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_fixture_2")
        self.assertTrue(selected["description_applied"])
        self.assertEqual(self.remote.created, 2)

    def test_cleanup_uses_gemini_credentials_after_switching_designers(self):
        self.remote.cleanup_error = GeminiVoiceError("Deletion temporarily unavailable")
        self.ensure()
        self.remote.cleanup_error = None
        self.config["voice_design"] = {"type": "elevenlabs", "api_key": "elevenlabs-only-secret"}
        settings_seen = []

        def create(settings):
            settings_seen.append(deepcopy(settings))
            return self.remote

        with patch("epub_tts_narrator.renderer.voice_design.create_designer", side_effect=create):
            lock = self.store.lock()
            try:
                workflow.ensure_voice(self.store, self.voice_id, self.profile, self.engine, self.verifier,
                                      self.config, self.work, notify=self.messages.append)
            finally:
                lock.rmdir()
        self.assertEqual(settings_seen[0]["api_key"], "fixture-gemini-secret")
        self.assertNotIn("elevenlabs-only-secret", repr(settings_seen))
        self.assertFalse(self.remote.hosted)

    def test_adding_book_fingerprint_preserves_accepted_legacy_reference(self):
        with patch.object(self.store, "publish", side_effect=OSError("fixture disk error")), \
                self.assertRaises(OSError):
            self.ensure()
        directory, = self.directories()
        (directory / "voice-request.json").unlink()
        events = list(self.remote.events)
        # The pre-fingerprint request did not carry a selected sample source.
        with patch("epub_tts_narrator.renderer.samples.select_sample", return_value={
                "reference_text": self.config["voices"]["reference_text"], "source": None}):
            selected = self.ensure(bundle={"manifest": {"source_fingerprint": "book-one"}})
        self.assertTrue(Path(selected["reference_audio"]).is_file())
        self.assertEqual(self.remote.events, events)

    def make_pending(self):
        pending = {"passed": False, "reasons": ["text_mismatch"],
                   "text_judge": {"decision": "unavailable"}}
        with patch.object(workflow, "validate_audio", return_value=pending), \
                self.assertRaises(workflow.VerificationUnavailable):
            self.ensure()
        directory, = self.directories()
        self.assertFalse(self.remote.hosted)
        return directory

    def test_pending_verification_rechecks_local_audio_after_remote_deletion(self):
        directory = self.make_pending()
        original = (directory / "attempt-001.wav").read_bytes()
        events = list(self.remote.events)
        selected = self.ensure()
        self.assertEqual(Path(selected["reference_audio"]).read_bytes(), original)
        self.assertEqual(self.remote.events, events)

    def test_rejected_pending_audio_creates_a_fresh_voice_without_reusing_deleted_id(self):
        self.make_pending()
        validate = workflow.validate_audio
        rechecked = []

        def reject_once(*args, **kwargs):
            if not rechecked:
                rechecked.append(True)
                return {"passed": False, "reasons": ["text_mismatch"]}
            return validate(*args, **kwargs)

        with patch.object(workflow, "validate_audio", side_effect=reject_once):
            self.ensure()
        self.assertEqual(self.remote.created, 2)
        self.assertEqual(self.remote.events.count(("synthesize", "voice_fixture_1")), 1)
        self.assertEqual(self.remote.events.count(("synthesize", "voice_fixture_2")), 1)
        self.assertFalse(self.remote.hosted)

    def test_cleanup_ignores_another_books_recorded_source_fingerprint(self):
        with patch.object(self.remote, "release_voice", return_value=[]), \
                patch("epub_tts_narrator.renderer.samples.select_sample", return_value={
                    "reference_text": self.config["voices"]["reference_text"], "source": {"method": "fixture"}}):
            self.ensure(bundle={"manifest": {"source_fingerprint": "book-one"}})
        self.ensure(bundle={"manifest": {"source_fingerprint": "book-two"}})
        self.assertIn("voice_fixture_1", self.remote.hosted)
        self.assertNotIn(("delete", "voice_fixture_1"), self.remote.events)

    def test_corrupt_original_request_is_not_used_to_delete_remote_voice(self):
        with patch.object(self.remote, "release_voice", return_value=[]):
            selected = self.ensure()
        directory, = self.directories()
        path = directory / "voice-request.json"
        request = read_json(path)
        request["voice_id"] = "changed"
        write_json(path, request)
        self.assertEqual(self.ensure(), selected)
        self.assertIn("voice_fixture_1", self.remote.hosted)
        self.assertTrue(any("integrity" in message for message in self.messages))


if __name__ == "__main__":
    unittest.main()
