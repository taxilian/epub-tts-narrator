"""Explicit recasting creates new remote identities; retrying one take does not."""

import base64
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS, select_voice_designer
from epub_tts_narrator.renderer.gemini_voice_design import GeminiVoiceDesigner, _RequestFailure
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.cli import export_bundle
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferenceDesigner, ReferencePlaybackEngine, TranscriptVerifier,
)


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting integration")
class GeminiFreshRegenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = deepcopy(DEFAULTS)
        self.config["voices"]["series_id"] = "fresh-gemini-fixture"
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.config["gemini"]["api_key"] = "fixture-credential-never-sent"
        self.config = select_voice_designer(self.config, "gemini")
        self.transcripts = {}
        self.engine = ReferencePlaybackEngine(self.transcripts)
        self.audio = ReferenceDesigner(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        directory = self.root / "book"
        export_bundle(read_json(EXAMPLE / "reviewed.json"), directory)
        self.bundle = prepare_bundle(load_bundle(directory))
        self.calls = []
        self.notices = []

    def designer(self, creation_responses, *, interaction_errors=()):
        remote = GeminiVoiceDesigner("fixture-credential-never-sent")
        creations, errors = list(creation_responses), list(interaction_errors)

        def post(endpoint, payload):
            self.calls.append((endpoint, deepcopy(payload)))
            if endpoint == "/voices":
                self.assertTrue(creations, "Unexpected new remote voice creation")
                response = creations.pop(0)
                if isinstance(response, Exception):
                    raise response
                return response
            self.assertEqual(endpoint, "/interactions")
            if errors:
                raise errors.pop(0)
            text = payload["input"][0]["content"][0]["text"]
            audio = self.root / f"remote-{len(self.calls)}.wav"
            self.audio.write_audio(text, audio, payload["generation_config"]["seed"])
            return {"status": "completed", "steps": [{"type": "model_output", "content": [{
                "type": "audio", "mime_type": "audio/wav",
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
            }]}]}

        remote._post = post
        return remote

    def session(self, designer):
        session = CastingSession(
            self.bundle, self.config, engine_factory=lambda _: self.engine,
            verifier_factory=lambda *_args, **_kwargs: self.verifier,
            designer_factory=lambda _: designer, notify=self.notices.append,
        )
        self.addCleanup(session.close)
        return session

    def pending_creation(self):
        designer = self.designer([_RequestFailure("Fixture creation response was lost")])
        session = self.session(designer)
        with self.assertRaisesRegex(ValueError, "may already have succeeded"):
            session.ensure_ready("voice_mara")
        pending = list(self.root.rglob("gemini-voice-pending.json"))
        self.assertEqual(len(pending), 1)
        return pending[0]

    def test_same_prompt_explicit_regeneration_creates_two_different_remote_voices(self):
        session = self.session(self.designer([{"id": "voice_first"}, {"id": "voice_second"}]))
        session.update_prompt("voice_mara", "A calm narrator with a gentle rasp.")
        first = session.regenerate("voice_mara")
        original_audio = Path(first["reference_audio"]).read_bytes()
        second = session.regenerate("voice_mara")
        self.assertEqual(first["creation"]["remote_voice_id"], "voice_first")
        self.assertEqual(second["creation"]["remote_voice_id"], "voice_second")
        self.assertNotEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(Path(first["reference_audio"]).read_bytes(), original_audio)
        self.assertEqual([endpoint for endpoint, _ in self.calls],
                         ["/voices", "/interactions", "/voices", "/interactions"])
        self.assertEqual(self.calls[0][1]["voice"]["prompted"], self.calls[2][1]["voice"]["prompted"])
        self.assertEqual(session.store.load("voice_mara")["creation"]["remote_voice_id"], "voice_second")
        self.assertIn("Selected Gemini voice ID: voice_first", self.notices)
        self.assertIn("Selected Gemini voice ID: voice_second", self.notices)

    def test_explicit_regeneration_leaves_repaired_pending_untouched_and_creates_new_remote_voice(self):
        pending = self.pending_creation()
        original_request = (pending.parent / "casting-request.json").read_bytes()
        state = read_json(pending)
        state["voice_id"] = "voice_repaired_old"
        write_json(pending, state)
        original_pending = pending.read_bytes()
        session = self.session(self.designer([{"id": "voice_fresh_replacement"}]))
        selected = session.regenerate("voice_mara")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_fresh_replacement")
        self.assertEqual([endpoint for endpoint, _ in self.calls], ["/voices", "/voices", "/interactions"])
        self.assertEqual(self.calls[-1][1]["generation_config"]["speech_config"],
                         [{"voice": "voice_fresh_replacement"}])
        self.assertEqual(pending.read_bytes(), original_pending)
        self.assertFalse((pending.parent / "gemini-voice.json").exists())
        self.assertEqual((pending.parent / "casting-request.json").read_bytes(), original_request)

    def test_explicit_regeneration_creates_fresh_voice_despite_unknown_prior_outcome(self):
        pending = self.pending_creation()
        before = pending.read_bytes()
        original_request = (pending.parent / "casting-request.json").read_bytes()
        session = self.session(self.designer([{"id": "voice_intentional_new"}]))
        selected = session.regenerate("voice_mara")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_intentional_new")
        self.assertEqual([endpoint for endpoint, _ in self.calls], ["/voices", "/voices", "/interactions"])
        self.assertEqual(pending.read_bytes(), before)
        self.assertEqual((pending.parent / "casting-request.json").read_bytes(), original_request)

    def test_startup_still_blocks_when_prior_creation_outcome_is_unknown(self):
        pending = self.pending_creation()
        before = pending.read_bytes()
        session = self.session(self.designer([]))
        with self.assertRaisesRegex(ValueError, "may already have succeeded"):
            session.ensure_ready("voice_mara")
        self.assertEqual([endpoint for endpoint, _ in self.calls], ["/voices"])
        self.assertEqual(pending.read_bytes(), before)

    def test_transient_synthesis_retry_reuses_remote_id_within_the_same_take(self):
        self.config["render"]["max_attempts"] = 2
        session = self.session(self.designer(
            [{"id": "voice_one_take"}],
            interaction_errors=[_RequestFailure("Fixture synthesis connection closed")],
        ))
        selected = session.regenerate("voice_mara")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_one_take")
        self.assertEqual([endpoint for endpoint, _ in self.calls],
                         ["/voices", "/interactions", "/interactions"])
        self.assertEqual([payload["generation_config"]["speech_config"]
                          for endpoint, payload in self.calls if endpoint == "/interactions"],
                         [[{"voice": "voice_one_take"}], [{"voice": "voice_one_take"}]])

    def test_explicit_regeneration_does_not_look_up_pending_verification(self):
        session = self.session(self.designer([{"id": "voice_explicit_new"}]))
        with patch.object(session, "_pending_confirmation", create=True,
                          side_effect=AssertionError("Explicit recasting must create a fresh take")) as lookup:
            selected = session.regenerate("voice_mara")
        lookup.assert_not_called()
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_explicit_new")

    def test_startup_generation_recovers_known_pending_id_without_new_creation(self):
        pending = self.pending_creation()
        state = read_json(pending)
        state["voice_id"] = "voice_startup_recovered"
        write_json(pending, state)
        session = self.session(self.designer([]))
        selected = session.ensure_ready("voice_mara")
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_startup_recovered")
        self.assertEqual([endpoint for endpoint, _ in self.calls], ["/voices", "/interactions"])
        self.assertFalse(pending.exists())

    def test_startup_recovers_legacy_pending_voice_with_original_unprefixed_description(self):
        original_prompt = "A quiet adult narrator with a measured pace."
        old_session = self.session(self.designer([_RequestFailure("Fixture old creation response lost")]))
        old_session.update_prompt("voice_mara", original_prompt)
        with patch("epub_tts_narrator.renderer.casting.audiobook_voice_prompt", side_effect=lambda prompt: prompt):
            with self.assertRaisesRegex(ValueError, "may already have succeeded"):
                old_session.ensure_ready("voice_mara")
        pending = next(self.root.rglob("gemini-voice-pending.json"))
        original_request = (pending.parent / "casting-request.json").read_bytes()
        state = read_json(pending)
        state["voice_id"] = "voice_legacy_unprefixed"
        write_json(pending, state)
        restarted = self.session(self.designer([]))
        selected = restarted.ensure_ready("voice_mara")
        self.assertEqual([endpoint for endpoint, _ in self.calls], ["/voices", "/interactions"])
        self.assertEqual(selected["creation"]["remote_voice_id"], "voice_legacy_unprefixed")
        self.assertEqual(selected["creation"]["requested_description"], original_prompt)
        self.assertEqual(restarted.rows()[0]["prompt"], original_prompt)
        self.assertEqual((pending.parent / "casting-request.json").read_bytes(), original_request)
        self.assertFalse(pending.exists())


if __name__ == "__main__":
    unittest.main()
