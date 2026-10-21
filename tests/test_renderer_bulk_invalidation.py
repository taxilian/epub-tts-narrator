"""Bulk recasting clears active choices and keeps failures visible across restarts."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.voices import VoiceStore
from epub_tts_narrator.renderer.workflow import RenderBlocked, ensure_voice
import test_renderer_casting as casting_fixtures


@unittest.skipUnless(casting_fixtures.FFMPEG, "FFmpeg is required for fixture voice verification")
class BulkInvalidationTests(unittest.TestCase):
    # Reuse only setup helpers, without importing another class's test methods.
    setUp = casting_fixtures.CastingTests.setUp
    new_session = casting_fixtures.CastingTests.new_session

    def row(self, session=None, voice_id=None):
        return next(row for row in (session or self.session).rows()
                    if row["voice_id"] == (voice_id or self.voice_id))

    def test_reset_preserves_custom_prompt_script_and_original_recording(self):
        prompt = "A patient, tired masculine narrator with a dry rasp."
        script = "The lantern is still burning, and I will wait here until everyone returns."
        self.session.update_prompt(self.voice_id, prompt, script)
        original = self.session.ensure_ready(self.voice_id)
        recording = Path(original["reference_audio"]).read_bytes()
        self.session._draft_path(self.voice_id).unlink()
        self.session.begin_regeneration([self.voice_id])
        row = self.row()
        self.assertIsNone(row["selected"])
        self.assertEqual(row["regeneration"]["state"], "pending")
        self.assertEqual(row["reference_text"], script)
        self.assertEqual(row["prompt"], original["creation"]["requested_description"])
        self.assertTrue(self.session._draft_path(self.voice_id).is_file())
        self.assertEqual(Path(original["reference_audio"]).read_bytes(), recording)
        self.assertEqual(self.session.store.list_candidates(self.voice_id)[0]["candidate_id"],
                         original["candidate_id"])
        self.assertFalse(self.session.store.list_candidates(self.voice_id)[0]["selected"])

    def test_failed_replacement_cannot_adopt_original_on_restart_or_render(self):
        original = self.session.ensure_ready(self.voice_id)
        original_audio = Path(original["reference_audio"]).read_bytes()
        self.session.begin_regeneration([self.voice_id])
        with patch.object(self.session, "_generate_locked", side_effect=RuntimeError("Provider refused the voice")):
            with self.assertRaisesRegex(RuntimeError, "Provider refused"):
                self.session.regenerate(self.voice_id)
        resumed = self.new_session()
        self.addCleanup(resumed.close)
        row = self.row(resumed)
        self.assertIsNone(row["selected"])
        self.assertEqual(row["regeneration"]["state"], "failed")
        self.assertIn("Provider refused the voice", row["regeneration"]["error"])
        with patch.object(resumed, "_generate_locked") as generation:
            with self.assertRaises(RenderBlocked):
                resumed.ensure_ready(self.voice_id)
            generation.assert_not_called()
        calls = len(self.engine.calls)
        with self.assertRaises(RenderBlocked):
            ensure_voice(resumed.store, self.voice_id, resumed._profile(self.voice_id),
                         self.engine, self.verifier, resumed.config, self.root / "work",
                         notify=lambda _: None, bundle=self.bundle)
        self.assertEqual(len(self.engine.calls), calls)
        self.assertEqual(Path(original["reference_audio"]).read_bytes(), original_audio)

    def test_successful_explicit_repair_clears_marker_and_selects_only_new_take(self):
        original = self.session.ensure_ready(self.voice_id)
        self.session.begin_regeneration([self.voice_id])
        with patch.object(self.session, "_generate_locked", side_effect=RuntimeError("First replacement failed")):
            with self.assertRaises(RuntimeError):
                self.session.regenerate(self.voice_id)
        replacement = self.session.regenerate(self.voice_id)
        self.assertNotEqual(replacement["candidate_id"], original["candidate_id"])
        self.assertIsNone(self.row().get("regeneration"))
        self.assertEqual(self.row()["selected"]["candidate_id"], replacement["candidate_id"])
        resumed = self.new_session()
        self.addCleanup(resumed.close)
        with patch.object(resumed, "_generate_locked") as generation:
            self.assertEqual(resumed.ensure_ready(self.voice_id)["candidate_id"], replacement["candidate_id"])
            generation.assert_not_called()

    def test_all_targets_are_cleared_before_first_generation_and_interrupt_leaves_rest_pending(self):
        voice_ids = [row["voice_id"] for row in self.session.rows()]
        for voice_id in voice_ids:
            self.session.ensure_ready(voice_id)
        calls = len(self.engine.calls)
        self.session.begin_regeneration(voice_ids)
        self.assertEqual(len(self.engine.calls), calls)
        self.assertTrue(all(row["selected"] is None for row in self.session.rows()))
        with patch.object(self.session, "_generate_locked", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.session.regenerate(voice_ids[0])
        resumed = self.new_session()
        self.addCleanup(resumed.close)
        for row in resumed.rows():
            self.assertIsNone(row["selected"])
            self.assertIn(row["regeneration"]["state"], {"pending", "failed"})
        with patch.object(resumed, "_generate_locked") as generation:
            with self.assertRaises(RenderBlocked):
                resumed.ensure_ready(voice_ids[-1])
            generation.assert_not_called()

    def test_bad_later_target_does_not_clear_valid_earlier_selection(self):
        original = self.session.ensure_ready(self.voice_id)
        with self.assertRaises(ValueError):
            self.session.begin_regeneration([self.voice_id, "unknown-voice"])
        self.assertEqual(self.session.store.load(self.voice_id)["candidate_id"], original["candidate_id"])
        self.assertIsNone(self.row().get("regeneration"))

    def test_bad_later_draft_does_not_clear_earlier_selection(self):
        voice_ids = [row["voice_id"] for row in self.session.rows()]
        self.assertGreaterEqual(len(voice_ids), 2)
        original = self.session.ensure_ready(voice_ids[0])
        self.session.update_prompt(voice_ids[1], "An edited sample voice.")
        self.session._draft_path(voice_ids[1]).write_text('{"invalid":"draft"}')
        with self.assertRaises(ValueError):
            self.session.begin_regeneration(voice_ids)
        self.assertEqual(self.session.store.load(voice_ids[0])["candidate_id"], original["candidate_id"])
        self.assertIsNone(self.session.store.regeneration_status(voice_ids[0]))

    def test_shared_roles_invalidate_only_their_canonical_voice_once(self):
        alias = "additional-alias"
        self.session._profiles[alias] = dict(deepcopy(self.session._profiles[self.voice_id]), id=alias)
        self.session.store = VoiceStore(
            self.session.store.directory, self.session.store.series_id,
            cast=self.bundle["cast"], explicit_aliases={alias: self.voice_id})
        self.session.ensure_ready(self.voice_id)
        with patch.object(self.session.store, "invalidate", wraps=self.session.store.invalidate) as invalidate:
            self.session.begin_regeneration([alias, self.voice_id, alias])
        invalidate.assert_called_once_with(self.voice_id)
        self.assertIsNone(self.session.store.load(alias))
        self.assertEqual(self.row(voice_id=alias)["regeneration"], self.row()["regeneration"])

    def test_saved_failure_keeps_provider_details_but_removes_credentials_and_terminal_escapes(self):
        secret = "fixture-very-private-provider-key"
        self.session.config["gemini"]["api_key"] = secret
        self.session.begin_regeneration([self.voice_id])
        message = 'HTTP 400\n{"message":"Voice prompt was blocked","key":"' + secret + '"}\x1b[2J'
        with patch.object(self.session, "_generate_locked", side_effect=RuntimeError(message)):
            with self.assertRaises(RuntimeError):
                self.session.regenerate(self.voice_id)
        resumed = self.new_session()
        self.addCleanup(resumed.close)
        error = self.row(resumed)["regeneration"]["error"]
        self.assertIn("HTTP 400\n", error)
        self.assertIn("Voice prompt was blocked", error)
        self.assertNotIn(secret, error)
        self.assertNotIn("\x1b", error)
        self.assertIn("[REDACTED]", error)
        for path in self.session.store.directory.rglob("*.json"):
            self.assertNotIn(secret, path.read_text())
