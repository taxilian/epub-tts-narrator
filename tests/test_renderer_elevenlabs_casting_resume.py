"""Startup recovers saved ElevenLabs previews; explicit G11 creates fresh voices."""

import base64
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.elevenlabs_voice_design import ElevenLabsVoiceDesigner, _publish
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import RenderBlocked
from epub_tts_narrator.renderer.assembly import normalize_clip
from epub_tts_narrator.cli import export_bundle
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferencePlaybackEngine, TranscriptVerifier,
)


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting integration")
class ElevenLabsCastingResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        export_bundle(read_json(EXAMPLE / "reviewed.json"), self.root / "book")
        self.bundle = prepare_bundle(load_bundle(self.root / "book"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["voice_design"]["type"] = "elevenlabs"
        self.config["elevenlabs"]["api_key"] = "fixture-elevenlabs-secret"
        self.config["render"].update(max_attempts=3, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.script = ("The lantern will guide our journey home. We can take the longer path "
                       "through the hills and reach the bridge before the storm comes.")
        self.prompt = "A warm adult female voice with a low pitch, clear articulation, and gentle pace."
        self.transcripts = defaultdict(lambda: self.script)
        self.posts = []
        self.notices = []

    def session(self, *, config=None, bundle=None, engine_model=None):
        engine = ReferencePlaybackEngine(self.transcripts)
        if engine_model:
            engine.identity = {**engine.identity, "model": engine_model}
        verifier = TranscriptVerifier(self.transcripts)

        def factory(settings):
            designer = ElevenLabsVoiceDesigner(**{key: value for key, value in settings.items() if key != "type"})

            def post(payload):
                # The exact request must already be durable before any paid call.
                requests = list(self.root.rglob("casting-request.json"))
                self.assertTrue(requests)
                for path in requests:
                    self.assertNotIn("fixture-elevenlabs-secret", path.read_text())
                self.posts.append(deepcopy(payload))
                pcm = struct.pack("<" + "h" * 24000, *[
                    round(4000 * math.sin(2 * math.pi * (200 + len(self.posts) * 100) * n / 24000))
                    for n in range(24000)])
                return {"text": payload["text"], "previews": [{
                    "audio_base_64": base64.b64encode(pcm).decode(), "media_type": "audio/pcm",
                    "generated_voice_id": f"fixture_preview_{len(self.posts)}",
                }]}

            designer._post = post
            return designer

        session = CastingSession(
            bundle or self.bundle, config or self.config,
            engine_factory=lambda _: engine, verifier_factory=lambda *_args, **_kwargs: verifier,
            designer_factory=factory, notify=self.notices.append)
        self.addCleanup(session.close)
        return session

    @staticmethod
    def interrupt_wav_publish(path, content):
        if path.suffix == ".wav":
            raise KeyboardInterrupt()
        _publish(path, content)

    def interrupted_request(self):
        session = self.session()
        session.update_prompt("voice_mara", self.prompt, self.script)
        with patch("epub_tts_narrator.renderer.elevenlabs_voice_design._publish", side_effect=self.interrupt_wav_publish):
            with self.assertRaises(KeyboardInterrupt):
                session.ensure_ready("voice_mara")
        session.close()
        requests = list(self.root.rglob("casting-request.json"))
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(self.posts), 1)
        return requests[0], read_json(requests[0])

    def test_restart_after_preview_saved_reuses_same_request_without_design_call(self):
        path, original = self.interrupted_request()
        self.assertTrue((path.parent / "elevenlabs-previews-001.json").exists())
        self.assertFalse(list(path.parent.glob("*.wav")))
        restarted = self.session()
        selected = restarted.ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(selected["candidate_id"], original["nonce"])
        self.assertEqual(selected["creation"]["request_hash"], path.parent.name)
        self.assertEqual(selected["creation"]["generated_voice_id"], "fixture_preview_1")
        self.assertTrue(selected["verification"]["passed"])
        self.assertTrue(any("Resuming the saved ElevenLabs" in note for note in self.notices))
        self.assertEqual(len(list(self.root.rglob("casting-request.json"))), 1)

    def test_explicit_g11_does_not_resume_interrupted_preview(self):
        _, original = self.interrupted_request()
        restarted = self.session()
        selected = restarted.regenerate("voice_mara", voice_designer="elevenlabs")
        self.assertEqual(len(self.posts), 2)
        self.assertNotEqual(selected["candidate_id"], original["nonce"])
        self.assertEqual(selected["creation"]["generated_voice_id"], "fixture_preview_2")

    def test_accepted_but_unpublished_candidate_is_recovered_without_generation(self):
        first = self.session()
        first.update_prompt("voice_mara", self.prompt, self.script)
        with patch.object(first.store, "publish_candidate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                first.ensure_ready("voice_mara")
        first.close()
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(len(list(self.root.rglob("accepted.json"))), 1)
        selected = self.session().ensure_ready("voice_mara")
        self.assertTrue(selected["verification"]["passed"])
        self.assertEqual(len(self.posts), 1)

    def verification_interruption(self):
        first = self.session()
        first.update_prompt("voice_mara", self.prompt, self.script)
        with patch("epub_tts_narrator.renderer.workflow.validate_audio", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                first.ensure_ready("voice_mara")
        first.close()
        records = list(self.root.rglob("attempt-001-generation.json"))
        self.assertEqual(len(records), 1)
        self.assertTrue((records[0].parent / "attempt-001.wav").is_file())
        self.assertFalse((records[0].parent / "attempt-001.json").exists())
        return records[0]

    def test_interrupted_verification_retains_generation_metadata_and_audio(self):
        record = self.verification_interruption()
        before = (record.parent / "attempt-001.wav").read_bytes()
        selected = self.session().ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(selected["creation"]["generated_voice_id"], "fixture_preview_1")
        self.assertTrue(selected["creation"]["description_applied"])
        self.assertEqual((record.parent / "attempt-001.wav").read_bytes(), before)

    def test_interruption_before_normalized_checksum_revalidates_raw_binding(self):
        first = self.session()
        first.update_prompt("voice_mara", self.prompt, self.script)

        def interrupted_normalize(*args, **kwargs):
            normalize_clip(*args, **kwargs)
            raise KeyboardInterrupt()

        with patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=interrupted_normalize):
            with self.assertRaises(KeyboardInterrupt):
                first.ensure_ready("voice_mara")
        first.close()
        path = next(self.root.rglob("attempt-001-generation.json"))
        self.assertIsNone(read_json(path)["normalized_sha256"])
        selected = self.session().ensure_ready("voice_mara")
        self.assertTrue(selected["creation"]["description_applied"])
        self.assertEqual(len(self.posts), 1)
        self.assertIsNotNone(read_json(path)["normalized_sha256"])

    def test_changed_generation_metadata_is_not_trusted(self):
        path = self.verification_interruption()
        record = read_json(path)
        record["engine_metadata"]["requested_description"] = "A different voice description."
        # Even a correctly checksummed record must match the exact request.
        record["record_sha256"] = digest({key: value for key, value in record.items() if key != "record_sha256"})
        write_json(path, record)
        with self.assertRaisesRegex(RenderBlocked, "generation metadata or audio changed"):
            self.session().ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 1)

    def test_changed_raw_audio_is_not_trusted(self):
        path = self.verification_interruption()
        record = read_json(path)
        raw = path.parent / record["raw_file"]
        raw.write_bytes(raw.read_bytes() + b"altered")
        with self.assertRaisesRegex(RenderBlocked, "generation metadata or audio changed"):
            self.session().ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 1)

    def test_changed_normalized_audio_is_not_trusted(self):
        path = self.verification_interruption()
        wav = path.parent / "attempt-001.wav"
        wav.write_bytes(wav.read_bytes() + b"altered")
        with self.assertRaisesRegex(RenderBlocked, "audio changed after ElevenLabs generation"):
            self.session().ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 1)

    def test_changed_prompt_does_not_reuse_stale_preview(self):
        _, original = self.interrupted_request()
        restarted = self.session()
        restarted.update_prompt("voice_mara", "A precise, bright female voice with a light nasal resonance.")
        selected = restarted.ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 2)
        self.assertNotEqual(selected["candidate_id"], original["nonce"])
        self.assertNotEqual(self.posts[0]["voice_description"], self.posts[1]["voice_description"])

    def test_changed_script_does_not_reuse_stale_preview(self):
        _, original = self.interrupted_request()
        self.script = self.script.replace("lantern", "torch")
        restarted = self.session()
        restarted.update_prompt("voice_mara", reference_text=self.script)
        selected = restarted.ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 2)
        self.assertNotEqual(selected["candidate_id"], original["nonce"])
        self.assertEqual(selected["reference_text"], self.script)

    def test_changed_source_engine_or_designer_is_not_a_matching_request(self):
        _, original = self.interrupted_request()
        for change in ("source", "engine", "designer"):
            with self.subTest(change=change):
                config, bundle = deepcopy(self.config), deepcopy(self.bundle)
                if change == "source":
                    bundle["manifest"]["source_fingerprint"] = "different-source-fingerprint"
                elif change == "designer":
                    config["elevenlabs"]["guidance_scale"] = 8.0
                restarted = self.session(config=config, bundle=bundle,
                                         engine_model="changed-local-model" if change == "engine" else None)
                before = len(self.posts)
                with patch("epub_tts_narrator.renderer.elevenlabs_voice_design._publish", side_effect=self.interrupt_wav_publish):
                    with self.assertRaises(KeyboardInterrupt):
                        restarted.ensure_ready("voice_mara")
                self.assertEqual(len(self.posts), before + 1)
                restarted.close()
        self.assertEqual(len(list(self.root.rglob("casting-request.json"))), 4)

    def test_corrupted_durable_request_blocks_recovery_without_another_api_call(self):
        path, request = self.interrupted_request()
        write_json(path, {**request, "text": self.script.replace("lantern", "torch")})
        with self.assertRaisesRegex(RenderBlocked, "Cannot safely resume ElevenLabs"):
            self.session().ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 1)

    def test_newest_unfinished_explicit_regeneration_supersedes_older_request(self):
        older, _ = self.interrupted_request()
        second = self.session()
        with patch("epub_tts_narrator.renderer.elevenlabs_voice_design._publish", side_effect=self.interrupt_wav_publish):
            with self.assertRaises(KeyboardInterrupt):
                second.regenerate("voice_mara", voice_designer="elevenlabs")
        second.close()
        requests = list(self.root.rglob("casting-request.json"))
        newest = next(path for path in requests if path != older)
        selected = self.session().ensure_ready("voice_mara")
        self.assertEqual(len(self.posts), 2)
        self.assertEqual(selected["candidate_id"], read_json(newest)["nonce"])


if __name__ == "__main__":
    unittest.main()
