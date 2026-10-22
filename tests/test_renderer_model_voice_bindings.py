"""Saved voices stay usable when changing compatible Drama 3 model ports."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS, select_voice_designer
from epub_tts_narrator.renderer.engines import Drama3MLXEngine
from epub_tts_narrator.renderer.mlx_audio_engine import MLXAudioDrama3Engine
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import RenderBlocked, ensure_voice, import_voice
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferenceDesigner, ReferencePlaybackEngine, TranscriptVerifier,
)


FOUR_BIT = "mlx-community/fish-audio-drama-3-4bit"
EIGHT_BIT = "mlx-community/fish-audio-drama-3-8bit"


class PlaybackFixture(ReferencePlaybackEngine):
    """Real adapter identities, with synthesis replaced at the model boundary."""

    def __init__(self, transcripts, provider):
        super().__init__(transcripts)
        self.provider = provider

    @property
    def identity(self):
        return self.provider.identity

    @property
    def voice_binding_identity(self):
        return self.provider.voice_binding_identity


class GeminiDesignerFixture(ReferenceDesigner):
    identity = {"engine": "gemini_voice_design", "model": "gemini-3.8-flash-tts"}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for voice binding integration")
class ModelVoiceBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps({"model_type": "fish_drama3"}))
        (self.checkpoint / "model.safetensors").write_bytes(b"model-free test identity")
        self.config = deepcopy(DEFAULTS)
        self.config["engine"].update(model=FOUR_BIT, checkpoint=str(self.checkpoint))
        self.config["verification"]["model"] = str(self.root / "asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="portable-voices")
        self.config["render"].update(work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
                                      max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.config["gemini"]["api_key"] = "model-free-test-key"
        self.config = select_voice_designer(self.config, "gemini")
        self.bundle = prepare_bundle(load_bundle(EXAMPLE / "bundle"))
        self.profile = self.bundle["cast"]["voices"][0]
        self.voice_id = self.profile["id"]
        self.transcripts = {}
        self.designer = GeminiDesignerFixture(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.engine = self.playback(FOUR_BIT)

    def playback(self, model):
        provider = (Drama3MLXEngine(self.checkpoint) if model == "fishaudio/drama-3"
                    else MLXAudioDrama3Engine(self.checkpoint, model=model))
        return PlaybackFixture(self.transcripts, provider)

    def session(self, engine=None):
        session = CastingSession(
            self.bundle, self.config, engine_factory=lambda _: engine or self.engine,
            verifier_factory=lambda *_args, **_kwargs: self.verifier,
            designer_factory=lambda _: self.designer, notify=lambda _: None,
        )
        self.addCleanup(session.close)
        return session

    def reuse(self, store, engine=None):
        return ensure_voice(
            store, self.voice_id, self.profile, engine or self.engine, self.verifier,
            self.config, self.root / "work", notify=lambda _: None, designer=self.designer,
        )

    def test_new_gemini_candidate_can_be_selected_and_used_by_quantized_model(self):
        session = self.session()
        selected = session.regenerate(self.voice_id)
        reused = self.reuse(session.store)

        self.assertEqual(reused["asset_hash"], selected["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)
        self.assertEqual(self.engine.calls, [])
        request = read_json(next((self.root / "work" / "casting-candidates").glob("*/accepted.json")))["request"]
        audit = read_json(next((session.store.directory / "casting-acceptance").glob("*.json")))
        self.assertEqual(request["engine"], self.engine.identity)
        self.assertEqual(audit["engine"], self.engine.identity)
        self.assertEqual(selected["creation"]["engine"], self.engine.identity)
        self.assertEqual(selected["creation"]["designer"], self.designer.identity)

    def test_existing_sogni_candidate_can_be_selected_after_switching_to_quantized_model(self):
        # Historical records stored the full execution identity, including weights.
        legacy = PlaybackFixture(self.transcripts, SimpleNamespace(identity={
            "engine": "drama3_mlx", "model": "fishaudio/drama-3",
            "checkpoint_files_sha256": {"weights": "original Sogni weights"},
        }))
        candidate = self.session(legacy).generate(self.voice_id)
        session = self.session()
        selected = session.accept(self.voice_id, candidate["candidate_id"])

        self.assertEqual(self.reuse(session.store)["asset_hash"], selected["asset_hash"])
        self.assertEqual(selected["asset_hash"], candidate["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_quantized_candidate_can_be_selected_by_another_quantization_and_sogni(self):
        candidate = self.session().generate(self.voice_id)
        for model in (EIGHT_BIT, "fishaudio/drama-3"):
            with self.subTest(model=model):
                engine = self.playback(model)
                session = self.session(engine)
                selected = session.accept(self.voice_id, candidate["candidate_id"])
                self.assertEqual(self.reuse(session.store, engine)["asset_hash"], candidate["asset_hash"])
                self.assertEqual(selected["reference_sha256"], candidate["reference_sha256"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_imported_reference_is_reusable_after_switching_the_model(self):
        source = self.root / "supplied.wav"
        text = "An existing recording keeps its voice."
        self.designer.write_audio(text, source, 1729)
        imported = import_voice(self.bundle, self.voice_id, source, text, self.config,
                                self.engine, self.verifier)
        session = self.session()

        self.assertEqual(self.reuse(session.store)["asset_hash"], imported["asset_hash"])
        self.assertEqual(self.reuse(session.store, self.playback(EIGHT_BIT))["asset_hash"], imported["asset_hash"])
        self.assertEqual(imported["creation"]["engine"], self.engine.identity)
        self.assertEqual(self.designer.calls, [])

    def test_automatically_created_reference_preserves_its_execution_identity(self):
        session = self.session()
        created = self.reuse(session.store)
        reused = self.reuse(session.store, self.playback("fishaudio/drama-3"))

        self.assertEqual(created["creation"]["engine"], self.engine.identity)
        self.assertEqual(reused["asset_hash"], created["asset_hash"])
        self.assertEqual(len(self.designer.calls), 1)

    def test_recovered_attempt_keeps_execution_evidence_and_creates_portable_reference(self):
        original = self.session()
        original.generate(self.voice_id)
        attempt = next((self.root / "work" / "casting-candidates").glob("*/attempt-001.json"))
        original_record = attempt.read_bytes()
        changed_engine = self.playback(EIGHT_BIT)
        session = self.session(changed_engine)
        recovered = session.recover_attempt(self.voice_id, attempt)

        self.assertEqual(self.reuse(session.store, changed_engine)["asset_hash"], recovered["asset_hash"])
        self.assertEqual(recovered["creation"]["engine"], self.engine.identity)
        self.assertEqual(attempt.read_bytes(), original_record)
        self.assertEqual(len(self.designer.calls), 1)

    def test_unrelated_model_cannot_select_a_portable_drama_reference(self):
        candidate = self.session().generate(self.voice_id)
        unrelated = PlaybackFixture(self.transcripts, SimpleNamespace(identity={
            "engine": "unrelated_tts", "model": "another-model",
        }))
        session = self.session(unrelated)
        with self.assertRaisesRegex(RenderBlocked, "different engine or model"):
            session.accept(self.voice_id, candidate["candidate_id"])
        self.assertIsNone(session.store.load(self.voice_id))


if __name__ == "__main__":
    unittest.main()
