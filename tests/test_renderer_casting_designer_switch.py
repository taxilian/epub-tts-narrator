"""Per-character designer overrides must not change the rest of a casting session."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.bundle import load_bundle
from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.voice_prompts import audiobook_voice_prompt
from epub_tts_narrator.cli import export_bundle
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferenceDesigner, TranscriptVerifier,
)


class LocalDesignerEngine(ReferenceDesigner):
    identity = {"engine": "drama_fixture", "model": "local-design-and-clone"}


class GeminiDesigner(ReferenceDesigner):
    identity = {"engine": "gemini_voice_design", "model": "gemini-3.8-flash-tts"}
    capabilities = {"voice_design": True}


class ElevenLabsDesigner(ReferenceDesigner):
    identity = {"engine": "elevenlabs_voice_design", "model": "eleven_ttv_v3"}
    capabilities = {"voice_design": True}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for casting integration")
class CastingDesignerSwitchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        document = read_json(EXAMPLE / "reviewed.json")
        document["cast"]["voices"].append({
            "id": "voice_mara_narration", "description": "Mara telling the story.", "traits": [],
        })
        for role in document["cast"]["roles"]:
            if role["id"] == "narrator_main":
                role["voice_id"] = "voice_mara_narration"
        for segment in document["segments"]:
            if segment["role_id"] == "narrator_main":
                segment["voice_id"] = "voice_mara_narration"
        export_bundle(document, self.root / "book")
        self.bundle = prepare_bundle(load_bundle(self.root / "book"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["gemini"]["api_key"] = "test-only-api-key"
        self.config["elevenlabs"]["api_key"] = "test-only-elevenlabs-key"
        self.config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.transcripts = {}
        self.engine = LocalDesignerEngine(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.configured_designer = ReferenceDesigner(self.transcripts)
        self.remote_designers = []
        self.factory_calls = []
        self.remote_error = None

    def session(self):
        def factory(settings):
            self.factory_calls.append(deepcopy(settings))
            if settings["type"] not in {"gemini", "elevenlabs"}:
                return self.configured_designer
            designer_class = GeminiDesigner if settings["type"] == "gemini" else ElevenLabsDesigner
            designer = designer_class(self.transcripts)
            designer.error = self.remote_error
            self.remote_designers.append(designer)
            return designer

        session = CastingSession(
            self.bundle, self.config,
            engine_factory=lambda _: self.engine,
            verifier_factory=lambda *_args, **_kwargs: self.verifier,
            designer_factory=factory, notify=lambda _: None,
        )
        self.addCleanup(session.close)
        return session

    def test_gemini_replaces_shared_character_then_local_uses_engine(self):
        session = self.session()
        original_config = deepcopy(session.config)
        prompt = "A confident adult female voice with a warm, low register."
        script = "The lantern will show us the way home."
        session.update_prompt("voice_mara_narration", prompt, script)

        remote = session.regenerate("voice_mara_narration", voice_designer="gemini")

        self.assertEqual(self.factory_calls[0], {"type": "gemini", **self.config["gemini"]})
        self.assertEqual(self.remote_designers[0].calls[0]["prompt"], audiobook_voice_prompt(prompt))
        self.assertEqual(self.remote_designers[0].calls[0]["text"], script)
        self.assertEqual(remote["created_with"], self.engine.identity)
        self.assertEqual(remote["creation"]["designer"], GeminiDesigner.identity)
        for voice_id in ("voice_mara", "voice_mara_narration"):
            self.assertEqual(session.store.load(voice_id)["asset_hash"], remote["asset_hash"])
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(session.config, original_config)
        self.assertIsNone(session._designer)
        self.assertTrue(self.remote_designers[0].closed)

        # A later G action is local, even after this character was made remotely.
        local = session.regenerate("voice_mara", voice_designer="local")
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.engine.calls[0]["prompt"], audiobook_voice_prompt(prompt))
        self.assertEqual(self.engine.calls[0]["text"], script)
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(local["creation"]["designer"], self.engine.identity)
        self.assertEqual(session.store.load("voice_mara_narration")["asset_hash"], local["asset_hash"])
        self.assertEqual(session.config, original_config)
        self.assertFalse(self.engine.closed)

    def test_override_restores_cached_configured_designer(self):
        self.config["voice_design"]["type"] = "qwen3_mlx"
        self.config["voice_design"]["checkpoint"] = str(self.root / "other-local-model")
        session = self.session()
        session.regenerate("voice_mara")
        original_config = deepcopy(session.config)
        self.assertIs(session._designer, self.configured_designer)

        session.regenerate("voice_mara", voice_designer="gemini")
        self.assertIs(session._designer, self.configured_designer)
        self.assertFalse(self.configured_designer.closed)
        self.assertEqual(session.config, original_config)
        self.assertTrue(self.remote_designers[0].closed)

        session.regenerate("voice_mara", voice_designer="local")
        self.assertIs(session._designer, self.configured_designer)
        self.assertFalse(self.configured_designer.closed)
        self.assertEqual(len(self.engine.calls), 1)
        session.regenerate("voice_mara")
        self.assertEqual(len(self.configured_designer.calls), 2)
        self.assertEqual(len(self.factory_calls), 2)

    def test_elevenlabs_replaces_shared_character_and_restores_configured_designer(self):
        self.config["voice_design"]["type"] = "qwen3_mlx"
        self.config["voice_design"]["checkpoint"] = str(self.root / "other-local-model")
        session = self.session()
        session.regenerate("voice_mara")
        original_config = deepcopy(session.config)
        remote = session.regenerate("voice_mara_narration", voice_designer="elevenlabs")
        self.assertEqual(self.factory_calls[-1], {"type": "elevenlabs", **self.config["elevenlabs"]})
        self.assertEqual(remote["creation"]["designer"], ElevenLabsDesigner.identity)
        self.assertEqual(remote["created_with"], self.engine.identity)
        for voice_id in ("voice_mara", "voice_mara_narration"):
            self.assertEqual(session.store.load(voice_id)["asset_hash"], remote["asset_hash"])
        self.assertEqual(session.config, original_config)
        self.assertIs(session._designer, self.configured_designer)
        self.assertFalse(self.configured_designer.closed)
        self.assertTrue(self.remote_designers[-1].closed)
        session.regenerate("voice_mara", voice_designer="local")
        self.assertEqual(len(self.engine.calls), 1)

    def test_elevenlabs_failure_preserves_existing_selection(self):
        session = self.session()
        original = session.regenerate("voice_mara", voice_designer="local")
        original_config = deepcopy(session.config)
        self.remote_error = RuntimeError("ElevenLabs request failed")
        with self.assertRaises((ValueError, RuntimeError)):
            session.regenerate("voice_mara_narration", voice_designer="elevenlabs")
        self.assertEqual(session.store.load("voice_mara")["asset_hash"], original["asset_hash"])
        self.assertEqual(session.config, original_config)
        self.assertTrue(self.remote_designers[-1].closed)

    def test_missing_elevenlabs_key_does_not_change_selected_voice(self):
        self.config["elevenlabs"]["api_key"] = None
        session = self.session()
        original = session.regenerate("voice_mara", voice_designer="local")
        with self.assertRaisesRegex(ValueError, r"elevenlabs\.api_key"):
            session.regenerate("voice_mara", voice_designer="elevenlabs")
        self.assertEqual(session.store.load("voice_mara")["asset_hash"], original["asset_hash"])
        self.assertEqual(self.factory_calls, [])

    def test_failed_or_interrupted_remote_replacement_preserves_selection_and_state(self):
        self.config["voice_design"]["type"] = "qwen3_mlx"
        self.config["voice_design"]["checkpoint"] = str(self.root / "other-local-model")
        session = self.session()
        original = session.regenerate("voice_mara")
        original_audio = Path(original["reference_audio"]).read_bytes()
        original_config = deepcopy(session.config)

        for error in (RuntimeError("Remote generation failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.remote_error = error
                expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else (ValueError, RuntimeError)
                with self.assertRaises(expected):
                    session.regenerate("voice_mara", voice_designer="gemini")
                self.assertEqual(session.store.load("voice_mara")["asset_hash"], original["asset_hash"])
                self.assertEqual(Path(original["reference_audio"]).read_bytes(), original_audio)
                self.assertEqual(session.config, original_config)
                self.assertIs(session._designer, self.configured_designer)
                self.assertFalse(self.configured_designer.closed)
                self.assertTrue(self.remote_designers[-1].closed)
                self.assertFalse((session.store.directory / ".lock").exists())

        self.assertEqual(len(session.store.list_candidates("voice_mara")), 1)
        session.regenerate("voice_mara", voice_designer="local")
        self.assertEqual(len(self.engine.calls), 1)

    def test_missing_gemini_key_does_not_change_selection_or_prevent_local_generation(self):
        self.config["gemini"]["api_key"] = None
        session = self.session()
        original = session.regenerate("voice_mara", voice_designer="local")
        original_config = deepcopy(session.config)

        with self.assertRaisesRegex(ValueError, r"gemini\.api_key"):
            session.regenerate("voice_mara", voice_designer="gemini")
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(session.config, original_config)
        self.assertEqual(session.store.load("voice_mara")["asset_hash"], original["asset_hash"])
        session.regenerate("voice_mara", voice_designer="local")
        self.assertEqual(len(self.engine.calls), 2)


if __name__ == "__main__":
    unittest.main()
