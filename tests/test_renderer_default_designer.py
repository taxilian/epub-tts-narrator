"""Automatic casting chooses Gemini without changing the local book engine."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle, prepare_document
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS, resolve_voice_design, select_voice_designer
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import ensure_voice, render
from epub_tts_narrator.cli import export_bundle
from test_renderer_voice_design_integration import (
    EXAMPLE, FFMPEG, FFPROBE, ReferenceDesigner, ReferencePlaybackEngine, TranscriptVerifier,
)


class DefaultDesignerConfigurationTests(unittest.TestCase):
    def test_default_resolves_from_configured_credentials_without_mutating_config(self):
        self.assertEqual(DEFAULTS["voice_design"]["type"], "auto")
        for key, expected in ((None, "engine"), ("", "engine"), ("  ", "engine"),
                              ("test-only-api-key", "gemini")):
            with self.subTest(key_present=bool(key and key.strip())):
                config = deepcopy(DEFAULTS)
                config["gemini"]["api_key"] = key
                original = deepcopy(config)
                resolved = resolve_voice_design(config)
                self.assertEqual(resolved["type"], expected)
                if expected == "gemini":
                    self.assertEqual(resolved, {"type": "gemini", **config["gemini"]})
                self.assertEqual(config, original)

    def test_explicit_local_and_other_designer_are_not_overridden_by_gemini_key(self):
        config = deepcopy(DEFAULTS)
        config["gemini"]["api_key"] = "test-only-api-key"
        self.assertEqual(resolve_voice_design(select_voice_designer(config, "local"))["type"],
                         "engine")
        config["voice_design"].update(type="qwen3_mlx", checkpoint="/fixture/qwen")
        self.assertEqual(resolve_voice_design(config), config["voice_design"])
        explicit = select_voice_designer(config, "gemini")
        self.assertEqual(resolve_voice_design(explicit), explicit["voice_design"])

    def test_no_cli_override_preserves_auto_for_later_resolution(self):
        config = deepcopy(DEFAULTS)
        config["gemini"]["api_key"] = "test-only-api-key"
        copied = select_voice_designer(config, None)
        self.assertEqual(copied, config)
        self.assertEqual(copied["voice_design"]["type"], "auto")
        copied["gemini"]["api_key"] = None
        self.assertEqual(config["gemini"]["api_key"], "test-only-api-key")
        self.assertEqual(resolve_voice_design(copied)["type"], "engine")


class LocalDesignerEngine(ReferencePlaybackEngine):
    capabilities = {"voice_design": True, "reference_voice": True}
    create_voice = ReferenceDesigner.create_voice


class GeminiDesigner(ReferenceDesigner):
    identity = {"engine": "gemini_voice_design", "model": "gemini-3.8-flash-tts"}
    capabilities = {"voice_design": True}


@unittest.skipUnless(FFMPEG, "FFmpeg is required for automatic casting integration")
class DefaultDesignerIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        document = prepare_document(read_json(EXAMPLE / "reviewed.json"))
        export_bundle(document, self.root / "book")
        self.bundle = prepare_bundle(load_bundle(self.root / "book"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "drama-model")
        self.config["verification"]["model"] = str(self.root / "asr-model")
        self.config["gemini"]["api_key"] = "test-only-api-key"
        self.config["render"].update(max_attempts=1, ffmpeg=FFMPEG, ffprobe=FFPROBE or "ffprobe")
        self.config["quality"] = validate_policy({})
        self.transcripts = {}
        self.engine = LocalDesignerEngine(self.transcripts)
        self.designer = GeminiDesigner(self.transcripts)
        self.verifier = TranscriptVerifier(self.transcripts)
        self.factory_calls = []

    def session(self):
        def factory(settings):
            self.factory_calls.append(deepcopy(settings))
            self.assertEqual(settings, {"type": "gemini", **self.config["gemini"]})
            return self.designer

        session = CastingSession(
            self.bundle, self.config, engine_factory=lambda _: self.engine,
            verifier_factory=lambda *_args, **_kwargs: self.verifier,
            designer_factory=factory, notify=lambda _: None,
        )
        self.addCleanup(session.close)
        return session

    def test_missing_samples_use_gemini_and_keep_local_clone_binding(self):
        session = self.session()
        session.generate_missing()
        self.assertEqual(len(self.factory_calls), 1)
        self.assertGreater(len(self.designer.calls), 0)
        self.assertEqual(self.engine.calls, [])
        for voice in self.bundle["cast"]["voices"]:
            selected = session.store.load(voice["id"])
            self.assertIsNotNone(selected)
            self.assertEqual(selected["created_with"], self.engine.identity)
            self.assertEqual(selected["creation"]["designer"], self.designer.identity)
        self.assertEqual(self.config["voice_design"]["type"], "auto")
        self.assertEqual(session.config["voice_design"]["type"], "gemini")

    def test_no_configured_key_uses_local_engine(self):
        self.config["gemini"]["api_key"] = None
        selected = self.session().ensure_ready("voice_mara")
        self.assertEqual(selected["creation"]["designer"], self.engine.identity)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(self.designer.calls, [])
        self.assertEqual(self.factory_calls, [])

    def test_gemini_failure_does_not_silently_create_a_local_voice(self):
        self.designer.error = RuntimeError("Gemini API unavailable")
        session = self.session()
        with self.assertRaises((ValueError, RuntimeError)):
            session.ensure_ready("voice_mara")
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(len(self.designer.calls), 1)
        self.assertEqual(self.engine.calls, [])
        self.assertIsNone(session.store.load("voice_mara"))

    def test_direct_render_voice_preparation_resolves_auto_and_reuses_saved_reference(self):
        session = self.session()
        profile = next(voice for voice in self.bundle["cast"]["voices"] if voice["id"] == "voice_mara")
        config = deepcopy(session.config)
        config["voice_design"] = deepcopy(self.config["voice_design"])
        with patch("epub_tts_narrator.renderer.voice_design.create_designer", return_value=self.designer) as factory:
            selected = ensure_voice(
                session.store, "voice_mara", profile, self.engine, self.verifier,
                config, config["render"]["work_dir"], bundle=self.bundle, notify=lambda _: None,
            )
        factory.assert_called_once_with({"type": "gemini", **self.config["gemini"]})
        self.assertEqual(selected["created_with"], self.engine.identity)
        self.assertEqual(selected["creation"]["designer"], self.designer.identity)
        with patch("epub_tts_narrator.renderer.voice_design.create_designer", side_effect=AssertionError("Must reuse saved voice")):
            reused = ensure_voice(
                session.store, "voice_mara", profile, self.engine, self.verifier,
                config, config["render"]["work_dir"], bundle=self.bundle, notify=lambda _: None,
            )
        self.assertEqual(reused["asset_hash"], selected["asset_hash"])
        self.assertEqual(self.engine.calls, [])

    @unittest.skipUnless(FFPROBE, "FFprobe is required for complete renderer validation")
    def test_audiobook_creation_uses_local_engine_with_saved_gemini_voices(self):
        session = self.session()
        session.generate_missing()
        generation_count = len(self.designer.calls)
        with patch("epub_tts_narrator.renderer.voice_design.create_designer", side_effect=AssertionError("No new voice required")):
            result = render(self.bundle, self.config, self.engine, self.verifier, notify=lambda _: None)
        self.assertGreater(len(self.engine.speech_calls), 0)
        self.assertEqual(len(self.designer.calls), generation_count)
        self.assertEqual(self.engine.calls, [])
        for call in self.engine.speech_calls:
            self.assertEqual(call["asset_hash"], result["voice_assets"][call["voice_id"]])
            self.assertTrue(Path(call["reference_audio"]).is_file())


if __name__ == "__main__":
    unittest.main()
