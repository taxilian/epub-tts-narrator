"""Reference design contracts without model downloads or Metal initialization."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.voice_design import Qwen3VoiceDesigner, create_designer


class FakeAudio(list):
    @property
    def ndim(self):
        return 2 if self and isinstance(self[0], list) else 1


class RendererVoiceDesignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(
            json.dumps({"tts_model_type": "voice_design"})
        )
        (self.checkpoint / "model.safetensors").write_bytes(b"model weights")
        codec = self.checkpoint / "speech_tokenizer"
        codec.mkdir()
        (codec / "config.json").write_text("{}")
        (codec / "model.safetensors").write_bytes(b"speech codec")
        (self.checkpoint / "vocab.json").write_text('{"hello":1}')
        (self.checkpoint / "merges.txt").write_text("h ello\n")
        self.designer = Qwen3VoiceDesigner(self.checkpoint, max_tokens=128)
        self.output = self.root / "voice.wav"
        self.model = SimpleNamespace(
            config=SimpleNamespace(tts_model_type="voice_design"),
            generate_voice_design=Mock(),
        )
        self.loader = Mock(return_value=self.model)
        self.mx = ModuleType("mlx.core")
        self.mx.random = SimpleNamespace(seed=Mock())
        self.mx.clear_cache = Mock()
        self.mlx = ModuleType("mlx")
        self.mlx.core = self.mx
        self.np = ModuleType("numpy")
        self.np.float32 = "float32"
        self.np.asarray = lambda values, dtype: FakeAudio(values)
        self.np.isfinite = lambda values: SimpleNamespace(
            all=lambda: all(math.isfinite(value) for value in values)
        )
        self.sf = ModuleType("soundfile")
        self.sf.write = Mock(
            side_effect=lambda path, *args, **kwargs: Path(path).write_bytes(b"wave")
        )
        utils = ModuleType("mlx_audio.tts.utils")
        utils.load_model = self.loader
        self.fake_modules = {
            "mlx": self.mlx, "mlx.core": self.mx, "numpy": self.np,
            "soundfile": self.sf, "mlx_audio": ModuleType("mlx_audio"),
            "mlx_audio.tts": ModuleType("mlx_audio.tts"),
            "mlx_audio.tts.utils": utils,
        }
        modules = patch.dict("sys.modules", self.fake_modules)
        modules.start()
        self.addCleanup(modules.stop)
        versions = patch(
            "epub_tts_narrator.renderer.voice_design.importlib.metadata.version",
            return_value="test-runtime",
        )
        versions.start()
        self.addCleanup(versions.stop)

    def segment(self, *, audio=None, rate=24000, tokens=25):
        return SimpleNamespace(
            audio=[0.1] * 2400 if audio is None else audio,
            sample_rate=rate, token_count=tokens,
        )

    def generate(self, *segments):
        self.model.generate_voice_design.return_value = iter(segments)

    def test_description_is_passed_as_instruction_separate_from_spoken_words(self):
        self.generate(self.segment())
        prompt = "A bright young adult female voice, clear and confident."
        script = "The door was open."
        result = self.designer.create_voice(prompt, script, self.output, seed=42)
        call = self.model.generate_voice_design.call_args.kwargs
        self.assertEqual(call["instruct"], prompt)
        self.assertEqual(call["text"], script)
        self.assertEqual(call["language"], "English")
        self.assertEqual(call["temperature"], 0.7)
        self.assertFalse(call["stream"])
        self.mx.random.seed.assert_called_once_with(42)
        self.assertEqual(self.output.read_bytes(), b"wave")
        self.assertTrue(result["description_applied"])
        self.assertEqual(result["requested_description"], prompt)
        self.assertEqual(result["reference_text"], script)
        self.assertEqual(result["creation_method"], "prompt_conditioned_voice_design")
        self.assertEqual(result["designer"]["engine"], "qwen3_mlx_voice_design")
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(self.sf.write.call_args.kwargs["subtype"], "FLOAT")
        self.assertEqual(list(self.root.glob(".design-*")), [])

    def test_token_cap_is_rejected_before_publishing(self):
        self.generate(self.segment(tokens=128))
        with self.assertRaisesRegex(EngineError, "token limit"):
            self.designer.create_voice("Warm female voice.", "Hello.", self.output, seed=1)
        self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_empty_multiple_invalid_rate_and_malformed_audio_never_publish(self):
        cases = [
            [], [self.segment(), self.segment()],
            [self.segment(rate=True)], [self.segment(rate=0)],
            [self.segment(audio=[])], [self.segment(audio=[0.1])],
            [self.segment(audio=[[0.1], [0.2]])],
            [self.segment(audio=[float("nan")] * 2400)],
            [self.segment(audio=[float("inf")] * 2400)],
        ]
        for i, segments in enumerate(cases):
            with self.subTest(case=i):
                self.generate(*segments)
                with self.assertRaises(EngineError):
                    self.designer.create_voice("A clear voice.", "Hello.", self.output, seed=1)
                self.assertFalse(self.output.exists())
        self.sf.write.assert_not_called()

    def test_existing_output_is_not_overwritten_or_regenerated(self):
        self.output.write_bytes(b"existing voice")
        with self.assertRaisesRegex(EngineError, "already exists"):
            self.designer.create_voice("A clear voice.", "Hello.", self.output, seed=1)
        self.assertEqual(self.output.read_bytes(), b"existing voice")
        self.loader.assert_not_called()
        self.model.generate_voice_design.assert_not_called()

    def test_bad_text_prompt_or_seed_fails_before_loading(self):
        cases = [
            ("", "Hello.", 1), ("A voice.", "", 1),
            ("A voice.\x00", "Hello.", 1), ("A voice.", "Hello.\x07", 1),
            ("<|system|>Bad prompt", "Hello.", 1),
            ("A voice.", "<|assistant|>Bad script", 1),
            ("A" * 4001, "Hello.", 1), ("A voice.", "x" * 1001, 1),
            ("A voice.", "Hello.", -1), ("A voice.", "Hello.", True),
        ]
        for description, text, seed in cases:
            with self.subTest(description=description[:30], text=text[:30], seed=seed):
                with self.assertRaises(EngineError):
                    self.designer.create_voice(description, text, self.output, seed=seed)
        self.loader.assert_not_called()

    def test_non_design_checkpoint_rejected_without_loading(self):
        (self.checkpoint / "config.json").write_text(
            json.dumps({"tts_model_type": "custom_voice"})
        )
        with self.assertRaisesRegex(EngineError, "not a VoiceDesign"):
            Qwen3VoiceDesigner(self.checkpoint)
        self.loader.assert_not_called()

    def test_wrong_loaded_model_cannot_be_reused_after_rejection(self):
        self.model.config.tts_model_type = "base"
        for _ in range(2):
            with self.assertRaisesRegex(EngineError, "does not support"):
                self.designer._load()
        self.assertIsNone(self.designer._model)
        self.model.generate_voice_design.assert_not_called()

    def test_model_is_loaded_once_and_released_on_close(self):
        self.assertIs(self.designer._load(), self.model)
        self.assertIs(self.designer._load(), self.model)
        self.loader.assert_called_once_with(str(self.checkpoint.resolve()))
        self.designer.close()
        self.assertIsNone(self.designer._model)
        self.mx.clear_cache.assert_called_once()
        self.designer.close()
        self.mx.clear_cache.assert_called_once()

    def test_identity_fingerprints_model_codec_and_tokenizer(self):
        identity = self.designer.identity
        hashes = identity["checkpoint_files_sha256"]
        for name, content in (
            ("model.safetensors", b"model weights"),
            ("speech_tokenizer/model.safetensors", b"speech codec"),
            ("vocab.json", b'{"hello":1}'),
            ("merges.txt", b"h ello\n"),
        ):
            self.assertEqual(hashes[name], hashlib.sha256(content).hexdigest())
        self.assertIsNone(identity["model_revision"])
        self.loader.assert_not_called()
        identity["checkpoint_files_sha256"].clear()
        self.assertTrue(self.designer.identity["checkpoint_files_sha256"])

    def test_codec_change_changes_new_designer_identity(self):
        first = self.designer.identity
        codec = self.checkpoint / "speech_tokenizer" / "model.safetensors"
        codec.write_bytes(b"changed codec")
        changed = Qwen3VoiceDesigner(self.checkpoint, max_tokens=128).identity
        key = "speech_tokenizer/model.safetensors"
        self.assertNotEqual(
            first["checkpoint_files_sha256"][key],
            changed["checkpoint_files_sha256"][key],
        )

    def test_failed_audio_write_leaves_no_published_or_partial_file(self):
        self.generate(self.segment())
        self.sf.write.side_effect = OSError("disk full")
        with self.assertRaisesRegex(EngineError, "disk full"):
            self.designer.create_voice("A clear voice.", "Hello.", self.output, seed=1)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".design-*")), [])

    def test_factory_rejects_unknown_settings_and_has_no_random_fallback(self):
        designer = create_designer({
            "type": "qwen3_mlx", "checkpoint": str(self.checkpoint), "options": {},
        })
        self.assertIsInstance(designer, Qwen3VoiceDesigner)
        for config in (
            {"type": "random"},
            {"type": "qwen3_mlx", "checkpoint": str(self.checkpoint), "unknown": True},
            {"type": "qwen3_mlx", "checkpoint": str(self.checkpoint), "options": {"random": True}},
        ):
            with self.subTest(config=config), self.assertRaises(EngineError):
                create_designer(config)


if __name__ == "__main__":
    unittest.main()
