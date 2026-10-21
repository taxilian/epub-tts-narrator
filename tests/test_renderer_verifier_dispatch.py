"""The global ASR backend applies to casting, local rendering, and chapter jobs."""

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.chapter_verification import FasterWhisperVerifier
from epub_tts_narrator.renderer.cli import doctor
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.model_setup import ModelSetup, ModelSetupError
from epub_tts_narrator.renderer.verification import (
    ReferenceCorroboratingVerifier, create_verifier,
)


def converted_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"lang_ids": [50259]}))
    (path / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}))
    (path / "model.bin").write_bytes(b"fixture CTranslate2 weights")
    return path


class SharedVerifierDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.primary = converted_model(self.root / "primary")
        self.settings = {**DEFAULTS["verification"], "type": "faster_whisper", "model": str(self.primary)}

    def test_shared_factory_uses_configured_faster_backend_without_loading_models(self):
        with (patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier") as mlx,
              patch.dict("sys.modules", {"faster_whisper": None, "mlx_whisper": None})):
            verifier = create_verifier(self.settings)
        self.assertIsInstance(verifier, FasterWhisperVerifier)
        self.assertEqual(verifier.identity["backend"], "faster-whisper")
        self.assertEqual(verifier.identity["compute_type"], "int8")
        self.assertIsNone(verifier._model)
        mlx.assert_not_called()

    def test_shared_factory_uses_faster_backend_for_secondary_recognition_too(self):
        reference = converted_model(self.root / "secondary")
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier") as mlx:
            verifier = create_verifier(self.settings | {"reference_model": str(reference)})
        self.assertIsInstance(verifier, ReferenceCorroboratingVerifier)
        self.assertIsInstance(verifier.primary, FasterWhisperVerifier)
        self.assertIsInstance(verifier.reference, FasterWhisperVerifier)
        self.assertEqual(verifier.reference.model_path, reference.resolve())
        mlx.assert_not_called()

    def test_existing_mlx_factory_remains_unchanged(self):
        settings = {"type": "mlx_whisper", "model": "/local/mlx-model", "language": "en", "reference_model": None}
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier") as mlx:
            self.assertIs(create_verifier(settings), mlx.return_value)
        mlx.assert_called_once_with("/local/mlx-model", language="en")

    def test_unknown_backend_is_never_silently_treated_as_mlx(self):
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier") as mlx:
            with self.assertRaisesRegex(ValueError, "Unsupported verifier"):
                create_verifier(self.settings | {"type": "mistyped-backend"})
        mlx.assert_not_called()


class FasterWhisperSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = deepcopy(DEFAULTS)
        self.config["verification"].update(type="faster_whisper", model=str(self.root / "model"))
        self.downloader = Mock()
        self.input = Mock()
        self.output = io.StringIO()
        self.setup = ModelSetup(self.config, input_stream=self.input, output_stream=self.output, downloader=self.downloader)

    def test_complete_ct2_models_are_checked_without_import_download_or_prompt(self):
        model = converted_model(self.root / "model")
        reference = converted_model(self.root / "reference")
        self.config["verification"]["reference_model"] = str(reference)
        before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                  for directory in (model, reference) for path in directory.iterdir()}
        with patch.dict("sys.modules", {"faster_whisper": None, "ctranslate2": None, "huggingface_hub": None}):
            self.setup.ensure("verification")
        after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                 for directory in (model, reference) for path in directory.iterdir()}
        self.assertEqual(before, after)
        self.downloader.assert_not_called()
        self.input.readline.assert_not_called()
        self.assertEqual(self.output.getvalue(), "")

    def test_missing_model_has_actionable_error_and_never_downloads(self):
        with self.assertRaisesRegex(ModelSetupError, "local CTranslate2|CTranslate2 Whisper") as raised:
            self.setup.ensure("verification")
        self.assertIn("model.bin", str(raised.exception))
        self.assertIn("tokenizer.json", str(raised.exception))
        self.assertIn("No model or tokenizer will be downloaded", str(raised.exception))
        self.assertFalse((self.root / "model").exists())
        self.downloader.assert_not_called()
        self.input.readline.assert_not_called()

    def test_missing_local_tokenizer_is_rejected_before_hub_fallback(self):
        model = converted_model(self.root / "model")
        (model / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ModelSetupError, "tokenizer.json"):
            self.setup.ensure("verification")
        self.downloader.assert_not_called()

    def test_missing_or_invalid_secondary_model_is_not_replaced_with_parakeet(self):
        converted_model(self.root / "model")
        reference = converted_model(self.root / "secondary")
        (reference / "config.json").write_text("not json")
        self.config["verification"]["reference_model"] = str(reference)
        with self.assertRaisesRegex(ModelSetupError, "verification.reference_model"):
            self.setup.ensure("verification")
        self.assertEqual((reference / "config.json").read_text(), "not json")
        self.downloader.assert_not_called()

    def test_mlx_weights_are_not_misidentified_as_ct2_weights(self):
        path = self.root / "model"
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"whisper"}')
        (path / "weights.safetensors").write_bytes(b"MLX fixture")
        with self.assertRaisesRegex(ModelSetupError, "model.bin"):
            self.setup.ensure("verification")
        self.downloader.assert_not_called()

    def test_unset_model_is_reported_without_type_error(self):
        self.config["verification"]["model"] = None
        with self.assertRaisesRegex(ModelSetupError, "verification.model"):
            self.setup.ensure("verification")


class FasterWhisperDoctorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = deepcopy(DEFAULTS)
        self.config["verification"].update(type="faster_whisper", model=str(converted_model(self.root / "model")))
        self.config["render_pool"]["workers"] = [{"type": "fish_http"}]

    def test_remote_speech_with_ct2_asr_does_not_require_mlx_packages(self):
        with (patch("epub_tts_narrator.renderer.cli.metadata.version", return_value="fixture") as version,
              patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool"),
              patch.dict("sys.modules", {"faster_whisper": None, "ctranslate2": None, "mlx": None})):
            report = doctor(self.config)
        self.assertTrue(report["ready"], report)
        self.assertEqual(set(report["packages"]), {"faster-whisper", "ctranslate2", "tokenizers", "av", "msgpack"})
        self.assertTrue(report["checks"]["whisper_tokenizer"])
        self.assertFalse(any("mlx" in call.args[0] for call in version.call_args_list))

    def test_secondary_ct2_requires_tokenizer_and_binary_weights(self):
        reference = converted_model(self.root / "reference")
        self.config["verification"]["reference_model"] = str(reference)
        with (patch("epub_tts_narrator.renderer.cli.metadata.version", return_value="fixture"),
              patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool")):
            self.assertTrue(doctor(self.config)["reference_verification"]["available"])
            (reference / "tokenizer.json").unlink()
            report = doctor(self.config)
        self.assertFalse(report["ready"])
        self.assertFalse(report["reference_verification"]["available"])
        self.assertFalse(report["checks"]["reference_asr_tokenizer"])

    def test_ct2_asr_with_local_drama_still_requires_mlx_for_synthesis(self):
        self.config["render_pool"]["workers"] = []
        with (patch("epub_tts_narrator.renderer.cli.metadata.version", return_value="fixture"),
              patch("epub_tts_narrator.renderer.cli.shutil.which", return_value="/fixture/tool")):
            report = doctor(self.config)
        self.assertIn("drama3-mlx", report["packages"])
        self.assertIn("mlx", report["packages"])
        self.assertNotIn("mlx-whisper", report["packages"])


if __name__ == "__main__":
    unittest.main()
