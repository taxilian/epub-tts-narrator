"""Reference ASR tests stub the Parakeet model so no audio is decoded."""

from importlib import metadata
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.reference_asr import MlxParakeetVerifier


class RendererReferenceAsrTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "weights.safetensors").write_bytes(b"test weights")
        self.audio = self.root / "reference.wav"
        self.audio.write_bytes(b"the fake model never decodes this file")
        self.result = SimpleNamespace(
            text="Hello there. A second sentence.",
            sentences=[
                SimpleNamespace(text="Hello there.", start=0.0, end=1.0),
                SimpleNamespace(text="A second sentence.", start=1.1, end=2.0),
            ],
        )
        self.fake_model = SimpleNamespace(generate=Mock(return_value=self.result))
        self.loader = Mock(return_value=self.fake_model)
        self.clear_cache = Mock()
        self.modules = {
            "mlx_audio": SimpleNamespace(),
            "mlx_audio.stt": SimpleNamespace(),
            "mlx_audio.stt.utils": SimpleNamespace(load_model=self.loader),
            "mlx": SimpleNamespace(core=SimpleNamespace(clear_cache=self.clear_cache)),
            "mlx.core": SimpleNamespace(clear_cache=self.clear_cache),
        }

    def test_construction_fingerprints_locally_without_loading_model(self):
        for name in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
            (self.model / name).write_text(name)
        with patch.dict(sys.modules, self.modules), patch(
            "epub_tts_narrator.renderer.reference_asr.metadata.version", return_value="test-version",
        ):
            verifier = MlxParakeetVerifier(self.model)
        self.loader.assert_not_called()
        self.assertIsNone(verifier._model)
        self.assertEqual(verifier.identity["backend"], "mlx-parakeet")
        self.assertEqual(verifier.identity["model_path"], str(self.model.resolve()))
        self.assertEqual(set(verifier.identity["model_sha256"]), {
            "config.json", "weights.safetensors", "tokenizer.model", "tokenizer.vocab", "vocab.txt",
        })
        self.assertTrue(all(len(value) == 64 for value in verifier.identity["model_sha256"].values()))
        self.assertEqual(verifier.identity["packages"], {
            "mlx-audio": "test-version", "mlx": "test-version", "numpy": "test-version",
        })
        self.assertEqual(verifier.identity["options"]["decoding"], "greedy")

    def test_transcription_loads_strict_local_path_once_and_never_supplies_text(self):
        verifier = MlxParakeetVerifier(self.model)
        with patch.dict(sys.modules, self.modules):
            first = verifier.transcribe_reference(self.audio)
            second = verifier.transcribe_reference(self.audio)
        self.loader.assert_called_once_with(self.model.resolve(), strict=True)
        self.assertIsInstance(self.loader.call_args.args[0], Path)
        self.assertEqual(self.fake_model.generate.call_args_list, [
            unittest.mock.call(str(self.audio)), unittest.mock.call(str(self.audio)),
        ])
        self.assertEqual(first, {
            "text": self.result.text,
            "segments": [
                {"text": "Hello there.", "start": 0.0, "end": 1.0},
                {"text": "A second sentence.", "start": 1.1, "end": 2.0},
            ],
        })
        self.assertEqual(first, second)
        self.assertFalse(hasattr(verifier, "transcribe"))

    def test_model_directory_config_and_weights_are_required_before_library_load(self):
        with patch.dict(sys.modules, self.modules):
            for path in (self.root / "missing", Path("not-downloaded/parakeet")):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    MlxParakeetVerifier(path)
            (self.model / "weights.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "weights"):
                MlxParakeetVerifier(self.model)
            (self.model / "weights.safetensors").write_bytes(b"weights")
            (self.model / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "config.json"):
                MlxParakeetVerifier(self.model)
        self.loader.assert_not_called()

    def test_changes_to_weights_config_and_tokenizer_are_detected_each_call(self):
        for name in ("weights.safetensors", "config.json", "tokenizer.model"):
            with self.subTest(name=name):
                path = self.model / name
                if not path.exists():
                    path.write_bytes(b"original")
                verifier = MlxParakeetVerifier(self.model)
                with patch.dict(sys.modules, self.modules):
                    verifier.transcribe_reference(self.audio)
                    calls = self.fake_model.generate.call_count
                    path.write_bytes(path.read_bytes() + b" changed")
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        verifier.transcribe_reference(self.audio)
                self.assertEqual(self.fake_model.generate.call_count, calls)
                self.assertNotEqual(
                    verifier.identity["model_sha256"], MlxParakeetVerifier(self.model).identity["model_sha256"])

    def test_removal_and_new_model_assets_invalidate_the_snapshot(self):
        for change in ("remove", "new_weights", "new_tokenizer"):
            with self.subTest(change=change):
                verifier = MlxParakeetVerifier(self.model)
                changed = self.model / "weights.safetensors"
                if change == "remove":
                    changed.unlink()
                elif change == "new_weights":
                    changed = self.model / "additional.safetensors"
                    changed.write_bytes(b"new shard")
                else:
                    changed = self.model / "vocab.txt"
                    changed.write_text("new vocabulary")
                with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(RuntimeError, "changed"):
                    verifier.transcribe_reference(self.audio)
                if change == "remove":
                    changed.write_bytes(b"restored")
                else:
                    changed.unlink()
        self.loader.assert_not_called()

    def test_missing_optional_packages_are_recorded_without_importing_them(self):
        with patch("epub_tts_narrator.renderer.reference_asr.metadata.version",
                   side_effect=metadata.PackageNotFoundError):
            verifier = MlxParakeetVerifier(self.model)
        self.assertEqual(verifier.identity["packages"], {
            "mlx-audio": None, "mlx": None, "numpy": None,
        })

    def test_missing_runtime_fails_only_when_transcription_is_requested(self):
        verifier = MlxParakeetVerifier(self.model)
        with patch.dict(sys.modules, {"mlx_audio.stt.utils": None}):
            with self.assertRaisesRegex(RuntimeError, "mlx-audio"):
                verifier.transcribe_reference(self.audio)

    def test_malformed_result_is_not_replaced_with_an_expected_transcript(self):
        verifier = MlxParakeetVerifier(self.model)
        for result in (
            None, SimpleNamespace(text=None, sentences=[]),
            SimpleNamespace(text="Hello", sentences=None),
            SimpleNamespace(text="Hello", sentences=[SimpleNamespace(text=None)]),
        ):
            with self.subTest(result=result), patch.dict(sys.modules, self.modules):
                self.fake_model.generate.return_value = result
                with self.assertRaises(ValueError):
                    verifier.transcribe_reference(self.audio)

    def test_close_releases_loaded_model_and_clears_cache_once(self):
        verifier = MlxParakeetVerifier(self.model)
        with patch.dict(sys.modules, self.modules):
            verifier.close()
            self.clear_cache.assert_not_called()
            verifier.transcribe_reference(self.audio)
            self.assertIs(verifier._model, self.fake_model)
            verifier.close()
            self.assertIsNone(verifier._model)
            verifier.close()
        self.clear_cache.assert_called_once_with()

    def test_cleanup_failure_does_not_hide_original_operation_error(self):
        verifier = MlxParakeetVerifier(self.model)
        with patch.dict(sys.modules, self.modules):
            verifier.transcribe_reference(self.audio)
            self.clear_cache.side_effect = RuntimeError("cache unavailable")
            verifier.close()
        self.assertIsNone(verifier._model)


if __name__ == "__main__":
    unittest.main()
