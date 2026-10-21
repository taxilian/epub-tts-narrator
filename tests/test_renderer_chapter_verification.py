"""CPU/CUDA recognizer contract without installing models or making API calls."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.chapter_verification import FasterWhisperVerifier, create_chapter_verifier
from epub_tts_narrator.renderer.verification import ReferenceCorroboratingVerifier, ReferenceTextJudgingVerifier


class FasterWhisperVerifierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        for name in ("config.json", "model.bin", "tokenizer.json"):
            (self.path / name).write_bytes(b"offline fixture")
        self.word = SimpleNamespace(word="Hello", start=0.0, end=0.8, probability=0.98)
        self.segment = SimpleNamespace(text=" Hello", start=0.0, end=1.0, avg_logprob=-0.1,
                                       no_speech_prob=0.0, compression_ratio=1.0, words=[self.word])
        self.model = Mock()
        self.model.transcribe.side_effect = lambda *args, **kwargs: (iter([self.segment]), SimpleNamespace(language="en"))
        self.constructor = Mock(return_value=self.model)
        self.module = SimpleNamespace(WhisperModel=self.constructor)

    def test_lazy_local_model_is_reused_and_audio_never_receives_expected_text(self):
        with patch.dict("sys.modules", {"faster_whisper": self.module}):
            verifier = FasterWhisperVerifier(self.path)
            self.constructor.assert_not_called()
            result = verifier.transcribe("speech.wav")
            reference = verifier.transcribe_reference("reference.flac")
        self.constructor.assert_called_once_with(str(self.path.resolve()), device="cpu", compute_type="int8",
                                                  cpu_threads=4, num_workers=1, local_files_only=True)
        self.assertEqual(result["text"], "Hello")
        self.assertEqual(result["segments"][0]["words"][0]["word"], "Hello")
        self.assertNotIn("words", reference["segments"][0])
        for call in self.model.transcribe.call_args_list:
            self.assertIsNone(call.kwargs["initial_prompt"])
            self.assertFalse(call.kwargs["condition_on_previous_text"])
            self.assertFalse(call.kwargs["vad_filter"])
        self.assertTrue(self.model.transcribe.call_args_list[0].kwargs["word_timestamps"])
        self.assertFalse(self.model.transcribe.call_args_list[1].kwargs["word_timestamps"])
        verifier.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            verifier.transcribe("speech.wav")

    def test_incomplete_model_is_rejected_before_an_implicit_download(self):
        (self.path / "tokenizer.json").unlink()
        with patch.dict("sys.modules", {"faster_whisper": self.module}):
            with self.assertRaisesRegex(ValueError, "tokenizer.json"):
                FasterWhisperVerifier(self.path)
        self.constructor.assert_not_called()

    def test_changed_model_is_rejected(self):
        verifier = FasterWhisperVerifier(self.path)
        (self.path / "model.bin").write_bytes(b"changed model bytes")
        with self.assertRaisesRegex(RuntimeError, "ASR model changed"):
            verifier.transcribe("speech.wav")

    def test_factory_preserves_secondary_recognition_and_jev(self):
        settings = {"type": "faster_whisper", "model": str(self.path), "reference_model": str(self.path),
                    "language": "en", "device": "cuda", "compute_type": "float16", "cpu_threads": 2}
        judge = SimpleNamespace(identity={"backend": "fixture-jev"}, judge=lambda *args: {"decision": "equivalent"})
        with patch("epub_tts_narrator.jev.JevTranscriptJudge", return_value=judge) as create_judge:
            verifier = create_chapter_verifier(settings, {"enabled": True})
        self.assertIsInstance(verifier, ReferenceTextJudgingVerifier)
        self.assertIsInstance(verifier.verifier, ReferenceCorroboratingVerifier)
        self.assertEqual(verifier.verifier.primary.identity["device"], "cuda")
        self.assertEqual(verifier.verifier.reference.identity["compute_type"], "float16")
        self.assertEqual(verifier.judge_reference("hello", "hello"), {"decision": "equivalent"})
        create_judge.assert_called_once_with({"enabled": True})


if __name__ == "__main__":
    unittest.main()
