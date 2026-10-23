"""Verification factory tests compose fake verifiers without loading ASR models."""

from copy import deepcopy
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

from epub_tts_narrator.renderer.quality import validate_audio, validation_key
from epub_tts_narrator.renderer.verification import ReferenceCorroboratingVerifier, create_verifier
from shared_fixtures import timed_transcript


def fake_primary(model="primary-v1"):
    return SimpleNamespace(
        identity={"backend": "mlx-whisper", "model_sha256": {"weights": model}},
        transcribe=Mock(return_value=timed_transcript("Hello there")),
        transcribe_reference=Mock(return_value={"text": "Hello there", "segments": [{"text": "Hello there"}]}),
        close=Mock(),
    )


def fake_reference(model="secondary-v1"):
    return SimpleNamespace(
        identity={"backend": "mlx-parakeet", "model_sha256": {"weights": model}},
        transcribe_reference=Mock(return_value={"text": "Hello there", "segments": [{"text": "Hello there"}]}),
        close=Mock(),
    )


class RendererVerificationFactoryTests(unittest.TestCase):
    def test_default_factory_uses_only_primary_verifier(self):
        for extra in ({}, {"reference_model": None}, {"reference_model": ""}):
            with self.subTest(extra=extra):
                primary = fake_primary()
                with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier", return_value=primary) as whisper, \
                     patch("epub_tts_narrator.renderer.reference_asr.MlxParakeetVerifier") as parakeet:
                    result = create_verifier({"model": "/local/whisper", "language": "en", **extra})
                self.assertIs(result, primary)
                whisper.assert_called_once_with("/local/whisper", language="en")
                parakeet.assert_not_called()
                primary.transcribe.assert_not_called()
                primary.transcribe_reference.assert_not_called()

    def test_configured_reference_model_builds_wrapper_without_transcribing(self):
        primary, secondary = fake_primary(), fake_reference()
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier", return_value=primary) as whisper, \
             patch("epub_tts_narrator.renderer.reference_asr.MlxParakeetVerifier", return_value=secondary) as parakeet:
            result = create_verifier({
                "model": "/local/whisper", "language": "fr", "reference_model": "/local/parakeet",
            })
        self.assertIsInstance(result, ReferenceCorroboratingVerifier)
        self.assertIs(result.primary, primary)
        self.assertIs(result.reference, secondary)
        whisper.assert_called_once_with("/local/whisper", language="fr")
        parakeet.assert_called_once_with("/local/parakeet")
        primary.transcribe.assert_not_called()
        primary.transcribe_reference.assert_not_called()
        secondary.transcribe_reference.assert_not_called()

    def test_wrapper_routes_each_method_to_its_own_backend_with_audio_only(self):
        primary, secondary = fake_primary(), fake_reference()
        wrapper = ReferenceCorroboratingVerifier(primary, secondary)
        path = Path("/local/reference.wav")
        self.assertEqual(wrapper.transcribe(path), primary.transcribe.return_value)
        primary.transcribe.assert_called_once_with(path)
        primary.transcribe_reference.assert_not_called()
        secondary.transcribe_reference.assert_not_called()
        self.assertEqual(wrapper.transcribe_reference(path), primary.transcribe_reference.return_value)
        primary.transcribe_reference.assert_called_once_with(path)
        secondary.transcribe_reference.assert_not_called()
        self.assertEqual(wrapper.corroborate_reference(path), secondary.transcribe_reference.return_value)
        secondary.transcribe_reference.assert_called_once_with(path)
        self.assertEqual(primary.transcribe.call_count, 1)
        self.assertEqual(primary.transcribe_reference.call_count, 1)

    def test_wrapper_identity_includes_both_backend_fingerprints(self):
        primary, secondary = fake_primary(), fake_reference()
        original_primary = deepcopy(primary.identity)
        wrapper = ReferenceCorroboratingVerifier(primary, secondary)
        self.assertEqual(wrapper.identity["backend"], "mlx-whisper")
        self.assertEqual(wrapper.identity["model_sha256"], primary.identity["model_sha256"])
        self.assertEqual(wrapper.identity["reference_corroborator"], secondary.identity)
        self.assertEqual(wrapper.corroboration_identity, secondary.identity)
        self.assertEqual(primary.identity, original_primary)

    def test_reference_cache_identity_changes_with_primary_secondary_and_configuration(self):
        original = ReferenceCorroboratingVerifier(fake_primary(), fake_reference())
        key = validation_key("Hello there", original, purpose="reference")
        changed_primary = ReferenceCorroboratingVerifier(fake_primary("primary-v2"), fake_reference())
        changed_secondary = ReferenceCorroboratingVerifier(fake_primary(), fake_reference("secondary-v2"))
        for verifier in (changed_primary, changed_secondary, fake_primary()):
            with self.subTest(identity=verifier.identity):
                self.assertNotEqual(key, validation_key("Hello there", verifier, purpose="reference"))
        self.assertEqual(key, validation_key(
            "Hello there", ReferenceCorroboratingVerifier(fake_primary(), fake_reference()), purpose="reference"))
        self.assertNotEqual(key, validation_key("Hello there", original, purpose="speech"))

    def test_close_releases_both_backends_and_tolerates_missing_close_methods(self):
        primary, secondary = fake_primary(), fake_reference()
        ReferenceCorroboratingVerifier(primary, secondary).close()
        primary.close.assert_called_once_with()
        secondary.close.assert_called_once_with()
        secondary.close.reset_mock()
        del primary.close
        ReferenceCorroboratingVerifier(primary, secondary).close()
        secondary.close.assert_called_once_with()
        del secondary.close
        ReferenceCorroboratingVerifier(primary, secondary).close()

    def test_pure_speech_never_consults_reference_backend_even_on_mismatch(self):
        primary, secondary = fake_primary(), fake_reference()
        secondary.transcribe_reference.side_effect = AssertionError("speech must not invoke secondary ASR")
        primary.transcribe_reference.side_effect = AssertionError("speech must request timed ASR")
        wrapper = ReferenceCorroboratingVerifier(primary, secondary)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            values = [round(4000 * math.sin(2 * math.pi * 220 * index / 44100)) for index in range(44100)]
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(struct.pack("<" + "h" * len(values), *values))
            correct = validate_audio(path, "Hello there", wrapper)
            mismatch = validate_audio(path, "Hello here", wrapper, purpose="speech")
        self.assertTrue(correct["passed"], correct)
        self.assertFalse(mismatch["passed"])
        self.assertIn("text_mismatch", mismatch["reasons"])
        self.assertEqual(primary.transcribe.call_count, 2)
        primary.transcribe_reference.assert_not_called()
        secondary.transcribe_reference.assert_not_called()


if __name__ == "__main__":
    unittest.main()
