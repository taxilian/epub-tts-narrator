"""A take's record is saved as soon as its audio exists, and audio with no record is never used."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.audio_naming import speech_take_filename
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.workflow import verified_take
import test_renderer_reference_consensus as consensus
from test_renderer_stall_recovery import TEXT, config, write_take

REQUEST = {"fixture": "unrecorded-take"}


class UnrecordedTakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "takes"
        self.directory.mkdir()
        self.generated = []

    def take(self, verifier, *, stop_after_normalizing=False):
        def generate(path, seed):
            self.generated.append(path)
            write_take(path, 0.2)
            return {"seed": seed, "made_by": "fixture-engine"}

        def normalize(raw, wav, **kwargs):
            Path(wav).write_bytes(Path(raw).read_bytes())
            if stop_after_normalizing:
                raise KeyboardInterrupt  # the render stopped here
        with patch("epub_tts_narrator.renderer.workflow.normalize_clip", side_effect=normalize):
            return verified_take(TEXT, generate, self.directory, REQUEST, consensus.PrimaryVerifier(TEXT),
                                 config(), purpose="speech", notify=lambda _: None)

    def test_a_take_stopped_before_verification_keeps_its_engine_record(self):
        with self.assertRaises(KeyboardInterrupt):
            self.take(None, stop_after_normalizing=True)
        # The stop came after the audio was written but before its record was saved.
        self.assertFalse((self.directory / "attempt-001.json").exists())
        path, accepted = self.take(None)
        self.assertEqual(accepted["engine_metadata"]["made_by"], "fixture-engine")
        self.assertEqual(len(self.generated), 2)  # the unrecorded audio was made again
        self.assertTrue((self.directory / speech_take_filename(REQUEST, 1)).exists())  # and kept on disk
        self.assertNotEqual(path.name, speech_take_filename(REQUEST, 1))

    def test_a_take_stopped_during_verification_resumes_with_its_record(self):
        class Stopping(consensus.PrimaryVerifier):
            def transcribe(self, path):
                raise KeyboardInterrupt
        with patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                   side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())):
            with self.assertRaises(KeyboardInterrupt):
                verified_take(TEXT, lambda path, seed: (self.generated.append(path), write_take(path, 0.2),
                                                        {"seed": seed, "made_by": "fixture-engine"})[2],
                              self.directory, REQUEST, Stopping(TEXT), config(), purpose="speech",
                              notify=lambda _: None)
        saved = read_json(self.directory / "attempt-001.json")
        self.assertEqual(saved["engine_metadata"]["made_by"], "fixture-engine")
        _, accepted = self.take(None)
        self.assertEqual(len(self.generated), 1)  # verified from the saved audio, nothing made again
        self.assertEqual(accepted["engine_metadata"]["made_by"], "fixture-engine")


if __name__ == "__main__":
    unittest.main()
