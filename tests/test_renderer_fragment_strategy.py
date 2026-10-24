from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.common import digest, read_json
from epub_tts_narrator.renderer.engines import _directed_text
from epub_tts_narrator.renderer.speech_identity import generation_identity
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, VerificationUnavailable
import test_renderer_fragment_recovery as fixtures


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe required")
class FragmentStrategyTests(unittest.TestCase):
    setUp = fixtures.FragmentRecoveryTests.setUp
    generate = fixtures.FragmentRecoveryTests.generate
    recover = fixtures.FragmentRecoveryTests.recover
    fragment_audit = fixtures.FragmentRecoveryTests.fragment_audit
    part_directory = fixtures.FragmentRecoveryTests.part_directory

    def plain(self, text, path, seed):
        self.calls.append(text)
        spoken = "This is—" if text == fixtures.PART else text
        metadata = self.engine._write(spoken, path, seed)
        directed, delivery = _directed_text(text.strip(), "")
        return {**metadata, "synthesis_text": directed, **delivery}

    def test_model_stop_does_not_discard_an_executable_untried_tail_strategy(self):
        self.config["verification"]["recurrence_threshold"] = 1
        self.client.proposal = {"action": "stop", "edits": [], "reason": "Whole takes are exhausted."}
        _, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(self.calls.count(fixtures.PART), 1)
        self.assertEqual(self.calls.count("This is—"), 1)
        self.assertEqual(self.calls.count("ack!"), 1)
        self.assertLessEqual(len(self.calls), self.config["verification"]["max_resolution_attempts"])

    def test_approved_parent_plain_delivery_is_bound_and_inherited_by_both_pieces(self):
        self.config["verification"]["recurrence_threshold"] = 1
        self.client.proposal = {"action": "plain_delivery", "edits": [], "reason": "Remove acting controls."}
        _, accepted = self.recover()
        _, audit = self.fragment_audit(accepted)
        self.assertEqual(audit["delivery_recovery"]["original_text"], fixtures.PART)
        self.assertEqual(audit["delivery_recovery"]["replacement_direction"], "")
        for row in audit["evidence"]:
            piece = row["acceptance"]
            self.assertEqual(piece["request"]["synthesis_request"]["chunk"]["segment"]["delivery"]["direction"], "")
            self.assertFalse(piece["engine_metadata"].get("direction_applied", False))
        self.assertEqual(self.request["chunk"]["segment"]["delivery"]["direction"], self.direction)
        self.assertEqual(generation_identity(accepted["request"], self.voices,
                                             engine_metadata=accepted["engine_metadata"]),
                         generation_identity(self.request, self.voices))
        bad = deepcopy(accepted["engine_metadata"])
        del bad["recovery"]["parts"][1]["acceptance"]["engine_metadata"]["fragment_recovery"]["delivery_recovery"]
        with self.assertRaises(ValueError):
            generation_identity(accepted["request"], self.voices, engine_metadata=bad)

    def test_failed_join_is_not_reconcatenated_or_reverified_on_resume(self):
        self.asr.final_transcript = "The requested final words are missing."
        with self.assertRaises(SpeechVerificationFailed):
            self.recover()
        directory = self.part_directory()
        before = {path.name: path.read_bytes() for path in directory.glob("attempt-*.json")}
        checks = len(self.asr.calls)
        with self.assertRaisesRegex(SpeechVerificationFailed, "Cutoff-tail recovery is exhausted"):
            self.recover()
        self.assertEqual({path.name: path.read_bytes() for path in directory.glob("attempt-*.json")}, before)
        self.assertEqual(len(self.asr.calls), checks)

    def test_join_verifier_outage_retries_same_assembled_waveform_without_more_synthesis(self):
        self.asr.final_error = RuntimeError("Temporary recognizer outage")
        for _ in range(2):
            with self.assertRaises(VerificationUnavailable):
                self.recover()
        directory = self.part_directory()
        pending = read_json(directory / "attempt-003.json")
        calls = list(self.calls)
        self.asr.final_error = None
        _, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(self.calls, calls)
        self.assertEqual(self.fragment_audit(accepted)[0]["acceptance"]["audio_sha256"], pending["audio_sha256"])


if __name__ == "__main__":
    unittest.main()
