"""Live-evaluation inputs mirror runtime evidence without pretending to label saved audio."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.engines import _directed_text
from epub_tts_narrator.renderer.recurring_resolution import failed_take_context
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_resolver import Client


SPEC = importlib.util.spec_from_file_location(
    "evaluate_speech_resolution", Path(__file__).resolve().parents[1] / "scripts/evaluate_speech_resolution.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class SpeechResolutionEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.expected = "Arrgh! The pain has returned."
        self.request = speech_request(reference(), self.expected)
        self.request["chunk"]["segment"]["delivery"]["direction"] = "pained, breaking into a cry"

    def save_record(self, number=1, *, reasons=None, heard="Oh! The pain has returned.", request=None):
        request = deepcopy(request or self.request)
        audio = self.directory / f"attempt-{number:03d}.wav"
        audio.write_bytes(f"saved waveform {number}".encode())
        text, delivery = _directed_text(self.expected, request["chunk"]["segment"]["delivery"]["direction"])
        record = {"attempt": number, "request": request, "request_hash": digest(request),
                  "audio_file": audio.name, "audio_sha256": file_hash(audio),
                  "engine_metadata": {"synthesis_text": text, **delivery},
                  "report": {"passed": False, "expected_text": self.expected,
                             "transcript": heard, "reasons": reasons or ["text_mismatch"],
                             "secondary_transcript": "The pain has returned.",
                             "corroboration": {"transcript": "Do not prefer this older recognizer field."},
                             "metrics": {"wer": 0.2, "duration_seconds": 2.0,
                                         "longest_internal_silence_seconds": 0.2,
                                         "alignment": [{"repeated": "x" * 10000}]}}}
        write_json(self.directory / f"attempt-{number:03d}.json", record)
        return record

    def test_saved_context_uses_shared_bounded_builder_and_includes_acoustic_takes(self):
        first = self.save_record()
        second = self.save_record(2, reasons=["internal_silence_stall"], heard=None)
        case = EVALUATION.saved_case(self.directory)
        self.assertEqual(case["failures"], [failed_take_context(self.expected, first),
                                             failed_take_context(self.expected, second)])
        self.assertNotIn("alignment", case["failures"][0]["metrics"])
        self.assertEqual(case["failures"][0]["secondary_transcript"], "The pain has returned.")
        self.assertEqual(case["failures"][0]["generation"]["direction"], "pained, breaking into a cry")
        self.assertIn("[pained, breaking into a cry]", case["failures"][0]["generation"]["synthesis_text"])
        self.assertTrue(case["can_plain_delivery"])
        self.assertIsNone(case["expected_actions"], "Unlabelled saved audio is not required to be accepted")

    def test_supported_capability_requires_a_bound_request_and_exact_source_text(self):
        first = self.save_record()
        path = self.directory / "attempt-001.json"
        first["request_hash"] = digest("other request")
        write_json(path, first)
        with self.assertRaisesRegex(ValueError, "request hash mismatch"):
            EVALUATION.saved_case(self.directory)
        first["request_hash"] = digest(first["request"])
        first["report"]["expected_text"] = "A different passage."
        write_json(path, first)
        with self.assertRaisesRegex(ValueError, "request text differs"):
            EVALUATION.saved_case(self.directory)

    def test_old_unbound_records_do_not_gain_plain_delivery_capability(self):
        first = self.save_record()
        first.pop("request")
        first.pop("request_hash")
        write_json(self.directory / "attempt-001.json", first)
        self.assertFalse(EVALUATION.saved_case(self.directory)["can_plain_delivery"])

    def test_valid_plain_proposal_passes_unlabelled_case_without_accepting_failed_audio(self):
        self.save_record()
        case = EVALUATION.saved_case(self.directory)
        client = Client("plain_delivery")
        resolver = SpeechResolver(client)
        with patch.object(EVALUATION, "create_speech_resolver", return_value=resolver):
            result = EVALUATION.evaluate(case, {})
        self.assertTrue(result["correct"])
        self.assertEqual(result["action"], "plain_delivery")
        self.assertEqual(result["evaluation_basis"], "valid_proposal")
        self.assertIn("plain_delivery", result["audit"]["payload"]["permitted_actions"])
        self.assertNotIn("acceptance_transcript", result["validated_proposal"])

    def test_invalid_proposal_still_fails_unlabelled_case(self):
        request = deepcopy(self.request)
        request["engine"]["engine"] = "unknown-engine"
        self.save_record(request=request)
        case = EVALUATION.saved_case(self.directory)
        self.assertFalse(case["can_plain_delivery"])
        with patch.object(EVALUATION, "create_speech_resolver", return_value=SpeechResolver(Client("plain_delivery"))):
            result = EVALUATION.evaluate(case, {})
        self.assertFalse(result["correct"])
        self.assertIsNone(result["validated_proposal"])

    def test_synthetic_expected_actions_remain_strict(self):
        case = EVALUATION.synthetic_case(EVALUATION.CASES[0])
        self.assertEqual(case["expected_actions"], ["accept"])
        self.assertFalse(case["can_plain_delivery"])
        with patch.object(EVALUATION, "create_speech_resolver", return_value=SpeechResolver(Client("retry"))):
            result = EVALUATION.evaluate(case, {})
        self.assertFalse(result["correct"])
        self.assertEqual(result["action"], "retry")
        self.assertEqual(result["evaluation_basis"], "expected_action")


if __name__ == "__main__":
    unittest.main()
