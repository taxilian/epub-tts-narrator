"""Quality gate tests accept or reject audio with fake verifiers and never decode real speech."""

from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.quality import (
    MlxWhisperVerifier, QUALITY_DEFAULTS, compare_words, normalize_words,
    validate_audio, validate_policy, validation_key,
)


def transcript(text, duration=1.0):
    words = text.split()
    return {"text": text, "segments": [{
        "start": 0.0, "end": duration, "text": text,
        "avg_logprob": -0.1, "no_speech_prob": 0.001,
        "compression_ratio": 1.1,
        "words": [{"word": word, "start": i * duration / len(words),
                   "end": (i + 1) * duration / len(words), "probability": 0.9}
                  for i, word in enumerate(words)],
    }]}


class FakeVerifier:
    identity = {"backend": "test", "version": 1}

    def __init__(self, response):
        self.response = response
        self.calls = []

    def transcribe(self, path):
        self.calls.append(path)
        return deepcopy(self.response)


class RendererQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        self.write_audio()

    def write_audio(self, amplitude=0.1, duration=1.0, values=None, channels=1, rate=44100):
        if values is None:
            values = [round(amplitude * 32767 * math.sin(2 * math.pi * 220 * n / rate))
                      for n in range(round(duration * rate))]
        with wave.open(str(self.audio), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack("<" + "h" * len(values), *values))

    def test_silent_audio_fails_without_asr_and_report_is_strict_json(self):
        self.write_audio(amplitude=0)
        verifier = FakeVerifier(transcript("Hello."))
        report = validate_audio(self.audio, "Hello.", verifier, {})
        self.assertFalse(report["passed"])
        self.assertIn("silent_audio", report["reasons"])
        self.assertIsNone(report["metrics"]["rms_dbfs"])
        self.assertEqual(verifier.calls, [])
        json.dumps(report, allow_nan=False)

    def test_voiced_short_correct_phrase_passes(self):
        verifier = FakeVerifier(transcript("No!"))
        report = validate_audio(self.audio, "“No.”", verifier, {})
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["metrics"]["wer"], 0)
        self.assertEqual(verifier.calls, [self.audio])

    def test_quiet_voiced_audio_can_pass_but_inaudible_signal_fails(self):
        self.write_audio(amplitude=0.005)
        self.assertTrue(validate_audio(self.audio, "Hello", FakeVerifier(transcript("Hello")), {})["passed"])
        self.write_audio(amplitude=0.0001)
        report = validate_audio(self.audio, "Hello", FakeVerifier(transcript("Hello")), {})
        self.assertFalse(report["passed"])
        self.assertIn("audio_too_quiet", report["reasons"])

    def test_insert_delete_substitute_and_repeat_are_rejected(self):
        cases = [
            ("Stay here.", "Stay here forever.", "insertions", 1),
            ("Do not leave.", "Do leave.", "deletions", 1),
            ("Meet John.", "Meet James.", "substitutions", 1),
            ("Stay here.", "Stay here stay here.", "insertions", 2),
            ("Go.", "No.", "substitutions", 1),
        ]
        for expected, actual, key, count in cases:
            with self.subTest(actual=actual):
                report = validate_audio(self.audio, expected, FakeVerifier(transcript(actual)), {})
                self.assertFalse(report["passed"])
                self.assertIn("text_mismatch", report["reasons"])
                self.assertEqual(report["metrics"][key], count)
                self.assertTrue(report["metrics"]["alignment"])

    def test_number_and_punctuation_equivalence_is_bounded(self):
        self.assertEqual(compare_words("Twenty-one cats and 15 dogs.", "21 cats and fifteen dogs")["wer"], 0)
        self.assertEqual(compare_words("-5 and 50%", "minus five and fifty percent")["wer"], 0)
        for expected, actual in [
            ("15", "fifty"), ("can", "can't"), ("can't", "cant"),
            ("1 2", "three"), ("0.5", "zero five"), ("007", "seven"),
            ("$5", "five"), ("-5", "five"), ("John", "Jon"),
        ]:
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_words(expected, actual)["wer"], 0)

    def test_hyphenated_identifier_does_not_invent_a_spoken_minus(self):
        expected = "“Confirmation A34785-Derrins-34r42,”"
        self.assertEqual(compare_words(expected, "Confirmation A34785 Derrins 34r42")["wer"], 0)
        self.assertEqual(compare_words("Sector-34r42", "Sector 34r42")["wer"], 0)
        self.assertEqual(compare_words("AB12-unit-8", "AB12 unit 8")["wer"], 0)
        for recognized in ("Confirmation A34785, Darren's 34R42.",
                           "Confirmation A34785, Darins 34R42."):
            with self.subTest(recognized=recognized):
                report = validate_audio(self.audio, expected, FakeVerifier(transcript(recognized)))
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["transcript"], recognized)

    def test_identifier_pronunciation_does_not_excuse_wrong_or_missing_code_parts(self):
        expected = "Confirmation A34785-Derrins-34r42"
        for recognized in ("Confirmation A34786 Darren's 34R42",
                           "Confirmation A34785 Darren's 34R43",
                           "Confirmation A34785 Darren's R42",
                           "Confirmation A34785 Darren's 34Q42",
                           "Confirmation A34785 Darren's",
                           "Confirmation A34785 Darren's minus 34R42"):
            with self.subTest(recognized=recognized):
                report = validate_audio(self.audio, expected, FakeVerifier(transcript(recognized)))
                self.assertFalse(report["passed"], report)
                self.assertIn("text_mismatch", report["reasons"])

    def test_identifier_separator_normalization_retains_actual_signs_and_subtraction(self):
        for expected, recognized in (("-34", "34"), ("(+34)", "34"),
                                     ("temperature -34", "temperature 34"),
                                     ("34-42", "34 42"), ("x-34", "x 34"),
                                     ("x-34", "x plus 34"), ("A1-34", "A1 34"),
                                     ("A1-B2-34", "A1 B2 34")):
            with self.subTest(expected=expected, recognized=recognized):
                self.assertGreater(compare_words(expected, recognized)["wer"], 0)
        self.assertEqual(compare_words("x-34", "x minus thirty four")["wer"], 0)

    def test_empty_speakable_text_is_a_control_event_not_speech(self):
        verifier = FakeVerifier(transcript("Hello"))
        report = validate_audio(self.audio, "", verifier, {})
        self.assertFalse(report["passed"])
        self.assertEqual(report["reasons"], ["empty_expected_text"])
        self.assertEqual(verifier.calls, [])

    def test_duration_and_clipping_gates(self):
        self.write_audio(duration=0.02)
        self.assertIn("audio_too_short", validate_audio(self.audio, "Hello", FakeVerifier({}), {})["reasons"])
        self.write_audio(values=[32767, -32768] * 22050)
        self.assertIn("excessive_clipping", validate_audio(self.audio, "Hello", FakeVerifier({}), {})["reasons"])
        self.write_audio(duration=1)
        self.assertIn("audio_too_long", validate_audio(
            self.audio, "Hello", FakeVerifier({}), {"max_duration_seconds": 0.5})["reasons"])

    def test_internal_silence_is_rejected(self):
        tone = [round(4000 * math.sin(2 * math.pi * 220 * n / 44100)) for n in range(44100)]
        self.write_audio(values=tone + [0] * (3 * 44100) + tone)
        report = validate_audio(self.audio, "Hello", FakeVerifier({}), {})
        self.assertIn("internal_silence_stall", report["reasons"])

    def test_bad_and_truncated_wav_fail_closed(self):
        self.audio.write_bytes(b"not a wav")
        self.assertFalse(validate_audio(self.audio, "Hello", FakeVerifier({}), {})["passed"])
        self.write_audio()
        self.audio.write_bytes(self.audio.read_bytes()[:-4])
        report = validate_audio(self.audio, "Hello", FakeVerifier({}), {})
        self.assertTrue(any("truncated" in reason for reason in report["reasons"]))
        self.write_audio(rate=16000)
        self.assertFalse(validate_audio(self.audio, "Hello", FakeVerifier({}), {})["passed"])

    def test_float_nan_wav_is_rejected(self):
        # IEEE-float WAVE (format tag 3) is invalid at this normalized boundary.
        data = struct.pack("<f", float("nan")) * 100
        fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
        self.audio.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
                              + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data)
        report = validate_audio(self.audio, "Hello", FakeVerifier({}), {})
        self.assertFalse(report["passed"])
        json.dumps(report, allow_nan=False)

    def test_malformed_asr_never_passes_or_falls_back_to_expected(self):
        examples = [None, {}, {"text": "Hello"}, {"text": "Hello", "segments": []}]
        bad = transcript("Hello")
        bad["segments"][0]["avg_logprob"] = float("-inf")
        examples.append(bad)
        bad = transcript("Hello")
        bad["text"] = "Different"
        examples.append(bad)
        for response in examples:
            with self.subTest(response=response):
                report = validate_audio(self.audio, "Hello", FakeVerifier(response), {})
                self.assertFalse(report["passed"])
                self.assertTrue(any(reason.startswith("verification_failed") for reason in report["reasons"]))
                self.assertIsNone(report["transcript"])
                json.dumps(report, allow_nan=False)

    def test_alignment_failure_keeps_verified_transcript_and_omits_bad_times(self):
        result = transcript("Hello there")
        result["segments"][0]["words"][1]["start"] = 0.1
        report = validate_audio(self.audio, "Hello there", FakeVerifier(result))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["raw_transcript"], "Hello there")
        self.assertEqual(report["transcript"], "Hello there")
        self.assertEqual(report["words"], [])
        self.assertEqual(report["word_timing_status"], "unavailable")
        self.assertIn("overlap", report["alignment_warnings"][0])
        # Timing failure must not hide wrong speech or fabricate the expected text.
        mismatch = validate_audio(self.audio, "Goodbye there", FakeVerifier(result))
        self.assertFalse(mismatch["passed"])
        self.assertEqual(mismatch["reasons"], ["text_mismatch"])

    def test_missing_invalid_and_nonfinite_alignment_is_optional_for_speech(self):
        examples = []
        for words in (None, [], "not alignment"):
            result = transcript("Hello there")
            result["segments"][0]["words"] = words
            examples.append(result)
        for key, value in (("start", 0.1), ("end", 0.5), ("end", 5), ("end", float("nan")), ("probability", 2)):
            result = transcript("Hello there")
            result["segments"][0]["words"][1][key] = value
            examples.append(result)
        for response in examples:
            for purpose in ("reference", "speech"):
                with self.subTest(response=response, purpose=purpose):
                    report = validate_audio(self.audio, "Hello there", FakeVerifier(response), purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["transcript"], "Hello there")
                    self.assertEqual(report["words"], [])
                    if purpose == "speech":
                        self.assertEqual(report["word_timing_status"], "unavailable")
                    json.dumps(report, allow_nan=False)

    def test_invalid_segment_timing_is_omitted_without_rejecting_speech(self):
        examples = []
        result = transcript("Hello")
        del result["segments"][0]["start"]
        del result["segments"][0]["end"]
        examples.append(result)
        for start, end in ((-1, 1), (1, 0), (0, 5), (float("nan"), float("inf"))):
            result = transcript("Hello")
            result["segments"][0].update(start=start, end=end)
            examples.append(result)
        for response in examples:
            for purpose in ("reference", "speech"):
                with self.subTest(response=response, purpose=purpose):
                    report = validate_audio(self.audio, "Hello", FakeVerifier(response), purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertNotIn("start", report["metrics"]["asr_segments"][0])
                    json.dumps(report, allow_nan=False)

    def test_reference_keeps_exact_text_gate(self):
        expected = "I listened to the wind in the trees."
        actual = "I listened to the wind and the trees."
        report = validate_audio(self.audio, expected, FakeVerifier(transcript(actual)), purpose="reference")
        self.assertFalse(report["passed"])
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertEqual(report["transcript"], actual)
        self.assertEqual(report["metrics"]["substitutions"], 1)
        self.assertGreater(report["metrics"]["wer"], 0)

    def test_reference_rejects_missing_inconsistent_and_malformed_text_evidence(self):
        examples = [
            None, {}, {"text": "Hello"}, {"text": "Hello", "segments": []},
            {"text": "", "segments": [{"text": ""}]},
            {"text": "Hello", "segments": [{"text": "Different"}]},
            {"text": "Hello", "segments": [{}]},
        ]
        for key, value in (
            ("avg_logprob", float("-inf")), ("no_speech_prob", 2),
            ("compression_ratio", -1), ("temperature", -1),
        ):
            result = transcript("Hello")
            result["segments"][0][key] = value
            examples.append(result)
        for response in examples:
            with self.subTest(response=response):
                report = validate_audio(self.audio, "Hello", FakeVerifier(response), purpose="reference")
                self.assertFalse(report["passed"])
                self.assertTrue(any(reason.startswith("verification_failed") for reason in report["reasons"]))
                self.assertIsNone(report["transcript"])
                json.dumps(report, allow_nan=False)

    def test_reference_keeps_acoustic_safeguards_without_calling_asr(self):
        verifier = FakeVerifier(transcript("Hello"))
        self.write_audio(amplitude=0)
        report = validate_audio(self.audio, "Hello", verifier, purpose="reference")
        self.assertIn("silent_audio", report["reasons"])
        self.write_audio(values=[32767, -32768] * 22050)
        report = validate_audio(self.audio, "Hello", verifier, purpose="reference")
        self.assertIn("excessive_clipping", report["reasons"])
        tone = [round(4000 * math.sin(2 * math.pi * 220 * n / 44100)) for n in range(44100)]
        self.write_audio(values=tone + [0] * (3 * 44100) + tone)
        report = validate_audio(self.audio, "Hello", verifier, purpose="reference")
        self.assertIn("internal_silence_stall", report["reasons"])
        self.assertEqual(verifier.calls, [])

    def test_reference_dispatch_receives_only_audio_and_speech_uses_normal_method(self):
        class ReferenceVerifier(FakeVerifier):
            def __init__(self):
                super().__init__(transcript("Hello"))
                self.reference_calls = []

            def transcribe_reference(self, path):
                self.reference_calls.append(path)
                return {"text": "Hello", "segments": [{"text": "Hello"}]}

        verifier = ReferenceVerifier()
        self.assertTrue(validate_audio(self.audio, "Hello", verifier, purpose="reference")["passed"])
        self.assertEqual(verifier.reference_calls, [self.audio])
        self.assertEqual(verifier.calls, [])
        self.assertTrue(validate_audio(self.audio, "Hello", verifier)["passed"])
        self.assertEqual(verifier.reference_calls, [self.audio])
        self.assertEqual(verifier.calls, [self.audio])

    def test_verification_keys_separate_purposes_and_bind_policy_text_and_model(self):
        verifier = FakeVerifier(transcript("Hello"))
        speech = validation_key("Hello", verifier)
        reference = validation_key("Hello", verifier, purpose="reference")
        self.assertEqual(speech, validation_key("Hello", verifier, QUALITY_DEFAULTS, purpose="speech"))
        self.assertNotEqual(speech, reference)
        self.assertNotEqual(reference, validation_key("Different", verifier, purpose="reference"))
        self.assertNotEqual(reference, validation_key("Hello", verifier, {"max_wer": 0.1}, purpose="reference"))
        verifier.identity = {**verifier.identity, "version": 2}
        self.assertNotEqual(reference, validation_key("Hello", verifier, purpose="reference"))
        for purpose in ("typo", None, {}, []):
            with self.subTest(purpose=purpose):
                with self.assertRaises(ValueError):
                    validation_key("Hello", verifier, purpose=purpose)
                with self.assertRaises(ValueError):
                    validate_audio(self.audio, "Hello", verifier, purpose=purpose)

    def test_asr_exception_is_reported(self):
        verifier = FakeVerifier({})
        with patch.object(verifier, "transcribe", side_effect=RuntimeError("model unavailable")):
            report = validate_audio(self.audio, "Hello", verifier, {})
        self.assertFalse(report["passed"])
        self.assertIsNone(report["transcript"])
        self.assertIn("model unavailable", report["reasons"][0])

    def test_timestamp_rounding_is_clamped_to_audio_bounds(self):
        result = transcript("Hello")
        result["segments"][0]["end"] = 1.02
        result["segments"][0]["words"][0]["end"] = 1.02
        report = validate_audio(self.audio, "Hello", FakeVerifier(result), {})
        self.assertTrue(report["passed"])
        self.assertEqual(report["words"][0]["end"], 1.0)

    def test_policy_is_strict_and_finite(self):
        self.assertEqual(validate_policy({}), QUALITY_DEFAULTS)
        for policy in [
            [], {"typo": 3}, {"max_wer": True}, {"max_wer": -1}, {"max_wer": 2},
            {"max_wer": float("nan")}, {"max_duration_seconds": float("inf")},
            {"min_duration_seconds": 0}, {"min_duration_seconds": 200},
            {"silence_threshold_dbfs": -40},
        ]:
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    validate_policy(policy)

    def test_lexical_alignment_retains_exact_operation_indexes(self):
        result = compare_words("One red ship", "one blue ship today")
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["insertions"], 1)
        edits = [item for item in result["alignment"] if item["operation"] != "equal"]
        self.assertEqual(edits[0]["expected"], ["red"])
        self.assertEqual(edits[0]["actual"], ["blue"])
        self.assertEqual(edits[1]["expected_start"], 3)

    def test_model_identity_pins_files_options_and_language(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "weights.safetensors").write_bytes(b"fake model for unit testing only")
        with patch("epub_tts_narrator.renderer.quality.metadata.version", return_value="test-version"):
            first = MlxWhisperVerifier(model)
            different_language = MlxWhisperVerifier(model, language="fr")
        self.assertNotEqual(first.identity, different_language.identity)
        self.assertEqual(first.identity["package_version"], "test-version")
        self.assertEqual(first.identity["options"]["initial_prompt"], None)
        self.assertFalse(first.identity["options"]["condition_on_previous_text"])
        self.assertTrue(first.identity["options"]["word_timestamps"])
        self.assertFalse(first.identity["reference_options"]["word_timestamps"])
        calls = []
        fake_module = SimpleNamespace(transcribe=lambda *args, **kwargs: calls.append((args, kwargs)) or {})
        with patch.dict(sys.modules, {"mlx_whisper": fake_module}):
            first.transcribe(self.audio)
            first.transcribe_reference(self.audio)
        self.assertEqual(calls[0][0], (str(self.audio),))
        self.assertEqual(calls[0][1]["path_or_hf_repo"], str(model.resolve()))
        self.assertEqual(calls[0][1]["language"], "en")
        self.assertIsNone(calls[0][1]["initial_prompt"])
        self.assertNotIn("prompt", calls[0][1])
        self.assertTrue(calls[0][1]["word_timestamps"])
        self.assertEqual(calls[1][0], (str(self.audio),))
        self.assertEqual(calls[1][1], {**calls[0][1], "word_timestamps": False})
        self.assertIsNone(calls[1][1]["initial_prompt"])
        self.assertFalse(calls[1][1]["condition_on_previous_text"])
        self.assertNotIn("prompt", calls[1][1])
        (model / "weights.safetensors").write_bytes(b"changed model")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            first.transcribe(self.audio)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            first.transcribe_reference(self.audio)
        self.assertNotEqual(first.identity["model_sha256"], MlxWhisperVerifier(model).identity["model_sha256"])

    def test_model_requires_existing_local_weights(self):
        with self.assertRaises(ValueError):
            MlxWhisperVerifier("not-a-local-model/model")
        model = self.root / "empty-model"
        model.mkdir()
        with self.assertRaises(ValueError):
            MlxWhisperVerifier(model)


if __name__ == "__main__":
    unittest.main()
