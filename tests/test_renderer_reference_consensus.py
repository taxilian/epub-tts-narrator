from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from audiobook_renderer.quality import validate_audio, validation_key


EXPECTED = (
    "The morning light fell across the quiet room. "
    "I opened the door and listened to the wind in the trees. "
    "There was still time to decide what to do next."
)
ACTUAL = EXPECTED.replace("wind in", "wind and")


def transcript(text):
    words = text.split()
    return {"text": text, "segments": [{
        "text": text, "start": 0.0, "end": 1.0,
        "words": [{"word": word, "start": index / len(words),
                   "end": (index + 1) / len(words)}
                  for index, word in enumerate(words)],
    }]}


class PrimaryVerifier:
    identity = {"backend": "primary-fixture"}

    def __init__(self, text=ACTUAL):
        self.primary = transcript(text)
        self.calls = []

    def transcribe(self, path):
        self.calls.append(("speech", path))
        return deepcopy(self.primary)

    def transcribe_reference(self, path):
        self.calls.append(("reference", path))
        return deepcopy(self.primary)


class IndependentVerifier(PrimaryVerifier):
    corroboration_identity = {"backend": "independent-fixture", "model": "second-model"}
    identity = {**PrimaryVerifier.identity, "corroboration": corroboration_identity}

    def __init__(self, primary=ACTUAL, secondary=ACTUAL):
        super().__init__(primary)
        self.secondary = transcript(secondary)

    def corroborate_reference(self, path):
        # A one-argument method also detects accidental expected-text prompts.
        self.calls.append(("corroboration", path))
        if isinstance(self.secondary, Exception):
            raise self.secondary
        return deepcopy(self.secondary)


class ReferenceConsensusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name) / "reference.wav"
        self.write_audio()

    def write_audio(self, amplitude=0.1):
        values = [
            round(amplitude * 32767 * math.sin(2 * math.pi * 220 * index / 44100))
            for index in range(44100)
        ]
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(44100)
            output.writeframes(struct.pack("<" + "h" * len(values), *values))

    def check(self, verifier, expected=EXPECTED, policy=None, purpose="reference"):
        report = validate_audio(self.audio, expected, verifier, policy, purpose=purpose)
        json.dumps(report, allow_nan=False)
        return report

    def test_agreed_substitution_saves_actual_words_and_original_edit_evidence(self):
        verifier = IndependentVerifier()
        report = self.check(verifier)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], ACTUAL)
        self.assertTrue(report["reference_text_adjusted"])
        self.assertEqual(report["expected_text"], EXPECTED)
        self.assertEqual(report["metrics"]["substitutions"], 1)
        self.assertGreater(report["metrics"]["wer"], 0)
        self.assertTrue(report["metrics"]["alignment"])
        self.assertEqual(report["corroboration"]["decision"], "agreed_reference_transcript")
        self.assertEqual(report["corroboration"]["identity"], verifier.corroboration_identity)
        self.assertEqual(report["corroboration"]["transcript"], ACTUAL)
        self.assertEqual(verifier.calls, [
            ("reference", self.audio), ("corroboration", self.audio)])

    def test_consensus_compares_complete_normalized_transcripts(self):
        verifier = IndependentVerifier(secondary=ACTUAL.upper().replace(".", "!"))
        report = self.check(verifier)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], ACTUAL)
        verifier = IndependentVerifier(secondary=ACTUAL.replace("quiet", "silent"))
        report = self.check(verifier)
        self.assertFalse(report["passed"])
        self.assertEqual(report["corroboration"]["decision"], "disagreed")

    def test_exact_reference_keeps_requested_format_without_second_decoder(self):
        verifier = IndependentVerifier(primary=EXPECTED.upper())
        report = self.check(verifier)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], EXPECTED)
        self.assertFalse(report["reference_text_adjusted"])
        self.assertNotIn("corroboration", report)
        self.assertEqual(verifier.calls, [("reference", self.audio)])

    def test_silence_does_not_call_either_decoder(self):
        self.write_audio(amplitude=0)
        verifier = IndependentVerifier()
        report = self.check(verifier)
        self.assertFalse(report["passed"])
        self.assertIn("silent_audio", report["reasons"])
        self.assertEqual(verifier.calls, [])
        self.assertNotIn("corroboration", report)

    def test_speech_mismatch_remains_strict_without_corroboration(self):
        verifier = IndependentVerifier()
        report = self.check(verifier, purpose="speech")
        self.assertFalse(report["passed"])
        self.assertIn("text_mismatch", report["reasons"])
        self.assertNotIn("reference_text", report)
        self.assertNotIn("corroboration", report)
        self.assertEqual(verifier.calls, [("speech", self.audio)])
        # Existing caller-specified speech tolerances retain their behavior.
        report = self.check(verifier, policy={"max_wer": 0.1}, purpose="speech")
        self.assertTrue(report["passed"], report)
        self.assertNotIn("reference_text", report)

    def test_reference_does_not_inherit_permissive_speech_text_tolerance(self):
        verifier = PrimaryVerifier()
        report = self.check(verifier, policy={"max_wer": 1.0})
        self.assertFalse(report["passed"])
        self.assertIn("text_mismatch", report["reasons"])
        self.assertEqual(report["corroboration"]["decision"], "unavailable")
        self.assertNotIn("reference_text", report)

    def test_insertions_deletions_or_large_mismatch_skip_second_decoder(self):
        cases = [
            (EXPECTED, EXPECTED + " Extra."),
            (EXPECTED, EXPECTED.replace("quiet ", "")),
            ("Go now.", "Stay now."),
            (EXPECTED, EXPECTED.replace("morning", "evening")
             .replace("quiet", "silent").replace("wind", "breeze")
             .replace("room", "chamber")),
        ]
        for expected, actual in cases:
            with self.subTest(actual=actual):
                verifier = IndependentVerifier(primary=actual, secondary=actual)
                report = self.check(verifier, expected=expected)
                self.assertFalse(report["passed"])
                self.assertIn("text_mismatch", report["reasons"])
                self.assertEqual(
                    report["corroboration"]["decision"], "mismatch_outside_reference_limits")
                self.assertEqual(verifier.calls, [("reference", self.audio)])
                self.assertNotIn("reference_text", report)

    def test_two_substitutions_at_ten_percent_boundary_can_be_corroborated(self):
        expected = "one two three four five six seven eight nine ten " * 2
        actual = expected.replace("one", "eleven", 1).replace("two", "twelve", 1)
        verifier = IndependentVerifier(primary=actual, secondary=actual)
        report = self.check(verifier, expected=expected)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["metrics"]["substitutions"], 2)
        self.assertEqual(report["metrics"]["wer"], 0.1)

    def test_exact_secondary_script_match_resolves_only_bounded_primary_discrepancy(self):
        verifier = IndependentVerifier(secondary=EXPECTED)
        report = self.check(verifier)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["corroboration"]["decision"], "secondary_matches_requested")
        self.assertEqual(report["reference_text"], EXPECTED)
        self.assertFalse(report["reference_text_adjusted"])
        self.assertEqual(report["transcript"], ACTUAL)
        self.assertEqual(report["corroboration"]["transcript"], EXPECTED)
        self.assertEqual(report["metrics"]["substitutions"], 1)
        self.assertEqual(verifier.calls, [
            ("reference", self.audio), ("corroboration", self.audio)])

        verifier = IndependentVerifier(primary=EXPECTED + " Extra.", secondary=EXPECTED)
        report = self.check(verifier)
        self.assertFalse(report["passed"])
        self.assertEqual(verifier.calls, [("reference", self.audio)])
        self.assertEqual(
            report["corroboration"]["decision"], "mismatch_outside_reference_limits")

    def test_invalid_or_failed_secondary_rejects_with_diagnostics(self):
        bad_segment_text = transcript(ACTUAL)
        bad_segment_text["segments"][0]["text"] = EXPECTED
        for secondary in (
            None, {}, {"text": ACTUAL, "segments": []},
            {"text": "", "segments": [{"text": ""}]},
            bad_segment_text, RuntimeError("independent decoder unavailable"),
        ):
            with self.subTest(secondary=secondary):
                verifier = IndependentVerifier()
                verifier.secondary = secondary
                report = self.check(verifier)
                self.assertFalse(report["passed"])
                self.assertIn("text_mismatch", report["reasons"])
                self.assertEqual(report["corroboration"]["decision"], "verification_failed")
                self.assertIn("error", report["corroboration"])
                self.assertNotIn("reference_text", report)

    def test_malformed_primary_never_reaches_secondary(self):
        verifier = IndependentVerifier()
        verifier.primary["segments"] = []
        report = self.check(verifier)
        self.assertFalse(report["passed"])
        self.assertEqual(verifier.calls, [("reference", self.audio)])
        self.assertNotIn("corroboration", report)

    def test_secondary_alignment_is_not_required_for_reference_words(self):
        verifier = IndependentVerifier()
        verifier.secondary["segments"] = [{"text": ACTUAL, "start": float("nan")}]
        report = self.check(verifier)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["corroboration"]["asr_segments"], [{}])

    def test_identity_change_invalidates_reference_cache(self):
        first = IndependentVerifier()
        second = IndependentVerifier()
        second.identity = {**second.identity, "corroboration": {"model": "different-model"}}
        self.assertNotEqual(
            validation_key(EXPECTED, first, purpose="reference"),
            validation_key(EXPECTED, second, purpose="reference"))


if __name__ == "__main__":
    unittest.main()
