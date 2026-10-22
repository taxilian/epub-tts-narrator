"""A text judge cannot excuse absent speech, even with maximum confidence."""

from array import array
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.jev import JevTranscriptJudge
from epub_tts_narrator.renderer.quality import validate_audio
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier


CONTEXT = (
    " We waited beside the window while the captain checked the doors."
    " Outside the rain fell steadily across the empty courtyard."
    " There was still time to finish our work before the travelers returned."
)

# Every pair represents a listener-visible loss or change, not a spelling choice.
UNACCEPTABLE = (
    ("missing narrator clause", "I understand, she added tiredly, but we must leave.",
     "I understand, but we must leave."),
    ("missing word", "Please wait here.", "Please wait."),
    ("missing intentional repetition", "She had had enough.", "She had enough."),
    ("added word repetition", "She had enough.", "She had had enough."),
    ("missing two clauses", "We refused, she explained quietly, then waited, I recalled, outside.",
     "We refused then waited outside."),
    ("added clause", "The captain opened the gate.",
     "The captain opened the gate and asked us to come inside."),
    ("repeated clause", "The captain opened the gate.",
     "The captain opened the gate. The captain opened the gate."),
    ("removed negation", "I will not open the door.", "I will open the door."),
    ("removed contracted negation", "I can't open the door.", "I can open the door."),
    ("removed negative command", "Don't open the door.", "Do open the door."),
    ("changed negation", "I will not open the door.", "I will now open the door."),
    ("changed number", "Bring seventeen maps.", "Bring seventy maps."),
    ("changed identifier digit", "Ask Derrins-34r42 to leave.", "Ask Derrins-35r42 to leave."),
    ("non soundalike replacement", "She put the book on the shelf.",
     "She put the candle on the shelf."),
    ("synonymous paraphrase", "She purchased a small automobile.", "She bought a little car."),
    ("spacing and name changes cannot excuse loss", "Derrins entered the sourcepalace, she added tiredly.",
     "Darren's entered the source palace."),
    ("compound spacing cannot hide nearby omitted word", "The sourcepalace gate opened.",
     "The source palace opened."),
)

ACCEPTABLE = (
    ("transliterated term", "She gathered qi in the courtyard.", "She gathered chi in the courtyard."),
    ("regional spelling", "Trust her judgement.", "Trust her judgment."),
    ("omitted written accent", "She studied techné.", "She studied techne."),
    ("compound boundary", "We entered the sourcepalace.", "We entered the source palace."),
    ("vowel recognition", "She put the doll on the shelf.", "She put the dull on the shelf."),
    ("stylized spelling", "Givve me the book.", "Give me the book."),
    ("word segmentation", "She admitted the mistake.", "She admit it the mistake."),
    ("invented name", "Derrins carried the book.", "Darren's carried the book."),
    ("short invented name", "Jas, wait here.", "Jace, wait here."),
    ("minor pronunciation", "Meet me at the libary.", "Meet me at the library."),
    ("contraction expansion", "I'm waiting beside the door.", "I am waiting beside the door."),
    ("negative contraction expansion", "Don't open the door.", "Do not open the door."),
)


def _recognized(text):
    """Fixture for the audio decoder boundary, with consistent timed segments."""
    words = text.split()
    return {"text": text, "segments": [{
        "text": text, "start": 0.0, "end": 1.0,
        "words": [{"word": word, "start": index / len(words),
                   "end": (index + 1) / len(words)}
                  for index, word in enumerate(words)],
    }]}


class RecordedRecognizer:
    identity = {"backend": "fixture-primary"}

    def __init__(self, transcript):
        self.result = _recognized(transcript)

    def transcribe(self, path):
        return deepcopy(self.result)

    def transcribe_reference(self, path):
        return deepcopy(self.result)


class TwoRecordedRecognizers(RecordedRecognizer):
    corroboration_identity = {"backend": "fixture-independent"}
    identity = {**RecordedRecognizer.identity, "corroboration": corroboration_identity}

    def __init__(self, primary, secondary):
        super().__init__(primary)
        self.secondary = _recognized(secondary)

    def corroborate_reference(self, path):
        return deepcopy(self.secondary)


class JevCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.audio = Path(temporary.name) / "speech.wav"
        samples = array("h", (round(3276 * math.sin(2 * math.pi * 220 * n / 44100))
                              for n in range(44100)))
        with wave.open(str(self.audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(44100)
            output.writeframes(samples.tobytes())
        self.judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-key"})
        # Only the external evaluator is substituted; all acceptance logic is real.
        evaluate = patch.object(self.judge._client, "evaluate", return_value={
            "model": "jev-fixture", "answers": {"roughly_same": {"type": "noul", "noul": .99}},
            "usage": {"input_tokens": 20, "output_tokens": 2},
        })
        evaluate.start()
        self.addCleanup(evaluate.stop)

    def test_high_judge_confidence_cannot_approve_missing_or_changed_content(self):
        for reason, expected, actual in UNACCEPTABLE:
            with self.subTest(reason=reason):
                decision = self.judge.judge(expected + CONTEXT, actual + CONTEXT)
                self.assertEqual(decision["decision"], "different", decision)

    def test_complete_soundalike_transcripts_remain_eligible(self):
        for reason, expected, actual in ACCEPTABLE:
            with self.subTest(reason=reason):
                decision = self.judge.judge(expected + CONTEXT, actual + CONTEXT)
                self.assertEqual(decision["decision"], "equivalent", decision)

    def test_audio_verification_cannot_be_overruled_by_high_judge_confidence(self):
        for purpose in ("reference", "speech"):
            for reason, expected, actual in UNACCEPTABLE:
                with self.subTest(purpose=purpose, reason=reason):
                    verifier = ReferenceTextJudgingVerifier(RecordedRecognizer(actual + CONTEXT), self.judge)
                    report = validate_audio(self.audio, expected + CONTEXT, verifier, purpose=purpose)
                    self.assertFalse(report["passed"], report)
                    self.assertIn("text_mismatch", report["reasons"])
                    self.assertNotIn("reference_text", report)

    def test_audio_verification_keeps_complete_spacing_and_pronunciation_variations(self):
        for purpose in ("reference", "speech"):
            for reason, expected, actual in ACCEPTABLE:
                with self.subTest(purpose=purpose, reason=reason):
                    verifier = ReferenceTextJudgingVerifier(RecordedRecognizer(actual + CONTEXT), self.judge)
                    report = validate_audio(self.audio, expected + CONTEXT, verifier, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    if purpose == "reference":
                        self.assertEqual(report["reference_text"], actual + CONTEXT)

    def test_independent_exact_recognition_can_rescue_faulty_primary_omission(self):
        expected = "I understand, she added tiredly, but we must leave." + CONTEXT
        partial = expected.replace("she added tiredly, ", "")
        for purpose in ("reference", "speech"):
            with self.subTest(purpose=purpose):
                verifier = ReferenceTextJudgingVerifier(TwoRecordedRecognizers(partial, expected), self.judge)
                report = validate_audio(self.audio, expected, verifier, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["verified_transcript"], expected)
                self.assertEqual(report["transcript"], partial)
                if purpose == "reference":
                    self.assertEqual(report["reference_text"], expected)

    def test_two_recognizers_agreeing_on_missing_speech_cannot_approve_it(self):
        expected = "I understand, she added tiredly, but we must leave." + CONTEXT
        partial = expected.replace("she added tiredly, ", "")
        for purpose in ("reference", "speech"):
            with self.subTest(purpose=purpose):
                verifier = ReferenceTextJudgingVerifier(TwoRecordedRecognizers(partial, partial), self.judge)
                report = validate_audio(self.audio, expected, verifier, purpose=purpose)
                self.assertFalse(report["passed"], report)
                self.assertIn("text_mismatch", report["reasons"])
                self.assertNotIn("reference_text", report)

    def test_rejection_message_identifies_local_coverage_failure(self):
        from epub_tts_narrator.renderer.workflow import _verification_failure_detail
        expected = "I understand, she added tiredly, but we must leave."
        partial = expected.replace("she added tiredly, ", "")
        verifier = ReferenceTextJudgingVerifier(RecordedRecognizer(partial), self.judge)
        report = validate_audio(self.audio, expected, verifier, purpose="speech")
        message = _verification_failure_detail(report)
        self.assertFalse(report["passed"])
        self.assertIn("coverage check failed", message)
        self.assertIn("she added tiredly", message)
        self.assertNotIn("JEV judged", message)


if __name__ == "__main__":
    unittest.main()
