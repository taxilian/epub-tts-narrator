"""Deterministic ordinal equivalence through the verification surface.

A transcript may spell a spoken ordinal either way ("seventh" / "7th").
Normalizers must treat identical spoken ordinals as equal while every changed
value, malformed suffix, cardinal confusion, leading zero, sign, decimal and
missing word keeps failing.
"""

from copy import deepcopy
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer import quality
from epub_tts_narrator.renderer.quality import (
    compare_words, normalize_words, validate_audio, validation_key,
)
from epub_tts_narrator.renderer.speech_coverage import speech_coverage


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


class OrdinalEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        self.write_audio()

    def write_audio(self, amplitude=0.1, duration=1.0):
        rate = 44100
        values = [round(amplitude * 32767 * math.sin(2 * math.pi * 220 * n / rate))
                  for n in range(round(duration * rate))]
        with wave.open(str(self.audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack("<" + "h" * len(values), *values))

    def test_identical_spoken_ordinals_match_either_spelling(self):
        for expected, actual in (
                ("seventh", "7th"), ("7th", "seventh"),
                ("fifth", "5th"), ("5th", "fifth"),
                ("eighth", "8th"), ("twelfth", "12th"),
                ("1st", "first"), ("2nd", "second"), ("3rd", "third"),
                ("4th", "fourth"),
                ("the seventh guest", "the 7th guest")):
            with self.subTest(expected=expected, actual=actual):
                self.assertEqual(compare_words(expected, actual)["wer"], 0)

    def test_compound_ordinals_expand_to_their_spoken_words(self):
        for digits, spoken in (
                ("20th", "twentieth"), ("21st", "twenty first"),
                ("22nd", "twenty second"), ("23rd", "twenty third"),
                ("100th", "one hundredth"), ("101st", "one hundred first"),
                ("110th", "one hundred tenth"), ("112th", "one hundred twelfth"),
                ("1,000th", "one thousandth"), ("10,000th", "ten thousandth"),
                ("999999th",
                 "nine hundred ninety nine thousand nine hundred ninety ninth")):
            with self.subTest(digits=digits):
                self.assertEqual(normalize_words(digits), normalize_words(spoken))
                self.assertEqual(compare_words(digits, spoken)["wer"], 0)
                self.assertEqual(compare_words(spoken, digits)["wer"], 0)

    def test_teens_and_ending_zero_suffixes_expand_while_wrong_teens_suffix_stays_literal(self):
        for digits, spoken in (
                ("10th", "tenth"), ("11th", "eleventh"), ("12th", "twelfth"),
                ("13th", "thirteenth"), ("111th", "one hundred eleventh"),
                ("211th", "two hundred eleventh"), ("100th", "one hundredth"),
                ("1,000th", "one thousandth")):
            self.assertEqual(normalize_words(digits), normalize_words(spoken))
        for malformed in ("11st", "12nd", "13rd", "112rd"):
            with self.subTest(malformed=malformed):
                self.assertEqual(normalize_words(malformed), [malformed])

    def test_changed_ordinal_values_are_not_equivalent(self):
        for expected, actual in (
                ("seventh", "8th"), ("8th", "seventh"), ("fifth", "4th"),
                ("21st", "22nd"), ("twelfth", "13th"),
                ("the 7th bell", "the eighth bell")):
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_words(expected, actual)["wer"], 0)

    def test_malformed_ordinal_suffixes_stay_literal_and_are_rejected(self):
        for malformed in ("7st", "7nd", "1rd", "2th", "3th", "4st",
                          "21th", "22th", "0st"):
            with self.subTest(malformed=malformed):
                self.assertEqual(normalize_words(malformed), [malformed])
        for expected, actual in (
                ("seventh", "7st"), ("first", "1rd"), ("twenty first", "21th")):
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_words(expected, actual)["wer"], 0)

    def test_cardinal_and_ordinal_confusions_are_not_equivalent(self):
        for expected, actual in (
                ("seven", "7th"), ("7th", "seven"), ("seventh", "7"),
                ("7", "seventh"), ("twenty one", "21st"), ("21", "21st"),
                ("chapter seven", "chapter 7th")):
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_words(expected, actual)["wer"], 0)

    def test_leading_zero_signed_decimal_and_out_of_range_ordinals_stay_rejected(self):
        for literal in ("07th", "007th", "-7th", "+7th", "7.5th",
                        "0,7th", "7,000,000th"):
            with self.subTest(literal=literal):
                self.assertEqual(normalize_words(literal), [literal])
        for expected, actual in (
                ("seventh", "07th"), ("minus seventh", "-7th"),
                ("seventh", "7.5th")):
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_words(expected, actual)["wer"], 0)

    def test_missing_or_extra_words_around_an_ordinal_still_fail(self):
        for expected, actual in (
                ("the seventh bell", "the 7th"), ("the 7th bell", "the seventh"),
                ("seventh", "seventh bell")):
            with self.subTest(expected=expected, actual=actual):
                self.assertGreater(compare_words(expected, actual)["wer"], 0)

    def test_validate_audio_accepts_identical_spoken_ordinals_and_still_rejects_changes(self):
        verifier = FakeVerifier(transcript("The 7th Bell."))
        report = validate_audio(self.audio, "The seventh Bell.", verifier, {})
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["metrics"]["wer"], 0)
        self.assertNotIn("text_mismatch", report["reasons"])
        verifier = FakeVerifier(transcript("The fifth guest arrived."))
        report = validate_audio(self.audio, "The 5th guest arrived.", verifier, {})
        self.assertTrue(report["passed"], report)
        verifier = FakeVerifier(transcript("The 8th Bell."))
        report = validate_audio(self.audio, "The seventh Bell.", verifier, {})
        self.assertFalse(report["passed"], report)
        self.assertIn("text_mismatch", report["reasons"])

    def test_speech_coverage_accepts_identical_spoken_ordinals_before_any_judgment(self):
        result = speech_coverage("the seventh year.", "the 7th year.")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["reason"], "exact_spoken_content")
        result = speech_coverage("the seventh year.", "the 7th")
        self.assertFalse(result["passed"], result)
        self.assertEqual(result["reason"], "unmatched_spoken_content")

    def test_validation_key_binds_the_policy_version_so_stale_evidence_cannot_be_reused(self):
        verifier = FakeVerifier(transcript("Hello"))
        keys = set()
        for version in (998, 999):
            with patch.object(quality, "QUALITY_VERSION", version):
                keys.add(validation_key("Chapter 7th.", verifier, {}))
        self.assertEqual(len(keys), 2)


if __name__ == "__main__":
    unittest.main()
