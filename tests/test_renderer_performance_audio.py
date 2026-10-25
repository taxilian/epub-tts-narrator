"""Intentional pauses excuse only their measured, word-aligned silence."""

from array import array
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
import wave

from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.renderer.assembly import compose_chapter
from epub_tts_narrator.renderer.pacing import audio_edges
from epub_tts_narrator.renderer.performance_audio import pause_edges
from epub_tts_narrator.renderer.quality import validate_audio, validation_key
from test_renderer_assembly import clip
from test_renderer_quality import FakeVerifier


RATE = 44100


def plan(text, *controls):
    return {"version": 1, "text_sha256": text_sha256(text), "controls": list(controls)}


def pause(offset, duration=3000):
    return {"kind": "pause", "offset": offset, "duration_ms": duration}


def transcript(text, intervals):
    tokens = text.split()
    return {"text": text, "segments": [{"text": text, "start": 0.0, "end": intervals[-1][1],
                                        "words": [{"word": word, "start": start, "end": end}
                                                  for word, (start, end) in zip(tokens, intervals)]}]}


class PerformanceAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "speech.wav"

    def audio_parts(self, *parts, rate=RATE):
        values = array("h")
        for seconds, active in parts:
            count = round(seconds * rate)
            values.extend(round(4000 * math.sin(2 * math.pi * 220 * number / rate)) if active else 0
                          for number in range(count))
        with wave.open(str(self.audio), "wb") as stream:
            stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            stream.writeframes(values.tobytes())

    def test_intentional_internal_pause_requires_its_exact_word_boundary(self):
        self.audio_parts((1, True), (3, False), (1, True))
        recognized = transcript("Hold now.", [(0, 1), (4, 5)])
        verifier = FakeVerifier(recognized)
        legacy = validate_audio(self.audio, "Hold now.", verifier)
        self.assertIn("internal_silence_stall", legacy["reasons"])
        self.assertEqual(verifier.calls, [])
        result = validate_audio(self.audio, "Hold now.", verifier, performance=plan("Hold now.", pause(4)))
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["performance_audio"]["matched_pauses"][0]["offset"], 4)
        self.assertGreater(result["metrics"]["longest_internal_silence_seconds"], 2.9)
        self.assertEqual(result["performance_audio"]["longest_internal_silence_seconds"], 0)

    def test_pause_at_a_different_word_or_at_the_clip_edge_cannot_excuse_a_stall(self):
        self.audio_parts((1, True), (3, False), (1, True))
        for offset in (0, 2, 9):
            with self.subTest(offset=offset):
                result = validate_audio(self.audio, "Hold now.", FakeVerifier(transcript("Hold now.", [(0, 1), (4, 5)])),
                                        performance=plan("Hold now.", pause(offset)))
                self.assertIn("internal_silence_stall", result["reasons"])
                self.assertEqual(result["performance_audio"]["matched_pauses"], [])

    def test_a_requested_pause_is_not_a_global_or_unbounded_silence_allowance(self):
        self.audio_parts((1, True), (4, False), (1, True))
        result = validate_audio(self.audio, "Hold now.", FakeVerifier(transcript("Hold now.", [(0, 1), (5, 6)])),
                                performance=plan("Hold now.", pause(4)))
        self.assertIn("internal_silence_stall", result["reasons"])
        self.audio_parts((1, True), (3, False), (1, True), (3, False), (1, True))
        result = validate_audio(self.audio, "Hold here now.", FakeVerifier(transcript("Hold here now.", [(0, 1), (4, 5), (8, 9)])),
                                performance=plan("Hold here now.", pause(4)))
        self.assertIn("internal_silence_stall", result["reasons"])
        self.assertEqual(len(result["performance_audio"]["matched_pauses"]), 1)

    def test_only_the_located_pause_is_removed_from_the_silence_fraction(self):
        self.audio_parts((.2, True), (3, False), (.2, True))
        result = validate_audio(self.audio, "Hold now.", FakeVerifier(transcript("Hold now.", [(0, .2), (3.2, 3.4)])),
                                performance=plan("Hold now.", pause(4)))
        self.assertTrue(result["passed"], result)
        self.assertGreater(result["metrics"]["silence_fraction"], .85)
        self.assertLess(result["performance_audio"]["silence_fraction"], .05)

    def test_silent_and_corrupt_audio_still_fail_before_transcription(self):
        self.audio_parts((4, False))
        verifier = FakeVerifier(transcript("Hold now.", [(0, 1), (3, 4)]))
        result = validate_audio(self.audio, "Hold now.", verifier, performance=plan("Hold now.", pause(4)))
        self.assertIn("silent_audio", result["reasons"])
        self.assertEqual(verifier.calls, [])
        self.audio.write_bytes(b"broken wave")
        self.assertFalse(validate_audio(self.audio, "Hold now.", verifier, performance=plan("Hold now.", pause(4)))["passed"])
        self.assertEqual(verifier.calls, [])

    def test_unusable_word_timing_leaves_the_normal_silence_gate_in_force(self):
        self.audio_parts((1, True), (3, False), (1, True))
        recognized = transcript("Hold now.", [(0, 1), (4, 5)])
        recognized["segments"][0]["words"][1]["start"] = .1
        result = validate_audio(self.audio, "Hold now.", FakeVerifier(recognized), performance=plan("Hold now.", pause(4)))
        self.assertIn("internal_silence_stall", result["reasons"])
        self.assertEqual(result["word_timing_status"], "unavailable")
        self.assertEqual(result["performance_audio"]["timing_status"], "unavailable")

    def test_nonexact_words_do_not_supply_speculative_pause_alignment(self):
        self.audio_parts((1, True), (3, False), (1, True))
        result = validate_audio(self.audio, "Hold now.", FakeVerifier(transcript("Hold here.", [(0, 1), (4, 5)])),
                                performance=plan("Hold now.", pause(4)))
        self.assertFalse(result["passed"])
        self.assertEqual(result["performance_audio"]["timing_status"], "lexical_alignment_unavailable")

    def test_inline_sounds_never_become_required_asr_words_or_allow_empty_speech(self):
        self.audio_parts((1, True))
        control = {"kind": "vocalization", "offset": 5, "sound": "laugh"}
        result = validate_audio(self.audio, "Hello", FakeVerifier(transcript("Hello", [(0, 1)])),
                                performance=plan("Hello", control))
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["expected_text"], "Hello")
        result = validate_audio(self.audio, "", FakeVerifier({}),
                                performance=plan("", {"kind": "vocalization", "offset": 0, "sound": "laugh"}))
        self.assertEqual(result["reasons"], ["empty_expected_text"])

    def test_validation_identity_binds_controls_and_rejects_stale_text(self):
        verifier = FakeVerifier({})
        legacy = validation_key("Hello now", verifier)
        self.assertEqual(legacy, validation_key("Hello now", verifier, performance=None))
        self.assertNotEqual(legacy, validation_key("Hello now", verifier, performance=plan("Hello now", pause(5))))
        self.assertNotEqual(validation_key("Hello now", verifier, performance=plan("Hello now", pause(5))),
                            validation_key("Hello now", verifier, performance=plan("Hello now", pause(5, 1000))))
        with self.assertRaises(ValueError):
            validate_audio(self.audio, "Different", verifier, performance=plan("Hello now", pause(5)))

    def test_assembly_retains_requested_outer_pause_lengths_without_unbounded_engine_padding(self):
        self.audio_parts((.8, False), (.1, True), (.9, False), rate=10000)
        original = self.audio.read_bytes()
        record = plan("Hello.", pause(0, 300), pause(6, 500))
        edges = audio_edges(self.audio, **pause_edges(record, "Hello."))
        self.assertEqual((edges["start"], edges["end"]), (5000, 14000))
        item = clip(self.audio, break_after="none")
        item["expressive_performance"] = record
        result = compose_chapter([item], self.root / "chapter.wav", sample_rate=10000)
        self.assertEqual(result["frames"], 9000)
        self.assertEqual(result["segments"][0]["source_audio_start_frame"], 5000)
        self.assertEqual(result["segments"][0]["source_audio_end_frame"], 14000)
        self.assertEqual(self.audio.read_bytes(), original)

    def test_outer_quotation_formatting_does_not_hide_an_intentional_edge_pause(self):
        text = '“Hello.”'
        self.assertEqual(pause_edges(plan(text, pause(1, 300), pause(7, 500)), text),
                         {"leading_pause_ms": 300, "trailing_pause_ms": 500})


if __name__ == "__main__":
    unittest.main()
