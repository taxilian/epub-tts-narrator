"""A take whose only fault is one long pause is verified for its words, not remade."""

import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer import stall_recovery
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.quality import validate_audio
from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, verified_take
import test_renderer_reference_consensus as consensus


TEXT = "The ship landed. Everyone waited for the doors to open."


def write_take(path, pause_seconds, tone_seconds=0.5):
    """Tone, a pause, then tone again."""
    rate = 44100
    tone = [round(0.1 * 32767 * math.sin(2 * math.pi * 220 * index / rate))
            for index in range(round(rate * tone_seconds))]
    values = tone + [0] * round(rate * pause_seconds) + tone
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack("<" + "h" * len(values), *values))


def config(**render):
    return {"quality": {}, "censorship": {"enabled": False},
            "render": {"max_attempts": 2, "seed": 7, "sample_rate": 44100, "ffmpeg": "unused",
                       "normalize_pauses": True, "normalize_take_pauses": True,
                       "sentence_pause_ms": 500, "clause_pause_ms": 250, **render}}


class StallRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wav = Path(self.temp.name) / "stalled-take-fixture.wav"

    def stalled(self, pause, heard=TEXT, settings=None, request=None):
        write_take(self.wav, pause, tone_seconds=2.0)
        verifier = consensus.PrimaryVerifier(heard)
        settings = settings or config()
        report = validate_audio(self.wav, TEXT, verifier, settings["quality"], purpose="speech")
        self.assertEqual(report["reasons"], ["internal_silence_stall"])
        self.assertEqual(verifier.calls, [])  # The stall gate runs before recognition.
        return stall_recovery.recover(report, self.wav, TEXT, verifier, settings,
                                      request or {"chunk": {}}, notify=lambda _: None), verifier

    def test_a_stalled_take_with_every_word_is_accepted_and_marked(self):
        report, verifier = self.stalled(3.0)
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(verifier.calls), 1)
        marker = report["silence_stall_recovery"]
        self.assertEqual((marker["decision"], marker["method"], marker["limit_seconds"]),
                         ("accepted", "assembly_pause_cap", 7.5))
        self.assertEqual(marker["stalled_report"]["reasons"], ["internal_silence_stall"])
        self.assertTrue(stall_recovery.recovered(report))

    def test_a_stalled_take_missing_words_still_fails_on_its_words(self):
        report, _ = self.stalled(3.0, heard="The ship landed.")
        self.assertFalse(report["passed"])
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertEqual(report["silence_stall_recovery"]["decision"], "words_checked")

    def test_a_very_long_pause_is_not_recovered(self):
        report, verifier = self.stalled(9.0)
        self.assertEqual(report["reasons"], ["internal_silence_stall"])
        self.assertEqual(report["silence_stall_recovery"]["decision"], "too_long")
        self.assertEqual(verifier.calls, [])
        # Assessed once: it is not offered again.
        self.assertFalse(stall_recovery.stalled_only(report))

    def test_recovery_needs_assembly_to_shorten_the_pause(self):
        for settings, request in ((config(normalize_take_pauses=False), {"chunk": {}}),
                                  (config(normalize_pauses=False), {"chunk": {}}),
                                  (config(sentence_pause_ms=0, clause_pause_ms=0), {"chunk": {}}),
                                  ({**config(), "censorship": {"enabled": True}}, {"chunk": {}}),
                                  (config(), {"chunk": {}, "performer": "lane"})):
            with self.subTest(settings=settings["render"], request=request):
                report, verifier = self.stalled(3.0, settings=settings, request=request)
                self.assertEqual(report["reasons"], ["internal_silence_stall"])
                self.assertNotIn("silence_stall_recovery", report)
                self.assertEqual(verifier.calls, [])


class StallRecoveryWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "takes"
        self.directory.mkdir()
        self.generated = []

    def take(self, settings, verifier):
        def generate(path, seed):
            self.generated.append(path)
            write_take(path, 3.0)
            return {"seed": seed}
        with patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                   side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())):
            return verified_take(TEXT, generate, self.directory, {"fixture": True}, verifier, settings,
                                 purpose="speech", notify=lambda _: None)

    def test_a_stalled_take_is_used_without_another_synthesis(self):
        verifier = consensus.PrimaryVerifier(TEXT)
        path, accepted = self.take(config(), verifier)
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(accepted["attempt"], 1)
        self.assertEqual(accepted["report"]["silence_stall_recovery"]["decision"], "accepted")

    def test_a_take_saved_as_stalled_is_rechecked_once_recovery_is_available(self):
        verifier = consensus.PrimaryVerifier(TEXT)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(config(normalize_take_pauses=False), verifier)
        self.assertEqual(len(self.generated), 2)
        saved = read_json(self.directory / "attempt-001.json")
        self.assertEqual(saved["report"]["reasons"], ["internal_silence_stall"])
        path, accepted = self.take(config(), verifier)
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(accepted["attempt"], 1)
        rechecked = read_json(self.directory / "attempt-001.json")
        self.assertEqual(rechecked["policy_recheck"]["previous_report"]["reasons"], ["internal_silence_stall"])

    def test_a_take_judged_too_long_under_a_lower_limit_is_rechecked(self):
        verifier = consensus.PrimaryVerifier(TEXT)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(config(normalize_take_pauses=False), verifier)
        for path in self.directory.glob("attempt-*.json"):
            record = read_json(path)
            record["report"]["silence_stall_recovery"] = {
                "version": 1, "decision": "too_long", "limit_seconds": 2.0,
                "longest_internal_silence_seconds": record["report"]["metrics"]["longest_internal_silence_seconds"]}
            write_json(path, record)
        path, accepted = self.take(config(), verifier)
        self.assertEqual(len(self.generated), 2)  # nothing new was synthesized
        self.assertEqual(accepted["attempt"], 1)
        self.assertEqual(accepted["report"]["silence_stall_recovery"]["decision"], "accepted")

    def test_a_too_long_assessment_is_not_repeated(self):
        verifier = consensus.PrimaryVerifier(TEXT)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(config(normalize_take_pauses=False), verifier)
        for path in self.directory.glob("attempt-*.json"):
            record = read_json(path)
            record["report"]["silence_stall_recovery"] = {"version": 1, "decision": "too_long"}
            write_json(path, record)
        before = len(verifier.calls)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(config(), verifier)
        self.assertEqual(len(verifier.calls), before)
        self.assertEqual(len(self.generated), 2)
        self.assertNotIn("policy_recheck", read_json(self.directory / "attempt-001.json"))


if __name__ == "__main__":
    unittest.main()
