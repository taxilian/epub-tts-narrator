"""A take carrying a steady high-pitched tone is rejected, never chosen, and retired when in use."""

import importlib.util
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
import wave

from epub_tts_narrator.renderer.common import read_json, write_json

RATE = 44100


def write_speechlike(path, *, tone_hz=None, tone_level=0.02, seconds=4.0):
    """Syllables of a 140 Hz voice with broad hiss now and then, plus an optional steady tone throughout."""
    rng = random.Random(7)
    samples = []
    for index in range(int(seconds * RATE)):
        t = index / RATE
        syllable = math.sin(math.pi * ((t * 4) % 1)) ** 2  # four syllables a second
        voice = sum(math.sin(2 * math.pi * 140 * h * t) / h for h in range(1, 25)) * 0.08 * syllable
        hiss = rng.uniform(-1, 1) * 0.05 if (t * 4) % 3 < 0.4 else 0.0  # a broad "s" now and then
        value = voice + hiss
        if tone_hz:
            value += tone_level * math.sin(2 * math.pi * tone_hz * t)
        samples.append(max(-32767, min(32767, int(value * 32767))))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy (renderer environment)")
class SteadyToneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_a_steady_high_tone_is_found_and_speech_alone_is_not(self):
        from epub_tts_narrator.renderer.steady_tone import check, measure
        clean, tonal = self.root / "clean-take.wav", self.root / "tonal-take.wav"
        write_speechlike(clean)
        write_speechlike(tonal, tone_hz=7100)
        self.assertIsNone(check(clean), measure(clean))
        found = check(tonal)
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found["frequency_hz"], 7100, delta=150)
        self.assertGreaterEqual(found["presence"], 0.5)

    def test_a_short_take_of_one_held_sound_is_not_judged(self):
        from epub_tts_narrator.renderer.steady_tone import check, measure
        held = self.root / "held-sound-take.wav"
        write_speechlike(held, tone_hz=7100, seconds=1.0)
        self.assertIsNone(measure(held))
        self.assertIsNone(check(held))

    def test_verification_rejects_it_before_recognition(self):
        from epub_tts_narrator.renderer.quality import validate_audio
        import test_renderer_reference_consensus as consensus
        tonal = self.root / "tonal-take.wav"
        write_speechlike(tonal, tone_hz=10000)
        verifier = consensus.PrimaryVerifier("Hello there.")
        report = validate_audio(tonal, "Hello there.", verifier, {}, purpose="speech")
        self.assertEqual(report["reasons"], ["steady_tone"])
        self.assertEqual(verifier.calls, [])
        from epub_tts_narrator.renderer.steady_tone import describe
        self.assertRegex(describe(report), r"^a steady 10\.[01] kHz tone")

    def test_takes_in_use_with_a_tone_are_retired_once_and_others_kept(self):
        from epub_tts_narrator.renderer.steady_tone import SWEEP_FILENAME, VERSION, retire_tonal
        work = self.root / "render-work"

        def accepted(name, *, tone=None, metadata=None):
            folder = work / "clips" / (name * 64)[:64]
            folder.mkdir(parents=True)
            write_speechlike(folder / "take.wav", tone_hz=tone)
            write_json(folder / "accepted.json", {
                "attempt": 1, "audio_sha256": name * 64, "audio_file": "take.wav",
                "engine_metadata": metadata or {}, "request": {"kind": "speech"},
                "report": {"passed": True, "reasons": []}})
            write_json(folder / "exhaustion-selection.json", {"audit": {}})
            return folder
        tonal, clean = accepted("a", tone=7100), accepted("b")
        chosen = accepted("c", tone=7100, metadata={"manual_selection": {}})
        self.assertEqual(retire_tonal(work), [tonal])
        self.assertFalse((tonal / "accepted.json").exists())
        self.assertFalse((tonal / "exhaustion-selection.json").exists())  # its choice cannot republish it
        evidence = read_json(next(tonal.glob("steady-tone-*.json")))
        self.assertEqual(evidence["evidence"]["reason"], "steady_tone")
        for folder in (clean, chosen):
            self.assertTrue((folder / "accepted.json").exists())
        self.assertEqual(read_json(work / SWEEP_FILENAME)["version"], VERSION)
        self.assertEqual(retire_tonal(work), [])  # measured once


if __name__ == "__main__":
    unittest.main()
