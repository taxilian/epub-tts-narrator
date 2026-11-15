"""Each take is brought toward the chapter's speech level, keeping contrast and directed soft or loud lines."""

import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave

RATE = 44100


def write_take(path, level_db, *, seconds=2.0):
    """A steady voiced tone at ``level_db`` (RMS dBFS) with short gaps, like syllables."""
    amplitude = 10 ** (level_db / 20) * math.sqrt(2)
    samples = []
    for index in range(int(seconds * RATE)):
        t = index / RATE
        on = (t * 4) % 1 < 0.7
        samples.append(int(max(-32767, min(32767, amplitude * 32767 * math.sin(2 * math.pi * 180 * t)))) if on else 0)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return str(path)


def clip(path, direction=""):
    return {"chunk_id": Path(path).stem, "wav_path": path, "spoken_text": "Words here.",
            "segment": {"delivery": {"direction": direction}}}


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires numpy (renderer environment)")
class TakeLevelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def take(self, name, level, direction=""):
        return clip(write_take(self.root / f"{name}-take.wav", level), direction)

    def test_an_unexplained_quiet_take_is_raised_to_the_edge_of_the_band_and_ordinary_ones_kept(self):
        from epub_tts_narrator.renderer.take_levels import gains
        clips = [self.take("a", -20), self.take("b", -20), self.take("c", -21.5), self.take("quiet", -30)]
        result, report = gains(clips, tolerance_db=3.0)
        self.assertEqual(set(result), {3})
        self.assertAlmostEqual(result[3], 7.0, delta=0.6)  # -30 → about -23 (3 dB under the typical -20)
        self.assertAlmostEqual(report["reference_db"], -20, delta=1.0)

    def test_a_directed_whisper_may_stay_quiet_and_a_shout_loud(self):
        from epub_tts_narrator.renderer.take_levels import expected_offset, gains
        whisper = self.take("whisper", -26, "whispered urgently")
        shout = self.take("shout", -15, "shouting")
        clips = [self.take("a", -20), self.take("b", -20), whisper, shout]
        result, report = gains(clips, tolerance_db=3.0)
        self.assertEqual(result, {})
        self.assertEqual([expected_offset(item) for item in (whisper, shout)], [-6.0, 4.0])
        # Not directed soft, the same quiet take would be raised.
        self.assertIn(2, gains([self.take("c", -20), self.take("d", -20), self.take("e", -26)], tolerance_db=3.0)[0])

    def test_a_raise_never_pushes_the_peak_past_the_ceiling(self):
        from epub_tts_narrator.renderer.take_levels import PEAK_CEILING_DBFS, gains, speech_level
        loud_peaks = self.take("peaky", -30)
        # A single near-full-scale click: the take may be raised only up to the ceiling.
        with wave.open(loud_peaks["wav_path"], "rb") as audio:
            params, frames = audio.getparams(), bytearray(audio.readframes(audio.getnframes()))
        frames[1000:1002] = struct.pack("<h", 30000)
        with wave.open(loud_peaks["wav_path"], "wb") as audio:
            audio.setparams(params)
            audio.writeframes(bytes(frames))
        result, _ = gains([self.take("a", -20), self.take("b", -20), loud_peaks], tolerance_db=3.0)
        peak = speech_level(loud_peaks["wav_path"])[1]
        self.assertLessEqual(result[2], PEAK_CEILING_DBFS - peak + 0.01)

    def test_a_gained_copy_has_the_same_length_and_the_new_level(self):
        from epub_tts_narrator.renderer.take_levels import gained_copy, speech_level
        source = write_take(self.root / "source-take.wav", -30)
        copy = gained_copy(source, self.root / "leveled-take.wav", 6.0)
        with wave.open(source, "rb") as a, wave.open(str(copy), "rb") as b:
            self.assertEqual(a.getnframes(), b.getnframes())
        self.assertAlmostEqual(speech_level(copy)[0] - speech_level(source)[0], 6.0, delta=0.2)

    def test_levelling_is_on_by_default_and_part_of_the_chapter_settings(self):
        from epub_tts_narrator.renderer.chapter_cache import chapter_settings
        settings = chapter_settings({})
        self.assertIs(settings["level_takes"], True)
        self.assertEqual(settings["take_level_tolerance_db"], 3.0)


if __name__ == "__main__":
    unittest.main()
