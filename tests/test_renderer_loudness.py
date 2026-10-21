"""Real audio tests for non-destructive, cached casting loudness matching."""

import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer.common import file_hash
from epub_tts_narrator.renderer.loudness import SampleNormalizer


def signal(path, amplitude=6000, *, seconds=4, transient=False):
    rate = 44100
    frames = int(seconds * rate)
    data = bytearray()
    for i in range(frames):
        envelope = .55 + .45 * math.sin(2 * math.pi * i / rate / .7)
        sample = amplitude * envelope * math.sin(2 * math.pi * 220 * i / rate)
        if transient and rate < i < rate + 200:
            sample = 29000 * math.sin(2 * math.pi * 600 * i / rate)
        data.extend(struct.pack("<h", round(sample)))
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(data)
    return path


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for real audio tests")
class SampleLoudnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.normalizer = SampleNormalizer(self.root / "work" / "voice-previews")

    def measurement(self, path):
        # Re-measure the encoded WAV, independently of the cached report.
        result = subprocess.run([
            "ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-i", str(path),
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=50:print_format=json", "-f", "null", "-",
        ], capture_output=True, check=True)
        report = result.stderr.decode()
        return json.JSONDecoder().raw_decode(report[report.rindex('{'):])[0]

    def test_quiet_and_loud_samples_match_without_changing_references_or_timing(self):
        quiet = signal(self.root / "quiet.wav", 500)
        loud = signal(self.root / "loud.wav", 20000)
        original = {path: file_hash(path) for path in (quiet, loud)}
        measured = []
        for source in (quiet, loud):
            preview = self.normalizer.prepare(source)
            stats = self.measurement(preview)
            measured.append(float(stats["input_i"]))
            self.assertAlmostEqual(measured[-1], -20, delta=.3)
            self.assertLessEqual(float(stats["input_tp"]), -1.4)
            with wave.open(str(source)) as before, wave.open(str(preview)) as after:
                self.assertEqual(before.getparams(), after.getparams())
            self.assertEqual(file_hash(source), original[source])
            self.assertNotEqual(preview, source)
        self.assertLess(abs(measured[0] - measured[1]), .3)

    def test_peak_limiter_handles_a_quiet_sample_with_a_loud_transient(self):
        source = signal(self.root / "transient.wav", 1600, transient=True)
        preview = self.normalizer.prepare(source)
        stats = self.measurement(preview)
        self.assertAlmostEqual(float(stats["input_i"]), -20, delta=1)
        self.assertLessEqual(float(stats["input_tp"]), -1.3)

    def test_aliases_and_restarts_reuse_content_cache_but_edits_and_settings_do_not(self):
        source = signal(self.root / "original.wav")
        copied = self.root / "inherited-copy.wav"
        shutil.copyfile(source, copied)
        preview = self.normalizer.prepare(source)
        restarted = SampleNormalizer(self.normalizer.cache_dir)
        with patch.object(restarted, "_run", side_effect=AssertionError("unexpected FFmpeg call")):
            self.assertEqual(restarted.prepare(copied), preview)
        changed = SampleNormalizer(self.normalizer.cache_dir, target_lufs=-23)
        self.assertNotEqual(changed.prepare(source), preview)
        signal(copied, 3000)
        self.assertNotEqual(restarted.prepare(copied), preview)

    def test_corrupted_cache_is_not_played(self):
        source = signal(self.root / "source.wav")
        preview = self.normalizer.prepare(source)
        data = bytearray(preview.read_bytes())
        data[-20] ^= 1
        preview.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.normalizer.prepare(source)

    def test_silent_and_too_short_samples_fail_without_publishing_or_altering_audio(self):
        for name, amplitude, seconds in (("silent", 0, 1), ("short", 6000, .05)):
            source = signal(self.root / f"{name}.wav", amplitude, seconds=seconds)
            original = file_hash(source)
            with self.assertRaisesRegex(ValueError, "measurable loudness"):
                self.normalizer.prepare(source)
            self.assertEqual(file_hash(source), original)
            self.assertEqual(list(self.normalizer.cache_dir.iterdir()), [])

    def test_timeout_or_interrupt_does_not_publish_partial_preview(self):
        source = signal(self.root / "source.wav")
        original = file_hash(source)
        run = self.normalizer._run
        for error in (KeyboardInterrupt(), ValueError("timed out")):
            calls = 0

            def fail_second(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    args[-1].write_bytes(b"partial wav")
                    raise error
                return run(*args)

            with patch.object(self.normalizer, "_run", side_effect=fail_second):
                with self.assertRaises(type(error)):
                    self.normalizer.prepare(source)
            self.assertEqual(list(self.normalizer.cache_dir.iterdir()), [])
            self.assertEqual(file_hash(source), original)
        self.assertTrue(self.normalizer.prepare(source).is_file())

    def test_external_process_failure_reports_error_and_leaves_no_partial_output(self):
        source = signal(self.root / "source.wav")
        with patch("epub_tts_narrator.renderer.loudness.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("ffmpeg", 120)):
            with self.assertRaisesRegex(ValueError, "Unable to normalize"):
                self.normalizer.prepare(source)
        self.assertEqual(list(self.normalizer.cache_dir.iterdir()), [])

    def test_preview_cache_cannot_follow_symlinks_into_an_ancestor_store(self):
        source = signal(self.root / "source.wav")
        parent = self.root / "parent-voices"
        parent.mkdir()
        self.normalizer.cache_dir.parent.mkdir()
        self.normalizer.cache_dir.symlink_to(parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.normalizer.prepare(source)
        self.assertEqual(list(parent.iterdir()), [])


class LoudnessSettingsTests(unittest.TestCase):
    def test_invalid_targets_are_rejected(self):
        for key, values in (("target_lufs", (True, "-20", float("nan"), -41, -9)),
                            ("true_peak_db", (False, float("inf"), -10, 1))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    SampleNormalizer("unused", **{key: value})


if __name__ == "__main__":
    unittest.main()
