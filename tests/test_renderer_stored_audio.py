"""Kept WAVs become FLAC files that restore them byte for byte; readers find either form."""

import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from epub_tts_narrator.renderer import stored_audio
from epub_tts_narrator.renderer.common import file_hash, pcm_sha256


def tone(path, *, seconds=0.5, channels=1, rate=44100, extra_chunk=b"", trailer=b""):
    """A PCM16 WAV; ``extra_chunk`` goes before the data chunk, ``trailer`` after it."""
    frames = int(seconds * rate)
    samples = [round(4000 * math.sin(2 * math.pi * 330 * i / rate)) for i in range(frames) for _ in range(channels)]
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    if extra_chunk or trailer:
        data = path.read_bytes()
        body = data[12:]
        data = b"RIFF" + struct.pack("<I", 4 + len(extra_chunk) + len(body) + len(trailer)) + b"WAVE" + extra_chunk + body + trailer
        path.write_bytes(data)
    return path


def float_wav(path):
    """The engines' raw output: 32-bit float WAV, which FLAC cannot hold exactly."""
    import numpy
    import soundfile
    soundfile.write(str(path), numpy.linspace(-.2, .2, 4410, dtype="float32"), 44100, subtype="FLOAT")
    return path


class StoredAudioTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_storing_replaces_the_wav_with_a_flac_that_restores_identical_bytes(self):
        cases = {"mono": {}, "stereo": {"channels": 2},
                 "info chunk": {"extra_chunk": b"LIST\x0c\x00\x00\x00INFOISFT\x00\x00\x00\x00"},
                 "trailing chunk": {"trailer": b"junk\x03\x00\x00\x00abc\x00"}}
        for label, options in cases.items():
            with self.subTest(label):
                wav = tone(self.root / f"{label}-take.wav", **options)
                original, checksum = wav.read_bytes(), file_hash(wav)
                frames = pcm_sha256(wav) if options.get("channels", 1) == 1 else None
                stored = stored_audio.store(wav)
                self.assertEqual(stored, wav.with_suffix(".flac"))
                self.assertFalse(wav.exists())
                self.assertLess(stored.stat().st_size, len(original))
                self.assertTrue(stored_audio.is_stored(wav))
                self.assertTrue(stored_audio.audio_exists(wav))
                self.assertEqual(stored_audio.locate(wav), stored)
                self.assertEqual(stored_audio.audio_hash(wav), checksum)
                self.assertEqual(stored_audio.read_bytes(wav), original)
                if frames is not None:
                    self.assertEqual(stored_audio.pcm_sha256(wav), frames)
                with stored_audio.open_wave(wav) as audio:
                    self.assertEqual(audio.getnchannels(), options.get("channels", 1))
                    self.assertEqual(audio.getnframes(), 22050)
                with stored_audio.materialize(wav) as copy:
                    self.assertEqual(copy.read_bytes(), original)
                    self.assertTrue(copy.name.endswith(".wav"))
                    self.assertIn(wav.stem, copy.name)
                self.assertFalse(copy.exists(), "a restored copy is deleted afterwards")

    def test_an_existing_wav_is_read_directly_and_stays_authoritative_beside_its_flac(self):
        wav = tone(self.root / "take.wav")
        self.assertEqual(stored_audio.locate(wav), wav)
        self.assertEqual(stored_audio.audio_hash(wav), file_hash(wav))
        with stored_audio.materialize(wav) as copy:
            self.assertEqual(copy, wav)
        backup = wav.read_bytes()
        stored_audio.store(wav)
        wav.write_bytes(backup)  # An interruption left both forms.
        self.assertEqual(stored_audio.locate(wav), wav)
        self.assertEqual(sorted(path.name for path in stored_audio.audio_files(wav)), ["take.flac", "take.wav"])
        self.assertEqual(stored_audio.store(wav), wav.with_suffix(".flac"))
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["take.flac"])

    def test_a_stale_flac_from_an_interruption_is_replaced_from_the_wav(self):
        wav = tone(self.root / "take.wav")
        other = tone(self.root / "other.wav", seconds=.25)
        stored_audio.store(other).rename(wav.with_suffix(".flac"))
        checksum = file_hash(wav)
        stored_audio.store(wav)
        self.assertEqual(stored_audio.audio_hash(wav), checksum)

    def test_audio_flac_cannot_hold_exactly_and_foreign_flacs_keep_the_wav(self):
        raw = float_wav(self.root / "take-raw.wav")
        before = raw.read_bytes()
        self.assertIsNone(stored_audio.store(raw))
        self.assertEqual(raw.read_bytes(), before)
        self.assertFalse(raw.with_suffix(".flac").exists())

        wav = tone(self.root / "take.wav")
        foreign = wav.with_suffix(".flac")
        foreign.write_bytes(b"fLaC not ours")
        self.assertIsNone(stored_audio.store(wav))
        self.assertTrue(wav.exists())
        self.assertEqual(foreign.read_bytes(), b"fLaC not ours")

    def test_a_wav_that_differs_from_its_record_is_never_stored(self):
        wav = tone(self.root / "take.wav")
        with self.assertRaises(stored_audio.StoredAudioError):
            stored_audio.store(wav, expected_sha256="0" * 64)
        self.assertTrue(wav.exists())
        self.assertFalse(wav.with_suffix(".flac").exists())

    def test_a_flac_that_does_not_restore_exactly_keeps_the_wav(self):
        wav = tone(self.root / "take.wav")
        before = wav.read_bytes()
        real = stored_audio._restore

        def corrupt(path, sink):
            if sink is None:
                raise stored_audio.StoredAudioError("mismatch")
            return real(path, sink)
        with patch.object(stored_audio, "_restore", side_effect=corrupt):
            with self.assertRaises(stored_audio.StoredAudioError):
                stored_audio.store(wav)
        self.assertEqual(wav.read_bytes(), before)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["take.wav"])

    def test_damaged_or_altered_stored_audio_is_refused_rather_than_trusted(self):
        wav = tone(self.root / "take.wav")
        stored = stored_audio.store(wav)
        data = bytearray(stored.read_bytes())
        data[-200] ^= 0x55
        stored.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            stored_audio.audio_hash(wav)
        with self.assertRaises(ValueError):
            with stored_audio.open_wave(wav):
                pass

        plain = tone(self.root / "plain.wav")
        import soundfile
        samples, rate = soundfile.read(str(plain), dtype="int16")
        soundfile.write(str(plain.with_suffix(".flac")), samples, rate, subtype="PCM_16")
        plain.unlink()
        self.assertFalse(stored_audio.is_stored_flac(plain.with_suffix(".flac")))
        with self.assertRaises(stored_audio.StoredAudioError):
            stored_audio.audio_hash(plain)

    def test_missing_and_symlinked_audio_behave_as_before(self):
        missing = self.root / "missing.wav"
        self.assertFalse(stored_audio.audio_present(missing))
        self.assertFalse(stored_audio.audio_exists(missing))
        with stored_audio.materialize(missing) as path:
            self.assertEqual(path, missing)
        with self.assertRaises(ValueError):  # As file_hash(regular_file(missing)) did.
            stored_audio.audio_hash(missing)
        target = tone(self.root / "target.wav")
        link = self.root / "link.wav"
        link.symlink_to(target)
        self.assertTrue(stored_audio.audio_present(link))
        self.assertFalse(stored_audio.audio_exists(link))
        with self.assertRaises(ValueError):
            stored_audio.audio_hash(link)
        with self.assertRaises(ValueError):
            stored_audio.regular_audio(link)

    def test_copies_links_and_removal_follow_the_form_each_take_is_kept_in(self):
        wav = tone(self.root / "attempt-001-take.wav")
        checksum = file_hash(wav)
        stored_audio.store(wav)
        staging = self.root / "staging"
        staging.mkdir()
        copied = stored_audio.copy_audio(wav, self.root / "selection-take.wav", expected_sha256=checksum,
                                         staging=staging)
        self.assertEqual(copied.name, "selection-take.flac")
        self.assertEqual(stored_audio.audio_hash(self.root / "selection-take.wav"), checksum)
        with self.assertRaises(stored_audio.StoredAudioError):
            stored_audio.copy_audio(wav, self.root / "wrong.wav", expected_sha256="1" * 64, staging=staging)
        self.assertFalse(stored_audio.audio_present(self.root / "wrong.wav"))
        linked = stored_audio.link_audio(wav, self.root / "linked.wav")
        self.assertEqual(linked.name, "linked.flac")
        self.assertEqual(linked.stat().st_ino, wav.with_suffix(".flac").stat().st_ino)
        self.assertEqual([path.name for path in stored_audio.remove_audio(wav)], ["attempt-001-take.flac"])
        self.assertFalse(stored_audio.audio_present(wav))

    def test_ffmpeg_decodes_a_stored_take_to_the_same_samples(self):
        import shutil
        import subprocess
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("FFmpeg is required")
        wav = tone(self.root / "take.wav")
        frames = pcm_sha256(wav)
        stored_audio.store(wav)
        result = subprocess.run([ffmpeg, "-v", "error", "-i", str(stored_audio.tool_path(wav)), "-map", "0:a:0",
                                 "-c:a", "pcm_s16le", "-f", "hash", "-hash", "sha256", "-"],
                                capture_output=True, check=True)
        self.assertEqual(result.stdout.decode().strip(), "SHA256=" + frames)


if __name__ == "__main__":
    unittest.main()


class StoredHashMemoTests(unittest.TestCase):
    """A FLAC take's WAV hash is restored once and remembered by the file's stat, across processes."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve() / "render-work"
        (self.work / "clips" / ("a" * 64)).mkdir(parents=True)
        self.addCleanup(stored_audio._HASHES.clear)
        stored_audio._HASHES.clear()

    def take(self, seconds=0.5):
        wav = tone(self.work / "clips" / ("a" * 64) / "attempt-001-take.wav", seconds=seconds)
        checksum = file_hash(wav)
        stored_audio.store(wav)
        return wav, checksum

    def test_an_unchanged_flac_is_not_restored_again_even_in_a_new_process(self):
        wav, checksum = self.take()
        self.assertEqual(stored_audio.audio_hash(wav), checksum)
        stored_audio._save_stored_hashes()
        stored_audio._HASHES.clear()  # a new process
        with patch.object(stored_audio, "_restore", side_effect=AssertionError("restored again")):
            self.assertEqual(stored_audio.audio_hash(wav), checksum)

    def test_a_replaced_flac_is_restored_and_hashed_again(self):
        wav, _ = self.take()
        stored_audio.audio_hash(wav)
        stored_audio.locate(wav).unlink()
        wav, checksum = self.take(seconds=0.7)
        self.assertEqual(stored_audio.audio_hash(wav), checksum)

    def test_audio_outside_a_render_work_is_restored_every_time(self):
        loose = Path(self.work.parent) / "loose"
        loose.mkdir()
        wav = tone(loose / "voice-sample-take.wav")
        checksum = file_hash(wav)
        stored_audio.store(wav)
        self.assertEqual(stored_audio.audio_hash(wav), checksum)
        self.assertEqual(stored_audio._HASHES, {})
