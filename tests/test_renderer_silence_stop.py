"""Ending a Drama 3 take whose audio has gone silent, from decoded frames."""

import unittest

import numpy as np

from epub_tts_narrator.renderer.silence_stop import (
    CONTEXT_FRAMES, SILENT_CHECK_FRAMES, STOP_SECONDS, SilenceWatch, stop_seconds)

RATE = 44100
HOP = 2048          # samples per codec frame, as in Drama 3's codec
FPS = RATE / HOP
LOUD, QUIET = 1, 0


class FakeCodec:
    """A frame whose first code is LOUD decodes to a tone, QUIET to digital silence.

    The first sample of every decoded window is loud, like a codec warming up
    without left context, so the watch must discard its context frames.
    """

    def __init__(self):
        self.calls = []

    def decode(self, frames):
        self.calls.append(len(frames))
        audio = np.concatenate([np.full(HOP, 0.3 if codes[0] == LOUD else 0.0) for codes in frames])
        audio[:64] = 0.3
        return audio.astype(np.float32)


def run(watch, pattern):
    """Feed frames until the watch stops; the frame count at which it stopped, or None."""
    for index, value in enumerate(pattern, 1):
        if watch.observe([value, 0, 0]):
            return index
    return None


def seconds(value, count):
    return [value] * round(count * FPS)


class SilenceWatchTests(unittest.TestCase):
    def watch(self, text="Hello there."):
        codec = FakeCodec()
        return SilenceWatch(text, codec.decode, sample_rate=RATE), codec

    def test_speech_that_goes_silent_is_stopped_within_a_check_of_the_limit(self):
        watch, _codec = self.watch()
        stopped = run(watch, seconds(LOUD, 2) + seconds(QUIET, 30))
        self.assertIsNotNone(stopped)
        silence = stopped / FPS - 2
        self.assertGreaterEqual(silence, STOP_SECONDS)
        # Silence is first noticed by a speech-rate check, then followed closely.
        self.assertLess(silence, STOP_SECONDS + 0.8)
        self.assertTrue(watch.stopped)

    def test_silence_that_began_between_checks_is_counted_from_its_start(self):
        # Silence starting just after a speech-rate check is first seen almost
        # two seconds later; the frames before the newest half second are then
        # decoded so the stop still comes about STOP_SECONDS after the speech.
        for lead in (2.1, 2.6, 3.0):
            with self.subTest(lead=lead):
                watch, _codec = self.watch()
                stopped = run(watch, seconds(LOUD, lead) + seconds(QUIET, 30))
                self.assertLess(stopped / FPS - lead, STOP_SECONDS + 0.8)

    def test_pauses_shorter_than_the_limit_never_stop_a_take(self):
        watch, _codec = self.watch()
        pattern = (seconds(LOUD, 1) + seconds(QUIET, STOP_SECONDS - 0.3)) * 6 + seconds(LOUD, 1)
        self.assertIsNone(run(watch, pattern))

    def test_requested_pauses_extend_the_allowed_silence(self):
        text = "Wait. [pause for 2.5 seconds] Now."
        self.assertEqual(stop_seconds(text), STOP_SECONDS + 2.5)
        self.assertEqual(stop_seconds("[pause for 1 seconds] a [pause for 0.5 seconds] b"), STOP_SECONDS + 1.5)
        self.assertEqual(stop_seconds("[Sheepish and slightly deflated] Fine."), STOP_SECONDS)
        watch, _codec = self.watch(text)
        self.assertIsNone(run(watch, seconds(LOUD, 1) + seconds(QUIET, 5) + seconds(LOUD, 1)))
        watch, _codec = self.watch(text)
        self.assertIsNotNone(run(watch, seconds(LOUD, 1) + seconds(QUIET, 8)))

    def test_continuous_speech_is_never_stopped(self):
        watch, _codec = self.watch()
        self.assertIsNone(run(watch, seconds(LOUD, 90)))
        self.assertEqual(watch.silent_seconds, 0)

    def test_speech_is_decoded_every_two_seconds_and_silence_every_half_second(self):
        watch, codec = self.watch()
        run(watch, seconds(LOUD, 20))
        self.assertIn(len(codec.calls), range(9, 12))
        # While speaking, each check decodes only the newest half second plus context.
        self.assertLessEqual(max(codec.calls), SILENT_CHECK_FRAMES + CONTEXT_FRAMES)
        calls = len(codec.calls)
        run(watch, seconds(QUIET, 2))
        self.assertGreaterEqual(len(codec.calls) - calls, 3)

    def test_a_decoded_warm_up_click_does_not_hide_silence(self):
        # Every decode starts with a loud warm-up sample (FakeCodec); only the
        # context frames carry it, so continuous silence still accumulates.
        watch, _codec = self.watch()
        self.assertIsNotNone(run(watch, seconds(LOUD, 1) + seconds(QUIET, 10)))

    def test_invalid_settings_are_rejected(self):
        codec = FakeCodec()
        for options in ({"sample_rate": 0}, {"sample_rate": RATE, "check_frames": 0},
                        {"sample_rate": RATE, "silent_check_frames": 0},
                        {"sample_rate": RATE, "context_frames": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SilenceWatch("Hi.", codec.decode, **options)
        with self.assertRaises(TypeError):
            stop_seconds(None)


if __name__ == "__main__":
    unittest.main()
