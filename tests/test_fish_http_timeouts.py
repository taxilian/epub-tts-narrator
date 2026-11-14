"""A long passage gets time in proportion to its length, and a take that runs out of time is sent once more."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from epub_tts_narrator.renderer.fish_http import (
    MAX_REQUEST_SECONDS, TIMEOUT_SECONDS_PER_CHARACTER, FishHTTPEngine, FishHTTPError)
from test_fish_http import wav_bytes, wire_response


class RequestTimeLimitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        reference = self.root / "reference.wav"
        reference.write_bytes(wav_bytes())
        self.voice = {"kind": "reference", "reference_audio": str(reference),
                      "reference_text": "Reference words.", "voice_id": "one"}
        self.answers = []  # one per send: an exception to raise, or None to answer
        self.limits, self.seeds = [], []
        opener = Mock()

        def respond(request, timeout=None):
            self.limits.append(timeout)
            self.seeds.append(request.get_header("X-audiobook-seed"))
            failure = self.answers.pop(0) if self.answers else None
            if failure is not None:
                raise failure
            return wire_response(request=request)

        opener.open.side_effect = respond
        patcher = patch("epub_tts_narrator.renderer.fish_http.build_opener", return_value=opener)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = FishHTTPEngine({"url": "http://127.0.0.1:8080/v1/tts", "name": "local",
                                      "timeout_seconds": 300})
        self.takes = 0

    def synthesize(self, text):
        self.takes += 1
        return self.client.synthesize(text, self.voice, self.root / f"take-{self.takes}.wav", seed=7)

    def test_a_longer_passage_gets_a_longer_limit_and_never_more_than_an_hour(self):
        self.synthesize("Short.")
        long_text = "word " * 260
        self.synthesize(long_text)
        short_limit, long_limit = self.limits
        self.assertLess(short_limit, 320)  # the wire text carries its speaker tag
        self.assertGreater(long_limit, short_limit + 600)  # about 1300 characters
        self.assertEqual(self.client.request_timeout("x" * 100_000), MAX_REQUEST_SECONDS)
        self.assertEqual(self.client.request_timeout("x" * 100), 300 + 100 * TIMEOUT_SECONDS_PER_CHARACTER)

    def test_a_take_that_runs_out_of_time_is_sent_again_with_twice_the_limit_and_the_same_seed(self):
        self.answers = [TimeoutError("timed out")]
        result = self.synthesize("A passage that took too long in a full batch.")
        self.assertEqual(len(self.limits), 2)
        self.assertEqual(self.limits[1], 2 * self.limits[0])
        self.assertEqual(self.seeds, ["7", "7"])
        self.assertEqual(result["seed"], 7)

    def test_running_out_of_time_twice_fails_the_attempt_and_says_how_long_it_waited(self):
        self.answers = [URLError(TimeoutError("timed out")), TimeoutError("timed out")]
        with self.assertRaises(FishHTTPError) as raised:
            self.synthesize("Still too slow.")
        self.assertTrue(raised.exception.timed_out)
        self.assertEqual(len(self.limits), 2)
        self.assertIn(f"timed out after {self.limits[1]:g}s", str(raised.exception))

    def test_a_refused_connection_is_not_a_timeout_and_is_not_sent_again(self):
        self.answers = [URLError(ConnectionRefusedError(61, "Connection refused"))]
        with self.assertRaises(FishHTTPError) as raised:
            self.synthesize("Nobody is listening.")
        self.assertFalse(raised.exception.timed_out)
        self.assertEqual(len(self.limits), 1)


if __name__ == "__main__":
    unittest.main()
