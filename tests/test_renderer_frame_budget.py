"""Per-take Drama 3 frame budget from the length of the spoken text."""

import unittest

from epub_tts_narrator.renderer.frame_budget import FRAMES_PER_SECOND, frame_budget


def seconds(text, limit=2048):
    return frame_budget(text, limit) / FRAMES_PER_SECOND


class FrameBudgetTests(unittest.TestCase):
    def test_short_lines_get_far_less_than_the_configured_limit(self):
        # A one-word line used to be allowed the whole 95-second limit.
        self.assertLess(seconds("Yes."), 15)
        self.assertLess(frame_budget("Yes.", 2048), 2048 // 5)

    def test_long_passages_keep_the_configured_limit(self):
        passage = "The ship drifted past the outer moons. " * 20
        self.assertEqual(frame_budget(passage, 2048), 2048)
        self.assertEqual(frame_budget(passage, 300), 300)

    def test_budget_never_exceeds_a_smaller_configured_or_context_limit(self):
        self.assertEqual(frame_budget("Yes.", 1), 1)
        self.assertEqual(frame_budget("Yes.", 50), 50)

    def test_slowest_measured_readings_fit_with_room_to_spare(self):
        # The slowest accepted take of a finished book read 0.48 seconds per
        # character; pauses, whispers and drawn-out words must still fit.
        for length in (1, 5, 20, 80, 160):
            text = "a" * length
            with self.subTest(length=length):
                self.assertGreater(seconds(text), 0.48 * length + 5)

    def test_unspoken_tags_and_directions_add_no_time(self):
        plain = frame_budget("Hello there.", 2048)
        self.assertEqual(frame_budget("<|speaker:0|> Hello there.", 2048), plain)
        self.assertEqual(frame_budget("[whispering, slowly, with long pauses] Hello there.", 2048), plain)
        self.assertEqual(frame_budget("<|speaker:0|> [sighs] Hello <|speaker:1|> there.", 2048), plain)

    def test_digits_are_read_as_words_not_letters(self):
        # "1923" is spoken as several words, not four letters.
        self.assertGreater(frame_budget("Year 1923.", 2048), frame_budget("Year abcd.", 2048))

    def test_punctuation_and_spacing_add_no_time(self):
        self.assertEqual(frame_budget("Wait... what?!", 2048), frame_budget("Wait what", 2048))

    def test_invalid_limits_are_rejected(self):
        for limit in (0, -1, 2.5, None, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                frame_budget("Yes.", limit)


if __name__ == "__main__":
    unittest.main()
