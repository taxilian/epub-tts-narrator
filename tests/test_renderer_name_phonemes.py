"""A chosen reading is heard by its phonemes: said as chosen, or misread."""

import unittest
from unittest import mock

from epub_tts_narrator.renderer import name_phonemes
from epub_tts_narrator.renderer.name_phonemes import MISREAD_COST, cost, verdicts
from test_renderer_name_shape import performance

READING = "M AH0 R EH1 N D R AH0"  # muh-REN-druh


def said_right(reading, heard):
    return cost(reading, heard.split()) < MISREAD_COST


class ReadingCostTests(unittest.TestCase):
    def test_the_reading_as_chosen_is_said_right_whatever_surrounds_it(self):
        self.assertTrue(said_right(READING, "m ə ɹ ɛ n d ɹ ə"))
        self.assertTrue(said_right(READING, "ð ə m ə ɹ ɛ n d ɹ ə n ɑː d ᵻ d"))

    def test_the_stressed_vowel_must_keep_its_kind(self):
        self.assertFalse(said_right(READING, "m ə ɹ iː n d ɹ ə"))  # ree
        self.assertFalse(said_right(READING, "m ə ɹ eɪ n d ɹ ə"))  # ray
        self.assertTrue(said_right(READING, "m ɪ ɹ ɛ n d ɹ ʌ"))  # unstressed vowels may be any vowel

    def test_consonants_moved_across_a_vowel_are_a_misreading(self):
        self.assertFalse(said_right(READING, "m ə ɹ n ɛ d ɹ ə"))
        self.assertFalse(said_right(READING, "m ə ɹ ɛ d ɹ ə"))  # a consonant missing inside the name

    def test_regular_consonant_variation_is_not_a_misreading(self):
        self.assertTrue(said_right(READING, "m ə ɹ ɛ n t ɹ ə"))  # devoiced
        self.assertTrue(said_right(READING, "m ə ɹ ɛ n d ə"))  # an R swallowed in a cluster
        self.assertTrue(said_right(READING, "ə ɹ ɛ n d ɹ ə"))  # the first sound clipped by word timings

    def test_vowels_that_merge_before_r_are_one_vowel(self):
        self.assertTrue(said_right("V EH1 R AH0 N", "v æ ɹ ə n"))
        self.assertTrue(said_right("K AO1 R IH0 N", "k oʊ ɹ ɪ n"))
        self.assertFalse(said_right("V EH1 R AH0 N", "v iː ɹ ə n"))

    def test_a_syllable_final_l_may_be_swallowed_and_colour_its_vowel(self):
        self.assertTrue(said_right("T AH1 L V AA0 R", "t uː v ɑː ɹ"))
        self.assertTrue(said_right("T AH1 L V AA0 R", "t əl v ɑː ɹ"))

    def test_two_neighbouring_vowels_may_glide_into_one(self):
        self.assertTrue(said_right("T AA1 AO0 R", "t aʊ ɹ"))
        self.assertFalse(said_right("T AA1 AO0 R", "t eɪ ɹ"))

    def test_tokens_of_other_languages_are_read_as_english_phones(self):
        self.assertTrue(said_right(READING, "m ə5 ɹ ɛ n ts. ɹ ə"))  # tone digit, stray period
        self.assertTrue(said_right("V AE0 L AO1 R AH0 N", "v ə l oːɹ ɹ ə n"))  # an r-coloured vowel before its R
        self.assertTrue(said_right("T AE1 N", "th æ n"))  # aspiration


class VerdictTests(unittest.TestCase):
    TEXT = "Marendra nodded."

    def report(self, words):
        return {"passed": True, "expected_text": self.TEXT, "transcript": " Marendra nodded.", "words": words}

    def test_a_name_clipped_by_its_word_timings_is_heard_again_with_its_neighbours(self):
        words = [{"word": "Marendra", "start": 0.0, "end": 0.3}, {"word": "nodded.", "start": 0.6, "end": 0.9}]
        heard = {(0.0, 0.3): ["n"], (0.0, 0.9): "m ə ɹ ɛ n d ɹ ə n ɑː d ᵻ d".split()}
        with mock.patch.object(name_phonemes, "phones", side_effect=lambda path, start=None, end=None: heard[(start, end)]):
            found = verdicts("take.wav", self.report(words), performance(self.TEXT, "Marendra", READING))
        self.assertEqual([(item["text"], item["ok"]) for item in found], [("Marendra", True)])

    def test_a_misread_name_stays_misread_with_its_neighbours(self):
        words = [{"word": "Marendra", "start": 0.0, "end": 0.3}, {"word": "nodded.", "start": 0.6, "end": 0.9}]
        heard = {(0.0, 0.3): "m ə ɹ iː n d ɹ ə".split(), (0.0, 0.9): "m ə ɹ iː n d ɹ ə n ɑː d ᵻ d".split()}
        with mock.patch.object(name_phonemes, "phones", side_effect=lambda path, start=None, end=None: heard[(start, end)]):
            found = verdicts("take.wav", self.report(words), performance(self.TEXT, "Marendra", READING))
        self.assertEqual([(item["ok"], item["heard"]) for item in found], [(False, "m ə ɹ iː n d ɹ ə")])

    def test_without_word_timings_the_whole_take_is_searched(self):
        with mock.patch.object(name_phonemes, "phones", return_value="m ə ɹ ɛ n d ɹ ə n ɑː d ᵻ d".split()) as heard:
            found = verdicts("take.wav", self.report([]), performance(self.TEXT, "Marendra", READING))
        heard.assert_called_once_with("take.wav")
        self.assertTrue(found[0]["ok"])


if __name__ == "__main__":
    unittest.main()
