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

    def test_an_er_vowel_may_be_said_without_its_r_or_with_it_written_after(self):
        for heard in ("d a n ɪ k", "d ɑː n ɪ k", "d oː n ɪ k", "d ɔ r n ɛ k", "d ɜː n ɪ k"):
            with self.subTest(heard=heard):
                self.assertTrue(said_right("D ER1 N IH0 K", heard))
        self.assertFalse(said_right("D ER1 N IH0 K", "d iː n ɪ k"))

    def test_a_nasal_or_a_breath_beside_a_vowel_is_not_a_misreading(self):
        self.assertTrue(said_right(READING, "m ə ɹ ɛ n n d ɹ ə"))
        self.assertTrue(said_right("G EH1 R IY0 AH0 N", "ɡ ɛ ɹ h ʌ m"))
        self.assertFalse(said_right(READING, "m ə ɹ ɛ k n d ɹ ə"))  # any other extra consonant still is
        self.assertTrue(said_right("TH AH1 L Z", "f aʊ l z"))

    def test_tokens_of_other_languages_are_read_as_english_phones(self):
        self.assertTrue(said_right(READING, "m ə5 ɹ ɛ n ts. ɹ ə"))  # tone digit, stray period
        self.assertTrue(said_right("V AE0 L AO1 R AH0 N", "v ə l oːɹ ɹ ə n"))  # an r-coloured vowel before its R
        self.assertTrue(said_right("T AE1 N", "th æ n"))  # aspiration
        self.assertTrue(said_right("M AE0 L AO1 R IY0 AH0 N Z", "m ə l oːɹ r i ə n z"))  # one R written twice


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

    def test_a_name_heard_in_another_languages_sounds_is_uncertain_not_misread(self):
        words = [{"word": "Marendra?", "start": 0.0, "end": 0.4}]
        report = dict(self.report(words), expected_text="Marendra?")
        with mock.patch.object(name_phonemes, "phones", return_value="x ai5 tɕh iou2".split()):
            [found] = verdicts("take.wav", report, performance("Marendra?", "Marendra", READING))
        self.assertTrue(found["ok"])
        self.assertTrue(found["uncertain"])
        with mock.patch.object(name_phonemes, "phones", return_value="m ə ɹ iː n d ɹ ə".split()):
            [found] = verdicts("take.wav", report, performance("Marendra?", "Marendra", READING))
        self.assertFalse(found["ok"])

    def test_without_word_timings_the_whole_take_is_searched(self):
        with mock.patch.object(name_phonemes, "phones", return_value="m ə ɹ ɛ n d ɹ ə n ɑː d ᵻ d".split()) as heard:
            found = verdicts("take.wav", self.report([]), performance(self.TEXT, "Marendra", READING))
        heard.assert_called_once_with("take.wav")
        self.assertTrue(found[0]["ok"])


class VerdictStoreTests(unittest.TestCase):
    def test_a_take_is_heard_once_and_its_verdicts_survive_a_new_renderer(self):
        import tempfile
        from pathlib import Path
        from epub_tts_narrator.renderer.name_phonemes import STORE_FILENAME, flush, judge
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name) / "work" / "clips" / ("c" * 64)
        folder.mkdir(parents=True)
        (folder / "take.wav").write_bytes(b"fixture")
        text = "Marendra nodded."
        record = {"audio_file": "take.wav", "audio_sha256": "a" * 64, "request_hash": "r" * 64,
                  "request": {"kind": "speech", "chunk": {"spoken_text": text, "expressive_direction": "",
                                                          "expressive_performance": performance(text, "Marendra", READING)}},
                  "report": {"passed": True, "expected_text": text, "words": []}}
        heard = "m ə ɹ ɛ n d ɹ ə".split()
        self.addCleanup(name_phonemes._STORES.clear)
        with mock.patch.object(name_phonemes, "available", return_value=True), \
                mock.patch("epub_tts_narrator.renderer.stored_audio.audio_present", return_value=True), \
                mock.patch.object(name_phonemes, "phones", return_value=heard) as listened:
            first = judge(record, folder)
            self.assertEqual(judge(record, folder), first)
            self.assertEqual(listened.call_count, 1)
            flush()
            name_phonemes._STORES.clear()  # a new renderer reads the store
            self.assertEqual(judge(record, folder), first)
            self.assertEqual(listened.call_count, 1)
            (folder.parents[1] / STORE_FILENAME).write_text("{cut short", encoding="utf-8")
            name_phonemes._STORES.clear()  # a damaged store is heard again, never trusted
            self.assertEqual(judge(record, folder), first)
            self.assertEqual(listened.call_count, 2)
        self.assertTrue(first[0]["ok"])


class ConfirmedSoundTests(unittest.TestCase):
    def test_a_take_a_person_marked_good_teaches_exactly_the_sound_it_was_heard_saying(self):
        import tempfile
        from pathlib import Path
        from epub_tts_narrator.renderer.common import write_json
        from epub_tts_narrator.renderer.take_feedback import record_verdict
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        work = Path(temporary.name) / "work"
        text = "Then Marendra nodded."
        words = [{"word": "Then", "start": 0.0, "end": 0.2}, {"word": "Marendra", "start": 0.2, "end": 0.6},
                 {"word": "nodded.", "start": 0.6, "end": 0.9}]
        report = {"passed": True, "expected_text": text, "words": words}
        control = performance(text, "Marendra", READING)

        def heard_as(sound, folder):
            folder.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(name_phonemes, "phones", return_value=sound.split()):
                return verdicts(folder / "take.wav", report, control)
        listened = work / "clips" / ("a" * 64)
        [first] = heard_as("m ə ɹ eɪ n d ɹ ə", listened)
        self.assertFalse(first["ok"])
        record = {"attempt": 1, "audio_sha256": "s" * 64,
                  "report": dict(report, passed=False, name_phonemes={"version": 4, "words": [first]})}
        write_json(listened / "attempt-001.json", record)
        record_verdict(work, "a" * 64, {"attempt": 1, "record": record}, "good")
        # The same sound in another take is now said right; any other sound is judged as before.
        [again] = heard_as("ð ɛ n m ə ɹ eɪ n d ɹ ə n ɑː", work / "clips" / ("b" * 64))
        self.assertEqual((again["ok"], again.get("confirmed")), (True, True))
        [other] = heard_as("m ə ɹ iː n d ɹ ə", work / "clips" / ("c" * 64))
        self.assertFalse(other["ok"])
        # Outside that book's render work nothing is confirmed.
        [elsewhere] = heard_as("m ə ɹ eɪ n d ɹ ə", Path(temporary.name) / "other" / "clips" / ("d" * 64))
        self.assertFalse(elsewhere["ok"])


class CommandScopeTests(unittest.TestCase):
    def test_a_command_never_leaves_the_phoneme_check_on_for_what_runs_next(self):
        from epub_tts_narrator.renderer import cli
        name_phonemes.configure(None)
        self.addCleanup(name_phonemes.configure, None)

        def command(argv, *, in_shell):
            name_phonemes.configure("/fixture/phoneme-model")
            return 0
        with mock.patch.object(cli, "_main", side_effect=command):
            self.assertEqual(cli.main(["render", "BOOK"]), 0)
        self.assertIsNone(name_phonemes.MODEL_DIRECTORY)


if __name__ == "__main__":
    unittest.main()
