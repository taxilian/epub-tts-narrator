"""A chosen reading is heard by its phonemes: said as chosen, or misread."""

from pathlib import Path
import unittest
from unittest import mock

from epub_tts_narrator.renderer import name_phonemes
from epub_tts_narrator.renderer.name_phonemes import MISREAD_COST, cost, verdicts
from test_renderer_name_shape import performance

READING = "M AH0 R EH1 N D R AH0"  # muh-REN-druh


def hearing(heard):
    """Stand in for the recognizer hearing ``heard`` in any window, evenly timed across it when asked."""
    sounds = heard.split() if isinstance(heard, str) else list(heard)

    def phones(path, start=None, end=None, *, timed=False):
        if not timed:
            return list(sounds)
        start, end = (0.0, 1.0) if start is None else (start, end)
        step = (end - start) / max(1, len(sounds))
        return [(phone, round(start + (index + 0.5) * step, 3)) for index, phone in enumerate(sounds)]
    return mock.patch.object(name_phonemes, "phones", side_effect=phones)


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

    def test_a_secondary_stress_may_reduce_but_not_cross_the_mouth(self):
        reading = "S EH2 N EH1 D R AH0"
        for heard in ("s ɪ n ɛ d ɹ ə", "s ə n ɛ d ɹ ə", "s iː n ɛ d ɹ ə", "s ɛ n ɛ d ɹ ə"):
            with self.subTest(heard=heard):
                self.assertTrue(said_right(reading, heard))
        for heard in ("s uː n ɛ d ɹ ə", "s ɚ n ɛ d ɹ ə", "s ʌ n ɛ d ɹ ɚ"):
            with self.subTest(heard=heard):
                self.assertFalse(said_right(reading, heard))
        self.assertTrue(said_right("M AE0 L AO1 R IY0 AH0 N", "m ʌ l ɔ ɹ i ə n"))  # unstressed stays free

    def test_a_stop_between_a_nasal_and_a_fricative_is_not_a_misreading(self):
        reading = "L AE2 N Z EH1 K IY0"
        self.assertTrue(said_right(reading, "aʊ t l æ n d z ɛ k i"))  # "lan-dzecki", heard right by a listener
        self.assertTrue(said_right("P R IH1 N S", "p ɹ ɪ n t s"))  # "prince" as "prints"
        self.assertTrue(said_right("K AH1 M F ER0 T", "k ʌ m p f ɚ t"))
        self.assertFalse(said_right(reading, "l æ n k z ɛ k i"))  # a stop made elsewhere still counts
        self.assertFalse(said_right(reading, "l æ n z d ɛ k i"))  # and so does one after the fricative

    def test_a_stressed_eh_may_be_written_as_the_open_front_a(self):
        reading = "L AE2 N Z EH1 K IY0"
        for heard in ("l a n z a k i s", "l a e n z a k i s", "m a n z a k iː s"):  # marked good by a listener
            with self.subTest(heard=heard):
                self.assertTrue(said_right(reading + " Z", heard))
        self.assertFalse(said_right(reading, "l a n z ɑː k i"))  # "ah" is still another vowel
        self.assertFalse(said_right(reading, "l a n s k iː"))  # and a dropped stressed syllable a misreading

    def test_tokens_of_other_languages_are_read_as_english_phones(self):
        self.assertTrue(said_right(READING, "m ə5 ɹ ɛ n ts. ɹ ə"))  # tone digit, stray period
        self.assertTrue(said_right("V AE0 L AO1 R AH0 N", "v ə l oːɹ ɹ ə n"))  # an r-coloured vowel before its R
        self.assertTrue(said_right("T AE1 N", "th æ n"))  # aspiration
        self.assertTrue(said_right("M AE0 L AO1 R IY0 AH0 N Z", "m ə l oːɹ r i ə n z"))  # one R written twice


class VerdictTests(unittest.TestCase):
    TEXT = "Marendra nodded."

    def report(self, words):
        return {"passed": True, "expected_text": self.TEXT, "transcript": " Marendra nodded.", "words": words}

    @staticmethod
    def listened(heard):
        """A ``phones`` stand-in: ``heard[(start, end)]`` is ``[(phone, seconds), ...]``."""
        def phones(path, start=None, end=None, *, timed=False):
            found = heard[(start, end)]
            return found if timed else [phone for phone, _ in found]
        return mock.patch.object(name_phonemes, "phones", side_effect=phones)

    @staticmethod
    def timed(text, start, step=0.06):
        return [(phone, round(start + index * step, 3)) for index, phone in enumerate(text.split())]

    def test_a_name_clipped_by_its_word_timings_is_heard_again_with_its_neighbours(self):
        # Said from 0.0 to 0.5 s, its timings end it at 0.3 s.
        words = [{"word": "Marendra", "start": 0.0, "end": 0.3}, {"word": "nodded.", "start": 0.6, "end": 0.9}]
        heard = {(0.0, 0.3): [("n", 0.2)], (0.0, 0.9): self.timed("m ə ɹ ɛ n d ɹ ə n ɑː d ᵻ d", 0.0)}
        with self.listened(heard):
            found = verdicts("take.wav", self.report(words), performance(self.TEXT, "Marendra", READING))
        self.assertEqual([(item["text"], item["ok"]) for item in found], [("Marendra", True)])

    def test_a_misread_name_stays_misread_with_its_neighbours(self):
        words = [{"word": "Marendra", "start": 0.0, "end": 0.3}, {"word": "nodded.", "start": 0.6, "end": 0.9}]
        heard = {(0.0, 0.3): self.timed("m ə ɹ iː n d ɹ ə", 0.0, 0.04),
                 (0.0, 0.9): self.timed("m ə ɹ iː n d ɹ ə n ɑː d ᵻ d", 0.0)}
        with self.listened(heard):
            found = verdicts("take.wav", self.report(words), performance(self.TEXT, "Marendra", READING))
        self.assertEqual([(item["ok"], item["heard"]) for item in found], [(False, "m ə ɹ iː n d ɹ ə")])

    def test_neighbouring_words_never_stand_in_for_the_name(self):
        # "the gray city of Riva, he sent" said as RAY-vuh: in the wider
        # window R IY V AH matched "-ty of" (ɾ i ʌ v), half of it before the
        # name; only the name's own sounds may decide (times as heard).
        text = "The gray city of Riva, he sent for them."
        words = [{"word": "city", "start": 7.0, "end": 7.34}, {"word": "of", "start": 7.34, "end": 7.54},
                 {"word": "Riva,", "start": 7.54, "end": 7.84}, {"word": "he", "start": 8.2, "end": 8.52}]
        report = {"passed": True, "expected_text": text, "words": words}
        heard = {(7.54, 7.84): [("v", 7.52), ("ɹ", 7.585), ("eɪ", 7.646), ("v", 7.788), ("ə", 7.83)],
                 (7.34, 8.52): [("ɪ", 7.261), ("ɾ", 7.321), ("i", 7.342), ("ʌ", 7.423), ("v", 7.504),
                                ("ɹ", 7.585), ("eɪ", 7.646), ("v", 7.788), ("ɐ", 7.829), ("h", 8.457),
                                ("iː", 8.498), ("s", 8.579)]}
        with self.listened(heard):
            [found] = verdicts("take.wav", report, performance(text, "Riva", "R IY1 V AH0"))
        self.assertFalse(found["ok"])
        self.assertEqual(found["heard"], "v ɹ eɪ v ə")

    def test_a_name_before_its_late_word_timing_is_still_heard(self):
        # Belgarath said 12.21-12.82 s, timed 12.64-13.12: the name sits in
        # the gap before its timing, where no other word was timed.
        text = "As usual. Belgarath, dressed in his robe."
        words = [{"word": "usual.", "start": 9.32, "end": 9.7}, {"word": "Belgarath,", "start": 12.64, "end": 13.12},
                 {"word": "dressed", "start": 13.12, "end": 13.6}]
        report = {"passed": True, "expected_text": text, "words": words}
        heard = {(12.64, 13.12): [("ɹ", 12.535), ("ɛ", 12.6), ("θ", 12.816)],
                 (9.32, 13.6): [("j", 9.361), ("uː", 9.401), ("ʒ", 9.521), ("əl", 9.642), ("b", 12.213),
                                ("ɛ", 12.274), ("l", 12.354), ("ɡ", 12.434), ("ɚ", 12.474), ("ɹ", 12.535),
                                ("æ", 12.595), ("θ", 12.816), ("d", 13.399), ("ɹ", 13.439), ("ɛ", 13.499)]}
        with self.listened(heard):
            [found] = verdicts("take.wav", report, performance(text, "Belgarath", "B EH1 L G AH0 R AE2 TH"))
        self.assertTrue(found["ok"])

    def test_another_words_sounds_never_pass_for_the_name(self):
        # "the army TOH Urgas": T AA AO R matched "ar-" of "army" (ɑː ɹ).
        text = "The bulk of the army Taur Urgas had brought."
        words = [{"word": "army", "start": 15.44, "end": 15.76}, {"word": "Taur", "start": 15.76, "end": 16.22},
                 {"word": "Urgas", "start": 16.22, "end": 16.68}]
        report = {"passed": True, "expected_text": text, "words": words}
        heard = {(15.76, 16.22): [("m", 15.685), ("i", 15.725), ("t", 15.888), ("oʊ", 15.989)],
                 (15.44, 16.68): [("ð", 15.34), ("ɪ", 15.381), ("ɑːɹ", 15.502), ("m", 15.685), ("i", 15.725),
                                  ("t", 15.888), ("oʊ", 15.989), ("ɜː", 16.374), ("ɡ", 16.536), ("ʌ", 16.618)]}
        with self.listened(heard):
            [found] = verdicts("take.wav", report, performance(text, "Taur", "T AA1 AO0 R"))
        self.assertFalse(found["ok"])

    def test_a_name_heard_in_another_languages_sounds_is_uncertain_not_misread(self):
        words = [{"word": "Marendra?", "start": 0.0, "end": 0.4}]
        report = dict(self.report(words), expected_text="Marendra?")
        with hearing("x ai5 tɕh iou2"):
            [found] = verdicts("take.wav", report, performance("Marendra?", "Marendra", READING))
        self.assertTrue(found["ok"])
        self.assertTrue(found["uncertain"])
        with hearing("m ə ɹ iː n d ɹ ə"):
            [found] = verdicts("take.wav", report, performance("Marendra?", "Marendra", READING))
        self.assertFalse(found["ok"])

    def test_without_word_timings_the_whole_take_is_searched(self):
        with hearing("m ə ɹ ɛ n d ɹ ə n ɑː d ᵻ d") as heard:
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
                hearing(heard) as listened:
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
            with hearing(sound):
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


class RecheckTests(unittest.TestCase):
    """A take the phoneme check rejected under earlier rules is heard again, not remade."""
    TEXT = "Lanzecki’s office was quiet."

    def setUp(self):
        import tempfile
        from pathlib import Path
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "render-work" / "clips" / ("a" * 64)
        self.directory.mkdir(parents=True)
        self.generated = []
        self.request = {"chunk": {"spoken_text": self.TEXT, "expressive_direction": "",
                                  "expressive_performance": performance(self.TEXT, "Lanzecki’s", "L AE2 N Z EH1 K IY0 Z")}}

    def take(self, verdict_version, ok):
        import test_renderer_reference_consensus as consensus
        from test_renderer_stall_recovery import config, write_take
        from epub_tts_narrator.renderer.workflow import verified_take

        def generate(path, seed):
            self.generated.append(path)
            write_take(path, 0.1)
            return {"seed": seed}

        def judged(report, path, performance):
            if report.get("passed") is not True:
                return report
            words = [{"text": "Lanzecki’s", "key": "L AE2 N Z EH1 K IY0", "heard": "l a n z a k i s",
                      "cost": 0.0 if ok else 1.0, "ok": ok}]
            report = {**report, "name_phonemes": {"version": verdict_version, "words": words}}
            return report if ok else {**report, "passed": False, "reasons": ["pronunciation_mismatch"]}
        verifier = consensus.PrimaryVerifier(self.TEXT)
        with mock.patch("epub_tts_narrator.renderer.workflow.normalize_clip",
                        side_effect=lambda raw, wav, **kw: Path(wav).write_bytes(Path(raw).read_bytes())), \
                mock.patch.object(name_phonemes, "available", return_value=True), \
                mock.patch.object(name_phonemes, "apply", side_effect=judged):
            return verified_take(self.TEXT, generate, self.directory, self.request, verifier, config(),
                                 purpose="speech", notify=lambda _: None)

    def test_a_take_rejected_under_earlier_rules_is_heard_again_without_a_new_take(self):
        from epub_tts_narrator.renderer.common import read_json
        from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, _verification_failure_detail
        with self.assertRaises(SpeechVerificationFailed):
            self.take(name_phonemes.VERSION - 1, ok=False)
        self.assertEqual(len(self.generated), 2)
        rejected = read_json(self.directory / "attempt-001.json")["report"]
        # The failure names what the phoneme check heard, not the word transcript that passed.
        self.assertEqual(_verification_failure_detail(rejected),
                         "Lanzecki’s sounded like \"l a n z a k i s\", not L AE2 N Z EH1 K IY0")
        _, accepted = self.take(name_phonemes.VERSION, ok=True)
        self.assertEqual(len(self.generated), 2)  # nothing new was synthesized
        self.assertEqual(accepted["attempt"], 1)
        self.assertEqual(read_json(self.directory / "attempt-001.json")["policy_recheck"]["previous_report"]["reasons"],
                         ["pronunciation_mismatch"])

    def test_a_take_rejected_under_todays_rules_is_not_heard_again(self):
        from epub_tts_narrator.renderer.common import read_json
        from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed
        with self.assertRaises(SpeechVerificationFailed):
            self.take(name_phonemes.VERSION, ok=False)
        with self.assertRaises(SpeechVerificationFailed):
            self.take(name_phonemes.VERSION, ok=False)
        self.assertEqual(len(self.generated), 2)
        self.assertNotIn("policy_recheck", read_json(self.directory / "attempt-001.json"))


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
