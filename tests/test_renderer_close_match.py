"""Close transcripts of chapter speech are accepted; wrong or missing speech is not.

The cases follow takes a listener judged: most rejected takes were accented
speech or invented names the recognizer misheard, while the takes they
judged bad lost words, replaced a short line or spoke another passage.
"""

import unittest

from epub_tts_narrator.renderer.close_match import close_transcript
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import QUALITY_DEFAULTS, validate_audio, validate_policy
import test_renderer_reference_consensus as consensus


def close(expected, heard, fraction=0.25):
    return close_transcript(expected, heard, fraction)


class CloseTranscriptTests(unittest.TestCase):
    def test_accent_and_name_mishearings_are_close(self):
        for expected, heard in (
                ("I remember some of those sleds. Faugh!", "I remember some of those sleds. Faw!"),
                ("You may come out of this affair better than you deserve to, Killashandra Ree,",
                 "You may come out of this affair better than you deserve to, Kilachandrari."),
                ("Killashandra waited by the Guild door.", "Kill a Chandra waited by the Gil door."),
                ("She proceeded to the sled and the Milekey Range.", "She proceeded to the sled and the Mile Key Range."),
                ("All but the very small detector units have gravity, ma'am.",
                 "All but the very small detector units have gravity, man."),
                ("I'm sure we'll find the cutters this way, Trag.", "Oh, I'm sure we'll find the cutters this way, Tragg."),
                ("We were always a crack in the flaw.", "We were always a crack in the floor."),
                ("Send an e-mail to the newsgroup.", "Send an email to the news group.")):
            with self.subTest(heard=heard):
                result = close(expected, heard)
                self.assertIsNotNone(result)
                self.assertEqual(result["method"], "close_match_v1")
                self.assertLessEqual(result["longest_difference_run"], 2)

    def test_a_few_scattered_differences_are_allowed(self):
        expected = "The rumor that crystal singers are immortal is exaggerated, but their longevity is increased."
        self.assertIsNotNone(close(expected, expected.replace("rumor", "roomer").replace("but", "bud")))
        # Three wrong words in a row is a replaced phrase, not mishearing.
        self.assertIsNone(close(expected, expected.replace("crystal singers are", "she brought them")))
        # More differences than the passage allows.
        # More unrelated words than a quarter of the passage, even when scattered.
        many = expected.replace("rumor", "table").replace("immortal", "purple").replace(
            "exaggerated", "window").replace("longevity", "carpet").replace("increased", "tomato")
        self.assertIsNone(close(expected, many))

    def test_missing_added_and_protected_words_are_never_close(self):
        for expected, heard in (
                ("I have never intended to invade your privacy. I enjoy you. Now", "I have never intended to invade your privacy. I enjoy you."),
                ("Special features on various authors and SF/F worlds.", "Special features on various authors and SF worlds."),
                ("The ship woman read seventeen minds.", "The ship woman read seventy minds."),
                ("Darin did not close the door.", "Darin did now close the door."),
                ("Darin closed the door.", "Darin never closed the door."),
                ("Darin found coins inside the box.", "Darin found fifty coins inside the box."),
                ("Carrik peered at the label.",
                 "He glanced at her skeptically just as a chilled bottle slid from the service panel.")):
            with self.subTest(heard=heard):
                self.assertIsNone(close(expected, heard))

    def test_short_lines_are_left_to_the_strict_checks(self):
        self.assertIsNone(close('"No."', "Here."))
        self.assertIsNone(close('"Rimbol!"', "I'm ball!"))
        self.assertIsNotNone(close('"High, Lanzecki,"', "Hi, Lanzeki."))

    def test_speech_from_another_passage_is_never_close(self):
        expected = "Is that how you do your appearing act? she asked, still shaken."
        echo = ("Then she noticed the fine white lines across his bare arm and chest, parallel in some places. "
                + expected)
        self.assertIsNone(close(expected, echo))

    def test_zero_turns_it_off(self):
        self.assertIsNone(close("We were always a crack in the flaw.", "We were always a crack in the floor.", 0))


class CloseMatchVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio

    EXPECTED = "Tomorrow, Trag will instruct you in installation procedures for the crystal cutters."
    HEARD = "Tomorrow, Tragg will instruct you in installation procedures for the crystal cutter."

    def test_chapter_speech_accepts_a_close_transcript_without_a_judge(self):
        policy = {"max_close_mismatch_fraction": 0.25}
        report = validate_audio(self.audio, self.EXPECTED, consensus.PrimaryVerifier(self.HEARD), policy)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["text_confirmation_method"], "close_match")
        self.assertEqual(report["verified_transcript"], self.HEARD)
        self.assertEqual(report["close_match"]["expected_words"], 12)

    def test_the_strict_default_still_requires_the_words(self):
        strict = validate_audio(self.audio, self.EXPECTED, consensus.PrimaryVerifier(self.HEARD), None)
        self.assertEqual(strict["reasons"], ["text_mismatch"])
        for purpose in ("speech", "reference"):
            report = validate_audio(self.audio, self.EXPECTED, consensus.PrimaryVerifier(self.HEARD), None,
                                    purpose=purpose)
            self.assertEqual(report["reasons"], ["text_mismatch"], purpose)

    def test_a_voice_sample_with_misheard_names_passes_and_keeps_its_script(self):
        # A recognizer that does not know a book's names says nothing about the voice being cloned.
        expected = ("We reached Erutown by nightfall, and Lars said the gates would be shut before we could "
                    "find an inn, so we camped by the river and waited for the light.")
        heard = ("We reached there a town by nightfall, and laws said the gates would be shut before we could "
                 "find an inn, so we camped by the river and waited for the light.")
        report = validate_audio(self.audio, expected, consensus.PrimaryVerifier(heard),
                                {"max_close_mismatch_fraction": 0.25}, purpose="reference")
        self.assertTrue(report["passed"], report["reasons"])
        self.assertEqual(report["text_confirmation_method"], "close_match")
        # The voice said the script: the script, not the misspelled transcript, goes with the recording.
        self.assertEqual((report["reference_text"], report["reference_text_adjusted"]), (expected, False))
        # A word never said still fails, as for chapter speech.
        missing = validate_audio(self.audio, expected, consensus.PrimaryVerifier(heard.replace(" by the river", "")),
                                 {"max_close_mismatch_fraction": 0.25}, purpose="reference")
        self.assertFalse(missing["passed"])

    def test_the_renderer_configuration_turns_it_on(self):
        self.assertEqual(QUALITY_DEFAULTS["max_close_mismatch_fraction"], 0.0)
        self.assertEqual(validate_policy(DEFAULTS["quality"])["max_close_mismatch_fraction"], 0.25)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            validate_policy({"max_close_mismatch_fraction": 2})


if __name__ == "__main__":
    unittest.main()
