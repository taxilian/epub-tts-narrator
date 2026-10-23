import unittest

from epub_tts_narrator.renderer.quality import word_spacing_equivalence
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
import test_renderer_reference_consensus as fixtures
from test_renderer_jev_verification import Judge


PAIRS = (
    ("The ship woman's voice.", "The shipwoman's voice."),
    ("The ship woman’s voice.", "THE SHIPWOMANS VOICE!"),
    ("The sun, flower grows.", "The sunflower grows."),
    ("Please re—sign.", "Please resign."),
    ("A part of it.", "Apart of it."),
    ("Some where over there.", "Somewhereoverthere."),
    ("Over taking.", "Overtak ing."),
    ("Sunflower.", "S u n f l o w e r."),
    ("I can't go.", "I cant go."),
)


class WrittenEquivalenceTests(unittest.TestCase):
    def test_boundary_placement_does_not_matter_but_every_letter_still_does(self):
        expected, letters = "The ship woman's voice.", "Theshipwomansvoice"
        for separator in (" ", "\t", "\n", "-", "—", ",", ".", "'", "’", "?!"):
            with self.subTest(separator=separator):
                actual = separator.join(letters)
                self.assertIsNotNone(word_spacing_equivalence(expected, actual))
                self.assertTrue(speech_coverage(expected, actual)["passed"])
        for position in range(len(letters)):
            with self.subTest(missing_position=position):
                self.assertIsNone(word_spacing_equivalence(expected, letters[:position] + letters[position + 1:]))

    def test_complete_written_content_ignores_case_spacing_and_punctuation(self):
        for first, second in PAIRS:
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    comparison = word_spacing_equivalence(expected, actual)
                    self.assertIsNotNone(comparison)
                    self.assertTrue(comparison["alignment"])
                    self.assertTrue(speech_coverage(expected, actual)["passed"])

    def test_changed_missing_repeated_and_reordered_content_still_fails(self):
        for expected, actual in (
            ("The ship woman's voice.", "The shipwoman's voices."),
            ("The ship woman's voice.", "The shipwoman's."),
            ("The ship woman's voice.", "The shipwoman's voice voice."),
            ("The ship woman's voice.", "The voice shipwoman's."),
            ("She did not hear the ship woman.", "She did hear the shipwoman."),
            ("12 34", "1234"), ("1.23 4", "1.234"),
            ("-7", "7"), ("$5", "5"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertIsNone(word_spacing_equivalence(expected, actual))


class WrittenEquivalenceVerificationTests(unittest.TestCase):
    setUp = fixtures.ReferenceConsensusTests.setUp
    write_audio = fixtures.ReferenceConsensusTests.write_audio
    check = fixtures.ReferenceConsensusTests.check

    def test_complete_equality_accepts_both_purposes_without_judges_or_secondary_asr(self):
        for expected, actual in PAIRS:
            for purpose in ("speech", "reference"):
                with self.subTest(expected=expected, purpose=purpose):
                    primary = fixtures.IndependentVerifier(actual, "Unused secondary transcript.")
                    primary.secondary = RuntimeError("Must not decode twice")
                    judge = Judge("unavailable")
                    report = self.check(ReferenceTextJudgingVerifier(primary, judge),
                                        expected=expected, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["text_confirmation_method"], "word_spacing")
                    self.assertEqual(report["transcript"], actual)
                    self.assertEqual(judge.calls, [])
                    self.assertEqual(primary.calls, [(purpose, self.audio)])

    def test_normalized_equality_never_overrides_failed_audio(self):
        self.write_audio(amplitude=0)
        report = self.check(fixtures.PrimaryVerifier("The shipwomans voice."),
                            expected="The ship woman's voice.", purpose="speech")
        self.assertFalse(report["passed"])
        self.assertIn("silent_audio", report["reasons"])


if __name__ == "__main__":
    unittest.main()
