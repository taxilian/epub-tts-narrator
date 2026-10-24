import unittest

from epub_tts_narrator.renderer.phonetic_match import phonetic_equivalence
from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer, substitution_pairs
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver, validate_resolution
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
import test_renderer_reference_consensus as fixtures
from test_renderer_jev_verification import Judge
from test_renderer_phonetic_fallback import PairClient
from test_renderer_speech_resolver import Client, failure


EXPECTED = "Know-know! Get help!"
HEARD = "No, no! Get help!"


class ProtectedHomophoneCoverageTests(unittest.TestCase):
    def test_complete_non_numeric_protected_sounds_reach_review_in_both_directions(self):
        for first, second in (("know", "no"), ("knot", "not"), ("known", "none")):
            for expected, actual in ((first, second), (second, first)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertIsNone(phonetic_equivalence(expected, actual))
                    coverage = speech_coverage(expected, actual)
                    self.assertTrue(coverage["passed"], coverage)
                    self.assertEqual(len(coverage["alignment"]), 1)
                    change = coverage["alignment"][0]
                    self.assertEqual(change["operation"], "pronunciation_candidate")
                    self.assertEqual(change["rule"], "protected_homophone")
                    self.assertEqual(change["expected"], [expected])
                    self.assertEqual(change["actual"], [actual])
                    pairs = substitution_pairs(expected, actual)
                    self.assertEqual([(p["expected_word"], p["recognized_word"]) for p in pairs],
                                     [(expected, actual)])

    def test_candidate_never_consumes_missing_extra_reordered_or_numeric_content(self):
        for expected, actual in (
            (EXPECTED, "No! Get help!"),
            (EXPECTED, HEARD + " Thank you."),
            (EXPECTED, "No, no! Get bang mean!"),
            (EXPECTED, "Get help! No, no!"),
            ("I did not know.", "I did know."),
            ("not", "now"), ("never", "ever"),
            ("four", "for"), ("two", "to"), ("first", "furst"),
            ("NO", "know"), ("KNOW", "no"),
        ):
            with self.subTest(expected=expected, actual=actual):
                self.assertFalse(speech_coverage(expected, actual)["passed"])
                self.assertIsNone(substitution_pairs(expected, actual))

    def test_resolver_requires_sound_votes_for_each_repeated_homophone(self):
        failures = [failure(HEARD)]
        for votes, approved in (([[True], [False], [True]], True),
                                ([[True], [False], [False]], False)):
            with self.subTest(approved=approved):
                client = Client(votes=votes)
                resolver = SpeechResolver(client)
                audit = resolver.resolve(EXPECTED, failures, can_split=False)
                proposal = validate_resolution(audit, EXPECTED, failures, resolver.identity,
                                               can_split=False)
                self.assertEqual(len(client.calls), 4)
                self.assertEqual(len(audit["reviews"]), 3)
                self.assertEqual(len(audit["reviews"][0]["payload"]["pairs"]), 2)
                self.assertEqual(proposal is not None, approved)
                if not approved:
                    self.assertEqual(audit["error_code"], "sound_review_rejected")


class ProtectedHomophoneVerificationTests(unittest.TestCase):
    setUp = fixtures.ReferenceConsensusTests.setUp
    write_audio = fixtures.ReferenceConsensusTests.write_audio
    check = fixtures.ReferenceConsensusTests.check

    def test_existing_judgment_paths_require_actual_approval(self):
        context = " We must bring assistance to everyone waiting outside the gate."
        for purpose in ("reference", "speech"):
            for votes, approved in (((True, False, True), True), ((True, False, False), False)):
                with self.subTest(purpose=purpose, approved=approved):
                    client = PairClient(votes)
                    verifier = ReferenceTextJudgingVerifier(fixtures.PrimaryVerifier(HEARD + context),
                        Judge("different"), PronunciationReviewer(client))
                    report = self.check(verifier, expected=EXPECTED + context, purpose=purpose)
                    self.assertEqual(report["passed"], approved, report)
                    self.assertEqual(len(client.calls), 3)
                    if approved:
                        self.assertEqual(report["text_confirmation_method"], "phonetic_primary")

    def test_no_judge_and_failed_acoustics_cannot_use_review_eligibility_as_approval(self):
        report = self.check(fixtures.PrimaryVerifier(HEARD), expected=EXPECTED, purpose="speech")
        self.assertFalse(report["passed"])
        self.write_audio(amplitude=0)
        report = self.check(ReferenceTextJudgingVerifier(fixtures.PrimaryVerifier(HEARD), Judge("equivalent")),
                            expected=EXPECTED, purpose="speech")
        self.assertFalse(report["passed"])
        self.assertIn("silent_audio", report["reasons"])


if __name__ == "__main__":
    unittest.main()
