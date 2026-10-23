"""Near pronunciations may reach review without authorizing missing speech."""

import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.phonetic_match import phonetic_equivalence
from epub_tts_narrator.renderer.phonetic_review import PronunciationReviewer, substitution_pairs
from epub_tts_narrator.renderer.quality import text_confirmation_unavailable, validation_key
from epub_tts_narrator.renderer.speech_coverage import SPEECH_COVERAGE_VERSION, speech_coverage
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from test_renderer_jev_verification import Judge
from test_renderer_phonetic_fallback import PairClient
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import PrimaryVerifier


EXPECTED = (
    "The ship woman could read their minds while the captain waited by the window. "
    "We kept the doors locked until all the travelers returned from the village."
)
RECOGNIZED = EXPECTED.replace("ship woman", "shipwoman").replace("minds", "mines")


class NearPhoneticCoverageTests(unittest.TestCase):
    def test_near_complete_word_sounds_are_reviewable_in_both_directions(self):
        # These all retain the complete word; a weak stop in the middle of a
        # consonant cluster may be absent from the recognizer's spelling.
        for left, right in (("minds", "mines"), ("hands", "hans"),
                            ("bands", "bans"), ("finds", "fines"), ("winds", "wins")):
            for expected, actual in ((left, right), (right, left)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertIsNone(phonetic_equivalence(expected, actual))
                    result = speech_coverage(expected, actual)
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(len(result["alignment"]), 1)
                    change = result["alignment"][0]
                    self.assertEqual(change["operation"], "pronunciation_candidate")
                    self.assertEqual(change["expected"], [expected])
                    self.assertEqual(change["actual"], [actual])
                    self.assertEqual((change["expected_start"], change["expected_end"]), (0, 1))
                    self.assertEqual((change["actual_start"], change["actual_end"]), (0, 1))

    def test_compound_boundary_does_not_hide_or_require_review_of_a_second_word(self):
        coverage = speech_coverage(EXPECTED, RECOGNIZED)
        self.assertTrue(coverage["passed"], coverage)
        self.assertEqual(len(coverage["alignment"]), 2)
        self.assertEqual(coverage["alignment"][0]["expected"], ["ship", "woman"])
        self.assertEqual(coverage["alignment"][0]["actual"], ["shipwoman"])
        self.assertEqual(coverage["alignment"][1]["operation"], "pronunciation_candidate")
        pairs = substitution_pairs(EXPECTED, RECOGNIZED)
        self.assertEqual([(pair["expected_word"], pair["recognized_word"])
                          for pair in pairs], [("minds", "mines")])
        pair = pairs[0]
        self.assertEqual(EXPECTED[pair["expected_source_start"]:pair["expected_source_end"]], "minds")
        self.assertEqual(RECOGNIZED[pair["recognized_source_start"]:pair["recognized_source_end"]], "mines")

    def test_short_different_function_protected_and_unrelated_words_stay_ineligible(self):
        cases = (("cat", "cap"), ("cold", "coal"), ("does", "is"),
                 ("about", "above"), ("without", "with"), ("not", "now"),
                 ("eleven", "eleventh"), ("seventeen", "seventy"),
                 ("Jasper", "Jasmine"), ("book", "candle"), ("weaponized", "weapon"))
        for left, right in cases:
            for expected, actual in ((left, right), (right, left)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertFalse(speech_coverage(expected, actual)["passed"])
                    self.assertIsNone(substitution_pairs(expected, actual))

    def test_near_keys_do_not_drop_a_spoken_initialism_letter(self):
        for longer, shorter in (("ABCDS", "ABCD"), ("MNOPS", "MNOP")):
            for left, right in ((longer, shorter), (longer, shorter.lower()),
                                (longer.lower(), shorter)):
                for expected, actual in ((left, right), (right, left)):
                    with self.subTest(expected=expected, actual=actual):
                        self.assertFalse(speech_coverage(expected, actual)["passed"])
                        self.assertIsNone(substitution_pairs(expected, actual))

    def test_near_word_candidate_cannot_consume_missing_extra_or_reordered_words(self):
        cases = (
            RECOGNIZED.replace("their mines", "mines"),
            RECOGNIZED.replace("the captain waited", "waited"),
            RECOGNIZED.replace("read their mines", "read their"),
            RECOGNIZED.replace("read their mines", "read their mines mines"),
            RECOGNIZED + " Thank you.",
            RECOGNIZED.replace("doors locked", "locked doors"),
        )
        for actual in cases:
            with self.subTest(actual=actual):
                self.assertFalse(speech_coverage(EXPECTED, actual)["passed"])
                self.assertIsNone(substitution_pairs(EXPECTED, actual))


class NearPhoneticVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual=RECOGNIZED, *, decision="equivalent", votes=None):
        judge = Judge(decision)
        client = PairClient(votes) if votes is not None else None
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier(actual), judge,
            PronunciationReviewer(client) if client is not None else None)
        return verifier, judge, client

    def test_verification_key_tracks_coverage_policy_even_without_analysis_reviewer(self):
        verifier, _, _ = self.wrapped()
        for purpose in ("reference", "speech"):
            with self.subTest(purpose=purpose):
                current = validation_key(EXPECTED, verifier, purpose=purpose)
                with patch("epub_tts_narrator.renderer.speech_coverage.SPEECH_COVERAGE_VERSION",
                           SPEECH_COVERAGE_VERSION + 1):
                    self.assertNotEqual(current, validation_key(EXPECTED, verifier, purpose=purpose))

    def test_external_confirmation_accepts_whole_speech_and_keeps_actual_clone_transcript(self):
        for purpose in ("reference", "speech"):
            with self.subTest(purpose=purpose):
                verifier, judge, _ = self.wrapped()
                report = self.check(verifier, expected=EXPECTED, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(judge.calls, [(EXPECTED, RECOGNIZED, None)])
                self.assertEqual(report["text_confirmation_method"], "jev_primary")
                self.assertEqual(report["expected_text"], EXPECTED)
                self.assertEqual(report["transcript"], RECOGNIZED)
                self.assertEqual(report["verified_transcript"], RECOGNIZED)
                self.assertGreater(report["metrics"]["wer"], 0)
                self.assertNotIn("phonetic_comparison", report)
                if purpose == "reference":
                    self.assertEqual(report["reference_text"], RECOGNIZED)
                    self.assertTrue(report["reference_text_adjusted"])
                else:
                    self.assertNotIn("reference_text", report)

    def test_eligibility_is_not_approval_when_the_judge_disagrees(self):
        for purpose in ("reference", "speech"):
            for decision in ("different", "uncertain"):
                with self.subTest(purpose=purpose, decision=decision):
                    verifier, judge, _ = self.wrapped(decision=decision)
                    report = self.check(verifier, expected=EXPECTED, purpose=purpose)
                    self.assertFalse(report["passed"], report)
                    self.assertEqual(report["reasons"], ["text_mismatch"])
                    self.assertEqual(judge.calls, [(EXPECTED, RECOGNIZED, None)])
                    self.assertNotIn("reference_text", report)
                    self.assertNotIn("text_judge_skipped", report)

    def test_existing_three_way_review_can_resolve_negative_judge_score(self):
        for purpose in ("reference", "speech"):
            with self.subTest(purpose=purpose):
                verifier, judge, client = self.wrapped(decision="different", votes=(True, False, True))
                report = self.check(verifier, expected=EXPECTED, purpose=purpose)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["text_confirmation_method"], "phonetic_primary")
                self.assertEqual(len(judge.calls), 1)
                self.assertEqual(len(client.calls), 3)
                self.assertEqual(report["phonetic_reviews"][0]["decision"], "accepted")
                self.assertEqual([(pair["expected_word"], pair["recognized_word"])
                                  for pair in report["phonetic_reviews"][0]["pairs"]],
                                 [("minds", "mines")])

    def test_negative_three_way_review_does_not_become_approval(self):
        verifier, _, client = self.wrapped(decision="different", votes=(False, True, False))
        report = self.check(verifier, expected=EXPECTED)
        self.assertFalse(report["passed"], report)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "rejected")
        self.assertNotIn("reference_text", report)

    def test_judge_outage_stays_a_verification_failure(self):
        verifier, judge, _ = self.wrapped(decision="unavailable")
        report = self.check(verifier, expected=EXPECTED)
        self.assertFalse(report["passed"], report)
        self.assertTrue(text_confirmation_unavailable(report))
        self.assertEqual(len(judge.calls), 1)
        self.assertNotIn("reference_text", report)

    def test_optimistic_judges_cannot_rescue_missing_or_extra_words_numbers_and_negation(self):
        cases = (
            (EXPECTED, RECOGNIZED.replace("the captain waited", "waited")),
            (EXPECTED, RECOGNIZED + " Thank you."),
            ("The ship woman could not read their minds.", "The shipwoman could read their mines."),
            ("The ship woman read seventeen minds.", "The shipwoman read seventy mines."),
            ("The ship woman read their minds.", "The shipwoman burned their mines."),
            ("The ship woman read their minds.", "Their mines read the shipwoman."),
        )
        for purpose in ("reference", "speech"):
            for expected, actual in cases:
                with self.subTest(purpose=purpose, expected=expected, actual=actual):
                    verifier, judge, client = self.wrapped(actual, votes=(True, True, True))
                    report = self.check(verifier, expected=expected, purpose=purpose)
                    self.assertFalse(report["passed"], report)
                    self.assertEqual(judge.calls, [])
                    self.assertEqual(client.calls, [])
                    self.assertEqual(report["text_judge_skipped"], "incomplete_spoken_coverage")


if __name__ == "__main__":
    unittest.main()
