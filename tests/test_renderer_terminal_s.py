"""Weak word endings can be reviewed without excusing absent spoken words."""

from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.phonetic_match import phonetic_equivalence
from epub_tts_narrator.renderer.phonetic_review import (
    PronunciationReviewer, substitution_pairs, validate_review,
)
from epub_tts_narrator.renderer.quality import text_confirmation_unavailable
from epub_tts_narrator.renderer.speech_coverage import speech_coverage
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier
from test_renderer_jev_verification import Judge
from test_renderer_phonetic_fallback import PairClient
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import PrimaryVerifier


REQUESTED = (
    "“I will turn my full attention to my ship-body’s sensors and weapons systems, "
    "so that I can support you as much as possible. "
)
RECOGNIZED = (
    " I will turn my full attention to my ship body's sensors and weapon systems "
    "so that I can support you as much as possible."
)
TWO_ENDINGS = RECOGNIZED.replace("body's", "body")


class TerminalSCoverageTests(unittest.TestCase):
    def test_final_s_variants_are_only_review_candidates_in_both_directions(self):
        for left, right in (("weapons", "weapon"), ("body's", "body"), ("dogs", "dog")):
            for expected, actual in ((left, right), (right, left)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertIsNone(phonetic_equivalence(expected, actual))
                    result = speech_coverage(expected, actual)
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(len(result["alignment"]), 1)
                    change = result["alignment"][0]
                    self.assertEqual(change["operation"], "pronunciation_candidate")
                    self.assertEqual(change["rule"], "terminal_s_variation")
                    self.assertEqual(change["expected"], [expected])
                    self.assertEqual(change["actual"], [actual])
                    self.assertEqual((change["expected_start"], change["expected_end"]), (0, 1))
                    self.assertEqual((change["actual_start"], change["actual_end"]), (0, 1))

    def test_real_passage_retains_both_exact_source_spans_for_review(self):
        pairs = substitution_pairs(REQUESTED, TWO_ENDINGS)
        self.assertEqual(len(pairs), 2)
        self.assertEqual([(pair["expected_word"], pair["recognized_word"])
                          for pair in pairs], [("body’s", "body"), ("weapons", "weapon")])
        for pair in pairs:
            self.assertEqual(pair["coverage_rule"], "terminal_s_variation")
            self.assertEqual(REQUESTED[pair["expected_source_start"]:pair["expected_source_end"]],
                             pair["expected_word"])
            self.assertEqual(TWO_ENDINGS[pair["recognized_source_start"]:pair["recognized_source_end"]],
                             pair["recognized_word"])

    def test_short_stems_contractions_function_words_numbers_and_initials_stay_blocked(self):
        cases = (
            ("gas", "ga"), ("hi", "his"), ("he's", "he"), ("let's", "let"),
            ("hers", "her"), ("theirs", "their"), ("there's", "there"),
            ("ones", "one"), ("dozens", "dozen"), ("nots", "not"),
            ("nevers", "never"), ("CCs", "CC"), ("ABCS", "ABC"),
            ("brrrs", "brrr"), ("weaponized", "weapon"),
        )
        for left, right in cases:
            for expected, actual in ((left, right), (right, left)):
                with self.subTest(expected=expected, actual=actual):
                    self.assertFalse(speech_coverage(expected, actual)["passed"])
                    self.assertIsNone(substitution_pairs(expected, actual))

    def test_suffix_candidate_does_not_hide_a_missing_extra_or_reordered_word(self):
        cases = (
            RECOGNIZED.replace("sensors and ", "and "),
            RECOGNIZED.replace("weapon systems", "weapon weapon systems"),
            RECOGNIZED + " Thank you.",
            RECOGNIZED.replace("full attention", "attention full"),
            RECOGNIZED.replace("support you", "support"),
        )
        for actual in cases:
            with self.subTest(actual=actual):
                self.assertFalse(speech_coverage(REQUESTED, actual)["passed"])
                self.assertIsNone(substitution_pairs(REQUESTED, actual))

    def test_initialism_letters_cannot_disappear_when_the_other_side_is_lowercase(self):
        for inflected, base in (("ABCDS", "ABCD"), ("MNOPS", "MNOP"), ("ABCS", "ABC")):
            for left, right in ((inflected, base), (inflected, base.lower()),
                                (inflected.lower(), base)):
                for expected, actual in ((left, right), (right, left)):
                    with self.subTest(expected=expected, actual=actual):
                        self.assertIsNone(phonetic_equivalence(expected, actual))
                        self.assertFalse(speech_coverage(expected, actual)["passed"])
                        self.assertIsNone(substitution_pairs(expected, actual))

    def test_pronunciation_review_keeps_existing_pair_count_and_fraction_limits(self):
        self.assertIsNone(substitution_pairs("Dogs cats birds wolves snakes.",
                                             "Dog cat bird wolve snake."))
        expected = "We checked dogs cats birds wolves snakes while the captain waited nearby."
        actual = "We checked dog cat bird wolve snake while the captain waited nearby."
        self.assertIsNone(substitution_pairs(expected, actual))


class PairSpecificClient(PairClient):
    """A majority for one ending must never approve a different disputed ending."""

    def complete(self, prompt, payload):
        self.calls.append((prompt, deepcopy(payload)))
        return {"pairs": [{"id": pair["id"], "plausible": index == 0,
                           "reason": "Fixture opinion about this word's terminal sound."}
                          for index, pair in enumerate(payload["pairs"])]}


class TerminalSVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual=RECOGNIZED, *, votes=(True, False, True), reviewer=True):
        judge = Judge("different")
        client = PairClient(votes)
        wrapped = ReferenceTextJudgingVerifier(
            PrimaryVerifier(actual), judge,
            PronunciationReviewer(client) if reviewer else None)
        return wrapped, judge, client

    def test_coverage_eligibility_alone_does_not_approve_audio(self):
        verifier, judge, client = self.wrapped(reviewer=False)
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["reasons"], ["text_mismatch"])
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(client.calls, [])

    def test_jury_can_resolve_suffix_recognition_with_original_metrics_and_audit(self):
        for actual in (RECOGNIZED, TWO_ENDINGS):
            for purpose in ("speech", "reference"):
                with self.subTest(actual=actual, purpose=purpose):
                    verifier, judge, client = self.wrapped(actual)
                    report = self.check(verifier, expected=REQUESTED, purpose=purpose)
                    self.assertTrue(report["passed"], report)
                    self.assertEqual(report["expected_text"], REQUESTED)
                    self.assertEqual(report["transcript"], actual)
                    self.assertEqual(report["raw_transcript"], actual)
                    self.assertEqual(report["verified_transcript"], actual)
                    self.assertGreater(report["metrics"]["wer"], 0)
                    self.assertEqual(report["text_confirmation_method"], "phonetic_primary")
                    self.assertEqual(report["text_judge"]["decision"], "different")
                    self.assertEqual(len(judge.calls), 1)
                    self.assertEqual(len(client.calls), 3)
                    audit = report["phonetic_reviews"][0]
                    self.assertEqual(audit["decision"], "accepted")
                    self.assertTrue(all(pair["coverage_rule"] == "terminal_s_variation"
                                        for pair in audit["pairs"]))

    def test_every_disputed_ending_needs_its_own_majority(self):
        client = PairSpecificClient()
        reviewer = PronunciationReviewer(client)
        review = reviewer.review(REQUESTED, TWO_ENDINGS)
        self.assertEqual(len(review["pairs"]), 2)
        self.assertEqual(review["decision"], "rejected")
        self.assertFalse(validate_review(review, REQUESTED, TWO_ENDINGS, reviewer.identity))
        verifier = ReferenceTextJudgingVerifier(
            PrimaryVerifier(TWO_ENDINGS), Judge("different"), reviewer)
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "rejected")

    def test_reviewer_outage_does_not_become_approval(self):
        verifier, _, client = self.wrapped()
        client.error = TimeoutError("fixture review unavailable")
        report = self.check(verifier, expected=REQUESTED, purpose="speech")
        self.assertFalse(report["passed"], report)
        self.assertTrue(text_confirmation_unavailable(report))
        self.assertEqual(report["phonetic_reviews"][0]["decision"], "unavailable")

    def test_suffix_variation_cannot_rescue_missing_words_or_changed_negation_and_number(self):
        cases = (
            (REQUESTED, RECOGNIZED.replace("sensors and ", "and ")),
            (REQUESTED, RECOGNIZED + " Thank you."),
            ("The weapons did not damage the wall.", "The weapon did damage the wall."),
            ("The weapons struck seventeen targets.", "The weapon struck seventy targets."),
        )
        for expected, actual in cases:
            with self.subTest(actual=actual):
                verifier, judge, client = self.wrapped(actual, votes=(True, True, True))
                report = self.check(verifier, expected=expected, purpose="speech")
                self.assertFalse(report["passed"], report)
                self.assertEqual(judge.calls, [])
                self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
