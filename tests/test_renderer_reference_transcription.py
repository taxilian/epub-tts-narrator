"""Reference-only transcription nomination and audit boundaries; no network."""

from copy import deepcopy
import unittest

from epub_tts_narrator.renderer import phonetic_review
from test_renderer_phonetic_review import Client


class ReferenceSubstitutionPairsTests(unittest.TestCase):
    def test_unaligned_reference_examples_nominate_complete_original_spans(self):
        examples = (
            ("Dalley checked the records.", "Darly checked the records.", "Dalley", "Darly"),
            ("Dalley checked the records.", "Darling checked the records.", "Dalley", "Darling"),
            ("I assume you're here about the records.", "I assume you hear about the records.",
             "you're here", "you hear"),
            ("You and probably the captain knew.", "You I'm probably the captain knew.", "and", "I'm"),
            ("The firepower she carried was impressive.", "The fire tower she carried was impressive.",
             "firepower", "fire tower"),
        )
        for expected, heard, expected_span, heard_span in examples:
            with self.subTest(expected=expected, heard=heard):
                pairs = phonetic_review.reference_substitution_pairs(expected, heard)
                self.assertEqual(len(pairs), 1)
                pair = pairs[0]
                self.assertEqual(pair["expected_word"], expected_span)
                self.assertEqual(pair["recognized_word"], heard_span)
                self.assertEqual(expected[pair["expected_source_start"]:pair["expected_source_end"]],
                                 expected_span)
                self.assertEqual(heard[pair["recognized_source_start"]:pair["recognized_source_end"]],
                                 heard_span)
                self.assertLessEqual(pair["expected_end"] - pair["expected_start"], 3)
                self.assertLessEqual(pair["actual_end"] - pair["actual_start"], 3)
                self.assertNotIn("plausible", pair)

    def test_strictly_covered_and_exact_transcripts_do_not_enter_reference_review(self):
        for expected, heard in (("Hello", "hello"), ("Nerys waited.", "Nerris waited.")):
            with self.subTest(expected=expected, heard=heard):
                self.assertIsNone(phonetic_review.reference_substitution_pairs(expected, heard))
        self.assertIsNone(phonetic_review.substitution_pairs(
            "Jasper waited beside the gate.", "Jasmine waited beside the gate."))

    def test_missing_extra_reordered_numeric_negated_and_nonalphabetic_edits_are_ineligible(self):
        examples = (
            ("Dalley left before sunrise.", "Darling left sunrise."),
            ("Dalley left before sunrise.", "Darling left before sunrise today."),
            ("She had had enough.", "She had enough."),
            ("Nerys gave Maren a parcel.", "Maren gave Nerys a parcel."),
            ("Bring seventeen maps.", "Bring seventy maps."),
            ("They said not before leaving.", "They said now before leaving."),
            ("They said can't before leaving.", "They said can before leaving."),
            ("Code alpha7 remains.", "Code alpha8 remains."),
            ("alpha beta gamma delta epsilon", "zeta eta theta iota kappa"),
        )
        for expected, heard in examples:
            with self.subTest(expected=expected, heard=heard):
                self.assertIsNone(phonetic_review.reference_substitution_pairs(expected, heard))

    def test_shared_changed_words_across_spans_reject_disguised_reordering(self):
        self.assertIsNone(phonetic_review.reference_substitution_pairs(
            "Nerys gave Maren", "Maren gave Nerris"))

    def test_pair_and_fraction_budgets_reject_the_whole_passage(self):
        expected = " ".join(["Dalley waited beside the old gate"] * 5)
        self.assertIsNone(phonetic_review.reference_substitution_pairs(
            expected, expected.replace("Dalley", "Darling")))
        self.assertIsNone(phonetic_review.reference_substitution_pairs(
            "Dalley waited beside Maren", "Darling waited beside Nerris"))

    def test_a_later_missing_word_vetoes_an_otherwise_valid_pair(self):
        expected = "Dalley checked records before everyone else arrived."
        heard = "Darling checked records before everyone else."
        self.assertIsNone(phonetic_review.reference_substitution_pairs(expected, heard))

    def test_invalid_and_oversized_text_is_ineligible(self):
        for expected, heard in ((None, "Dalley"), ("Dalley", {}), ("", ""),
                                ("word " * 2001, "herd " * 2001)):
            with self.subTest(kind=(type(expected), type(heard))):
                self.assertIsNone(phonetic_review.reference_substitution_pairs(expected, heard))


class ReferencePronunciationReviewerTests(unittest.TestCase):
    expected = "Dalley checked the records beside the old gate."
    heard = "Darling checked the records beside the old gate."

    def test_reference_acceptance_uses_three_identical_blind_payloads_and_validates(self):
        client = Client((True, False, True))
        reviewer = phonetic_review.PronunciationReviewer(client)

        record = reviewer.review_reference(self.expected, self.heard)

        self.assertEqual(record["decision"], "accepted")
        self.assertTrue(phonetic_review.validate_reference_review(
            record, self.expected, self.heard, reviewer.reference_identity))
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertEqual(client.calls[1], client.calls[2])
        self.assertEqual(set(client.calls[0][1]), {"pairs"})
        self.assertNotEqual(reviewer.reference_identity, reviewer.identity)

    def test_negative_reference_votes_reject_and_keep_the_complete_audit(self):
        reviewer = phonetic_review.PronunciationReviewer(Client((False, True, False)))

        record = reviewer.review_reference(self.expected, self.heard)

        self.assertEqual(record["decision"], "rejected")
        self.assertEqual(len(record["votes"]), 3)
        self.assertFalse(phonetic_review.validate_reference_review(
            record, self.expected, self.heard, reviewer.reference_identity))

    def test_ineligible_reference_passages_never_call_the_provider(self):
        client = Client()
        reviewer = phonetic_review.PronunciationReviewer(client)
        for heard in ("Darling checked the records beside the gate.",
                      "Darling checked the records beside the old gate today."):
            with self.subTest(heard=heard):
                record = reviewer.review_reference(self.expected, heard)
                self.assertEqual(record["decision"], "ineligible")
                self.assertEqual(record["votes"], [])
        self.assertEqual(client.calls, [])

    def test_reference_outages_and_malformed_answers_fail_closed(self):
        for response in ({}, {"pairs": []}, RuntimeError("fixture-secret-do-not-log")):
            with self.subTest(response=response):
                client = Client((True, response, True))
                reviewer = phonetic_review.PronunciationReviewer(client)
                record = reviewer.review_reference(self.expected, self.heard)
                self.assertEqual(record["decision"], "unavailable")
                self.assertEqual(len(record["votes"]), 1)
                self.assertNotIn("fixture-secret", str(record))
                self.assertFalse(phonetic_review.validate_reference_review(
                    record, self.expected, self.heard, reviewer.reference_identity))

    def test_reference_audit_tampering_and_cross_route_validation_are_rejected(self):
        reviewer = phonetic_review.PronunciationReviewer(Client())
        record = reviewer.review_reference(self.expected, self.heard)
        mutations = (
            lambda value: value.update(prompt="different"),
            lambda value: value["pairs"][0].update(expected_source_start=1),
            lambda value: value["pairs"][0].update(recognized_word="Darly"),
            lambda value: value["votes"].pop(),
            lambda value: value["votes"][0]["pairs"][0].update(plausible=1),
            lambda value: value.update(extra=True),
        )
        for mutate in mutations:
            altered = deepcopy(record)
            mutate(altered)
            self.assertFalse(phonetic_review.validate_reference_review(
                altered, self.expected, self.heard, reviewer.reference_identity))
        self.assertFalse(phonetic_review.validate_review(
            record, self.expected, self.heard, reviewer.identity))
        self.assertFalse(phonetic_review.validate_reference_review(
            record, self.expected, self.heard, reviewer.identity))

        strict = phonetic_review.PronunciationReviewer(Client())
        strict_expected = "Nerys waited beside the old gate."
        strict_record = strict.review(strict_expected, "Nerris waited beside the old gate.")
        self.assertEqual(strict_record["decision"], "accepted")
        self.assertFalse(phonetic_review.validate_reference_review(
            strict_record, strict_expected, "Nerris waited beside the old gate.", strict.reference_identity))
        self.assertFalse(phonetic_review.validate_review(
            strict_record, strict_expected, "Nerris waited beside the old gate.", strict.reference_identity))

    def test_reference_identity_is_copied_and_every_policy_component_is_validated(self):
        client = Client()
        reviewer = phonetic_review.PronunciationReviewer(client)
        record = reviewer.review_reference(self.expected, self.heard)
        identity = reviewer.reference_identity
        identity["provider"]["model"] = "changed"
        identity["nomination"]["version"] += 1
        self.assertNotEqual(identity, reviewer.reference_identity)

        malformed = (None, {}, reviewer.identity,
                     {**reviewer.reference_identity, "required_votes": 1},
                     {**reviewer.reference_identity, "coverage_version": -1})
        for candidate in malformed:
            with self.subTest(identity=candidate):
                altered = deepcopy(record)
                altered["identity"] = candidate
                self.assertFalse(phonetic_review.validate_reference_review(
                    altered, self.expected, self.heard, candidate))
        nomination = deepcopy(reviewer.reference_identity)
        nomination["nomination"]["max_span_words"] = 4
        altered = deepcopy(record)
        altered["identity"] = nomination
        self.assertFalse(phonetic_review.validate_reference_review(
            altered, self.expected, self.heard, nomination))

    def test_reference_provider_change_preserves_the_existing_unavailable_contract(self):
        client = Client()
        reviewer = phonetic_review.PronunciationReviewer(client)
        client.identity = {**client.identity, "model": "changed"}

        record = reviewer.review_reference(self.expected, self.heard)

        self.assertEqual(record["decision"], "unavailable")
        self.assertEqual(record["error_code"], "provider_changed")
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
