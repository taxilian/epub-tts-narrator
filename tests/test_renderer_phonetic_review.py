"""Coverage gates and auditable votes; no network or audio synthesis."""

from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.phonetic_review import (
    PROMPT, PronunciationReviewer, substitution_pairs, validate_review,
)


class Client:
    identity = {"model": "fixture-model", "base_url": "https://fixture.invalid/v1"}

    def __init__(self, answers=(True, True, True)):
        self.answers = iter(answers)
        self.calls = []

    def complete(self, prompt, payload):
        self.calls.append((prompt, deepcopy(payload)))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, dict):
            return answer
        return {"pairs": [{"id": pair["id"], "plausible": answer,
                           "reason": "The word sounds plausibly similar."}
                          for pair in payload["pairs"]]}


class SubstitutionPairsTests(unittest.TestCase):
    def test_unfamiliar_names_reach_review_without_being_locally_approved(self):
        for requested, heard in (("Derrins", "darins"), ("Derrins", "Darren’s"),
                                  ("Maelin", "Maylin"), ("Nerys", "Nerris"),
                                  ("flower", "flour")):
            with self.subTest(requested=requested, heard=heard):
                pairs = substitution_pairs(requested, heard)
                self.assertEqual(pairs[0]["expected_word"], requested)
                self.assertEqual(pairs[0]["recognized_word"], heard)
                self.assertEqual(pairs[0]["word_index"], 0)
                self.assertNotIn("plausible", pairs[0])

    def test_multiword_sound_candidates_bind_the_complete_original_spans(self):
        for requested, heard in (("Insang", "In saying"), ("Vandire", "Van Dyer"),
                                  ("admitted", "admit it"),
                                  ("moonlightflour", "moon light flower")):
            for left, right in ((requested, heard), (heard, requested)):
                expected = f"Before noon, “{left}” was heard beside the gate."
                actual = f"Before noon, “{right}” was heard beside the gate."
                with self.subTest(expected=expected, actual=actual):
                    pairs = substitution_pairs(expected, actual)
                    self.assertEqual(len(pairs), 1)
                    pair = pairs[0]
                    self.assertEqual(pair["expected_word"], left)
                    self.assertEqual(pair["recognized_word"], right)
                    self.assertEqual(pair["word_index"], 2)
                    self.assertEqual(pair["expected_start"], 2)
                    self.assertEqual(pair["expected_end"], 2 + len(left.split()))
                    self.assertEqual(pair["actual_start"], 2)
                    self.assertEqual(pair["actual_end"], 2 + len(right.split()))
                    self.assertEqual(expected[pair["expected_source_start"]:pair["expected_source_end"]], left)
                    self.assertEqual(actual[pair["recognized_source_start"]:pair["recognized_source_end"]], right)
                    self.assertIn(left, pair["requested_context"])
                    self.assertIn(right, pair["recognized_context"])
                    self.assertIn("was heard beside", pair["recognized_context"])
                    self.assertNotIn("the gate", pair["recognized_context"])

    def test_multiple_boundary_candidates_keep_their_own_shifted_positions(self):
        expected = "Insang greeted Vandire before everyone else arrived at the city gate."
        actual = "In saying greeted Van Dyer before everyone else arrived at the city gate."
        pairs = substitution_pairs(expected, actual)
        self.assertEqual([(p["expected_start"], p["expected_end"], p["actual_start"], p["actual_end"])
                          for p in pairs], [(0, 1, 0, 2), (2, 3, 3, 5)])
        self.assertEqual([p["id"] for p in pairs], ["pair_1", "pair_2"])
        self.assertEqual(pairs[1]["expected_word"], "Vandire")
        self.assertEqual(pairs[1]["recognized_word"], "Van Dyer")

    def test_proven_boundaries_and_contractions_need_no_model_vote(self):
        for expected, actual in (("What'cha said I can't do?", "What you said I can not do?"),
                                 ("The sourcepalace glowed.", "The source palace glowed.")):
            self.assertIsNone(substitution_pairs(expected, actual))
        expected = "Insang said I can't leave before sunrise."
        actual = "In saying said I can not leave before sunrise."
        pairs = substitution_pairs(expected, actual)
        self.assertEqual([(p["expected_word"], p["recognized_word"]) for p in pairs],
                         [("Insang", "In saying")])

    def test_original_casing_offsets_context_and_normalized_number_expansion(self):
        requested = "Before sunset 24 people followed Nerys along the narrow river into town."
        heard = "Before sunset twenty four people followed Nerris along the narrow river into town."
        pairs = substitution_pairs(requested, heard)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["word_index"], 6)
        self.assertEqual(pairs[0]["expected_word"], "Nerys")
        self.assertEqual(pairs[0]["requested_context"], "24 people followed Nerys along the narrow")
        self.assertEqual(pairs[0]["recognized_context"], "four people followed Nerris along the narrow")
        self.assertNotIn("sunset", pairs[0]["requested_context"])

    def test_identifier_name_review_preserves_original_spelling_and_context(self):
        requested = "“Confirmation A34785-Derrins-34r42,”"
        heard = "Confirmation A34785, Darren's 34R42."
        pairs = substitution_pairs(requested, heard)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["expected_word"], "Derrins")
        self.assertEqual(pairs[0]["recognized_word"], "Darren's")
        self.assertIn(pairs[0]["requested_context"], requested)
        self.assertIn("-Derrins-", pairs[0]["requested_context"])
        self.assertIn(pairs[0]["recognized_context"], heard)

    def test_missing_extra_and_reordered_words_are_ineligible(self):
        cases = (
            ("Nerys will not leave before sunrise.", "Nerris will leave before sunrise."),
            ("Nerys left before sunrise.", "Nerris left before sunrise today."),
            ("Nerys gave Maren a parcel by the old gate.", "Maren gave Nerys a parcel by the old gate."),
            ("Wait by the green gate beside the long road.", "Wait by the gate green beside the long road."),
            ("Nerys left before sunrise.", "Nerris left."),
            ("Jasper waited beside the gate.", "Jasmine waited beside the gate."),
            ("Insang Mortuary Services! What does it—", "In saying Mortuary Services! What is it?"),
            ("Insang left quietly before sunrise.", "In saying left before sunrise."),
            ("Insang left before sunrise.", "In saying left before sunrise today."),
            ("Insang left before sunrise.", "In, saying left before sunrise."),
        )
        for expected, heard in cases:
            with self.subTest(expected=expected, heard=heard):
                self.assertIsNone(substitution_pairs(expected, heard))

    def test_numbers_and_negations_never_reach_phonetic_review(self):
        for left, right in (("fifteen", "fifty"), ("15", "50"), ("first", "third"),
                            ("minus", "plus"), ("not", "now"), ("can't", "can"),
                            ("never", "ever"), ("no", "know"), ("won't", "want")):
            with self.subTest(left=left, right=right):
                self.assertIsNone(substitution_pairs("They said " + left + " before leaving the room.",
                                                     "They said " + right + " before leaving the room."))

    def test_fraction_and_pair_bounds_do_not_approve_similar_strings(self):
        self.assertIsNone(substitution_pairs("Jasper wore blue clothes today.", "Jasmine wore red clothes today."))
        expected = " ".join([f"Name{letter} waited beside the old gate" for letter in "ABCDE"])
        heard = expected.replace("Name", "Naym")
        self.assertIsNone(substitution_pairs(expected, heard))
        # Four nonnumeric substitutions across 24 words are eligible for judgment.
        expected = " ".join([f"Nerys{letter} waited beside the old gate" for letter in "ABCD"])
        heard = expected.replace("Nerys", "Nerris")
        self.assertEqual(len(substitution_pairs(expected, heard)), 4)

    def test_invalid_exact_and_oversized_text_are_ineligible(self):
        for expected, heard in ((None, "hello"), ("hello", {}), ("", ""),
                                ("Hello", "hello"), ("word " * 2001, "herd " * 2001)):
            with self.subTest(kind=(type(expected), type(heard))):
                self.assertIsNone(substitution_pairs(expected, heard))


class PronunciationReviewerTests(unittest.TestCase):
    expected = "Nerys stood beside the gate while everyone else hurried across the courtyard."
    heard = expected.replace("Nerys", "Nerris")

    def test_majority_requires_three_isolated_calls_with_only_local_context(self):
        client = Client((True, False, True))
        reviewer = PronunciationReviewer(client)
        record = reviewer.review(self.expected, self.heard)
        self.assertEqual(record["decision"], "accepted")
        self.assertTrue(validate_review(record, self.expected, self.heard, reviewer.identity))
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(client.calls[0], client.calls[1])
        self.assertEqual(client.calls[1], client.calls[2])
        self.assertEqual(set(client.calls[0][1]), {"pairs"})
        self.assertNotIn("courtyard", str(client.calls))
        self.assertEqual(record["prompt"], PROMPT)
        self.assertEqual([v["reviewer"] for v in record["votes"]], [1, 2, 3])
        self.assertNotIn("probability", record)

    def test_one_vote_cannot_approve_and_majorities_are_per_pair(self):
        reviewer = PronunciationReviewer(Client((True, False, False)))
        record = reviewer.review(self.expected, self.heard)
        self.assertEqual(record["decision"], "rejected")
        self.assertFalse(validate_review(record, self.expected, self.heard, reviewer.identity))
        expected = "Nerys stood beside Maelin while everyone else hurried across the courtyard."
        heard = expected.replace("Nerys", "Nerris").replace("Maelin", "Maylin")
        answers = [{"pairs": [{"id": f"pair_{i + 1}", "plausible": vote, "reason": "Sound comparison."}
                              for i, vote in enumerate(votes)]}
                   for votes in ((True, True), (True, False), (False, False))]
        reviewer = PronunciationReviewer(Client(answers))
        self.assertEqual(reviewer.review(expected, heard)["decision"], "rejected")

    def test_multiword_review_has_three_bound_votes_and_coverage_failure_makes_no_call(self):
        expected = "“Insang Mortuary Services! What does it—”"
        actual = "In saying mortuary services! What does it?"
        client = Client((True, False, True))
        reviewer = PronunciationReviewer(client)
        record = reviewer.review(expected, actual)
        self.assertEqual(record["decision"], "accepted")
        self.assertTrue(validate_review(record, expected, actual, reviewer.identity))
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(record["pairs"][0]["expected_word"], "Insang")
        self.assertEqual(record["pairs"][0]["recognized_word"], "In saying")
        self.assertEqual(record["pairs"][0]["coverage_operation"], "phonetic_group")
        self.assertEqual(record["pairs"][0]["coverage_rule"], "complete_group_sound")
        from epub_tts_narrator.renderer.speech_coverage import SPEECH_COVERAGE_VERSION
        self.assertEqual(reviewer.identity["coverage_version"], SPEECH_COVERAGE_VERSION)
        self.assertNotIn("JEV", PROMPT)
        for bad in (actual.replace("does", "is"), actual.replace("What ", ""),
                    actual + " Thanks."):
            with self.subTest(bad=bad):
                rejected = reviewer.review(expected, bad)
                self.assertEqual(rejected["decision"], "ineligible")
                self.assertEqual(rejected["votes"], [])
                self.assertEqual(len(client.calls), 3)

    def test_whole_passage_is_rechecked_when_an_audit_is_validated(self):
        expected = "Insang waited beside the gate while everyone else hurried across the courtyard."
        actual = expected.replace("Insang", "In saying")
        reviewer = PronunciationReviewer(Client())
        record = reviewer.review(expected, actual)
        self.assertTrue(validate_review(record, expected, actual, reviewer.identity))
        # The changed ending is outside the three-word context sent for Insang.
        actual = actual.replace("the courtyard", "")
        record["transcript"] = actual
        self.assertFalse(validate_review(record, expected, actual, reviewer.identity))

    def test_ineligible_passages_never_call_provider(self):
        client = Client()
        reviewer = PronunciationReviewer(client)
        for heard in (self.heard.replace("stood ", ""), self.heard + " Hello."):
            record = reviewer.review(self.expected, heard)
            self.assertEqual(record["decision"], "ineligible")
            self.assertFalse(validate_review(record, self.expected, heard, reviewer.identity))
        self.assertEqual(client.calls, [])

    def test_repeated_substitution_occurrences_are_bounded_before_any_api_call(self):
        client = Client()
        reviewer = PronunciationReviewer(client)
        expected = " ".join([self.expected] * 5)
        heard = expected.replace("Nerys", "Nerris")
        self.assertIsNone(substitution_pairs(expected, heard))
        record = reviewer.review(expected, heard)
        self.assertEqual(record["decision"], "ineligible")
        self.assertEqual(client.calls, [])
        self.assertEqual(reviewer.identity["limits"]["max_pairs"], 4)
        self.assertNotIn("max_distinct_pairs", reviewer.identity["limits"])

    def test_invalid_responses_and_transport_failures_fail_closed_without_leaking(self):
        good = {"id": "pair_1", "plausible": True, "reason": "Plausible sound."}
        bad = (None, {}, {"pairs": []}, {"pairs": [good, good]},
               {"pairs": [dict(good, id="invented")]}, {"pairs": [dict(good, plausible=1)]},
               {"pairs": [dict(good, plausible="true")]}, {"pairs": [dict(good, reason=" ")]},
               {"pairs": [dict(good, reason="x" * 401)]}, {"pairs": [dict(good, extra=True)]},
               {"pairs": [good], "extra": True}, RuntimeError("fixture-secret-do-not-log"))
        for response in bad:
            with self.subTest(response=response):
                client = Client((True, response, True))
                # Client supports raw None separately from regular boolean fixtures.
                original = client.complete
                if response is None:
                    def complete(prompt, payload):
                        value = original(prompt, payload)
                        return None if len(client.calls) == 2 else value
                    client.complete = complete
                reviewer = PronunciationReviewer(client)
                record = reviewer.review(self.expected, self.heard)
                self.assertEqual(record["decision"], "unavailable")
                self.assertEqual(len(record["votes"]), 1)
                self.assertEqual(len(client.calls), 2)
                self.assertNotIn("fixture-secret", str(record))
                self.assertFalse(validate_review(record, self.expected, self.heard, reviewer.identity))

    def test_audit_tampering_cannot_turn_a_record_into_approval(self):
        reviewer = PronunciationReviewer(Client())
        record = reviewer.review(self.expected, self.heard)
        mutations = (
            lambda x: x.update(identity={}), lambda x: x.update(prompt="different prompt"),
            lambda x: x.update(expected_text=self.heard), lambda x: x.update(transcript=self.expected),
            lambda x: x["pairs"][0].update(word_index=3),
            lambda x: x["pairs"][0].update(expected_start=3),
            lambda x: x["pairs"][0].update(actual_end=2),
            lambda x: x["pairs"][0].update(expected_source_start=1),
            lambda x: x["pairs"][0].update(recognized_source_end=1),
            lambda x: x["pairs"][0].update(coverage_operation="spelling_boundary"),
            lambda x: x["pairs"][0].update(coverage_rule="invented"),
            lambda x: x["pairs"][0].update(recognized_word="Jasmine"),
            lambda x: x["votes"].pop(), lambda x: x["votes"][1].update(reviewer=1),
            lambda x: x["votes"][0]["pairs"][0].update(plausible=1),
            lambda x: x["votes"][0]["pairs"][0].update(id="unbound"),
            lambda x: x.update(extra=True),
        )
        for mutate in mutations:
            altered = deepcopy(record)
            mutate(altered)
            self.assertFalse(validate_review(altered, self.expected, self.heard, reviewer.identity))
        for vote in record["votes"][:2]:
            vote["pairs"][0]["plausible"] = False
        self.assertFalse(validate_review(record, self.expected, self.heard, reviewer.identity))

    def test_identity_is_copied_and_provider_configuration_cannot_change_mid_review(self):
        client = Client()
        reviewer = PronunciationReviewer(client)
        identity = reviewer.identity
        identity["provider"]["model"] = "changed"
        self.assertNotEqual(identity, reviewer.identity)
        client.identity = {**client.identity, "model": "changed"}
        record = reviewer.review(self.expected, self.heard)
        self.assertEqual(record["decision"], "unavailable")
        self.assertEqual(record["error_code"], "provider_changed")
        self.assertEqual(client.calls, [])

    def test_malformed_identity_cannot_bypass_policy_or_raise_from_validation(self):
        reviewer = PronunciationReviewer(Client())
        record = reviewer.review(self.expected, self.heard)
        for identity in (None, {}, {"provider": {"bad": object()}},
                         {**reviewer.identity, "required_votes": 1},
                         {**reviewer.identity, "policy_sha256": "changed"}):
            with self.subTest(identity=identity):
                altered = deepcopy(record)
                altered["identity"] = identity
                self.assertFalse(validate_review(altered, self.expected, self.heard, identity))


if __name__ == "__main__":
    unittest.main()
