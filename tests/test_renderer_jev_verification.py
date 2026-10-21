"""A text judge can veto reference variations, never replace independent hearing."""

from copy import deepcopy
import json
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from audiobook_renderer.quality import normalize_words, validation_key
from audiobook_renderer.verification import ReferenceTextJudgingVerifier, create_verifier
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import (
    EXPECTED, IndependentVerifier, PrimaryVerifier,
)
from test_renderer_verification_factory import fake_primary, fake_reference


SCRIPT = (
    "The bloodbeast waited quietly just outside the station while the captain checked every locked "
    "door and window. A sourcevow bound the traveler to remain nearby until the night train "
    "arrived with fresh supplies for the village."
)
SPLIT = SCRIPT.replace("bloodbeast", "blood beast").replace("sourcevow", "source vow")


class Judge:
    def __init__(self, decision="equivalent", *, threshold=.9, model="jev-fixture"):
        self.identity = {"provider": "typesafe", "model": model, "policy_version": 3,
                         "threshold": threshold, "endpoint": "https://offline.invalid", "timeout_seconds": 10}
        score = {"equivalent": .98, "different": .01, "uncertain": .6, "unavailable": 0}[decision]
        self.response = {"decision": decision, "identity": deepcopy(self.identity), "policy_version": 3,
                         "threshold": threshold, "model": model if decision != "unavailable" else None,
                         "probabilities": {} if decision == "unavailable" else
                         {"harmless_variation": score}, "method": "jev_changed_spans",
                         "offset_unit": "normalized_words", "normalization": "source_explicit_compounds_v1",
                         "differences": []}
        if decision != "unavailable":
            expected, actual = normalize_words(SCRIPT), normalize_words(SPLIT)
            for number, (word, parts) in enumerate((("bloodbeast", ["blood", "beast"]),
                                                     ("sourcevow", ["source", "vow"]))):
                start, actual_start = expected.index(word), actual.index(parts[0])
                self.response["differences"].append({
                    "id": f"difference_{number}", "expected_text": word, "primary_transcript": " ".join(parts),
                    "context": " ".join(expected[max(0, start - 2):start + 3]),
                    "probability": score, "model": model, "policy_version": 3,
                    "occurrences": [{"transcript": transcript, "operation": "split",
                                     "expected_start": start, "expected_end": start + 1,
                                     "actual_start": actual_start, "actual_end": actual_start + 2,
                                     "expected": [word], "actual": parts}
                                    for transcript in ("primary", "secondary")]})
        self.calls = []

    def judge(self, expected, primary, secondary=None):
        self.calls.append((expected, primary, secondary))
        if isinstance(self.response, Exception):
            raise self.response
        return deepcopy(self.response)


class JevReferenceVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual=SPLIT, secondary=SPLIT, judge=None):
        recognizer = IndependentVerifier(primary=actual, secondary=secondary)
        return ReferenceTextJudgingVerifier(recognizer, judge or Judge()), recognizer

    def test_equivalent_compound_splits_still_require_independent_hearing(self):
        judge = Judge()
        verifier, recognizer = self.wrapped(judge=judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], SPLIT)
        self.assertEqual(report["text_judge"], judge.response)
        self.assertEqual(judge.calls, [(SCRIPT, SPLIT, SPLIT)])
        self.assertEqual(recognizer.calls, [("reference", self.audio), ("corroboration", self.audio)])

    def test_different_uncertain_unavailable_and_failed_judges_reject(self):
        for decision in ("different", "uncertain", "unavailable", "exception"):
            with self.subTest(decision=decision):
                judge = Judge(decision if decision != "exception" else "equivalent")
                if decision == "exception":
                    judge.response = RuntimeError("PRIVATE_API_KEY PRIVATE_REQUEST_TEXT")
                verifier, _ = self.wrapped(judge=judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertNotIn("reference_text", report)
                self.assertNotIn("PRIVATE", json.dumps(report, allow_nan=False))

    def test_each_changed_phrase_must_clear_threshold_and_audit_survives(self):
        from epub_audiobook.jev import JevTranscriptJudge

        for scores, passed in (((.94, .92), True), ((.94, .47), False)):
            with self.subTest(scores=scores):
                judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})

                def evaluate(state, questions):
                    self.assertEqual(set(questions), {"harmless_variation"})
                    self.assertEqual(set(state), {"expected_text", "primary_transcript", "context"})
                    score = dict(zip(("bloodbeast", "sourcevow"), scores))[state["expected_text"]]
                    return {"model": "jev-fixture", "usage": {"input_tokens": 40, "output_tokens": 2},
                            "answers": {"harmless_variation": {"type": "noul", "noul": score}}}

                verifier, _ = self.wrapped(judge=judge)
                with patch.object(judge._client, "evaluate", side_effect=evaluate) as evaluate_mock:
                    report = self.check(verifier, expected=SCRIPT)

                self.assertEqual(evaluate_mock.call_count, 2)
                self.assertEqual([call.args[0]["expected_text"] for call in evaluate_mock.call_args_list],
                                 ["bloodbeast", "sourcevow"])
                self.assertEqual(report["passed"], passed, report)
                audit = report["text_judge"]
                self.assertEqual(audit["probabilities"], {"harmless_variation": min(scores)})
                self.assertEqual([item["probability"] for item in audit["differences"]], list(scores))
                self.assertEqual(audit["decision"], "equivalent" if passed else "uncertain")
                self.assertEqual(audit["offset_unit"], "normalized_words")
                self.assertEqual(audit["usage"], {"input_tokens": 80, "output_tokens": 4})
                self.assertNotIn("fixture-private-key", json.dumps(audit, allow_nan=False))

    def test_equivalent_label_or_high_aggregate_cannot_hide_uncertain_difference(self):
        for aggregate in (.47, .98):
            with self.subTest(aggregate=aggregate):
                judge = Judge()
                judge.response["differences"][1]["probability"] = .47
                judge.response["probabilities"]["harmless_variation"] = aggregate
                verifier, _ = self.wrapped(judge=judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertNotIn("reference_text", report)

    def test_later_judge_outage_retains_successful_partial_audit_and_safe_cause(self):
        from epub_audiobook.jev import JevTranscriptJudge, QUESTIONS
        from test_jev import Opener, response

        secret = "PRIVATE_API_KEY_MUST_NOT_APPEAR"
        judge = JevTranscriptJudge({"enabled": True, "api_key": secret})
        first = response(QUESTIONS, {"harmless_variation": .94})
        judge._client._opener = Opener(first, TimeoutError(secret))
        judge._client.max_retries = 0
        verifier, _ = self.wrapped(judge=judge)
        report = self.check(verifier, expected=SCRIPT)

        self.assertFalse(report["passed"], report)
        self.assertNotIn("reference_text", report)
        self.assertEqual(len(judge._client._opener.calls), 2)
        audit = report["text_judge"]
        self.assertEqual(audit["decision"], "unavailable")
        self.assertEqual(audit["error_code"], "transport_error")
        self.assertEqual(audit["model"], first["model"])
        self.assertEqual(audit["probabilities"], {"harmless_variation": .94})
        self.assertEqual(len(audit["differences"]), 1)
        self.assertEqual(audit["differences"][0]["expected_text"], "bloodbeast")
        self.assertEqual(audit["differences"][0]["probability"], .94)
        self.assertEqual(audit["usage"], first["usage"])
        self.assertNotIn(secret, json.dumps(report, allow_nan=False))

    def test_later_judge_model_change_retains_both_model_audits_without_approving(self):
        from epub_audiobook.jev import JevTranscriptJudge, QUESTIONS
        from test_jev import Opener, response

        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        first = response(QUESTIONS)
        second = dict(response(QUESTIONS), model="jev-other-version")
        judge._client._opener = Opener(first, second)
        verifier, _ = self.wrapped(judge=judge)
        report = self.check(verifier, expected=SCRIPT)

        self.assertFalse(report["passed"], report)
        audit = report["text_judge"]
        self.assertEqual(audit["decision"], "unavailable")
        self.assertEqual(audit["error_code"], "model_changed")
        self.assertEqual([row["model"] for row in audit["differences"]], [first["model"], second["model"]])

    def test_malformed_partial_judgments_are_discarded_without_approving(self):
        for mutate in (
            lambda r: dict(r, probabilities={}),
            lambda r: dict(r, model=None),
            lambda r: dict(r, differences=[]),
            lambda r: dict(r, probabilities={"harmless_variation": float("nan")}),
            lambda r: dict(r, probabilities={"harmless_variation": .1}),
            lambda r: dict(r, differences=[dict(r["differences"][0], occurrences=[])]),
            lambda r: dict(r, differences=[dict(r["differences"][0], model="jev-unbound")]),
            lambda r: dict(r, usage={"input_tokens": True, "output_tokens": 2}),
            lambda r: dict(r, usage={"input_tokens": 1, "output_tokens": float("nan")}),
            lambda r: dict(r, usage={"input_tokens": 1, "output_tokens": 2, "private": "PRIVATE_DATA"}),
            lambda r: dict(r, error_code="PRIVATE_ERROR_TEXT"),
            lambda r: dict(r, error_code="model_changed"),  # Neither model actually changed.
        ):
            judge = Judge()
            judge.response.update(decision="unavailable", error_code="transport_error")
            judge.response = mutate(judge.response)
            with self.subTest(response=judge.response):
                verifier, _ = self.wrapped(judge=judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["text_judge"]["error_code"], "invalid_judge_record")
                self.assertNotIn("PRIVATE", json.dumps(report, allow_nan=False))

    def test_malformed_equivalence_and_inconsistent_probabilities_never_pass(self):
        for mutate in (
            lambda r: None,
            lambda r: {"decision": "equivalent"},
            lambda r: dict(r, decision="approved"),
            lambda r: dict(r, decision=[]),
            lambda r: dict(r, probabilities={"harmless_variation": float("nan")}),
            lambda r: dict(r, probabilities={"harmless_variation": 1, "spoken_word_error": 1}),
            lambda r: dict(r, probabilities={"harmless_variation": .5}),
            lambda r: dict(r, identity={"provider": "unbound"}),
            lambda r: dict(r, threshold=.1),
            lambda r: dict(r, model=None),
            lambda r: dict(r, differences=[]),
            lambda r: dict(r, differences=[dict(r["differences"][0], probability=float("nan"))]),
            lambda r: dict(r, differences=[dict(r["differences"][0], model="jev-unbound")]),
            lambda r: dict(r, differences=[dict(r["differences"][0], occurrences=[])]),
        ):
            judge = Judge()
            judge.response = mutate(judge.response)
            with self.subTest(response=judge.response):
                verifier, _ = self.wrapped(judge=judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertNotIn("reference_text", report)
                json.dumps(report, allow_nan=False)

    def test_exact_primary_or_exact_secondary_requested_text_needs_no_judge_call(self):
        for primary, secondary in ((SCRIPT, SCRIPT), (SPLIT, SCRIPT)):
            with self.subTest(primary=primary):
                judge = Judge()
                judge.response = AssertionError("No JEV call is needed for an exact requested transcript")
                verifier, _ = self.wrapped(primary, secondary, judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertTrue(report["passed"], report)
                self.assertEqual(report["reference_text"], SCRIPT)
                self.assertEqual(judge.calls, [])
                self.assertNotIn("text_judge", report)

    def test_judge_cannot_override_missing_added_or_unrelated_speech(self):
        for actual in ("Thank you.", SCRIPT.removesuffix(" village.") + ".", SCRIPT + " Again.",
                       SCRIPT.replace("locked door and", "locked door")):
            with self.subTest(actual=actual):
                judge = Judge()
                verifier, recognizer = self.wrapped(actual, actual, judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(recognizer.calls, [("reference", self.audio)])
                self.assertEqual(judge.calls, [])

    def test_negation_and_numeric_substitutions_are_not_accepted_as_equivalent(self):
        expected = EXPECTED + " I will not wait for seven passengers."
        for actual in (expected.replace("not", "now"), expected.replace("seven", "eleven")):
            with self.subTest(actual=actual):
                judge = Judge("different")
                verifier, _ = self.wrapped(actual, actual, judge)
                report = self.check(verifier, expected=expected)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["text_judge"]["decision"], "different")
                self.assertEqual(judge.calls, [(expected, actual, actual)])

    def test_no_or_disagreeing_secondary_cannot_be_replaced_by_judge(self):
        judge = Judge()
        verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(SPLIT), judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertFalse(report["passed"], report)
        self.assertEqual(judge.calls, [])
        verifier, _ = self.wrapped(SPLIT, SPLIT.replace("quietly", "quickly"), judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertFalse(report["passed"], report)
        self.assertEqual(judge.calls, [])

    def test_silent_audio_and_book_speech_never_reach_text_judge(self):
        judge = Judge()
        verifier, recognizer = self.wrapped(judge=judge)
        speech = self.check(verifier, expected=SCRIPT, purpose="speech")
        self.assertFalse(speech["passed"], speech)
        self.assertEqual(recognizer.calls, [("speech", self.audio)])
        self.assertEqual(judge.calls, [])
        self.write_audio(amplitude=0)
        silent = self.check(verifier, expected=SCRIPT)
        self.assertFalse(silent["passed"], silent)
        self.assertIn("silent_audio", silent["reasons"])
        self.assertEqual(judge.calls, [])

    def test_only_reference_cache_identity_depends_on_judge_policy(self):
        recognizer = IndependentVerifier(primary=SPLIT, secondary=SPLIT)
        original = ReferenceTextJudgingVerifier(recognizer, Judge())
        ref_key = validation_key(SCRIPT, original, purpose="reference")
        speech_key = validation_key(SCRIPT, original, purpose="speech")
        self.assertEqual(speech_key, validation_key(SCRIPT, recognizer, purpose="speech"))
        previous_policy = Judge()
        previous_policy.identity["policy_version"] = 2
        for judge in (Judge(threshold=.95), Judge(model="jev-updated"), previous_policy):
            verifier = ReferenceTextJudgingVerifier(recognizer, judge)
            self.assertNotEqual(validation_key(SCRIPT, verifier, purpose="reference"), ref_key)
            self.assertEqual(validation_key(SCRIPT, verifier, purpose="speech"), speech_key)
        self.assertNotEqual(validation_key(SCRIPT, recognizer, purpose="reference"), ref_key)


class JevVerifierFactoryTests(unittest.TestCase):
    def test_actual_judge_identity_redacts_credentials_and_factory_never_calls_api(self):
        settings = {"model": "/local/whisper", "language": "en"}
        secret = "fixture-private-jev-key"
        with patch("audiobook_renderer.verification.MlxWhisperVerifier", side_effect=lambda *a, **kw: fake_primary()), \
                patch("epub_audiobook.jev.JevClient._request",
                      side_effect=AssertionError("Factory construction cannot make a JEV request")) as request:
            verifier = create_verifier(settings, {"enabled": True, "api_key": secret})
            rotated = create_verifier(settings, {"enabled": True, "api_key": "fixture-new-key"})
        request.assert_not_called()
        self.assertNotIn(secret, json.dumps(verifier.reference_judge_identity))
        self.assertNotIn("api_key", json.dumps(verifier.reference_judge_identity))
        self.assertEqual(verifier.reference_judge_identity["model"], "jev-latest")
        self.assertEqual(verifier.reference_judge_identity["policy_version"], 3)
        self.assertEqual(verifier.reference_judge_identity["threshold"], .9)
        self.assertEqual(validation_key(SCRIPT, verifier, purpose="reference"),
                         validation_key(SCRIPT, rotated, purpose="reference"))

    def test_factory_is_lazy_and_disabled_jev_keeps_existing_verifier(self):
        module = ModuleType("epub_audiobook.jev")
        module.JevTranscriptJudge = Mock(return_value=Judge())
        primary, secondary = fake_primary(), fake_reference()
        settings = {"model": "/local/whisper", "language": "en", "reference_model": "/local/parakeet"}
        with patch.dict("sys.modules", {"epub_audiobook.jev": module}), \
                patch("audiobook_renderer.verification.MlxWhisperVerifier", return_value=primary), \
                patch("audiobook_renderer.reference_asr.MlxParakeetVerifier", return_value=secondary):
            disabled = create_verifier(settings, {"enabled": False, "api_key": "PRIVATE_KEY"})
            module.JevTranscriptJudge.assert_not_called()
            enabled = create_verifier(settings, {"enabled": True, "api_key": "PRIVATE_KEY"})
        self.assertEqual(enabled.judge.calls, [])
        primary.transcribe.assert_not_called()
        primary.transcribe_reference.assert_not_called()
        secondary.transcribe_reference.assert_not_called()
        self.assertEqual(enabled.identity, disabled.identity)
        self.assertNotIn("PRIVATE_KEY", json.dumps(enabled.identity))
        self.assertNotIn("PRIVATE_KEY", json.dumps(enabled.reference_judge_identity))
        self.assertFalse(hasattr(disabled, "reference_judge_identity"))

    def test_disabled_reference_variation_keeps_existing_corroboration_behavior(self):
        test = JevReferenceVerificationTests()
        test.setUp()
        try:
            report = test.check(IndependentVerifier(primary=SPLIT, secondary=SPLIT), expected=SCRIPT)
            self.assertTrue(report["passed"], report)
            self.assertNotIn("text_judge", report)
        finally:
            test.doCleanups()


if __name__ == "__main__":
    unittest.main()
