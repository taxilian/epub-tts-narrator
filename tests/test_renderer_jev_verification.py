"""Whole-passage JEV decisions govern sample and chapter transcript differences."""

from copy import deepcopy
import json
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.quality import normalize_words, validation_key
from epub_tts_narrator.renderer.verification import ReferenceTextJudgingVerifier, create_verifier
import test_renderer_reference_consensus as consensus
from test_renderer_reference_consensus import EXPECTED, IndependentVerifier, PrimaryVerifier
from test_renderer_verification_factory import fake_primary, fake_reference


SCRIPT = (
    "The bloodbeast waited quietly just outside the station while the captain checked every locked "
    "door and window. A sourcevow bound the traveler to remain nearby until the night train "
    "arrived with fresh supplies for the village."
)
SPLIT = SCRIPT.replace("bloodbeast", "blood beast").replace("sourcevow", "source vow")
ALTERNATE = SPLIT.replace("quietly", "quickly")


class Judge:
    """A schema-complete judge fixture that does not know the verifier's audio."""

    def __init__(self, decision="equivalent", *, threshold=.9, model="jev-fixture"):
        self.identity = {"provider": "typesafe", "model": model, "policy_version": 8,
                         "threshold": threshold, "endpoint": "https://offline.invalid", "timeout_seconds": 10}
        self.decision = decision
        self.mutate = lambda record: record
        self.error = None
        self.calls = []
        self.response = None

    def judge(self, expected, primary, secondary=None):
        self.calls.append((expected, primary, secondary))
        if self.error is not None:
            raise self.error
        score = {"equivalent": .98, "different": .01, "uncertain": .6, "unavailable": 0}[self.decision]
        record = {"decision": self.decision, "identity": deepcopy(self.identity), "policy_version": 8,
                  "threshold": self.identity["threshold"],
                  "model": self.identity["model"] if self.decision != "unavailable" else None,
                  "probabilities": {} if self.decision == "unavailable" else {"roughly_same": score},
                  "method": "jev_spoken_passage", "comparisons": []}
        if self.decision == "unavailable":
            record["error_code"] = "transport_error"
        else:
            for label, transcript in (("primary", primary), ("secondary", secondary)):
                if transcript is None or normalize_words(transcript) == normalize_words(expected):
                    continue
                previous = next((row for row in record["comparisons"]
                                 if row["recognized_passage"] == transcript), None)
                if previous is not None:
                    previous["transcripts"].append(label)
                    continue
                record["comparisons"].append({
                    "id": f"comparison_{len(record['comparisons'])}", "requested_passage": expected,
                    "recognized_passage": transcript, "transcripts": [label],
                    "probability": score, "model": self.identity["model"], "policy_version": 8})
        self.response = self.mutate(record)
        return deepcopy(self.response)


class JevReferenceVerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    check = consensus.ReferenceConsensusTests.check

    def wrapped(self, actual=ALTERNATE, secondary=ALTERNATE, judge=None):
        recognizer = IndependentVerifier(primary=actual, secondary=secondary)
        return ReferenceTextJudgingVerifier(recognizer, judge or Judge()), recognizer

    def test_equivalent_compound_splits_keep_recognized_reference_text_and_full_audit(self):
        judge = Judge()
        verifier, recognizer = self.wrapped(SPLIT, SPLIT, judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], SPLIT)
        self.assertNotIn("text_judge", report)
        self.assertEqual(report["word_boundary_comparison"]["boundary_changes"], 2)
        self.assertGreater(report["metrics"]["wer"], 0)
        self.assertEqual(judge.calls, [])
        self.assertEqual(recognizer.calls, [("reference", self.audio)])

    def test_different_uncertain_unavailable_and_failed_judges_reject(self):
        for decision in ("different", "uncertain", "unavailable", "exception"):
            with self.subTest(decision=decision):
                judge = Judge(decision if decision != "exception" else "equivalent")
                if decision == "exception":
                    judge.error = RuntimeError("PRIVATE_API_KEY PRIVATE_REQUEST_TEXT")
                verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(ALTERNATE), judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["reasons"], ["text_mismatch"])
                self.assertNotIn("reference_text", report)
                self.assertNotIn("PRIVATE", json.dumps(report, allow_nan=False))

    def judge_report(self, judge, *, secondary=ALTERNATE):
        from epub_tts_narrator.renderer.quality import _judge_reference
        verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(SPLIT), judge)
        report = {"expected_text": SCRIPT, "transcript": SPLIT, "purpose": "reference"}
        report["passed"] = _judge_reference(verifier, report, secondary)
        return report

    def test_one_confirmed_whole_transcript_can_establish_equivalence(self):
        from epub_tts_narrator.jev import JevTranscriptJudge
        for scores, passed in (((.47, .92), True), ((.47, .60), False)):
            with self.subTest(scores=scores):
                judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
                def evaluate(state, questions):
                    self.assertEqual(set(questions), {"roughly_same"})
                    self.assertEqual(set(state), {"requested_passage", "recognized_passage"})
                    self.assertEqual(state["requested_passage"], SCRIPT)
                    score = dict(zip((SPLIT, ALTERNATE), scores))[state["recognized_passage"]]
                    return {"model": "jev-fixture", "usage": {"input_tokens": 40, "output_tokens": 2},
                            "answers": {"roughly_same": {"type": "noul", "noul": score}}}
                with patch.object(judge._client, "evaluate", side_effect=evaluate) as evaluate_mock:
                    report = self.judge_report(judge)
                self.assertEqual(evaluate_mock.call_count, 2)
                self.assertEqual(report["passed"], passed, report)
                audit = report["text_judge"]
                self.assertEqual(audit["probabilities"], {"roughly_same": max(scores)})
                self.assertEqual([row["probability"] for row in audit["comparisons"]], list(scores))
                self.assertEqual([row["recognized_passage"] for row in audit["comparisons"]], [SPLIT, ALTERNATE])
                self.assertEqual(audit["decision"], "equivalent" if passed else "uncertain")
                self.assertEqual(audit["usage"], {"input_tokens": 80, "output_tokens": 4})
                self.assertNotIn("fixture-private-key", json.dumps(audit, allow_nan=False))

    def test_approved_primary_needs_only_one_whole_passage_judgment(self):
        from epub_tts_narrator.jev import JevTranscriptJudge
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        verifier, recognizer = self.wrapped(judge=judge)
        result = {"model": "jev-fixture", "usage": {"input_tokens": 40, "output_tokens": 2},
                  "answers": {"roughly_same": {"type": "noul", "noul": .95}}}
        with patch.object(judge._client, "evaluate", return_value=result) as evaluate:
            report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[0], {"requested_passage": SCRIPT, "recognized_passage": ALTERNATE})
        self.assertEqual(recognizer.calls, [("reference", self.audio)])
        self.assertEqual(report["text_judge"]["comparisons"][0]["transcripts"], ["primary"])

    def test_high_aggregate_cannot_fabricate_a_confirmed_transcript(self):
        judge = Judge()
        def mutate(record):
            for row in record["comparisons"]:
                row["probability"] = .47
            record["probabilities"]["roughly_same"] = .98
            return record
        judge.mutate = mutate
        report = self.judge_report(judge)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["text_judge"]["error_code"], "invalid_judge_record")

    def test_later_judge_outage_retains_successful_partial_audit_and_safe_cause(self):
        from epub_tts_narrator.jev import JevTranscriptJudge, QUESTIONS
        from test_jev import Opener, response
        secret = "PRIVATE_API_KEY_MUST_NOT_APPEAR"
        judge = JevTranscriptJudge({"enabled": True, "api_key": secret})
        first = response(QUESTIONS, {"roughly_same": .94})
        judge._client._opener = Opener(first, TimeoutError(secret))
        judge._client.max_retries = 0
        report = self.judge_report(judge)
        self.assertFalse(report["passed"], report)
        self.assertNotIn("reference_text", report)
        self.assertEqual(len(judge._client._opener.calls), 2)
        audit = report["text_judge"]
        self.assertEqual(audit["decision"], "unavailable")
        self.assertEqual(audit["error_code"], "transport_error")
        self.assertEqual(audit["model"], first["model"])
        self.assertEqual(audit["probabilities"], {"roughly_same": .94})
        self.assertEqual(len(audit["comparisons"]), 1)
        self.assertEqual(audit["comparisons"][0]["requested_passage"], SCRIPT)
        self.assertEqual(audit["comparisons"][0]["recognized_passage"], SPLIT)
        self.assertEqual(audit["comparisons"][0]["probability"], .94)
        self.assertEqual(audit["usage"], first["usage"])
        self.assertNotIn(secret, json.dumps(report, allow_nan=False))

    def test_later_judge_model_change_retains_both_audits_without_approving(self):
        from epub_tts_narrator.jev import JevTranscriptJudge, QUESTIONS
        from test_jev import Opener, response
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        first = response(QUESTIONS)
        second = dict(response(QUESTIONS), model="jev-other-version")
        judge._client._opener = Opener(first, second)
        report = self.judge_report(judge)
        self.assertFalse(report["passed"], report)
        audit = report["text_judge"]
        self.assertEqual(audit["decision"], "unavailable")
        self.assertEqual(audit["error_code"], "model_changed")
        self.assertEqual([row["model"] for row in audit["comparisons"]], [first["model"], second["model"]])

    def test_malformed_partial_judgments_are_discarded_without_approving(self):
        mutations = (
            lambda r: dict(r, probabilities={}),
            lambda r: dict(r, model=None),
            lambda r: dict(r, comparisons=[]),
            lambda r: dict(r, probabilities={"roughly_same": float("nan")}),
            lambda r: dict(r, probabilities={"roughly_same": .1}),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], transcripts=[])]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], model="jev-unbound")]),
            lambda r: dict(r, usage={"input_tokens": True, "output_tokens": 2}),
            lambda r: dict(r, usage={"input_tokens": 1, "output_tokens": float("nan")}),
            lambda r: dict(r, usage={"input_tokens": 1, "output_tokens": 2, "private": "PRIVATE_DATA"}),
            lambda r: dict(r, error_code="PRIVATE_ERROR_TEXT"),
            lambda r: dict(r, error_code="model_changed"),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                judge = Judge()
                judge.mutate = lambda record: mutation(dict(record, decision="unavailable", error_code="transport_error"))
                verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(ALTERNATE), judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["text_judge"]["error_code"], "invalid_judge_record")
                self.assertNotIn("PRIVATE", json.dumps(report, allow_nan=False))

    def test_malformed_equivalence_and_inconsistent_probabilities_never_pass(self):
        mutations = (
            lambda r: None,
            lambda r: {"decision": "equivalent"},
            lambda r: dict(r, decision="approved"),
            lambda r: dict(r, decision=[]),
            lambda r: dict(r, probabilities={"roughly_same": float("nan")}),
            lambda r: dict(r, probabilities={"roughly_same": 1, "unrequested_score": 1}),
            lambda r: dict(r, probabilities={"roughly_same": .5}),
            lambda r: dict(r, identity={"provider": "unbound"}),
            lambda r: dict(r, threshold=.1),
            lambda r: dict(r, model=None),
            lambda r: dict(r, comparisons=[]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], probability=float("nan"))]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], model="jev-unbound")]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], transcripts=[])]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], transcripts=["secondary"])]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], requested_passage=SCRIPT + " Changed.")]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], recognized_passage=SPLIT)]),
            lambda r: dict(r, comparisons=[dict(r["comparisons"][0], policy_version=6)]),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                judge = Judge()
                judge.mutate = mutation
                verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(ALTERNATE), judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertNotIn("reference_text", report)
                json.dumps(report, allow_nan=False)

    def test_success_cannot_omit_the_second_distinct_transcript(self):
        judge = Judge()
        judge.mutate = lambda record: dict(record, comparisons=record["comparisons"][:1])
        report = self.judge_report(judge)
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["text_judge"]["error_code"], "invalid_judge_record")
        self.assertNotIn("reference_text", report)

    def test_exact_primary_requested_text_needs_no_judge_call(self):
        judge = Judge()
        judge.error = AssertionError("Exact requested transcript does not need a JEV call")
        verifier, _ = self.wrapped(SCRIPT, SCRIPT, judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["reference_text"], SCRIPT)
        self.assertEqual(judge.calls, [])
        self.assertNotIn("text_judge", report)

    def test_judge_decides_unrelated_or_truncated_text_without_a_distance_gate(self):
        for actual in ("Thank you.", SCRIPT.removesuffix(" village.") + "."):
            with self.subTest(actual=actual):
                judge = Judge("different")
                recognizer = PrimaryVerifier(actual)
                verifier = ReferenceTextJudgingVerifier(recognizer, judge)
                report = self.check(verifier, expected=SCRIPT)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["text_judge"]["decision"], "different")
                self.assertEqual(judge.calls, [(SCRIPT, actual, None)])

    def test_negative_judgment_of_negation_or_numeric_changes_is_respected(self):
        expected = EXPECTED + " I will not wait for seven passengers."
        for actual in (expected.replace("not", "now"), expected.replace("seven", "eleven")):
            with self.subTest(actual=actual):
                judge = Judge("different")
                verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(actual), judge)
                report = self.check(verifier, expected=expected)
                self.assertFalse(report["passed"], report)
                self.assertEqual(report["text_judge"]["decision"], "different")
                self.assertEqual(judge.calls, [(expected, actual, None)])

    def test_approved_primary_does_not_trigger_optional_secondary(self):
        judge = Judge()
        verifier = ReferenceTextJudgingVerifier(PrimaryVerifier(ALTERNATE), judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(judge.calls, [(SCRIPT, ALTERNATE, None)])
        judge.calls.clear()
        verifier, recognizer = self.wrapped(ALTERNATE, "Completely unrelated secondary recognition.", judge)
        report = self.check(verifier, expected=SCRIPT)
        self.assertTrue(report["passed"], report)
        self.assertEqual(judge.calls, [(SCRIPT, ALTERNATE, None)])
        self.assertEqual(recognizer.calls, [("reference", self.audio)])

    def test_secondary_confirmation_rescues_uncertain_primary_and_keeps_both_audits(self):
        from epub_tts_narrator.jev import JevTranscriptJudge
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        verifier, recognizer = self.wrapped("Chessed!", "Jase!", judge)
        def evaluate(state, questions):
            score = {"Chessed!": .4, "Jase!": .95}[state["recognized_passage"]]
            return {"model": "jev-fixture", "usage": {"input_tokens": 20, "output_tokens": 2},
                    "answers": {"roughly_same": {"type": "noul", "noul": score}}}
        with patch.object(judge._client, "evaluate", side_effect=evaluate) as evaluate_mock:
            report = self.check(verifier, expected="Jas!")
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["transcript"], "Chessed!")
        self.assertEqual(report["verified_transcript"], "Jase!")
        self.assertEqual(report["reference_text"], "Jase!")
        self.assertEqual(report["text_judge_source"], "secondary")
        self.assertEqual(report["text_judge"]["decision"], "equivalent")
        self.assertEqual(report["text_judge_history"][0]["recognizer"], "primary")
        self.assertEqual(report["text_judge_history"][0]["judgment"]["decision"], "uncertain")
        self.assertEqual(recognizer.calls, [("reference", self.audio), ("corroboration", self.audio)])
        self.assertEqual([call.args[0]["recognized_passage"] for call in evaluate_mock.call_args_list],
                         ["Chessed!", "Jase!"])

    def test_identical_uncertain_secondary_does_not_repeat_the_same_judgment(self):
        from epub_tts_narrator.jev import JevTranscriptJudge
        judge = JevTranscriptJudge({"enabled": True, "api_key": "fixture-private-key"})
        verifier, recognizer = self.wrapped("Chessed!", "Chessed!", judge)
        response = {"model": "jev-fixture", "usage": {"input_tokens": 20, "output_tokens": 2},
                    "answers": {"roughly_same": {"type": "noul", "noul": .4}}}
        with patch.object(judge._client, "evaluate", return_value=response) as evaluate:
            report = self.check(verifier, expected="Jas!")
        self.assertFalse(report["passed"], report)
        self.assertEqual(report["text_judge"]["decision"], "uncertain")
        self.assertEqual(recognizer.calls, [("reference", self.audio), ("corroboration", self.audio)])
        evaluate.assert_called_once()

    def test_book_speech_reaches_text_judge_but_silence_does_not(self):
        judge = Judge()
        verifier, recognizer = self.wrapped(judge=judge)
        speech = self.check(verifier, expected=SCRIPT, purpose="speech")
        self.assertTrue(speech["passed"], speech)
        self.assertEqual(recognizer.calls, [("speech", self.audio)])
        self.assertEqual(judge.calls, [(SCRIPT, ALTERNATE, None)])
        judge.calls.clear()
        self.write_audio(amplitude=0)
        silent = self.check(verifier, expected=SCRIPT)
        self.assertFalse(silent["passed"], silent)
        self.assertIn("silent_audio", silent["reasons"])
        self.assertEqual(judge.calls, [])

    def test_reference_and_speech_cache_identity_depend_on_judge_policy(self):
        recognizer = IndependentVerifier(primary=SPLIT, secondary=SPLIT)
        original = ReferenceTextJudgingVerifier(recognizer, Judge())
        ref_key = validation_key(SCRIPT, original, purpose="reference")
        speech_key = validation_key(SCRIPT, original, purpose="speech")
        self.assertNotEqual(speech_key, validation_key(SCRIPT, recognizer, purpose="speech"))
        previous_policy = Judge()
        previous_policy.identity["policy_version"] = 7
        for judge in (Judge(threshold=.95), Judge(model="jev-updated"), previous_policy):
            verifier = ReferenceTextJudgingVerifier(recognizer, judge)
            self.assertNotEqual(validation_key(SCRIPT, verifier, purpose="reference"), ref_key)
            self.assertNotEqual(validation_key(SCRIPT, verifier, purpose="speech"), speech_key)
            self.assertNotEqual(verifier.identity, original.identity)
        self.assertNotEqual(validation_key(SCRIPT, recognizer, purpose="reference"), ref_key)


class JevVerifierFactoryTests(unittest.TestCase):
    def test_actual_judge_identity_redacts_credentials_and_factory_never_calls_api(self):
        settings = {"model": "/local/whisper", "language": "en"}
        secret = "fixture-private-jev-key"
        with patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier", side_effect=lambda *a, **kw: fake_primary()), \
                patch("epub_tts_narrator.jev.JevClient._request",
                      side_effect=AssertionError("Factory construction cannot make a JEV request")) as request:
            verifier = create_verifier(settings, {"enabled": True, "api_key": secret})
            rotated = create_verifier(settings, {"enabled": True, "api_key": "fixture-new-key"})
        request.assert_not_called()
        self.assertNotIn(secret, json.dumps(verifier.reference_judge_identity))
        self.assertNotIn("api_key", json.dumps(verifier.reference_judge_identity))
        self.assertEqual(verifier.reference_judge_identity["model"], "jev-latest")
        from epub_tts_narrator.jev import POLICY_VERSION
        self.assertEqual(verifier.reference_judge_identity["policy_version"], POLICY_VERSION)
        self.assertEqual(verifier.reference_judge_identity["threshold"], .75)
        self.assertEqual(validation_key(SCRIPT, verifier, purpose="reference"),
                         validation_key(SCRIPT, rotated, purpose="reference"))

    def test_factory_is_lazy_and_disabled_jev_keeps_existing_verifier(self):
        module = ModuleType("epub_tts_narrator.jev")
        module.JevTranscriptJudge = Mock(return_value=Judge())
        primary, secondary = fake_primary(), fake_reference()
        settings = {"model": "/local/whisper", "language": "en", "reference_model": "/local/parakeet"}
        with patch.dict("sys.modules", {"epub_tts_narrator.jev": module}), \
                patch("epub_tts_narrator.renderer.verification.MlxWhisperVerifier", return_value=primary), \
                patch("epub_tts_narrator.renderer.reference_asr.MlxParakeetVerifier", return_value=secondary):
            disabled = create_verifier(settings, {"enabled": False, "api_key": "PRIVATE_KEY"})
            module.JevTranscriptJudge.assert_not_called()
            enabled = create_verifier(settings, {"enabled": True, "api_key": "PRIVATE_KEY"})
        self.assertEqual(enabled.judge.calls, [])
        primary.transcribe.assert_not_called()
        primary.transcribe_reference.assert_not_called()
        secondary.transcribe_reference.assert_not_called()
        self.assertEqual(enabled.identity, {**disabled.identity, "text_judge": enabled.reference_judge_identity})
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
