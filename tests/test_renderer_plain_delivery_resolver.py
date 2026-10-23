"""The recovery planner can change performance without waiving word coverage."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.speech_resolver import (
    PROMPT, SpeechResolver, validate_resolution,
)
from test_renderer_speech_resolver import Client, failure


class PlainDeliveryResolverTests(unittest.TestCase):
    def setUp(self):
        self.expected = "Arrgh! The pain!"
        self.failures = [failure(
            "Oh", tts_text="Aargh! The pain!",
            generation={"synthesis_text": "[crying] Aargh! The pain!",
                        "direction": "Cry out in pain.", "applied_control": ""})]
        self.client = Client("plain_delivery")
        self.resolver = SpeechResolver(self.client)

    def test_failed_respelling_and_actual_directions_reach_planner(self):
        audit = self.resolver.resolve(self.expected, self.failures, can_plain_delivery=True)
        proposal = validate_resolution(audit, self.expected, self.failures, self.resolver.identity,
                                       can_plain_delivery=True)
        self.assertEqual(proposal["action"], "plain_delivery")
        self.assertEqual(proposal["edits"], [])
        self.assertEqual(proposal["validated_pairs"], [])
        self.assertNotIn("acceptance_transcript", proposal)
        self.assertNotIn("tts_text", proposal)
        self.assertEqual(audit["reviews"], [])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0][1]["failures"], self.failures)
        self.assertIn("crying", self.client.calls[0][0])
        self.assertIn("never accepts the failed audio", self.client.calls[0][0])

    def test_caller_must_enable_delivery_control_and_cannot_be_overridden(self):
        blocked = self.resolver.resolve(self.expected, self.failures)
        self.assertEqual(blocked["decision"], "blocked")
        self.assertEqual(blocked["error_code"], "invalid_proposal")
        self.assertNotIn("plain_delivery", blocked["payload"]["permitted_actions"])
        allowed = self.resolver.resolve(self.expected, self.failures, can_plain_delivery=True)
        self.assertIsNone(validate_resolution(allowed, self.expected, self.failures,
                                              self.resolver.identity, can_plain_delivery=False))
        self.assertIsNotNone(validate_resolution(allowed, self.expected, self.failures,
                                                 self.resolver.identity, can_plain_delivery=True))

    def test_new_take_can_change_style_after_acoustic_failure(self):
        evidence = [failure("", reasons=["internal_silence_stall"],
                            generation=self.failures[0]["generation"])]
        audit = self.resolver.resolve(self.expected, evidence, can_plain_delivery=True)
        self.assertEqual(audit["decision"], "resolved")
        self.assertNotIn("accept", audit["payload"]["permitted_actions"])
        self.assertEqual(validate_resolution(audit, self.expected, evidence, self.resolver.identity,
                                             can_plain_delivery=True)["action"], "plain_delivery")

    def test_plain_delivery_never_allows_text_edits_or_acceptance_of_missing_words(self):
        self.client.proposal["edits"] = [{"start": 0, "end": 5, "source": "Arrgh",
                                          "replacement": "Oh"}]
        audit = self.resolver.resolve(self.expected, self.failures, can_plain_delivery=True)
        self.assertEqual(audit["error_code"], "invalid_proposal")
        self.client.proposal = {"action": "accept", "edits": [],
                                "reason": "Assume a groan fills all absent words."}
        audit = self.resolver.resolve(self.expected, self.failures, can_plain_delivery=True)
        self.assertEqual(audit["error_code"], "invalid_proposal")
        self.assertIn("complete word coverage", audit["validation_error"])

    def test_persisted_capability_can_be_replayed_without_engine_context(self):
        audit = self.resolver.resolve(self.expected, self.failures, can_plain_delivery=True)
        self.assertEqual(validate_resolution(audit, self.expected, self.failures,
                                             self.resolver.identity)["action"], "plain_delivery")
        for mutate in (lambda row: row["payload"]["permitted_actions"].remove("plain_delivery"),
                       lambda row: row["reviews"].append({"reviewer": 1}),
                       lambda row: row["payload"]["failures"][0]["generation"].update(direction="")):
            changed = deepcopy(audit)
            mutate(changed)
            self.assertIsNone(validate_resolution(changed, self.expected, self.failures,
                                                  self.resolver.identity))

    def test_rewrite_audit_retains_enabled_delivery_capability_on_replay(self):
        client = Client("rewrite", edits=[{"start": 0, "end": 5, "source": "Xzqra",
                                          "replacement": "Zekra"}])
        resolver = SpeechResolver(client)
        expected, evidence = "Xzqra waited.", [failure("Zekra waited.")]
        audit = resolver.resolve(expected, evidence, can_plain_delivery=True)
        proposal = validate_resolution(audit, expected, evidence, resolver.identity)
        self.assertEqual(proposal["tts_text"], "Zekra waited.")
        self.assertEqual(len(audit["reviews"]), 3)
        self.assertIsNone(validate_resolution(audit, expected, evidence, resolver.identity,
                                              can_plain_delivery=False))

    def test_invalid_capability_type_never_calls_model(self):
        for value in (1, "true", [], None):
            with self.subTest(value=value):
                audit = self.resolver.resolve(self.expected, self.failures, can_plain_delivery=value)
                self.assertEqual(audit["error_code"], "invalid_request")
        self.assertEqual(self.client.calls, [])

    def test_fixed_previous_stutter_rewrite_remains_valid(self):
        saved = json.loads((Path(__file__).parent / "fixtures/speech-rewrite-policy-v4.json").read_text())
        audit = saved["audit"]
        proposal = validate_resolution(audit, saved["expected_text"], saved["failures"], audit["identity"])
        self.assertEqual(proposal["tts_text"], "Buh-but Mother")
        self.assertNotEqual(audit["identity"]["planner_prompt_sha256"], digest(PROMPT))
        self.assertIsNone(validate_resolution(audit, saved["expected_text"], saved["failures"],
                                              audit["identity"], can_plain_delivery=True))
        for mutate in (lambda row: row["proposal"].update(action="plain_delivery", edits=[]),
                       lambda row: row["reviews"].pop(),
                       lambda row: row["payload"]["permitted_actions"].append("plain_delivery")):
            changed = deepcopy(audit)
            mutate(changed)
            self.assertIsNone(validate_resolution(changed, saved["expected_text"], saved["failures"],
                                                  changed["identity"]))


if __name__ == "__main__":
    unittest.main()
