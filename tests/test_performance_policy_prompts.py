"""Exercise policy delivery and precedence contracts, not model judgment quality."""

from copy import deepcopy
import unittest

from epub_tts_narrator.attribution_review import audit_attribution
from epub_tts_narrator.attribution_review_prompts import SCAN, VOTE
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import DEFAULT_POLICY, segment
from epub_tts_narrator.prompts import ATTRIBUTE, COMMON, DISCOVER, PERFORMANCE_POLICY_GUIDANCE
from test_attribution_review import Requests, fixture


class PerformancePolicyPromptTests(unittest.TestCase):
    def test_all_performance_phases_share_explicit_conditional_policy_precedence(self):
        # Keep the hierarchy out of COMMON: identity-only requests need no
        # performance policy and must retain their independent prompt/cache keys.
        self.assertNotIn(PERFORMANCE_POLICY_GUIDANCE, COMMON)
        for prompt in (DISCOVER, ATTRIBUTE, SCAN, VOTE):
            with self.subTest(prompt=prompt[:60]):
                self.assertEqual(prompt.count(PERFORMANCE_POLICY_GUIDANCE), 1)
                self.assertIn("performance_policy takes precedence over\ncast and role descriptions", prompt)
                self.assertIn("narrator-character's own mental replies", prompt)
                self.assertIn("A different configured policy\nmust also be respected", prompt)
                self.assertIn("retrospective\nnarrative reflection remain narration", prompt)
        self.assertIn("Refine a\ncontradictory role description without changing its ID, character, or voice", DISCOVER)
        for prompt in (SCAN, VOTE):
            self.assertIn("Do not reject a source-supported, policy-correct role assignment", prompt)
            self.assertIn("do not edit the conflicting description here", prompt)

    def test_mental_reply_review_receives_policy_despite_conflicting_scope_without_mutating_cast(self):
        for thought_role in ("mara", "narrator_main"):
            with self.subTest(thought_role=thought_role):
                document = fixture(count=1)
                block = document["source"]["blocks"][0]
                frame, reply = "I, Mara, sent a reply in my mind: ", "I agree."
                block["text"] = frame + reply
                narrator = document["cast"]["roles"][0]
                narrator["character_id"] = "c1"
                narrator["description"] = (
                    "Mara's narration includes all her own directly rendered mental replies."
                    if thought_role == "mara" else
                    "Mara narrates actions only; her own mental replies always use her dialogue role."
                )
                document["segments"] = [segment(document["source"], "b1", 0, block["text"], narrator)]
                policy = deepcopy(DEFAULT_POLICY)
                if thought_role == "narrator_main":
                    policy["thoughts"] = "Use the applicable narration role for explicit thoughts and mental messages."
                document["provenance"]["performance_policy"] = policy
                before = deepcopy(document)
                evidence = [{"block_id": "b1", "quote": "I, Mara, sent a reply in my mind:"}]

                class MentalRequests(Requests):
                    def __init__(self):
                        super().__init__()
                        self.systems = []

                    def run(self, system, payload, validate, *, cancelled=None):
                        self.systems.append(system)
                        return super().run(system, payload, validate, cancelled=cancelled)

                    def response(self, payload):
                        if payload["mode"] == "attribution_review_vote":
                            return {"votes": [{"block_id": "b1", "verdict": "approve",
                                              "reason": "The explicit mental reply follows the supplied thought policy.",
                                              "evidence": evidence}]}
                        def piece(text, kind, role):
                            return {"text": text, "kind": kind, "role_id": role, "basis": "explicit",
                                    "reason": "The source separates the sending tag from the directly rendered reply.",
                                    "evidence": evidence, "candidates": [], "review_required": False,
                                    "delivery": {"direction": "", "evidence": []}}
                        return {"corrections": [{
                            "block_id": "b1", "reason": "The direct mental reply must follow performance policy.",
                            "evidence": evidence,
                            "segments": [piece(frame, "narration", "narrator_main"), piece(reply, "thought", thought_role)],
                        }]}

                requests = MentalRequests()
                result = audit_attribution(document, requests)
                self.assertEqual(document, before)
                self.assertEqual(result["cast"], before["cast"])
                self.assertEqual(result["source"], before["source"])
                self.assertEqual(result["provenance"]["performance_policy"], policy)
                self.assertEqual(validate_document(result), [])
                self.assertEqual([(item["kind"], item["role_id"]) for item in result["segments"]],
                                 [("narration", "narrator_main"), ("thought", thought_role)])
                self.assertEqual("".join(item["text"] for item in result["segments"]), block["text"])
                self.assertEqual(len(requests.calls), 4)
                self.assertTrue(all(payload["performance_policy"] == policy for payload in requests.calls))
                self.assertTrue(all(PERFORMANCE_POLICY_GUIDANCE in system for system in requests.systems))
                self.assertEqual(len({row["reviewer_id"] for row in requests.calls if "reviewer_id" in row}), 3)
                self.assertTrue(all("votes" not in payload for payload in requests.calls))
                self.assertEqual(result["provenance"]["attribution_review"]["applied_blocks"], 1)
