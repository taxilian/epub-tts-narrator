"""Identity consensus must gate the cast actually used for attribution."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_audiobook.contracts import readiness, validate_document
from epub_audiobook.pipeline import analyze, make_units, merge_patch, seed_cast
from test_contracts import valid_document
from test_pipeline import FakeClient


class JuryClient(FakeClient):
    def __init__(self, *, verdicts=None, **kwargs):
        super().__init__(**kwargs)
        self.verdicts = verdicts or ["approve", "approve", "approve"]

    def reconcile(self, payload):
        response = super().reconcile(payload)
        if response["merges"]:
            # Discovery proposes the change; it cannot fold the other person's
            # aliases or roles into the survivor before independent review.
            response["characters"] = []
        return response

    def complete(self, system, payload):
        if payload.get("mode") == "identity_scan":
            self.calls.append((system, deepcopy(payload)))
            return {"cases": []}
        if payload.get("mode") == "identity_vote":
            self.calls.append((system, deepcopy(payload)))
            slot = int(payload["reviewer_id"].rsplit("_", 1)[1])
            return {"verdict": self.verdicts[slot - 1],
                    "reason": "The named speaker and the speaker description refer to the same person in this scene.",
                    "evidence": [{"block_id": "b1", "quote": "Mara said softly"}]}
        return super().complete(system, payload)


class IdentityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"
        self.source = deepcopy(valid_document()["source"])

    def run_analysis(self, client, **kwargs):
        return analyze(self.source, client, self.cache, **kwargs)

    def test_default_runs_identity_scan_before_attribution(self):
        client = JuryClient()
        document = self.run_analysis(client)
        modes = [payload.get("mode", "attribute") for _, payload in client.calls]
        self.assertEqual(modes, ["discover", "reconcile", "identity_scan", "attribute"])
        self.assertTrue(document["provenance"]["identity_consensus_enabled"])
        self.assertIn("identity_consensus", document["provenance"])
        self.assertEqual(validate_document(document), [])

    def test_proposed_merge_reaches_three_judges_before_attribution(self):
        client = JuryClient(duplicate=True)
        document = self.run_analysis(client, attribution_workers=5)
        votes = [payload for _, payload in client.calls if payload.get("mode") == "identity_vote"]
        self.assertEqual(len(votes), 3)
        self.assertEqual({vote["reviewer_id"] for vote in votes},
                         {"identity_reviewer_1", "identity_reviewer_2", "identity_reviewer_3"})
        self.assertTrue(all(len(vote["original_cast"]["characters"]) == 2 for vote in votes))
        self.assertEqual(len(document["cast"]["characters"]), 1)
        attributed = next(payload for _, payload in client.calls if "mode" not in payload)
        self.assertEqual(attributed["final_cast"], document["cast"])
        self.assertTrue(readiness(document)["ready"])
        self.assertEqual(validate_document(document), [])

    def test_dissent_keeps_both_characters_and_retains_advisory_notes(self):
        for dissent in ("reject", "uncertain"):
            with self.subTest(dissent=dissent):
                client = JuryClient(duplicate=True, verdicts=["approve", "approve", dissent])
                output = analyze(self.source, client, self.cache / dissent, attribution_workers=5)
                self.assertEqual(len(output["cast"]["characters"]), 2)
                self.assertTrue(readiness(output)["ready"])
                self.assertTrue(any(issue["kind"] == "identity_consensus" for issue in output["issues"]))
                self.assertEqual(validate_document(output), [])

    def test_resume_reuses_three_distinct_validated_votes(self):
        client = JuryClient(duplicate=True)
        original = self.run_analysis(client)
        self.assertEqual(sum(payload.get("mode") == "identity_vote" for _, payload in client.calls), 3)
        replay = JuryClient(duplicate=True)
        self.assertEqual(original, self.run_analysis(replay))
        self.assertEqual(replay.calls, [])

    def test_audit_approval_does_not_clear_other_identity_issues(self):
        class PriorIssueClient(JuryClient):
            def discover(self, payload):
                result = super().discover(payload)
                if result["characters"]:
                    result["issues"].append({"kind": "identity", "target": result["characters"][1]["id"],
                                             "reason": "A separate unresolved identity concern."})
                return result

        document = self.run_analysis(PriorIssueClient(duplicate=True))
        self.assertEqual(len(document["cast"]["characters"]), 1)
        self.assertTrue(readiness(document)["ready"])
        self.assertTrue(any(issue["reason"] == "A separate unresolved identity concern."
                            and not issue["resolved"] for issue in document["issues"]))
        concern = next(issue for issue in document["issues"] if issue["reason"] == "A separate unresolved identity concern.")
        self.assertEqual(concern["target"], document["cast"]["characters"][0]["id"])
        self.assertNotEqual(concern["target"], concern["identity_original_target"])

    def test_deferred_merge_keeps_original_records_and_validates_quotation(self):
        client = FakeClient(duplicate=True)
        available = {"characters": ["c1", "c2"], "roles": ["r1", "r2"], "voices": ["v1", "v2"]}
        payload = {"known_cast": seed_cast(), "available_ids": available,
                   "target_units": make_units(self.source)[0]}
        cast, _, _ = merge_patch(self.source, seed_cast(), client.discover(payload), available)
        proposal = client.patch(merges=[{"from_character_id": "c2", "into_character_id": "c1",
                                        "evidence": [{"block_id": "b1", "quote": "Mara"}]}])
        updated, issues, _ = merge_patch(self.source, cast, proposal, available, defer_merges=True)
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])
        proposal["merges"][0]["evidence"][0]["quote"] = "Invented quotation"
        with self.assertRaisesRegex(ValueError, "does not occur"):
            merge_patch(self.source, cast, proposal, available, defer_merges=True)

    def test_deferred_merge_cannot_sneak_through_record_upserts(self):
        client = FakeClient(duplicate=True)
        available = {"characters": ["c1", "c2"], "roles": ["r1", "r2"], "voices": ["v1", "v2"]}
        cast, _, _ = merge_patch(self.source, seed_cast(), client.discover({
            "known_cast": seed_cast(), "available_ids": available,
            "target_units": make_units(self.source)[0]}), available)
        merger = {"from_character_id": "c2", "into_character_id": "c1",
                  "evidence": [{"block_id": "b1", "quote": "Mara said softly"}]}
        for collection, index, changes in (
            ("characters", 0, {"aliases": ["the speaker"]}),
            ("roles", 1, {"character_id": "c2", "voice_id": "v2"}),
            ("voices", 1, {"description": "A conflated character voice."}),
        ):
            record = dict(deepcopy(cast[collection][index]), **changes)
            for proposal in (client.patch(merges=[merger], **{collection: [record]}),
                             client.patch(**{collection: [record]})):
                with self.subTest(collection=collection), self.assertRaisesRegex(ValueError, "must remain unchanged"):
                    merge_patch(self.source, cast, proposal, available, defer_merges=True,
                                protected_character_ids={"c1", "c2"})

    def test_invalid_consensus_flag_fails_before_any_call(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "identity_consensus"):
                client = JuryClient()
                self.run_analysis(client, identity_consensus=value)
            self.assertEqual(client.calls, [])

    def test_merge_preserves_continuity_hypotheses_and_flags_conflicting_prior_ids(self):
        client = FakeClient(duplicate=True)
        available = {"characters": ["c1", "c2"], "roles": ["r1", "r2"], "voices": ["v1", "v2"]}
        cast, _, _ = merge_patch(self.source, seed_cast(), client.discover({
            "known_cast": seed_cast(), "available_ids": available,
            "target_units": make_units(self.source)[0]}), available)
        prior = {"character_id": "earlier_mara", "basis": "context", "review_required": False,
                 "reason": "The continuing speaker.", "evidence": [{"block_id": "b1", "quote": "Mara said softly"}]}
        cast["characters"][1]["series_identity"] = deepcopy(prior)
        merger = client.patch(merges=[{"from_character_id": "c2", "into_character_id": "c1",
                                     "evidence": [{"block_id": "b1", "quote": "Mara said softly"}]}])
        updated, _, _ = merge_patch(self.source, cast, merger, available)
        self.assertEqual(updated["characters"][0]["series_identity"], prior)
        cast["characters"][0]["series_identity"] = dict(deepcopy(prior), character_id="different_mara")
        updated, issues, _ = merge_patch(self.source, cast, merger, available)
        person = updated["characters"][0]
        self.assertTrue(person["series_identity"]["review_required"])
        self.assertEqual({p["character_id"] for p in person["series_identity_candidates"]},
                         {"different_mara", "earlier_mara"})
        self.assertTrue(any(issue["kind"] == "series_identity" for issue in issues))


if __name__ == "__main__":
    unittest.main()
