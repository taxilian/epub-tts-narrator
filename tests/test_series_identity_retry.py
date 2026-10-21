"""One bounded recheck can repair a dissenting series proposal, never force it."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.identity import (audit_identities, SERIES_SCAN, SERIES_VOTE, VOTE,
                                     SERIES_IDENTITY_GUIDANCE, SERIES_RETRY_SCOPE_GUIDANCE)
from epub_tts_narrator.pipeline import CachedRequests, fingerprint, make_units
from epub_tts_narrator.prompts import PROMPT_VERSION
from epub_tts_narrator.series import reconcile_series_cast
from test_series_identity_scan import fixture, decision, ScanRequests


def retry_fixture():
    source, cast, registry = fixture()
    source["blocks"][0]["text"] = "The warbling visitor spoke at the first door. It was the same being who had warbled behind the third door before."
    source["blocks"][1]["text"] = "Bex returned as the Dawn's baker, just as he had promised the captain."
    cast["characters"][0].update(name="First Door Visitor", aliases=[], description="The visitor with the warbling speech.")
    for index, character in enumerate(cast["characters"]):
        character["evidence"] = [{"block_id": f"b{index + 1}", "quote": source["blocks"][index]["text"]}]
    prior = registry["books"][0]["cast"]
    prior["characters"][0].update(name="First Door Whisperer", aliases=[], description="The visitor who hissed behind the first door.")
    for identity, name, description in (("prior_warbler", "Third Door Whisperer", "The visitor who warbled behind the third door."),
                                       ("prior_bex", "Bex", "The Dawn's baker, who promised to return.")):
        prior["characters"].append({"id": identity, "name": name, "aliases": [], "description": description,
                                    "voice_id": identity + "_voice", "evidence": []})
        prior["voices"].append({"id": identity + "_voice", "description": "Established distinct voice.", "traits": []})
    return source, cast, registry


class RetryRequests(ScanRequests):
    def __init__(self, *, reject_second=False, uncertain_second=False, invalid_first=False, same_proposal=False, old_issue=False):
        super().__init__()
        self.reject_second, self.uncertain_second = reject_second, uncertain_second
        self.invalid_first, self.same_proposal, self.old_issue = invalid_first, same_proposal, old_issue
        self.system_calls = []

    def response(self, payload):
        review_round = payload.get("series_review_round", 1)
        if payload["mode"] == "identity_scan" and self.old_issue:
            return {"cases": [{"kind": "uncertain", "character_ids": ["c2"], "reason": "Existing within-book uncertainty remains.",
                               "evidence": [{"block_id": "b2", "quote": "Bex returned as the Dawn's baker"}]}]}
        if payload["mode"] == "series_identity_scan":
            rows = []
            for character in payload["current_characters"]:
                candidate = "prior_bex" if character["id"] == "c2" else (
                    "prior_warbler" if review_round == 2 or self.same_proposal else "prior_mara")
                row = decision(character, "match", [candidate])
                row["reason"] = "The source describes the returning visitor's distinctive speech and the baker's promise."
                if review_round == 2 and character["id"] == "c1" and self.uncertain_second:
                    row.update(decision="uncertain", prior_character_ids=["prior_mara", "prior_warbler"])
                rows.append(row)
            return {"decisions": rows}
        if payload["mode"] == "identity_vote":
            response = super().response(payload)
            if payload["case"]["character_ids"] == ["c1"] and payload["reviewer_id"] == "identity_reviewer_2":
                if review_round == 1 and self.invalid_first:
                    response["evidence"] = [{"block_id": "b1", "quote": "fabricated citation"}]
                elif review_round == 1 or self.reject_second:
                    response.update(verdict="reject", reason="The source visitor warbles. Prior Third Door warbled, while prior First Door hissed; encounter order is insufficient.")
            return response
        return super().response(payload)

    def run(self, system, payload, validate, *, cancelled=None):
        self.system_calls.append((system, deepcopy(payload)))
        return super().run(system, payload, validate, cancelled=cancelled)


class ScannerUncertaintyRequests(RetryRequests):
    """One candidate is withheld by the scanner; all first juries approve."""
    def __init__(self, *, final_decision="match", **kwargs):
        super().__init__(**kwargs)
        self.final_decision = final_decision

    def response(self, payload):
        response = super().response(payload)
        if payload["mode"] == "series_identity_scan":
            for row in response["decisions"]:
                if row["character_id"] == "c1":
                    first = payload.get("series_review_round", 1) == 1
                    row.update(decision="uncertain" if first else self.final_decision,
                               prior_character_ids=[] if not first and self.final_decision == "new"
                               else ["prior_warbler"],
                               reason="The visitor returns, but its older doorway identity remains unresolved."
                               if first else "The source identifies the returning visitor behind the former third door.")
        return response


class SeriesIdentityRetryTests(unittest.TestCase):
    def audit(self, requests, **kwargs):
        source, cast, registry = retry_fixture()
        result = audit_identities(source, cast, requests, make_units(source), whole_book=True,
                                  series_registry=registry, workers=5, **kwargs)
        return source, cast, registry, result

    def test_dissent_triggers_one_full_scan_and_fresh_blind_reviews_for_all_final_matches(self):
        requests = RetryRequests(old_issue=True)
        source, cast, registry, (updated, issues, provenance) = self.audit(requests, review_notes="Check the visitor's distinctive vocal manner.")
        scans = [row for row in requests.calls if row["mode"] == "series_identity_scan"]
        self.assertEqual(len(scans), 2)
        self.assertNotIn("series_review_round", scans[0])
        self.assertNotIn("series_review_feedback", scans[0])
        self.assertNotIn("review_notes", scans[0])
        self.assertEqual(scans[1]["series_review_round"], 2)
        self.assertEqual(scans[1]["review_notes"], "Check the visitor's distinctive vocal manner.")
        self.assertEqual([row["id"] for row in scans[1]["current_characters"]], ["c1", "c2"])
        self.assertEqual(scans[1]["source_units"], scans[0]["source_units"])
        self.assertEqual(scans[1]["series_review_feedback"][0]["character_id"], "c1")
        juries = [row for row in requests.calls if row["mode"] == "identity_vote"]
        self.assertEqual(len(juries), 12)
        final_juries = [row for row in juries if row.get("series_review_round") == 2]
        self.assertEqual(len(final_juries), 6)
        self.assertEqual({row["case"]["character_ids"][0] for row in final_juries}, {"c1", "c2"})
        self.assertTrue(all("series_review_feedback" not in row and "review_notes" not in row for row in juries))
        self.assertLessEqual(requests.max_active, 5)
        self.assertEqual(provenance["series_review_rounds"], 2)
        history = provenance["series_review_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["series_cases"][0]["status"], "unresolved")
        self.assertEqual(history[0]["series_cases"][0]["prior_character_id"], "prior_mara")
        self.assertEqual(provenance["series_cases"][0]["prior_character_id"], "prior_warbler")
        self.assertTrue(all(row["status"] == "confirmed" for row in provenance["series_cases"]))
        self.assertEqual(len(issues), 1)
        self.assertIn("Existing within-book uncertainty", issues[0]["reason"])
        linked, _, _ = reconcile_series_cast(source, updated, registry, confirmed_identity_matches=provenance)
        self.assertEqual([row["voice_id"] for row in linked["characters"]], ["prior_warbler_voice", "prior_bex_voice"])
        self.assertTrue(all("series_identity" not in row for row in cast["characters"]))

    def test_second_round_dissent_stops_without_third_scan_and_binds_other_accepted_match(self):
        requests = RetryRequests(reject_second=True)
        source, _, registry, (updated, issues, provenance) = self.audit(requests)
        self.assertEqual(len([row for row in requests.calls if row["mode"] == "series_identity_scan"]), 2)
        self.assertEqual(len([row for row in requests.calls if row["mode"] == "identity_vote"]), 12)
        self.assertEqual(provenance["series_cases"][0]["status"], "unresolved")
        self.assertEqual(provenance["series_cases"][1]["status"], "confirmed")
        self.assertEqual(len(issues), 1)
        self.assertTrue(updated["characters"][0]["series_identity"]["review_required"])
        linked, _, _ = reconcile_series_cast(source, updated, registry, confirmed_identity_matches=provenance)
        self.assertNotEqual(linked["characters"][0]["voice_id"], "prior_warbler_voice")
        self.assertEqual(linked["characters"][1]["voice_id"], "prior_bex_voice")

    def test_second_scan_uncertainty_removes_superseded_case_and_keeps_other_match_bound(self):
        requests = RetryRequests(uncertain_second=True)
        source, _, registry, (updated, issues, provenance) = self.audit(requests)
        self.assertNotIn("series_identity", updated["characters"][0])
        self.assertEqual([row["character_ids"] for row in provenance["series_cases"]], [["c2"]])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["kind"], "series_identity")
        linked, _, _ = reconcile_series_cast(source, updated, registry, confirmed_identity_matches=provenance)
        self.assertNotEqual(linked["characters"][0]["voice_id"], "prior_warbler_voice")
        self.assertEqual(linked["characters"][1]["voice_id"], "prior_bex_voice")

    def test_invalid_quote_retries_with_fresh_votes_even_when_proposed_cast_is_identical(self):
        requests = RetryRequests(invalid_first=True, same_proposal=True)
        _, _, _, (_, issues, provenance) = self.audit(requests)
        self.assertEqual(issues, [])
        history = provenance["series_review_history"][0]["series_cases"][0]
        current = provenance["series_cases"][0]
        self.assertEqual(history["votes"][1]["verdict"], "error")
        self.assertEqual(history["votes"][0]["original_cast_fingerprint"], current["votes"][0]["original_cast_fingerprint"])
        self.assertTrue({row["request_fingerprint"] for row in history["votes"]}.isdisjoint(
                        {row["request_fingerprint"] for row in current["votes"]}))
        self.assertTrue(all(row["series_review_round"] == 2 for row in current["votes"]))

    def test_first_round_current_prompt_caches_are_reused_and_only_second_round_calls_are_new(self):
        initial_requests = RetryRequests()
        _, _, _, baseline = self.audit(initial_requests)
        class Client(RetryRequests):
            identity = {"model": "test", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                return self.response(payload)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            for system, payload in initial_requests.system_calls:
                if payload.get("series_review_round") == 2:
                    continue
                if payload["mode"] == "series_identity_scan":
                    self.assertEqual(system, SERIES_SCAN)
                elif payload["mode"] == "identity_vote":
                    self.assertEqual(system, SERIES_VOTE)
                key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION, "system": system, "payload": payload})
                Path(directory, key + ".json").write_text(json.dumps(client.response(payload)), encoding="utf-8")
            requests = CachedRequests(client, directory, repair_attempts=0)
            _, _, _, result = self.audit(requests)
            self.assertEqual(len(client.calls), 7)
            self.assertTrue(all(row.get("series_review_round") == 2 for row in client.calls))
            _, _, _, repeated = self.audit(requests)
        self.assertEqual(len(client.calls), 7)
        self.assertEqual(result, repeated)
        self.assertEqual(result, baseline)

    def test_anti_ordinal_guidance_applies_to_both_series_rounds_but_not_within_book_votes(self):
        requests = RetryRequests()
        self.audit(requests)
        for system, payload in requests.system_calls:
            if payload["mode"] == "series_identity_scan" or payload["mode"] == "identity_vote":
                self.assertIn(SERIES_IDENTITY_GUIDANCE, system)
                self.assertEqual(system.count(SERIES_IDENTITY_GUIDANCE), 1)
        self.assertNotIn(SERIES_IDENTITY_GUIDANCE, VOTE)

    def test_scanner_uncertainty_with_candidates_rechecks_once_and_stays_separate_if_unresolved(self):
        source, cast, registry = fixture()
        requests = ScanRequests({"c1": decision(cast["characters"][0], "uncertain", ["prior_mara"])})
        _, issues, provenance = audit_identities(source, cast, requests, make_units(source), series_registry=registry)
        self.assertEqual(len([row for row in requests.calls if row["mode"] == "series_identity_scan"]), 2)
        self.assertEqual(provenance["series_review_history"][0]["retry_triggers"],
                         ["scanner_uncertainty_with_candidates"])
        self.assertEqual(provenance["series_cases"], [])
        self.assertEqual(len(issues), 1)
        self.assertFalse(any(row["mode"] == "identity_vote" for row in requests.calls))

    def test_scanner_candidate_retry_can_confirm_specific_performer_without_resolving_older_issue(self):
        source, cast, registry = retry_fixture()
        old_issue = {"id": "old-door-link", "kind": "series_identity", "target": "prior_warbler",
                     "reason": "Link to another older doorway identity is unresolved.", "resolved": False}
        registry["books"][0]["issues"] = [old_issue]
        before = deepcopy((source, cast, registry))
        requests = ScannerUncertaintyRequests(old_issue=True)
        updated, issues, provenance = audit_identities(source, cast, requests, make_units(source),
            whole_book=True, series_registry=registry, workers=5)
        self.assertEqual((source, cast, registry), before)
        self.assertEqual(provenance["series_review_rounds"], 2)
        history = provenance["series_review_history"][0]
        self.assertTrue(all(row["status"] == "confirmed" for row in history["series_cases"]))
        self.assertEqual(history["series_scan"]["decisions"][0]["decision"], "uncertain")
        self.assertEqual(len(history["issues"]), 1)
        self.assertTrue(all(row["status"] == "confirmed" for row in provenance["series_cases"]))
        # Only superseded scanner issues are retired; older and within-book
        # uncertainty remains independent of this current-to-prior match.
        self.assertEqual(len(issues), 1)
        self.assertIn("Existing within-book uncertainty", issues[0]["reason"])
        linked, inherited, _ = reconcile_series_cast(source, updated, registry, confirmed_identity_matches=provenance)
        self.assertEqual([row["voice_id"] for row in linked["characters"]], ["prior_warbler_voice", "prior_bex_voice"])
        self.assertTrue(any(row["target"] == "prior_warbler" and old_issue["reason"] in row["reason"]
                            for row in inherited))
        scans = [row for row in requests.calls if row["mode"] == "series_identity_scan"]
        feedback = scans[1]["series_review_feedback"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0]["kind"], "scanner_uncertainty")
        self.assertEqual(feedback[0]["candidate_prior_character_ids"], ["prior_warbler"])
        self.assertNotIn("dissenting_reviews", feedback[0])
        juries = [row for row in requests.calls if row["mode"] == "identity_vote"]
        self.assertEqual(len(juries), 9)  # 3 existing match + 6 final-round votes
        self.assertEqual(len([row for row in juries if row.get("series_review_round") == 2]), 6)
        self.assertLessEqual(requests.max_active, 5)
        for system, payload in requests.system_calls:
            if payload.get("series_review_round") == 2:
                self.assertIn(SERIES_RETRY_SCOPE_GUIDANCE, system)
            else:
                self.assertNotIn(SERIES_RETRY_SCOPE_GUIDANCE, system)
            if payload["mode"] == "identity_vote":
                self.assertNotIn("series_review_feedback", payload)
                self.assertNotIn("review_notes", payload)

    def test_scanner_candidate_retry_still_requires_three_final_approvals(self):
        requests = ScannerUncertaintyRequests(reject_second=True)
        source, _, registry, (updated, issues, provenance) = self.audit(requests)
        self.assertEqual(provenance["series_review_rounds"], 2)
        self.assertEqual(provenance["series_cases"][0]["status"], "unresolved")
        self.assertEqual(len(issues), 1)
        linked, _, _ = reconcile_series_cast(source, updated, registry, confirmed_identity_matches=provenance)
        self.assertNotEqual(linked["characters"][0]["voice_id"], "prior_warbler_voice")
        self.assertEqual(linked["characters"][1]["voice_id"], "prior_bex_voice")

    def test_scanner_retry_may_conclude_new_without_forcing_candidate_match(self):
        requests = ScannerUncertaintyRequests(final_decision="new")
        source, _, registry, (updated, issues, provenance) = self.audit(requests)
        self.assertEqual(issues, [])
        self.assertNotIn("series_identity", updated["characters"][0])
        self.assertEqual([row["character_ids"] for row in provenance["series_cases"]], [["c2"]])
        linked, _, _ = reconcile_series_cast(source, updated, registry, confirmed_identity_matches=provenance)
        self.assertNotEqual(linked["characters"][0]["voice_id"], "prior_warbler_voice")
        self.assertEqual(len([row for row in requests.calls if row["mode"] == "series_identity_scan"]), 2)

    def test_new_and_uncertain_without_candidates_do_not_trigger_retry(self):
        for kind in ("new", "uncertain"):
            with self.subTest(kind=kind):
                source, cast, registry = fixture()
                requests = ScanRequests({"c1": decision(cast["characters"][0], kind, [])})
                _, _, provenance = audit_identities(source, cast, requests, make_units(source), series_registry=registry)
                self.assertEqual(len([row for row in requests.calls if row["mode"] == "series_identity_scan"]), 1)
                self.assertNotIn("series_review_history", provenance)

    def test_uncertainty_retry_reuses_first_round_cache_and_replays_deterministically(self):
        initial = ScannerUncertaintyRequests()
        _, _, _, expected = self.audit(initial)
        class Client(ScannerUncertaintyRequests):
            identity = {"model": "test", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                return self.response(payload)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            for system, payload in initial.system_calls:
                if payload.get("series_review_round") == 2:
                    continue
                key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                   "system": system, "payload": payload})
                Path(directory, key + ".json").write_text(json.dumps(client.response(payload)), encoding="utf-8")
            requests = CachedRequests(client, directory, repair_attempts=0)
            _, _, _, actual = self.audit(requests)
            self.assertEqual(len(client.calls), 7)
            self.assertTrue(all(row.get("series_review_round") == 2 for row in client.calls))
            _, _, _, replayed = self.audit(requests)
            self.assertEqual(len(client.calls), 7)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, replayed)


if __name__ == "__main__":
    unittest.main()
