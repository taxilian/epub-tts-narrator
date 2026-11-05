"""Adaptive collision resolution stays bounded and never merges or splits by name."""

from concurrent.futures import CancelledError
from copy import deepcopy
from json import dumps
from pathlib import Path
from types import SimpleNamespace
import tempfile
from threading import Lock
import unittest

from epub_tts_narrator.identity_collisions import (COLLISION_VOTE_SYSTEM, COLLISION_VERSION,
                                                   resolve_collisions)
from epub_tts_narrator.identity_prompts import LENSES
from epub_tts_narrator.pipeline import fingerprint, merge_patch
from epub_tts_narrator.provider import ResponseCapacityError
from test_identity import fixture


def collision_fixture(count, name="Trask"):
    return fixture(tuple([name] * count))


class BatchRequests:
    """Offline fake for identity_collision_votes batch requests."""

    def __init__(self, verdicts=None, *, limit=200000, fail=None, defect=None, capacity_over=None,
                 cache_dir=None):
        self.verdicts = dict(verdicts or {})
        self.fail = fail
        self.defect = defect
        self.capacity_over = capacity_over
        self.calls = []
        self.lock = Lock()
        self.client = SimpleNamespace(identity={"model": "offline-collision"})
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / f"identity-collision-{id(self):x}"
        self.max_request_chars = limit

    def response(self, payload):
        votes = []
        for entry in payload["cases"]:
            votes.append({"case_id": entry["case_id"],
                          "verdict": self.verdicts.get(tuple(entry["character_ids"]), "approve"),
                          "reason": "The source follows one person across both records beyond the shared name.",
                          "evidence": deepcopy(entry["evidence"][:1])})
        result = {"votes": votes}
        if self.defect is not None:
            result = self.defect(payload, result)
        return result

    def run(self, system, payload, validate, *, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Fixture request cancelled")
        with self.lock:
            self.calls.append(deepcopy(payload))
        if self.fail is not None and self.fail(payload) is not None:
            raise self.fail(payload)
        if self.capacity_over is not None and len(payload["cases"]) > self.capacity_over:
            raise ResponseCapacityError("The model output was truncated. Increase the output token limit or reduce the input.")
        return validate(self.response(payload))


def negative_case(kind, ids, status):
    row = {"kind": kind, "character_ids": list(ids), "reason": "Source-established constraint.",
           "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}
    if status is not None:
        row["status"] = status
    return row


class CollisionResolutionTests(unittest.TestCase):
    def test_thirty_two_same_name_duplicates_merge_in_bounded_waves_with_few_batched_requests(self):
        source, cast = collision_fixture(32)
        requests = BatchRequests()
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        self.assertEqual(len(updated["characters"]), 1)
        self.assertEqual(updated["characters"][0]["id"], "c1")
        self.assertEqual(issues, [])
        records = provenance["cases"]
        self.assertEqual(len(records), 31)  # K - 1 nominated cases, never 496 pairwise screens.
        self.assertTrue(all(record["status"] == "applied" for record in records))
        self.assertEqual([wave["request_count"] for wave in provenance["waves"]], [3, 3, 3, 3, 3, 0])
        self.assertEqual(provenance["request_count"], 15)
        call_fingerprints = {fingerprint(payload) for payload in requests.calls}
        for record in records:
            self.assertEqual(len(record["votes"]), 3)
            self.assertEqual({vote["reviewer_id"] for vote in record["votes"]}, {row[0] for row in LENSES})
            self.assertEqual(len({vote["lens"] for vote in record["votes"]}), 3)
            self.assertTrue(all(vote["verdict"] == "approve" for vote in record["votes"]))
            for vote in record["votes"]:
                self.assertIn(vote["request_fingerprint"], call_fingerprints)
        self.assertEqual(provenance["state"]["lineage_members"],
                         {"c1": sorted(f"c{index}" for index in range(1, 33))})
        self.assertEqual(provenance["state"]["retired_ids"], [])

    def test_rejected_wave_pair_retires_both_participants_and_never_tries_new_partners(self):
        source, cast = collision_fixture(4)
        requests = BatchRequests(verdicts={("c1", "c2"): "reject"})
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        records = provenance["cases"]
        self.assertEqual([record["character_ids"] for record in records], [["c1", "c2"], ["c3", "c4"]])
        self.assertEqual([record["status"] for record in records], ["unresolved", "applied"])
        self.assertEqual(sorted(row["id"] for row in updated["characters"]), ["c1", "c2", "c3"])
        self.assertEqual(next(row for row in updated["characters"] if row["id"] == "c1"), cast["characters"][0])
        self.assertEqual(next(row for row in updated["characters"] if row["id"] == "c2"), cast["characters"][1])
        self.assertEqual(provenance["state"]["retired_ids"], ["c1", "c2"])
        self.assertEqual(provenance["state"]["unresolved_pairs"], [["c1", "c2"]])
        self.assertEqual(provenance["state"]["lineage_members"], {"c3": ["c3", "c4"]})
        self.assertEqual(len(issues), 1)
        self.assertIn("Three reviewers did not unanimously approve", issues[0]["reason"])
        self.assertEqual(provenance["waves"][1]["case_ids"], [])
        self.assertEqual(provenance["waves"][1]["live_lineages"], 1)

    def test_confirmed_separate_negative_blocks_transitive_merge_after_an_approved_merge(self):
        source, cast = collision_fixture(3)
        source_cases = [negative_case("keep_separate", ("c2", "c3"), "confirmed_separate")]
        requests = BatchRequests()
        updated, issues, provenance = resolve_collisions(source, cast, requests, source_cases=source_cases)
        records = provenance["cases"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["character_ids"], ["c1", "c2"])
        self.assertEqual(records[0]["status"], "applied")
        self.assertEqual(sorted(row["id"] for row in updated["characters"]), ["c1", "c3"])
        self.assertEqual(issues, [])
        self.assertEqual(provenance["state"]["forbidden_pairs"], [["c2", "c3"]])
        self.assertEqual(provenance["state"]["lineage_members"], {"c1": ["c1", "c2"]})
        # The A-C hypothesis was considered against two live lineages and refused:
        # the approved A-B merge inherited B's negative constraint with C.
        self.assertEqual(len(provenance["waves"]), 2)
        self.assertEqual(provenance["waves"][1]["live_lineages"], 2)
        self.assertEqual(provenance["waves"][1]["case_ids"], [])

    def test_unapproved_pairs_stay_unresolved_without_exhaustive_certification(self):
        source, cast = collision_fixture(2)
        requests = BatchRequests(verdicts={("c1", "c2"): "uncertain"})
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        self.assertEqual(updated, cast)
        self.assertEqual(provenance["cases"][0]["status"], "unresolved")
        self.assertEqual({vote["verdict"] for vote in provenance["cases"][0]["votes"]}, {"uncertain"})
        self.assertEqual(provenance["state"]["unresolved_pairs"], [["c1", "c2"]])
        self.assertEqual(provenance["disclosure"]["pair_certification"], "non_exhaustive")
        self.assertIn("does not claim", provenance["disclosure"]["note"])
        self.assertIn("not proven separate", provenance["disclosure"]["note"])

    def test_prior_state_round_trip_prevents_renomination(self):
        source, cast = collision_fixture(4)
        first = BatchRequests(verdicts={("c1", "c2"): "reject"})
        updated1, _, provenance1 = resolve_collisions(source, cast, first)
        for prior in (provenance1, provenance1["state"]):
            with self.subTest(prior="provenance" if prior is provenance1 else "state"):
                second = BatchRequests()
                updated2, issues2, provenance2 = resolve_collisions(source, updated1, second, prior=prior)
                self.assertEqual(second.calls, [])
                self.assertEqual(updated2, updated1)
                self.assertEqual(issues2, [])
                self.assertEqual(provenance2["cases"], [])
                self.assertEqual(provenance2["state"]["retired_ids"], ["c1", "c2"])

    def test_source_case_negatives_block_same_name_pairs_without_a_jury(self):
        for status in ("confirmed_separate", "unresolved", "invalid_proposal", "overlapping_cases", None):
            with self.subTest(status=status):
                source, cast = collision_fixture(2)
                requests = BatchRequests()
                updated, issues, provenance = resolve_collisions(
                    source, cast, requests, source_cases=[negative_case("merge", ("c1", "c2"), status)])
                self.assertEqual(provenance["cases"], [])
                self.assertEqual(requests.calls, [])
                self.assertEqual(updated, cast)
                self.assertEqual(issues, [])
                self.assertEqual(provenance["state"]["forbidden_pairs"], [["c1", "c2"]])

    def test_unresolved_split_or_uncertain_freezes_the_affected_lineage(self):
        for kind in ("split", "uncertain"):
            with self.subTest(kind=kind):
                source, cast = collision_fixture(4)
                requests = BatchRequests()
                updated, issues, provenance = resolve_collisions(
                    source, cast, requests, source_cases=[negative_case(kind, ("c1",), "unresolved")])
                nominated = [tuple(record["character_ids"]) for record in provenance["cases"]]
                self.assertEqual(nominated, [("c2", "c3"), ("c2", "c4")])
                self.assertTrue(all("c1" not in pair for pair in nominated))
                self.assertIn("c1", provenance["state"]["frozen_ids"])
                self.assertEqual(sorted(row["id"] for row in updated["characters"]), ["c1", "c2"])
                self.assertEqual(provenance["state"]["lineage_members"], {"c2": ["c2", "c3", "c4"]})

    def test_indivisible_case_budget_stays_unresolved_and_never_approves(self):
        source, cast = collision_fixture(4)
        requests = BatchRequests(limit=400)
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        records = provenance["cases"]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["status"] == "unresolved" for record in records))
        self.assertTrue(all(vote["verdict"] == "error" for record in records for vote in record["votes"]))
        self.assertEqual(requests.calls, [])
        self.assertEqual(provenance["request_count"], 0)
        self.assertEqual(updated, cast)
        self.assertEqual(provenance["state"]["retired_ids"], ["c1", "c2", "c3", "c4"])
        self.assertEqual(len(issues), len(records))

    def test_fatal_provider_failure_propagates_without_fanout(self):
        source, cast = collision_fixture(8)
        failure = RuntimeError("Provider rejected the revoked key")
        requests = BatchRequests(fail=lambda payload: failure)
        with self.assertRaises(RuntimeError) as raised:
            resolve_collisions(source, cast, requests, workers=1)
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(requests.calls), 1)

    def test_provider_capacity_subdivides_batches_and_still_judges_every_case(self):
        source, cast = collision_fixture(8)
        with tempfile.TemporaryDirectory() as directory:
            requests = BatchRequests(capacity_over=2, cache_dir=Path(directory))
            updated, issues, provenance = resolve_collisions(source, cast, requests)
            markers = list((Path(directory) / "adaptive-subdivisions").glob("*.json"))
        self.assertEqual([record["status"] for record in provenance["cases"]], ["applied"] * 7)
        self.assertTrue(all({vote["verdict"] for vote in record["votes"]} == {"approve"}
                            for record in provenance["cases"]))
        self.assertEqual(provenance["request_count"], 12)  # Wave 1 subdivides to two 2-case leaves; waves 2-3 fit.
        self.assertTrue(markers)
        self.assertEqual(len(updated["characters"]), 1)

    def test_payloads_stay_pair_scoped_with_truthful_proposed_fingerprints(self):
        source, cast = collision_fixture(4)
        requests = BatchRequests()
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        wave1_calls = [payload for payload in requests.calls
                       if payload["original_cast_fingerprint"] == fingerprint(cast)]
        self.assertEqual(len(wave1_calls), 3)
        for payload in wave1_calls:
            self.assertEqual(payload["mode"], "identity_collision_votes")
            self.assertEqual(payload["source_fingerprint"], fingerprint(source))
            self.assertEqual(list(payload)[-2:], ["reviewer_id", "lens"])
            serialized = dumps(payload, ensure_ascii=False)
            self.assertTrue(serialized.startswith('{"mode": "identity_collision_votes"'))
            for entry in payload["cases"]:
                pair = set(entry["character_ids"])
                self.assertEqual({row["id"] for row in entry["original_cast"]["characters"]}, pair)
                self.assertFalse(entry["cast_scope"]["complete"])
                self.assertEqual(entry["cast_scope"]["included_ids"], sorted(pair))
                self.assertEqual([row["id"] for row in entry["proposed_cast"]["characters"]],
                                 [entry["character_ids"][0]])
                expected = merge_patch(source, cast, entry["patch"],
                                       {"characters": [], "roles": [], "voices": []})[0]
                self.assertEqual(entry["proposed_cast_scope"]["full_cast_fingerprint"], fingerprint(expected))
                self.assertFalse(entry["proposed_cast_scope"]["complete"])
                self.assertEqual(entry["source_scope"], "pair_evidence_and_chapter_neighbors")
                # Complete pair evidence: cited blocks plus same-chapter neighbors.
                self.assertEqual({unit["block_id"] for unit in entry["source_units"]}, {"b1", "b2"})
        self.assertEqual(len({dumps(payload, ensure_ascii=False).partition(', "reviewer_id":')[0]
                              for payload in wave1_calls}), 1)
        self.assertEqual({payload["reviewer_id"] for payload in wave1_calls}, {row[0] for row in LENSES})
        self.assertEqual({payload["lens"] for payload in wave1_calls}, {row[1] for row in LENSES})

    def test_malformed_batch_response_leaves_cases_unresolved_never_approved(self):
        source, cast = collision_fixture(4)
        requests = BatchRequests(defect=lambda payload, result: {"votes": result["votes"][:-1]})
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        records = provenance["cases"]
        self.assertEqual([record["status"] for record in records], ["unresolved", "unresolved"])
        self.assertTrue(all(vote["verdict"] == "error" for record in records for vote in record["votes"]))
        self.assertEqual(provenance["state"]["retired_ids"], ["c1", "c2", "c3", "c4"])
        self.assertEqual(updated, cast)
        self.assertEqual(len(issues), 2)

    def test_input_validation_rejects_invalid_workers_prior_and_source_cases(self):
        source, cast = collision_fixture(2)
        requests = BatchRequests()
        for workers in (0, 9, "2", True, None):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "integer from 1 to 8"):
                    resolve_collisions(source, cast, requests, workers=workers)
        for prior in ("state", 5, {"version": COLLISION_VERSION}, {"state": {"version": COLLISION_VERSION}}):
            with self.subTest(prior=prior):
                with self.assertRaisesRegex(ValueError, "prior"):
                    resolve_collisions(source, cast, requests, prior=prior)
        with self.assertRaisesRegex(ValueError, "case records"):
            resolve_collisions(source, cast, requests, source_cases=["x"])
        with self.assertRaisesRegex(ValueError, "character IDs"):
            resolve_collisions(source, cast, requests, source_cases=[{}])
        with self.assertRaisesRegex(ValueError, "original_cast"):
            resolve_collisions(source, cast, requests, original_cast={"characters": "no"})

    def test_distinct_names_nominate_nothing(self):
        source, cast = fixture(("Mara", "Zephyrine", "Bolt"))
        requests = BatchRequests()
        updated, issues, provenance = resolve_collisions(source, cast, requests)
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])
        self.assertEqual(provenance["cases"], [])
        self.assertEqual(provenance["request_count"], 0)
        self.assertEqual(provenance["disclosure"]["pair_certification"], "non_exhaustive")
        self.assertEqual(provenance["state"]["forbidden_pairs"], [])


if __name__ == "__main__":
    unittest.main()
