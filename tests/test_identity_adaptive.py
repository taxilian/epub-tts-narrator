"""Adaptive identity scans cover every candidate without hidden evidence trimming."""

from copy import deepcopy
from json import dumps
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from epub_tts_narrator.cast_scope import CAST_SCOPE_GUIDANCE, scoped_cast
from epub_tts_narrator.identity import (_scan_series_identities, _series_scan_payload,
                                        _series_scan_requests, audit_identities)
from epub_tts_narrator.identity_prompts import SCAN, SERIES_SCAN, VOTE
from epub_tts_narrator.pipeline import fingerprint, make_units
from test_identity import Requests, case, fixture
from test_series_identity_scan import ScanRequests as SeriesRequests


def all_units(source):
    return [unit for batch in make_units(source) for unit in batch]


def payload_size(system, payload):
    return len(system) + len(dumps(payload, ensure_ascii=False))


def budget_floor(size):
    """Smallest request limit whose repair-reserved series budget still admits ``size``."""
    limit = size
    while limit - min(4000, limit // 10) < size:
        limit += 1
    return limit


class Bounded(Requests):
    """The test Requests fake plus the attributes the shared adaptive runner needs."""

    def __init__(self, cases=(), limit=200000, cache=None):
        super().__init__(cases)
        self.client = SimpleNamespace(identity={"model": "offline-adaptive"})
        self.cache_dir = Path(cache or tempfile.gettempdir()) / f"identity-adaptive-{id(self):x}"
        self.max_request_chars = limit

    def response(self, payload):
        if payload["mode"] == "identity_collision_votes":
            return {"votes": [{"case_id": row["case_id"], "verdict": "approve",
                                "reason": "The narrative follows the same captain across both records.",
                                "evidence": [{"block_id": "b1", "quote": "The captain was called Red."}]}
                               for row in payload["cases"]]}
        return super().response(payload)


def wide_fixture(count=8):
    source, cast = fixture(tuple(f"Person{index}" for index in range(1, count + 1)))
    blocks = source["blocks"]
    for index, character in enumerate(cast["characters"], 1):
        block = blocks[(index - 1) % len(blocks)]
        character["evidence"] = [{"block_id": block["id"], "quote": block["text"][:20]}]
    return source, cast


class RegionScanRequests(Bounded):
    """Only report the case from leaves where its citation is actually visible."""

    def response(self, payload):
        result = super().response(payload)
        if payload["mode"] == "identity_scan":
            visible = {unit["block_id"] for unit in payload["source_units"]}
            return {"cases": deepcopy(self.cases)} if "b1" in visible else {"cases": []}
        return result


class AdaptiveIdentityTests(unittest.TestCase):
    def measure_full_leaf(self, source, cast, cases=()):
        with tempfile.TemporaryDirectory() as directory:
            probe = Bounded(cases, cache=directory)
            audit_identities(source, cast, probe, make_units(source), adaptive=True)
            calls = deepcopy(probe.calls)
        return next(payload for payload in calls if payload["mode"] == "identity_scan")

    def test_adaptive_scans_partition_the_complete_source_exactly_once(self):
        source, cast = fixture()
        units = all_units(source)
        full = self.measure_full_leaf(source, cast)
        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded(limit=payload_size(SCAN + CAST_SCOPE_GUIDANCE, full) - 1, cache=directory)
            updated, issues, provenance = audit_identities(source, cast, requests, make_units(source), adaptive=True)
            calls = deepcopy(requests.calls)
        scans = [payload for payload in calls if payload["mode"] == "identity_scan"]
        self.assertGreater(len(scans), 1)
        self.assertEqual([unit for payload in scans for unit in payload["source_units"]], units)
        self.assertTrue(all(payload["source_scope"] == "source_batch" for payload in scans))
        self.assertTrue(all(payload["cast_scope"]["complete"] for payload in scans))
        self.assertTrue(all("cast_scope" in payload for payload in scans))
        self.assertEqual([entry["index"] for entry in provenance["scans"]], list(range(len(scans))))
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])

    def test_scoped_leaves_keep_complete_records_and_cover_every_candidate(self):
        source, cast = wide_fixture()
        full = self.measure_full_leaf(source, cast)
        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded(limit=payload_size(SCAN + CAST_SCOPE_GUIDANCE, full) - 700, cache=directory)
            updated, issues, provenance = audit_identities(source, cast, requests, make_units(source), adaptive=True)
            calls = deepcopy(requests.calls)
        scans = [payload for payload in calls if payload["mode"] == "identity_scan"]
        scoped = [payload for payload in scans if not payload["cast_scope"]["complete"]]
        self.assertTrue(scoped)
        originals = {row["id"]: row for row in cast["characters"]}
        for payload in scoped:
            scope = payload["cast_scope"]
            self.assertEqual(scope["full_cast_fingerprint"], fingerprint(cast))
            view = payload["original_cast"]
            self.assertEqual(len(view["characters"]), len(scope["included_ids"]))
            for row in view["characters"]:
                self.assertEqual(row, originals[row["id"]])
            self.assertEqual(len(view["roles"]), len(cast["roles"]) - scope["omitted_counts"]["roles"])
            self.assertEqual(len(view["voices"]), len(cast["voices"]) - scope["omitted_counts"]["voices"])
        covered = {character for payload in scans for character in payload["cast_scope"]["included_ids"]}
        self.assertEqual(covered, {row["id"] for row in cast["characters"]})
        self.assertFalse(any(entry["candidate_character_ids"] for entry in provenance["scans"]))
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])

    def test_invisible_candidates_get_an_additional_group_scan_without_trimming(self):
        source, cast = wide_fixture()
        cast["characters"].append({"id": "c9", "name": "Quillon", "aliases": [],
            "description": "Never mentioned anywhere in the book.", "voice_id": "v9", "evidence": []})
        cast["roles"].append({"id": "r9", "kind": "dialogue", "character_id": "c9",
                              "voice_id": "v9", "description": "Character dialogue."})
        cast["voices"].append({"id": "v9", "description": "A distinct clear voice.", "traits": []})
        full = self.measure_full_leaf(source, cast)
        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded(limit=payload_size(SCAN + CAST_SCOPE_GUIDANCE, full) - 700, cache=directory)
            updated, issues, provenance = audit_identities(source, cast, requests, make_units(source), adaptive=True)
            calls = deepcopy(requests.calls)
        scans = [payload for payload in calls if payload["mode"] == "identity_scan"]
        group_entries = [entry for entry in provenance["scans"] if entry["candidate_character_ids"]]
        self.assertTrue(group_entries)
        self.assertEqual(sorted({character for entry in group_entries
                                 for character in entry["candidate_character_ids"]}), ["c9"])
        covered = {character for payload in scans for character in payload["cast_scope"]["included_ids"]}
        self.assertEqual(covered, {row["id"] for row in cast["characters"]})
        quillon_scans = [payload for payload in scans if "c9" in payload["cast_scope"]["included_ids"]]
        self.assertTrue(quillon_scans)
        original = next(row for row in cast["characters"] if row["id"] == "c9")
        for payload in quillon_scans:
            supplied = next(row for row in payload["original_cast"]["characters"] if row["id"] == "c9")
            self.assertEqual(supplied, original)
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])

    def test_case_scan_indexes_reference_actual_adaptive_leaves(self):
        source, cast = fixture()
        units = all_units(source)
        with tempfile.TemporaryDirectory() as directory:
            requests = RegionScanRequests([case()], cache=directory)
            updated, issues, provenance = audit_identities(source, cast, requests, make_units(source),
                                                           adaptive=True, scan_regions=[[units[0]], [units[1]]])
            calls = deepcopy(requests.calls)
        scans = [payload for payload in calls if payload["mode"] == "identity_scan"]
        self.assertEqual([payload["source_units"] for payload in scans], [[units[0]], [units[1]]])
        self.assertEqual(provenance["cases"][0]["scan_indexes"], [0])
        votes = [payload for payload in calls if payload["mode"] == "identity_vote"]
        self.assertEqual(len(votes), 3)
        self.assertTrue(all({unit["block_id"] for unit in payload["source_units"]} == {"b1", "b2"}
                            for payload in votes))
        self.assertEqual(len(updated["characters"]), 1)
        self.assertEqual(issues, [])

    def test_same_case_found_in_multiple_leaves_merges_their_leaf_indexes(self):
        source, cast = fixture()
        units = all_units(source)
        local = case()

        class EveryLeafRequests(Requests):
            def response(self, payload):
                if payload["mode"] == "identity_scan":
                    found = deepcopy(local)
                    unit = payload["source_units"][0]
                    found["evidence"] = [{"block_id": unit["block_id"], "quote": unit["text"][:20]}]
                    return {"cases": [found]}
                return super().response(payload)

        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded([local], cache=directory)
            requests.__class__ = EveryLeafRequests
            _, _, provenance = audit_identities(source, cast, requests, make_units(source),
                                                adaptive=True, scan_regions=[[units[0]], [units[1]]])
        self.assertEqual(len(provenance["scans"]), 2)
        self.assertEqual(len(provenance["cases"]), 1)
        self.assertEqual(provenance["cases"][0]["scan_indexes"], [0, 1])

    def test_cross_partition_name_resemblance_reaches_one_blind_jury(self):
        source, cast = fixture(("Mara", "Mara"))
        cast["characters"][0]["evidence"] = [{"block_id": "b1", "quote": "Mara, the captain"}]
        cast["characters"][1]["evidence"] = [{"block_id": "b2", "quote": "Red returned after a year"}]
        units = all_units(source)
        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded(cache=directory)
            updated, issues, provenance = audit_identities(source, cast, requests, make_units(source),
                                                           adaptive=True, scan_regions=[[units[0]], [units[1]]])
            calls = deepcopy(requests.calls)
        self.assertEqual(len(updated["characters"]), 1)
        self.assertEqual(issues, [])
        self.assertEqual(provenance["cases"][0]["origin"], "identity_collision_screen")
        self.assertEqual(provenance["cases"][0]["status"], "applied")
        votes = [payload for payload in calls if payload["mode"] == "identity_collision_votes"]
        self.assertEqual(len(votes), 3)
        self.assertEqual({payload["reviewer_id"] for payload in votes},
                         {"identity_reviewer_1", "identity_reviewer_2", "identity_reviewer_3"})
        self.assertTrue(all(len(payload["cases"]) == 1 for payload in votes))

    def test_oversized_cases_stay_unresolved_without_a_partial_jury(self):
        source, cast = fixture()
        for kind, ids in (("merge", ("c1", "c2")), ("split", ("c1",))):
            oversized = case(kind, ids)
            oversized["reason"] = "Narrative evidence identifies the people. " * 55
            with tempfile.TemporaryDirectory() as directory:
                probe = RegionScanRequests([oversized], limit=200000, cache=directory)
                audit_identities(source, cast, probe, make_units(source), adaptive=True)
                probe_calls = deepcopy(probe.calls)
            full = next(payload for payload in probe_calls if payload["mode"] == "identity_scan")
            vote = next(payload for payload in probe_calls if payload["mode"] == "identity_vote")
            scan_total = payload_size(SCAN + CAST_SCOPE_GUIDANCE, full)
            vote_total = payload_size(VOTE, vote)
            self.assertGreater(vote_total, scan_total)
            with tempfile.TemporaryDirectory() as directory:
                requests = RegionScanRequests([oversized], limit=(scan_total + vote_total) // 2, cache=directory)
                updated, issues, provenance = audit_identities(source, cast, requests, make_units(source),
                                                               adaptive=True)
                calls = deepcopy(requests.calls)
            self.assertTrue(any(payload["mode"] == "identity_scan" for payload in calls))
            self.assertFalse(any(payload["mode"] == "identity_vote" for payload in calls))
            self.assertEqual(updated, cast)
            self.assertEqual(len(provenance["cases"]), 1)
            self.assertEqual(provenance["cases"][0]["status"], "unresolved")
            self.assertEqual(provenance["cases"][0]["votes"], [])
            self.assertEqual(len(issues), 1)
            self.assertIn("no reviewer was dispatched", issues[0]["reason"])

    def test_scoped_vote_keeps_identical_complete_records_for_all_three_reviewers(self):
        source, cast = wide_fixture()
        cast["characters"][1]["evidence"] = deepcopy(cast["characters"][0]["evidence"])
        for character in cast["characters"]:
            character["description"] += " Source-bound regional identity context." * 60
        with tempfile.TemporaryDirectory() as directory:
            probe = RegionScanRequests([case()], cache=directory)
            audit_identities(source, cast, probe, make_units(source), adaptive=True)
            probe_calls = deepcopy(probe.calls)
        vote = next(payload for payload in probe_calls if payload["mode"] == "identity_vote")
        scan = next(payload for payload in probe_calls if payload["mode"] == "identity_scan")
        leaf_sizes = []
        for unit in all_units(source):
            view, scope = scoped_cast(source, cast, [unit])
            leaf = dict(scan, original_cast=view, cast_scope=scope, source_units=[unit], source_scope="source_batch")
            leaf_sizes.append(payload_size(SCAN + CAST_SCOPE_GUIDANCE, leaf))
        budget = max([payload_size(VOTE + CAST_SCOPE_GUIDANCE, vote), *leaf_sizes]) + 200
        self.assertGreater(payload_size(VOTE, dict(vote, original_cast=cast)), budget)
        with tempfile.TemporaryDirectory() as directory:
            requests = RegionScanRequests([case()], limit=budget, cache=directory)
            updated, issues, _ = audit_identities(source, cast, requests, make_units(source), adaptive=True)
            calls = deepcopy(requests.calls)
        votes = [payload for payload in calls if payload["mode"] == "identity_vote"]
        self.assertEqual(len(votes), 3)
        self.assertEqual({payload["reviewer_id"] for payload in votes},
                         {"identity_reviewer_1", "identity_reviewer_2", "identity_reviewer_3"})
        views = {dumps(payload["original_cast"], sort_keys=True, ensure_ascii=False) for payload in votes}
        self.assertEqual(len(views), 1)
        view = votes[0]["original_cast"]
        self.assertEqual([row["id"] for row in view["characters"]], ["c1", "c2"])
        self.assertFalse(votes[0]["cast_scope"]["complete"])
        self.assertEqual(votes[0]["cast_scope"]["included_ids"], ["c1", "c2"])
        self.assertEqual(votes[0]["original_cast_fingerprint"], fingerprint(cast))
        originals = {row["id"]: row for row in cast["characters"]}
        for row in view["characters"]:
            self.assertEqual(row, originals[row["id"]])
        self.assertEqual(len({payload["lens"] for payload in votes}), 3)
        self.assertEqual([row["id"] for row in updated["characters"]],
                         [row["id"] for row in cast["characters"] if row["id"] != "c2"])
        self.assertEqual(issues, [])

    def test_explicit_scan_regions_reaudit_only_those_units_and_still_validate_batches(self):
        source, cast = fixture()
        units = all_units(source)
        with tempfile.TemporaryDirectory() as directory:
            requests = Bounded(cache=directory)
            updated, issues, provenance = audit_identities(source, cast, requests, make_units(source),
                                                           adaptive=True, scan_regions=[[units[1]]])
            calls = deepcopy(requests.calls)
        scans = [payload for payload in calls if payload["mode"] == "identity_scan"]
        self.assertEqual([payload["source_units"] for payload in scans], [[units[1]]])
        self.assertEqual(scans[0]["source_scope"], "source_batch")
        self.assertEqual(updated, cast)
        self.assertEqual(issues, [])
        foreign = dict(units[0], unit_id="invented")
        with self.assertRaisesRegex(ValueError, "audited batches"):
            audit_identities(source, cast, Bounded(), make_units(source), adaptive=True,
                             scan_regions=[[foreign]])
        with self.assertRaisesRegex(ValueError, "requires adaptive"):
            audit_identities(source, cast, Bounded(), make_units(source), scan_regions=[[units[0]]])
        with self.assertRaisesRegex(ValueError, "list of source-unit region lists"):
            audit_identities(source, cast, Bounded(), make_units(source), adaptive=True, scan_regions="regions")

    def test_adaptive_series_scan_decides_every_character_exactly_once(self):
        names = ("Alma", "Bram", "Cato", "Dara", "Enid", "Faro", "Gail", "Hugo", "Ivy", "Jules", "Kip", "Lora")
        source, cast = fixture(names)
        prior = deepcopy(cast)
        prior["characters"][0]["id"] = "prior_alma"
        decisions = {"c1": {"character_id": "c1", "decision": "match", "prior_character_ids": ["prior_alma"],
                     "reason": "The same returning person with unchanged relationships and history.",
                     "evidence": deepcopy(cast["characters"][0]["evidence"])}}
        units = all_units(source)
        single = _series_scan_payload(source, cast, prior, "registry-hash", units,
                                      cast["characters"][:1], "complete_source")
        full_probe = Requests()
        full_probe.max_request_chars = 200000
        full = _series_scan_requests(source, cast, prior, "registry-hash", units, full_probe, False)[0]
        self.assertGreater(payload_size(SERIES_SCAN, full), payload_size(SERIES_SCAN, single))
        requests = SeriesRequests(decisions=decisions)
        requests.max_request_chars = payload_size(SERIES_SCAN, full) - 1
        updated, _, continuity = _scan_series_identities(source, cast, "registry", prior, requests, units,
                                                         False, lambda message: None, adaptive=True)
        covered = [character for entry in continuity["requests"] for character in entry["character_ids"]]
        self.assertGreater(len(continuity["requests"]), 1)
        self.assertEqual(sorted(covered), sorted(character["id"] for character in cast["characters"]))
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(len(continuity["decisions"]), len(cast["characters"]))
        self.assertEqual(updated["characters"][0]["series_identity"]["character_id"], "prior_alma")

    def test_series_adaptive_groups_characters_instead_of_one_request_per_character(self):
        names = ("Alma", "Bram", "Cato", "Dara", "Enid", "Faro", "Gail", "Hugo", "Ivy", "Jules", "Kip", "Lora")
        source, cast = fixture(names)
        prior = {"characters": deepcopy(cast["characters"])}
        units = all_units(source)
        full_probe = Requests()
        full_probe.max_request_chars = 200000
        full = _series_scan_requests(source, cast, prior, "registry-hash", units, full_probe, False)
        self.assertEqual(len(full), 1)
        requests = Requests()
        requests.max_request_chars = payload_size(SERIES_SCAN, full[0]) - 1
        payloads = _series_scan_requests(source, cast, prior, "registry-hash", units, requests, False, adaptive=True)
        self.assertEqual(len(payloads), 2)
        supplied = [row["id"] for payload in payloads for row in payload["current_characters"]]
        self.assertEqual(sorted(supplied), sorted(row["id"] for row in cast["characters"]))
        self.assertEqual(len(supplied), len(set(supplied)))
        self.assertTrue(all(payload["source_scope"] == "complete_source" for payload in payloads))
        self.assertTrue(all(len(payload["current_characters"]) == 6 for payload in payloads))

    def test_series_adaptive_single_character_leaf_keeps_selected_blocks_explicit(self):
        source, cast = fixture(("Imogen", "Tobias", "Cassandra", "Perrin"))
        blocks = []
        for copy in range(10):
            for block in source["blocks"]:
                blocks.append(dict(block, id=f"{block['id']}_x{copy}"))
        source = dict(source, blocks=blocks)
        units = [{"unit_id": f"s{index:04d}", "block_id": block["id"], "chapter_id": block["chapter_id"],
                  "start": 0, "text": block["text"]} for index, block in enumerate(blocks)]
        for index, character in enumerate(cast["characters"]):
            block = blocks[index * 2]
            character["evidence"] = [{"block_id": block["id"], "quote": block["text"][:20]}]
        prior = {"characters": deepcopy(cast["characters"])}
        full_probe = Requests()
        full_probe.max_request_chars = 200000
        full = _series_scan_requests(source, cast, prior, "registry-hash", units, full_probe, False)[0]
        legacy = Requests()
        legacy.max_request_chars = payload_size(SERIES_SCAN, full) - 1
        selected = _series_scan_requests(source, cast, prior, "registry-hash", units, legacy, False)
        self.assertEqual(len(selected), 4)
        measured = payload_size(SERIES_SCAN, selected[0])
        requests = Requests()
        requests.max_request_chars = budget_floor(measured) + 2000
        payloads = _series_scan_requests(source, cast, prior, "registry-hash", units, requests, False, adaptive=True)
        self.assertEqual(len(payloads), 4)
        supplied = [row["id"] for payload in payloads for row in payload["current_characters"]]
        self.assertEqual(sorted(supplied), sorted(row["id"] for row in cast["characters"]))
        self.assertEqual(len(supplied), len(set(supplied)))
        self.assertTrue(all(payload["source_scope"] == "selected_blocks" for payload in payloads))
        for payload in payloads:
            self.assertGreater(payload["context_selection"]["omitted_source_blocks"], 0)
            self.assertEqual(payload["context_selection"]["total_source_blocks"], len(blocks))


if __name__ == "__main__":
    unittest.main()
