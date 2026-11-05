"""Batch screening retains global identity clues without authorizing hidden citations."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.identity import audit_identities
from epub_tts_narrator.pipeline import CachedRequests, make_units
from test_identity import Requests, case, fixture


class IdentityScanScopeTests(unittest.TestCase):
    def test_scan_discloses_actual_source_coverage_without_dropping_cast_context(self):
        # Given complete source divided either across scans or in one request.
        source, cast = fixture()
        units = [unit for batch in make_units(source) for unit in batch]
        for batches, whole_book, expected in (
                ([[units[0]], [units[1]]], False, ["source_batch", "source_batch"]),
                ([[units[0]], [units[1]]], True, ["complete_source"]),
                ([units], False, ["complete_source"])):
            with self.subTest(whole_book=whole_book, batch_count=len(batches)):
                requests = Requests()

                # When screening the cast through the real audit entry point.
                updated, issues, provenance = audit_identities(
                    source, cast, requests, batches, whole_book=whole_book)

                # Then scope is accurate, with every source unit and cast record retained.
                scans = [p for p in requests.calls if p["mode"] == "identity_scan"]
                self.assertEqual([p["source_scope"] for p in scans], expected)
                self.assertEqual([u for p in scans for u in p["source_units"]], units)
                self.assertTrue(all(p["original_cast"] == cast for p in scans))
                self.assertEqual(updated, cast)
                self.assertEqual(issues, [])
                self.assertEqual(len(provenance["scans"]), len(expected))

    def test_later_batch_can_link_dissimilar_names_with_broader_blind_review(self):
        # Given an earlier identity quotation and a later explicit nickname revelation.
        source, cast = fixture()
        source["blocks"][0]["text"] = "Mara commanded the survey ship."
        source["blocks"][1]["text"] = "Red was the name her crew used for Captain Mara."
        for character, block in zip(cast["characters"], source["blocks"]):
            character["evidence"] = [{"block_id": block["id"], "quote": block["text"]}]
        units = [unit for batch in make_units(source) for unit in batch]
        local_case = case()
        local_case["evidence"] = deepcopy(cast["characters"][1]["evidence"])

        class LaterLinkRequests(Requests):
            def response(self, payload):
                if payload["mode"] == "identity_scan":
                    return {"cases": [deepcopy(local_case)]
                            if payload["source_units"][0]["block_id"] == "b2" else []}
                result = super().response(payload)
                result["evidence"] = deepcopy(local_case["evidence"])
                return result

        requests = LaterLinkRequests()

        # When the only linking evidence occurs in the later scan.
        updated, issues, provenance = audit_identities(
            source, cast, requests, [[units[0]], [units[1]]])

        # Then a locally anchored proposal still reaches three broader blind reviews.
        self.assertEqual(len(updated["characters"]), 1)
        self.assertEqual(issues, [])
        self.assertEqual(provenance["cases"][0]["scan_indexes"], [1])
        votes = [p for p in requests.calls if p["mode"] == "identity_vote"]
        self.assertEqual(len({p["reviewer_id"] for p in votes}), 3)
        self.assertTrue(all({u["block_id"] for u in p["source_units"]} == {"b1", "b2"}
                            for p in votes))

    def test_hidden_cast_citation_is_repaired_and_only_valid_result_is_reused(self):
        # Given real cast evidence from another batch and one allowed repair.
        source, cast = fixture()
        units = [unit for batch in make_units(source) for unit in batch]
        batches = [[units[0]], [units[1]]]
        local_case = case("uncertain")
        hidden_case = deepcopy(local_case)
        hidden_case["evidence"] = [{"block_id": "b2", "quote": "She was still the captain."}]
        cast["characters"][1]["evidence"] = deepcopy(hidden_case["evidence"])

        class Client:
            identity = {"model": "offline-scope-repair"}

            def __init__(self):
                self.calls = []

            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                if payload["source_units"][0]["block_id"] == "b2":
                    return {"cases": []}
                return {"cases": [deepcopy(local_case if "correction" in payload else hidden_case)]}

        client = Client()
        with tempfile.TemporaryDirectory() as temporary:
            requests = CachedRequests(client, Path(temporary), repair_attempts=1)

            # When the model cites a real but hidden cast quotation, then reruns.
            first = audit_identities(source, cast, requests, batches)
            second = audit_identities(source, cast, requests, batches)

        # Then hidden evidence never enters accepted provenance and replay makes no new calls.
        self.assertEqual(len(client.calls), 3)
        repaired = [p for p in client.calls if "correction" in p]
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0]["source_units"], [units[0]])
        self.assertEqual(repaired[0]["original_cast"], cast)
        self.assertEqual(first, second)
        self.assertEqual(first[0], cast)
        self.assertEqual(first[2]["cases"][0]["evidence"], local_case["evidence"])
        self.assertEqual(first[2]["cases"][0]["status"], "unresolved")

    def test_mixed_visible_and_hidden_evidence_still_stops_the_scan(self):
        # Given one local quotation mixed with one genuine out-of-batch quotation.
        source, cast = fixture()
        units = [unit for batch in make_units(source) for unit in batch]
        mixed = case()
        mixed["evidence"].append({"block_id": "b2", "quote": "She was still the captain."})

        # When the model returns that evidence, a local anchor cannot authorize the hidden item.
        with self.assertRaisesRegex(ValueError, "outside the source text visible"):
            audit_identities(source, cast, Requests([mixed]), [[units[0]], [units[1]]])


if __name__ == "__main__":
    unittest.main()
