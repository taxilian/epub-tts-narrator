"""Empty model delivery objects need no inference; asserted cues remain strict."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_audiobook.attribution_review import audit_attribution
from epub_audiobook.contracts import validate_document
from epub_audiobook.pipeline import (
    CachedRequests, align_attribution_response, align_units, base_document,
)
from test_attribution_review import Requests, fixture as review_fixture
from test_repeated_runs import fixture


class DeliveryDefaultTests(unittest.TestCase):
    def setUp(self):
        self.source, self.cast, self.units, self.response = fixture(
            'Mara said, "I ssssaid stop."', ['Mara said, ', '"I ssssaid stop."'])

    def accept(self, response):
        return align_attribution_response(self.source, self.cast, self.units, response)

    def test_empty_model_defaults_preserve_text_roles_and_raw_response(self):
        expected = self.accept(self.response)
        for delivery in ({}, {"direction": ""}, {"evidence": []}, {"direction": "", "evidence": []}):
            with self.subTest(delivery=delivery):
                response = deepcopy(self.response)
                for piece in response["units"][0]["segments"]:
                    piece["delivery"] = deepcopy(delivery)
                before = deepcopy(response)
                accepted = self.accept(response)
                self.assertEqual(accepted, expected)
                self.assertEqual(response, before)
                self.assertEqual(validate_document(base_document(self.source, self.cast, accepted)), [])
                self.assertFalse(any(row["attribution"]["review_required"] for row in accepted))

    def test_missing_malformed_or_nonempty_partial_delivery_is_not_invented(self):
        evidence = [{"block_id": "b1", "quote": "Mara"}]
        invalid = [None, "", "softly", [], False, 0,
                   {"direction": None}, {"direction": False}, {"direction": 0},
                   {"evidence": None}, {"evidence": ""}, {"evidence": {}},
                   {"direction": " "}, {"direction": "softly"}, {"evidence": evidence},
                   {"direction": "", "evidence": None}, {"direction": "", "evidence": False},
                   {"direction": "", "unknown": ""}, {"style": ""}]
        for delivery in invalid:
            with self.subTest(delivery=delivery):
                response = deepcopy(self.response)
                response["units"][0]["segments"][0]["delivery"] = delivery
                before = deepcopy(response)
                with self.assertRaises(ValueError):
                    self.accept(response)
                self.assertEqual(response, before)
        del self.response["units"][0]["segments"][0]["delivery"]
        with self.assertRaisesRegex(ValueError, "fields are missing"):
            self.accept(self.response)

    def test_bad_source_and_evidence_still_fail_with_empty_defaults(self):
        mutations = [
            lambda p: p.update(text="Mara whispered, "),
            lambda p: p.update(role_id="invented"),
            lambda p: p.update(evidence=[{"block_id": "b1", "quote": "mara said"}]),
            lambda p: p.update(evidence=[{"block_id": "unknown", "quote": "Mara"}]),
            lambda p: p.update(delivery={"direction": "softly", "evidence": []}),
            lambda p: p.update(delivery={"direction": "softly", "evidence": [{"block_id": "b1", "quote": "softly"}]}),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                response = deepcopy(self.response)
                piece = response["units"][0]["segments"][0]
                piece["delivery"] = {}
                mutate(piece)
                with self.assertRaises(ValueError):
                    self.accept(response)

    def test_raw_cache_preserved_and_replayed_without_repair_call(self):
        response = deepcopy(self.response)
        response["units"][0]["segments"][0]["delivery"] = {}
        class Client:
            identity = {"model": "offline-delivery-fixture"}
            calls = 0
            def complete(client, system, payload):
                client.calls += 1
                return deepcopy(response)
        client, progress = Client(), []
        with tempfile.TemporaryDirectory() as temp:
            requests = CachedRequests(client, temp, repair_attempts=0, progress=progress.append)
            first = requests.run("original prompt", {"target_units": self.units}, self.accept)
            again = requests.run("original prompt", {"target_units": self.units}, self.accept)
            cached = json.loads(next(Path(temp).glob("*.json")).read_text())
            self.assertFalse((Path(temp) / "failed").exists())
        self.assertEqual(client.calls, 1)
        self.assertEqual(first, again)
        self.assertEqual(cached, response)
        self.assertEqual(first[0]["delivery"], {"direction": "", "evidence": []})
        self.assertTrue(any("Cache reuse:" in message for message in progress))
        self.assertFalse(any("Repairing validation" in message for message in progress))

    def test_published_document_and_user_edit_schema_remain_strict(self):
        document = base_document(self.source, self.cast, self.accept(self.response))
        document["segments"][0]["delivery"] = {}
        self.assertTrue(validate_document(document))
        self.response["units"][0]["segments"][0]["delivery"] = {}
        with self.assertRaises(ValueError):
            align_units(self.source, self.cast, self.units, self.response, allow_user=True)

    def test_semantic_scan_defaults_still_require_three_valid_votes(self):
        def partial_defaults(response, payload):
            if "corrections" in response:
                for correction in response["corrections"]:
                    for piece in correction["segments"]:
                        piece["delivery"] = {"direction": ""}
            return response
        document = review_fixture()
        before = deepcopy(document)
        baseline = audit_attribution(document, Requests())
        requests = Requests(modify=partial_defaults)
        result = audit_attribution(document, requests)
        self.assertEqual(document, before)
        self.assertEqual(result["segments"], baseline["segments"])
        self.assertEqual(validate_document(result), [])
        votes = [p for p in requests.calls if p["mode"] == "attribution_review_vote"]
        self.assertEqual(len(votes), 3)
        for request in votes:
            for proposal in request["proposals"]:
                self.assertTrue(all(p["delivery"] == {"direction": "", "evidence": []}
                                    for p in proposal["proposed_segments"]))
        changes = result["provenance"]["attribution_review"]["changes"]
        self.assertTrue(all(len(change["votes"]) == 3 and change["status"] == "applied" for change in changes))

        dissent = audit_attribution(document, Requests(modify=partial_defaults,
                                  verdicts={"attribution_reviewer_3": "reject"}))
        self.assertEqual(dissent["segments"], before["segments"])
        self.assertEqual(dissent["provenance"]["attribution_review"]["applied_blocks"], 0)
