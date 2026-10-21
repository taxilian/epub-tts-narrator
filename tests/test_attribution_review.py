"""Semantic review is source-bound and cannot turn dissent into a correction."""

from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
from threading import Event, Lock
import tempfile
import time
import unittest

from epub_audiobook.attribution_review import (
    LENSES, _accept_scan, _batches, audit_attribution, count_review_batches,
)
from epub_audiobook.contracts import validate_document
from epub_audiobook.performance import apply_bracket_delivery
from epub_audiobook.pipeline import CachedRequests, fingerprint, segment
from test_contracts import valid_document


def fixture(count=2, long=False, bracket=False):
    document = valid_document()
    document["cast"]["roles"][0]["id"] = "narrator_main"
    document["cast"]["voices"][1]["traits"] = []
    blocks = []
    for index in range(count):
        text = "Mara said softly, “" + ("Hello " * 60 if long else "Hello") + (" [Friend].”" if bracket else ".”")
        blocks.append({"id": f"b{index + 1}", "chapter_id": "ch1", "text": text,
                       "source": {"href": "text/ch1.xhtml", "element_id": None, "tag": "p"}})
    document["source"]["blocks"] = blocks
    document["source"]["chapters"][0]["block_ids"] = [block["id"] for block in blocks]
    narrator = document["cast"]["roles"][0]
    document["segments"] = [segment(document["source"], block["id"], 0, block["text"], narrator) for block in blocks]
    document["readiness"] = {"ready": True, "errors": [], "review_count": 0}
    return apply_bracket_delivery(document) if bracket else document


def correction(unit):
    before, after = unit["text"].split("“", 1)
    evidence = [{"block_id": unit["block_id"], "quote": "Mara said softly"}]
    def piece(text, role, kind):
        return {"text": text, "kind": kind, "role_id": role, "basis": "explicit",
                "reason": "The speech tag explicitly identifies Mara; the tag itself is narration.",
                "evidence": evidence, "candidates": [], "review_required": False,
                "delivery": {"direction": "", "evidence": []}}
    return {"block_id": unit["block_id"], "reason": "Mara's spoken words were incorrectly assigned to narration.",
            "evidence": evidence, "segments": [piece(before, "narrator_main", "narration"), piece("“" + after, "mara", "dialogue")]}


class Requests:
    def __init__(self, *, verdicts=None, modify=None, empty=False):
        self.verdicts, self.modify, self.empty = verdicts or {}, modify, empty
        self.calls, self.lock = [], Lock()
        self.active = self.max_active = 0

    def response(self, payload):
        if payload["mode"] == "attribution_review_scan":
            result = {"corrections": [] if self.empty else [correction(unit) for unit in payload["target_units"]]}
        else:
            result = {"votes": [{"block_id": item["block_id"], "verdict": self.verdicts.get(payload["reviewer_id"], "approve"),
                                 "reason": "The speech tag names Mara while the narrative tag belongs to the narrator.",
                                 "evidence": [{"block_id": item["block_id"], "quote": "Mara said softly"}]}
                                for item in payload["proposals"]]}
        return self.modify(result, payload) if self.modify else result

    def run(self, system, payload, validate, *, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Cancelled before fake request")
        with self.lock:
            self.calls.append(deepcopy(payload))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(.005)
            return validate(self.response(payload))
        finally:
            with self.lock:
                self.active -= 1


class AttributionReviewTests(unittest.TestCase):
    def test_three_blind_votes_fix_speech_and_preserve_narration_source_and_cast(self):
        document, requests = fixture(), Requests()
        before = deepcopy(document)
        result = audit_attribution(document, requests, review_notes="Untrusted suggestion about b1")
        self.assertEqual(document, before)
        self.assertEqual(result["source"], before["source"])
        self.assertEqual(result["cast"], before["cast"])
        self.assertEqual(validate_document(result), [])
        self.assertEqual([item["role_id"] for item in result["segments"]], ["narrator_main", "mara"] * 2)
        audit = result["provenance"]["attribution_review"]
        self.assertEqual(audit["applied_blocks"], 2)
        self.assertEqual(audit["input_script_fingerprint"], fingerprint(document["segments"]))
        juries = [row for row in requests.calls if row["mode"] == "attribution_review_vote"]
        self.assertEqual(len(juries), 3)
        self.assertEqual({row["reviewer_id"] for row in juries}, {row[0] for row in LENSES})
        self.assertTrue(all("review_notes" not in row and "votes" not in row for row in juries))
        self.assertTrue(all(list(row).index("target_units") < list(row).index("reviewer_id") for row in juries))
        for change in audit["changes"]:
            self.assertEqual(len(change["votes"]), 3)
            self.assertEqual(len({vote["request_fingerprint"] for vote in change["votes"]}), 3)
            self.assertEqual(change["status"], "applied")

    def test_dissent_and_uncertainty_keep_original_assignments_and_advisory_notes(self):
        for verdict in ("reject", "uncertain"):
            with self.subTest(verdict=verdict):
                document = fixture()
                result = audit_attribution(document, Requests(verdicts={"attribution_reviewer_2": verdict}))
                self.assertEqual(result["segments"], document["segments"])
                self.assertEqual(result["provenance"]["attribution_review"]["applied_blocks"], 0)
                self.assertEqual(len(result["issues"]), 2)
                self.assertTrue(result["readiness"]["ready"])
                self.assertEqual(validate_document(result), [])

    def test_invalid_jury_quote_cannot_approve_and_is_saved_as_error_vote(self):
        def modify(response, payload):
            if payload["mode"] == "attribution_review_vote" and payload["reviewer_id"] == "attribution_reviewer_3":
                response["votes"][0]["evidence"][0]["quote"] = "fabricated words"
            return response
        document = fixture()
        result = audit_attribution(document, Requests(modify=modify))
        self.assertEqual(result["segments"], document["segments"])
        self.assertEqual(result["provenance"]["attribution_review"]["changes"][0]["votes"][-1]["verdict"], "error")

    def test_dissent_is_per_block_and_does_not_block_other_unanimous_corrections(self):
        def modify(response, payload):
            if payload["mode"] == "attribution_review_vote" and payload["reviewer_id"] == "attribution_reviewer_1":
                response["votes"][1]["verdict"] = "reject"
            return response
        document = fixture()
        result = audit_attribution(document, Requests(modify=modify))
        self.assertEqual([row["role_id"] for row in result["segments"] if row["block_id"] == "b1"], ["narrator_main", "mara"])
        self.assertEqual([row for row in result["segments"] if row["block_id"] == "b2"], [document["segments"][1]])
        self.assertEqual(result["provenance"]["attribution_review"]["applied_blocks"], 1)
        self.assertEqual(result["issues"][0]["target"], "b2")

    def test_missing_duplicate_and_reordered_jury_votes_never_apply(self):
        def mutate(response, payload):
            if payload["mode"] == "attribution_review_vote":
                response["votes"].reverse()
            return response
        result = audit_attribution(fixture(), Requests(modify=mutate))
        self.assertEqual(result["provenance"]["attribution_review"]["applied_blocks"], 0)
        self.assertTrue(all(vote["verdict"] == "error" for change in result["provenance"]["attribution_review"]["changes"] for vote in change["votes"]))

    def test_scan_rejects_forged_text_quotes_cast_mutations_and_unknown_roles(self):
        mutations = [
            lambda result: result["corrections"][0]["segments"][1].update(text="New text"),
            lambda result: result["corrections"][0]["evidence"][0].update(quote="fabricated"),
            lambda result: result.update(cast={}),
            lambda result: result["corrections"][0]["segments"][1].update(role_id="invented"),
            lambda result: result["corrections"][0]["segments"][1].update(voice_id="invented"),
            lambda result: result["corrections"].reverse(),
            lambda result: result["corrections"].append(deepcopy(result["corrections"][0])),
            lambda result: result["corrections"][0]["segments"].pop(),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                def modify(response, payload):
                    mutation(response)
                    return response
                with self.assertRaises(ValueError):
                    audit_attribution(fixture(), Requests(modify=modify))

    def test_context_blocks_are_read_only_and_unseen_quotes_are_rejected(self):
        document = fixture(count=4, long=True)
        batch = _batches(document["source"], 256)[0]
        original = {row["block_id"]: [row] for row in document["segments"]}
        self.assertTrue(batch["context_units"])
        with self.assertRaisesRegex(ValueError, "context blocks"):
            _accept_scan({"corrections": [correction(batch["context_units"][0])]}, document, batch, original)
        batch["context_units"] = []
        proposed = correction(batch["target_units"][0])
        proposed["segments"][0]["evidence"] = [{"block_id": "b4", "quote": "Mara"}]
        with self.assertRaisesRegex(ValueError, "visible source"):
            _accept_scan({"corrections": [proposed]}, document, batch, original)

    def test_existing_segment_reviews_and_issues_are_not_resolved_by_jury(self):
        document = fixture()
        document["segments"][0]["attribution"].update(review_required=True, reason="Existing uncertain narrator decision.")
        document["issues"] = [{"id": "old", "kind": "casting", "target": "c1", "reason": "Existing review.", "resolved": False}]
        result = audit_attribution(document, Requests())
        self.assertEqual(result["issues"][0], document["issues"][0])
        self.assertTrue(all(row["attribution"]["review_required"] for row in result["segments"] if row["block_id"] == "b1"))
        self.assertTrue(result["readiness"]["ready"])

    def test_explicit_user_corrections_cannot_be_replaced(self):
        document = fixture()
        document["segments"][0]["attribution"]["basis"] = "user"
        with self.assertRaisesRegex(ValueError, "user corrections"):
            audit_attribution(document, Requests())

    def test_bracket_cues_are_reapplied_without_changing_cast_or_source(self):
        document = fixture(bracket=True)
        result = audit_attribution(document, Requests())
        self.assertEqual(validate_document(result), [])
        self.assertEqual(result["cast"], document["cast"])
        marked = [row for row in result["segments"] if "performance" in row]
        self.assertEqual(len(marked), 2)
        self.assertTrue(all(row["role_id"] == "mara" and row["spoken_text"] == "Friend" for row in marked))
        for change in result["provenance"]["attribution_review"]["changes"]:
            self.assertEqual(change["applied_segments"], [row for row in result["segments"] if row["block_id"] == change["block_id"]])

    def test_no_proposals_skips_juries_and_does_not_claim_guaranteed_accuracy(self):
        requests = Requests(empty=True)
        result = audit_attribution(fixture(), requests)
        self.assertEqual(len(requests.calls), 1)
        self.assertEqual(result["provenance"]["attribution_review"]["proposed_blocks"], 0)
        self.assertEqual(result["provenance"]["attribution_review"]["scans"][0]["target_block_ids"], ["b1", "b2"])

    def test_global_scan_and_jury_queues_never_exceed_five_workers(self):
        requests = Requests()
        result = audit_attribution(fixture(count=10, long=True), requests, workers=5, review_chunk_chars=256)
        self.assertEqual(requests.max_active, 5)
        self.assertEqual(len(requests.calls), 40)
        self.assertEqual(result["provenance"]["attribution_review"]["applied_blocks"], 10)
        self.assertEqual(validate_document(result), [])

    def test_plan_counts_full_block_batches_and_bounds_neighbor_context(self):
        document = fixture(count=100, long=True)
        batches = _batches(document["source"], 256)
        self.assertEqual(count_review_batches(document["source"], 256), 100)
        self.assertEqual(sum(len(unit["text"]) for unit in batches[50]["context_units"]), 24000)
        self.assertEqual(batches[50]["target_units"][0]["text"], document["source"]["blocks"][50]["text"])
        with self.assertRaises(ValueError):
            count_review_batches(document["source"], 255)

    def test_fatal_error_stops_new_dispatch_and_preserves_original_exception(self):
        started = Event()
        class FatalRequests(Requests):
            def run(self, system, payload, validate, *, cancelled=None):
                with self.lock:
                    self.calls.append(payload)
                if payload["batch_index"] == 0:
                    self.assert_started = started.wait(2)
                    raise RuntimeError("Provider unavailable")
                started.set()
                cancelled.wait(2)
                raise CancelledError("Sibling failed")
        requests = FatalRequests()
        with self.assertRaisesRegex(RuntimeError, "Provider unavailable"):
            audit_attribution(fixture(count=10, long=True), requests, workers=2, review_chunk_chars=256)
        self.assertTrue(requests.assert_started)
        self.assertEqual(len(requests.calls), 2)

    def test_cache_replay_reproduces_audit_without_new_model_calls(self):
        class Client(Requests):
            identity = {"model": "test", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                with self.lock:
                    self.calls.append(deepcopy(payload))
                return self.response(payload)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = audit_attribution(fixture(), requests)
            count = len(client.calls)
            second = audit_attribution(fixture(), requests)
        self.assertEqual(count, 4)
        self.assertEqual(len(client.calls), count)
        self.assertEqual(first, second)

    def test_targeted_review_uses_one_scan_and_three_votes_preserving_other_blocks_and_history(self):
        for bracket in (False, True):
            with self.subTest(bracket=bracket):
                document, requests = fixture(count=30, bracket=bracket), Requests()
                document["segments"][0]["id"] = "preserve-this-original-id"
                document["segments"][0]["attribution"].update(review_required=True, reason="Existing outside review.")
                old_audit = {"version": "1", "input_script_fingerprint": "older-audit"}
                document["provenance"]["attribution_review"] = deepcopy(old_audit)
                document["provenance"]["attribution_review_history"] = [{"version": "older"}]
                document["issues"] = [{"id": "old", "kind": "casting", "target": "b1", "reason": "Existing review.", "resolved": False}]
                before = deepcopy(document)
                result = audit_attribution(document, requests, review_chunk_chars=256,
                                           target_block_ids=["b15"], review_notes="Check this target independently.")
                self.assertEqual(document, before)
                self.assertEqual(result["source"], before["source"])
                self.assertEqual(result["cast"], before["cast"])
                self.assertEqual(result["issues"], before["issues"])
                self.assertEqual([row for row in result["segments"] if row["block_id"] != "b15"],
                                 [row for row in before["segments"] if row["block_id"] != "b15"])
                self.assertEqual(validate_document(result), [])
                self.assertEqual(len(requests.calls), 4)
                scan = next(row for row in requests.calls if row["mode"] == "attribution_review_scan")
                juries = [row for row in requests.calls if row["mode"] == "attribution_review_vote"]
                self.assertEqual([row["block_id"] for row in scan["target_units"]], ["b15"])
                self.assertEqual([row["block_id"] for row in scan["context_units"]],
                                 [f"b{i}" for i in range(1, 31) if i != 15])
                self.assertEqual({row["block_id"] for row in scan["original_segments"]}, {"b15"})
                self.assertEqual(len(juries), 3)
                self.assertTrue(all("review_notes" not in row for row in juries))
                audit = result["provenance"]["attribution_review"]
                self.assertEqual(audit["applied_blocks"], 1)
                self.assertEqual(audit["scope"], {"mode": "target_blocks", "target_block_ids": ["b15"],
                                 "reviewed_blocks": 1, "total_source_blocks": 30,
                                 "source_batch_count": count_review_batches(document["source"], 256),
                                 "reviewed_batch_count": 1})
                self.assertEqual(audit["source_fingerprint"], fingerprint(before["source"]))
                self.assertEqual(audit["cast_fingerprint"], fingerprint(before["cast"]))
                self.assertEqual(audit["input_script_fingerprint"], fingerprint(before["segments"]))
                self.assertEqual(audit["output_script_fingerprint"], fingerprint(result["segments"]))
                self.assertEqual(audit["scans"][0]["request_fingerprint"], fingerprint(scan))
                self.assertEqual(result["provenance"]["attribution_review_history"], [{"version": "older"}, old_audit])
                if bracket:
                    self.assertEqual(result["provenance"]["bracket_delivery"], before["provenance"]["bracket_delivery"])
                    self.assertTrue(any("performance" in row and row["role_id"] == "mara"
                                        for row in result["segments"] if row["block_id"] == "b15"))
                else:
                    self.assertNotIn("bracket_delivery", result["provenance"])

    def test_targeted_review_rejects_changes_to_unselected_original_batch_blocks(self):
        def modify(response, payload):
            if payload["mode"] == "attribution_review_scan":
                response["corrections"] = [correction(payload["context_units"][0])]
            return response
        requests = Requests(modify=modify)
        with self.assertRaisesRegex(ValueError, "context blocks"):
            audit_attribution(fixture(), requests, target_block_ids=["b2"])
        self.assertEqual(len(requests.calls), 1)

    def test_targeted_review_accepts_order_independent_collections_and_replays_cache(self):
        class Client(Requests):
            identity = {"model": "test", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                return self.response(payload)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = audit_attribution(fixture(count=3), requests, target_block_ids=["b3", "b1"])
            for ids in (("b1", "b3"), {"b3", "b1"}, frozenset({"b3", "b1"})):
                self.assertEqual(audit_attribution(fixture(count=3), requests, target_block_ids=ids), first)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(first["provenance"]["attribution_review"]["scope"]["target_block_ids"], ["b1", "b3"])

    def test_invalid_target_ids_fail_before_any_model_request(self):
        for ids in ([], (), set(), "b1", {"b1": True}, ["b1", "b1"], ["missing"],
                    ["b1", None], [1], True, iter(["b1"])):
            with self.subTest(ids=ids):
                requests = Requests()
                with self.assertRaisesRegex(ValueError, "target_block_ids"):
                    audit_attribution(fixture(), requests, target_block_ids=ids)
                self.assertEqual(requests.calls, [])

    def test_malformed_input_and_invalid_limits_are_rejected_before_calls(self):
        for options in ({"workers": 6}, {"workers": True}, {"review_chunk_chars": 20}, {"review_notes": []}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                audit_attribution(fixture(), Requests(), **options)
        document = fixture()
        document["segments"][0]["text"] = "wrong"
        requests = Requests()
        with self.assertRaisesRegex(ValueError, "Invalid input analysis"):
            audit_attribution(document, requests)
        self.assertEqual(requests.calls, [])


if __name__ == "__main__":
    unittest.main()
