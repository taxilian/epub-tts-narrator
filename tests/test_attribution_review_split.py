"""Bounded scan subdivision preserves source scope, votes, and valid caches."""

from copy import deepcopy
from pathlib import Path
from threading import get_ident
import json
import tempfile
import unittest

from epub_tts_narrator.attribution_review import MAX_SCAN_SPLIT_DEPTH, _batches, audit_attribution, record_scan_truncation
from epub_tts_narrator.attribution_review_prompts import SCAN
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import CachedRequests, fingerprint
from epub_tts_narrator.prompts import PROMPT_VERSION
from epub_tts_narrator.provider import ProviderError
from test_attribution_review import Requests, correction, fixture


TRUNCATED = "The model output was truncated. Increase the output token limit or reduce the input."


class ScanSubdivisionTests(unittest.TestCase):
    def test_two_split_levels_keep_whole_blocks_context_and_one_blind_jury(self):
        threads = []
        def truncate(response, payload):
            if payload["mode"] == "attribution_review_scan":
                threads.append(get_ident())
                if len(payload["target_units"]) > 2:
                    raise ProviderError(TRUNCATED)
            return response
        document = fixture(count=8, bracket=True)
        original = deepcopy(document)
        requests = Requests(modify=truncate)
        result = audit_attribution(document, requests, review_notes="Scanner-only lead.")
        scans = [row for row in requests.calls if row["mode"] == "attribution_review_scan"]
        juries = [row for row in requests.calls if row["mode"] == "attribution_review_vote"]
        self.assertEqual([len(row["target_units"]) for row in scans], [8, 4, 2, 2, 4, 2, 2])
        self.assertEqual(len(set(threads)), 1)
        self.assertEqual(len(juries), 3)
        self.assertLessEqual(requests.max_active, 5)
        all_ids = {row["id"] for row in document["source"]["blocks"]}
        for row in scans:
            target_ids = {unit["block_id"] for unit in row["target_units"]}
            context_ids = [unit["block_id"] for unit in row["context_units"]]
            self.assertEqual(target_ids | set(context_ids), all_ids)
            self.assertFalse(target_ids & set(context_ids))
            self.assertEqual(len(context_ids), len(set(context_ids)))
            self.assertEqual(context_ids, sorted(context_ids))
            self.assertEqual({item["block_id"] for item in row["original_segments"]}, target_ids)
        for jury in juries:
            self.assertEqual(jury["target_units"], scans[0]["target_units"])
            self.assertEqual(jury["context_units"], scans[0]["context_units"])
            self.assertEqual(len(jury["proposals"]), 8)
            self.assertNotIn("review_notes", jury)
            self.assertNotIn("subscans", jury)
        self.assertEqual(document, original)
        self.assertEqual(result["source"], original["source"])
        self.assertEqual(result["cast"], original["cast"])
        self.assertEqual(validate_document(result), [])
        audit = result["provenance"]["attribution_review"]
        self.assertEqual(audit["applied_blocks"], 8)
        root = audit["scans"][0]
        self.assertEqual(root["original_attempt_status"], "truncated")
        self.assertEqual(root["request_fingerprint"], fingerprint(scans[0]))
        leaves = [leaf for child in root["subscans"] for leaf in child["subscans"]]
        self.assertEqual(len(leaves), 4)
        self.assertEqual([block for leaf in leaves for block in leaf["target_block_ids"]],
                         [block["id"] for block in document["source"]["blocks"]])
        actual_leaf_hashes = {fingerprint(row) for row in scans if len(row["target_units"]) == 2}
        self.assertEqual({leaf["request_fingerprint"] for leaf in leaves}, actual_leaf_hashes)
        for change in audit["changes"]:
            leaf = next(row for row in leaves if change["block_id"] in row["target_block_ids"])
            self.assertEqual(change["scanner_request_fingerprint"], leaf["request_fingerprint"])
            self.assertEqual(len(change["votes"]), 3)

    def test_interrupted_split_resumes_children_without_repeating_parent_or_other_batch(self):
        document = fixture(count=12)
        batches = _batches(document["source"], 256)
        self.assertEqual(len(batches), 2)
        parent_ids = [row["block_id"] for row in batches[1]["target_units"]]
        second_half = parent_ids[len(parent_ids) // 2:]
        class Client(Requests):
            identity = {"model": "offline", "base_url": "https://example.invalid"}
            fail_child = True
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                if payload["mode"] == "attribution_review_scan" and payload["batch_index"] == 1:
                    ids = [row["block_id"] for row in payload["target_units"]]
                    if ids == parent_ids:
                        raise ProviderError(TRUNCATED)
                    if ids == second_half and self.fail_child:
                        raise ProviderError("Disconnected")
                return self.response(payload)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            requests = CachedRequests(client, cache, repair_attempts=0)
            with self.assertRaisesRegex(ProviderError, "Disconnected"):
                audit_attribution(document, requests, workers=1, review_chunk_chars=256)
            self.assertEqual(len(client.calls), 4)
            parent_payload = client.calls[1]
            normal_payload = client.calls[0]
            normal_key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                      "system": SCAN, "payload": normal_payload})
            self.assertTrue((cache / f"{normal_key}.json").exists())
            parent_key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                      "system": SCAN, "payload": parent_payload})
            self.assertFalse((cache / f"{parent_key}.json").exists())
            marker_path = cache / "attribution-review-subdivisions" / f"{parent_key}.json"
            self.assertEqual(json.loads(marker_path.read_text())["reason"], "model_output_truncated")
            first_call_count = len(client.calls)
            client.fail_child = False
            resumed = audit_attribution(document, requests, workers=1, review_chunk_chars=256)
            new_calls = client.calls[first_call_count:]
            self.assertEqual(len(new_calls), 7)  # One remaining scan + three juries per original batch.
            self.assertEqual([row["block_id"] for row in new_calls[0]["target_units"]], second_half)
            self.assertEqual(sum(row == parent_payload for row in client.calls), 1)
            self.assertEqual(sum(row == normal_payload for row in client.calls), 1)
            count = len(client.calls)
            self.assertEqual(audit_attribution(document, requests, workers=1, review_chunk_chars=256), resumed)
            self.assertEqual(len(client.calls), count)
            # A marker for another payload is not trusted as a skip instruction.
            marker = json.loads(marker_path.read_text())
            marker["request_fingerprint"] = "wrong"
            marker_path.write_text(json.dumps(marker))
            refreshed = audit_attribution(document, requests, workers=1, review_chunk_chars=256)
            self.assertEqual(refreshed["segments"], resumed["segments"])
            self.assertFalse(refreshed["provenance"]["attribution_review"]["scans"][1]["truncation_observation"]["replayed_hint"])
            self.assertEqual(len(client.calls), count + 1)
        self.assertEqual(validate_document(resumed), [])

    def test_verified_prior_truncation_can_be_imported_without_repeating_failed_scan(self):
        document = fixture(count=4)
        capture = Requests(empty=True)
        audit_attribution(document, capture)
        payload = capture.calls[0]
        class Client(Requests):
            identity = {"model": "offline", "base_url": "https://example.invalid"}
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                if payload["mode"] == "attribution_review_scan" and len(payload["target_units"]) == 4:
                    raise AssertionError("Known-truncated parent must not be requested again")
                return self.response(payload)
        client = Client()
        observation = fingerprint({"log": "Prior run observed exact request output truncation", "payload": payload})
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            marker = record_scan_truncation(requests, payload, prior_observation_fingerprint=observation)
            self.assertEqual(client.calls, [])
            self.assertEqual(marker["observation_origin"], "prior_run")
            result = audit_attribution(document, requests)
            self.assertEqual(len(client.calls), 5)
            audit = result["provenance"]["attribution_review"]
            self.assertEqual(audit["scans"][0]["truncation_observation"],
                             {"origin": "prior_run", "prior_observation_fingerprint": observation, "replayed_hint": True})
            self.assertEqual(audit["scans"][0]["request_fingerprint"], fingerprint(payload))
            self.assertEqual(audit["applied_blocks"], 4)
            for options in ({"prior_observation_fingerprint": "bad"}, {"prior_observation_fingerprint": 3}):
                with self.assertRaises(ValueError):
                    record_scan_truncation(requests, payload, **options)
            with self.assertRaises(ValueError):
                record_scan_truncation(requests, dict(payload, mode="attribution_review_vote"),
                                       prior_observation_fingerprint=observation)
        self.assertEqual(validate_document(result), [])

    def test_child_cannot_correct_promoted_read_only_context(self):
        def modify(response, payload):
            if payload["mode"] == "attribution_review_scan":
                if len(payload["target_units"]) == 4:
                    raise ProviderError(TRUNCATED)
                response["corrections"] = [correction(payload["context_units"][0])]
            return response
        requests = Requests(modify=modify)
        with self.assertRaisesRegex(ValueError, "context blocks"):
            audit_attribution(fixture(count=4), requests)
        self.assertEqual(len(requests.calls), 2)

    def test_subdivision_stops_at_two_levels_or_single_block(self):
        for blocks, expected_sizes in ((8, [8, 4, 2]), (1, [1])):
            def truncate(response, payload):
                raise ProviderError(TRUNCATED)
            requests = Requests(modify=truncate)
            with self.subTest(blocks=blocks), self.assertRaisesRegex(ProviderError, "truncated"):
                audit_attribution(fixture(count=blocks), requests)
            self.assertEqual([len(row["target_units"]) for row in requests.calls], expected_sizes)
            self.assertLessEqual(len(requests.calls), MAX_SCAN_SPLIT_DEPTH + 1)
            self.assertTrue(all(row["mode"] == "attribution_review_scan" for row in requests.calls))

    def test_only_exact_provider_truncation_can_split_and_jury_never_splits(self):
        for error in (ProviderError("Unknown failure"), ProviderError(TRUNCATED, status_code=401),
                      ProviderError(TRUNCATED, code="invalid_api_key"),
                      ProviderError(TRUNCATED + " untrusted detail"), ValueError("Invalid source")):
            def fail(response, payload):
                raise error
            requests = Requests(modify=fail)
            with self.subTest(error=str(error)), self.assertRaises(type(error)) as caught:
                audit_attribution(fixture(count=4), requests)
            self.assertIs(caught.exception, error)
            self.assertEqual(len(requests.calls), 1)
        def jury_failure(response, payload):
            if payload["mode"] == "attribution_review_vote":
                raise ProviderError(TRUNCATED)
            return response
        requests = Requests(modify=jury_failure)
        with self.assertRaisesRegex(ProviderError, "truncated"):
            audit_attribution(fixture(count=4), requests, workers=1)
        self.assertEqual(len(requests.calls), 2)


if __name__ == "__main__":
    unittest.main()
