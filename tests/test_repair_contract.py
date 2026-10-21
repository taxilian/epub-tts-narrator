"""Repair hints retain complete attribution scope without invalidating caches."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_audiobook.pipeline import CachedRequests, align_units, fingerprint, make_units, seed_cast, write_json
from epub_audiobook.prompts import ATTRIBUTE, PROMPT_VERSION
from test_contracts import valid_document


class Client:
    identity = {"model": "offline-repair-contract"}

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        return deepcopy(self.responses[len(self.calls) - 1])


class RepairContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        self.source = deepcopy(valid_document()["source"])
        template = self.source["blocks"][0]
        self.source["blocks"] = [dict(template, id=f"b{index:03}", text=f'"Sssstay," said the rider {index}.')
                                 for index in range(3)]
        self.source["chapters"][0]["block_ids"] = [row["id"] for row in self.source["blocks"]]
        self.units = [unit for batch in make_units(self.source) for unit in batch]
        self.cast = seed_cast()
        self.payload = {"final_cast": self.cast, "target_units": self.units,
                        "context_units": [], "prior_segments": []}
        self.response = {"units": [{"unit_id": unit["unit_id"], "segments": [{
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Synthetic fixture.", "evidence": [],
            "candidates": [], "review_required": False, "delivery": {"direction": "", "evidence": []},
        }]} for unit in self.units]}

    def validate(self, response):
        return align_units(self.source, self.cast, self.units, response)

    def key(self):
        return fingerprint({"provider": Client.identity, "prompt_version": PROMPT_VERSION,
                            "system": ATTRIBUTE, "payload": self.payload})

    @staticmethod
    def size(system, payload):
        return len(system) + len(json.dumps(payload, ensure_ascii=False))

    def test_repair_includes_full_contract_but_first_request_and_cache_key_are_unchanged(self):
        partial = {"units": self.response["units"][:1]}
        client = Client([partial, self.response])
        original = deepcopy(self.payload)
        requests = CachedRequests(client, self.cache, repair_attempts=1)
        result = requests.run(ATTRIBUTE, self.payload, self.validate)
        self.assertEqual(result, self.validate(self.response))
        self.assertEqual(self.payload, original)
        self.assertEqual(client.calls[0], (ATTRIBUTE, original))
        self.assertEqual(client.calls[1][0], ATTRIBUTE)
        repair = client.calls[1][1]
        self.assertEqual({key: value for key, value in repair.items() if key != "correction"}, original)
        correction = repair["correction"]
        self.assertEqual(correction["previous_response"], partial)
        self.assertEqual(correction["target_unit_count"], len(self.units))
        self.assertEqual(correction["target_unit_ids"], [unit["unit_id"] for unit in self.units])
        self.assertIn("including unaffected units", correction["instruction"])
        self.assertIn("not a patch", correction["instruction"])
        self.assertIn("Do not emit context_units or prior_segments", correction["instruction"])
        guidance = " ".join(correction["guidance"])
        self.assertIn("straight/curly", guidance)
        self.assertIn("unusual spelling", guidance)
        self.assertIn("repeated letters", guidance)
        self.assertIn('EVERY delivery object', guidance)
        self.assertIn('{"direction":"","evidence":[]}', guidance)
        self.assertIn("Copy evidence verbatim", guidance)
        self.assertIn("short, sufficient exact substring", guidance)
        self.assertEqual(json.loads((self.cache / f"{self.key()}.json").read_text()), self.response)

    def test_partial_repairs_remain_rejected_with_the_same_bounded_call_limit(self):
        partial = {"units": self.response["units"][:1]}
        client = Client([partial, partial, partial])
        requests = CachedRequests(client, self.cache, repair_attempts=2)
        with self.assertRaisesRegex(ValueError, "failed validation after 3 attempt.*Expected 3 unit"):
            requests.run(ATTRIBUTE, self.payload, self.validate)
        self.assertEqual(len(client.calls), 3)
        self.assertFalse((self.cache / f"{self.key()}.json").exists())
        self.assertEqual(len(list((self.cache / "failed").glob("*.json"))), 3)
        self.assertTrue(all(call[1]["target_units"] == self.units for call in client.calls))

    def test_repair_guidance_does_not_accept_changed_source(self):
        altered = deepcopy(self.response)
        altered["units"][0]["segments"][0]["text"] = self.units[0]["text"].replace('"', '“')
        client = Client([altered, altered])
        requests = CachedRequests(client, self.cache, repair_attempts=1)
        with self.assertRaisesRegex(ValueError, "Text changed or omitted"):
            requests.run(ATTRIBUTE, self.payload, self.validate)
        self.assertEqual(len(client.calls), 2)
        self.assertFalse((self.cache / f"{self.key()}.json").exists())

    def test_existing_success_is_revalidated_and_replayed_without_a_request(self):
        write_json(self.cache / f"{self.key()}.json", self.response)
        write_json(self.cache / "failed" / f"{self.key()}-attempt1.json", {"units": []})
        client = Client([])
        requests = CachedRequests(client, self.cache)
        self.assertEqual(requests.run(ATTRIBUTE, self.payload, self.validate), self.validate(self.response))
        self.assertEqual(client.calls, [])

    def test_failed_exact_request_resumes_with_new_contract_and_does_not_promote_failure(self):
        partial = {"units": self.response["units"][:1]}
        path = self.cache / "failed" / f"{self.key()}-attempt2.json"
        write_json(path, partial)
        client = Client([self.response])
        requests = CachedRequests(client, self.cache, repair_attempts=0)
        self.assertEqual(requests.run(ATTRIBUTE, self.payload, self.validate), self.validate(self.response))
        self.assertEqual(len(client.calls), 1)
        correction = client.calls[0][1]["correction"]
        self.assertEqual(correction["previous_response"], partial)
        self.assertEqual(correction["target_unit_count"], 3)
        self.assertEqual(json.loads(path.read_text()), partial)

    def test_discovery_and_semantic_modes_keep_original_schema_and_upsert_scope(self):
        requests = CachedRequests(Client([]), self.cache)
        for mode in ("discovery", "reconciliation", "attribution_review_scan", "attribution_review_vote"):
            with self.subTest(mode=mode):
                payload = dict(self.payload, mode=mode)
                repaired = requests._repair_payload("System", payload, {}, ValueError("Invalid evidence quote"))
                correction = repaired["correction"]
                self.assertIn("original schema and scope", correction["instruction"])
                self.assertIn("upsert contract", correction["instruction"])
                self.assertNotIn("units array", correction["instruction"])
                self.assertNotIn("target_unit_ids", correction)
                self.assertNotIn("target_unit_count", correction)
                self.assertNotIn("delivery", " ".join(correction["guidance"]))
                self.assertIn("Copy evidence verbatim", " ".join(correction["guidance"]))

    def test_budget_drops_previous_response_before_duplicate_ids_and_guidance(self):
        system, error = "System", ValueError("Invalid evidence quote")
        previous = {"units": [], "large": "x" * 50000}
        requests = CachedRequests(Client([]), self.cache)
        full = requests._repair_payload(system, self.payload, previous, error)
        without_previous = deepcopy(full)
        without_previous["correction"].pop("previous_response")
        without_ids = deepcopy(without_previous)
        without_ids["correction"].pop("target_unit_ids")
        without_guidance = deepcopy(without_ids)
        without_guidance["correction"].pop("guidance")
        for expected in (without_previous, without_ids, without_guidance):
            with self.subTest(keys=sorted(expected["correction"])):
                budget = self.size(system, expected)
                requests.max_request_chars = budget
                repaired = requests._repair_payload(system, self.payload, previous, error)
                self.assertEqual(repaired, expected)
                self.assertLessEqual(self.size(system, repaired), budget)
                self.assertEqual({key: value for key, value in repaired.items() if key != "correction"}, self.payload)

    def test_tight_budgets_keep_base_source_and_use_compact_complete_response_instruction(self):
        system, error = "System", ValueError("Invalid evidence quote. " * 200)
        requests = CachedRequests(Client([]), self.cache)
        base_size = self.size(system, self.payload)
        for additional in (0, 25, 100, 240, 400, 800, 1500):
            with self.subTest(additional=additional):
                requests.max_request_chars = base_size + additional
                repaired = requests._repair_payload(system, self.payload, self.response, error)
                self.assertLessEqual(self.size(system, repaired), requests.max_request_chars)
                self.assertEqual({key: value for key, value in repaired.items() if key != "correction"}, self.payload)
                if "correction" in repaired:
                    self.assertIn("complete", repaired["correction"]["instruction"])
                    self.assertIn("unaffected", repaired["correction"]["instruction"])
        self.assertIs(requests._repair_payload("x" * requests.max_request_chars, self.payload,
                                              self.response, error), self.payload)


if __name__ == "__main__":
    unittest.main()
