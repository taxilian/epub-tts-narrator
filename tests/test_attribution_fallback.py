"""Raw attribution subdivision never accepts partial or out-of-scope output."""

from concurrent.futures import CancelledError
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock, get_ident
import unittest

from epub_tts_narrator.attribution_fallback import (
    MAX_ATTRIBUTION_SPLIT_DEPTH, TRUNCATED_OUTPUT, attribute_with_subdivision,
    record_attribution_truncation,
)
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import (
    CachedRequests, DEFAULT_POLICY, align_attribution_response, analyze, base_document,
    fingerprint, make_units, seed_cast,
)
from epub_tts_narrator.prompts import ATTRIBUTE, PROMPT_VERSION
from epub_tts_narrator.provider import ProviderError
from test_contracts import valid_document
from test_pipeline import FakeClient
from test_simultaneous_attribution import group_correction, simultaneous_fixture


def fixture(count=6, *, text=None):
    source = deepcopy(valid_document()["source"])
    template = source["blocks"][0]
    source["blocks"] = [dict(template, id=f"b{index:03d}",
                             text=text or f"Narrative paragraph {index}.")
                        for index in range(count)]
    source["chapters"][0]["block_ids"] = [block["id"] for block in source["blocks"]]
    units = [unit for batch in make_units(source) for unit in batch]
    cast = seed_cast()
    return source, cast, {
        "final_cast": cast, "performance_policy": deepcopy(DEFAULT_POLICY),
        "target_units": units, "context_units": [], "prior_segments": [],
        "scene_summary": {"summary": "Offline narrative fixture."},
    }


def response(payload):
    return {"units": [{"unit_id": unit["unit_id"], "segments": [{
        "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
        "basis": "explicit", "reason": "Narrative frame.", "evidence": [],
        "candidates": [], "review_required": False, "delivery": {"direction": "", "evidence": []},
    }]} for unit in payload["target_units"]]}


def cache_key(client, payload, system=ATTRIBUTE):
    return fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                        "system": system, "payload": payload})


class Client:
    identity = {"base_url": "https://offline.invalid/v1", "model": "fixture"}

    def __init__(self, maximum=2, modify=None):
        self.maximum = maximum
        self.modify = modify
        self.calls = []
        self.threads = []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        self.threads.append(get_ident())
        if len(payload["target_units"]) > self.maximum:
            raise ProviderError(TRUNCATED_OUTPUT)
        result = response(payload)
        return self.modify(result, payload) if self.modify else result


class AttributionFallbackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"

    def run_request(self, source, cast, payload, client, **kwargs):
        requests = CachedRequests(client, self.cache, repair_attempts=0)
        return attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload, **kwargs)

    def test_two_levels_preserve_whole_units_context_order_source_and_child_provenance(self):
        source, cast, payload = fixture(10)
        visible = deepcopy(payload["target_units"])
        payload["target_units"] = visible[1:-1]
        payload["context_units"] = [visible[0], visible[-1]]
        original = deepcopy((source, cast, payload))
        client = Client(maximum=2)
        segments, record = self.run_request(source, cast, payload, client)
        self.assertEqual([len(row["target_units"]) for row in client.calls], [8, 4, 2, 2, 4, 2, 2])
        self.assertEqual(len(set(client.threads)), 1)
        all_ids = {unit["unit_id"] for unit in visible}
        for row in client.calls:
            target_ids = {unit["unit_id"] for unit in row["target_units"]}
            context_ids = [unit["unit_id"] for unit in row["context_units"]]
            self.assertEqual(target_ids | set(context_ids), all_ids)
            self.assertFalse(target_ids & set(context_ids))
            self.assertEqual(context_ids, sorted(set(context_ids)))
            for unit in row["target_units"] + row["context_units"]:
                self.assertEqual(unit, next(original for original in visible if original["unit_id"] == unit["unit_id"]))
        self.assertEqual((source, cast, payload), original)
        self.assertEqual([item["text"] for item in segments], [unit["text"] for unit in visible[1:-1]])
        self.assertEqual(record["original_attempt_status"], "truncated")
        self.assertEqual(record["request_cache_fingerprint"], cache_key(client, payload))
        self.assertEqual(record["max_split_depth"], MAX_ATTRIBUTION_SPLIT_DEPTH)
        leaves = [leaf for child in record["children"] for leaf in child["children"]]
        self.assertEqual(len(leaves), 4)
        self.assertTrue(all(leaf["status"] == "validated" and leaf["subdivision_depth"] == 2 for leaf in leaves))
        self.assertEqual([unit for leaf in leaves for unit in leaf["target_unit_ids"]],
                         [unit["unit_id"] for unit in visible[1:-1]])
        self.assertEqual({leaf["request_cache_fingerprint"] for leaf in leaves},
                         {cache_key(client, row) for row in client.calls if len(row["target_units"]) == 2})
        self.assertFalse((self.cache / f"{cache_key(client, payload)}.json").exists())
        self.assertTrue(all((self.cache / f"{leaf['request_cache_fingerprint']}.json").exists() for leaf in leaves))

    def test_interrupted_fallback_reuses_normal_batch_and_valid_child_cache(self):
        source, cast, payload = fixture(6)
        normal = dict(payload, target_units=payload["target_units"][:2])
        split = dict(payload, target_units=payload["target_units"][2:])
        fail = True
        def modify(result, current):
            if current["target_units"][0]["unit_id"] == split["target_units"][2]["unit_id"] and fail:
                raise ProviderError("temporary failure")
            return result
        client = Client(modify=modify)
        normal_result = self.run_request(source, cast, normal, client)
        self.assertIsNone(normal_result[1])
        with self.assertRaisesRegex(ProviderError, "temporary failure"):
            self.run_request(source, cast, split, client)
        self.assertEqual(len(client.calls), 4)
        self.assertTrue((self.cache / f"{cache_key(client, normal)}.json").exists())
        fail = False
        self.assertEqual(self.run_request(source, cast, normal, client), normal_result)
        segments, record = self.run_request(source, cast, split, client)
        self.assertEqual(len(client.calls), 5)  # Only the previously failed right child.
        self.assertTrue(record["truncation_observation"]["replayed_hint"])
        self.assertEqual(self.run_request(source, cast, split, client), (segments, record))
        self.assertEqual(len(client.calls), 5)

    def test_simultaneous_performers_survive_subdivision_combination_and_cache_replay(self):
        document = simultaneous_fixture()
        source, cast = document["source"], document["cast"]
        units = [unit for batch in make_units(source) for unit in batch]
        payload = {"final_cast": cast, "performance_policy": deepcopy(DEFAULT_POLICY),
                   "target_units": units, "context_units": [], "prior_segments": [],
                   "scene_summary": {"summary": "Mara and Lena speak together."}}
        before = deepcopy((source, cast, payload))

        def modify(result, current):
            for row, unit in zip(result["units"], current["target_units"]):
                if unit["block_id"] == "b1":
                    row["segments"] = group_correction(document)["segments"]
            return result

        client = Client(maximum=1, modify=modify)
        segments, record = self.run_request(source, cast, payload, client, sequential_prior=True)
        self.assertEqual([len(row["target_units"]) for row in client.calls], [2, 1, 1])
        expected = [{"role_id": "mara", "voice_id": "v1"}, {"role_id": "lena", "voice_id": "v2"}]
        self.assertEqual(segments[0]["simultaneous_speakers"], expected)
        self.assertIsNone(segments[0]["role_id"])
        self.assertIsNone(segments[0]["voice_id"])
        self.assertEqual(client.calls[-1]["prior_segments"][0]["simultaneous_speakers"], expected)
        self.assertEqual(validate_document(base_document(source, cast, segments)), [])
        self.assertEqual((source, cast, payload), before)
        self.assertEqual(record["status"], "subdivided")
        self.assertEqual(record["segment_count"], len(segments))
        self.assertTrue(all(child["status"] == "validated" for child in record["children"]))
        self.assertEqual(self.run_request(source, cast, payload, client, sequential_prior=True)[0], segments)
        self.assertEqual(len(client.calls), 3)

    def test_depth_and_singleton_failures_are_bounded(self):
        for count, maximum, expected_calls in [(8, 1, 3), (1, 0, 1)]:
            with self.subTest(count=count):
                source, cast, payload = fixture(count)
                client = Client(maximum=maximum)
                with tempfile.TemporaryDirectory() as cache:
                    requests = CachedRequests(client, cache, repair_attempts=0)
                    with self.assertRaisesRegex(ProviderError, "model output was truncated"):
                        attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload)
                self.assertEqual(len(client.calls), expected_calls)
                self.assertEqual(min(len(row["target_units"]) for row in client.calls), 2 if count == 8 else 1)

    def test_only_exact_uncoded_provider_truncation_splits(self):
        for failure in [
            ValueError(TRUNCATED_OUTPUT), ProviderError("The model output was truncated"),
            ProviderError("Incomplete response"), ProviderError(TRUNCATED_OUTPUT, status_code=401),
            ProviderError(TRUNCATED_OUTPUT, code="server_error"),
        ]:
            with self.subTest(type=type(failure).__name__, code=getattr(failure, "code", None)):
                source, cast, payload = fixture(4)
                class Failing(Client):
                    def complete(self, system, payload):
                        self.calls.append(deepcopy(payload))
                        raise failure
                client = Failing()
                with tempfile.TemporaryDirectory() as cache:
                    requests = CachedRequests(client, cache, repair_attempts=0)
                    with self.assertRaises(type(failure)):
                        attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload)
                    self.assertFalse((Path(cache) / "attribution-subdivisions").exists())
                self.assertEqual(len(client.calls), 1)

    def test_context_output_reordering_and_rewritten_source_are_rejected(self):
        for mutation in ["context", "reordered", "rewritten"]:
            with self.subTest(mutation=mutation):
                source, cast, payload = fixture(6)
                def modify(result, current):
                    if mutation == "context":
                        result["units"] += response(dict(current, target_units=current["context_units"][:1]))["units"]
                    elif mutation == "reordered":
                        result["units"].reverse()
                    else:
                        result["units"][0]["segments"][0]["text"] = "Changed source."
                    return result
                client = Client(maximum=3, modify=modify)
                with tempfile.TemporaryDirectory() as cache:
                    requests = CachedRequests(client, cache, repair_attempts=0)
                    with self.assertRaisesRegex(ValueError, "failed validation"):
                        attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload)
                    self.assertEqual(list(Path(cache).glob("*.json")), [])
                self.assertEqual(len(client.calls), 2)

    def test_multiple_units_from_one_block_are_kept_distinct_and_revalidated(self):
        source, cast, payload = fixture(1, text="One long narrative sentence. " * 31)
        units = [unit for batch in make_units(source, 256) for unit in batch]
        self.assertEqual(len(units), 4)
        payload["target_units"] = units
        client = Client(maximum=1)
        segments, record = self.run_request(source, cast, payload, client)
        self.assertEqual("".join(item["text"] for item in segments), source["blocks"][0]["text"])
        self.assertEqual(validate_document(base_document(source, cast, segments)), [])
        self.assertEqual(record["segment_count"], 4)
        for call in client.calls[1:]:
            all_units = call["target_units"] + call["context_units"]
            self.assertEqual({unit["unit_id"] for unit in all_units}, {unit["unit_id"] for unit in units})
            self.assertEqual(len(all_units), 4)

    def test_serial_prior_segments_advance_but_independent_children_keep_empty_prior(self):
        source, cast, payload = fixture(10)
        prior_units = payload["target_units"][:6]
        prior = align_attribution_response(source, cast, prior_units, response(dict(payload, target_units=prior_units)))
        payload["target_units"] = payload["target_units"][6:]
        payload["prior_segments"] = prior
        client = Client()
        _, record = self.run_request(source, cast, payload, client, sequential_prior=True)
        left = align_attribution_response(source, cast, client.calls[1]["target_units"], response(client.calls[1]))
        self.assertEqual(client.calls[1]["prior_segments"], prior)
        self.assertEqual(client.calls[2]["prior_segments"], (prior + left)[-6:])
        self.assertEqual(record["prior_segment_mode"], "serial")
        payload["prior_segments"] = []
        concurrent = Client()
        _, record = self.run_request(source, cast, payload, concurrent)
        self.assertTrue(all(call["prior_segments"] == [] for call in concurrent.calls))
        self.assertEqual(record["prior_segment_mode"], "independent")

    def test_repeated_run_source_restoration_survives_combined_validation_and_cache_replay(self):
        source, cast, payload = fixture(4, text="The voice ssssaid nothing else.")
        def modify(result, current):
            for unit in result["units"]:
                unit["segments"][0]["text"] = unit["segments"][0]["text"].replace("ssss", "sss")
            return result
        client = Client(modify=modify)
        segments, _ = self.run_request(source, cast, payload, client)
        self.assertEqual(validate_document(base_document(source, cast, segments)), [])
        self.assertTrue(all(item["attribution"]["review_required"] for item in segments))
        self.assertTrue(all(item["source_restoration"]["changes"][0]["source_count"] == 4 for item in segments))
        for call in client.calls[1:]:
            raw = json.loads((self.cache / f"{cache_key(client, call)}.json").read_text())
            self.assertIn("sssaid", raw["units"][0]["segments"][0]["text"])
            self.assertNotIn("source_restoration", raw["units"][0]["segments"][0])
        self.assertEqual(self.run_request(source, cast, payload, client)[0], segments)
        self.assertEqual(len(client.calls), 3)

    def test_cancellation_after_first_leaf_stops_next_child(self):
        source, cast, payload = fixture(4)
        cancelled = Event()
        def modify(result, current):
            cancelled.set()
            return result
        client = Client(modify=modify)
        with self.assertRaises(CancelledError):
            self.run_request(source, cast, payload, client, cancelled=cancelled)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 1)  # The validated left child only.

    def test_imported_exact_request_hint_skips_parent_and_success_cache_takes_precedence(self):
        source, cast, payload = fixture(4)
        client = Client()
        requests = CachedRequests(client, self.cache, repair_attempts=0)
        observation = fingerprint("Verified offline log + exact captured request")
        marker = record_attribution_truncation(requests, ATTRIBUTE, payload,
                                             prior_observation_fingerprint=observation)
        self.assertEqual(client.calls, [])
        self.assertEqual(marker["request_cache_fingerprint"], cache_key(client, payload))
        segments, record = attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(record["truncation_observation"],
                         {"origin": "prior_run", "prior_observation_fingerprint": observation, "replayed_hint": True})
        # A deliberately obtained successful parent response overrides a hint.
        client.maximum = 4
        parent = requests.run(ATTRIBUTE, payload, lambda raw: align_attribution_response(source, cast, payload["target_units"], raw))
        self.assertEqual(parent, segments)
        self.assertEqual(attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload), (parent, None))
        self.assertEqual(len(client.calls), 3)

    def test_corrupt_stale_and_symlink_hints_are_ignored_and_bad_imports_rejected(self):
        source, cast, payload = fixture(4)
        client = Client(maximum=4)
        for corrupt in ("garbage", "stale", "oversized", "symlink"):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory)
                requests = CachedRequests(client, cache, repair_attempts=0)
                marker = record_attribution_truncation(requests, ATTRIBUTE, payload)
                path = cache / "attribution-subdivisions" / f"{cache_key(client, payload)}.json"
                if corrupt == "garbage":
                    path.write_text("{")
                elif corrupt == "stale":
                    marker["request_fingerprint"] = "0" * 64
                    path.write_text(json.dumps(marker))
                elif corrupt == "oversized":
                    path.write_text(" " * 2049 + json.dumps(marker))
                else:
                    target = cache / "other.json"
                    path.rename(target)
                    path.symlink_to(target)
                before = len(client.calls)
                self.assertIsNone(attribute_with_subdivision(source, cast, requests, ATTRIBUTE, payload)[1])
                self.assertEqual(len(client.calls), before + 1)
        requests = CachedRequests(client, self.cache, repair_attempts=0)
        for bad in ("not-a-hash", {"secret": "not-an-id"}):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                record_attribution_truncation(requests, ATTRIBUTE, payload, prior_observation_fingerprint=bad)
        duplicate = dict(payload, target_units=[payload["target_units"][0]] * 2)
        with self.assertRaisesRegex(ValueError, "distinct target units"):
            record_attribution_truncation(requests, ATTRIBUTE, duplicate)

    def test_pipeline_five_workers_bound_children_and_reuse_all_caches(self):
        source, _, _ = fixture(30, text="Mara walked slowly along the path. " * 2)
        gate, lock = Barrier(5), Lock()
        class ParallelClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.active = self.maximum = 0
            def attribute(self, payload):
                with lock:
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                try:
                    if len(payload["target_units"]) > 2:
                        # Ten parent batches; the first five arrive together.
                        if payload["target_units"][0]["unit_id"] <= "u0000013":
                            gate.wait(timeout=5)
                        raise ProviderError(TRUNCATED_OUTPUT)
                    return super().attribute(payload)
                finally:
                    with lock:
                        self.active -= 1
        client = ParallelClient()
        result = analyze(source, client, self.cache, chunk_chars=256, whole_book_discovery=True,
                         attribution_workers=5, identity_consensus=False, repair_attempts=0)
        self.assertEqual(validate_document(result), [])
        self.assertEqual(client.maximum, 5)
        records = result["provenance"]["attribution_subdivisions"]
        self.assertEqual(len(records), 10)
        self.assertEqual([row["batch_index"] for row in records], list(range(1, 11)))
        calls = len(client.calls)
        resumed = analyze(source, client, self.cache, chunk_chars=256, whole_book_discovery=True,
                          attribution_workers=5, identity_consensus=False, repair_attempts=0)
        self.assertEqual(len(client.calls), calls)
        self.assertEqual(resumed["segments"], result["segments"])
        self.assertTrue(all(row["truncation_observation"]["replayed_hint"]
                            for row in resumed["provenance"]["attribution_subdivisions"]))

    def test_pipeline_child_failure_redacts_transport_details(self):
        source = deepcopy(valid_document()["source"])
        class Failing(FakeClient):
            def attribute(self, payload):
                if len(payload["target_units"]) > 1:
                    raise ProviderError(TRUNCATED_OUTPUT)
                raise ProviderError("secret-url credential-token private-book")
        with self.assertRaises(ProviderError) as caught:
            analyze(source, Failing(), self.cache, identity_consensus=False, repair_attempts=0)
        message = str(caught.exception)
        self.assertIn("Attribution batch 1/1", message)
        self.assertIn("Provider request failed", message)
        for secret in ("secret-url", "credential-token", "private-book"):
            self.assertNotIn(secret, message)


if __name__ == "__main__":
    unittest.main()
