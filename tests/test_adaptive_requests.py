"""Adaptive partitioning preserves exact coverage and resumes failed parents."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.adaptive_requests import capacity_error, request_size, run_partitioned
from epub_tts_narrator.pipeline import CachedRequests, fingerprint
from epub_tts_narrator.prompts import PROMPT_VERSION
from epub_tts_narrator.provider import OutputTruncatedError, ProviderError, ResponseCapacityError


TRUNCATED = "The model output was truncated. Increase the output token limit or reduce the input."


class Client:
    def __init__(self, maximum=1, failure=OutputTruncatedError):
        self.maximum = maximum
        self.failure = failure
        self.calls = []
        self.identity = {"model": "offline", "base_url": "https://example.invalid"}

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if len(payload["items"]) > self.maximum:
            raise self.failure(TRUNCATED)
        return {"items": deepcopy(payload["items"])}


def build(items):
    return "Adaptive test", {"items": list(items)}


def validate(response):
    if not isinstance(response, dict) or set(response) != {"items"} or not isinstance(response["items"], list):
        raise ValueError("response.items must be an array")
    return response["items"]


class AdaptiveRequestTests(unittest.TestCase):
    def test_root_success_returns_one_validated_leaf_without_partitioning(self):
        calls = []

        def execute(system, payload):
            calls.append(payload["items"])
            return validate({"items": payload["items"]})

        leaves = run_partitioned([1, 2, 3], build, execute, fits=lambda system, payload: True)

        self.assertEqual(leaves, [([1, 2, 3], [1, 2, 3])])
        self.assertEqual(calls, [[1, 2, 3]])

    def test_recursive_capacity_splits_cover_every_item_once_in_original_order(self):
        calls = []

        def execute(system, payload):
            calls.append(list(payload["items"]))
            if len(payload["items"]) > 1:
                raise OutputTruncatedError(TRUNCATED)
            return list(payload["items"])

        leaves = run_partitioned(list(range(9)), build, execute, fits=lambda system, payload: True)

        self.assertEqual(len(calls), 17)
        self.assertEqual(calls[:7], [list(range(9)), [0, 1, 2, 3], [0, 1], [0], [1], [2, 3], [2]])
        self.assertEqual([item for selected, _ in leaves for item in selected], list(range(9)))
        self.assertEqual([item for _, result in leaves for item in result], list(range(9)))
        self.assertTrue(all(len(selected) == 1 for selected, _ in leaves))

    def test_exact_preflight_partitions_without_calling_oversized_parents(self):
        calls = []

        def execute(system, payload):
            calls.append(list(payload["items"]))
            return list(payload["items"])

        largest_leaf = request_size(*build(["cccc"]))
        leaves = run_partitioned(
            ["aa", "b", "cccc"], build, execute,
            fits=lambda system, payload: request_size(system, payload) <= largest_leaf,
        )

        self.assertEqual(calls, [["aa"], ["b"], ["cccc"]])
        self.assertEqual([selected for selected, _ in leaves], [["aa"], ["b"], ["cccc"]])

    def test_stage_fit_predicate_is_enforced_alongside_the_request_cache_budget(self):
        client = Client(maximum=3)
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, directory, repair_attempts=0)

            def execute(system, payload):
                return requests.run(system, payload, validate)

            leaves = run_partitioned([0, 1, 2], build, execute, requests=requests,
                                     fits=lambda system, payload: len(payload["items"]) <= 2)

            self.assertEqual([call["items"] for call in client.calls], [[0], [1, 2]])
            self.assertEqual([item for selected, _ in leaves for item in selected], [0, 1, 2])

    def test_indivisible_preflight_and_capacity_failures_identify_the_request_fields(self):
        with self.assertRaises(ValueError) as caught:
            run_partitioned(["oversized"], build, lambda system, payload: [],
                            fits=lambda system, payload: False)
        self.assertIn("items", str(caught.exception))
        self.assertIn("fits(system, payload)", str(caught.exception))

        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(Client(), directory, max_request_chars=1, repair_attempts=0)
            with self.assertRaises(ValueError) as caught:
                run_partitioned(["oversized"], build, lambda system, payload: [], requests=requests)
            self.assertIn("items", str(caught.exception))
            self.assertIn("requests.max_request_chars", str(caught.exception))

        def execute(system, payload):
            raise ResponseCapacityError("response capacity")

        with self.assertRaisesRegex(ValueError, "system/payload request_size") as caught:
            run_partitioned(["one"], build, execute, fits=lambda system, payload: True)
        self.assertIsInstance(caught.exception.__cause__, ResponseCapacityError)

    def test_noncapacity_failures_propagate_without_splitting_or_creating_markers(self):
        for failure in (ProviderError(TRUNCATED + " private detail"),
                        ProviderError(TRUNCATED, status_code=400),
                        ProviderError("untyped context overflow"), ValueError("validator failed")):
            calls = []

            def execute(system, payload):
                calls.append(list(payload["items"]))
                raise failure

            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                client = Client(maximum=4)
                requests = CachedRequests(client, directory, repair_attempts=0)
                with self.assertRaises(type(failure)) as caught:
                    run_partitioned([1, 2], build, execute, requests=requests)
                self.assertIs(caught.exception, failure)
                self.assertEqual(calls, [[1, 2]])
                self.assertFalse((Path(directory) / "adaptive-subdivisions").exists())

    def test_capacity_classification_accepts_only_explicit_typed_or_legacy_signals(self):
        accepted = [OutputTruncatedError(TRUNCATED), ResponseCapacityError("limit"),
                    ProviderError("context", code="context_length_exceeded", status_code=400),
                    ProviderError(TRUNCATED)]
        rejected = [ProviderError(TRUNCATED + " detail"), ProviderError(TRUNCATED, code="server_error"),
                    ProviderError(TRUNCATED, status_code=401), ProviderError("context_length_exceeded"),
                    ValueError(TRUNCATED)]
        self.assertTrue(all(capacity_error(error) for error in accepted))
        self.assertFalse(any(capacity_error(error) for error in rejected))

    def test_explicit_context_capacity_uses_the_same_sequential_partition_recovery(self):
        def failure(message):
            return ProviderError(message, code="context_length_exceeded", status_code=413)

        client = Client(maximum=2, failure=failure)
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, directory, repair_attempts=0,
                                      progress=progress.append)

            def execute(system, payload):
                return requests.run(system, payload, validate)

            leaves = run_partitioned(list(range(5)), build, execute, requests=requests,
                                     progress=progress.append)

        self.assertEqual([item for selected, _ in leaves for item in selected], list(range(5)))
        self.assertEqual([len(call["items"]) for call in client.calls], [5, 2, 3, 1, 2])
        messages = "\n".join(progress)
        self.assertIn("Provider request failed immediately: HTTP 413; code context_length_exceeded", messages)
        self.assertIn("after provider failure (HTTP 413; code context_length_exceeded)", messages)

    def test_rate_limit_failure_is_reported_before_propagation(self):
        class RateLimited:
            identity = {"model": "offline", "base_url": "https://example.invalid"}

            def complete(self, system, payload):
                raise ProviderError(
                    "private provider detail", status_code=429,
                    code="rate_limit_exceeded", retryable=True,
                    rate_limits={"concurrent_active": 4, "concurrent_max": 4},
                )

        progress = []
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(RateLimited(), directory, repair_attempts=0,
                                      progress=progress.append)
            with self.assertRaises(ProviderError):
                requests.run("system", {"items": [1]}, validate)
        self.assertTrue(any(
            "HTTP 429" in message and "code rate_limit_exceeded" in message
            and "concurrent_active=4" in message for message in progress), progress)
        self.assertNotIn("private", "\n".join(progress))

    def test_resume_skips_a_known_truncated_parent_and_reuses_validated_children(self):
        client = Client(maximum=1)
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, directory, repair_attempts=0)

            def execute(system, payload):
                return requests.run(system, payload, validate)

            original = run_partitioned([0, 1, 2], build, execute, requests=requests)
            parent_payload = client.calls[0]
            parent_key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                      "system": "Adaptive test", "payload": parent_payload})
            marker = Path(directory) / "adaptive-subdivisions" / f"{parent_key}.json"
            self.assertEqual(json.loads(marker.read_text())["request_cache_fingerprint"], parent_key)
            marker_text = marker.read_text()
            self.assertNotIn('"items"', marker_text)
            self.assertNotIn(client.identity["base_url"], marker_text)

            class FailParent(Client):
                def complete(self, system, payload):
                    if payload == parent_payload:
                        raise AssertionError("known truncated parent was requested again")
                    return super().complete(system, payload)

            resumed_client = FailParent(maximum=1)
            requests.client = resumed_client
            resumed = run_partitioned([0, 1, 2], build, execute, requests=requests)
            self.assertEqual(resumed, original)
            self.assertEqual(resumed_client.calls, [])

            client.maximum = 3
            requests.client = client
            parent = requests.run("Adaptive test", parent_payload, validate)
            self.assertEqual(parent, [0, 1, 2])
            before_hits = requests.cache_hits
            root = run_partitioned([0, 1, 2], build, execute, requests=requests,
                                   cache_validator=lambda system, payload, response: validate(response))
            self.assertEqual(root, [([0, 1, 2], [0, 1, 2])])
            self.assertEqual(requests.cache_hits - before_hits, 1)
            self.assertEqual(client.calls[-1], parent_payload)

    def test_invalid_parent_cache_never_reissues_a_known_failed_parent(self):
        client = Client(maximum=1)
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, directory, repair_attempts=0)

            def execute(system, payload):
                return requests.run(system, payload, validate)

            original = run_partitioned([0, 1, 2], build, execute, requests=requests)
            system, payload = build([0, 1, 2])
            key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                               "system": system, "payload": payload})
            (Path(directory) / f"{key}.json").write_text('{"invalid": true}')
            client.calls.clear()

            result = run_partitioned([0, 1, 2], build, execute, requests=requests,
                                     cache_validator=lambda system, payload, response: validate(response))

            self.assertEqual(result, original)
            self.assertEqual(client.calls, [])

    def test_corrupt_stale_oversized_and_symlink_markers_are_not_replay_instructions(self):
        for corruption in ("garbage", "stale", "oversized", "symlink"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                client = Client(maximum=1)
                requests = CachedRequests(client, directory, repair_attempts=0)

                def execute(system, payload):
                    return requests.run(system, payload, validate)

                run_partitioned([0, 1], build, execute, requests=requests)
                parent_payload = client.calls[0]
                key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                   "system": "Adaptive test", "payload": parent_payload})
                marker = Path(directory) / "adaptive-subdivisions" / f"{key}.json"
                if corruption == "garbage":
                    marker.write_text("{")
                elif corruption == "stale":
                    saved = json.loads(marker.read_text())
                    saved["request_fingerprint"] = "0" * 64
                    marker.write_text(json.dumps(saved))
                elif corruption == "oversized":
                    marker.write_text(" " * 2049 + marker.read_text())
                else:
                    target = Path(directory) / "target.json"
                    marker.rename(target)
                    marker.symlink_to(target)
                client.maximum = 2
                before = len(client.calls)
                leaves = run_partitioned([0, 1], build, execute, requests=requests)
                self.assertEqual(leaves, [([0, 1], [0, 1])])
                self.assertEqual(len(client.calls), before + 1)

    def test_empty_input_and_missing_fit_predicate_are_handled_without_model_work(self):
        calls = []
        self.assertEqual(run_partitioned([], build, calls.append, fits=lambda system, payload: True), [])
        with self.assertRaisesRegex(ValueError, "fits"):
            run_partitioned([1], build, calls.append)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
