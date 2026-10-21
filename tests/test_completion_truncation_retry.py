"""An isolated truncated completion retries once without invalidating prior work."""

from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
from threading import Event
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.provider import ProviderError


TRUNCATED = "The model output was truncated. Increase the output token limit or reduce the input."


class Client:
    identity = {"model": "offline-completion-retry", "max_output_tokens": 65536}

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


class CompletionTruncationRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payload = {"mode": "speaker_completion", "target_units": [{"block_id": "b1", "text": "PRIVATE"}]}

    @staticmethod
    def validate(response):
        if response.get("valid") is not True:
            raise ValueError("Invalid response")
        return response

    def test_retry_uses_identical_request_and_reuses_success_without_new_calls(self):
        client = Client(ProviderError(TRUNCATED), {"valid": True})
        progress = []
        requests = CachedRequests(client, self.root, progress=progress.append)
        result = requests.run("system", self.payload, self.validate)
        self.assertEqual(result, {"valid": True})
        self.assertEqual(client.calls, [("system", self.payload)] * 2)
        self.assertEqual(requests.run("system", self.payload, self.validate), result)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(requests.cache_hits, 1)
        self.assertEqual(len(list(self.root.glob("*.json"))), 1)
        self.assertIn("retrying the identical request once", progress[0])
        self.assertTrue(all("PRIVATE" not in item for item in progress))

    def test_second_truncation_propagates_without_caching_partial_result(self):
        client = Client(ProviderError(TRUNCATED), ProviderError(TRUNCATED))
        requests = CachedRequests(client, self.root, repair_attempts=4)
        with self.assertRaisesRegex(ProviderError, "truncated"):
            requests.run("system", self.payload, self.validate)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(list(self.root.rglob("*.json")), [])

    def test_retry_budget_is_shared_across_validation_repairs(self):
        client = Client(ProviderError(TRUNCATED), {"valid": False}, ProviderError(TRUNCATED))
        requests = CachedRequests(client, self.root, repair_attempts=4)
        with self.assertRaisesRegex(ProviderError, "truncated"):
            requests.run("system", self.payload, self.validate)
        self.assertEqual(len(client.calls), 3)
        self.assertIn("correction", client.calls[-1][1])
        self.assertEqual(len(list((self.root / "failed").glob("*.json"))), 1)
        self.assertEqual(list(self.root.glob("*.json")), [])

    def test_truncated_validation_repair_retries_that_exact_repair(self):
        client = Client({"valid": False}, ProviderError(TRUNCATED), {"valid": True})
        requests = CachedRequests(client, self.root, repair_attempts=1)
        requests.run("system", self.payload, self.validate)
        self.assertEqual(len(client.calls), 3)
        self.assertNotEqual(client.calls[0], client.calls[1])
        self.assertEqual(client.calls[1], client.calls[2])

    def test_other_requests_and_errors_do_not_retry(self):
        for payload, message in (
            ({"target_units": self.payload["target_units"]}, TRUNCATED),
            (dict(self.payload, mode="attribution_semantic_scan"), TRUNCATED),
            (dict(self.payload, target_units=self.payload["target_units"] * 2), TRUNCATED),
            (self.payload, "The API request failed or timed out."),
            (self.payload, "The model output was blocked by the provider's content filter."),
        ):
            with self.subTest(payload=payload, message=message):
                client = Client(ProviderError(message))
                requests = CachedRequests(client, self.root)
                with self.assertRaises(ProviderError):
                    requests.run("system", payload, self.validate)
                self.assertEqual(len(client.calls), 1)

    def test_cancellation_between_generations_prevents_retry(self):
        cancelled = Event()
        client = Client(ProviderError(TRUNCATED))
        requests = CachedRequests(client, self.root, progress=lambda message: cancelled.set())
        with self.assertRaises(CancelledError):
            requests.run("system", self.payload, self.validate, cancelled=cancelled)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(list(self.root.rglob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
