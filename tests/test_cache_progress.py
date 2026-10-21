"""Resume progress distinguishes validated local reuse from fresh model calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests, fingerprint, write_json
from epub_tts_narrator.prompts import PROMPT_VERSION


class Client:
    identity = {"model": "offline-cache-progress"}

    def __init__(self, response=None):
        self.response = response or {"valid": True}
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        return deepcopy(self.response)


class CacheProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def cache(self, payload, response):
        key = fingerprint({"provider": Client.identity, "prompt_version": PROMPT_VERSION,
                           "system": "original system", "payload": payload})
        write_json(self.root / f"{key}.json", response)

    @staticmethod
    def validate(response):
        if response.get("valid") is not True:
            raise ValueError("PRIVATE rejected text must not appear in progress")
        return response

    def test_first_and_periodic_cache_reuse_is_reported_without_api_calls(self):
        for number in range(51):
            self.cache({"index": number}, {"valid": True, "private": "PRIVATE source material"})
        client, progress = Client(), []
        requests = CachedRequests(client, self.root, progress=progress.append)
        for number in range(51):
            requests.run("original system", {"index": number}, self.validate)
        self.assertEqual(requests.cache_hits, 51)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(progress), 3)
        for message, count in zip(progress, (1, 25, 50)):
            self.assertIn(f"Cache reuse: {count} ", message)
            self.assertIn("no API calls for these responses", message)
            self.assertNotIn("PRIVATE", message)

    def test_concurrent_cache_hits_count_only_this_run(self):
        for number in range(75):
            self.cache({"index": number}, {"valid": True})
        client, progress = Client(), []
        requests = CachedRequests(client, self.root, progress=progress.append)
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda n: requests.run("original system", {"index": n}, self.validate), range(50)))
        self.assertEqual(requests.cache_hits, 50)
        self.assertEqual(client.calls, [])
        self.assertEqual([int(message.split(": ")[1].split()[0]) for message in progress], [1, 25, 50])

    def test_corrupted_cache_is_reported_and_valid_replacement_is_saved(self):
        payload = {"index": 1}
        self.cache(payload, {"valid": False, "private": "PRIVATE response"})
        client, progress = Client(), []
        requests = CachedRequests(client, self.root, progress=progress.append)
        accepted = requests.run("original system", payload, self.validate)
        self.assertEqual(accepted, {"valid": True})
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(requests.cache_hits, 0)
        self.assertEqual(progress, ["Cached response failed current validation; requesting a replacement."])
        requests.run("original system", payload, self.validate)
        self.assertEqual(requests.cache_hits, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(all("PRIVATE" not in message for message in progress))

    def test_first_fresh_call_is_not_reported_as_cache_reuse(self):
        client, progress = Client(), []
        requests = CachedRequests(client, self.root, progress=progress.append)
        requests.run("original system", {"index": 1}, self.validate)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(requests.cache_hits, 0)
        self.assertEqual(progress, [])
