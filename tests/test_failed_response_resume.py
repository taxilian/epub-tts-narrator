"""Failed material can seed a fresh repair, but never becomes a cache hit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_audiobook.pipeline import CachedRequests, fingerprint, write_json
from epub_audiobook.prompts import PROMPT_VERSION


class Client:
    identity = {"base_url": "https://offline.invalid", "model": "fixture"}

    def __init__(self, response=None):
        self.response = {"valid": True} if response is None else response
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        return deepcopy(self.response)


class FailedResponseResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        self.system, self.payload = "Unchanged application prompt.", {"source": "Canonical source."}
        self.client = Client()

    def key(self, client=None, system=None, payload=None):
        return fingerprint({"provider": (client or self.client).identity, "prompt_version": PROMPT_VERSION,
                            "system": self.system if system is None else system,
                            "payload": self.payload if payload is None else payload})

    def failure(self, value, attempt=1):
        path = self.cache / "failed" / f"{self.key()}-attempt{attempt}.json"
        write_json(path, value, overwrite=True)
        return path

    @staticmethod
    def validate(value):
        if value.get("valid") is not True:
            raise ValueError('Fresh check: characters[6].series_identity.evidence[0] (block_id="b000123") is not exact.')
        return value

    def run_request(self, client=None, validate=None, **kwargs):
        requests = CachedRequests(client or self.client, self.cache, **kwargs)
        return requests.run(self.system, self.payload, validate or self.validate)

    def test_newest_failure_gets_current_validator_feedback_then_success_cache_replays(self):
        old = self.failure({"valid": False, "revision": "older"}, attempt=3)
        latest_raw = {"valid": False, "revision": "latest", "quote": "PRIVATE REJECTED TEXT"}
        latest = self.failure(latest_raw, attempt=1)
        os.utime(old, ns=(100, 100))
        os.utime(latest, ns=(200, 200))
        progress = []
        def current_validator(value):
            if not value.get("valid"):
                value["validator_mutation"] = "must not alter the raw previous response"
            return self.validate(value)
        result = self.run_request(validate=current_validator, repair_attempts=0, progress=progress.append)
        self.assertEqual(result, {"valid": True})
        self.assertEqual(len(self.client.calls), 1)
        system, request = self.client.calls[0]
        self.assertEqual(system, self.system)
        self.assertEqual(request["source"], self.payload["source"])
        self.assertEqual(request["correction"]["previous_response"], latest_raw)
        self.assertIn("characters[6].series_identity.evidence[0]", request["correction"]["validator_error"])
        self.assertIn("Fresh check", request["correction"]["validator_error"])
        self.assertEqual(json.loads(latest.read_text()), latest_raw)
        self.assertTrue(any("Resuming validation repair" in entry for entry in progress))
        self.assertTrue(all("PRIVATE" not in entry and "quote" not in entry for entry in progress))
        repeat = Client()
        self.assertEqual(self.run_request(repeat, repair_attempts=0), result)
        self.assertEqual(repeat.calls, [])

    def test_failures_from_other_prompt_payload_or_provider_are_never_borrowed(self):
        self.failure({"valid": False, "private": "OTHER REQUEST"})
        cases = [
            (Client(), self.system + " Changed.", self.payload),
            (Client(), self.system, {"source": "Different source."}),
            (Client(), self.system, self.payload),
        ]
        cases[2][0].identity = dict(Client.identity, model="different-model")
        for client, system, payload in cases:
            with self.subTest(system=system, payload=payload, provider=client.identity):
                requests = CachedRequests(client, self.cache, repair_attempts=0)
                self.assertEqual(requests.run(system, payload, self.validate), {"valid": True})
                self.assertEqual(client.calls, [(system, payload)])

    def test_corrupt_nonobject_nonfinite_oversized_and_misnamed_failures_are_ignored(self):
        path = self.failure({"valid": False})
        for content in (b"{", b"\xff", b"[]", b'{"number":NaN}', b"x" * 257):
            with self.subTest(content=content[:20]):
                path.write_bytes(content)
                success = self.cache / f"{self.key()}.json"
                success.unlink(missing_ok=True)
                client = Client()
                with patch("epub_audiobook.pipeline.MAX_RESPONSE_BYTES", 256):
                    self.run_request(client, repair_attempts=0)
                self.assertEqual(client.calls[0][1], self.payload)
        path.unlink()
        (self.cache / f"{self.key()}.json").unlink()
        for name in (f"{self.key()}-attempt0.json", f"{self.key()}-attempt01.json",
                     f"{self.key()}-attempt1-extra.json", f"{self.key()}-attemptx.json"):
            (path.parent / name).write_text('{"valid":false}')
        client = Client()
        self.run_request(client, repair_attempts=0)
        self.assertEqual(client.calls[0][1], self.payload)

    def test_symlink_failure_is_never_read_into_repair_context(self):
        external = self.cache / "unrelated.json"
        external.write_text('{"valid":false,"private":"UNRELATED FILE"}')
        directory = self.cache / "failed"
        directory.mkdir()
        (directory / f"{self.key()}-attempt1.json").symlink_to(external)
        self.run_request(repair_attempts=0)
        self.assertEqual(self.client.calls, [(self.system, self.payload)])

    def test_deeply_nested_failed_hint_is_ignored_without_masking_current_validation_errors(self):
        path = self.failure({"valid": False})
        path.write_text('{"nested":' + "[" * 500 + "0" + "]" * 500 + "}")
        self.assertIsInstance(json.loads(path.read_text()), dict)
        progress = []
        self.run_request(repair_attempts=0, progress=progress.append)
        self.assertEqual(self.client.calls, [(self.system, self.payload)])
        self.assertFalse(any("Resuming validation repair" in message for message in progress))

        (self.cache / f"{self.key()}.json").unlink()
        path.unlink()
        client = Client()
        def recursive_validator(value):
            raise RecursionError("Current response validation failed")
        with self.assertRaisesRegex(RecursionError, "Current response"):
            self.run_request(client, validate=recursive_validator, repair_attempts=0)
        self.assertEqual(client.calls, [(self.system, self.payload)])
        self.assertFalse((self.cache / f"{self.key()}.json").exists())

    def test_overflowing_json_numbers_are_ignored_but_finite_numbers_can_seed_repairs(self):
        path = self.failure({"valid": False})
        success = self.cache / f"{self.key()}.json"
        for raw in ('{"nested":1e999}', '{"nested":[-1e999]}', '{"nested":{"value":1.8e308}}'):
            with self.subTest(raw=raw):
                path.write_text(raw)
                success.unlink(missing_ok=True)
                client = Client()
                self.run_request(client, repair_attempts=0)
                self.assertEqual(client.calls, [(self.system, self.payload)])
        success.unlink()
        path.write_text('{"valid":false,"nested":1e100}')
        client = Client()
        self.run_request(client, repair_attempts=0)
        self.assertEqual(client.calls[0][1]["correction"]["previous_response"]["nested"], 1e100)
        json.dumps(client.calls[0][1], allow_nan=False)

    def test_valid_failed_material_is_not_promoted_and_success_cache_has_priority(self):
        self.failure({"valid": True, "origin": "previous failure"})
        success = self.cache / f"{self.key()}.json"
        class ObservingClient(Client):
            def complete(client, system, payload):
                self.assertFalse(success.exists())
                return super().complete(system, payload)
        client = ObservingClient({"valid": True, "origin": "fresh model"})
        self.assertEqual(self.run_request(client), client.response)
        self.assertEqual(client.calls, [(self.system, self.payload)])
        self.failure({"valid": False, "origin": "do not inspect when success exists"}, attempt=2)
        repeat = Client()
        self.assertEqual(self.run_request(repeat), client.response)
        self.assertEqual(repeat.calls, [])

    def test_resume_preserves_call_limit_and_revalidates_every_new_response(self):
        self.failure({"valid": False, "revision": "old"})
        client = Client({"valid": False, "revision": "still invalid"})
        with self.assertRaisesRegex(ValueError, "failed validation after 3 attempt"):
            self.run_request(client, repair_attempts=2)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all("correction" in request for _, request in client.calls))
        self.assertEqual(client.calls[0][1]["correction"]["previous_response"]["revision"], "old")
        self.assertTrue(all(request["correction"]["previous_response"]["revision"] == "still invalid"
                            for _, request in client.calls[1:]))
        self.assertFalse((self.cache / f"{self.key()}.json").exists())

    def test_request_budget_omits_previous_response_or_ignores_hint_when_feedback_wont_fit(self):
        raw = {"valid": False, "large_previous_response": "x" * 5000}
        self.failure(raw)
        client, progress = Client(), []
        self.run_request(client, repair_attempts=0, max_request_chars=500, progress=progress.append)
        request = client.calls[0][1]
        self.assertIn("correction", request)
        self.assertNotIn("previous_response", request["correction"])
        self.assertLessEqual(len(self.system) + len(json.dumps(request, ensure_ascii=False)), 500)
        self.assertTrue(any("omitted to fit" in entry for entry in progress))
        (self.cache / f"{self.key()}.json").unlink()
        client = Client()
        self.run_request(client, repair_attempts=0, max_request_chars=100)
        self.assertEqual(client.calls[0][1], self.payload)
        (self.cache / f"{self.key()}.json").unlink()
        client = Client()
        with self.assertRaisesRegex(ValueError, "max_request_chars"):
            self.run_request(client, max_request_chars=1)
        self.assertEqual(client.calls, [])

    def test_seed_feedback_is_bounded_and_cancellation_stops_before_any_call(self):
        self.failure({"valid": False})
        def validate(value):
            if not value.get("valid"):
                raise ValueError("x" * 10000)
            return value
        self.run_request(validate=validate, repair_attempts=0)
        self.assertEqual(len(self.client.calls[0][1]["correction"]["validator_error"]), 2000)
        from concurrent.futures import CancelledError
        from threading import Event
        cancelled = Event()
        cancelled.set()
        client = Client()
        with self.assertRaises(CancelledError):
            CachedRequests(client, self.cache).run(self.system, self.payload, validate, cancelled=cancelled)
        self.assertEqual(client.calls, [])
