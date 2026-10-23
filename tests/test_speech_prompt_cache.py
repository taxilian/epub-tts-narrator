from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests, fingerprint, write_json
from epub_tts_narrator.prompts import PROMPT_VERSION
from epub_tts_narrator.speech_normalization import normalize_document_speech
from test_speech_normalization import document, Requests


class Client:
    identity = {"model": "offline-compact-cache"}

    def __init__(self, *, forbid_calls=False):
        self.calls = []
        self.forbid_calls = forbid_calls
        self.behavior = Requests({"wull": "well"})

    def complete(self, system, payload):
        if self.forbid_calls:
            raise AssertionError("Validated legacy requests must not call the model")
        self.calls.append((system, deepcopy(payload)))
        return self.behavior.run(system, payload, lambda result: result)


class CaptureLegacy(Requests):
    def __init__(self):
        super().__init__({"wull": "well"})
        self.legacy = []

    def cached(self, system, payload, validate, **kwargs):
        self.legacy.append((system, deepcopy(payload)))
        return False, None


class SpeechPromptCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.source = document("wull one", "well two")
        self.capture = CaptureLegacy()
        self.expected = normalize_document_speech(self.source, self.capture, workers=1)

    def key(self, system, payload):
        return fingerprint({"provider": Client.identity, "prompt_version": PROMPT_VERSION,
                            "system": system, "payload": payload})

    def test_all_validated_legacy_scan_and_review_responses_reuse_without_model_calls(self):
        behavior = Requests({"wull": "well"})
        for system, payload in self.capture.legacy:
            write_json(self.cache / f"{self.key(system, payload)}.json",
                       behavior.run(system, payload, lambda result: result))
        client = Client(forbid_calls=True)
        requests = CachedRequests(client, self.cache)
        result = normalize_document_speech(self.source, requests, workers=1)
        self.assertEqual(result, self.expected)
        self.assertEqual(requests.cache_hits, 4)
        self.assertEqual(client.calls, [])
        self.assertEqual(normalize_document_speech(result, requests, workers=1), result)

    def test_invalid_legacy_scan_is_rejected_and_replaced_by_compact_request(self):
        system, payload = self.capture.legacy[0]
        write_json(self.cache / f"{self.key(system, payload)}.json", {"segments": []})
        client = Client()
        result = normalize_document_speech(self.source, CachedRequests(client, self.cache), workers=1)
        self.assertEqual(result, self.expected)
        self.assertEqual(client.calls[0][1]["version"], 2)
        self.assertIn("roles", client.calls[0][1])

    def test_failed_legacy_response_is_never_promoted_to_validated_cache(self):
        system, payload = self.capture.legacy[0]
        key = self.key(system, payload)
        write_json(self.cache / "failed" / f"{key}-attempt1.json",
                   Requests({"wull": "well"}).run(system, payload, lambda result: result))
        client = Client()
        normalize_document_speech(self.source, CachedRequests(client, self.cache), workers=1)
        self.assertEqual(len(client.calls), 4)
        self.assertNotIn("correction", client.calls[0][1])

    def test_legacy_parent_truncation_hint_selects_compact_children(self):
        system, payload = self.capture.legacy[0]
        key = self.key(system, payload)
        write_json(self.cache / "speech-spelling-subdivisions" / f"{key}.json",
                   {"version": 1, "reason": "model_output_truncated", "request_cache_fingerprint": key})
        client = Client()
        result = normalize_document_speech(self.source, CachedRequests(client, self.cache), workers=1)
        scans = [row for _, row in client.calls if row["mode"] == "speech_spelling_scan"]
        self.assertEqual([len(row["target_segments"]) for row in scans], [1, 1])
        self.assertTrue(all(row["version"] == 2 for row in scans))
        self.assertEqual(result["provenance"]["speech_normalization"]["applied_edits"], 1)


if __name__ == "__main__":
    unittest.main()
