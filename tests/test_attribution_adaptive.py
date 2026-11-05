from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.attribution_compact import compact_cast
from epub_tts_narrator.attribution_fallback import attribute_with_subdivision
from epub_tts_narrator.pipeline import CachedRequests, make_units
from epub_tts_narrator.provider import OutputTruncatedError, ProviderError
from test_contracts import valid_document


class CapacityClient:
    identity = {"model": "offline-adaptive", "base_url": "http://offline.invalid"}

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if self.error is not None:
            raise self.error
        if len(payload["target_units"]) > 1:
            raise OutputTruncatedError("The model output was truncated. Increase the output token limit or reduce the input.")
        return {"units": [{"unit_id": unit["unit_id"], "segments": [
            {"text": unit["text"], "kind": "narration", "role_id": "narrator_main", "basis": "explicit",
             "reason": "Source narration.", "evidence": [], "candidates": [], "review_required": False,
             "delivery": {"direction": "", "evidence": []}}]}
            for unit in payload["target_units"]]}


class AttributionAdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.document = valid_document()
        self.document["cast"]["roles"][0]["id"] = "narrator_main"
        source = self.document["source"]
        first = source["blocks"][0]
        source["blocks"] = [dict(deepcopy(first), id=f"b{index + 1}") for index in range(17)]
        source["chapters"][0]["block_ids"] = [block["id"] for block in source["blocks"]]
        self.units = [unit for batch in make_units(source) for unit in batch]
        self.payload = {"final_cast": compact_cast(self.document["cast"]), "performance_policy": {},
                        "target_units": self.units, "context_units": [], "prior_segments": [],
                        "scene_summary": {"summary": "Source narration."}}

    def test_compact_wire_retains_narrator_source_scope_without_mutating_the_cast(self):
        scope = {"unit_ids": ["u0000001"], "block_ids": ["b1"]}
        self.document["cast"]["roles"][0]["source_scope"] = deepcopy(scope)

        wire = compact_cast(self.document["cast"])

        self.assertEqual(wire["roles"][0]["source_scope"], scope)
        wire["roles"][0]["source_scope"]["unit_ids"].append("u0000002")
        self.assertEqual(self.document["cast"]["roles"][0]["source_scope"], scope)

    def test_more_than_eight_leaves_preserve_source_and_replay_without_model_calls(self):
        client = CapacityClient()
        requests = CachedRequests(client, Path(self.directory.name), max_request_chars=100000)
        source_before = deepcopy(self.document["source"])

        segments, provenance = attribute_with_subdivision(
            self.document["source"], self.document["cast"], requests, "Attribute source.",
            self.payload, adaptive=True)

        self.assertEqual(len(segments), 17)
        self.assertEqual([row["text"] for row in segments], [unit["text"] for unit in self.units])
        self.assertEqual(len(client.calls), 33)
        self.assertEqual(len(provenance["regions"]), 17)
        self.assertEqual(len(provenance["subdivision_observations"]), 16)
        self.assertFalse(provenance["truncation_observation"]["replayed_hint"])
        self.assertEqual(self.document["source"], source_before)
        client.calls.clear()
        replay, replay_provenance = attribute_with_subdivision(
            self.document["source"], self.document["cast"], requests, "Attribute source.",
            self.payload, adaptive=True)
        self.assertEqual(replay, segments)
        self.assertEqual(client.calls, [])
        self.assertTrue(replay_provenance["truncation_observation"]["replayed_hint"])

    def test_generic_provider_failure_does_not_fan_out(self):
        client = CapacityClient(ProviderError("Unavailable", status_code=503))
        requests = CachedRequests(client, Path(self.directory.name))

        with self.assertRaises(ProviderError):
            attribute_with_subdivision(self.document["source"], self.document["cast"], requests,
                                       "Attribute source.", self.payload, adaptive=True)

        self.assertEqual(len(client.calls), 1)

    def test_model_cannot_assign_a_canonical_role_omitted_from_its_request(self):
        class HiddenRoleClient(CapacityClient):
            def complete(self, system, payload):
                response = super().complete(system, payload)
                row = response["units"][0]["segments"][0]
                row.update(role_id="mara", kind="dialogue", evidence=[{"block_id": "b1", "quote": "Mara"}])
                return response

        client = HiddenRoleClient()
        requests = CachedRequests(client, Path(self.directory.name), repair_attempts=0)
        payload = dict(deepcopy(self.payload), target_units=self.units[:1])
        payload["final_cast"] = {"characters": [], "roles": [deepcopy(self.document["cast"]["roles"][0])],
                                 "voices": [{"id": "vn"}]}

        with self.assertRaisesRegex(ValueError, "role"):
            attribute_with_subdivision(self.document["source"], self.document["cast"], requests,
                                       "Attribute source.", payload, adaptive=True)

        self.assertEqual(len(client.calls), 1)
        self.assertFalse(list(Path(self.directory.name).glob("*.json")))
