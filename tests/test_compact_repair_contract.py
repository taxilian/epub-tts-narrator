"""Compact attribution generation, repair, and frozen-prompt cache contracts."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.attribution_adaptive import SCOPE_GUIDANCE
from epub_tts_narrator.attribution_compact import align_compact_units, compact_cast
from epub_tts_narrator.attribution_compact_prompts import ATTRIBUTE_COMPACT
from epub_tts_narrator.attribution_fallback import attribute_with_subdivision
from epub_tts_narrator.pipeline import (
    CachedRequests, DEFAULT_POLICY, analyze, fingerprint, make_units, seed_cast, write_json,
)
from epub_tts_narrator.prompts import ATTRIBUTE, PROMPT_VERSION
from test_compact_attribution import CompactClient, compact_source


COMPACT_CONTRACT = {
    "protocol": "compact-attribution-v1",
    "segment_text": "forbidden",
    "delivery_evidence": "nonempty_direction_requires_nonempty_object_array",
}


class MalformedEvidenceClient(CompactClient):
    def __init__(self, *, delivery=False):
        super().__init__()
        self.delivery = delivery

    def attribute(self, payload):
        response = super().attribute(payload)
        if "correction" not in payload:
            piece = response["units"][0]["segments"][0]
            if self.delivery:
                piece["delivery"] = {"direction": "urgently", "evidence": ["not an evidence object"]}
            else:
                piece["evidence"] = ["not an evidence object"]
        return response


class MissingUnitClient(CompactClient):
    def attribute(self, payload):
        response = super().attribute(payload)
        if "correction" not in payload:
            response["units"].pop()
        return response


class CompactRepairContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name) / "cache"
        self.source = compact_source()
        self.cast = seed_cast()
        self.units = [unit for batch in make_units(self.source) for unit in batch]

    def payload(self):
        return {"final_cast": compact_cast(self.cast), "performance_policy": deepcopy(DEFAULT_POLICY),
                "target_units": self.units, "context_units": [], "prior_segments": [],
                "scene_summary": {"summary": "Offline narrative fixture."}}

    def response(self, units):
        return {"units": [{"unit_id": unit["unit_id"], "segments": [
            CompactClient.piece(unit, 0, len(unit["text"]), "narration", "narrator_main",
                                "Narrative frame.", {}),
        ]} for unit in units]}

    def analyze_compact(self, client, cache=None):
        return analyze(deepcopy(self.source), client, cache or self.cache,
                       identity_consensus=False, repair_attempts=1)

    def attribute_calls(self, client):
        return [payload for _, payload in client.calls
                if "final_cast" in payload and "target_units" in payload]

    def test_compact_evidence_repair_carries_structured_anchor_and_object_contract(self):
        for delivery in (False, True):
            with self.subTest(delivery=delivery):
                client = MalformedEvidenceClient(delivery=delivery)

                document = self.analyze_compact(client, self.cache / ("delivery" if delivery else "segment"))

                calls = self.attribute_calls(client)
                self.assertEqual(len(calls), 2)
                systems = [system for system, payload in client.calls
                           if "final_cast" in payload and "target_units" in payload]
                self.assertEqual(systems, [ATTRIBUTE_COMPACT, ATTRIBUTE_COMPACT])
                correction = calls[1]["correction"]
                contract = correction["compact_contract"]
                self.assertEqual({key: contract[key] for key in COMPACT_CONTRACT}, COMPACT_CONTRACT)
                self.assertEqual(set(contract["evidence_entry"]), {"block_id", "quote"})
                self.assertEqual(correction["target_unit_ids"], [unit["unit_id"] for unit in self.units])
                piece = correction["previous_response"]["units"][0]["segments"][0]
                container = piece["delivery"] if delivery else piece
                self.assertEqual(container["evidence"], ["not an evidence object"])
                self.assertEqual(document["source"], self.source)

    def test_compact_missing_unit_repair_retains_complete_ordered_target_scope(self):
        client = MissingUnitClient()

        document = self.analyze_compact(client)

        calls = self.attribute_calls(client)
        self.assertEqual(len(calls), 2)
        correction = calls[1]["correction"]
        contract = correction["compact_contract"]
        self.assertEqual({key: contract[key] for key in COMPACT_CONTRACT}, COMPACT_CONTRACT)
        self.assertEqual(set(contract["evidence_entry"]), {"block_id", "quote"})
        self.assertEqual(correction["target_unit_count"], len(self.units))
        self.assertEqual(correction["target_unit_ids"], [unit["unit_id"] for unit in self.units])
        self.assertEqual([row["text"] for row in document["segments"]],
                         ["“Nooo,”", " Mara whispered with a laugh.", "They left."])

    def test_legacy_exact_text_repair_does_not_acquire_the_compact_contract(self):
        requests = CachedRequests(CompactClient(), self.cache)
        payload = dict(self.payload(), final_cast=self.cast)

        repaired = requests._repair_payload(ATTRIBUTE, payload, {"units": []}, ValueError("Invalid response"))

        self.assertNotIn("compact_contract", repaired["correction"])
        self.assertEqual(repaired["correction"]["target_unit_ids"],
                         [unit["unit_id"] for unit in self.units])

    def test_validated_v1_compact_caches_replay_at_adaptive_leaves_without_model_calls(self):
        from epub_tts_narrator import attribution_compact_prompts as prompts

        old_prompt = prompts.ATTRIBUTE_COMPACT_V1
        self.assertEqual(fingerprint(old_prompt), "5797d78829931dbd5d6a4d5d95cec7b035ec579591caeacd85f91f1c2645467f")
        for scoped in (False, True):
            with self.subTest(scoped=scoped):
                cache = self.cache / ("scoped" if scoped else "ordinary")
                client = CompactClient()
                requests = CachedRequests(client, cache, repair_attempts=0)
                payload = self.payload()
                suffix = ""
                if scoped:
                    suffix = SCOPE_GUIDANCE
                    payload["cast_scope"] = {"unit_ids": [unit["unit_id"] for unit in self.units],
                                             "block_ids": [unit["block_id"] for unit in self.units]}
                response = self.response(self.units)
                key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                                   "system": old_prompt + suffix, "payload": payload})
                write_json(cache / f"{key}.json", response, overwrite=True)

                segments, _ = attribute_with_subdivision(
                    self.source, self.cast, requests, ATTRIBUTE_COMPACT + suffix, payload,
                    validator=align_compact_units, legacy_compact_system=old_prompt + suffix,
                    repair_protocol=COMPACT_CONTRACT["protocol"], adaptive=True)

                self.assertEqual([row["text"] for row in segments], [unit["text"] for unit in self.units])
                self.assertEqual(requests.cache_hits, 1)
                self.assertEqual(client.calls, [])

    def test_malformed_compact_evidence_entries_remain_strictly_rejected(self):
        unit = self.units[0]
        for delivery in (False, True):
            for malformed in ("bare quotation", ["block ID", "quotation"]):
                with self.subTest(delivery=delivery, shape=type(malformed).__name__):
                    response = self.response([unit])
                    piece = response["units"][0]["segments"][0]
                    container = piece["delivery"] if delivery else piece
                    container["evidence"] = [malformed]
                    before = deepcopy(response)
                    with self.assertRaises(ValueError) as raised:
                        align_compact_units(self.source, self.cast, [unit], response)
                    self.assertIn("must be an object", str(raised.exception))
                    self.assertEqual(response, before)

    def test_compact_response_with_a_missing_target_unit_is_still_rejected(self):
        response = self.response(self.units[:1])

        with self.assertRaises(ValueError):
            align_compact_units(self.source, self.cast, self.units, response)


if __name__ == "__main__":
    unittest.main()
