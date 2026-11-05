"""The compact attribution wire reconstructs exact source and stays cache-compatible."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Barrier, Lock
import unittest

from epub_tts_narrator.attribution_compact import align_compact_units, compact_cast
from epub_tts_narrator.attribution_compact_prompts import ATTRIBUTE_COMPACT
from epub_tts_narrator.attribution_fallback import attribute_with_subdivision
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.performance_plan import validate_performance
from epub_tts_narrator.pipeline import DEFAULT_POLICY, CachedRequests, analyze, fingerprint, make_units, seed_cast, write_json
from epub_tts_narrator.prompts import ATTRIBUTE, PROMPT_VERSION
from epub_tts_narrator.provider import ProviderError
from test_contracts import valid_document
from test_pipeline import FakeClient


def compact_source():
    source = deepcopy(valid_document()["source"])
    source["blocks"][0]["text"] = "“Nooo,” Mara whispered with a laugh."
    return source


class CompactClient(FakeClient):
    """Answers attribution with anchored segments instead of echoed text."""

    def __init__(self, *, mutation=None, reviewer="keep"):
        super().__init__()
        self.mutation = mutation
        self.reviewer = reviewer

    def attribute(self, payload):
        cast = payload["final_cast"]
        dialogue = next(role["id"] for role in cast["roles"] if role["kind"] == "dialogue")
        units = []
        for unit in payload["target_units"]:
            quote_end = unit["text"].find("”")
            if quote_end == -1:
                pieces = [self.piece(unit, 0, len(unit["text"]), "narration", "narrator_main",
                                     "Narrative frame.", {})]
            else:
                quote = unit["text"][:quote_end + 1]
                tag = unit["text"][quote_end + 1:]
                quote_proposals = {
                    "edits": [{"original": "Nooo", "replacement": "No", "direction": "drawn-out refusal",
                               "reason": "Stretched refusal word."}],
                    "controls": [{"kind": "delivery", "tag": "whisper", "original": "Nooo",
                                  "reason": "She whispered."}],
                }
                tag_proposals = {
                    "edits": [{"original": "laugh", "replacement": "larf", "direction": "dialect vowel",
                               "reason": "Stylized dialect respelling."}],
                    "controls": [
                        {"kind": "pause", "duration_ms": 300, "position": "before", "original": "Mara",
                         "reason": "A beat after the refusal."},
                        {"kind": "vocalization", "sound": "laugh", "position": "after", "original": "laugh",
                         "reason": "The passage states a laugh."},
                    ],
                }
                pieces = [self.piece(unit, 0, len(quote), "dialogue", dialogue,
                                     "The quotation names Mara.", quote_proposals),
                          self.piece(unit, len(quote), len(unit["text"]), "narration", "narrator_main",
                                     "The speech tag narrates the delivery.", tag_proposals)]
            units.append({"unit_id": unit["unit_id"], "segments": pieces})
        response = {"units": units}
        if self.mutation:
            self.mutation(response)
        return response

    @staticmethod
    def piece(unit, start, end, kind, role_id, reason, proposals):
        result = {"kind": kind, "role_id": role_id, "basis": "explicit", "reason": reason,
                  "evidence": [{"block_id": unit["block_id"], "quote": "Mara"}] if role_id != "narrator_main" else [],
                  "candidates": [], "review_required": False,
                  "delivery": {"direction": "", "evidence": []}}
        if start > 0:
            result["start_anchor"] = unit["text"][start:start + 12]
        if proposals:
            result["proposals"] = proposals
        return result

    def review(self, payload):
        verdict = {"keep": "keep", "remove": "remove", "uncertain": "uncertain"}[self.reviewer]
        return {"decisions": [{"id": item["id"], "verdict": verdict} for item in payload["proposals"]]}

    def complete(self, system, payload):
        if payload.get("mode") == "inline_performance_review":
            self.calls.append((system, deepcopy(payload)))
            return self.review(payload)
        return super().complete(system, payload)


class LegacyEchoClient(CompactClient):
    """An older model that still echoes text on the compact wire."""

    def attribute(self, payload):
        units = []
        for unit in payload["target_units"]:
            piece = {"text": unit["text"], "kind": "narration", "role_id": "narrator_main",
                     "basis": "explicit", "reason": "Legacy echo.", "evidence": [], "candidates": [],
                     "review_required": False, "delivery": {"direction": "", "evidence": []}}
            units.append({"unit_id": unit["unit_id"], "segments": [piece]})
        return {"units": units}


class BadAnchorClient(CompactClient):
    """First response anchors a segment outside the unit, then recovers."""

    def __init__(self):
        super().__init__()
        self.attempts = 0

    def attribute(self, payload):
        response = super().attribute(payload)
        if self.attempts == 0:
            self.attempts += 1
            unit = response["units"][0]
            if len(unit["segments"]) > 1:
                unit["segments"][1]["start_anchor"] = "absent from the unit entirely"
        return response


class CompactAttributionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"
        self.source = compact_source()

    def analyze_compact(self, client, **kwargs):
        # These tests isolate the compact wire mechanics; consensus has its own suites.
        kwargs.setdefault("identity_consensus", False)
        return analyze(deepcopy(self.source), client, self.cache, **kwargs)

    def test_compact_wire_publishes_approved_edits_controls_and_exact_source(self):
        client = CompactClient()
        document = self.analyze_compact(client)
        self.assertEqual(validate_document(document), [])
        original = compact_source()
        self.assertEqual(document["source"]["blocks"], original["blocks"])
        self.assertEqual([row["text"] for row in document["segments"]],
                         ["“Nooo,”", " Mara whispered with a laugh.", "They left."])
        quote_row = document["segments"][0]
        self.assertEqual(quote_row["speech_normalization"], {"version": 1, "edits": [
            {"start": 1, "end": 5, "original": "Nooo", "replacement": "No",
             "direction": "drawn-out refusal", "reason": "Stretched refusal word."},
        ]})
        spoken = quote_row["spoken_text"]
        self.assertEqual(spoken, "“No,”")
        controls = quote_row["performance_controls"]
        self.assertEqual(validate_performance(controls, spoken), controls)
        whisper = controls["controls"][0]
        self.assertEqual((whisper["kind"], whisper["tag"], spoken[whisper["start"]:whisper["end"]]),
                         ("delivery", "whisper", "No"))
        tag_row = document["segments"][1]
        tag_spoken = tag_row["spoken_text"]
        self.assertEqual(tag_spoken, " Mara whispered with a larf.")
        tag_controls = tag_row["performance_controls"]
        self.assertEqual(validate_performance(tag_controls, tag_spoken), tag_controls)
        self.assertEqual(sorted(control["kind"] for control in tag_controls["controls"]),
                         ["pause", "vocalization"])
        self.assertEqual([row.get("speech_normalization") for row in document["segments"][1:2]][0]["edits"],
                         [{"start": 23, "end": 28, "original": "laugh", "replacement": "larf",
                           "direction": "dialect vowel", "reason": "Stylized dialect respelling."}])
        for row in document["segments"]:
            self.assertNotIn("inline_proposals", row)
            self.assertNotIn("start_anchor", json.dumps(row))
        audit = document["provenance"]["inline_performance"]
        self.assertEqual(audit["counts"], {"kept": 4, "mechanical": 1})
        self.assertEqual([item["id"] for item in audit["outcomes"]][:1], ["r001"])
        attribute_calls = [payload for _, payload in client.calls
                           if "final_cast" in payload and "target_units" in payload]
        self.assertTrue(attribute_calls)
        wire_voices = attribute_calls[0]["final_cast"]["voices"]
        self.assertTrue(all("speechTraits" not in voice and "traits" not in voice for voice in wire_voices))
        wire_characters = attribute_calls[0]["final_cast"]["characters"]
        self.assertTrue(all("evidence" not in character for character in wire_characters))
        self.assertTrue(all(character["name"] == "Mara" for character in wire_characters))
        review_calls = [payload for _, payload in client.calls if payload.get("mode") == "inline_performance_review"]
        self.assertEqual(len(review_calls), 1)
        self.assertEqual([item["id"] for item in review_calls[0]["proposals"]],
                         ["r002", "r003", "r004", "r005"])
        modes = {payload.get("mode", "attribute") for _, payload in client.calls}
        self.assertEqual(modes, {"casting_discovery", "casting_coverage", "voice_ensemble", "attribute",
                                 "inline_performance_review"})

    def test_inline_review_honors_five_worker_cap_and_replays_identical_output(self):
        source = deepcopy(self.source)
        template = source["blocks"][0]
        source["blocks"] = [dict(template, id=f"b{index:03d}") for index in range(25)]
        source["chapters"][0]["block_ids"] = [block["id"] for block in source["blocks"]]
        self.source = source

        class ConcurrentReviewClient(CompactClient):
            def __init__(self):
                super().__init__()
                self.barrier = Barrier(5)
                self.active_lock = Lock()
                self.active = self.max_active = 0

            def complete(self, system, payload):
                if payload.get("mode") != "inline_performance_review":
                    return super().complete(system, payload)
                with self.active_lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    self.barrier.wait(timeout=2)
                    return super().complete(system, payload)
                finally:
                    with self.active_lock:
                        self.active -= 1

        client = ConcurrentReviewClient()
        document = self.analyze_compact(client, whole_book_discovery=True, attribution_workers=8,
                                        repair_attempts=0)

        self.assertEqual(client.max_active, 5)
        self.assertEqual(client.active, 0)
        self.assertEqual(len(document["provenance"]["inline_performance"]["batches"]), 5)
        self.assertEqual(document["source"], source)
        replay = CompactClient()
        self.assertEqual(self.analyze_compact(replay, whole_book_discovery=True, attribution_workers=8,
                                              repair_attempts=0), document)
        self.assertEqual(replay.calls, [])

    def test_compact_cast_request_is_smaller_than_the_legacy_full_cast(self):
        client = CompactClient()
        document = self.analyze_compact(client)
        observed = next(payload for _, payload in client.calls
                        if "final_cast" in payload and "target_units" in payload)
        legacy = dict(observed, final_cast=document["cast"])
        self.assertLess(len(json.dumps(observed, ensure_ascii=False)), len(json.dumps(legacy, ensure_ascii=False)))

    def test_exact_legacy_cache_entry_replays_before_any_new_request(self):
        source = deepcopy(self.source)
        units = [unit for batch in make_units(source) for unit in batch]
        cast = seed_cast()
        payload = {"final_cast": cast, "performance_policy": deepcopy(DEFAULT_POLICY),
                   "target_units": units, "context_units": [], "prior_segments": [],
                   "scene_summary": {"summary": "Offline narrative fixture."}}
        legacy_response = {"units": [{"unit_id": unit["unit_id"], "segments": [{
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narrative frame.", "evidence": [],
            "candidates": [], "review_required": False, "delivery": {"direction": "", "evidence": []},
        }]} for unit in units]}
        requests = CachedRequests(CompactClient(), self.cache, repair_attempts=0)
        key = fingerprint({"provider": requests.client.identity, "prompt_version": PROMPT_VERSION,
                           "system": ATTRIBUTE, "payload": payload})
        write_json(self.cache / f"{key}.json", legacy_response, overwrite=True)
        compact_payload = dict(payload, final_cast=compact_cast(cast))
        segments, record = attribute_with_subdivision(
            source, cast, requests, ATTRIBUTE_COMPACT, compact_payload,
            validator=align_compact_units, legacy=(ATTRIBUTE, payload))
        self.assertIsNone(record)
        self.assertEqual([row["text"] for row in segments], [unit["text"] for unit in units])
        self.assertEqual(requests.cache_hits, 1)

    def test_legacy_text_echo_response_stays_valid_on_the_compact_wire(self):
        client = LegacyEchoClient()
        document = self.analyze_compact(client)
        self.assertEqual(validate_document(document), [])
        self.assertEqual([row["text"] for row in document["segments"]],
                         [block["text"] for block in self.source["blocks"]])
        self.assertNotIn("inline_performance", document["provenance"])

    def test_bad_anchor_is_rejected_then_repaired_through_the_normal_loop(self):
        client = BadAnchorClient()
        document = self.analyze_compact(client, repair_attempts=1)
        self.assertEqual(validate_document(document), [])
        self.assertEqual(client.attempts, 1)

    def test_unrepaired_bad_anchor_fails_without_silent_acceptance(self):
        client = BadAnchorClient()
        with self.assertRaises(ValueError) as context:
            self.analyze_compact(client, repair_attempts=0)
        self.assertIn("start_anchor", str(context.exception))

    def test_truncated_compact_request_subdivides_and_retains_private_metadata(self):
        source = deepcopy(self.source)
        template = source["blocks"][0]
        source["blocks"] = [dict(template, id=f"b{index:03d}", text=f"Narrative paragraph {index}.")
                            for index in range(8)]
        source["chapters"][0]["block_ids"] = [block["id"] for block in source["blocks"]]
        units = [unit for batch in make_units(source) for unit in batch]
        cast = seed_cast()
        payload = {"final_cast": compact_cast(cast), "performance_policy": deepcopy(DEFAULT_POLICY),
                   "target_units": units, "context_units": [], "prior_segments": [],
                   "scene_summary": {"summary": "Offline narrative fixture."}}

        class TruncatingClient:
            identity = {"base_url": "https://offline.invalid/v1", "model": "fixture"}

            def __init__(self):
                self.calls = []

            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                if len(payload["target_units"]) > 2:
                    raise ProviderError("The model output was truncated. "
                                        "Increase the output token limit or reduce the input.")
                proposals = {"controls": [{"kind": "pause", "duration_ms": 200, "position": "before",
                                           "original": "paragraph", "reason": "Deliberate opening beat."}]}
                return {"units": [{"unit_id": unit["unit_id"], "segments": [{
                    "kind": "narration", "role_id": "narrator_main", "basis": "explicit",
                    "reason": "Narrative frame.", "evidence": [], "candidates": [],
                    "review_required": False, "delivery": {"direction": "", "evidence": []},
                    "proposals": proposals,
                }]} for unit in payload["target_units"]]}

        client = TruncatingClient()
        requests = CachedRequests(client, self.cache, repair_attempts=0)
        segments, record = attribute_with_subdivision(
            source, cast, requests, ATTRIBUTE_COMPACT, payload, validator=align_compact_units,
            sequential_prior=True)
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "subdivided")
        self.assertEqual(len(record["children"]), 2)
        self.assertEqual([row["text"] for row in segments], [unit["text"] for unit in units])
        self.assertTrue(all("inline_proposals" in row for row in segments))
        for row, unit in zip(segments, units):
            self.assertEqual(row["inline_proposals"]["controls"][0]["original"], "paragraph")
        attribute_calls = [payload for payload in client.calls if "final_cast" in payload]
        self.assertTrue(all("inline_proposals" not in json.dumps(row)
                            for payload in attribute_calls for row in payload["prior_segments"]))


if __name__ == "__main__":
    unittest.main()
