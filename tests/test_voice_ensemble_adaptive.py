"""Adaptive ensemble partitions preserve the canonical cast and voice evidence."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests, DEFAULT_POLICY, fingerprint
from epub_tts_narrator.provider import OutputTruncatedError, ProviderError
from epub_tts_narrator.voice_ensemble import _contrast_context, design_ensemble
from epub_tts_narrator.voice_ensemble_prompts import COMPACT_CONTRAST_GUIDANCE
from test_contracts import valid_document


TRUNCATED = "The model output was truncated. Increase the output token limit or reduce the input."


def ensemble_fixture(local_count=6, *, inherited=False):
    document = valid_document()
    source = deepcopy(document["source"])
    cast = deepcopy(document["cast"])
    cast["roles"][0]["id"] = "narrator_main"
    cast["voices"][1]["user_note"] = "deliberate local metadata"
    for index in range(2, local_count):
        voice_id = f"v{index}"
        cast["characters"].append({"id": f"c{index}", "name": f"Speaker {index}", "aliases": [],
                                   "description": "A source-linked speaker.",
                                   "evidence": [{"block_id": "b1", "quote": "Mara"}], "voice_id": voice_id})
        cast["roles"].append({"id": f"r{index}", "kind": "dialogue", "character_id": f"c{index}",
                              "voice_id": voice_id, "description": "Dialogue role."})
        cast["voices"].append({"id": voice_id, "description": "A measured local voice.",
                               "traits": [{"name": "pace", "value": "measured", "basis": "casting", "evidence": []}]})
    if inherited:
        cast["voices"].append({"id": "old_voice", "description": "DO NOT LEAK INHERITED PROFILE",
                               "series_origin": {"kind": "series_casting_reuse", "series_id": "s", "voice_id": "old"},
                               "traits": [{"name": "volume", "value": "soft", "basis": "text",
                                           "evidence": [{"block_id": "b1", "quote": "softly"}]}]})
    return source, cast


def compact_request_size(system, payload):
    compact = deepcopy(payload)
    compact["contrast_findings"] = _contrast_context(payload["contrast_findings"])
    if isinstance(compact["contrast_findings"], dict):
        system += COMPACT_CONTRAST_GUIDANCE
    return len(system) + len(json.dumps(compact, ensure_ascii=False))


class FakeClient:
    identity = {"base_url": "https://offline.invalid/v1", "model": "adaptive-ensemble-fixture"}

    def __init__(self, *, max_targets=None, mutate=None, failure=None):
        self.max_targets = max_targets
        self.mutate = mutate
        self.failure = failure
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if self.failure is not None:
            raise self.failure
        targets = payload.get("scope", {}).get("target_voice_ids")
        if targets is None:
            targets = [voice["id"] for voice in payload["cast"]["voices"]
                       if voice.get("series_origin", {}).get("kind") != "series_casting_reuse"]
        if self.max_targets is not None and len(targets) > self.max_targets:
            raise OutputTruncatedError(TRUNCATED)
        profiles = {voice["id"]: voice for voice in payload["cast"]["voices"]}
        upserts = [{"id": voice_id, "description": f"Coordinated voice {voice_id} with clear resonance.",
                    "speechTraits": "clear measured phrases", "traits": deepcopy(profiles[voice_id]["traits"])}
                   for voice_id in targets]
        response = {"voices": upserts, "reasons": {voice_id: "Coordinate this target with the canonical ensemble."
                                                    for voice_id in targets}}
        return self.mutate(response, payload) if self.mutate else response


class VoiceEnsembleAdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def requests(self, client, name, *, maximum=200000):
        return CachedRequests(client, Path(self.directory.name) / name, repair_attempts=0, max_request_chars=maximum)

    @staticmethod
    def target_ids(provenance):
        return [voice_id for group in provenance["adaptive"]["groups"] for voice_id in group["target_voice_ids"]]

    def test_root_reuses_legacy_request_bytes_and_result_when_it_fits(self):
        source, cast = ensemble_fixture()
        original = deepcopy(cast)
        legacy_client, adaptive_client = FakeClient(), FakeClient()
        legacy, legacy_provenance = design_ensemble(source, cast, self.requests(legacy_client, "legacy"), policy=DEFAULT_POLICY)
        updated, provenance = design_ensemble(source, cast, self.requests(adaptive_client, "adaptive"),
                                              policy=DEFAULT_POLICY, adaptive=True)
        self.assertEqual(adaptive_client.calls, legacy_client.calls)
        self.assertEqual(json.dumps(adaptive_client.calls[0][1], ensure_ascii=False),
                         json.dumps(legacy_client.calls[0][1], ensure_ascii=False))
        self.assertEqual(len(adaptive_client.calls), 1)
        self.assertEqual(updated, legacy)
        self.assertEqual({key: value for key, value in provenance.items() if key != "adaptive"}, legacy_provenance)
        self.assertEqual(provenance["adaptive"]["groups"][0]["scope"], None)
        self.assertEqual(cast, original)

    def test_root_that_fits_after_contrast_compaction_also_matches_legacy(self):
        source, cast = ensemble_fixture(40)
        probe = FakeClient()
        design_ensemble(source, cast, self.requests(probe, "compact-probe"), policy=DEFAULT_POLICY)
        raw_system, raw_payload = probe.calls[0]
        raw_size = len(raw_system) + len(json.dumps(raw_payload, ensure_ascii=False))
        budget = compact_request_size(raw_system, raw_payload)
        self.assertLess(budget, raw_size)
        legacy_client, adaptive_client = FakeClient(), FakeClient()
        legacy = design_ensemble(source, cast, self.requests(legacy_client, "compact-legacy", maximum=budget),
                                 policy=DEFAULT_POLICY)
        adaptive = design_ensemble(source, cast, self.requests(adaptive_client, "compact-adaptive", maximum=budget),
                                   policy=DEFAULT_POLICY, adaptive=True)
        self.assertEqual(len(adaptive_client.calls), 1)
        self.assertEqual(adaptive_client.calls, legacy_client.calls)
        self.assertEqual(json.dumps(adaptive_client.calls[0][1], ensure_ascii=False),
                         json.dumps(legacy_client.calls[0][1], ensure_ascii=False))
        self.assertIsInstance(adaptive_client.calls[0][1]["contrast_findings"], dict)
        self.assertEqual(adaptive[0], legacy[0])
        self.assertEqual({key: value for key, value in adaptive[1].items() if key != "adaptive"}, legacy[1])

    def test_many_singleton_leaves_cover_each_target_once_and_preserve_the_full_cast(self):
        source, cast = ensemble_fixture(13, inherited=True)
        original = deepcopy(cast)
        client = FakeClient(max_targets=1)
        updated, provenance = design_ensemble(source, cast, self.requests(client, "many"), policy=DEFAULT_POLICY, adaptive=True)
        designable = [voice["id"] for voice in cast["voices"] if "series_origin" not in voice]
        self.assertEqual(len(client.calls), 2 * len(designable) - 1)
        self.assertEqual(self.target_ids(provenance), designable)
        self.assertEqual(len(provenance["adaptive"]["groups"]), len(designable))
        self.assertEqual([voice["id"] for voice in updated["voices"]], [voice["id"] for voice in cast["voices"]])
        self.assertEqual(updated["characters"], cast["characters"])
        self.assertEqual(updated["roles"], cast["roles"])
        self.assertEqual(next(voice for voice in updated["voices"] if voice["id"] == "v1")["user_note"],
                         "deliberate local metadata")
        self.assertEqual(next(voice for voice in updated["voices"] if voice["id"] == "old_voice"),
                         next(voice for voice in cast["voices"] if voice["id"] == "old_voice"))
        self.assertEqual(next(voice for voice in updated["voices"] if voice["id"] == "v1")["traits"],
                         next(voice for voice in cast["voices"] if voice["id"] == "v1")["traits"])
        for voice in updated["voices"]:
            if voice["id"] in designable:
                self.assertEqual(voice["description"], f"Coordinated voice {voice['id']} with clear resonance.")
                self.assertIn(voice["id"], provenance["updated"])
        for group in provenance["adaptive"]["groups"]:
            scope = group["scope"]
            self.assertEqual(scope["full_cast_fingerprint"], fingerprint(cast))
            self.assertEqual(scope["omitted_counts"]["voices"], len(cast["voices"]) - 1)
            self.assertEqual(group["reasons"], {voice_id: "Coordinate this target with the canonical ensemble."
                                                for voice_id in group["target_voice_ids"]})
        self.assertEqual(cast, original)

    def test_empty_group_upserts_keep_the_cast_unchanged_without_losing_coverage(self):
        source, cast = ensemble_fixture(7, inherited=True)
        client = FakeClient(max_targets=1, mutate=lambda response, payload: {"voices": [], "reasons": {}})
        updated, provenance = design_ensemble(source, cast, self.requests(client, "empty"),
                                              policy=DEFAULT_POLICY, adaptive=True)
        self.assertEqual(updated, cast)
        self.assertEqual(provenance["status"], "unchanged")
        self.assertEqual(self.target_ids(provenance), [voice["id"] for voice in cast["voices"]
                                                       if "series_origin" not in voice])
        self.assertTrue(all(group["reasons"] == {} for group in provenance["adaptive"]["groups"]))

    def test_input_capacity_partitions_without_trimming_selected_records(self):
        source, cast = ensemble_fixture(24)
        source["blocks"][0]["text"] += " Softly." * 100
        evidence = [{"block_id": "b1", "quote": source["blocks"][0]["text"]}]
        evidence += [{"block_id": "b1", "quote": " Softly." * count} for count in range(1, 14)]
        cast["characters"][0]["evidence"] = deepcopy(evidence)
        cast["voices"][1]["traits"][0]["evidence"] = deepcopy(evidence)
        probe = FakeClient(max_targets=1)
        design_ensemble(source, cast, self.requests(probe, "probe"), policy=DEFAULT_POLICY, adaptive=True)
        sizes = [len(system) + len(json.dumps(payload, ensure_ascii=False)) for system, payload in probe.calls]
        leaf_maximum = max(sizes[1:])
        root_system, root_payload = probe.calls[0]
        compact_root_size = compact_request_size(root_system, root_payload)
        self.assertGreater(compact_root_size, leaf_maximum)
        client = FakeClient()
        requests = self.requests(client, "input", maximum=leaf_maximum)
        updated, provenance = design_ensemble(source, cast, requests, policy=DEFAULT_POLICY, adaptive=True)
        self.assertTrue(client.calls)
        self.assertTrue(all(payload.get("scope") is not None for _, payload in client.calls))
        self.assertTrue(all(len(system) + len(json.dumps(payload, ensure_ascii=False)) <= leaf_maximum
                            for system, payload in client.calls))
        self.assertEqual(self.target_ids(provenance), [voice["id"] for voice in cast["voices"]])
        for _, payload in client.calls:
            for collection in ("characters", "roles", "voices"):
                for record in payload["cast"][collection]:
                    self.assertEqual(record, next(item for item in cast[collection] if item["id"] == record["id"]))
            self.assertTrue(payload["cross_group_contrast"]["read_only"])
        self.assertEqual(updated["characters"], cast["characters"])
        self.assertEqual(updated["roles"], cast["roles"])

    def test_group_rejects_other_local_and_inherited_upserts_without_fanout(self):
        for unauthorized in ("v1", "old_voice"):
            with self.subTest(unauthorized=unauthorized):
                source, cast = ensemble_fixture(3, inherited=True)
                original = deepcopy(cast)
                def inject(response, payload):
                    response["voices"].append({"id": unauthorized, "description": "Unauthorized partial-group edit.",
                                               "speechTraits": "steady", "traits": []})
                    response["reasons"][unauthorized] = "Attempt an out-of-scope edit."
                    return response
                client = FakeClient(max_targets=1, mutate=inject)
                with self.assertRaisesRegex(ValueError, "may upsert only its target voice IDs"):
                    design_ensemble(source, cast, self.requests(client, unauthorized), policy=DEFAULT_POLICY, adaptive=True)
                self.assertEqual(len(client.calls), 2)
                self.assertEqual(cast, original)

    def test_group_still_rejects_changed_text_traits(self):
        source, cast = ensemble_fixture(2)
        def drop_text_traits(response, payload):
            if payload.get("scope", {}).get("target_voice_ids") == ["v1"]:
                response["voices"][0]["traits"] = []
            return response
        client = FakeClient(max_targets=1, mutate=drop_text_traits)
        with self.assertRaisesRegex(ValueError, "text-supported traits"):
            design_ensemble(source, cast, self.requests(client, "traits"), policy=DEFAULT_POLICY, adaptive=True)

    def test_noncapacity_failures_do_not_partition(self):
        for failure in (ProviderError("offline transport failure"), ValueError("invalid fixture response")):
            with self.subTest(failure=type(failure).__name__):
                source, cast = ensemble_fixture(5)
                client = FakeClient(failure=failure)
                with self.assertRaises(type(failure)):
                    design_ensemble(source, cast, self.requests(client, type(failure).__name__),
                                    policy=DEFAULT_POLICY, adaptive=True)
                self.assertEqual(len(client.calls), 1)

    def test_single_target_that_cannot_fit_fails_without_trimming_or_source_echo(self):
        source, cast = ensemble_fixture(2)
        original = deepcopy(cast)
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "complete profile and mandatory text-trait evidence") as caught:
            design_ensemble(source, cast, self.requests(client, "singleton", maximum=100),
                            policy=DEFAULT_POLICY, adaptive=True)
        self.assertEqual(client.calls, [])
        self.assertNotIn(source["blocks"][0]["text"], str(caught.exception))
        self.assertEqual(cast, original)

    def test_truncation_recovery_reuses_leaf_cache_without_leaking_omitted_profiles(self):
        source, cast = ensemble_fixture(5, inherited=True)
        original = deepcopy(cast)
        client = FakeClient(max_targets=1)
        progress = []
        requests = self.requests(client, "resume")
        first = design_ensemble(source, cast, requests, policy=DEFAULT_POLICY, progress=progress.append, adaptive=True)
        call_count = len(client.calls)
        scoped = [payload for _, payload in client.calls if payload.get("scope") is not None]
        self.assertEqual(len(first[1]["adaptive"]["groups"]), 5)
        for payload in scoped:
            wire = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("DO NOT LEAK INHERITED PROFILE", wire)
            self.assertNotIn("correction", payload)
            self.assertEqual(payload["cast_fingerprint"], fingerprint(cast))
        self.assertFalse(any("DO NOT LEAK INHERITED PROFILE" in message for message in progress))
        second = design_ensemble(source, cast, requests, policy=DEFAULT_POLICY, adaptive=True)
        self.assertEqual(second, first)
        self.assertEqual(len(client.calls), call_count)
        self.assertEqual(cast, original)


if __name__ == "__main__":
    unittest.main()
