"""The whole-cast ensemble pass designs the final resolved cast before attribution."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import CachedRequests, DEFAULT_POLICY, analyze, fingerprint
from epub_tts_narrator.voice_ensemble import (
    ENSEMBLE_SYSTEM, VERSION, _payload, _validate_response, design_ensemble,
)
from test_contracts import valid_document


def cast_patch(**values):
    return dict({"characters": [], "roles": [], "voices": [], "merges": [],
                 "issues": [], "summary": "Mara speaks softly, then the group leaves."}, **values)


class EnsembleClient:
    """A deterministic fixture model; the ensemble handler is configurable."""

    identity = {"base_url": "http://offline.invalid/v1", "model": "fixture-model"}

    def __init__(self, *, ensemble=None, text_traits=False):
        self.ensemble = ensemble
        self.text_traits = text_traits
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        mode = payload.get("mode")
        if mode == "discover":
            return self.discover(payload)
        if mode == "reconcile":
            return cast_patch()
        if mode == "identity_scan":
            return {"cases": []}
        if mode == "voice_ensemble":
            return self.ensemble(payload) if self.ensemble else {"voices": [], "reasons": {}}
        return self.attribute(payload)

    def discover(self, payload):
        if payload["known_cast"]["characters"]:
            return cast_patch()
        unit = payload["target_units"][0]
        evidence = [{"block_id": unit["block_id"], "quote": "Mara"}]
        basis = "text" if self.text_traits else "casting"
        trait_evidence = deepcopy(evidence) if basis == "text" else []
        character_id, voice_id = payload["available_ids"]["characters"][0], payload["available_ids"]["voices"][0]
        return cast_patch(
            characters=[{"id": character_id, "name": "Mara", "aliases": [],
                         "description": "A character who speaks softly.",
                         "evidence": deepcopy(evidence), "voice_id": voice_id}],
            roles=[{"id": payload["available_ids"]["roles"][0], "kind": "dialogue",
                    "character_id": character_id, "voice_id": voice_id,
                    "description": "Dialogue spoken by Mara."}],
            voices=[{"id": voice_id, "description": "Proposed distinct voice for Mara.",
                     "traits": [{"name": "pace", "value": "measured", "basis": basis,
                                 "evidence": trait_evidence}]}])

    @staticmethod
    def attribute(payload):
        return {"units": [{"unit_id": unit["unit_id"], "segments": [{
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narrative prose.", "evidence": [], "candidates": [],
            "review_required": False, "delivery": {"direction": "", "evidence": []},
        }]} for unit in payload["target_units"]]}


class RecordingRequests:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])
        self.max_request_chars = 200000

    def run(self, system, payload, validate, *, cancelled=None):
        self.calls.append((system, deepcopy(payload)))
        return validate(deepcopy(self.responses.pop(0)) if self.responses else {"voices": [], "reasons": {}})


class VoiceEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"
        self.source = deepcopy(valid_document()["source"])

    def run_analysis(self, client, **kwargs):
        kwargs.setdefault("identity_consensus", False)
        return analyze(self.source, client, self.cache, **kwargs)

    def modes(self, client):
        return [payload.get("mode", "attribute") for _, payload in client.calls]

    def test_ensemble_runs_on_the_resolved_cast_after_identity_and_before_attribution(self):
        client = EnsembleClient()
        document = self.run_analysis(client, identity_consensus=True)
        self.assertEqual(self.modes(client),
                         ["discover", "reconcile", "identity_scan", "voice_ensemble", "attribute"])
        system, payload = client.calls[3]
        self.assertEqual(payload["mode"], "voice_ensemble")
        self.assertEqual(payload["performance_policy"], DEFAULT_POLICY)
        self.assertEqual(payload["book"], self.source["book"])
        self.assertEqual({section["id"] for section in payload["cast"]["voices"]},
                         {voice["id"] for voice in document["cast"]["voices"]})
        self.assertEqual(payload["cast"], document["cast"])
        self.assertEqual(payload["contrast_findings"],
                         document["provenance"]["voice_ensemble"]["contrast_findings_before"])
        self.assertEqual(document["provenance"]["voice_ensemble"]["contrast_findings_after"],
                         document["provenance"]["voice_ensemble"]["contrast_findings_before"])
        attribution = next(item for _, item in client.calls if "mode" not in item)
        self.assertEqual(attribution["final_cast"], document["cast"])
        self.assertEqual(validate_document(document), [])

    def test_ensemble_upserts_coordinate_profiles_and_provenance_traces_changes(self):
        def redesign(payload):
            voice_id = payload["cast"]["voices"][1]["id"]
            return {"voices": [{"id": voice_id, "description": "A coordinated, clearly distinct voice.",
                                "speechTraits": "measured pace with rounded vowels",
                                "traits": [{"name": "resonance", "value": "bright and forward", "basis": "casting",
                                            "evidence": []},
                                           {"name": "articulation", "value": "precise but relaxed", "basis": "casting",
                                            "evidence": []},
                                           {"name": "pace", "value": "brisk", "basis": "casting", "evidence": []}]}],
                    "reasons": {voice_id: "Separate the two measured voices by resonance and pace as casting."}}
        client = EnsembleClient(ensemble=redesign)
        document = self.run_analysis(client)
        ensemble = document["provenance"]["voice_ensemble"]
        voice = next(item for item in document["cast"]["voices"] if item["id"] != "voice_narrator_main")
        self.assertEqual(ensemble["status"], "applied")
        self.assertEqual(voice["description"], "A coordinated, clearly distinct voice.")
        self.assertEqual(voice["speechTraits"], "measured pace with rounded vowels")
        self.assertEqual({trait["name"]: trait["value"] for trait in voice["traits"]},
                         {"resonance": "bright and forward", "articulation": "precise but relaxed",
                          "pace": "brisk"})
        narrator = next(item for item in document["cast"]["voices"] if item["id"] == "voice_narrator_main")
        self.assertEqual(narrator["description"],
                         "A clear speaking voice with even and natural resonance, precise but relaxed "
                         "articulation, and a steady conversational pace.")
        self.assertEqual(ensemble["updated"][voice["id"]]["reason"], "Separate the two measured voices by resonance and pace as casting.")
        self.assertNotEqual(ensemble["cast_fingerprint_before"], ensemble["cast_fingerprint_after"])
        self.assertEqual(ensemble["cast_fingerprint_after"], fingerprint(document["cast"]))
        self.assertEqual([finding["kind"] for finding in ensemble["contrast_findings_before"]],
                         ["insufficient_known_contrast"])
        self.assertEqual([finding["kind"] for finding in ensemble["contrast_findings_after"]], [])
        attribution = next(item for _, item in client.calls if "mode" not in item)
        self.assertEqual(attribution["final_cast"], document["cast"])

    def test_ensemble_rejects_upserts_that_change_text_supported_traits(self):
        def rewrite(payload):
            voice_id = payload["cast"]["voices"][1]["id"]
            return {"voices": [{"id": voice_id, "description": "A sharper contrasting voice.",
                                "speechTraits": "brisk careful pace",
                                "traits": [{"name": "pace", "value": "brisk", "basis": "text", "evidence": []}]}],
                    "reasons": {voice_id: "Contrast."}}
        client = EnsembleClient(ensemble=rewrite, text_traits=True)
        with self.assertRaisesRegex(ValueError, "text-supported traits"):
            self.run_analysis(client, repair_attempts=0)
        # The rejected ensemble response must not enter the success cache; only
        # discovery and reconciliation hold validated entries.
        self.assertEqual(len(list(self.cache.glob("*.json"))), 2)
        self.assertTrue(list(self.cache.glob("failed/*.json")))
        repaired = EnsembleClient(text_traits=True)
        document = self.run_analysis(repaired, repair_attempts=0)
        self.assertEqual(validate_document(document), [])
        self.assertEqual(self.modes(repaired).count("voice_ensemble"), 1)

    def test_ensemble_rejects_invalid_upserts_before_touching_the_cast(self):
        source = deepcopy(valid_document()["source"])
        cast = deepcopy(valid_document())["cast"]
        cast["roles"][0]["id"] = "narrator_main"
        inherited = deepcopy(valid_document())["cast"]["voices"][1]
        inherited.update(id="old_v1", series_origin={"kind": "series_casting_reuse",
                                                     "series_id": "s", "voice_id": "prior"})
        cast["voices"].append(inherited)
        original = deepcopy(cast)
        requests = RecordingRequests()
        complete = {"description": "A plain coordinated direction.", "speechTraits": "steady even pace",
                    "traits": [{"name": "pace", "value": "brisk", "basis": "casting", "evidence": []}]}
        cases = [
            {"voices": [{"id": "new_voice", **complete}], "reasons": {"new_voice": "Invented voice."}},
            {"voices": [{"id": "v1", **complete}, {"id": "v1", **complete}], "reasons": {"v1": "Twice."}},
            {"voices": [{"id": "old_v1", **complete}], "reasons": {"old_v1": "Redesigned inherited profile."}},
            {"voices": [{"id": "vn", **complete, "extra": True}], "reasons": {"vn": "Extra field."}},
            {"voices": [{"id": "vn", "description": "   ", "speechTraits": "steady", "traits": []}],
             "reasons": {"vn": "Blank."}},
            {"voices": [{"id": "vn", "description": "A voice.", "speechTraits": "two\nlines", "traits": []}],
             "reasons": {"vn": "Broken traits."}},
            {"voices": [{"id": "vn", "description": "A voice.", "traits": []}], "reasons": {"vn": "No speechTraits."}},
            {"voices": [{"id": "vn", "description": "A voice.", "speechTraits": "steady"}], "reasons": {"vn": "No traits."}},
            {"voices": [dict(id="vn", speechTraits="steady", traits=[], **{"description": '"A voice."'})],
             "reasons": {"vn": "Quoted description."}},
            {"voices": [{"id": "vn", "description": "A voice. [whisper]", "speechTraits": "steady", "traits": []}],
             "reasons": {"vn": "Markup description."}},
            {"voices": [{"id": "vn", **complete,
                         "traits": [{"name": "pace", "value": "brisk", "basis": "text", "evidence": []}]}],
             "reasons": {"vn": "Fabricated text trait."}},
            {"voices": [{"id": "vn", **complete,
                         "traits": [{"name": "pace", "value": "brisk", "basis": "casting",
                                     "evidence": [{"block_id": "b1", "quote": "Mara"}]}]}],
             "reasons": {"vn": "Casting trait with evidence."}},
            {"voices": [{"id": "vn", **complete,
                         "traits": [{"name": "pace", "value": "brisk", "basis": "casting", "evidence": [],
                                     "note": "extra"}]}],
             "reasons": {"vn": "Extra trait field."}},
            {"voices": [{"id": "vn", "description": "A voice.", "speechTraits": "steady",
                         "traits": [{"name": "pace", "value": "br\u200bisk", "basis": "casting", "evidence": []}]}],
             "reasons": {"vn": "Invisible character in trait value."}},
            {"voices": [{"id": "vn", **complete}], "reasons": {}},
            {"voices": [{"id": "vn", **complete}], "reasons": {"vn": "x" * 1001}},
            {"voices": [{"id": "vn", **complete}], "reasons": {"vn": "Control\u0007 character."}},
            {"voices": [{"id": "vn", **complete}], "reasons": {"vn": 7}},
            {"voices": [], "reasons": {"vn": "No upsert."}},
            {"updates": []},
        ]
        for response in cases:
            with self.subTest(response=str(response)[:80]):
                requests.responses = [response]
                with self.assertRaises(ValueError):
                    design_ensemble(source, cast, requests, policy=DEFAULT_POLICY)
        self.assertEqual(len(requests.calls), len(cases))
        self.assertEqual(cast, original)

    def test_valid_upsert_applies_completely_and_preserves_profile_metadata(self):
        source = deepcopy(valid_document()["source"])
        cast = deepcopy(valid_document())["cast"]
        cast["roles"][0]["id"] = "narrator_main"
        cast["voices"][1]["user_note"] = "deliberate user choice"
        cast["voices"].append({"id": "v2", "description": "Another local voice.", "traits": []})
        inherited = {"id": "old_v1", "description": "An established series voice.", "traits": [],
                     "series_origin": {"kind": "series_casting_reuse", "series_id": "s", "voice_id": "prior"}}
        cast["voices"].append(inherited)
        requests = RecordingRequests([{"voices": [{"id": "v2", "description": "A distinct second voice.",
                                                   "speechTraits": "bright quick phrases",
                                                   "traits": [{"name": "pitch", "value": "high", "basis": "casting",
                                                               "evidence": []}]}],
                                       "reasons": {"v2": "Contrast with the narrator."}}])
        updated, provenance = design_ensemble(source, cast, requests, policy=DEFAULT_POLICY)
        self.assertEqual(provenance["status"], "applied")
        revised = next(voice for voice in updated["voices"] if voice["id"] == "v2")
        self.assertEqual(revised["description"], "A distinct second voice.")
        self.assertEqual(revised["speechTraits"], "bright quick phrases")
        self.assertEqual(next(voice for voice in updated["voices"] if voice["id"] == "v1")["user_note"],
                         "deliberate user choice")
        self.assertEqual(len(requests.calls), 1)
        payload = requests.calls[0][1]
        self.assertEqual({voice["id"] for voice in payload["cast"]["voices"]}, {"vn", "v1", "v2", "old_v1"})
        self.assertTrue(payload["contrast_findings"])
        self.assertEqual(provenance["contrast_findings_before"], payload["contrast_findings"])

    def test_single_voice_cast_skips_the_request_but_records_the_audit(self):
        class NarratorOnly(EnsembleClient):
            def discover(self, payload):
                return cast_patch()
        client = NarratorOnly()
        document = self.run_analysis(client)
        self.assertNotIn("voice_ensemble", self.modes(client))
        ensemble = document["provenance"]["voice_ensemble"]
        self.assertEqual(ensemble["status"], "skipped")
        self.assertIn("fewer than two", ensemble["reason"])
        self.assertEqual(ensemble["contrast_findings_before"], [])
        self.assertEqual(ensemble["contrast_findings_after"], [])

    def test_redesign_can_echo_long_and_numerous_text_evidence_exactly(self):
        source = deepcopy(valid_document()["source"])
        source["blocks"][0]["text"] += " Softly." * 100
        cast = deepcopy(valid_document())["cast"]
        cast["roles"][0]["id"] = "narrator_main"
        evidence = [{"block_id": source["blocks"][0]["id"],
                     "quote": source["blocks"][0]["text"]}]
        evidence += [{"block_id": source["blocks"][0]["id"], "quote": " Softly." * count}
                     for count in range(1, 14)]
        cast["voices"][1]["traits"][0]["evidence"] = evidence

        class EchoRequests(RecordingRequests):
            def run(self, system, payload, validate, *, cancelled=None):
                profile = deepcopy(payload["cast"]["voices"][1])
                profile.update(description="A warmer coordinated voice.", speechTraits="steady even pace")
                return validate({"voices": [profile], "reasons": {profile["id"]: "Keep established evidence."}})

        updated, _ = design_ensemble(source, cast, EchoRequests(), policy=DEFAULT_POLICY)
        self.assertEqual(updated["voices"][1]["traits"][0]["evidence"], evidence)
        self.assertEqual(updated["voices"][1]["description"], "A warmer coordinated voice.")

    def test_skipped_ensemble_records_fingerprints_without_a_request(self):
        source = deepcopy(valid_document()["source"])
        cast = deepcopy(valid_document())["cast"]
        for voice in cast["voices"]:
            voice["series_origin"] = {"kind": "series_casting_reuse"}
        requests = RecordingRequests()
        updated, provenance = design_ensemble(source, cast, requests, policy=DEFAULT_POLICY)
        self.assertEqual(updated, cast)
        self.assertEqual(provenance["status"], "skipped")
        self.assertEqual(provenance["cast_fingerprint_before"], fingerprint(cast))
        self.assertEqual(provenance["cast_fingerprint_after"], fingerprint(cast))
        self.assertEqual(provenance["source_fingerprint"], fingerprint(source))
        self.assertEqual(provenance["request_characters"], 0)
        self.assertEqual(provenance["updated"], {})
        self.assertEqual(requests.calls, [])

    def test_oversized_request_fails_clearly_before_any_model_call(self):
        source = deepcopy(valid_document()["source"])
        cast = deepcopy(valid_document())["cast"]
        requests = RecordingRequests()
        requests.max_request_chars = 100
        with self.assertRaisesRegex(ValueError, "Voice ensemble request"):
            design_ensemble(source, cast, requests, policy=DEFAULT_POLICY)
        self.assertEqual(requests.calls, [])

    def test_completed_ensemble_responses_are_cached_and_replayed_exactly(self):
        def redesign(payload):
            voice_id = payload["cast"]["voices"][1]["id"]
            return {"voices": [{"id": voice_id, "description": "A coordinated voice.",
                                "speechTraits": "brisk careful phrases",
                                "traits": [{"name": "pace", "value": "brisk", "basis": "casting", "evidence": []}]}],
                    "reasons": {voice_id: "Casting coordination."}}
        first = EnsembleClient(ensemble=redesign)
        document = self.run_analysis(first)
        self.assertEqual(self.modes(first).count("voice_ensemble"), 1)
        replay = EnsembleClient(ensemble=redesign)
        repeated = self.run_analysis(replay)
        self.assertEqual(repeated, document)
        self.assertEqual(replay.calls, [])

    def test_cached_replay_revalidates_the_complete_cast(self):
        def redesign(payload):
            voice_id = payload["cast"]["voices"][1]["id"]
            echoed = deepcopy(next(trait for trait in payload["cast"]["voices"][1]["traits"]
                                   if trait["basis"] == "text"))
            return {"voices": [{"id": voice_id, "description": "A coordinated voice.",
                                "speechTraits": "brisk careful phrases",
                                "traits": [echoed, {"name": "pace", "value": "brisk", "basis": "casting",
                                                    "evidence": []}]}],
                    "reasons": {voice_id: "Casting coordination keeps the soft volume evidence."}}
        source = deepcopy(valid_document()["source"])
        cast = deepcopy(valid_document())["cast"]
        cast["roles"][0]["id"] = "narrator_main"
        client = EnsembleClient(ensemble=redesign)
        validator_calls = []
        inner = _validate_response(source, cast)

        def counting_validator(response):
            validator_calls.append(deepcopy(response))
            return inner(response)

        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            payload = _payload(source, cast, DEFAULT_POLICY, [])
            first = requests.run(ENSEMBLE_SYSTEM, payload, counting_validator)
            second = requests.run(ENSEMBLE_SYSTEM, payload, counting_validator)
        self.assertEqual(first, second)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(validator_calls), 2)
        self.assertEqual(first["cast"]["voices"][1]["description"], "A coordinated voice.")
        self.assertEqual(set(first), {"cast", "reasons"})

    def test_identical_declared_profiles_surface_before_and_resolve_after_design(self):
        def twin_voices(payload):
            unit = payload["target_units"][0]
            evidence = [{"block_id": unit["block_id"], "quote": "Mara"}]
            character_id, voice_id = payload["available_ids"]["characters"][0], payload["available_ids"]["voices"][0]
            traits = [{"name": name, "value": value, "basis": "casting", "evidence": []}
                      for name, value in (("resonance", "even and natural"),
                                          ("articulation", "precise but relaxed"),
                                          ("pace", "steady conversational"))]
            return cast_patch(
                characters=[{"id": character_id, "name": "Mara", "aliases": [],
                             "description": "A measured speaker.", "evidence": evidence, "voice_id": voice_id}],
                roles=[{"id": payload["available_ids"]["roles"][0], "kind": "dialogue",
                        "character_id": character_id, "voice_id": voice_id, "description": "Mara speaking."}],
                voices=[{"id": voice_id, "description": "A measured reading voice.", "traits": traits}])

        def redesign(payload):
            twin = next(voice for voice in payload["cast"]["voices"]
                        if voice["id"] != "voice_narrator_main")
            return {"voices": [{"id": twin["id"],
                                "description": "A brighter, quicker voice kept distinct from the narrator.",
                                "speechTraits": "bright quick phrases",
                                "traits": [{"name": "resonance", "value": "bright and forward", "basis": "casting",
                                            "evidence": []},
                                           {"name": "pace", "value": "brisk", "basis": "casting", "evidence": []}]}],
                    "reasons": {twin["id"]: "Resolve the identical known profile by resonance and pace."}}
        client = EnsembleClient(ensemble=redesign)
        client.discover = twin_voices
        document = self.run_analysis(client)
        ensemble = document["provenance"]["voice_ensemble"]
        self.assertEqual([finding["kind"] for finding in ensemble["contrast_findings_before"]],
                         ["identical_known_profile"])
        self.assertEqual(ensemble["contrast_findings_after"], [])
        _, payload = next(item for item in client.calls if item[1].get("mode") == "voice_ensemble")
        self.assertEqual(payload["contrast_findings"], ensemble["contrast_findings_before"])


if __name__ == "__main__":
    unittest.main()
