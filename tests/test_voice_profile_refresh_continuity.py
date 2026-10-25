from copy import deepcopy
import unittest

from epub_tts_narrator.voice_profile_refresh import apply_profile_updates, refresh_voice_profiles
from epub_tts_narrator.analysis_revision import profile_revision_differences
from test_series import fixture
from test_voice_profile_refresh import RESPONSE, Requests, bundle


class VoiceProfileRefreshContinuityTests(unittest.TestCase):
    def test_profile_revision_does_not_allow_unrelated_cast_or_format_metadata(self):
        original = fixture()
        for mutate in (lambda candidate: candidate["cast"].update(unrelated="change"),
                       lambda candidate: candidate.update(format="another-format")):
            candidate = deepcopy(original)
            mutate(candidate)
            with self.subTest(candidate_format=candidate["format"]):
                self.assertTrue(profile_revision_differences(original, candidate))

    def test_stale_audition_provenance_is_not_silently_rebound(self):
        document = fixture()
        document["provenance"]["voice_samples"] = {"cast_fingerprint": "stale", "scripts": {}, "samples": {}}
        updates = {voice["id"]: deepcopy(RESPONSE) for voice in document["cast"]["voices"]}

        with self.assertRaisesRegex(ValueError, "pre-refresh cast"):
            apply_profile_updates(document, updates, provider={"model": "test"},
                                  guidance_fingerprint="a" * 64)

    def test_existing_text_supported_traits_survive_profile_refresh(self):
        document = fixture()
        before = deepcopy(document)
        evidence_trait = deepcopy(document["cast"]["voices"][1]["traits"][0])
        updates = {voice["id"]: deepcopy(RESPONSE) for voice in document["cast"]["voices"]}
        updates["v1"]["traits"].append({"name": evidence_trait["name"], "value": "different casting choice",
                                        "basis": "casting", "evidence": []})

        refreshed = apply_profile_updates(document, updates, provider={"model": "test"},
                                          guidance_fingerprint="a" * 64)

        traits = refreshed["cast"]["voices"][1]["traits"]
        self.assertIn(evidence_trait, traits)
        self.assertEqual([trait for trait in traits if trait["name"] == evidence_trait["name"]],
                         [evidence_trait])
        self.assertEqual(document, before)

    def test_requests_receive_parent_voice_and_the_supplied_cast(self):
        data, requests = bundle(), Requests()
        previous = {"id": "v1", "description": "An established clear voice.", "traits": [],
                    "speechTraits": "Neutral General American accent; deliberate rhythm"}
        data["series_context"] = {"voices": [previous]}

        refresh_voice_profiles(data, requests, workers=1)

        payload = next(payload for _, payload in requests.calls if payload["voice"]["id"] == "v1")
        self.assertEqual(payload["prior_series_voice"], previous)
        self.assertEqual({voice["id"] for voice in payload["known_cast"]}, {"vn", "v1"})

    def test_parent_without_initial_traits_is_rejected_before_model_calls(self):
        data, requests = bundle(), Requests()
        data["series_context"] = {"voices": [{"id": "v1", "description": "Legacy voice", "traits": []}]}

        with self.assertRaisesRegex(ValueError, "originating"):
            refresh_voice_profiles(data, requests, workers=1)

        self.assertEqual(requests.calls, [])
