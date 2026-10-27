"""Accent regressions exercise real payloads and guards, never fake model judgment."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.pipeline import CachedRequests, check_cast
from epub_tts_narrator.prompts import DISCOVER, VOICE_PROFILE_GUIDANCE
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.voice_ensemble import _validate_response
from epub_tts_narrator.voice_profile_refresh import refresh_voice_profiles, validate_profile_response
from epub_tts_narrator.voice_profile_refresh_prompts import SYSTEM as PROFILE_SYSTEM
from scripts import validate_voice_accents as evaluation
from test_voice_ensemble import RecordingRequests
from test_voice_profile_refresh import Requests as ProfileRequests, RESPONSE as PROFILE_RESPONSE
from test_voice_prompt_refresh import Requests as DescriptionRequests


INHERITED_RESPONSE = {
    "description": "An adult performer with a light mid-range tone, a General American accent, and a brisk clipped rhythm.",
    "traits": [{"name": "accent", "value": "General American accent", "basis": "casting", "evidence": []},
               {"name": "pace", "value": "brisk clipped rhythm", "basis": "casting", "evidence": []}],
    "speechTraits": "General American accent; brisk clipped rhythm",
    "reason": "Preserves the inherited accent while refining compatible cadence.",
}


class VoiceAccentGuidanceTests(unittest.TestCase):
    def test_contrastive_fixtures_are_valid_and_distinguish_background_from_name(self):
        for case_id in evaluation.CASES:
            with self.subTest(case=case_id):
                bundle = evaluation.fixture(case_id)
                check_cast(bundle["source"], bundle["cast"])
        immigrant = evaluation.fixture("japanese_immigrant")["cast"]["characters"][0]
        local = evaluation.fixture("same_name_local")["cast"]["characters"][0]
        unknown = evaluation.fixture("name_only")["cast"]["characters"][0]
        self.assertEqual({immigrant["name"], local["name"], unknown["name"]}, {"Himari"})
        self.assertEqual(len({immigrant["description"], local["description"], unknown["description"]}), 3)

    def test_immigration_context_and_guidance_reach_every_refinement_without_book_mutation(self):
        for case_id in ("japanese_immigrant", "mexican_immigrant"):
            for stage, request_type in (("profile", ProfileRequests), ("prompt", DescriptionRequests),
                                        ("ensemble", RecordingRequests)):
                with self.subTest(case=case_id, stage=stage):
                    requests = request_type()
                    evaluation.run_case(case_id, stage, requests)
                    self.assertEqual(len(requests.calls), 1)
                    system, payload = requests.calls[0]
                    characters = payload["cast"]["characters"] if stage == "ensemble" else payload["characters"]
                    self.assertEqual(characters[0]["description"], evaluation.CASES[case_id]["text"])
                    self.assertEqual(characters[0]["evidence"], [{"block_id": "b1", "quote": evaluation.CASES[case_id]["text"]}])
                    self.assertEqual(system.count(VOICE_PROFILE_GUIDANCE), 1)
                    normalized = " ".join(system.split())
                    self.assertIn("Fluent, clear, or grammatical English does not imply General American pronunciation", normalized)
                    self.assertIn("A quotation establishing nationality or upbringing supports the background, not an unmentioned audible accent", normalized)
                    self.assertIn("not independent evidence or proof of user approval", normalized)

    def test_discovery_receives_background_in_source_and_the_same_accent_contract(self):
        class Requests:
            def run(self, system, payload, validate):
                self.system, self.payload = system, deepcopy(payload)
                proposal = evaluation.fixture("japanese_immigrant")["cast"]
                return validate({**proposal, "merges": [], "issues": [], "summary": "Synthetic casting."})

        requests = Requests()
        evaluation.run_case("japanese_immigrant", "discovery", requests)
        self.assertEqual(requests.system, DISCOVER)
        self.assertIn(VOICE_PROFILE_GUIDANCE, requests.system)
        self.assertEqual(requests.payload["target_units"][0]["text"], evaluation.CASES["japanese_immigrant"]["text"])
        self.assertNotIn("General American", json.dumps(requests.payload["known_cast"]))

    def test_user_prompt_correction_remains_distinct_from_automatic_profile_default(self):
        requests = DescriptionRequests()
        evaluation.run_case("user_scottish", "prompt", requests)
        system, payload = requests.calls[0]
        self.assertEqual(payload["current_prompt"], evaluation.CASES["user_scottish"]["current_prompt"])
        self.assertIn("Edinburgh", payload["current_prompt"])
        self.assertIn("General American", payload["voice"]["description"])
        self.assertIn("Preserve its distinct, explicit", system)
        self.assertIn("description-only response", system)

    def test_inherited_profile_remains_supplied_as_established_context(self):
        class Requests:
            def run(self, system, payload, validate, **kwargs):
                self.calls = [(system, payload)]
                return validate(deepcopy(INHERITED_RESPONSE))

        requests = Requests()
        evaluation.run_case("inherited_american", "profile", requests)
        system, payload = requests.calls[0]
        self.assertIn("General American", payload["prior_series_voice"]["speechTraits"])
        self.assertEqual(payload["prior_series_voice"]["series_origin"]["kind"], "series_casting_reuse")
        self.assertIn("Preserve that voice's accent", system)
        self.assertIn("originally labeled basis=casting", system)
        self.assertIn("Those background facts", system)
        self.assertIn("do not describe a change in the established speaking voice", system)

    def test_inherited_declared_accent_is_guarded_without_freezing_other_performance_traits(self):
        data = evaluation.fixture("inherited_american")
        prior = data["series_context"]["voices"][0]
        accepted = validate_profile_response(INHERITED_RESPONSE, prior_series_voice=prior)
        self.assertEqual(accepted, INHERITED_RESPONSE)
        self.assertNotEqual(accepted["speechTraits"], prior["speechTraits"])
        mutations = (
            lambda value: value["traits"][0].update(value="Mexican Spanish-influenced accent"),
            lambda value: value["traits"].pop(0),
            lambda value: value["traits"].append({"name": "dialect", "value": "Mexican Spanish", "basis": "casting", "evidence": []}),
            lambda value: value.update(description="An adult performer with a Mexican Spanish-influenced accent."),
            lambda value: value.update(speechTraits="Mexican Spanish-influenced accent; brisk rhythm"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = deepcopy(INHERITED_RESPONSE)
                mutate(value)
                with self.assertRaisesRegex(ValueError, "Inherited accent"):
                    validate_profile_response(value, prior_series_voice=prior)

    def test_previously_cached_recast_is_revalidated_against_inherited_accent_guard(self):
        class Client:
            identity = {"model": "offline-fixture", "base_url": "https://example.invalid"}

            def __init__(self):
                self.calls = 0

            def complete(self, system, payload):
                self.calls += 1
                return deepcopy(PROFILE_RESPONSE if self.calls == 1 else INHERITED_RESPONSE)

        class Capture:
            def run(self, system, payload, validate, **kwargs):
                self.payload = payload
                return validate(deepcopy(INHERITED_RESPONSE))

        data, capture, client = evaluation.fixture("inherited_american"), Capture(), Client()
        refresh_voice_profiles(data, capture, voice_ids=["v1"], workers=1)
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, directory, repair_attempts=0)
            # Simulate a success cached before the inherited-identity guard existed.
            requests.run(PROFILE_SYSTEM, capture.payload, validate_profile_response)
            result = refresh_voice_profiles(data, requests, voice_ids=["v1"], workers=1)
            self.assertEqual(result["profiles"]["v1"], INHERITED_RESPONSE)
        self.assertEqual(client.calls, 2)

    def test_ensemble_cannot_recast_source_explicit_or_inherited_accents(self):
        for case_id, error in (("explicit_american", "text-supported traits"),
                               ("inherited_american", "established series voice")):
            with self.subTest(case=case_id):
                bundle = evaluation.fixture(case_id)
                voice = deepcopy(bundle["cast"]["voices"][-1])
                voice.pop("series_origin", None)
                voice["traits"] = [{"name": "accent", "value": "Japanese-influenced English", "basis": "casting", "evidence": []}]
                voice["description"] = "An adult performer with a Japanese-influenced English accent."
                voice["speechTraits"] = "Japanese-influenced English accent"
                with self.assertRaisesRegex(ValueError, error):
                    _validate_response(bundle["source"], bundle["cast"])(
                        {"voices": [voice], "reasons": {"v1": "Ignore the established accent."}})

    def test_offline_fixture_check_never_loads_configuration_or_writes_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(evaluation, "load_config", side_effect=AssertionError("No configuration needed")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.main(["--check-fixtures", "-o", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_live_harness_resumes_validated_requests_and_rejects_different_run_overwrite(self):
        class Client:
            identity = {"model": "offline-fixture", "base_url": "https://example.invalid"}

            def __init__(self):
                self.calls, self.interrupt = [], True

            def complete(self, system, payload):
                book_id = payload["book"]["id"]
                self.calls.append(book_id)
                if book_id.endswith("mexican_immigrant") and self.interrupt:
                    self.interrupt = False
                    raise ProviderError("Synthetic interruption")
                return deepcopy(PROFILE_RESPONSE)

        client = Client()
        settings = deepcopy(DEFAULTS)
        settings["analysis"]["repair_attempts"] = 0
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            args = ["--config", "unused.toml", "-o", str(output), "--case", "japanese_immigrant", "--case", "mexican_immigrant"]
            with patch.object(evaluation, "load_config", return_value=settings), patch.object(evaluation, "OpenAICompatibleClient", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ProviderError, "Synthetic interruption"):
                    evaluation.main(args)
                partial = json.loads(output.read_text())
                self.assertEqual(len(partial["results"]), 1)
                self.assertEqual(partial["status"], "running")
                self.assertEqual(evaluation.main(args), 0)
                complete = json.loads(output.read_text())
                self.assertEqual(complete["status"], "complete")
                self.assertEqual(complete["assessment"], "manual_review_required")
                self.assertEqual(len(complete["results"]), 2)
                before = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "different evaluation"):
                    evaluation.main(args + ["--case", "same_name_local"])
                self.assertEqual(output.read_bytes(), before)
        self.assertEqual(client.calls, ["accent-eval-japanese_immigrant", "accent-eval-mexican_immigrant", "accent-eval-mexican_immigrant"])
