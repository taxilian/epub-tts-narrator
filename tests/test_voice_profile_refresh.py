"""Voice profile refresh changes only descriptions, traits, speechTraits, and their bindings."""

from copy import deepcopy
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (_completed_directory, continuation_directories,
    continuation_registry, fresh_registry, publish_directory)
from epub_tts_narrator.analysis_revision import _install, profile_revision_differences, revise_directories
from epub_tts_narrator.audition_scripts import FILENAME
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.pipeline import CachedRequests, fingerprint
from epub_tts_narrator.prompts import COMMON, VOICE_PROFILE_GUIDANCE
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.series import link_analysis
from epub_tts_narrator.voice_profile_refresh import (apply_profile_updates, audition_sidecar_rebind,
    profiles_changed, refresh_book_voice_profiles, refresh_voice_profiles, validate_profile_response)
from epub_tts_narrator.voice_profile_refresh_prompts import SYSTEM
from test_audition_scripts import bundle
from test_series import fixture, propose


DESCRIPTION = "A mature feminine voice with a bright forward tone, light gravel, and brisk, precise phrasing."
TRAITS = [{"name": "timbre", "value": "bright with light gravel", "basis": "casting", "evidence": []},
          {"name": "pace", "value": "brisk", "basis": "casting", "evidence": []}]
SPEECH_TRAITS = "light Irish accent; brisk, clipped delivery"
RESPONSE = {"description": DESCRIPTION, "traits": deepcopy(TRAITS), "speechTraits": SPEECH_TRAITS,
            "reason": "Keeps the established timbre under the current profile contract."}
AUDITION_TEXT = ("The tide is turning already, and we should reach the far light before it fails. "
                 "Keep the lantern low and tell me what you see along the water. "
                 "We can still choose the safer channel if you are unsure.")
AUDITION_SCRIPT = {"text": AUDITION_TEXT, "basis": "casting", "reason": "Calm original audition for the keeper."}


class Requests:
    def __init__(self):
        self.calls, self.active, self.max_active = [], 0, 0
        self.lock = Lock()

    def run(self, system, payload, validate, *, cancelled=None):
        with self.lock:
            self.calls.append((system, deepcopy(payload)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(.01)
            return validate(deepcopy(RESPONSE))
        finally:
            with self.lock:
                self.active -= 1


class Client:
    identity = {"model": "test", "base_url": "https://example.invalid"}
    model, base_url = "test", "https://example.invalid"

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def complete(self, system, payload):
        voice = payload["voice"]["id"]
        self.calls.append(voice)
        if self.fail_on == voice:
            self.fail_on = None
            raise ProviderError("Temporary interruption")
        return deepcopy(RESPONSE)


class DeadClient(Client):
    def complete(self, system, payload):
        raise ProviderError("endpoint down")


class BadClient(Client):
    def complete(self, system, payload):
        self.calls.append(payload["voice"]["id"])
        return {key: value for key, value in RESPONSE.items() if key != "speechTraits"}


class VoiceProfileRefreshTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.paths, self.documents = [], []
        for number in range(1, 4):
            path = self.root / f"book-{number}"
            document = fixture(sequel=number > 1)
            document["source"]["book"]["id"] = f"profile-book-{number}"
            if number == 1:
                registry = fresh_registry(path)
            else:
                registry = continuation_registry(self.paths[-1], document["source"])
                propose(document, self.documents[0]["cast"]["characters"][0]["id"])
            linked = link_analysis(document, registry)
            linked["provenance"]["voice_samples"] = {
                "version": 1, "source_fingerprint": fingerprint(linked["source"]),
                "cast_fingerprint": fingerprint(linked["cast"]),
                "scripts": {linked["cast"]["characters"][0]["voice_id"]: deepcopy(AUDITION_SCRIPT)},
                "samples": {}}
            publish_directory(linked, registry, path, export_bundle,
                              continue_from=self.paths[-1] if self.paths else None)
            self.paths.append(path)
            self.documents.append(linked)
        self.sidecar_voice = self.documents[0]["cast"]["characters"][0]["voice_id"]
        for path in self.paths:
            (path / "voices").mkdir()
            (path / "voices" / "reference.wav").write_bytes(b"untouched voice")
            (path / "render-work").mkdir()
            (path / "render-work" / "accepted.json").write_text("untouched cache")
            (path / "voice-aliases.json").write_text("untouched aliases")
        (self.paths[0] / FILENAME).write_text(json.dumps({
            "version": 1, "source_fingerprint": fingerprint(self.documents[0]["source"]),
            "cast_fingerprint": fingerprint(self.documents[0]["cast"]),
            "scripts": {self.sidecar_voice: deepcopy(AUDITION_SCRIPT)}, "samples": {}}), encoding="utf-8")

    def updates(self, document=None):
        document = document if document is not None else self.documents[0]
        return {voice["id"]: deepcopy(RESPONSE) for voice in document["cast"]["voices"]}

    def revised(self, number=0):
        return apply_profile_updates(self.documents[number], self.updates(self.documents[number]),
                                     provider={"model": "test"}, guidance_fingerprint="a" * 64)

    def sidecar_record(self):
        document = self.documents[0]
        return audition_sidecar_rebind(
            self.paths[0] / FILENAME, source_fingerprint=fingerprint(document["source"]),
            cast_fingerprint_before=fingerprint(document["cast"]),
            cast_fingerprint_after=fingerprint(self.revised()["cast"]),
            voice_ids=[voice["id"] for voice in document["cast"]["voices"]])

    def managed(self):
        names = ["analysis.json", "series-cast.json", "manifest.json", "review.json", "source.json",
                 "cast.json", "scripts/chapter-0001.jsonl", FILENAME, "voices/reference.wav",
                 "render-work/accepted.json", "voice-aliases.json"]
        return {path: {name: (path / name).read_bytes() for name in names if (path / name).exists()}
                for path in self.paths}

    def test_profile_scope_allowlist_rejects_every_other_change(self):
        base = deepcopy(self.documents[0])
        before = self.managed()
        self.assertEqual(profile_revision_differences(base, self.revised()), [])
        mutations = (
            ("source", lambda doc: doc["source"]["blocks"][0].update(text="changed source")),
            ("segments", lambda doc: doc["segments"][0].update(text="changed segment")),
            ("issues", lambda doc: doc["issues"].append(
                {"id": "x", "kind": "attribution", "target": "b1", "reason": "r", "resolved": False})),
            ("cast.characters", lambda doc: doc["cast"]["characters"][0].update(name="Changed")),
            ("cast.roles", lambda doc: doc["cast"]["roles"][0].update(description="Changed")),
            ("cast.voices identity or order", lambda doc: doc["cast"]["voices"].reverse()),
            ("cast.voices identity or order", lambda doc: doc["cast"]["voices"].pop()),
            ("cast.voice fields outside description/traits/speechTraits",
                lambda doc: doc["cast"]["voices"][0].update(sample_script=deepcopy(AUDITION_SCRIPT))),
            ("provenance", lambda doc: doc["provenance"].update(tool_version="changed")),
            ("provenance.voice_samples scripts or binding",
                lambda doc: doc["provenance"]["voice_samples"]["scripts"].clear()),
            ("provenance.voice_samples scripts or binding",
                lambda doc: doc["provenance"]["voice_samples"].update(cast_fingerprint="0" * 64)),
            ("document metadata", lambda doc: doc.update(extra={"unexpected": True})),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                candidate = deepcopy(base)
                mutate(candidate)
                differences = profile_revision_differences(base, candidate)
                self.assertTrue(any(expected in difference for difference in differences), differences)
                with self.assertRaisesRegex(ValueError, "only cast voice description, traits"):
                    revise_directories({self.paths[0]: candidate}, directories=self.paths,
                                       scope="voice-profiles", progress=lambda _: None)
                self.assertEqual(self.managed(), before)

    def test_profile_revision_publishes_and_propagates_the_chain(self):
        before = self.managed()
        result = revise_directories({self.paths[0]: self.revised()}, directories=self.paths,
                                    scope="voice-profiles",
                                    sidecars={self.paths[0]: {FILENAME: self.sidecar_record()}},
                                    progress=lambda _: None)
        for index, path in enumerate(self.paths):
            document, registry, _ = _completed_directory(path)
            self.assertEqual(continuation_directories(path), list(reversed(self.paths[:index])))
            if index == 0:
                for voice in document["cast"]["voices"]:
                    self.assertEqual(voice["description"], DESCRIPTION)
                    original = next(item for item in self.documents[0]["cast"]["voices"]
                                    if item["id"] == voice["id"])
                    factual = [trait for trait in original["traits"] if trait["basis"] == "text"]
                    self.assertEqual(voice["traits"], factual + TRAITS)
                    self.assertEqual(voice["speechTraits"], SPEECH_TRAITS)
                self.assertNotIn("speechTraits", self.documents[0]["cast"]["voices"][0])
            else:
                self.assertEqual(document["cast"], self.documents[index]["cast"])
            self.assertEqual((path / "voices/reference.wav").read_bytes(), b"untouched voice")
            self.assertEqual((path / "render-work/accepted.json").read_text(), "untouched cache")
            self.assertEqual((path / "voice-aliases.json").read_text(), "untouched aliases")
            self.assertEqual((path / "source.json").read_bytes(), before[path]["source.json"])
            self.assertEqual((path / "scripts/chapter-0001.jsonl").read_bytes(),
                             before[path]["scripts/chapter-0001.jsonl"])
        parent_document = _completed_directory(self.paths[0])[0]
        _, book2_registry, _ = _completed_directory(self.paths[1])
        self.assertEqual(book2_registry["books"][0]["cast"], parent_document["cast"])
        samples = parent_document["provenance"]["voice_samples"]
        self.assertEqual(samples["scripts"], {self.sidecar_voice: deepcopy(AUDITION_SCRIPT)})
        self.assertEqual(samples["cast_fingerprint"], fingerprint(parent_document["cast"]))
        audit = parent_document["provenance"]["voice_profile_refresh"]
        self.assertEqual(audit["guidance_fingerprint"], "a" * 64)
        self.assertEqual(audit["voices"][self.sidecar_voice]["reason"], RESPONSE["reason"])
        sidecar = json.loads((self.paths[0] / FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["scripts"][self.sidecar_voice]["text"], AUDITION_TEXT)
        self.assertEqual(sidecar["source_fingerprint"], fingerprint(self.documents[0]["source"]))
        self.assertEqual(sidecar["cast_fingerprint"], fingerprint(parent_document["cast"]))
        backup = Path(result["backup_directory"]) / "book-0001" / "before"
        self.assertEqual((backup / FILENAME).read_bytes(), before[self.paths[0]][FILENAME])
        self.assertEqual((backup / "analysis.json").read_bytes(), before[self.paths[0]]["analysis.json"])

    def test_default_scope_and_bad_scope_still_guard_their_contracts(self):
        before = self.managed()
        with self.assertRaisesRegex(ValueError, "preserve source, cast"):
            revise_directories({self.paths[0]: self.revised()}, directories=self.paths,
                               progress=lambda _: None)
        with self.assertRaisesRegex(ValueError, "Revision scope"):
            revise_directories({self.paths[0]: self.revised()}, directories=self.paths,
                               scope="everything", progress=lambda _: None)
        self.assertEqual(self.managed(), before)

    def test_interrupted_publication_restores_the_replaced_sidecar(self):
        before = self.managed()
        calls = 0

        def fail_on_ninth(source, destination):
            nonlocal calls
            calls += 1
            if calls == 9:
                raise OSError("interrupted publication")
            return _install(source, destination)

        with patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_on_ninth):
            with self.assertRaisesRegex(OSError, "interrupted publication"):
                revise_directories({self.paths[0]: self.revised()}, directories=self.paths,
                                   scope="voice-profiles",
                                   sidecars={self.paths[0]: {FILENAME: self.sidecar_record()}},
                                   progress=lambda _: None)
        self.assertEqual(self.managed(), before)
        for path in self.paths:
            _completed_directory(path)
            continuation_directories(path)
            self.assertFalse((path / ".analysis-update.lock").exists())

    def test_response_contract_requires_the_three_profile_fields_and_a_reason(self):
        self.assertEqual(validate_profile_response(deepcopy(RESPONSE)), RESPONSE)
        at_limit = deepcopy(RESPONSE)
        at_limit["speechTraits"] = "x" * 400
        self.assertEqual(validate_profile_response(at_limit)["speechTraits"], "x" * 400)
        long_description = dict(RESPONSE, description="A voice. " * 600)
        cases = [
            {key: value for key, value in RESPONSE.items() if key != "speechTraits"},
            {**RESPONSE, "extra": 1},
            {**RESPONSE, "description": ""},
            {**RESPONSE, "description": "123"},
            {**RESPONSE, "description": "[soft] A voice."},
            {**RESPONSE, "description": '"A voice."'},
            {**RESPONSE, "description": "A voice.\u200b"},
            long_description,
            {**RESPONSE, "traits": []},
            {**RESPONSE, "traits": RESPONSE["traits"] * 5},
            {**RESPONSE, "traits": [{"name": "timbre", "value": "soft", "basis": "text", "evidence": []}]},
            {**RESPONSE, "traits": [{"name": "timbre", "value": "soft", "basis": "casting",
                                     "evidence": [{"block_id": "b1", "quote": "x"}]}]},
            {**RESPONSE, "traits": [{"name": "timbre", "value": "soft", "basis": "casting",
                                     "evidence": [], "extra": 1}]},
            {**RESPONSE, "speechTraits": ""},
            {**RESPONSE, "speechTraits": "two\nlines"},
            {**RESPONSE, "speechTraits": "[accent] Irish"},
            {**RESPONSE, "speechTraits": "x" * 401},
            {**RESPONSE, "reason": ""},
            {**RESPONSE, "reason": "x" * 1001},
        ]
        for value in cases:
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                validate_profile_response(value)

    def test_requests_carry_profile_context_and_never_mutate_the_bundle(self):
        data, requests = bundle(), Requests()
        before = deepcopy(data)
        result = refresh_voice_profiles(data, requests, workers=1)
        self.assertEqual(data, before)
        self.assertEqual(set(result["profiles"]), {"vn", "v1"})
        self.assertEqual(result["guidance_fingerprint"], fingerprint(SYSTEM))
        self.assertEqual(result["cast_fingerprint_before"], fingerprint(data["cast"]))
        for system, payload in requests.calls:
            self.assertIn(COMMON, system)
            self.assertEqual(system.count(VOICE_PROFILE_GUIDANCE), 1)
            self.assertEqual(payload["mode"], "voice_profile_refresh")
            voice = payload["voice"]
            self.assertNotIn("sample_script", voice)
            profile = next(row for row in data["cast"]["voices"] if row["id"] == voice["id"])
            self.assertEqual(payload["current_description"], profile["description"])
            self.assertEqual(payload["current_traits"], profile.get("traits", []))
            self.assertEqual(payload["current_speech_traits"], profile.get("speechTraits"))
        payload = next(row for _, row in requests.calls if row["voice"]["id"] == "v1")
        self.assertEqual(payload["characters"][0]["name"], "Mara")
        self.assertEqual(payload["roles"][0]["id"], "mara")
        self.assertEqual(payload["source_evidence"],
                         [{"block_id": "b1", "quote": "softly"}, {"block_id": "b1", "quote": "Mara"}])

    def test_invalid_model_output_fails_without_writing_a_response_cache(self):
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaisesRegex(ValueError, "speechTraits"):
                refresh_voice_profiles(bundle(), CachedRequests(BadClient(), cache, repair_attempts=0))
            self.assertEqual(list(Path(cache).glob("*.json")), [])
            self.assertTrue(list((Path(cache) / "failed").glob("*.json")))

    def test_worker_bounds_and_selection_validation(self):
        data, requests = bundle(), Requests()
        for index in range(8):
            data["cast"]["voices"].append({"id": f"extra_{index}", "description": "A clear voice.", "traits": []})
        result = refresh_voice_profiles(data, requests, workers=5)
        self.assertEqual(requests.max_active, 5)
        self.assertEqual(len(result["profiles"]), 10)
        for kwargs in ({"workers": 0}, {"workers": 6}, {"workers": True}, {"workers": 1.5},
                       {"voice_ids": ["unknown"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                refresh_voice_profiles(bundle(), Requests(), **kwargs)
        with self.assertRaisesRegex(ValueError, "needs the configured analysis model"):
            refresh_voice_profiles(bundle(), None)
        self.assertEqual(refresh_voice_profiles(bundle(), None, voice_ids=[])["profiles"], {})

    def test_apply_updates_cover_exactly_the_existing_voices_without_secrets(self):
        document = fixture()
        document["cast"]["voices"][0]["sample_script"] = deepcopy(AUDITION_SCRIPT)
        document["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(document["source"]),
            "cast_fingerprint": fingerprint(document["cast"]),
            "scripts": {"v1": "kept"}, "samples": {"v1": "kept"}}
        before = deepcopy(document)
        revised = apply_profile_updates(document, self.updates(document), provider={"model": "test"},
                                        guidance_fingerprint="a" * 64)
        self.assertEqual(document, before)
        for key in ("source", "segments", "issues"):
            self.assertEqual(revised[key], document[key])
        self.assertEqual(revised["cast"]["characters"], document["cast"]["characters"])
        self.assertEqual(revised["cast"]["roles"], document["cast"]["roles"])
        self.assertEqual([voice["id"] for voice in revised["cast"]["voices"]], ["vn", "v1"])
        self.assertEqual(revised["cast"]["voices"][0]["sample_script"],
                         document["cast"]["voices"][0]["sample_script"])
        samples = revised["provenance"]["voice_samples"]
        self.assertEqual(samples["scripts"], {"v1": "kept"})
        self.assertEqual(samples["samples"], {"v1": "kept"})
        self.assertEqual(samples["cast_fingerprint"], fingerprint(revised["cast"]))
        self.assertFalse(profiles_changed(revised, self.updates(revised)))
        for updates, provider in (({"vn": RESPONSE}, {"model": "test"}),
                                  (dict(self.updates(document), v9=RESPONSE), {"model": "test"}),
                                  (self.updates(document), {"model": "test", "api_key": "do-not-persist"})):
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                apply_profile_updates(document, updates, provider=provider, guidance_fingerprint="a" * 64)

    def test_sidecar_rebind_policy(self):
        arguments = {"source_fingerprint": fingerprint(self.documents[0]["source"]),
                     "cast_fingerprint_before": fingerprint(self.documents[0]["cast"]),
                     "cast_fingerprint_after": "b" * 64,
                     "voice_ids": [voice["id"] for voice in self.documents[0]["cast"]["voices"]]}
        bound = audition_sidecar_rebind(self.paths[0] / FILENAME, **arguments)
        self.assertEqual(bound["scripts"], {self.sidecar_voice: deepcopy(AUDITION_SCRIPT)})
        self.assertEqual(bound["cast_fingerprint"], "b" * 64)
        stale = json.loads((self.paths[0] / FILENAME).read_text(encoding="utf-8"))
        stale["cast_fingerprint"] = "0" * 64
        (self.paths[0] / FILENAME).write_text(json.dumps(stale), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match this book's pre-refresh source and cast"):
            audition_sidecar_rebind(self.paths[0] / FILENAME, **arguments)
        stale["cast_fingerprint"] = fingerprint(self.documents[0]["cast"])
        stale["scripts"]["v9"] = deepcopy(AUDITION_SCRIPT)
        (self.paths[0] / FILENAME).write_text(json.dumps(stale), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid saved character audition script"):
            audition_sidecar_rebind(self.paths[0] / FILENAME, **arguments)
        del stale["scripts"]["v9"]
        stale["scripts"][self.sidecar_voice]["basis"] = "text"
        (self.paths[0] / FILENAME).write_text(json.dumps(stale), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid saved character audition script"):
            audition_sidecar_rebind(self.paths[0] / FILENAME, **arguments)
        missing = self.root / "absent" / FILENAME
        with self.assertRaises(ValueError):
            audition_sidecar_rebind(missing, **arguments)
        if os.name == "posix":
            link = self.root / "linked-sidecar.json"
            link.symlink_to(self.paths[0] / FILENAME)
            with self.assertRaisesRegex(ValueError, "regular"):
                audition_sidecar_rebind(link, **arguments)

    def invoke_refresh(self, client, *, argv_extra=()):
        config = deepcopy(DEFAULTS)
        config["provider"].update(model="test-model", api_key="test-key")
        config["analysis"]["cache_dir"] = str(self.root / "cache")
        output, errors = io.StringIO(), io.StringIO()
        with (patch("epub_tts_narrator.config.load_config", return_value=config),
              patch("epub_tts_narrator.provider.OpenAICompatibleClient", return_value=client),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["refresh-voice-profiles", str(self.paths[0]), "--config", "global.toml",
                         *argv_extra])
        return code, output.getvalue(), errors.getvalue()

    def test_book_workflow_updates_profiles_conditioning_and_sidecar_atomically(self):
        from epub_tts_narrator.renderer.bundle import load_bundle
        from epub_tts_narrator.renderer.speech_traits import bind_plan_speech_traits
        client = Client()
        code, output, _ = self.invoke_refresh(client)
        self.assertEqual(code, 0)
        self.assertIn("Updated voice profiles in", output)
        document, _, _ = _completed_directory(self.paths[0])
        for voice in document["cast"]["voices"]:
            self.assertEqual(voice["description"], DESCRIPTION)
            self.assertEqual(voice["speechTraits"], SPEECH_TRAITS)
        sidecar = json.loads((self.paths[0] / FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["cast_fingerprint"], fingerprint(document["cast"]))
        self.assertEqual(sidecar["scripts"][self.sidecar_voice]["text"], AUDITION_TEXT)
        self.assertEqual(len(list((self.root / "cache").glob("*.json"))), 2)
        self.assertEqual((self.paths[0] / "voices/reference.wav").read_bytes(), b"untouched voice")
        loaded = load_bundle(self.paths[0])
        self.assertEqual([voice["speechTraits"] for voice in loaded["cast"]["voices"]],
                         [SPEECH_TRAITS] * 2)
        row = document["segments"][0]
        plan = {"chapters": [{"chunks": [{"segment": row}]}]}
        self.assertNotIn("speech_traits",
                         bind_plan_speech_traits(plan, self.documents[0]["cast"])["chapters"][0]["chunks"][0])
        bound = bind_plan_speech_traits(plan, document["cast"])["chapters"][0]["chunks"][0]
        self.assertEqual(bound["speech_traits"], {row["voice_id"]: SPEECH_TRAITS})
        manifest_hash = hashlib.sha256((self.paths[0] / "manifest.json").read_bytes()).hexdigest()
        before_calls = len(client.calls)
        code, output, _ = self.invoke_refresh(client)
        self.assertEqual(code, 0)
        self.assertIn("no changes to publish", output)
        self.assertEqual(len(client.calls), before_calls + 2)
        self.assertEqual(hashlib.sha256((self.paths[0] / "manifest.json").read_bytes()).hexdigest(),
                         manifest_hash)

    def test_transport_failure_publishes_nothing_and_releases_the_lock(self):
        before = self.managed()
        code, output, errors = self.invoke_refresh(DeadClient())
        self.assertEqual(code, 1)
        self.assertIn("endpoint down", errors)
        self.assertEqual(self.managed(), before)
        self.assertFalse((self.paths[0] / ".analysis-update.lock").exists())
        self.assertFalse((self.paths[0] / ".analysis-revisions").exists())

    def test_cli_rejects_out_of_range_workers(self):
        code, _, errors = self.invoke_refresh(Client(), argv_extra=("--workers", "0"))
        self.assertEqual(code, 1)
        self.assertIn("workers", errors)


if __name__ == "__main__":
    unittest.main()
