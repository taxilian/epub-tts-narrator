"""Audition preparation preserves book text and persists only casting material."""

from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
import tempfile
import time
from unittest.mock import patch
import unittest

from epub_audiobook.audition_scripts import (
    FILENAME, PROMPT_VERSION, SYSTEM, VERSION, ensure_bundle_scripts, prepare_document_samples,
    prepare_samples, validate_script,
)
from epub_audiobook.contracts import validate_document
from epub_audiobook.pipeline import CachedRequests, fingerprint
from epub_audiobook.provider import ProviderError
from test_contracts import valid_document


SCRIPT = ("I have checked the path ahead, and the bridge is still standing. "
          "Keep close as we cross, then tell me what you saw by the river. "
          "We have time to listen before deciding where to go next.")
BOOK_TEXT = ("There is still time to find our friends and bring them home. "
             "I remember the road that brought us here, and I know where the old bridge stands. "
             "Follow me quietly, and we will find our way together.")


def bundle(document=None):
    document = valid_document() if document is None else document
    positions = {block["id"]: block["chapter_id"] for block in document["source"]["blocks"]}
    return {"source": deepcopy(document["source"]), "cast": deepcopy(document["cast"]),
            "manifest": {"source_fingerprint": fingerprint(document["source"]), "provenance": deepcopy(document["provenance"])},
            "chapters": [{"entry": {"title": "Chapter One", "path": "scripts/chapter-0001.jsonl"},
                          "rows": [dict(deepcopy(item), chapter_id=positions[item["block_id"]], sequence=index)
                                   for index, item in enumerate(document["segments"])]}]}


class Requests:
    def __init__(self):
        self.calls, self.active, self.max_active = [], 0, 0
        self.lock = Lock()

    def run(self, system, payload, validate, *, cancelled=None):
        with self.lock:
            self.calls.append(deepcopy(payload))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(.01)
            return validate({"text": SCRIPT, "reason": "A calm original audition matching this character's cautious manner."})
        finally:
            with self.lock:
                self.active -= 1


class Client:
    identity = {"model": "test", "base_url": "https://example.invalid"}
    model, base_url = "test", "https://example.invalid"

    def __init__(self, fail_once=None):
        self.calls, self.fail_once = [], fail_once

    def complete(self, system, payload):
        voice_id = payload["voice"]["id"]
        self.calls.append(voice_id)
        if self.fail_once == voice_id:
            self.fail_once = None
            raise ProviderError("Temporary interruption")
        return {"text": SCRIPT, "reason": "A character-specific original audition for " + voice_id + "."}


class AuditionScriptTests(unittest.TestCase):
    def test_sufficient_exact_book_speech_uses_no_model_and_keeps_source_mapping(self):
        document = valid_document()
        document["source"]["blocks"][0]["text"] = BOOK_TEXT
        document["cast"]["characters"][0]["evidence"] = []
        document["cast"]["voices"][1]["traits"] = []
        speech = document["segments"][1]
        speech.update(text=BOOK_TEXT, start=0, end=len(BOOK_TEXT))
        document["segments"] = [speech, document["segments"][2]]
        data, requests = bundle(document), Requests()
        before = deepcopy(data)
        result = prepare_samples(data, requests, voice_ids=["v1"])
        self.assertEqual(requests.calls, [])
        self.assertEqual(result["scripts"], {})
        sample = result["samples"]["v1"]
        self.assertEqual(sample["reference_text"], BOOK_TEXT)
        self.assertEqual(sample["source"]["kind"], "book_excerpt")
        self.assertEqual(sample["source"]["source_ranges"], [{"start": 0, "end": len(BOOK_TEXT)}])
        self.assertEqual(data, before)

    def test_insufficient_text_generates_character_specific_casting_script_with_no_book_edits(self):
        document = valid_document()
        before = deepcopy(document)
        requests = Requests()
        result = prepare_document_samples(document, requests)
        self.assertEqual(validate_document(result), [])
        self.assertEqual(document, before)
        for key in ("source", "cast", "segments", "issues"):
            self.assertEqual(result[key], document[key])
        self.assertEqual(len(requests.calls), 2)
        payload = next(row for row in requests.calls if row["voice"]["id"] == "v1")
        self.assertEqual(payload["mode"], "audition_script")
        self.assertEqual(payload["characters"][0]["name"], "Mara")
        self.assertEqual(payload["roles"][0]["id"], "mara")
        self.assertEqual(payload["short_speech_examples"], ["“Hello.”"])
        prepared = result["provenance"]["voice_samples"]
        self.assertEqual(prepared["source_fingerprint"], fingerprint(document["source"]))
        self.assertEqual(prepared["cast_fingerprint"], fingerprint(document["cast"]))
        for voice_id, script in prepared["scripts"].items():
            self.assertEqual(script["basis"], "casting")
            self.assertTrue(25 <= len(script["text"].split()) <= 60)
            self.assertEqual(prepared["samples"][voice_id]["source"]["kind"], "casting_script")
            self.assertIn("not a quotation", prepared["samples"][voice_id]["source"]["detail"])

    def test_cached_scripts_replay_without_new_calls(self):
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            first = prepare_samples(bundle(), requests, workers=1)
            second = prepare_samples(bundle(), requests, workers=1)
        self.assertEqual(first, second)
        self.assertEqual(client.calls, ["vn", "v1"])

    def test_interrupted_run_reuses_every_successful_cached_script(self):
        client = Client(fail_once="v1")
        with tempfile.TemporaryDirectory() as directory:
            requests = CachedRequests(client, Path(directory), repair_attempts=0)
            with self.assertRaises(ProviderError):
                prepare_samples(bundle(), requests, workers=1)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            result = prepare_samples(bundle(), requests, workers=1)
        self.assertEqual(client.calls, ["vn", "v1", "v1"])
        self.assertEqual(set(result["scripts"]), {"vn", "v1"})

    def test_voice_aliases_share_one_generated_script_and_one_request(self):
        data, requests = bundle(), Requests()
        alias = lambda value: "v1" if value == "vn" else value
        result = prepare_samples(data, requests, canonical_id=alias)
        self.assertEqual(len(requests.calls), 1)
        self.assertEqual(requests.calls[0]["voice"]["id"], "v1")
        self.assertEqual(result["scripts"]["vn"], result["scripts"]["v1"])
        self.assertEqual(result["samples"]["vn"], result["samples"]["v1"])
        self.assertEqual({row["id"] for row in requests.calls[0]["roles"]}, {"narrator", "mara"})

    def test_refresh_replaces_only_targeted_group_in_every_saved_location(self):
        data, requests = bundle(), Requests()
        original = validate_script({"text": BOOK_TEXT, "reason": "Older casting instructions."})
        data["cast"]["voices"].append({"id": "unrelated", "description": "Other", "traits": [],
                                         "sample_script": deepcopy(original)})
        for voice in data["cast"]["voices"]:
            voice["sample_script"] = deepcopy(original)
        scripts = {voice["id"]: deepcopy(original) for voice in data["cast"]["voices"]}
        data["manifest"]["provenance"]["voice_samples"] = {"scripts": deepcopy(scripts)}
        data["audition_scripts"] = deepcopy(scripts)
        before = deepcopy(data)
        canonical = lambda value: "v1" if value == "vn" else value
        result = prepare_samples(data, requests, canonical_id=canonical, voice_ids=["vn", "vn"], refresh=True)
        self.assertEqual(len(requests.calls), 1)
        self.assertEqual(requests.calls[0]["version"], PROMPT_VERSION)
        self.assertNotIn("sample_script", requests.calls[0]["voice"])
        self.assertEqual(result["scripts"]["v1"]["text"], SCRIPT)
        self.assertEqual(result["scripts"]["vn"], result["scripts"]["v1"])
        self.assertEqual(result["scripts"]["unrelated"], original)
        self.assertEqual(result["samples"]["vn"]["reference_text"], SCRIPT)
        self.assertEqual(result["replaced_voice_ids"], ["v1", "vn"])
        self.assertEqual(result["cast_fingerprint"], fingerprint(before["cast"]))
        self.assertEqual(data, before)

    def test_refresh_source_only_persists_old_fallback_suppression_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data = bundle()
            data["directory"] = directory
            old = validate_script({"text": SCRIPT, "reason": "Older creative fallback."})
            for voice in data["cast"]["voices"]:
                voice["sample_script"] = deepcopy(old)
            data["manifest"]["provenance"]["voice_samples"] = {
                "scripts": {voice["id"]: deepcopy(old) for voice in data["cast"]["voices"]}}
            before = deepcopy(data)
            source_sample = {"reference_text": BOOK_TEXT, "source": {"kind": "book_excerpt"}}
            with patch("audiobook_renderer.samples.select_book_sample", return_value=source_sample), \
                    patch("audiobook_renderer.samples.select_sample", return_value=source_sample), \
                    patch("epub_audiobook.config.load_config", side_effect=AssertionError("No model needed")):
                result = ensure_bundle_scripts(data, voice_ids=["v1"], refresh=True)
            record = json.loads((Path(directory) / FILENAME).read_text())
            self.assertEqual(record["version"], VERSION)
            self.assertEqual(record["replaced_voice_ids"], ["v1"])
            self.assertEqual(record["scripts"], {"vn": old})
            self.assertEqual(record["cast_fingerprint"], fingerprint(data["cast"]))
            self.assertEqual(result["audition_script_overrides"], ["v1"])
            self.assertEqual(data, before)
            # Once the new selector declines source speech, old provenance and
            # inline fallbacks must not silently return on a subsequent run.
            with patch("epub_audiobook.config.load_config", side_effect=RuntimeError("Model needed now")), \
                    self.assertRaisesRegex(RuntimeError, "Model needed now"):
                ensure_bundle_scripts(data, voice_ids=["v1"])
            self.assertFalse((Path(directory) / ".audition-scripts.lock").exists())
            self.assertEqual(json.loads((Path(directory) / FILENAME).read_text()), record)

    def test_refresh_v1_sidecar_uses_current_cached_guidance_and_preserves_unselected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = bundle()
            data["directory"] = directory
            old = validate_script({"text": BOOK_TEXT, "reason": "Old guidance."})
            record = {"version": 1, "source_fingerprint": data["manifest"]["source_fingerprint"],
                      "cast_fingerprint": fingerprint(data["cast"]), "scripts": {"v1": old, "vn": old}}
            (root / FILENAME).write_text(json.dumps(record))
            config = {"provider": {}, "analysis": {"cache_dir": root / "cache", "repair_attempts": 0,
                      "max_request_chars": 200000, "attribution_workers": 5}}
            client = Client()
            with patch("epub_audiobook.config.load_config", return_value=config), \
                    patch("epub_audiobook.provider.OpenAICompatibleClient", return_value=client):
                first = ensure_bundle_scripts(data, voice_ids=["v1"], refresh=True, progress=lambda text: None)
                second = ensure_bundle_scripts(data, voice_ids=["v1"], refresh=True, progress=lambda text: None)
            self.assertEqual(client.calls, ["v1"])
            self.assertEqual(first, second)
            self.assertEqual(first["audition_scripts"]["v1"]["text"], SCRIPT)
            self.assertEqual(first["audition_scripts"]["vn"], old)
            self.assertEqual(first["cast"], data["cast"])
            self.assertFalse((root / ".audition-scripts.lock").exists())

    def test_refresh_failure_keeps_original_sidecar_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = bundle()
            data["directory"] = directory
            old = validate_script({"text": BOOK_TEXT, "reason": "Old guidance."})
            record = {"version": VERSION, "source_fingerprint": data["manifest"]["source_fingerprint"],
                      "cast_fingerprint": fingerprint(data["cast"]), "scripts": {"v1": old, "vn": old}}
            path = root / FILENAME
            path.write_text(json.dumps(record))
            before = path.read_bytes()
            config = {"provider": {}, "analysis": {"cache_dir": root / "cache", "repair_attempts": 0,
                      "max_request_chars": 200000, "attribution_workers": 5}}
            with patch("epub_audiobook.config.load_config", return_value=config), \
                    patch("epub_audiobook.provider.OpenAICompatibleClient", return_value=Client(fail_once="v1")), \
                    self.assertRaises(ProviderError):
                ensure_bundle_scripts(data, voice_ids=["v1"], refresh=True, progress=lambda text: None)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / ".audition-scripts.lock").exists())

    def test_refresh_rejects_invalid_override_metadata_before_model(self):
        for replaced in (None, "v1", ["unknown"], [{}]):
            with self.subTest(replaced=replaced), tempfile.TemporaryDirectory() as directory:
                data = bundle()
                data["directory"] = directory
                record = {"version": VERSION, "source_fingerprint": data["manifest"]["source_fingerprint"],
                          "cast_fingerprint": fingerprint(data["cast"]), "scripts": {},
                          "replaced_voice_ids": replaced}
                (Path(directory) / FILENAME).write_text(json.dumps(record))
                with patch("epub_audiobook.config.load_config", side_effect=AssertionError("No model allowed")), \
                        self.assertRaisesRegex(ValueError, "replaced character"):
                    ensure_bundle_scripts(data, refresh=True)

    def test_new_creative_scripts_strip_only_balanced_outer_speech_quotes(self):
        inner = SCRIPT.replace("the path ahead", 'the path called "Moon Road"')
        for opening, closing in (("“", "”"), ('"', '"')):
            text = SCRIPT if opening == '"' else inner
            self.assertEqual(validate_script({"text": opening + text + closing, "reason": "Audition."})["text"], text)
        self.assertIn("Do not wrap the script or individual statements", SYSTEM)

    def test_new_creative_scripts_reject_separately_wrapped_statements(self):
        sentences = SCRIPT.split(". ")
        for opening, closing in (("“", "”"), ('"', '"'), ("‘", "’"), ("'", "'")):
            text = " ".join(opening + sentence.rstrip(".") + "." + closing for sentence in sentences)
            with self.subTest(opening=opening), self.assertRaisesRegex(ValueError, "separately quoted statements"):
                validate_script({"text": text, "reason": "Audition."})
        mixed = " ".join(opening + sentence.rstrip(".") + "." + closing for sentence, (opening, closing)
                         in zip(sentences, (("“", "”"), ('"', '"'), ("‘", "’"))))
        with self.assertRaisesRegex(ValueError, "separately quoted statements"):
            validate_script({"text": mixed, "reason": "Audition."})

    def test_new_creative_script_preserves_quoted_wording_and_apostrophes(self):
        text = SCRIPT.replace("the path ahead", 'the path called "Moon Road"').replace(
            "the bridge", "our friends’ bridge").replace("We have", "We've still got")
        self.assertEqual(validate_script({"text": text, "reason": "Audition."})["text"], text)
        quoted = "“" + text + "”"
        self.assertEqual(validate_script({"text": quoted, "reason": "Audition."})["text"], text)
        quoted_with_connective = SCRIPT.replace("the path ahead", '“the path ahead” or “the safer road”')
        self.assertEqual(validate_script({"text": quoted_with_connective, "reason": "Audition."})["text"],
                         quoted_with_connective)

    def test_existing_quoted_creative_scripts_can_be_loaded_and_refreshed(self):
        text = " ".join("“" + sentence.rstrip(".") + ".”" for sentence in SCRIPT.split(". "))
        old = {"text": text, "reason": "Legacy audition.", "basis": "casting"}
        data = bundle()
        for voice in data["cast"]["voices"]:
            voice["sample_script"] = deepcopy(old)
        before = deepcopy(data)
        existing = prepare_samples(data, None)
        self.assertEqual(existing["scripts"]["v1"], old)
        requests = Requests()
        refreshed = prepare_samples(data, requests, voice_ids=["v1"], refresh=True)
        self.assertEqual(refreshed["scripts"]["v1"]["text"], SCRIPT)
        self.assertEqual(refreshed["scripts"]["vn"], old)
        self.assertEqual(len(requests.calls), 1)
        self.assertEqual(data, before)

    def test_prepared_provenance_sidecar_or_inline_script_needs_no_configuration_or_api(self):
        for location in ("provenance", "sidecar", "inline"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                data = bundle()
                data["directory"] = directory
                scripts = {voice["id"]: validate_script({"text": SCRIPT, "reason": "Prepared for this character."})
                           for voice in data["cast"]["voices"]}
                if location == "provenance":
                    data["manifest"]["provenance"]["voice_samples"] = {"scripts": scripts}
                elif location == "sidecar":
                    record = {"version": VERSION, "source_fingerprint": data["manifest"]["source_fingerprint"],
                              "cast_fingerprint": fingerprint(data["cast"]), "scripts": scripts}
                    (Path(directory) / FILENAME).write_text(json.dumps(record), encoding="utf-8")
                else:
                    for voice in data["cast"]["voices"]:
                        voice["sample_script"] = scripts[voice["id"]]
                with patch("epub_audiobook.config.load_config", side_effect=AssertionError("No config required")), \
                        patch("epub_audiobook.provider.OpenAICompatibleClient", side_effect=AssertionError("No model required")):
                    prepared = ensure_bundle_scripts(data)
                self.assertEqual(prepared["audition_scripts"], scripts)

    def test_legacy_upgrade_writes_bound_sidecar_and_never_changes_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = bundle()
            data["directory"] = str(root / "book")
            Path(data["directory"]).mkdir()
            sentinels = {name: (Path(data["directory"]) / name) for name in ("source.json", "cast.json", "analysis.json")}
            for name, path in sentinels.items():
                path.write_text("Original " + name, encoding="utf-8")
            before = {name: path.read_bytes() for name, path in sentinels.items()}
            config = {"provider": {}, "analysis": {"cache_dir": root / "cache", "repair_attempts": 0,
                      "max_request_chars": 200000, "attribution_workers": 5}}
            client = Client()
            with patch("epub_audiobook.config.load_config", return_value=config), \
                    patch("epub_audiobook.provider.OpenAICompatibleClient", return_value=client):
                result = ensure_bundle_scripts(data, progress=lambda message: None)
            path = Path(data["directory"]) / FILENAME
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["source_fingerprint"], data["manifest"]["source_fingerprint"])
            self.assertEqual(record["cast_fingerprint"], fingerprint(data["cast"]))
            self.assertEqual(result["audition_scripts"], record["scripts"])
            self.assertEqual({name: path.read_bytes() for name, path in sentinels.items()}, before)
            self.assertFalse((Path(data["directory"]) / ".audition-scripts.lock").exists())

    def test_corrupt_foreign_or_nonregular_sidecar_is_blocked_without_model(self):
        for mutation in ("source", "cast", "version", "scripts", "missing_scripts", "invalid_json", "symlink", "directory"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                data = bundle()
                data["directory"] = directory
                path = Path(directory) / FILENAME
                record = {"version": VERSION, "source_fingerprint": data["manifest"]["source_fingerprint"],
                          "cast_fingerprint": fingerprint(data["cast"]), "scripts": {}}
                if mutation in {"source", "cast"}:
                    record[mutation + "_fingerprint"] = "foreign"
                elif mutation == "version":
                    record["version"] = 999
                elif mutation == "scripts":
                    record["scripts"] = []
                elif mutation == "missing_scripts":
                    record.pop("scripts")
                if mutation == "symlink":
                    path.symlink_to(Path(directory) / "absent.json")
                elif mutation == "directory":
                    path.mkdir()
                else:
                    path.write_text("not json" if mutation == "invalid_json" else json.dumps(record), encoding="utf-8")
                with patch("epub_audiobook.config.load_config", side_effect=AssertionError("No config allowed")), \
                        self.assertRaises(ValueError):
                    ensure_bundle_scripts(data)

    def test_generated_script_validation_rejects_unsafe_or_unusable_text_before_cache_acceptance(self):
        for text in ("Too short.", "word " * 70, SCRIPT.upper(), "Sssstay " + SCRIPT,
                     "https://example.invalid " + SCRIPT, "[softly] " + SCRIPT, SCRIPT + "\x1b",
                     SCRIPT.rstrip(".") + "—"):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                validate_script({"text": text, "reason": "Original audition."})
        self.assertEqual(validate_script({"text": SCRIPT, "reason": "Original audition."})["basis"], "casting")

    def test_workers_are_bounded_and_missing_voices_run_concurrently(self):
        data = bundle()
        for index in range(8):
            data["cast"]["voices"].append({"id": f"extra_{index}", "description": "Another voice", "traits": []})
        requests = Requests()
        result = prepare_samples(data, requests, workers=5)
        self.assertEqual(requests.max_active, 5)
        self.assertEqual(len(result["scripts"]), 10)
        for workers in (0, 6, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                prepare_samples(data, requests, workers=workers)


if __name__ == "__main__":
    unittest.main()
