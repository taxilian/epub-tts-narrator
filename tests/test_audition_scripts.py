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
    FILENAME, VERSION, ensure_bundle_scripts, prepare_document_samples, prepare_samples, validate_script,
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
