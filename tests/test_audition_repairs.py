"""Audition repairs explain the actual defect and retain partial successes."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.audition_scripts import prepare_samples, validate_script
from epub_tts_narrator.pipeline import CachedRequests
from test_audition_scripts import SCRIPT, Client, bundle


class RepairClient(Client):
    def __init__(self):
        super().__init__()
        self.payloads = []

    def complete(self, system, payload):
        self.payloads.append(deepcopy(payload))
        if "correction" not in payload:
            return {"text": SCRIPT.replace("We have", "Yesss, we have"), "reason": "Hissing voice."}
        return {"text": SCRIPT, "reason": "Ordinary spelling; the voice profile supplies the hiss."}


class AuditionRepairTests(unittest.TestCase):
    def test_length_correct_hissing_script_gets_specific_repair_and_caches_only_valid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            client = RepairClient()
            requests = CachedRequests(client, directory, repair_attempts=1)
            prepared = prepare_samples(bundle(), requests, voice_ids=["v1"], workers=1)
            self.assertEqual(prepared["scripts"]["v1"]["text"], SCRIPT)
            error = client.payloads[1]["correction"]["validator_error"]
            self.assertIn("Yesss", error)
            self.assertIn("ordinary spelling", error)
            self.assertNotIn("25–60 words", error)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            replayed = prepare_samples(bundle(), requests, voice_ids=["v1"], workers=1)
            self.assertEqual(replayed, prepared)
            self.assertEqual(len(client.payloads), 2)

    def test_whitespace_formatting_is_normalized_without_a_model_retry(self):
        text = " \n\t" + SCRIPT.replace(". ", ".\n\n") + "\r\n "
        result = validate_script({"text": text, "reason": "An audition."})
        self.assertEqual(result["text"], SCRIPT)
        with self.assertRaisesRegex(ValueError, "U\\+001B"):
            validate_script({"text": "\x1b" + SCRIPT, "reason": "An audition."})

    def test_word_and_character_limit_errors_include_actual_measurements(self):
        for text, expected in (("Too short.", "2 words"), ("word " * 61, "61 words"),
                               (("characteristically " * 40).strip(), "characters")):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                validate_script({"text": text, "reason": "An audition."})

    def test_optional_preparation_keeps_good_scripts_and_records_failed_voice_for_retry(self):
        class MixedClient(Client):
            fail = True

            def complete(self, system, payload):
                result = super().complete(system, payload)
                if payload["voice"]["id"] == "v1" and self.fail:
                    result["text"] = "Too short."
                return result

        with tempfile.TemporaryDirectory() as directory:
            client = MixedClient()
            requests = CachedRequests(client, directory, repair_attempts=0)
            data = bundle()
            before = deepcopy(data)
            first = prepare_samples(data, requests, workers=1, allow_incomplete=True)
            self.assertEqual(data, before)
            self.assertEqual(set(first["scripts"]), {"vn"})
            self.assertEqual(set(first["samples"]), {"vn"})
            self.assertEqual(set(first["pending"]), {"v1"})
            self.assertEqual(first["pending"]["v1"]["name"], "Mara")
            self.assertIn("2 words", first["pending"]["v1"]["error"])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            data["manifest"]["provenance"]["voice_samples"] = first
            no_calls = prepare_samples(data, None, allow_incomplete=True)
            self.assertEqual(no_calls["pending"], first["pending"])
            client.fail = False
            complete = prepare_samples(data, requests, workers=1, allow_incomplete=True)
            self.assertNotIn("pending", complete)
            self.assertEqual(set(complete["samples"]), {"vn", "v1"})
            self.assertEqual(client.calls, ["vn", "v1", "v1"])

    def test_optional_preparation_does_not_swallow_corrupt_saved_scripts(self):
        data = bundle()
        data["manifest"]["provenance"]["voice_samples"] = {
            "scripts": {"v1": {"text": "Too short.", "basis": "casting", "reason": "Corrupt."}}}
        with self.assertRaises(ValueError):
            prepare_samples(data, None, allow_incomplete=True)


if __name__ == "__main__":
    unittest.main()
