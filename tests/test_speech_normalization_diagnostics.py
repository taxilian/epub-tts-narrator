"""Spelling repairs identify the faulty field without discarding source scope."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.pipeline import CachedRequests, fingerprint, write_json
from epub_tts_narrator.prompts import PROMPT_VERSION
from epub_tts_narrator.speech_normalization import _validate_scan, normalize_document_speech
from epub_tts_narrator.speech_prompts_v1 import SCAN
from test_speech_normalization import document, edit


class Client:
    identity = {"model": "offline-diagnostics", "base_url": "https://example.invalid"}

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        return self.respond(payload)


def empty_scan(payload):
    return {"segments": [{"segment_id": row["segment_id"], "edits": []}
                         for row in payload["target_segments"]]}


class SpeechNormalizationDiagnosticTests(unittest.TestCase):
    def diagnosis(self, response, rows):
        with self.assertRaises(ValueError) as caught:
            _validate_scan(response, rows, [])
        return str(caught.exception)

    def test_segment_schema_errors_are_not_misreported_as_id_ordering(self):
        row = document("wull")["segments"][0]
        cases = [
            ({"segment_id": row["id"]}, "edits", "missing"),
            ({"segment_id": row["id"], "edits": [], "text": "well"}, "text", "unexpected|extra"),
            ({"segment_id": row["id"], "edits": None}, "edits", "array|list"),
        ]
        for value, field, explanation in cases:
            with self.subTest(value=value):
                message = self.diagnosis({"segments": [value]}, [row])
                self.assertIn("segments[0]", message)
                self.assertIn(field, message)
                self.assertRegex(message.lower(), explanation)
                self.assertNotIn("all segment IDs in source order", message)

    def test_coverage_error_lists_counts_missing_duplicate_and_unknown_ids(self):
        rows = document("One.", "Two.", "Three.")["segments"]
        received = [rows[0]["id"], rows[0]["id"], "unknown-segment"]
        message = self.diagnosis({"segments": [
            {"segment_id": identifier, "edits": []} for identifier in received]}, rows)
        self.assertRegex(message.lower(), r"expected\s+3")
        self.assertRegex(message.lower(), r"received\s+3")
        self.assertIn("missing", message.lower())
        self.assertIn("duplicate", message.lower())
        self.assertRegex(message.lower(), "unknown|unexpected")
        for identifier in [row["id"] for row in rows] + ["unknown-segment"]:
            self.assertIn(identifier, message)
        self.assertIn("segments[1]", message)

    def test_order_error_identifies_first_expected_and_received_segment(self):
        rows = document("One.", "Two.")["segments"]
        message = self.diagnosis({"segments": [
            {"segment_id": row["id"], "edits": []} for row in reversed(rows)]}, rows)
        self.assertIn("segments[0]", message)
        self.assertRegex(message, "(?i)expected.*" + rows[0]["id"])
        self.assertRegex(message, "(?i)(received|got).*" + rows[1]["id"])

    def test_original_mismatch_identifies_edit_source_and_exact_slice(self):
        row = document('“wull,” he said.')["segments"][0]
        # Segment-relative coordinates must remain useful for a passage that
        # starts partway through its parent source block.
        row.update(start=50, end=50 + len(row["text"]))
        change = edit(row["text"], "wull", "well")
        change["original"] = "wul"
        message = self.diagnosis({"segments": [{"segment_id": row["id"], "edits": [change]}]}, [row])
        for value in ("segments[0].edits[0]", row["id"], row["block_id"], "original", "wul", "wull"):
            self.assertIn(value, message)
        self.assertRegex(message, r"text\[1:5\]")
        self.assertRegex(message.lower(), "source|excerpt")

    def test_noop_edit_explains_that_it_must_be_omitted(self):
        row = document("wull")["segments"][0]
        change = edit(row["text"], "wull", "wull", direction="")
        message = self.diagnosis({"segments": [{"segment_id": row["id"], "edits": [change]}]}, [row])
        self.assertIn("segments[0].edits[0]", message)
        self.assertIn("original", message)
        self.assertIn("replacement", message)
        self.assertRegex(message.lower(), "identical|equal|same")
        self.assertRegex(message.lower(), "omit|remove")

    def test_overlap_error_identifies_both_ranges_and_second_edit(self):
        row = document("wull wull")["segments"][0]
        change = edit(row["text"], "wull", "well")
        message = self.diagnosis({"segments": [{"segment_id": row["id"], "edits": [change, change]}]}, [row])
        self.assertIn("segments[0].edits[1]", message)
        self.assertRegex(message.lower(), "overlap")
        self.assertIn("0", message)
        self.assertIn("4", message)
        self.assertRegex(message.lower(), "previous|prior")

    def test_repair_receives_precise_diagnosis_and_all_original_targets(self):
        source = document("wull", "Another ordinary line.")
        before = deepcopy(source)

        def respond(payload):
            if "correction" in payload:
                return empty_scan(payload)
            response = empty_scan(payload)
            response["segments"][0]["edits"] = None
            return response

        client = Client(respond)
        with tempfile.TemporaryDirectory() as temporary:
            requests = CachedRequests(client, temporary, repair_attempts=1)
            normalize_document_speech(source, requests, workers=1)
            self.assertEqual(len(client.calls), 2)
            initial, repaired = [payload for _, payload in client.calls]
            self.assertEqual(repaired["target_segments"], initial["target_segments"])
            self.assertEqual(repaired["context"], initial["context"])
            self.assertEqual(repaired["correction"]["previous_response"]["segments"][0]["edits"], None)
            self.assertIn("segments[0].edits", repaired["correction"]["validator_error"])
            self.assertRegex(repaired["correction"]["validator_error"].lower(), "array|list")
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 1)
        self.assertEqual(source, before)

    def test_exhausted_failure_names_batch_chapter_and_inspectable_response(self):
        source = document("Ordinary speech.", "wull")
        source["source"]["chapters"] = [{"id": "ch1", "title": "Chapter of Whispers"}]

        def respond(payload):
            response = empty_scan(payload)
            if payload["target_segments"][0]["text"] == "wull":
                response["segments"][0]["edits"] = None
            return response

        client = Client(respond)
        with tempfile.TemporaryDirectory() as temporary:
            requests = CachedRequests(client, temporary, repair_attempts=4)
            with patch("epub_tts_narrator.speech_normalization.BATCH_SEGMENTS", 1), \
                 self.assertRaises(ValueError) as caught:
                normalize_document_speech(source, requests, workers=1)
            message = str(caught.exception)
            self.assertRegex(message.lower(), r"batch\s+2\s*/\s*2")
            self.assertIn("Chapter of Whispers", message)
            self.assertIn("segments[0].edits", message)
            failed = sorted((Path(temporary) / "failed").glob("*-attempt5.json"))
            self.assertEqual(len(failed), 1)
            self.assertIn(str(failed[0]), message)
            saved = json.loads(failed[0].read_text())
            self.assertIsNone(saved["segments"][0]["edits"])
            # The other completed batch remains reusable despite this failure.
            self.assertEqual(len(list(Path(temporary).glob("*.json"))), 1)
        self.assertNotIn("speech_normalization", source["provenance"])

    def test_existing_successful_cache_key_remains_usable_without_provider_calls(self):
        source = document("Ordinary speech.", "wull")
        # Freeze the pre-diagnostic request shape: context belongs in failures,
        # not in a changed scan request that needlessly invalidates saved work.
        payload = {
            "mode": "speech_spelling_scan", "version": 1, "batch": 1,
            "book": deepcopy(source["source"]["book"]), "established_names": ["Vern"],
            "target_segments": [{
                "segment_id": row["id"], "block_id": row["block_id"],
                "text": row["text"], "kind": row["kind"],
                "role": deepcopy(source["cast"]["roles"][0]), "delivery": deepcopy(row["delivery"]),
            } for row in source["segments"]],
            "context": [{"block_id": row["id"], "text": row["text"]}
                        for row in source["source"]["blocks"]],
        }
        client = Client(lambda payload: self.fail("Already validated response should be reused"))
        key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                           "system": SCAN, "payload": payload})
        with tempfile.TemporaryDirectory() as temporary:
            write_json(Path(temporary) / f"{key}.json", empty_scan(payload))
            requests = CachedRequests(client, temporary)
            normalize_document_speech(source, requests, workers=1)
            self.assertEqual(requests.cache_hits, 1)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
