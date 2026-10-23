from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.pipeline import CachedRequests, fingerprint, write_json
from epub_tts_narrator.prompts import PROMPT_VERSION
from epub_tts_narrator.provider import OutputTruncatedError, ProviderError
from epub_tts_narrator.speech_normalization import normalize_document_speech
from epub_tts_narrator.speech_prompts_v1 import SCAN
from epub_tts_narrator.speech_prompts import SCAN as compact_scan
from test_speech_normalization import document, edit


class Client:
    identity = {"model": "offline-spelling-truncation"}

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        return self.respond(payload)


def response(payload):
    if payload["mode"] == "speech_spelling_scan":
        return {"segments": [{"segment_id": row["segment_id"],
                              "edits": [edit(row["text"], "wull", "well")]}
                             for row in payload["target_segments"]]}
    return {"votes": [{"edit_id": row["edit_id"], "verdict": "approve", "reason": "Same word."}
                      for row in payload["proposals"]]}


class SpeechNormalizationTruncationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)

    def test_truncated_scan_splits_whole_segments_and_retains_context_and_jury(self):
        source = document("wull one", "wull two", "wull three", "wull four")
        before = deepcopy(source)

        def respond(payload):
            if payload["mode"] == "speech_spelling_scan" and len(payload["target_segments"]) > 1:
                raise OutputTruncatedError("fixture output limit")
            return response(payload)

        client, progress = Client(respond), []
        result = normalize_document_speech(source, CachedRequests(client, self.cache),
                                           workers=1, progress=progress.append)
        scans = [row for row in client.calls if row["mode"] == "speech_spelling_scan"]
        self.assertEqual([len(row["target_segments"]) for row in scans], [4, 2, 1, 1, 2, 1, 1])
        originals = scans[0]["target_segments"]
        for child in scans[1:]:
            self.assertEqual(child["context"], scans[0]["context"])
            visible = child["target_segments"] + child["context_segments"]
            self.assertEqual({row["segment_id"] for row in visible}, {row["segment_id"] for row in originals})
            for row in visible:
                self.assertIn(row, originals)
        self.assertEqual(source, before)
        self.assertEqual([row["spoken_text"] for row in result["segments"]],
                         ["well one", "well two", "well three", "well four"])
        self.assertEqual(len([row for row in client.calls if row["mode"] == "speech_spelling_review"]), 3)
        self.assertTrue(any("subdivid" in row for row in progress))

    def test_resume_skips_known_truncated_parent_and_reuses_validated_child(self):
        source = document("wull one", "wull two")

        def interrupted(payload):
            if len(payload["target_segments"]) > 1:
                raise OutputTruncatedError("fixture output limit")
            if payload["target_segments"][0]["text"] == "wull two":
                raise ProviderError("fixture transport unavailable", status_code=503)
            return response(payload)

        with self.assertRaises(ProviderError):
            normalize_document_speech(source, CachedRequests(Client(interrupted), self.cache), workers=1)
        resumed = Client(response)
        result = normalize_document_speech(source, CachedRequests(resumed, self.cache), workers=1)
        scans = [row for row in resumed.calls if row["mode"] == "speech_spelling_scan"]
        self.assertEqual([row["target_segments"][0]["text"] for row in scans], ["wull two"])
        self.assertEqual(result["provenance"]["speech_normalization"]["applied_edits"], 2)

    def test_single_segment_truncation_is_contextual_and_never_splits_source(self):
        source = document("wull")
        client = Client(lambda payload: (_ for _ in ()).throw(OutputTruncatedError("fixture output limit")))
        with self.assertRaises(OutputTruncatedError) as caught:
            normalize_document_speech(source, CachedRequests(client, self.cache), workers=1)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Speech spelling scan batch 1/1", str(caught.exception))
        self.assertIn("provider.max_output_tokens", str(caught.exception))
        self.assertNotIn("speech_normalization", source["provenance"])

    def test_subdivision_stops_after_two_levels(self):
        client = Client(lambda payload: (_ for _ in ()).throw(OutputTruncatedError("fixture output limit")))
        with self.assertRaises(OutputTruncatedError):
            normalize_document_speech(document(*[f"wull {number}" for number in range(8)]),
                                       CachedRequests(client, self.cache), workers=1)
        self.assertEqual([len(row["target_segments"]) for row in client.calls], [8, 4, 2])

    def test_other_provider_errors_do_not_trigger_subdivision(self):
        client = Client(lambda payload: (_ for _ in ()).throw(ProviderError("quota", status_code=429)))
        with self.assertRaises(ProviderError) as caught:
            normalize_document_speech(document("wull one", "wull two"),
                                       CachedRequests(client, self.cache), workers=1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(caught.exception.status_code, 429)

    def test_child_cannot_return_edits_for_read_only_sibling(self):
        def respond(payload):
            if len(payload["target_segments"]) > 1:
                raise OutputTruncatedError("fixture output limit")
            wrong = response(payload)
            wrong["segments"][0]["segment_id"] = payload["context_segments"][0]["segment_id"]
            return wrong

        source = document("wull one", "wull two")
        with self.assertRaises(ValueError):
            normalize_document_speech(source, CachedRequests(Client(respond), self.cache, repair_attempts=0), workers=1)
        self.assertNotIn("speech_normalization", source["provenance"])

    def test_truncation_during_validation_repair_recovers_without_accepting_failed_edits(self):
        def respond(payload):
            if payload["mode"] == "speech_spelling_scan" and len(payload["target_segments"]) > 1:
                if "correction" in payload:
                    raise OutputTruncatedError("fixture repair output limit")
                invalid = response(payload)
                invalid["segments"][0]["edits"] = None
                return invalid
            return response(payload)

        client = Client(respond)
        result = normalize_document_speech(document("wull one", "wull two"),
                                           CachedRequests(client, self.cache, repair_attempts=4), workers=1)
        self.assertEqual(result["provenance"]["speech_normalization"]["applied_edits"], 2)
        scans = [row for row in client.calls if row["mode"] == "speech_spelling_scan"]
        self.assertEqual([len(row["target_segments"]) for row in scans], [2, 2, 1, 1])
        self.assertEqual(len(list((self.cache / "failed").glob("*.json"))), 1)

    def test_successful_parent_cache_takes_precedence_over_truncation_hint(self):
        def respond(payload):
            if payload["mode"] == "speech_spelling_scan" and len(payload["target_segments"]) > 1:
                raise OutputTruncatedError("fixture output limit")
            return response(payload)

        source, client = document("wull one", "wull two"), Client(respond)
        normalize_document_speech(source, CachedRequests(client, self.cache), workers=1)
        parent = client.calls[0]
        key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                           "system": compact_scan, "payload": parent})
        write_json(self.cache / f"{key}.json", response(parent))
        cached = Client(lambda payload: self.fail("Validated cache should avoid all model calls"))
        result = normalize_document_speech(source, CachedRequests(cached, self.cache), workers=1)
        self.assertNotIn("scan_subdivisions", result["provenance"]["speech_normalization"])
        self.assertEqual(cached.calls, [])

    def test_truncated_jury_is_contextual_without_splitting_or_applying_edits(self):
        def respond(payload):
            if payload["mode"] == "speech_spelling_review":
                raise OutputTruncatedError("fixture jury output limit")
            return response(payload)

        source, client = document("wull one", "wull two"), Client(respond)
        with self.assertRaises(OutputTruncatedError) as caught:
            normalize_document_speech(source, CachedRequests(client, self.cache), workers=1)
        self.assertIn("review (speech_spelling_reviewer_1) batch 1/1", str(caught.exception))
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("speech_normalization", source["provenance"])


if __name__ == "__main__":
    unittest.main()
