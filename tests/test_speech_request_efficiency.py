from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from epub_tts_narrator.speech_prompts import pack_payload
from epub_tts_narrator.speech_normalization import normalize_document_speech
from epub_tts_narrator.speech_prompts_v1 import REVIEW as legacy_review, SCAN as legacy_scan
from test_speech_normalization import document, Requests


def scan_payload():
    role = {"id": "speaker", "kind": "dialogue", "description": "A quiet, careful traveler.",
            "voice_id": "voice", "character_id": "character"}
    segments = [{"segment_id": f"b{number}:0:120", "block_id": f"b{number}",
                 "text": f"wull, the quiet traveler waited beside the stone bridge while the morning mist drifted over the river. Passage {number}.",
                 "kind": "dialogue", "role": deepcopy(role),
                 "delivery": {"direction": "", "evidence": []}} for number in range(20)]
    return {"mode": "speech_spelling_scan", "version": 1, "batch": 1,
            "book": {"id": "fixture-book", "title": "Invented examples"},
            "established_names": ["Vern"], "target_segments": segments,
            "context": [{"block_id": row["block_id"], "text": row["text"]} for row in segments]}


def context_texts(payload):
    segments = {row["segment_id"]: row for key in ("target_segments", "context_segments")
                for row in payload.get(key, [])}
    return {row["block_id"]: (row["text"] if "text" in row
                              else "".join(segments[identifier]["text"] for identifier in row["segment_ids"]))
            for row in payload["context"]}


class SpeechRequestEfficiencyTests(unittest.TestCase):
    def test_scan_deduplicates_roles_and_exact_source_without_losing_data(self):
        original = scan_payload()
        before = deepcopy(original)
        packed = pack_payload(original)
        self.assertEqual(original, before)
        self.assertEqual(context_texts(packed), context_texts(original))
        self.assertEqual(packed["roles"], {"speaker": original["target_segments"][0]["role"]})
        for expected, actual in zip(original["target_segments"], packed["target_segments"]):
            self.assertEqual(actual["text"], expected["text"])
            self.assertEqual(actual["segment_id"], expected["segment_id"])
            self.assertEqual(packed["roles"][actual["role_id"]], expected["role"])
            self.assertEqual(actual.get("delivery", {"direction": "", "evidence": []}), expected["delivery"])
        size = lambda value: len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
        self.assertLess(size(packed), size(original) * .75)

    def test_review_targets_only_proposed_segments_and_preserves_surrounding_source(self):
        original = scan_payload()
        selected = original["target_segments"][7]
        original.update(mode="speech_spelling_review", reviewer_id="reviewer-1", private_lens="fixture lens",
                        proposals=[{"edit_id": selected["segment_id"] + ":0", "segment_id": selected["segment_id"],
                                    "start": 0, "end": 4, "original": "wull", "replacement": "well",
                                    "direction": "Use a quiet regional drawl.", "reason": "Same spoken word."}])
        packed = pack_payload(original)
        self.assertEqual([row["segment_id"] for row in packed["target_segments"]], [selected["segment_id"]])
        self.assertEqual(context_texts(packed), context_texts(original))
        self.assertEqual(packed["proposals"], original["proposals"])
        self.assertEqual(packed["private_lens"], original["private_lens"])
        visible = packed["target_segments"] + packed["context_segments"]
        self.assertEqual({row["segment_id"] for row in visible},
                         {row["segment_id"] for row in original["target_segments"]})
        for row in visible:
            expected = next(item for item in original["target_segments"] if item["segment_id"] == row["segment_id"])
            self.assertEqual(row["text"], expected["text"])
            self.assertEqual(packed["roles"][row["role_id"]], expected["role"])

    def test_partial_blocks_and_repeated_text_are_not_discarded_as_redundant(self):
        original = scan_payload()
        original["context"][0]["text"] += " Narration beyond the target."
        original["context"][1]["text"] *= 2
        self.assertEqual(context_texts(pack_payload(original)), context_texts(original))

    def test_review_retains_unedited_segment_beyond_truncated_block_context(self):
        original = scan_payload()
        sibling = original["target_segments"][0]
        sibling["text"] = "The unedited source remains visible. " * 100
        sibling["delivery"] = {"direction": "Speak quietly.", "evidence": [{"block_id": "b0", "quote": "unedited source"}]}
        original["context"][0]["text"] = sibling["text"][:1800]
        selected = original["target_segments"][1]
        original.update(mode="speech_spelling_review", reviewer_id="reviewer-1", private_lens="fixture lens",
                        proposals=[{"edit_id": selected["segment_id"] + ":0", "segment_id": selected["segment_id"]}])
        packed = pack_payload(original)
        retained = next(row for row in packed["context_segments"] if row["segment_id"] == sibling["segment_id"])
        self.assertEqual(retained["text"], sibling["text"])
        self.assertEqual(retained["delivery"], sibling["delivery"])
        self.assertEqual(context_texts(packed), context_texts(original))

    def test_subdivision_context_retains_sibling_segments_and_delivery_evidence(self):
        original = scan_payload()
        original["context_segments"] = original["target_segments"][1:]
        original["target_segments"] = original["target_segments"][:1]
        original["target_segments"][0]["delivery"] = {
            "direction": "Speak urgently.", "evidence": [{"block_id": "b0", "quote": "quiet traveler"}]}
        packed = pack_payload(original)
        self.assertEqual(context_texts(packed), context_texts(original))
        self.assertEqual(len(packed["context_segments"]), 19)
        self.assertEqual(packed["target_segments"][0]["delivery"], original["target_segments"][0]["delivery"])

    def test_actual_scan_and_review_requests_shrink_without_changing_normalization(self):
        for changed in (1, 20):
            with self.subTest(changed=changed):
                source = document(*[("wull" if number < changed else "well")
                                    + f", the quiet traveler waited beside the stone bridge while morning mist crossed the river. Passage {number}."
                                    for number in range(20)])
                legacy = Requests({"wull": "well"})
                with patch("epub_tts_narrator.speech_prompts.SCAN", legacy_scan), \
                        patch("epub_tts_narrator.speech_prompts.REVIEW", legacy_review), \
                        patch("epub_tts_narrator.speech_prompts.pack_payload", side_effect=lambda value: value):
                    before = normalize_document_speech(source, legacy, workers=1)
                compact = Requests({"wull": "well"})
                after = normalize_document_speech(source, compact, workers=1)
                # The stage marker's input fingerprint records the guidance that
                # produced it, so prompt generations legitimately differ here.
                for result in (after, before):
                    result["provenance"]["speech_normalization"].pop("input_fingerprint")
                self.assertEqual(after, before)
                size = lambda calls: sum(len(system.encode()) + len(json.dumps(payload, ensure_ascii=False).encode())
                                         for system, payload in calls)
                self.assertLess(size(compact.calls), size(legacy.calls) * .75)
                reviews = [payload for _, payload in compact.calls if payload["mode"] == "speech_spelling_review"]
                self.assertEqual(len(reviews), 3)
                self.assertTrue(all(len(payload["target_segments"]) == changed for payload in reviews))
                for review in reviews:
                    self.assertNotIn("votes", review)
                    for index, key in enumerate(review):
                        if key == "reviewer_id":
                            self.assertEqual(list(review)[index:], ["reviewer_id", "private_lens"])


if __name__ == "__main__":
    unittest.main()
