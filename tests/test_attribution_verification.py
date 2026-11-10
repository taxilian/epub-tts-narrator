"""Focused attribution verification uses the real audit and publication boundaries."""

from copy import deepcopy
from threading import Lock
import unittest

from epub_tts_narrator.attribution_verification import (
    chapter_block_ids, findings, restrict_findings, risk_blocks, sample_blocks,
)
from test_contracts import valid_document
from test_series import fixture as series_fixture


def add_tom(document):
    character = deepcopy(document["cast"]["characters"][0])
    character.update(id="c2", name="Tom", voice_id="v2", description="Another speaker.")
    voice = deepcopy(document["cast"]["voices"][1])
    voice.update(id="v2", description="Another distinct voice.", traits=[])
    role = deepcopy(document["cast"]["roles"][1])
    role.update(id="tom", character_id="c2", voice_id="v2", description="Tom speaking.")
    document["cast"]["characters"].append(character)
    document["cast"]["voices"].append(voice)
    document["cast"]["roles"].append(role)
    return document


def risk_document(risk=True):
    document = add_tom(series_fixture())
    for row in document["segments"]:
        row["id"] = f"{row['block_id']}:{row['start']}:{row['end']}"
        row["attribution"]["basis"] = "explicit"
    document["segments"][2].update(kind="dialogue", role_id="mara", voice_id="v1")
    if risk:
        for row in document["segments"][1:]:
            row["attribution"]["basis"] = "context"
    return document


def evidence(block_id):
    quote = "Mara said softly" if block_id == "b1" else "They left."
    return [{"block_id": block_id, "quote": quote}]


def correction(unit):
    block_id = unit["block_id"]
    def piece(text, role_id, kind, citations):
        return {"text": text, "kind": kind, "role_id": role_id, "basis": "explicit",
                "reason": "The source evidence supports this speaker.", "evidence": citations,
                "candidates": [], "review_required": False,
                "delivery": {"direction": "", "evidence": []}}
    if block_id == "b1":
        before, after = unit["text"].split("“", 1)
        segments = [piece(before, "narrator_main", "narration", []),
                    piece("“" + after, "tom", "dialogue", evidence(block_id))]
    else:
        segments = [piece(unit["text"], "tom", "dialogue", evidence(block_id))]
    return {"block_id": block_id, "reason": "The context speaker was misattributed.",
            "evidence": evidence(block_id), "segments": segments}


class AuditClient:
    identity = {"model": "test", "base_url": "https://example.invalid"}

    def __init__(self, model="test-model", base_url="https://example.invalid", **kwargs):
        self.model, self.base_url = model, base_url
        self.calls, self.lock = [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        if payload["mode"] == "attribution_review_scan":
            return {"corrections": [correction(unit) for unit in payload["target_units"]]}
        if payload["mode"] == "attribution_review_vote":
            return {"votes": [{"block_id": row["block_id"], "verdict": "approve",
                               "reason": "The exact source evidence supports the correction.",
                               "evidence": evidence(row["block_id"])}
                              for row in payload["proposals"]]}
        raise AssertionError(f"Unexpected request mode: {payload['mode']}")


class AttributionVerificationTests(unittest.TestCase):
    def test_risk_blocks_include_context_dialogue_and_flagged_lines_but_exclude_explicit_only_blocks(self):
        document = valid_document()
        self.assertEqual(risk_blocks(document), [])
        document["segments"][1]["attribution"]["basis"] = "context"
        self.assertEqual(risk_blocks(document), ["b1"])
        document["segments"][2]["attribution"]["review_required"] = True
        self.assertEqual(risk_blocks(document), ["b1", "b2"])
        document["segments"][2]["attribution"].update(basis="unresolved", review_required=False)
        self.assertEqual(risk_blocks(document), ["b1", "b2"])

    def test_risk_blocks_include_context_simultaneous_rows_and_require_two_neighborhood_speakers(self):
        document = valid_document()
        document["segments"][1]["attribution"]["basis"] = "context"
        for row in document["segments"]:
            row["role_id"] = "mara"
        self.assertEqual(risk_blocks(document), [])
        document = valid_document()
        document["segments"][1]["attribution"]["basis"] = "context"
        document["segments"][1].update(role_id=None, simultaneous_speakers=[
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "narrator", "voice_id": "vn"}])
        self.assertEqual(risk_blocks(document), ["b1"])

    def test_chapter_mapping_preserves_source_order_and_lists_unknown_ids(self):
        document = valid_document()
        document["source"]["chapters"][0]["block_ids"] = ["b1"]
        document["source"]["chapters"].append({"id": "ch2", "title": "Two", "source_href": "two.xhtml", "block_ids": ["b2"]})
        self.assertEqual(chapter_block_ids(document, ["ch2", "ch1"]), {"ch1": ["b1"], "ch2": ["b2"]})
        with self.assertRaisesRegex(ValueError, "missing.*unknown"):
            chapter_block_ids(document, ["missing", "unknown"])

    def test_sampling_is_deterministic_hash_selected_and_input_ordered(self):
        self.assertEqual(sample_blocks(["b1", "b2", "b3", "b4"], 2), ["b2", "b4"])
        self.assertEqual(sample_blocks(["b4", "b1", "b3", "b2"], 2), ["b4", "b2"])
        self.assertEqual(sample_blocks(["b1"], 3), ["b1"])
        with self.assertRaisesRegex(ValueError, "at least 1"):
            sample_blocks(["b1"], 0)

    def test_findings_report_display_labels_evidence_and_review_only_changes(self):
        original = valid_document()
        audited = deepcopy(original)
        audited["segments"][1].update(role_id="narrator", voice_id="vn")
        audited["segments"][1]["attribution"].update(basis="context", evidence=[{"block_id": "b1", "quote": "softly"}])
        audited["segments"][2]["attribution"]["review_required"] = True
        result = findings(original, audited)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], {"segment_id": "s2", "chapter_id": "ch1", "block_id": "b1",
            "text": "“Hello.”", "from_label": "Mara", "to_label": "Narrator", "basis": "context",
            "evidence": [{"block_id": "b1", "quote": "softly"}], "review_only": False})
        self.assertTrue(result[1]["review_only"])
        self.assertEqual(result[1]["from_label"], result[1]["to_label"])

    def test_findings_display_unassigned_and_comma_joined_simultaneous_labels(self):
        original = add_tom(valid_document())
        audited = deepcopy(original)
        audited["segments"][1].update(role_id=None, voice_id=None)
        audited["segments"][1]["attribution"].update(basis="unresolved", review_required=True)
        audited["segments"][2].update(kind="dialogue", role_id=None, voice_id=None, simultaneous_speakers=[
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "tom", "voice_id": "v2"}])
        result = findings(original, audited)
        self.assertEqual(result[0]["to_label"], "unassigned")
        self.assertEqual(result[1]["to_label"], "Mara, Tom")

    def test_restrict_findings_replaces_only_chosen_segments_and_rejects_unpaired_ids(self):
        original = valid_document()
        audited = deepcopy(original)
        audited["segments"][1]["role_id"] = "narrator"
        audited["segments"][2]["role_id"] = "mara"
        revised, applied = restrict_findings(original, audited, ["s2"])
        self.assertEqual(applied, ["s2"])
        self.assertEqual(revised["segments"][1], audited["segments"][1])
        self.assertEqual(revised["segments"][2], original["segments"][2])
        self.assertEqual(original["segments"][1]["role_id"], "mara")
        with self.assertRaisesRegex(ValueError, "missing"):
            restrict_findings(original, audited, ["missing"])


if __name__ == "__main__":
    unittest.main()
