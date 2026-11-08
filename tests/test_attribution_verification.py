"""Focused attribution verification uses the real audit and publication boundaries."""

import contextlib
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.attribution_verification import (
    chapter_block_ids, findings, restrict_findings, risk_blocks, sample_blocks,
)
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.series import link_analysis
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


class AttributionVerificationCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.book = self.root / "book"

    def make_book(self, risk=True):
        document = risk_document(risk)
        registry = fresh_registry(self.book)
        linked = link_analysis(document, registry)
        publish_directory(linked, registry, self.book, export_bundle)
        return linked

    def invoke(self, *arguments, input_text=""):
        config = deepcopy(DEFAULTS)
        config["provider"].update(model="test-model", base_url="https://example.invalid")
        client, output, errors = AuditClient(), io.StringIO(), io.StringIO()
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client),
              patch("sys.stdin", io.StringIO(input_text)),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["verify-attribution", str(self.book), *map(str, arguments)])
        return code, output.getvalue(), errors.getvalue(), client

    def test_report_mode_prints_findings_without_revising_the_book(self):
        self.make_book()
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, _ = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertIn("from Mara to Tom", output)
        self.assertIn("2 attribution finding(s).", output)
        self.assertIn("instructions=no", output)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())

    def test_apply_mode_publishes_the_whole_audit_with_a_revision_journal(self):
        original = self.make_book()
        code, output, errors, _ = self.invoke("--apply")
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        roles = {row["id"]: row["role_id"] for row in revised["segments"]}
        self.assertEqual(roles[original["segments"][1]["id"]], "tom")
        self.assertEqual(roles[original["segments"][2]["id"]], "tom")
        self.assertIn("attribution_review", revised["provenance"])
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))
        self.assertIn("Original analysis files and revision journal:", output)

    def test_interactive_wins_over_apply_and_publishes_only_accepted_findings(self):
        original = self.make_book()
        code, _, errors, _ = self.invoke("--apply", "--interactive", input_text="y\nn\n")
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        roles = {row["id"]: row["role_id"] for row in revised["segments"]}
        self.assertEqual(roles[original["segments"][1]["id"]], "tom")
        self.assertEqual(roles[original["segments"][2]["id"]], "mara")
        self.assertNotIn("attribution_review", revised["provenance"])
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))

    def test_chapter_scoped_instructions_reach_the_scan_as_review_notes(self):
        self.make_book()
        code, output, errors, client = self.invoke("--chapter", 1, "--instructions", "Check the visitor carefully.")
        self.assertEqual(code, 0, errors)
        scans = [row for row in client.calls if row["mode"] == "attribution_review_scan"]
        self.assertTrue(scans)
        self.assertTrue(all(row["review_notes"] == "Check the visitor carefully." for row in scans))
        self.assertIn("instructions=yes", output)

    def test_sample_mode_audits_the_deterministic_risk_subset(self):
        self.make_book()
        code, _, errors, client = self.invoke("--sample", 1)
        self.assertEqual(code, 0, errors)
        scans = [row for row in client.calls if row["mode"] == "attribution_review_scan"]
        self.assertEqual(len(scans), 1)
        self.assertEqual([row["block_id"] for row in scans[0]["target_units"]], ["b2"])

    def test_empty_risk_set_exits_before_building_model_requests(self):
        self.make_book(risk=False)
        code, output, errors, client = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertIn("No context-basis or flagged dialogue lines to verify.", output)
        self.assertEqual(client.calls, [])
        self.assertFalse((self.book / ".analysis-cache").exists())

    def test_invalid_instruction_lengths_and_worker_counts_return_errors_without_model_calls(self):
        self.make_book()
        for arguments, message in [(("--instructions", ""), "--instructions"),
                                   (("--instructions", "x" * 2001), "--instructions"),
                                   (("--workers", 0), "workers"), (("--workers", 6), "workers")]:
            with self.subTest(arguments=arguments):
                code, _, errors, client = self.invoke(*arguments)
                self.assertEqual(code, 1)
                self.assertIn(message, errors)
                self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
