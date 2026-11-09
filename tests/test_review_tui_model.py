"""Pure behavior tests for the published-book review model."""

from copy import deepcopy
from typing import Any
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.review_tui_model import (
    XTERM_RGB, apply_edits, contrast_ok, contrast_ratio, edit_direction,
    edit_review, edit_speaker, metadata, palette, passages, toc,
)
from tests.test_contracts import valid_document


Document = dict[str, Any]


def review_document() -> Document:
    return valid_document()


def add_lena(document: Document) -> None:
    document["cast"]["characters"].append({
        "id": "c2", "name": "Lena", "aliases": [], "description": "Another speaker.",
        "evidence": [], "voice_id": "v2",
    })
    document["cast"]["voices"].append({"id": "v2", "description": "Second voice.", "traits": []})
    document["cast"]["roles"].append({
        "id": "lena", "kind": "dialogue", "voice_id": "v2", "character_id": "c2",
        "description": "Lena speaking.",
    })


def role_context(document: Document) -> tuple[Document, dict[str, str]]:
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    labels = {role_id: role_id.split("_")[0].title() for role_id in roles}
    return roles, labels


def make_unresolved(segment: Document) -> Document:
    segment.update(role_id=None, voice_id=None)
    segment["attribution"].update(basis="unresolved", review_required=True,
                                  reason="No consensus.", candidates=["mara"])
    return segment


class ReviewModelTests(unittest.TestCase):
    def test_toc_counts_each_rendered_chapter_review_and_unassigned_passages(self):
        document = review_document()
        make_unresolved(document["segments"][1])
        document["segments"][2]["attribution"]["review_required"] = True
        document["source"]["chapters"].append({
            "id": "ch2", "title": "Chapter Two", "source_href": "text/ch2.xhtml", "block_ids": ["b3"],
        })
        document["source"]["blocks"].append({
            "id": "b3", "chapter_id": "ch2", "text": "Later.",
            "source": {"href": "text/ch2.xhtml", "element_id": None, "tag": "p"},
        })
        document["segments"].append({
            "id": "s4", "block_id": "b3", "start": 0, "end": 6, "text": "Later.",
            "kind": "narration", "role_id": "narrator", "voice_id": "vn",
            "delivery": {"direction": "", "evidence": []},
            "attribution": {"basis": "explicit", "reason": "Narrative.", "evidence": [],
                            "candidates": [], "review_required": False},
        })

        rows = toc(document)

        self.assertEqual(validate_document(document), [])
        self.assertEqual(rows, [
            {"id": 0, "title": "Chapter One", "passage_count": 3,
             "review_count": 2, "unassigned_count": 1},
            {"id": 1, "title": "Chapter Two", "passage_count": 1,
             "review_count": 0, "unassigned_count": 0},
        ])

    def test_one_source_chapter_splits_into_rendered_chapters_at_numbered_headings(self):
        document = review_document()
        document["source"]["chapters"][0]["title"] = "Volume One"
        chapter = document["source"]["chapters"][0]
        additions = (
            ("b3", "CHAPTER ONE", "h1", "heading"),
            ("b4", "First chapter prose.", "p", "narration"),
            ("b5", "CHAPTER TWO", "h1", "heading"),
            ("b6", "Second chapter prose.", "p", "narration"),
        )
        for block_id, text, tag, kind in additions:
            chapter["block_ids"].append(block_id)
            document["source"]["blocks"].append({
                "id": block_id, "chapter_id": "ch1", "text": text,
                "source": {"href": "text/ch1.xhtml", "element_id": None, "tag": tag},
            })
            document["segments"].append({
                "id": f"s_{block_id}", "block_id": block_id, "start": 0, "end": len(text),
                "text": text, "kind": kind, "role_id": "narrator", "voice_id": "vn",
                "delivery": {"direction": "", "evidence": []},
                "attribution": {"basis": "explicit", "reason": "Narrative.", "evidence": [],
                                "candidates": [], "review_required": False},
            })

        rows = toc(document)
        second_chapter = passages(document, 1)

        self.assertEqual(validate_document(document), [])
        self.assertEqual([(row["id"], row["title"], row["passage_count"]) for row in rows], [
            (0, "Volume One", 3), (1, "CHAPTER ONE", 2), (2, "CHAPTER TWO", 2),
        ])
        self.assertEqual([row["sequence"] for row in second_chapter], [3, 4])
        self.assertEqual([row["text"] for row in second_chapter], ["CHAPTER ONE", "First chapter prose."])

    def test_passage_rows_keep_cast_palette_slots_and_join_simultaneous_labels(self):
        document = review_document()
        add_lena(document)
        segment = document["segments"][1]
        segment.update(role_id=None, voice_id=None, simultaneous_speakers=[
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "lena", "voice_id": "v2"},
        ])
        segment["attribution"].update(basis="user", review_required=False)

        rows = passages(document, 0)

        self.assertEqual([row["sequence"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0]["role_index"], 0)
        self.assertEqual(rows[1]["from_label"], "Mara, Lena")
        self.assertEqual(rows[1]["role_index"], 1)
        self.assertEqual(rows[1]["role_indices"], [1, 2])
        self.assertEqual(rows[1]["basis"], "user")
        self.assertFalse(rows[1]["review_required"])

    def test_passage_rows_mark_unresolved_text_unassigned(self):
        document = review_document()
        make_unresolved(document["segments"][1])

        row = passages(document, 0)[1]

        self.assertEqual(row["from_label"], "Unassigned")
        self.assertIsNone(row["role_index"])
        self.assertTrue(row["review_required"])

    def test_palette_is_deterministic_and_every_choice_reads_on_black_and_white(self):
        first = palette(40)

        self.assertEqual(first, palette(40))
        self.assertEqual(len(first), 40)
        self.assertEqual(len(set(first)), 40)
        self.assertTrue(all(contrast_ok(index) for index in first))
        self.assertEqual(palette(0), [])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            palette(-1)

    def test_xterm_rgb_derivation_and_contrast_reference_pairs(self):
        self.assertEqual(len(XTERM_RGB), 256)
        self.assertEqual(XTERM_RGB[16], (0, 0, 0))
        self.assertEqual(XTERM_RGB[21], (0, 0, 255))
        self.assertEqual(XTERM_RGB[196], (255, 0, 0))
        self.assertEqual(XTERM_RGB[231], (255, 255, 255))
        self.assertEqual(XTERM_RGB[255], (238, 238, 238))
        self.assertAlmostEqual(contrast_ratio((255, 255, 255), (0, 0, 0)), 21.0)
        self.assertLess(contrast_ratio((0, 0, 128), (0, 0, 0)), 3.0)
        self.assertFalse(contrast_ok(15))
        self.assertFalse(contrast_ok(18))

    def test_metadata_exposes_speaker_voice_attribution_delivery_and_controls(self):
        document = review_document()
        segment = document["segments"][1]
        segment["delivery"] = {"direction": "Softly", "evidence": [{"block_id": "b1", "quote": "softly"}]}
        segment["performance_controls"] = {"version": 1, "text_sha256": "fixture", "controls": [
            {"kind": "delivery"}, {"kind": "pause"}, {"kind": "delivery"},
        ]}

        result = metadata(document, "s2")

        self.assertEqual(result["chapter_id"], "ch1")
        self.assertEqual(result["offsets"], {"start": segment["start"], "end": segment["end"]})
        self.assertEqual(result["sequence"], 1)
        self.assertEqual(result["speaker_labels"], ["Mara"])
        self.assertEqual(result["role_ids"], ["mara"])
        self.assertEqual(result["voice_ids"], ["v1"])
        self.assertEqual(result["voice_descriptions"], ["Proposed light, distinct voice."])
        self.assertEqual(result["delivery"], segment["delivery"])
        self.assertEqual(result["attribution"], segment["attribution"])
        self.assertEqual(result["performance_controls"],
                         {"total": 3, "kinds": {"delivery": 2, "pause": 1}, "read_only": True})
        self.assertEqual(result["editable"], ["speaker", "direction", "review_required"])
        with self.assertRaisesRegex(ValueError, "Unknown segment"):
            metadata(document, "missing")

    def test_single_speaker_choice_records_user_basis_and_preserves_evidence(self):
        document = review_document()
        segment = make_unresolved(document["segments"][1])
        segment["attribution"]["evidence"] = [{"block_id": "b1", "quote": "Hello"}]
        roles, labels = role_context(document)
        before = deepcopy(segment)

        result = edit_speaker(segment, ["mara"], roles, labels)

        self.assertEqual(segment, before)
        self.assertEqual((result["role_id"], result["voice_id"]), ("mara", "v1"))
        self.assertNotIn("simultaneous_speakers", result)
        self.assertEqual(result["attribution"], {
            "basis": "user", "reason": "User selected Mara.",
            "evidence": [{"block_id": "b1", "quote": "Hello"}],
            "candidates": [], "review_required": False,
        })

    def test_single_role_choice_adjusts_kind_like_the_existing_speaker_menu(self):
        document = review_document()
        roles, labels = role_context(document)

        narrated = edit_speaker(document["segments"][1], ["narrator"], roles, labels)
        spoken = edit_speaker(document["segments"][0], ["mara"], roles, labels)

        self.assertEqual(narrated["kind"], "narration")
        self.assertEqual(spoken["kind"], "dialogue")

    def test_simultaneous_choice_preserves_selection_order(self):
        document = review_document()
        add_lena(document)
        roles, labels = role_context(document)

        result = edit_speaker(document["segments"][1], ["lena", "mara"], roles, labels)

        self.assertIsNone(result["role_id"])
        self.assertIsNone(result["voice_id"])
        self.assertEqual(result["simultaneous_speakers"], [
            {"role_id": "lena", "voice_id": "v2"}, {"role_id": "mara", "voice_id": "v1"},
        ])
        self.assertEqual(result["attribution"]["reason"], "User selected Lena; Mara.")

    def test_simultaneous_choice_rejects_groups_the_document_contract_cannot_save(self):
        document = review_document()
        add_lena(document)
        roles, labels = role_context(document)
        roles["mara_copy"] = dict(roles["mara"], id="mara_copy")
        labels["mara_copy"] = "Other Mara"
        cases = [
            (["mara", "mara"], "distinct"),
            (["narrator", "mara"], "dialogue roles"),
            (["mara", "mara_copy"], "voices must be distinct"),
        ]
        for role_ids, message in cases:
            with self.subTest(role_ids=role_ids):
                with self.assertRaisesRegex(ValueError, message):
                    edit_speaker(document["segments"][1], role_ids, roles, labels)
        with self.assertRaisesRegex(ValueError, "dialogue, thought, or letter"):
            edit_speaker(document["segments"][0], ["mara", "lena"], roles, labels)

    def test_unresolved_choice_preserves_reason_evidence_and_candidates(self):
        document = review_document()
        add_lena(document)
        segment = document["segments"][1]
        segment["attribution"].update(reason="Existing uncertainty.", candidates=["mara"],
                                      evidence=[{"block_id": "b1", "quote": "Hello"}])
        segment["simultaneous_speakers"] = [{"role_id": "mara", "voice_id": "v1"}]
        roles, labels = role_context(document)

        preserved = edit_speaker(segment, [], roles, labels)
        supplied = edit_speaker(segment, None, roles, labels, candidates=["lena"])

        self.assertIsNone(preserved["role_id"])
        self.assertIsNone(preserved["voice_id"])
        self.assertNotIn("simultaneous_speakers", preserved)
        self.assertEqual(preserved["attribution"]["basis"], "unresolved")
        self.assertTrue(preserved["attribution"]["review_required"])
        self.assertEqual(preserved["attribution"]["reason"], "Existing uncertainty.")
        self.assertEqual(preserved["attribution"]["candidates"], ["mara"])
        self.assertEqual(preserved["attribution"]["evidence"], segment["attribution"]["evidence"])
        self.assertEqual(supplied["attribution"]["candidates"], ["lena"])

    def test_direction_edit_trims_preserves_evidence_and_enforces_the_limit(self):
        document = review_document()
        segment = document["segments"][1]
        segment["delivery"]["evidence"] = [{"block_id": "b1", "quote": "softly"}]
        before = deepcopy(segment)

        result = edit_direction(segment, "  Quietly, then faster.  ")
        maximum = edit_direction(segment, "x" * 240)

        self.assertEqual(segment, before)
        self.assertEqual(result["delivery"]["direction"], "Quietly, then faster.")
        self.assertEqual(result["delivery"]["evidence"], before["delivery"]["evidence"])
        self.assertEqual(len(maximum["delivery"]["direction"]), 240)
        with self.assertRaisesRegex(ValueError, "240 characters"):
            edit_direction(segment, "x" * 241)

    def test_review_edit_toggles_assigned_segments_and_guards_unresolved_segments(self):
        document = review_document()
        flagged = edit_review(document["segments"][1], True)
        cleared = edit_review(flagged, False)
        unresolved = make_unresolved(document["segments"][1])

        self.assertTrue(flagged["attribution"]["review_required"])
        self.assertFalse(cleared["attribution"]["review_required"])
        with self.assertRaisesRegex(ValueError, "Unresolved attribution"):
            edit_review(unresolved, False)

    def test_apply_edits_replaces_segments_without_mutating_the_document(self):
        document = review_document()
        replacement = edit_review(document["segments"][1], True)
        before = deepcopy(document)

        result = apply_edits(document, {"s2": replacement})

        self.assertEqual(document, before)
        self.assertEqual(result["segments"][1], replacement)
        self.assertEqual(result["segments"][0], before["segments"][0])
        with self.assertRaisesRegex(ValueError, "Unknown segment"):
            apply_edits(document, {"missing": replacement})
        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_edits(document, {"s2": dict(replacement, id="other")})


if __name__ == "__main__":
    unittest.main()
