"""Known joint speakers preserve distinct identities and one source-text span."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_audiobook.attribution_review import (
    _accept_scan, _assignment_signature, _batches, _compact_segments, audit_attribution,
)
from epub_audiobook.contracts import has_assigned_speakers, readiness, review_items, validate_document
from epub_audiobook.pipeline import align_units, apply_edits, fingerprint, segment
from epub_audiobook.series import link_analysis, load_registry, save_registry
from test_contracts import valid_document


def simultaneous_fixture():
    document = valid_document()
    cast, source = document["cast"], document["source"]
    cast["roles"][0]["id"] = "narrator_main"
    cast["voices"][1]["traits"] = []
    cast["characters"].append({"id": "c2", "name": "Lena", "aliases": [],
        "description": "Lena speaks with Mara.", "evidence": [{"block_id": "b1", "quote": "Lena"}], "voice_id": "v2"})
    cast["roles"].append({"id": "lena", "kind": "dialogue", "character_id": "c2", "voice_id": "v2", "description": "Lena speaking."})
    cast["voices"].append({"id": "v2", "description": "A bright voice.", "traits": []})
    source["blocks"][0]["text"] = '“Run!” Mara and Lena said together.'
    spoken = '“Run!” '
    narrator = cast["roles"][0]
    document["segments"] = [
        segment(source, "b1", 0, spoken, None, "dialogue", basis="unresolved",
                reason="No group role exists.", candidates=["mara", "lena"], review_required=True),
        segment(source, "b1", len(spoken), source["blocks"][0]["text"][len(spoken):], narrator),
        segment(source, "b2", 0, source["blocks"][1]["text"], narrator),
    ]
    return document


def group_correction(document):
    evidence = [{"block_id": "b1", "quote": "Mara and Lena said together."}]
    pieces = []
    for row in document["segments"][:2]:
        pieces.append({"text": row["text"], "kind": row["kind"], "role_id": row["role_id"],
                       "basis": "explicit", "reason": "The tag names both simultaneous speakers.",
                       "evidence": evidence, "candidates": [], "review_required": False,
                       "delivery": {"direction": "", "evidence": []}})
    pieces[0]["simultaneous_role_ids"] = ["mara", "lena"]
    return {"block_id": "b1", "reason": "Both named characters speak these same words together.",
            "evidence": evidence, "segments": pieces}


def aligned_group(document):
    batch = _batches(document["source"], 60000)[0]
    unit = batch["target_units"][0]
    result = deepcopy(document)
    result["segments"] = align_units(document["source"], document["cast"], [unit],
        {"units": [{"unit_id": unit["unit_id"], "segments": group_correction(document)["segments"]}]}) + result["segments"][2:]
    return result


class GroupRequests:
    def __init__(self, document, *, empty=False, verdict="approve"):
        self.document, self.empty, self.verdict = document, empty, verdict
        self.calls = []

    def run(self, system, payload, validate, *, cancelled=None):
        self.calls.append(deepcopy(payload))
        if payload["mode"] == "attribution_review_scan":
            response = {"corrections": [] if self.empty else [group_correction(self.document)]}
        else:
            response = {"votes": [{"block_id": "b1", "verdict": self.verdict,
                                  "reason": "The exact tag identifies both people speaking together.",
                                  "evidence": [{"block_id": "b1", "quote": "Mara and Lena said together."}]}]}
        return validate(response)


class SimultaneousAttributionTests(unittest.TestCase):
    def test_series_registration_remaps_every_performer_and_reimport_reuses_the_same_voices(self):
        original = aligned_group(simultaneous_fixture())
        before = deepcopy(original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.json"
            registry = load_registry(path)
            linked = link_analysis(original, registry)
            roles = {role["id"]: role for role in linked["cast"]["roles"]}
            group = linked["segments"][0]
            expected = [{"role_id": role_id, "voice_id": roles[role_id]["voice_id"]}
                        for role_id in ("mara", "lena")]
            self.assertEqual(group["simultaneous_speakers"], expected)
            self.assertTrue(all(row["voice_id"].startswith("series_voice_") for row in expected))
            self.assertEqual(len({row["voice_id"] for row in expected}), 2)
            self.assertIsNone(group["role_id"])
            self.assertIsNone(group["voice_id"])
            self.assertEqual(group["text"], original["segments"][0]["text"])
            self.assertEqual(linked["source"], original["source"])
            self.assertEqual(validate_document(linked), [])
            save_registry(path, registry, linked)
            saved = load_registry(path)
            self.assertEqual(link_analysis(original, saved), linked)
            self.assertEqual(link_analysis(linked, saved), linked)
        self.assertEqual(original, before)

    def test_sequel_simultaneous_performers_reuse_each_matched_character_voice(self):
        original = aligned_group(simultaneous_fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.json"
            registry = load_registry(path)
            first = link_analysis(original, registry)
            save_registry(path, registry, first)
            sequel = deepcopy(original)
            sequel["source"]["book"].update(id="book2", title="The Return")
            prior = {character["name"]: character for character in first["cast"]["characters"]}
            for character in sequel["cast"]["characters"]:
                character["series_identity"] = {
                    "character_id": prior[character["name"]]["id"], "basis": "context", "review_required": False,
                    "reason": "The named returning character is identified in this sequel.",
                    "evidence": [{"block_id": "b1", "quote": "Mara and Lena said together."}],
                }
            linked = link_analysis(sequel, load_registry(path))
            self.assertEqual(linked["segments"][0]["simultaneous_speakers"], first["segments"][0]["simultaneous_speakers"])
            self.assertEqual(validate_document(linked), [])

    def test_aligned_group_is_ready_without_merging_characters_or_duplicating_text(self):
        original = simultaneous_fixture()
        self.assertEqual(validate_document(original), [])
        result = aligned_group(original)
        self.assertEqual(result["cast"], original["cast"])
        self.assertEqual(result["source"], original["source"])
        group = result["segments"][0]
        self.assertIsNone(group["role_id"])
        self.assertIsNone(group["voice_id"])
        self.assertEqual(group["simultaneous_speakers"], [
            {"role_id": "mara", "voice_id": "v1"}, {"role_id": "lena", "voice_id": "v2"}])
        self.assertTrue(has_assigned_speakers(group))
        self.assertEqual(review_items(result), [])
        self.assertTrue(readiness(result)["ready"])
        self.assertEqual("".join(s["text"] for s in result["segments"] if s["block_id"] == "b1"),
                         original["source"]["blocks"][0]["text"])

    def test_direct_group_contract_rejects_invalid_performers_and_ambiguous_groups(self):
        changes = [
            lambda s: s.update(simultaneous_speakers=[]),
            lambda s: s.update(simultaneous_speakers=[{"role_id": "mara", "voice_id": "v1"}]),
            lambda s: s["simultaneous_speakers"][1].update(role_id="mara", voice_id="v1"),
            lambda s: s["simultaneous_speakers"][1].update(role_id="narrator_main", voice_id="vn"),
            lambda s: s["simultaneous_speakers"][1].update(voice_id="v1"),
            lambda s: s["simultaneous_speakers"][1].update(role_id="missing"),
            lambda s: s["simultaneous_speakers"][1].update(untrusted=True),
            lambda s: s.update(role_id="mara", voice_id="v1"),
            lambda s: s.update(kind="narration"),
            lambda s: s["attribution"].update(basis="unresolved"),
            lambda s: s["attribution"].update(review_required=True),
            lambda s: s["attribution"].update(candidates=["mara", "lena"]),
            lambda s: s["attribution"].update(evidence=[]),
        ]
        for change in changes:
            result = aligned_group(simultaneous_fixture())
            change(result["segments"][0])
            with self.subTest(change=change):
                self.assertTrue(validate_document(result))
                self.assertFalse(readiness(result)["ready"])

    def test_model_group_input_rejects_unknown_roles_duplicates_and_wrong_types(self):
        doc = simultaneous_fixture()
        unit = _batches(doc["source"], 60000)[0]["target_units"][0]
        for ids in (None, [], ["mara"], ["mara", "mara"], ["mara", "absent"],
                    ["mara", "narrator_main"], ["mara", {}], "mara,lena"):
            pieces = group_correction(doc)["segments"]
            pieces[0]["simultaneous_role_ids"] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                align_units(doc["source"], doc["cast"], [unit], {"units": [{"unit_id": "b1", "segments": pieces}]})

    def test_alias_roles_of_same_character_do_not_create_duplicate_performers(self):
        doc = simultaneous_fixture()
        doc["cast"]["roles"].append(dict(doc["cast"]["roles"][1], id="mara_alias"))
        pieces = group_correction(doc)["segments"]
        pieces[0]["simultaneous_role_ids"] = ["mara", "mara_alias"]
        unit = _batches(doc["source"], 60000)[0]["target_units"][0]
        with self.assertRaisesRegex(ValueError, "distinct character voices"):
            align_units(doc["source"], doc["cast"], [unit], {"units": [{"unit_id": "b1", "segments": pieces}]})

    def test_group_still_requires_exact_words_and_exact_source_evidence(self):
        doc = simultaneous_fixture()
        batch = _batches(doc["source"], 60000)[0]
        originals = {b["id"]: [s for s in doc["segments"] if s["block_id"] == b["id"]]
                     for b in doc["source"]["blocks"]}
        for change in (lambda p: p.update(text="Run away!"),
                       lambda p: p.update(evidence=[{"block_id": "b1", "quote": "They sang as one"}]),
                       lambda p: p.update(unexpected="value")):
            correction = group_correction(doc)
            change(correction["segments"][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                _accept_scan({"corrections": [correction]}, doc, batch, originals)

    def test_compact_review_and_change_detection_keep_group_members(self):
        result = aligned_group(simultaneous_fixture())
        group = result["segments"][0]
        self.assertEqual(_compact_segments([group])[0]["simultaneous_role_ids"], ["mara", "lena"])
        reordered = deepcopy(group)
        reordered["simultaneous_speakers"].reverse()
        self.assertEqual(_assignment_signature([group]), _assignment_signature([reordered]))
        changed = deepcopy(group)
        changed["simultaneous_speakers"][1]["role_id"] = "different"
        self.assertNotEqual(_assignment_signature([group]), _assignment_signature([changed]))

    def test_three_approvals_resolve_group_uncertainty_and_keep_audit_evidence(self):
        doc = simultaneous_fixture()
        before = deepcopy(doc)
        requests = GroupRequests(doc)
        result = audit_attribution(doc, requests)
        self.assertEqual(doc, before)
        self.assertEqual(result["cast"], doc["cast"])
        self.assertEqual(result["source"], doc["source"])
        self.assertFalse(result["segments"][0]["attribution"]["review_required"])
        self.assertIn("Previous uncertainty resolved", result["segments"][0]["attribution"]["reason"])
        review = result["provenance"]["attribution_review"]
        self.assertEqual(review["remaining_unassigned_segments"], 0)
        self.assertEqual(len(review["changes"][0]["votes"]), 3)
        self.assertIsNone(review["changes"][0]["original_segments"][0]["voice_id"])
        self.assertTrue(readiness(result)["ready"])

    def test_rejected_group_proposal_stays_unassigned_and_is_reported(self):
        doc = simultaneous_fixture()
        result = audit_attribution(doc, GroupRequests(doc, verdict="reject"))
        review = result["provenance"]["attribution_review"]
        self.assertEqual(review["rejected_correction_blocks"], 1)
        self.assertEqual(review["remaining_unassigned_segments"], 1)
        self.assertEqual(review["remaining_unassigned_block_ids"], ["b1"])
        self.assertFalse(has_assigned_speakers(result["segments"][0]))

    def test_empty_scan_reports_remaining_unassigned_despite_zero_rejected_corrections(self):
        doc = simultaneous_fixture()
        progress = []
        result = audit_attribution(doc, GroupRequests(doc, empty=True), progress=progress.append)
        review = result["provenance"]["attribution_review"]
        self.assertEqual(review["unresolved_blocks"], 0)
        self.assertEqual(review["remaining_unassigned_segments"], 1)
        self.assertIn("1 spoken segment(s) in 1 block(s) still need a speaker", progress[-1])

    def test_explicit_single_speaker_edit_removes_old_group_and_updates_voice(self):
        doc = aligned_group(simultaneous_fixture())
        edited = apply_edits(doc, {"source_fingerprint": fingerprint(doc["source"]),
            "cast_fingerprint": fingerprint(doc["cast"]),
            "assignments": [{"segment_id": doc["segments"][0]["id"], "role_id": "mara"}]})
        self.assertNotIn("simultaneous_speakers", edited["segments"][0])
        self.assertEqual(edited["segments"][0]["voice_id"], "v1")
        self.assertTrue(readiness(edited)["ready"])


if __name__ == "__main__":
    unittest.main()
