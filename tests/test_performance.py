"""Bracket performance changes delivery without rewriting or recasting text."""

from copy import deepcopy
import tempfile
from pathlib import Path
import json
import unittest

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.performance import apply_bracket_delivery, bracket_ranges
from epub_tts_narrator.pipeline import apply_edits, fingerprint
from test_contracts import valid_document


def book_with_text(*texts):
    doc = valid_document()
    template = deepcopy(doc["source"]["blocks"][0])
    template_segment = deepcopy(doc["segments"][0])
    doc["source"]["blocks"], doc["segments"] = [], []
    for index, text in enumerate(texts):
        block = dict(template, id=f"b{index+1}", text=text)
        doc["source"]["blocks"].append(block)
        doc["segments"].append(dict(template_segment, id=f"s{index}", block_id=block["id"],
                                    start=0, end=len(text), text=text))
    doc["source"]["chapters"][0]["block_ids"] = [block["id"] for block in doc["source"]["blocks"]]
    for character in doc["cast"]["characters"]:
        character["evidence"] = []
    for voice in doc["cast"]["voices"]:
        voice["traits"] = []
    return doc


class BracketPerformanceTests(unittest.TestCase):
    def test_multiple_nested_and_adjacent_groups_keep_exact_text(self):
        doc = book_with_text("Cast [Fireball]s, [Ice [Storm]][Shield].")
        before = deepcopy(doc)
        result = apply_bracket_delivery(doc)
        self.assertEqual(doc, before)
        self.assertEqual(validate_document(result), [])
        self.assertEqual("".join(s["text"] for s in result["segments"]), doc["source"]["blocks"][0]["text"])
        tagged = [s for s in result["segments"] if "performance" in s]
        self.assertEqual([s["spoken_text"] for s in tagged], ["Fireball", "Ice Storm", "Shield"])
        self.assertEqual(len({s["performance"]["group_id"] for s in tagged}), 3)
        self.assertTrue(all(s["voice_id"] == "vn" for s in result["segments"]))
        self.assertEqual(apply_bracket_delivery(result), result)

    def test_group_crosses_paragraphs_and_existing_voice_boundaries(self):
        doc = book_with_text("A [message", "continues] here.")
        second = doc["segments"][1]
        second.update(role_id="mara", voice_id="v1", kind="dialogue")
        result = apply_bracket_delivery(doc)
        tagged = [s for s in result["segments"] if "performance" in s]
        self.assertEqual(validate_document(result), [])
        self.assertEqual([s["voice_id"] for s in tagged], ["vn", "v1"])
        self.assertEqual([s["performance"]["enter"] for s in tagged], [True, False])
        self.assertEqual([s["performance"]["exit"] for s in tagged], [False, True])
        self.assertEqual(tagged[0]["performance"]["group_id"], tagged[1]["performance"]["group_id"])

    def test_unbalanced_delimiters_do_not_swallow_later_chapters(self):
        doc = book_with_text("An [unfinished message", "A closing ] and [valid] group.")
        doc["source"]["blocks"][1]["chapter_id"] = "ch2"
        doc["source"]["chapters"][0]["block_ids"] = ["b1"]
        doc["source"]["chapters"].append(dict(doc["source"]["chapters"][0], id="ch2", block_ids=["b2"]))
        ranges, warnings = bracket_ranges(doc["source"])
        self.assertEqual(len(warnings), 2)
        self.assertEqual(ranges["b1"], [])
        result = apply_bracket_delivery(doc)
        self.assertEqual(validate_document(result), [])
        self.assertEqual(len(result["issues"]), 2)
        self.assertEqual([s["spoken_text"] for s in result["segments"] if "spoken_text" in s], ["valid"])

    def test_validator_rejects_rewrites_missing_cues_and_wrong_boundaries(self):
        result = apply_bracket_delivery(book_with_text("A [Fireball] spell."))
        tagged = next(s for s in result["segments"] if "performance" in s)
        tagged["spoken_text"] = "Flame"
        self.assertTrue(any("spoken_text" in e for e in validate_document(result)))
        tagged["spoken_text"] = "Fireball"
        tagged["performance"]["enter"] = False
        self.assertTrue(any("boundaries" in e for e in validate_document(result)))
        tagged.pop("performance")
        tagged.pop("spoken_text")
        self.assertTrue(any("required for source bracket" in e for e in validate_document(result)))

    def test_export_preserves_spoken_text_cues_and_source(self):
        result = apply_bracket_delivery(book_with_text("[Fireball]"))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bundle"
            export_bundle(result, target)
            manifest = json.loads((target / "manifest.json").read_text())
            item = json.loads((target / manifest["files"]["scripts"][0]["path"]).read_text())
            self.assertEqual(item["text"], "[Fireball]")
            self.assertEqual(item["spoken_text"], "Fireball")
            self.assertEqual(item["performance"]["kind"], "bracketed")

    def test_nested_group_keeps_same_block_speaker_and_delivery_boundaries(self):
        doc = book_with_text("[A [B] C]")
        template = doc["segments"][0]
        doc["segments"] = []
        for index, (start, end, role, voice) in enumerate([
                (0, 1, "narrator", "vn"), (1, 6, "mara", "v1"),
                (6, 9, "narrator", "vn")]):
            piece = deepcopy(template)
            piece.update(id=f"s{index}", start=start, end=end,
                         text=doc["source"]["blocks"][0]["text"][start:end],
                         role_id=role, voice_id=voice)
            piece["delivery"] = {"direction": "", "evidence": []}
            doc["segments"].append(piece)
        result = apply_bracket_delivery(doc)
        self.assertEqual(validate_document(result), [])
        self.assertEqual([s["spoken_text"] for s in result["segments"]], ["", "A B", " C"])
        self.assertEqual([s["voice_id"] for s in result["segments"]], ["vn", "v1", "vn"])
        for before, after in zip(doc["segments"], result["segments"]):
            self.assertEqual(before["delivery"], after["delivery"])
            self.assertEqual(before["attribution"], after["attribution"])
        self.assertEqual([s["performance"]["enter"] for s in result["segments"]], [True, False, False])
        self.assertEqual([s["performance"]["exit"] for s in result["segments"]], [False, False, True])

    def test_custom_cue_survives_whole_block_correction_and_export(self):
        doc = book_with_text("A [Shield] grows.")
        doc["cast"]["roles"][0]["id"] = "narrator_main"
        doc["segments"][0]["role_id"] = "narrator_main"
        doc = apply_bracket_delivery(doc, "Subtly distinguish this skill name.")
        edits = {"source_fingerprint": fingerprint(doc["source"]),
                 "cast_fingerprint": fingerprint(doc["cast"]),
                 "blocks": [{"block_id": "b1", "segments": [
                     {"text": doc["source"]["blocks"][0]["text"],
                      "role_id": "mara", "kind": "dialogue"}]}]}
        result = apply_edits(doc, edits)
        self.assertEqual(validate_document(result), [])
        self.assertEqual(result["source"], doc["source"])
        self.assertTrue(all(s["voice_id"] == "v1" for s in result["segments"]))
        marked = [s for s in result["segments"] if "performance" in s]
        self.assertEqual(len(marked), 1)
        self.assertEqual(marked[0]["spoken_text"], "Shield")
        self.assertEqual(marked[0]["performance"]["direction"], "Subtly distinguish this skill name.")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bundle"
            export_bundle(result, target)
            manifest = json.loads((target / "manifest.json").read_text())
            exported = [json.loads(line) for chapter in manifest["files"]["scripts"]
                        for line in (target / chapter["path"]).read_text().splitlines()]
            self.assertEqual("".join(s["text"] for s in exported), "A [Shield] grows.")
            self.assertEqual("".join(s.get("spoken_text", s["text"]) for s in exported), "A Shield grows.")
            self.assertTrue(all(s["voice_id"] == "v1" for s in exported))

    def test_resolved_unmatched_warning_stays_resolved_after_corrections(self):
        doc = apply_bracket_delivery(book_with_text("An unmatched [ remains literal."))
        issue_id = doc["issues"][0]["id"]
        edits = {"source_fingerprint": fingerprint(doc["source"]),
                 "cast_fingerprint": fingerprint(doc["cast"]),
                 "resolved_issues": [issue_id]}
        result = apply_edits(doc, edits)
        self.assertTrue(result["issues"][0]["resolved"])
        self.assertEqual(apply_bracket_delivery(result), result)
        self.assertEqual(result["segments"][0]["text"], "An unmatched [ remains literal.")
        self.assertNotIn("spoken_text", result["segments"][0])

    def test_malformed_performance_metadata_is_rejected_without_crashing(self):
        original = apply_bracket_delivery(book_with_text("[Shield]"))
        for field, value in [("kind", []), ("direction", None), ("group_id", {}),
                             ("enter", 1), ("exit", "true"), ("delimiters", [])]:
            with self.subTest(field=field):
                doc = deepcopy(original)
                doc["segments"][0]["performance"][field] = value
                self.assertTrue(validate_document(doc))
        for field, value in [("performance", []), ("spoken_text", [])]:
            with self.subTest(field=field):
                doc = deepcopy(original)
                doc["segments"][0][field] = value
                self.assertTrue(validate_document(doc))

    def test_cues_require_provenance_before_correction_or_export(self):
        doc = apply_bracket_delivery(book_with_text("[Shield]"))
        doc["provenance"].pop("bracket_delivery")
        edits = {"source_fingerprint": fingerprint(doc["source"]),
                 "cast_fingerprint": fingerprint(doc["cast"]),
                 "blocks": [{"block_id": "b1", "segments": [
                     {"text": "[Shield]", "role_id": "mara", "kind": "dialogue"}]}]}
        self.assertTrue(any("bracket_delivery" in e for e in validate_document(doc)))
        with self.assertRaises(ValueError):
            apply_edits(doc, edits)
        self.assertIn("performance", doc["segments"][0])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bundle"
            with self.assertRaises(ValueError):
                export_bundle(doc, target)
            self.assertFalse(target.exists())

    def test_malformed_policy_marker_is_rejected_before_apply(self):
        original = apply_bracket_delivery(book_with_text("[Shield]"))
        invalid = [None, "bogus", {}, {"version": 1}]
        for field, value in [("version", True), ("basis", "guessed"),
                             ("direction", []), ("groups", 0), ("groups", True),
                             ("warnings", [{"reason": "invented"}])]:
            marker = deepcopy(original["provenance"]["bracket_delivery"])
            marker[field] = value
            invalid.append(marker)
        for marker in invalid:
            with self.subTest(marker=marker):
                doc = deepcopy(original)
                doc["provenance"]["bracket_delivery"] = marker
                self.assertTrue(validate_document(doc))
                with self.assertRaises(ValueError):
                    apply_edits(doc, {"source_fingerprint": fingerprint(doc["source"]),
                                      "cast_fingerprint": fingerprint(doc["cast"])})

    def test_conflicting_directions_cannot_change_during_unrelated_edit(self):
        doc = apply_bracket_delivery(book_with_text("[Shield]"))
        doc["segments"][0]["performance"]["direction"] = "A conflicting delivery."
        before = deepcopy(doc)
        self.assertTrue(any("direction" in e for e in validate_document(doc)))
        with self.assertRaises(ValueError):
            apply_edits(doc, {"source_fingerprint": fingerprint(doc["source"]),
                              "cast_fingerprint": fingerprint(doc["cast"])})
        self.assertEqual(doc, before)


if __name__ == "__main__":
    unittest.main()
