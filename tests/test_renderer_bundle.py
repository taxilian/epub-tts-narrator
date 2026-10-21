"""Renderer plans must preserve text, attribution uncertainty and source offsets."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audiobook_renderer.bundle import load_bundle, make_plan, ranges, spoken_mapping, split_text
from audiobook_renderer.config import DEFAULTS
from epub_audiobook.cli import export_bundle
from epub_audiobook.performance import apply_bracket_delivery
from test_chapter_export import chapter_document
from test_contracts import valid_document
from test_performance import book_with_text


class RendererBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0
        self.config = deepcopy(DEFAULTS)
        self.config["voices"]["series_id"] = "test-series"

    def export(self, document=None):
        self.counter += 1
        target = self.root / f"bundle-{self.counter}"
        export_bundle(document or valid_document(), target, allow_review=True)
        return target

    def rewrite_json(self, target, name, value):
        path = target / name
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        if name != "manifest.json":
            manifest = json.loads((target / "manifest.json").read_text())
            manifest["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.rewrite_json(target, "manifest.json", manifest)

    def rewrite_rows(self, target, rows):
        path = target / "scripts/chapter-0001.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["sha256"]["scripts/chapter-0001.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.rewrite_json(target, "manifest.json", manifest)

    def test_native_bundle_roundtrip_and_plan_do_not_change_source_or_segments(self):
        document = valid_document()
        before = deepcopy(document)
        bundle = load_bundle(self.export(document))
        self.assertEqual(bundle["source"], document["source"])
        self.assertEqual(bundle["cast"], document["cast"])
        plan = make_plan(bundle, self.config)
        self.assertEqual(plan["errors"], [])
        self.assertEqual(plan["voice_ids"], ["v1", "vn"])
        self.assertFalse(plan["partial_book"])
        rows = [chunk["segment"] for chapter in plan["chapters"] for chunk in chapter["chunks"]]
        for original, rendered in zip(document["segments"], rows):
            self.assertEqual(original, {key: rendered[key] for key in original})
        self.assertEqual(document, before)

    def test_logical_chapters_in_one_source_resource_keep_order_and_titles(self):
        document = chapter_document([
            ("CHAPTER FIVE", "heading", "p"),
            ("The first scene ends here.", "narration", "p"),
            ("CHAPTER SIX", "heading", "p"),
            ("The next scene begins.", "narration", "p"),
        ])
        bundle = load_bundle(self.export(document))
        plan = make_plan(bundle, self.config)
        self.assertEqual([chapter["title"] for chapter in plan["chapters"]],
                         ["CHAPTER FIVE", "CHAPTER SIX"])
        self.assertEqual([chunk["segment"]["chapter_id"] for chapter in plan["chapters"]
                          for chunk in chapter["chunks"]], ["ch1"] * 4)
        self.config["render"]["chapter_numbers"] = [2]
        selected = make_plan(bundle, self.config)
        self.assertTrue(selected["partial_book"])
        self.assertEqual([chapter["title"] for chapter in selected["chapters"]], ["CHAPTER SIX"])

    def test_empty_source_chapters_are_retained_without_stealing_following_rows(self):
        document = valid_document()
        document["source"]["chapters"].insert(0, {
            "id": "empty", "title": "Illustration", "source_href": "cover.xhtml", "block_ids": [],
        })
        bundle = load_bundle(self.export(document))
        plan = make_plan(bundle, self.config)
        self.assertEqual(plan["chapters"][0]["title"], "Illustration")
        self.assertEqual(plan["chapters"][0]["chunks"], [])
        self.assertEqual(len(plan["chapters"][1]["chunks"]), 3)

    def test_changed_text_is_rejected_even_with_updated_manifest_checksum(self):
        target = self.export()
        rows = [json.loads(line) for line in (target / "scripts/chapter-0001.jsonl").read_text().splitlines()]
        rows[1]["text"] = "Different dialogue."
        self.rewrite_rows(target, rows)
        with self.assertRaises(ValueError):
            load_bundle(target)

    def test_sequence_must_be_an_integer_not_an_equal_boolean(self):
        target = self.export()
        rows = [json.loads(line) for line in (target / "scripts/chapter-0001.jsonl").read_text().splitlines()]
        rows[0]["sequence"] = False
        self.rewrite_rows(target, rows)
        with self.assertRaises(ValueError):
            load_bundle(target)

    def test_manifest_cannot_relabel_or_reorder_native_chapter_metadata(self):
        for key, value in (("title", "Invented title"), ("heading_block_id", "b2"),
                           ("chapter_id", "different"), ("sequence_end", 2),
                           ("path", "../chapter.jsonl")):
            with self.subTest(key=key):
                target = self.export()
                manifest = json.loads((target / "manifest.json").read_text())
                manifest["files"]["scripts"][0][key] = value
                self.rewrite_json(target, "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    load_bundle(target)

    def test_source_break_before_cannot_be_removed_or_invented_in_script(self):
        for source_break, row_break in (("blank_paragraph", None), (None, "blank_paragraph")):
            with self.subTest(source_break=source_break):
                document = valid_document()
                if source_break:
                    document["source"]["blocks"][0]["break_before"] = source_break
                target = self.export(document)
                rows = [json.loads(line) for line in (target / "scripts/chapter-0001.jsonl").read_text().splitlines()]
                if row_break:
                    rows[0]["break_before"] = row_break
                else:
                    rows[0].pop("break_before", None)
                self.rewrite_rows(target, rows)
                with self.assertRaises(ValueError):
                    load_bundle(target)

    def test_flagged_attribution_remains_visible_without_blocking_assigned_speech(self):
        document = valid_document()
        document["segments"][0]["attribution"]["review_required"] = True
        target = self.export(document)
        self.rewrite_json(target, "review.json", [])
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["readiness"] = {"ready": True, "errors": [], "review_count": 0}
        self.rewrite_json(target, "manifest.json", manifest)
        bundle = load_bundle(target)
        plan = make_plan(bundle, self.config)
        self.assertEqual(plan["errors"], [])
        self.assertEqual(plan["review_items"], 1)

    def test_advisory_issues_and_legacy_draft_status_need_no_override(self):
        document = valid_document()
        document["issues"].append({
            "id": "identity-question", "kind": "cast", "target": "c1",
            "reason": "Check possible alias", "resolved": False,
        })
        target = self.export(document)
        manifest = json.loads((target / "manifest.json").read_text())
        manifest["readiness"] = {"ready": False, "errors": [], "review_count": 1}
        self.rewrite_json(target, "manifest.json", manifest)
        bundle = load_bundle(target)
        for allow_review in (False, True):
            with self.subTest(allow_review=allow_review):
                self.config["render"]["allow_review"] = allow_review
                plan = make_plan(bundle, self.config)
                self.assertEqual(plan["errors"], [])
                self.assertEqual(plan["review_items"], 1)

    def test_null_voice_still_blocks_speech_when_review_override_is_enabled(self):
        document = valid_document()
        row = document["segments"][1]
        row.update(role_id=None, voice_id=None)
        row["attribution"].update(basis="unresolved", review_required=True)
        bundle = load_bundle(self.export(document))
        for allow_review in (False, True):
            with self.subTest(allow_review=allow_review):
                self.config["render"]["allow_review"] = allow_review
                errors = make_plan(bundle, self.config)["errors"]
                self.assertTrue(any("Missing voice" in error for error in errors), errors)

    def test_unknown_role_still_blocks_loading_despite_advisory_review_policy(self):
        target = self.export()
        rows = [json.loads(line) for line in (target / "scripts/chapter-0001.jsonl").read_text().splitlines()]
        rows[1]["role_id"] = "not-in-cast"
        self.rewrite_rows(target, rows)
        with self.assertRaisesRegex(ValueError, "unknown reference"):
            load_bundle(target)

    def test_unassigned_speech_blocks_its_chapter_without_blocking_other_chapters(self):
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The first scene has an assigned narrator.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("This later passage needs a speaker.", "narration", "p"),
        ])
        document["segments"][-1].update(role_id=None, voice_id=None)
        document["segments"][-1]["attribution"].update(basis="unresolved", review_required=True)
        bundle = load_bundle(self.export(document))
        self.assertFalse(bundle["manifest"]["readiness"]["ready"])
        self.config["render"]["chapter_numbers"] = [1]
        self.assertEqual(make_plan(bundle, self.config)["errors"], [])
        self.config["render"]["chapter_numbers"] = [2]
        self.assertTrue(any("Missing voice" in message for message in make_plan(bundle, self.config)["errors"]))

    def test_series_provenance_prevents_silent_voice_namespace_switch(self):
        document = valid_document()
        document["provenance"]["series"] = {"series_id": "actual-series"}
        bundle = load_bundle(self.export(document))
        self.config["voices"]["series_id"] = "accidental-new-series"
        try:
            plan = make_plan(bundle, self.config)
        except ValueError:
            return
        self.assertTrue(plan["errors"], "A namespace mismatch must not silently create replacement voices")
        self.config["voices"]["series_id"] = "actual-series"
        self.assertEqual(make_plan(bundle, self.config)["errors"], [])

    def test_nested_bracket_mapping_preserves_unicode_and_canonical_offsets(self):
        document = apply_bracket_delivery(book_with_text("Élise cast [水 [火] élève] twice."))
        bundle = load_bundle(self.export(document))
        self.config["render"]["max_chunk_chars"] = 5
        plan = make_plan(bundle, self.config)
        chunks = plan["chapters"][0]["chunks"]
        self.assertEqual("".join(chunk["spoken_text"] for chunk in chunks), "Élise cast 水 火 élève twice.")
        block = document["source"]["blocks"][0]["text"]
        for chunk in chunks:
            rebuilt = "".join(block[item["start"]:item["end"]] for item in chunk["source_ranges"])
            self.assertEqual(rebuilt, chunk["spoken_text"])
            self.assertLessEqual(len(chunk["spoken_text"]), 5)
        marked = [chunk for chunk in chunks if chunk["segment"].get("performance")]
        self.assertTrue(marked)
        self.assertTrue(all(chunk["segment"]["voice_id"] == "vn" for chunk in marked))

    def test_unrecognized_spoken_text_rewrite_needs_explicit_mapping(self):
        row = {"text": "[Fireball]", "spoken_text": "Flame", "start": 12, "end": 22}
        with self.assertRaises(ValueError):
            spoken_mapping(row)

    def test_split_ranges_reconstruct_exact_text_without_unicode_loss(self):
        samples = (
            "A short line. Another sentence follows.\nA final line.",
            "é水🪐" * 100, "Word " * 100, " " * 75,
            "He said, “Go!”  She stayed. Then they left.",
        )
        for text in samples:
            for maximum in (1, 7, 40):
                with self.subTest(maximum=maximum, text=text[:20]):
                    spans = list(split_text(text, maximum))
                    self.assertEqual("".join(text[a:b] for a, b in spans), text)
                    self.assertEqual(spans[0][0], 0)
                    self.assertEqual(spans[-1][1], len(text))
                    self.assertTrue(all(0 < b - a <= maximum for a, b in spans))
                    self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
        self.assertEqual(ranges([1, 2, 4, 5, 7]), [
            {"start": 1, "end": 3}, {"start": 4, "end": 6}, {"start": 7, "end": 8},
        ])

    def test_meaningful_symbol_only_segments_are_not_silently_discarded(self):
        for text in ("%", "+", "&", "=", "$"):
            with self.subTest(text=text):
                bundle = load_bundle(self.export(book_with_text(text)))
                plan = make_plan(bundle, self.config)
                chunks = plan["chapters"][0]["chunks"]
                self.assertTrue(plan["errors"] or any(not chunk["silent_control"] for chunk in chunks),
                                "Meaningful text must be spoken or explicitly blocked, never dropped")

    def test_checksum_and_symlink_fail_closed_before_reading_book(self):
        target = self.export()
        (target / "cast.json").write_text("{}")
        with self.assertRaises(ValueError):
            load_bundle(target)
        target = self.export()
        real = self.root / "cast-copy.json"
        real.write_bytes((target / "cast.json").read_bytes())
        (target / "cast.json").unlink()
        (target / "cast.json").symlink_to(real)
        with self.assertRaises(ValueError):
            load_bundle(target)


if __name__ == "__main__":
    unittest.main()
