"""Logical chapter partitions never alter the native source or script rows."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.cli import export_bundle, script_partitions
from test_contracts import valid_document


def chapter_document(items):
    document = valid_document()
    template_block = document["source"]["blocks"][0]
    template_segment = document["segments"][0]
    blocks, segments = [], []
    for index, (text, kind, tag) in enumerate(items):
        identifier = f"b{index + 1}"
        blocks.append(dict(template_block, id=identifier, text=text,
                           source={"href": "chapter.xhtml", "element_id": None, "tag": tag}))
        segments.append(dict(deepcopy(template_segment), id=f"s{index + 1}", block_id=identifier,
                             text=text, start=0, end=len(text), kind=kind))
    document["source"]["blocks"] = blocks
    document["source"]["chapters"][0].update(
        title="Unknown", block_ids=[block["id"] for block in blocks])
    document["segments"] = segments
    document["cast"]["characters"][0]["evidence"] = []
    document["cast"]["voices"][1]["traits"] = []
    return document


class ChapterExportTests(unittest.TestCase):
    def test_two_story_chapters_inside_one_resource_get_distinct_files(self):
        document = chapter_document([
            ("CHAPTER FIVE", "heading", "p"),
            ("The first scene ends here.", "narration", "p"),
            ("CHAPTER SIX", "heading", "p"),
            ("The next scene begins.", "narration", "p"),
        ])
        before = deepcopy(document)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "bundle"
            export_bundle(document, target)
            manifest = json.loads((target / "manifest.json").read_text())
            entries = manifest["files"]["scripts"]
            self.assertEqual([entry["chapter_id"] for entry in entries], ["ch1", "ch1"])
            self.assertEqual([entry["title"] for entry in entries], ["CHAPTER FIVE", "CHAPTER SIX"])
            self.assertEqual([entry["heading_block_id"] for entry in entries], ["b1", "b3"])
            self.assertEqual([(entry["sequence_start"], entry["sequence_end"]) for entry in entries],
                             [(0, 2), (2, 4)])
            self.assertEqual(manifest["chapters"], document["source"]["chapters"])
            rows = [json.loads(line) for entry in entries
                    for line in (target / entry["path"]).read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], [0, 1, 2, 3])
            self.assertTrue(all(row["chapter_id"] == "ch1" for row in rows))
            for original, row in zip(document["segments"], rows):
                self.assertEqual(original, {key: row[key] for key in original})
        self.assertEqual(document, before)

    def test_contents_lists_remain_one_section(self):
        for suffix in ("", ":"):
            for extra in ([], [("Afterword", "heading", "p")],
                          [("Table of Contents", "narration", "p")]):
                with self.subTest(suffix=suffix, extra=extra):
                    document = chapter_document([
                        (f"Chapter {index}{suffix}", "heading", "p") for index in range(1, 20)
                    ] + extra)
                    partitions = script_partitions(document)
                    self.assertEqual(len(partitions), 1)
                    self.assertEqual(partitions[0]["title"], "Contents")
                    self.assertIsNone(partitions[0]["heading_block_id"])
                    self.assertEqual(partitions[0]["block_ids"],
                                     document["source"]["chapters"][0]["block_ids"])

    def test_adjacent_labels_do_not_create_empty_story_chapters(self):
        document = chapter_document([
            ("Chapter 1", "heading", "p"), ("Chapter 2", "heading", "p"),
            ("There is only one body passage.", "narration", "p"),
        ])
        self.assertEqual(len(script_partitions(document)), 1)

    def test_numbered_prose_and_partial_chapter_phrases_do_not_split(self):
        document = chapter_document([
            ("Chapter 1", "heading", "p"),
            ("A book was open beside her.", "narration", "p"),
            ("Chapter 2", "dialogue", "p"),
            ("Chapter two is where the story begins.", "heading", "p"),
            ("“CHAPTER THREE”", "dialogue", "p"),
        ])
        self.assertEqual(len(script_partitions(document)), 1)
        self.assertEqual(script_partitions(document)[0]["title"], "Chapter 1")

    def test_structural_headings_support_roman_and_spelled_chapters(self):
        for labels in (("Chapter IV", "Chapter V"), ("CHAPTER TWENTY-ONE", "CHAPTER TWENTY TWO"),
                       ("Chapter 1:", "Chapter 2:")):
            with self.subTest(labels=labels):
                document = chapter_document([
                    (labels[0], "narration", "h1"), ("Prose one.", "narration", "p"),
                    (labels[1], "narration", "h2"), ("Prose two.", "narration", "p"),
                ])
                self.assertEqual([part["title"] for part in script_partitions(document)], list(labels))

    def test_source_boundaries_and_empty_chapters_are_not_merged(self):
        document = chapter_document([
            ("Chapter 1", "heading", "p"), ("Opening prose.", "narration", "p"),
            ("A continued passage.", "narration", "p"),
        ])
        document["source"]["chapters"][0]["block_ids"] = ["b1", "b2"]
        document["source"]["chapters"].extend([
            {"id": "empty", "title": "Illustration", "source_href": "empty.xhtml", "block_ids": []},
            {"id": "ch2", "title": "Continuation", "source_href": "second.xhtml", "block_ids": ["b3"]},
        ])
        document["source"]["blocks"][2]["chapter_id"] = "ch2"
        partitions = script_partitions(document)
        self.assertEqual([part["chapter_id"] for part in partitions], ["ch1", "empty", "ch2"])
        self.assertEqual([part["block_ids"] for part in partitions], [["b1", "b2"], [], ["b3"]])
        self.assertEqual(partitions[1]["title"], "Illustration")
        self.assertIsNone(partitions[1]["heading_block_id"])

    def test_unknown_title_uses_existing_heading_and_preserves_intro_prose(self):
        document = chapter_document([
            ("An introduction precedes the numbered chapters.", "narration", "p"),
            ("Chapter 1", "heading", "p"), ("The story begins.", "narration", "p"),
        ])
        partitions = script_partitions(document)
        self.assertEqual([part["block_ids"] for part in partitions], [["b1"], ["b2", "b3"]])
        self.assertEqual([part["title"] for part in partitions], ["Unknown", "Chapter 1"])
        document = chapter_document([
            ("Afterword", "heading", "p"), ("Thank you for reading.", "narration", "p"),
        ])
        self.assertEqual(script_partitions(document)[0]["title"], "Afterword")


if __name__ == "__main__":
    unittest.main()
