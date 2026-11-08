"""Narrator-read publisher apparatus is dropped wherever it clusters."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.front_matter import drop_front_matter, is_front_matter_text


STORY = ("The road climbed gently toward the lighthouse while Mara listened to gulls above the water "
         "and thought about the letter waiting for her at home.")
LONG_STORY = STORY * 6


def make_source(paragraphs_by_chapter):
    blocks, chapters = [], []
    for number, paragraphs in enumerate(paragraphs_by_chapter, 1):
        chapter_id = f"ch{number:04d}"
        identifiers = []
        for text, tag in paragraphs:
            block_id = f"b{len(blocks) + 1:06d}"
            blocks.append({"id": block_id, "chapter_id": chapter_id, "text": text,
                           "source": {"href": f"{chapter_id}.xhtml", "element_id": None, "tag": tag}})
            identifiers.append(block_id)
        chapters.append({"id": chapter_id, "title": f"Chapter {number}",
                         "source_href": f"{chapter_id}.xhtml", "block_ids": identifiers})
    return {"book": {"id": "front-matter-fixture", "title": "Fixture", "authors": ["A"], "language": "en"},
            "chapters": chapters, "blocks": blocks, "exclusions": [], "warnings": [],
            "extraction_policy": "fixture"}


def block_ids(source):
    return [block["id"] for block in source["blocks"]]


def kept_ids(source):
    return {block["id"] for block in drop_front_matter(source)["blocks"]}


class FrontMatterTextTests(unittest.TestCase):
    def test_isbn_alone_is_strong_with_or_without_separators(self):
        for text in ("ISBN 978-0-345-45745-5", "ISBN-13: 9780345457455",
                     "eISBN 0-345-45745-5", "ISBN 0 345 45745 5"):
            with self.subTest(text=text):
                self.assertTrue(is_front_matter_text(text))

    def test_copyright_signals_alone_are_strong(self):
        for text in ("Copyright © 1982 by Example Author", "© Example Author",
                     "All rights reserved under international conventions."):
            with self.subTest(text=text):
                self.assertTrue(is_front_matter_text(text))

    def test_each_weak_publisher_signal_marks_its_block(self):
        for text in ("Published by The Example Publishing Group", "Printed in the United States",
                     "at delrey@randomhouse.com", "www.randomhouse.com/delrey/",
                     "1745 Broadway, New York, NY 10019", "A Del Rey Book",
                     "THE WHITE DRAGON", "Del Rey and colophon are registered trademarks"):
            with self.subTest(text=text):
                self.assertTrue(is_front_matter_text(text))

    def test_story_text_carries_no_signal(self):
        for text in (STORY, "He printed out fax sheets in the office.", "Her first novel was published in 1967."):
            with self.subTest(text=text):
                self.assertFalse(is_front_matter_text(text))


class FrontMatterClusterTests(unittest.TestCase):
    def test_matching_blocks_are_dropped_wherever_they_appear(self):
        source = make_source([
            [(STORY, "p"), ('"I know," Mara said.', "p"), (STORY, "p")],
            [("Copyright © 1982 by Example Author", "p"),
             ("Published by The Ballantine Publishing Group", "p"),
             ("eISBN 0-345-45745-5", "p"), (STORY, "p")],
        ])
        original = deepcopy(source)

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), ["b000001", "b000002", "b000003", "b000007"])
        self.assertEqual(result["chapters"][1]["block_ids"], ["b000007"])
        self.assertEqual({entry["block_id"] for entry in result["exclusions"]},
                         {"b000004", "b000005", "b000006"})
        self.assertEqual(source, original)

    def test_story_prose_adjacent_to_back_matter_survives(self):
        source = make_source([[
            (STORY, "p"), (STORY, "p"),
            ("End of Book One.", "p"),
            ("Copyright © 2019 by Example Author", "p"),
            ("You can join my facebook page at www.facebook.com/groups/readers", "p"),
        ]])

        self.assertEqual(kept_ids(source), {b["id"] for b in source["blocks"]} - {"b000004", "b000005"})

    def test_a_single_weak_signal_cluster_drops_nothing(self):
        source = make_source([[("Published by The Example Publishing Group", "p"), (STORY, "p")]])

        self.assertEqual(kept_ids(source), {b["id"] for b in source["blocks"]})

    def test_two_weak_signal_types_in_two_blocks_qualify(self):
        source = make_source([[
            ("at delrey@randomhouse.com", "p"),
            ("We post updates at www.example.com", "p"),
            (STORY, "p"),
        ]])

        self.assertEqual(kept_ids(source), {"b000003"})

    def test_dialogue_lines_break_clusters_and_are_never_dropped(self):
        source = make_source([[
            ("at delrey@randomhouse.com", "p"),
            ('"Write to help@example.com," Mara said.', "p"),
            ("We post updates at www.example.com", "p"),
        ]])

        self.assertEqual(kept_ids(source), {b["id"] for b in source["blocks"]})

    def test_a_quoted_word_does_not_shield_neighboring_apparatus(self):
        source = make_source([[
            ("We post updates at www.example.com", "p"),
            ('Subscribe by sending "subscribe" to drin@example.com', "p"),
            ("You can also read www.example.com/news", "p"),
        ]])

        self.assertEqual(kept_ids(source), {"b000002"})

    def test_caps_catalog_titles_drop_only_inside_a_qualifying_cluster(self):
        source = make_source([
            [("THE WHITE DRAGON", "p"), ('"I know," Mara said.', "p"), (STORY, "p")],
            [("A Del Rey Book", "p"), ("THE WHITE DRAGON", "p"), ("Copyright © 1982", "p"), (STORY, "p")],
        ])

        self.assertEqual(kept_ids(source),
                         {b["id"] for b in source["blocks"]} - {"b000004", "b000005", "b000006"})

    def test_long_prose_blocks_break_clusters(self):
        source = make_source([[
            ("at delrey@randomhouse.com", "p"),
            (LONG_STORY, "p"),
            ("We post updates at www.example.com", "p"),
        ]])

        self.assertEqual(kept_ids(source), {b["id"] for b in source["blocks"]})


class FrontMatterClient:
    def __init__(self):
        self.model = "fixture"
        self.base_url = "http://offline.invalid/v1"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        mode = payload.get("mode")
        if mode == "narrator_prep":
            return {"new": [], "voice": [], "nar": ["third", None]}
        if mode == "narrator_voices":
            return {"v": [[row[0], "Androgynous voice with a clear, even tone.", "even pace", "pace=even"]
                          for row in payload["voices"]]}
        if mode == "audition_script":
            return {"text": STORY, "reason": "A representative narration sample."}
        raise AssertionError(f"Unexpected model request: {mode}")


class NarratorFrontMatterCliTests(unittest.TestCase):
    def test_narrator_publication_omits_matching_rows_in_any_chapter(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = make_source([
            [("Copyright © 1982 by Example Author", "p"), ("Chapter One", "h1"), (STORY, "p")],
            [(STORY, "p"), ("Copyright © 1982 by Example Author", "p"), (STORY, "p")],
        ])
        source_path = root / "front-matter-source.json"
        source_path.write_text(json.dumps({"format": "epub-audiobook-source", "version": 1, "source": source}),
                               encoding="utf-8")
        config = root / "epub-tts-narrator.toml"
        config.write_text('[provider]\nmodel="fixture"\nbase_url="http://offline.invalid/v1"\n'
                          '[analysis]\nrepair_attempts=0\nattribution_workers=1\n', encoding="utf-8")
        output = root / "book"
        client = FrontMatterClient()
        stdout, stderr = io.StringIO(), io.StringIO()

        with (contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr),
              patch("sys.stdin", io.StringIO()),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client)):
            status = main(["analyze", str(source_path), "-o", str(output),
                           "--strategy", "narrator", "--config", str(config)])

        self.assertEqual(status, 0, stderr.getvalue())
        document = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_document(document), [])
        self.assertEqual(block_ids(document["source"]), ["b000002", "b000003", "b000004", "b000006"])
        self.assertEqual({row["block_id"] for row in document["segments"]},
                         {"b000002", "b000003", "b000004", "b000006"})
        script_rows = [json.loads(line) for path in (output / "scripts").glob("*.jsonl")
                       for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["block_id"] for row in script_rows},
                         {"b000002", "b000003", "b000004", "b000006"})


if __name__ == "__main__":
    unittest.main()
