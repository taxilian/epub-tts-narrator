"""Narrator-read front matter is dropped only inside a conservative window."""

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

    def test_one_weak_signal_does_not_drop_a_block(self):
        for text in ("Published by The Example Publishing Group", "Printed in the United States",
                     "at delrey@randomhouse.com", "1745 Broadway, New York, NY 10019"):
            with self.subTest(text=text):
                self.assertFalse(is_front_matter_text(text))

    def test_two_distinct_weak_signals_drop_a_block(self):
        for text in ("Published by Example Press\nat delrey@randomhouse.com",
                     "Printed in the United States\n1745 Broadway, New York, NY 10019"):
            with self.subTest(text=text):
                self.assertTrue(is_front_matter_text(text))


class FrontMatterWindowTests(unittest.TestCase):
    def test_matching_copyright_publisher_and_email_pages_before_the_heading_are_dropped(self):
        source = make_source([[
            ("Copyright © 1982 by Example Author", "p"),
            ("Published by Example Press\n1745 Broadway, New York, NY 10019", "p"),
            ("at delrey@randomhouse.com\nPrinted in the United States", "p"),
            ("Chapter One", "h1"),
            (STORY, "p"),
        ]])
        original = deepcopy(source)

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), ["b000004", "b000005"])
        self.assertEqual(result["chapters"][0]["block_ids"], ["b000004", "b000005"])
        self.assertEqual({entry["block_id"] for entry in result["exclusions"]},
                         {"b000001", "b000002", "b000003"})
        self.assertEqual(source, original)

    def test_identical_text_after_the_heading_or_in_a_later_chapter_is_kept(self):
        copyright_page = "Copyright © 1982 by Example Author"
        contact_page = "at delrey@randomhouse.com\nPrinted in the United States"
        source = make_source([
            [(copyright_page, "p"), ("Chapter One", "h1"),
             (copyright_page, "p"), (contact_page, "p")],
            [(copyright_page, "p"), (contact_page, "p"), ("Chapter Two", "h1")],
        ])

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), ["b000002", "b000003", "b000004", "b000005", "b000006", "b000007"])

    def test_isbn_alone_drops_inside_the_window(self):
        source = make_source([[("eISBN 0-345-45745-5", "p"), ("Chapter One", "h1"), (STORY, "p")]])

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), ["b000002", "b000003"])

    def test_one_weak_signal_stays_inside_the_window(self):
        source = make_source([[("Published by The Example Publishing Group", "p"),
                               ("Chapter One", "h1"), (STORY, "p")]])

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), block_ids(source))

    def test_headingless_chapter_drops_only_before_the_first_dialogue(self):
        source = make_source([[
            ("Copyright © 1982 by Example Author", "p"),
            ('"Write to delrey@randomhouse.com," Mara said.', "p"),
            ("Copyright © 1982 by Example Author", "p"),
            ("Published by Example Press\nPrinted in the United States", "p"),
        ]])

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), ["b000002", "b000003", "b000004"])

    def test_headingless_chapter_without_dialogue_drops_nothing(self):
        source = make_source([[("Copyright © 1982 by Example Author", "p"),
                               ("Published by Example Press\nPrinted in the United States", "p"),
                               (STORY, "p")]])

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), block_ids(source))

    def test_an_empty_first_chapter_does_not_move_the_window_to_the_next_chapter(self):
        source = make_source([[], [("Copyright © 1982 by Example Author", "p"), (STORY, "p")]])

        result = drop_front_matter(source)

        self.assertEqual(block_ids(result), block_ids(source))


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
    def test_narrator_publication_omits_only_the_matching_preheading_script_rows(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = make_source([
            [("Copyright © 1982 by Example Author", "p"), ("Chapter One", "h1"), (STORY, "p")],
            [("Copyright © 1982 by Example Author", "p"), (STORY, "p")],
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
        self.assertEqual(block_ids(document["source"]), ["b000002", "b000003", "b000004", "b000005"])
        self.assertEqual({row["block_id"] for row in document["segments"]},
                         {"b000002", "b000003", "b000004", "b000005"})
        script_rows = [json.loads(line) for path in (output / "scripts").glob("*.jsonl")
                       for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["block_id"] for row in script_rows},
                         {"b000002", "b000003", "b000004", "b000005"})


if __name__ == "__main__":
    unittest.main()
