"""Saved books whose speech split contradicts their quotation marks are read again by analyze."""

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator import quote_spans
from epub_tts_narrator.cli import main
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.narration_boundaries import repair_narration_boundaries
from epub_tts_narrator.pipeline import segment
from epub_tts_narrator.quote_spans import double_quote_marks
from epub_tts_narrator.split_check import misread_blocks
from shared_fixtures import make_serial_epub
from test_contracts import valid_document
from test_narrator_read import NarratorClient


def quoted(text):
    return [text[start:end] for start, end, _ in double_quote_marks(text)[0]]


def old_double_quotes(text):
    """How quotation marks were read before mixed marks and split paragraphs were understood."""
    spans, start = [], None
    for index, char in enumerate(text):
        if char == '"':
            if start is None:
                start = index
            else:
                spans.append((start, index + 1, True))
                start = None
    if start is not None:
        spans.append((start, len(text), False))
    return spans


class QuotationMarkTests(unittest.TestCase):
    def test_mixed_curly_and_straight_marks_pair(self):
        self.assertEqual(quoted('“What?" Essia cried. "Go.”'), ['“What?"', '"Go.”'])
        self.assertEqual(quoted('"Hello,” he said.'), ['"Hello,”'])
        self.assertTrue(double_quote_marks('“What?" Essia cried.')[1])

    def test_a_paragraph_opening_inside_speech_starts_with_that_speech(self):
        text = 'Wolves are much more dignified." He paused. "You need not tell her," he added.'
        self.assertEqual(quoted(text), ['Wolves are much more dignified."', '"You need not tell her,"'])
        self.assertEqual(quoted('\'Is he all right?" Durnik asked.'), ['\'Is he all right?"'])

    def test_a_mark_typed_facing_the_wrong_way_still_pairs(self):
        self.assertEqual(quoted('“I ought to-“ He stopped. “Never mind,” he added.'),
                         ['“I ought to-“', '“Never mind,”'])

    def test_ordinary_quotations_and_measurements_are_unchanged(self):
        self.assertEqual(quoted('He called it "the end." Then he left.'), ['"the end."'])
        self.assertEqual(quoted('"Hello," Mara said, "again."'), ['"Hello,"', '"again."'])
        spans, clean = double_quote_marks('He was 6\'2" tall.')
        self.assertFalse(clean)
        self.assertNotEqual(spans[0][0], 0)


def fixture(text, rows):
    """A valid document whose only paragraph is ``text``, split as ``rows`` of (start, end, kind)."""
    document = valid_document()
    document["cast"]["characters"][0]["evidence"] = []
    document["cast"]["voices"][1]["traits"] = []
    document["source"]["blocks"][0]["text"] = text
    roles = {role["kind"]: role for role in document["cast"]["roles"]}
    document["segments"][0:2] = [segment(document["source"], "b1", start, text[start:end], roles[kind], kind=kind)
                                 for start, end, kind in rows]
    return document


class SplitCheckTests(unittest.TestCase):
    def test_mixed_marks_tag_is_repaired_without_a_model(self):
        text = '“What happened?" Mara asked quietly.'
        result = repair_narration_boundaries(fixture(text, [(0, len(text), "dialogue")]))
        pieces = [row for row in result["segments"] if row["block_id"] == "b1"]
        self.assertEqual([(row["kind"], row["text"]) for row in pieces],
                         [("dialogue", '“What happened?"'), ("narration", " Mara asked quietly.")])
        self.assertEqual(validate_document(result), [])
        self.assertEqual(misread_blocks(result), {})

    def test_speech_read_by_the_narrator_is_misread(self):
        text = 'Wolves are dignified." He paused. "Tell nobody," he added.'
        document = fixture(text, [(0, 21, "narration"), (21, 34, "dialogue"), (34, 48, "narration"),
                                  (48, len(text), "dialogue")])
        self.assertEqual(set(misread_blocks(document)), {"b1"})

    def test_scare_quotes_and_a_persons_choices_are_left_alone(self):
        text = 'She found the "sleepy" pills Amon had mentioned.'
        self.assertEqual(misread_blocks(fixture(text, [(0, len(text), "narration")])), {})
        text = '"Hello," Mara said.'
        document = fixture(text, [(0, len(text), "dialogue")])
        self.assertIn("b1", misread_blocks(document))
        document["segments"][0]["attribution"]["basis"] = "user"
        self.assertEqual(misread_blocks(document), {})


class ReReadTests(unittest.TestCase):
    PARAGRAPHS = ['"Jon, the tide is turning," Mara said. "We should go',
                  'before the light fails." Mara looked at the sea. "Come," she said.',
                  'Jon nodded and followed Mara down the path.']

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "epub-tts-narrator.toml").write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nrepair_attempts = 0\nattribution_workers = 1\n', encoding="utf-8")
        self.clients = []
        self.book = self.root / "book"

    def client(self, *args, **kwargs):
        client = NarratorClient(*args, **kwargs)
        self.clients.append(client)
        return client

    def calls(self):
        return sum(len(client.calls) for client in self.clients)

    def run_cli(self, *args):
        previous = Path.cwd()
        stdout, stderr = io.StringIO(), io.StringIO()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                 patch("sys.stdin", io.StringIO()), \
                 patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
                try:
                    status = main([str(arg) for arg in args])
                except SystemExit as error:
                    status = error.code
        finally:
            os.chdir(previous)
        return status, stdout.getvalue(), stderr.getvalue()

    def analysis(self):
        return json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))

    def test_rerunning_analyze_reads_a_misread_paragraph_again_once(self):
        epub = make_serial_epub(self.root / "book.epub", "Tide", "book-tide", [("One", self.PARAGRAPHS)])
        with patch.dict(quote_spans._FINDERS, {"double": old_double_quotes}):
            status, stdout, stderr = self.run_cli("analyze", epub, "-o", self.book, "--strategy", "narrator")
        self.assertIn(status, (0, 2), stderr + stdout)
        before = self.analysis()
        broken = next(block for block in before["source"]["blocks"] if block["text"].startswith("before the light"))
        self.assertIn(broken["id"], misread_blocks(before))
        untouched = {row["id"]: row for row in before["segments"] if row["block_id"] != broken["id"]}

        status, stdout, stderr = self.run_cli("analyze", self.book)

        self.assertIn(status, (0, 2), stderr + stdout)
        self.assertIn("reading them again", stdout)
        after = self.analysis()
        self.assertEqual(misread_blocks(after), {})
        self.assertEqual(after["source"], before["source"])
        rows = sorted((row for row in after["segments"] if row["block_id"] == broken["id"]), key=lambda row: row["start"])
        spoken = {row["text"].strip() for row in rows if row["kind"] == "dialogue"}
        narrated = " ".join(row["text"] for row in rows if row["kind"] == "narration")
        self.assertIn('before the light fails."', spoken)
        self.assertIn("Mara looked at the sea.", narrated)
        self.assertIn("she said.", narrated)
        # Context paragraphs keep their saved segments; only the misread one changed.
        kept = {row["id"]: row for row in after["segments"] if row["id"] in untouched}
        self.assertEqual(kept, untouched)
        self.assertIn(broken["id"], after["provenance"]["split_rereads"]["blocks"])
        self.assertEqual(after["provenance"]["book_updates"][-1]["reread_paragraphs"], {
            broken["id"]: misread_blocks(before)[broken["id"]]})
        self.assertFalse((self.book / ".analysis-update").exists())

        clients = len(self.clients)
        published = (self.book / "manifest.json").read_bytes()
        status, stdout, stderr = self.run_cli("analyze", self.book)
        self.assertIn(status, (0, 2), stderr + stdout)
        self.assertNotIn("reading them again", stdout)
        # Only the fixture's still-missing auditions are retried; nothing is read again.
        self.assertEqual({call.get("mode") for client in self.clients[clients:] for call in client.calls} - {"audition_script"},
                         set())
        self.assertEqual((self.book / "manifest.json").read_bytes(), published)


if __name__ == "__main__":
    unittest.main()
