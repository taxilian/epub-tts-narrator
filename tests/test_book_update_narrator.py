"""Narrator-read source edits preserve a published continuation chain."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import main
from shared_fixtures import make_serial_epub
from test_narrator_read import NarratorClient


class SourceEvidenceNarratorClient(NarratorClient):
    def complete(self, system, payload):
        result = super().complete(system, payload)
        if payload.get("mode") == "narrator_series_check":
            result["r"] = [[identifier, answer, block, "Jon said"] if answer == "y"
                           else [identifier, answer] for identifier, answer, block, _ in result["r"]]
        return result


class NarratorSourceEditTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "epub-tts-narrator.toml").write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nrepair_attempts = 0\nattribution_workers = 1\n', encoding="utf-8")
        self.clients = []
        self.ancestor = self.root / "zero"
        self.parent = self.root / "one"
        self.child = self.root / "two"

    def client(self, *args, **kwargs):
        client = SourceEvidenceNarratorClient(*args, **kwargs)
        self.clients.append(client)
        return client

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

    def epub(self, name, title, identifier, chapters):
        return make_serial_epub(self.root / f"{name}.epub", title, identifier, chapters)

    def seed_chain(self):
        ancestor = self.epub("zero-v1", "Zero", "book-zero", [("Zero", ['"Earlier," Jon said.'])])
        first = self.epub("one-v1", "One", "book-one", [("One", ['"You nevver listen," Jon said.'])])
        second = self.epub("two-v1", "Two", "book-two", [
            ("Two", ['Jon returned to the lighthouse. "Back again," Jon said.'])])
        status, _, stderr = self.run_cli("analyze", ancestor, "-o", self.ancestor, "--strategy", "narrator")
        self.assertEqual(status, 0, stderr)
        status, _, stderr = self.run_cli(
            "analyze", first, "-o", self.parent, "--strategy", "narrator", "--continue-from", self.ancestor)
        self.assertEqual(status, 0, stderr)
        status, _, stderr = self.run_cli(
            "analyze", second, "-o", self.child, "--strategy", "narrator", "--continue-from", self.parent)
        self.assertEqual(status, 0, stderr)
        original = json.loads((self.parent / "analysis.json").read_text(encoding="utf-8"))
        cited = next(block for block in original["source"]["blocks"] if "nevver" in block["text"])
        self.assertTrue(any(citation["block_id"] == cited["id"] for character in original["cast"]["characters"]
                            for citation in character.get("series_identity", {}).get("evidence", [])))
        self.child_analysis = (self.child / "analysis.json").read_bytes()
        self.child_scripts = {path.name: path.read_bytes() for path in (self.child / "scripts").glob("*.jsonl")}
        fixed = self.epub("one-v2", "One", "book-one-fixed", [("One", ['"You never listen," Jon said.'])])
        return original, fixed

    def assert_preserved_update(self, original):
        from epub_tts_narrator.analysis_directory import _completed_directory, continuation_directories

        updated, _, _ = _completed_directory(self.parent)
        self.assertEqual(updated["source"]["book"], original["source"]["book"])
        self.assertEqual(updated["cast"], original["cast"])
        self.assertEqual(updated["provenance"]["series"], original["provenance"]["series"])
        self.assertTrue(any(block["text"] == '"You never listen," Jon said.'
                            for block in updated["source"]["blocks"]))
        self.assertEqual(updated["provenance"]["book_updates"][-1]["strategy"], "narrator")
        self.assertEqual((self.child / "analysis.json").read_bytes(), self.child_analysis)
        self.assertEqual({path.name: path.read_bytes() for path in (self.child / "scripts").glob("*.jsonl")},
                         self.child_scripts)
        _completed_directory(self.child)
        self.assertEqual(continuation_directories(self.child), [self.parent.resolve(), self.ancestor.resolve()])

    def test_narrator_typo_update_preserves_cast_and_rebinds_continuation(self):
        original, fixed = self.seed_chain()

        status, stdout, stderr = self.run_cli("analyze", fixed, "-o", self.parent)

        self.assertEqual(status, 0, stderr + stdout)
        self.assert_preserved_update(original)

    def test_interrupted_narrator_typo_update_resumes_without_model_or_cast_changes(self):
        original, fixed = self.seed_chain()

        from epub_tts_narrator import analysis_revision
        install = analysis_revision._install
        interrupted = False

        def fail_once(source, destination):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise OSError("disk interrupted")
            return install(source, destination)

        with patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_once):
            status, _, stderr = self.run_cli("analyze", fixed, "-o", self.parent)
        self.assertEqual(status, 1, stderr)
        self.assertIn("disk interrupted", stderr)
        calls = sum(len(client.calls) for client in self.clients)

        status, stdout, stderr = self.run_cli("analyze", fixed, "-o", self.parent)

        self.assertEqual(status, 0, stderr + stdout)
        self.assertEqual(sum(len(client.calls) for client in self.clients), calls)
        self.assert_preserved_update(original)


if __name__ == "__main__":
    unittest.main()
