"""Continuation directory tests link series books without modifying earlier outputs."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator import analysis_directory
from epub_tts_narrator.analysis_directory import (
    continuation_directories, continuation_registry, fresh_registry, publish_directory,
)
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.series import link_analysis
from test_series import fixture, propose


class ContinuationDirectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def read(self, path):
        return json.loads(path.read_text())

    def book(self, number, previous=None, *, output=None):
        output = output or self.root / "series" / f"book-{number:02}"
        document = fixture(sequel=previous is not None)
        document["source"]["book"].update(id=f"book{number}", title=f"Book {number}")
        if previous is None:
            registry = fresh_registry(output)
        else:
            registry = continuation_registry(previous, document["source"])
            propose(document, registry["books"][0]["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, output, export_bundle, continue_from=previous)
        return output

    def chain(self):
        first = self.book(1)
        second = self.book(2, first)
        third = self.book(3, second)
        return first, second, third

    def rewrite_link(self, child, parent):
        parent_manifest = self.read(parent / "manifest.json")
        parent_registry = self.read(parent / "series-cast.json")
        manifest = self.read(child / "manifest.json")
        checksums = parent_manifest["analysis_directory"]["sha256"]
        manifest["continuation"] = {
            "version": 1, "path": os.path.relpath(parent, child),
            "series_id": parent_registry["series_id"],
            "source_fingerprint": parent_manifest["source_fingerprint"],
            "analysis_sha256": checksums["analysis.json"],
            "series_cast_sha256": checksums["series-cast.json"],
        }
        write_json(child / "manifest.json", manifest, overwrite=True)

    def test_published_relative_links_and_nearest_first_chain(self):
        first, second, third = self.chain()
        self.assertIsNone(self.read(first / "manifest.json")["continuation"])
        self.assertEqual(self.read(second / "manifest.json")["continuation"]["path"], "../book-01")
        self.assertEqual(self.read(third / "manifest.json")["continuation"]["path"], "../book-02")
        self.assertEqual(continuation_directories(first), [])
        self.assertEqual(continuation_directories(second), [first])
        self.assertEqual(continuation_directories(third), [second, first])

    def test_parent_link_is_relative_to_final_directory_with_nested_output(self):
        first = self.book(1)
        second = self.book(2, first, output=self.root / "different" / "nested" / "second")
        self.assertEqual(continuation_directories(second), [first])
        self.assertEqual((second / self.read(second / "manifest.json")["continuation"]["path"]).resolve(), first)

    def test_series_can_move_as_a_unit_without_rewriting_links(self):
        first, second, third = self.chain()
        moved = self.root / "moved-series"
        shutil.move(first.parent, moved)
        self.assertEqual(continuation_directories(moved / third.name),
                         [moved / second.name, moved / first.name])

    def test_voice_assets_can_be_created_after_every_analysis(self):
        first, second, third = self.chain()
        before = (first / "manifest.json").read_bytes()
        for book in (first, second):
            voices = book / "voices"
            voices.mkdir()
            (voices / "reference.wav").write_bytes(b"later-created audio")
        self.assertEqual(continuation_directories(third), [second, first])
        self.assertEqual((first / "manifest.json").read_bytes(), before)

    def test_existing_output_without_link_has_no_ancestors(self):
        first = self.book(1)
        manifest = self.read(first / "manifest.json")
        del manifest["continuation"]
        write_json(first / "manifest.json", manifest, overwrite=True)
        self.assertEqual(continuation_directories(first), [])
        # Older renderer-only bundles need not acquire analysis.json or a
        # registry merely to establish that no ancestry was recorded.
        legacy = self.root / "legacy-bundle"
        export_bundle(fixture(), legacy, allow_review=True)
        self.assertEqual(continuation_directories(legacy), [])

    def test_missing_grandparent_blocks_render_lookup_and_continued_analysis(self):
        first, _, third = self.chain()
        first.rename(self.root / "removed-first")
        next_source = fixture()["source"]
        next_source["book"]["id"] = "book4"
        with self.assertRaisesRegex(ValueError, "Cannot load continuation parent.*book-01"):
            continuation_directories(third)
        with self.assertRaisesRegex(ValueError, "Cannot load continuation parent.*book-01"):
            continuation_registry(third, next_source)

    def test_replaced_parent_directory_is_rejected_by_recorded_identity(self):
        first, second, _ = self.chain()
        shutil.rmtree(first)
        self.book(99, output=first)
        with self.assertRaisesRegex(ValueError, "parent identity does not match"):
            continuation_directories(second)

    def test_each_recorded_identity_field_is_checked(self):
        _, second, _ = self.chain()
        original = self.read(second / "manifest.json")
        for field in ("series_id", "source_fingerprint", "analysis_sha256", "series_cast_sha256"):
            manifest = deepcopy(original)
            manifest["continuation"][field] = "0" * 64
            write_json(second / "manifest.json", manifest, overwrite=True)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "parent identity does not match"):
                continuation_directories(second)

    def test_wrong_ancestor_cannot_be_substituted_even_with_valid_identity(self):
        first, _, third = self.chain()
        self.rewrite_link(third, first)
        with self.assertRaisesRegex(ValueError, "parent history does not match"):
            continuation_directories(third)

    def test_cycle_is_rejected_instead_of_recursing(self):
        first, _, third = self.chain()
        self.rewrite_link(first, third)
        with self.assertRaisesRegex(ValueError, "Cycle in analysis directory continuation chain"):
            continuation_directories(third)

    def test_malformed_links_are_rejected(self):
        first = self.book(1)
        self.rewrite_link(first, first)
        original = self.read(first / "manifest.json")
        invalid = ["../other", {}, {**original["continuation"], "path": "/absolute"},
                   {**original["continuation"], "path": ""},
                   {**original["continuation"], "path": "nul\x00path"},
                   {**original["continuation"], "version": 2},
                   {**original["continuation"], "analysis_sha256": []}]
        for value in invalid:
            manifest = deepcopy(original)
            manifest["continuation"] = value
            write_json(first / "manifest.json", manifest, overwrite=True)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid analysis directory continuation"):
                continuation_directories(first)

    def test_each_book_is_validated_once_per_chain_traversal(self):
        first, second, third = self.chain()
        original = analysis_directory._completed_directory
        with patch.object(analysis_directory, "_completed_directory", wraps=original) as completed:
            self.assertEqual(continuation_directories(third), [second, first])
        self.assertEqual([call.args[0] for call in completed.call_args_list], [third, second, first])

    def test_publication_rejects_registry_that_differs_from_parent(self):
        first = self.book(1)
        source = fixture(sequel=True)["source"]
        registry = continuation_registry(first, source)
        document = link_analysis(fixture(sequel=True), registry)
        registry["books"][0]["book"]["title"] = "Changed history"
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(ValueError, "parent changed during analysis"):
            publish_directory(document, registry, output, export_bundle, continue_from=first)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".audiobook-analysis-*")))


if __name__ == "__main__":
    unittest.main()
