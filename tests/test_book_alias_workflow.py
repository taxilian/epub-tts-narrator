"""Book-owned voice mappings stay live along a series continuation chain."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest

from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main as renderer_main
from epub_tts_narrator.analysis_directory import (
    continuation_registry, fresh_registry, publish_directory,
)
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.series import link_analysis
from test_book_voice_workflow import BookFixture
from test_renderer_voice_design_integration import FFMPEG
from test_series import fixture, propose


class BookAliasWorkflowTests(BookFixture):
    def publish(self, number, parent=None):
        directory = self.root / f"book-{number:02}"
        document = fixture(sequel=parent is not None)
        document["source"]["book"].update(id=f"book-{number}", title=f"Journey {number}")
        alternate = deepcopy(document["cast"]["voices"][0])
        alternate["id"] = "alternate-voice"
        document["cast"]["voices"].append(alternate)
        role = deepcopy(document["cast"]["roles"][0])
        role.update(id="alternate-narrator", voice_id="alternate-voice",
                    description="The alternate narrator.")
        document["cast"]["roles"].append(role)
        if parent is None:
            registry = fresh_registry(directory)
        else:
            registry = continuation_registry(parent["directory"], document["source"])
            propose(document, parent["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        publish_directory(document, registry, directory, export_bundle,
                          continue_from=parent["directory"] if parent else None)
        return load_bundle(directory)

    @staticmethod
    def ids(bundle):
        roles = bundle["cast"]["roles"]
        narrator = next(role["voice_id"] for role in roles
                        if role["description"] == "The book's narrator.")
        alternate = next(role["voice_id"] for role in roles
                         if role["description"] == "The alternate narrator.")
        return narrator, bundle["cast"]["characters"][0]["voice_id"], alternate

    def aliases(self, bundle, aliases):
        path = Path(bundle["directory"]) / "voice-aliases.json"
        path.write_text(json.dumps({
            "format": "audiobook-book-voice-aliases", "version": 1,
            "series_id": bundle["manifest"]["provenance"]["series"]["series_id"],
            "book_id": bundle["source"]["book"]["id"], "aliases": aliases,
        }), encoding="utf-8")
        return path

    def test_late_parent_mapping_survives_child_persistence_and_local_override(self):
        first, second, third, sibling = self.lineage()
        narrator, character, alternate = self.ids(first)
        stores = {book["source"]["book"]["id"]: self.store(book)
                  for book in (first, second, third, sibling)}
        child = stores["book-2"]
        child.persist_aliases()
        self.assertEqual(stores["book-3"].canonical_id(narrator), narrator)

        # A mapping may be added after every analysis and renderer store exists.
        self.aliases(first, {narrator: character})
        self.assertEqual(stores["book-3"].canonical_id(narrator), character)
        self.assertEqual(stores["book-4"].canonical_id(narrator), character)
        first_before = self.snapshot(Path(first["directory"]))
        sibling_before = self.snapshot(Path(sibling["directory"]))

        self.aliases(second, {narrator: alternate})
        child.persist_aliases()
        self.assertEqual(stores["book-2"].canonical_id(narrator), alternate)
        self.assertEqual(stores["book-3"].canonical_id(narrator), alternate)
        self.assertEqual(stores["book-1"].canonical_id(narrator), character)
        self.assertEqual(stores["book-4"].canonical_id(narrator), character)
        self.assertEqual(self.snapshot(Path(first["directory"])), first_before)
        self.assertEqual(self.snapshot(Path(sibling["directory"])), sibling_before)

        # Persisting a render store must not turn the manual override into a
        # second, hidden source that prevents the user's removal from working.
        self.aliases(second, {})
        self.assertEqual(stores["book-3"].canonical_id(narrator), character)
        self.aliases(first, {narrator: alternate})
        self.assertEqual(stores["book-3"].canonical_id(narrator), alternate)
        self.assertEqual(stores["book-4"].canonical_id(narrator), alternate)

    @unittest.skipUnless(FFMPEG, "FFmpeg is required for fixture voice normalization")
    def test_late_mapping_routes_saved_audio_and_child_override_is_isolated(self):
        first, second, third, sibling = self.lineage()
        narrator, character, alternate = self.ids(first)
        third_store = self.store(third)
        with contextlib.closing(self.session(first)) as session:
            session.generate_missing()
            character_asset = session.store.load(character)
            alternate_asset = session.store.load(alternate)
        self.aliases(first, {narrator: character})
        self.assertEqual(third_store.load(narrator)["asset_hash"], character_asset["asset_hash"])
        self.aliases(second, {narrator: alternate})
        self.assertEqual(third_store.load(narrator)["asset_hash"], alternate_asset["asset_hash"])
        self.assertEqual(self.store(sibling).load(narrator)["asset_hash"], character_asset["asset_hash"])
        self.assertFalse((Path(third["directory"]) / "voices").exists())

    def test_unknown_identity_and_ancestry_cycle_are_rejected(self):
        first, second, third, _ = self.lineage()
        narrator, character, _ = self.ids(first)
        self.aliases(first, {narrator: "unknown-voice"})
        with self.assertRaises(ValueError):
            self.store(third)
        self.aliases(first, {narrator: character})
        self.aliases(second, {character: narrator})
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.store(third)

    def test_file_is_bound_to_its_own_book_and_protected_from_output_overrides(self):
        first, second, _, _ = self.lineage()
        narrator, character, _ = self.ids(first)
        path = self.aliases(first, {narrator: character})
        target = Path(second["directory"]) / "voice-aliases.json"
        target.write_bytes(path.read_bytes())
        with self.assertRaises(ValueError):
            self.store(second)
        target.unlink()
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_book_config(second, self.config, output=target)
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_book_config(second, self.config, work_dir=target)

    @staticmethod
    def cli(*args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = renderer_main([str(arg) for arg in args])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_cli_edits_book_only_and_removal_restores_inherited_mapping(self):
        first, second, third, _ = self.lineage()
        narrator, character, alternate = self.ids(first)
        config = self.root / "epub-audiobook.toml"
        config.write_text('[engine]\ncheckpoint = "./tts-model"\n'
                          '[verification]\nmodel = "./asr-model"\n', encoding="utf-8")
        config_before = config.read_bytes()
        with contextlib.chdir(self.root):
            result, _, error = self.cli("voice-aliases", first["directory"],
                                        "--set", narrator, character)
            self.assertEqual(result, 0, error)
            first_before = self.snapshot(Path(first["directory"]))
            result, _, error = self.cli("voice-aliases", second["directory"],
                                        "--set", narrator, alternate)
            self.assertEqual(result, 0, error)
            self.assertEqual(self.store(third).canonical_id(narrator), alternate)
            result, report, error = self.cli("voice-aliases", third["directory"])
            self.assertEqual(result, 0, error)
            self.assertIn(narrator, report)
            self.assertIn(alternate, report)
            result, _, error = self.cli("voice-aliases", second["directory"],
                                        "--remove", narrator)
            self.assertEqual(result, 0, error)
            self.assertEqual(self.store(third).canonical_id(narrator), character)
        self.assertEqual(config.read_bytes(), config_before)
        self.assertEqual(self.snapshot(Path(first["directory"])), first_before)

    def test_cli_batch_edits_validate_before_writing_and_support_multiple_removals(self):
        first = self.publish(1)
        narrator, character, alternate = self.ids(first)
        config = self.root / "epub-audiobook.toml"
        config.write_text('[engine]\ncheckpoint = "./tts-model"\n'
                          '[verification]\nmodel = "./asr-model"\n', encoding="utf-8")
        with contextlib.chdir(self.root):
            result, _, error = self.cli("voice-aliases", first["directory"],
                                        "--set", narrator, character,
                                        "--set", alternate, character)
            self.assertEqual(result, 0, error)
            path = Path(first["directory"]) / "voice-aliases.json"
            before = path.read_bytes()
            result, _, error = self.cli("voice-aliases", first["directory"],
                                        "--set", narrator, alternate,
                                        "--set", alternate, narrator)
            self.assertNotEqual(result, 0)
            self.assertIn("cycle", error)
            self.assertEqual(path.read_bytes(), before)
            result, _, error = self.cli("voice-aliases", first["directory"],
                                        "--remove", narrator, "--remove", alternate)
            self.assertEqual(result, 0, error)
            self.assertEqual(self.store(first).canonical_id(narrator), narrator)
            self.assertEqual(self.store(first).canonical_id(alternate), alternate)


if __name__ == "__main__":
    unittest.main()
