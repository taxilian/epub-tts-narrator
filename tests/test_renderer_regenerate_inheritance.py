"""Bulk recasting preserves inherited voices unless the caller opts in."""

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.analysis_directory import (
    continuation_registry, fresh_registry, publish_directory,
)
from epub_tts_narrator.audition_scripts import ensure_bundle_scripts
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.series import link_analysis
from audition_fixtures import prepare_document
import test_book_voice_workflow as book_fixtures
from test_renderer_voice_design_integration import FFMPEG
from test_series import fixture, propose


class RegenerateInheritanceTests(book_fixtures.BookFixture):
    def setUp(self):
        super().setUp()
        self.config_path = self.root / "renderer.toml"
        self.config_path.write_text(
            '[engine]\ncheckpoint="./tts-model"\n'
            '[verification]\nmodel="./asr-model"\n'
            '[voice_design]\ntype="engine"\n', encoding="utf-8")
        self.sample = self.root / "seed.wav"
        # Pre-existing selections are checked for immutability, never played.
        self.sample.write_bytes(b"immutable inherited reference fixture")

    def publish(self, number, parent=None, *, prepared=True):
        directory = self.root / f"book-{number:02}"
        document = fixture(sequel=parent is not None)
        document["source"]["book"].update(id=f"book-{number}", title=f"Journey {number}")
        document["cast"]["voices"].append({
            "id": "additional-voice", "description": "Additional warm voice.", "traits": [],
        })
        document["cast"]["roles"].append({
            "id": "additional-role", "kind": "narration", "character_id": None,
            "voice_id": "additional-voice", "description": "Additional narration.",
        })
        if parent is None:
            registry = fresh_registry(directory)
        else:
            registry = continuation_registry(parent["directory"], document["source"])
            propose(document, parent["cast"]["characters"][0]["id"])
        document = link_analysis(document, registry)
        if prepared:
            document = prepare_document(document)
        publish_directory(document, registry, directory, export_bundle,
                          continue_from=parent["directory"] if parent else None)
        return load_bundle(directory)

    @staticmethod
    def ids(bundle):
        roles = bundle["cast"]["roles"]
        narrator = next(role["voice_id"] for role in roles if role["id"] == "narrator_main")
        extra = next(role["voice_id"] for role in roles if role["id"] == "additional-role")
        return narrator, bundle["cast"]["characters"][0]["voice_id"], extra

    def seed(self, book, voice_id):
        store = self.store(book)
        profile = next(profile for profile in book["cast"]["voices"] if profile["id"] == voice_id)
        lock = store.lock()
        try:
            return store.publish(
                voice_id, profile, self.sample, "An existing verified reference.",
                self.engine.identity, {"passed": True}, {"description_applied": True})
        finally:
            lock.rmdir()

    def alias_narrator(self, book, *, extra_target=None):
        narrator, character, extra = self.ids(book)
        aliases = {narrator: character}
        if extra_target is not None:
            aliases[extra] = extra_target
        (Path(book["directory"]) / "voice-aliases.json").write_text(json.dumps({
            "format": "audiobook-book-voice-aliases", "version": 1,
            "series_id": book["manifest"]["provenance"]["series"]["series_id"],
            "book_id": book["source"]["book"]["id"], "aliases": aliases,
        }), encoding="utf-8")

    def run_cast(self, book, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(["cast", book["directory"], "--config", str(self.config_path),
                           "--action", "regenerate-all", *arguments])
        return status, stdout.getvalue(), stderr.getvalue()

    @unittest.skipUnless(FFMPEG, "FFmpeg is required for fixture voice normalization")
    def test_default_recasts_local_and_missing_but_excludes_inherited_scripts_and_audio(self):
        first = self.publish(1)
        second = self.publish(2, first)
        narrator, character, missing = self.ids(second)
        inherited = self.seed(first, character)
        original_local = self.seed(second, narrator)
        before = self.snapshot(Path(first["directory"]))
        with patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"), \
                patch("epub_tts_narrator.renderer.engines.create_engine", return_value=self.engine), \
                patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=self.verifier), \
                patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts",
                      wraps=ensure_bundle_scripts) as scripts:
            status, output, error = self.run_cast(second)
        self.assertEqual(status, 0, error)
        self.assertEqual(set(scripts.call_args.kwargs["voice_ids"]), {narrator, missing})
        self.assertEqual(len(self.engine.calls), 2)
        store = self.store(second)
        self.assertTrue(store.load(character)["inherited"])
        self.assertEqual(store.load(character)["asset_hash"], inherited["asset_hash"])
        self.assertEqual(store.list_candidates(character), [])
        self.assertNotEqual(store.load(narrator)["asset_hash"], original_local["asset_hash"])
        self.assertFalse(store.load(missing)["inherited"])
        self.assertEqual(self.snapshot(Path(first["directory"])), before)
        self.assertIn("Skipped 1 inherited voice", output)
        self.assertIn("--include-inherited", output)
        self.assertIn("This run targets 2 distinct voices.", error)

    @unittest.skipUnless(FFMPEG, "FFmpeg is required for fixture voice normalization")
    def test_include_inherited_creates_one_local_override_for_shared_roles(self):
        first = self.publish(1)
        second = self.publish(2, first)
        third = self.publish(3, second)
        self.alias_narrator(first)
        narrator, character, extra = self.ids(second)
        inherited = self.seed(first, character)
        before = self.snapshot(Path(first["directory"]))
        with patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"), \
                patch("epub_tts_narrator.renderer.engines.create_engine", return_value=self.engine), \
                patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=self.verifier):
            status, _, error = self.run_cast(second, "--include-inherited")
        self.assertEqual(status, 0, error)
        self.assertEqual(len(self.engine.calls), 2)
        store = self.store(second)
        replacement = store.load(character)
        self.assertFalse(replacement["inherited"])
        self.assertNotEqual(replacement["asset_hash"], inherited["asset_hash"])
        self.assertEqual(store.load(narrator)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(len(store.list_candidates(character)), 1)
        self.assertFalse(store.load(extra)["inherited"])
        self.assertEqual(self.store(third).load(character)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.snapshot(Path(first["directory"])), before)
        self.assertIn("Linked roles in this book", error)
        self.assertIn("This run targets 2 distinct voices.", error)

    def test_all_inherited_from_grandparent_is_read_only_without_preparing_scripts_or_models(self):
        first = self.publish(1, prepared=False)
        second = self.publish(2, first, prepared=False)
        third = self.publish(3, second, prepared=False)
        self.alias_narrator(first)
        _, character, extra = self.ids(first)
        self.seed(first, character)
        self.seed(first, extra)
        # Secondary narration needs an explicit identity mapping across books.
        self.alias_narrator(third, extra_target=extra)
        before = self.snapshot(self.root)
        with patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts") as scripts, \
                patch("epub_tts_narrator.renderer.engines.create_engine") as engine, \
                patch("epub_tts_narrator.renderer.voice_design.create_designer") as designer, \
                patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier:
            status, output, error = self.run_cast(third)
        self.assertEqual(status, 0, error)
        scripts.assert_not_called()
        engine.assert_not_called()
        designer.assert_not_called()
        verifier.assert_not_called()
        self.assertEqual(self.snapshot(self.root), before)
        self.assertFalse((Path(third["directory"]) / "voices").exists())
        self.assertIn("Skipped 2 inherited voice", output)
        self.assertIn("No voices to regenerate", output)
        self.assertNotIn("WARNING", error)

    def failed_parent_lineage(self):
        first = self.publish(1)
        second = self.publish(2, first)
        self.alias_narrator(first)
        _, character, extra = self.ids(first)
        self.alias_narrator(second, extra_target=extra)
        self.seed(first, character)
        self.seed(first, extra)
        store = self.store(first)
        lock = store.lock()
        try:
            store.invalidate(character)
            store.mark_regeneration_failed(character, "Parent voice creation was rejected.")
        finally:
            lock.rmdir()
        self.assertTrue(self.store(second).regeneration_status(character)["inherited"])
        return first, second, character

    def test_default_bulk_skips_inherited_failure_without_scripts_or_generation(self):
        first, second, character = self.failed_parent_lineage()
        before = self.snapshot(self.root)
        with patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts") as scripts, \
                patch("epub_tts_narrator.renderer.engines.create_engine") as engine, \
                patch("epub_tts_narrator.renderer.voice_design.create_designer") as designer, \
                patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier:
            status, output, error = self.run_cast(second)
        self.assertEqual(status, 0, error)
        scripts.assert_not_called()
        engine.assert_not_called()
        designer.assert_not_called()
        verifier.assert_not_called()
        self.assertEqual(self.snapshot(self.root), before)
        self.assertIn("Skipped 2 inherited voice", output)
        self.assertIn("No voices to regenerate", output)
        self.assertIsNone(self.store(second).load(character))
        self.assertEqual(self.store(first).regeneration_status(character)["state"], "failed")

    @unittest.skipUnless(FFMPEG, "FFmpeg is required for fixture voice normalization")
    def test_include_inherited_replaces_failed_ancestor_voice_only_in_child(self):
        first, second, character = self.failed_parent_lineage()
        before = self.snapshot(Path(first["directory"]))
        with patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"), \
                patch("epub_tts_narrator.renderer.engines.create_engine", return_value=self.engine), \
                patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=self.verifier):
            status, output, error = self.run_cast(second, "--include-inherited")
        self.assertEqual(status, 0, error)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertIn("Regenerated 2", output)
        replacement = self.store(second).load(character)
        self.assertIsNotNone(replacement)
        self.assertFalse(replacement["inherited"])
        self.assertIsNone(self.store(second).regeneration_status(character))
        self.assertIsNone(self.store(first).load(character))
        self.assertEqual(self.store(first).regeneration_status(character)["state"], "failed")
        self.assertEqual(self.snapshot(Path(first["directory"])), before)

    @unittest.skipUnless(FFMPEG, "FFmpeg is required for fixture voice normalization")
    def test_child_retry_records_its_own_failure_then_clears_it_without_changing_parent(self):
        first, second, character = self.failed_parent_lineage()
        before = self.snapshot(Path(first["directory"]))
        with contextlib.closing(self.session(second)) as session:
            with patch.object(session, "_generate_locked", side_effect=RuntimeError("Child provider rejected voice")):
                with self.assertRaisesRegex(RuntimeError, "Child provider rejected voice"):
                    session.regenerate(character)
            marker = session.store.regeneration_status(character)
            self.assertFalse(marker["inherited"])
            self.assertEqual(marker["state"], "failed")
            self.assertIn("Child provider rejected voice", marker["error"])
            self.assertEqual(self.snapshot(Path(first["directory"])), before)
        with contextlib.closing(self.session(second)) as resumed:
            replacement = resumed.regenerate(character)
            self.assertFalse(resumed.store.load(character)["inherited"])
            self.assertEqual(resumed.store.load(character)["asset_hash"], replacement["asset_hash"])
            self.assertIsNone(resumed.store.regeneration_status(character))
        self.assertEqual(self.snapshot(Path(first["directory"])), before)
        self.assertEqual(self.store(first).regeneration_status(character)["state"], "failed")
        self.assertIsNone(self.store(first).load(character))

    def test_include_inherited_rejected_for_other_actions_before_scripts_or_session(self):
        first = self.publish(1)
        for action in ("list", "generate-missing", "interactive", "prepare-samples", "normalize-samples"):
            with self.subTest(action=action), \
                    patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts") as scripts, \
                    patch("epub_tts_narrator.renderer.casting.CastingSession") as session:
                status, _, error = self.run_cast(first, "--action", action, "--include-inherited")
            self.assertEqual(status, 2)
            self.assertIn("--include-inherited", error)
            self.assertIn("regenerate-all", error)
            scripts.assert_not_called()
            session.assert_not_called()
