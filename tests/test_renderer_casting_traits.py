"""Manual casting traits survive reopening and condition only affected speech."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.casting import CastingSession
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.speech_traits import save_speech_traits, speech_trait_overrides
from epub_tts_narrator.renderer.workflow import _preflight, make_render_plan
from audition_fixtures import prepare_bundle
import test_book_context


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"
TRAITS = "Light Mexican Spanish-influenced English accent; relaxed conversational cadence"


class CastingTraitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "book"
        shutil.copytree(EXAMPLE, self.directory)
        self.bundle = prepare_bundle(load_bundle(self.directory))
        self.config = resolve_book_config(self.bundle, deepcopy(DEFAULTS))
        self.session = CastingSession(self.bundle, self.config, notify=lambda _: None)
        self.addCleanup(self.session.close)
        self.voice_id = self.bundle["cast"]["voices"][0]["id"]

    def originals(self):
        return {str(path.relative_to(self.directory)): path.read_bytes()
                for path in self.directory.rglob("*") if path.is_file()}

    def test_traits_can_be_edited_without_a_sample_or_model_and_reopen_immediately(self):
        before = self.originals()
        with patch.object(self.session, "_draft", side_effect=AssertionError("no sample needed")):
            self.session.update_prompt(self.voice_id, speech_traits=TRAITS)
        self.assertEqual(speech_trait_overrides(self.bundle, self.config), {self.voice_id: TRAITS})
        reopened = CastingSession(self.bundle, self.config, notify=lambda _: None)
        self.addCleanup(reopened.close)
        row = next(row for row in reopened.rows() if row["voice_id"] == self.voice_id)
        self.assertEqual(row["speech_traits"], TRAITS)
        after = self.originals()
        self.assertTrue(all(after[path] == data for path, data in before.items()))
        self.assertIsNone(self.session._engine)
        self.assertIsNone(self.session._verifier)
        self.assertIsNone(self.session._designer)

    def test_invalid_traits_fail_before_prompt_or_sidecar_changes(self):
        before = self.originals()
        for value in (" ", "a\nb", "[shouting]", "a" * 401, 10):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.session.update_prompt(self.voice_id, prompt="A new prompt", speech_traits=value)
            self.assertEqual(self.originals(), before)

    def test_prompt_script_and_traits_save_together_without_putting_traits_in_draft_schema(self):
        self.session.update_prompt(self.voice_id, prompt="A measured tenor voice.",
                                   reference_text="Here is the new sample.", speech_traits=TRAITS)
        row = next(row for row in self.session.rows() if row["voice_id"] == self.voice_id)
        self.assertEqual(row["prompt"], "A measured tenor voice.")
        self.assertEqual(row["reference_text"], "Here is the new sample.")
        self.assertEqual(row["speech_traits"], TRAITS)
        self.assertNotIn("speech_traits", read_json(self.session._draft_path(self.voice_id)))

    def test_trait_override_changes_only_affected_plan_chunks_and_can_be_cleared(self):
        original = make_render_plan(self.bundle, self.config, multi_speaker=False)
        self.session.update_prompt(self.voice_id, speech_traits=TRAITS)
        edited = make_render_plan(self.bundle, self.config, multi_speaker=False)
        affected = 0
        for old_chapter, new_chapter in zip(original["chapters"], edited["chapters"]):
            for old, new in zip(old_chapter["chunks"], new_chapter["chunks"]):
                if old["segment"]["voice_id"] == self.voice_id:
                    self.assertEqual(new["speech_traits"][self.voice_id], TRAITS)
                    self.assertEqual({k: v for k, v in new.items() if k != "speech_traits"},
                                     {k: v for k, v in old.items() if k != "speech_traits"})
                    affected += 1
                else:
                    self.assertEqual(new, old)
        self.assertGreater(affected, 0)
        self.session.update_prompt(self.voice_id, speech_traits="")
        cleared = make_render_plan(self.bundle, self.config, multi_speaker=False)
        for chapter in cleared["chapters"]:
            for chunk in chapter["chunks"]:
                self.assertNotIn(self.voice_id, chunk.get("speech_traits", {}))

    def test_preflight_freezes_traits_for_recovery_but_new_invocation_reads_edits(self):
        self.session.update_prompt(self.voice_id, speech_traits=TRAITS)
        frozen, first, _ = _preflight(self.bundle, self.config, chapters_only=True, multi_speaker=False)
        self.session.update_prompt(self.voice_id, speech_traits="Careful Japanese-influenced English")
        self.assertEqual(make_render_plan(self.bundle, frozen, multi_speaker=False), first)
        _, second, _ = _preflight(self.bundle, frozen, chapters_only=True, multi_speaker=False)
        self.assertNotEqual(first, second)

    def test_symlink_foreign_identity_unknown_voice_and_lock_fail_without_overwriting(self):
        target = self.directory / "voice-traits.json"
        outside = self.root / "outside.json"
        outside.write_text("{}")
        target.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.session.update_prompt(self.voice_id, speech_traits=TRAITS)
        self.assertEqual(outside.read_text(), "{}")
        target.unlink()
        self.session.update_prompt(self.voice_id, speech_traits=TRAITS)
        good = read_json(target)
        for key, value in (("book_id", "wrong"), ("series_id", "wrong"),
                           ("speech_traits", {"unknown": TRAITS})):
            write_json(target, {**good, key: value})
            before = target.read_bytes()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.session.update_prompt(self.voice_id, speech_traits="new")
            self.assertEqual(target.read_bytes(), before)
        write_json(target, good)
        lock = self.directory / ".voice-traits.lock"
        lock.mkdir()
        try:
            with self.assertRaisesRegex(ValueError, "being edited"):
                self.session.update_prompt(self.voice_id, speech_traits="new")
            self.assertEqual(read_json(target), good)
        finally:
            lock.rmdir()

    def test_failed_combined_edit_restores_existing_or_absent_draft(self):
        lock = self.directory / ".voice-traits.lock"
        for has_draft in (False, True):
            if has_draft:
                self.session.update_prompt(self.voice_id, prompt="Original direction")
            path = self.session._draft_path(self.voice_id)
            before = path.read_bytes() if path.exists() else None
            lock.mkdir()
            try:
                with self.assertRaisesRegex(ValueError, "being edited"):
                    self.session.update_prompt(self.voice_id, prompt="Replacement direction", speech_traits=TRAITS)
                self.assertEqual(path.read_bytes() if path.exists() else None, before)
            finally:
                lock.rmdir()


class CastingTraitInheritanceTests(unittest.TestCase):
    make_book = test_book_context.BookContextTests.make_book

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.first = self.make_book("first")
        self.second = self.make_book("second", self.first)
        self.third = self.make_book("third", self.second)

    def test_nearest_manual_traits_inherit_live_without_changing_ancestors_or_cast(self):
        books = [load_bundle(path) for path in (self.first, self.second, self.third)]
        configs = [resolve_book_config(book, deepcopy(DEFAULTS)) for book in books]
        voice_id = books[0]["cast"]["characters"][0]["voice_id"]
        self.assertTrue(all(voice_id in {v["id"] for v in b["cast"]["voices"]} for b in books))
        originals = {path: (path / "cast.json").read_bytes() for path in (self.first, self.second, self.third)}
        save_speech_traits(books[0], configs[0], voice_id, TRAITS)
        self.assertEqual(speech_trait_overrides(books[2], configs[2])[voice_id], TRAITS)
        save_speech_traits(books[1], configs[1], voice_id, "Measured British accent")
        with patch("epub_tts_narrator.analysis_directory._completed_directory",
                   side_effect=AssertionError("menu lookup must not reparse book scripts")):
            self.assertEqual(speech_trait_overrides(books[2], configs[2])[voice_id], "Measured British accent")
        self.assertEqual(speech_trait_overrides(books[0], configs[0])[voice_id], TRAITS)
        save_speech_traits(books[2], configs[2], voice_id, "")
        self.assertEqual(speech_trait_overrides(books[2], configs[2])[voice_id], "")
        self.assertEqual(speech_trait_overrides(books[1], configs[1])[voice_id], "Measured British accent")
        self.assertTrue(all((path / "cast.json").read_bytes() == data for path, data in originals.items()))

    def test_parent_pending_marker_and_changed_cast_are_rejected_by_lightweight_lookup(self):
        first, third = load_bundle(self.first), load_bundle(self.third)
        first_config, third_config = [resolve_book_config(book, deepcopy(DEFAULTS)) for book in (first, third)]
        voice_id = first["cast"]["characters"][0]["voice_id"]
        save_speech_traits(first, first_config, voice_id, TRAITS)
        marker = self.first / ".voice-profile-refresh.json"
        marker.write_text("{}")
        with self.assertRaisesRegex(ValueError, "refresh is pending"):
            speech_trait_overrides(third, third_config)
        marker.unlink()
        with (self.first / "cast.json").open("a") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "cast identity changed"):
            speech_trait_overrides(third, third_config)
