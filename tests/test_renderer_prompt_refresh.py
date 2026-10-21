"""Refreshing casting text is explicit and never replaces accepted recordings."""

from copy import deepcopy
from contextlib import closing
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from audiobook_renderer.common import read_json
from audiobook_renderer.voice_prompts import audiobook_voice_prompt
from audiobook_renderer.samples import MissingSampleScript
from epub_audiobook.provider import ProviderError
import test_renderer_sample_drafts as drafts_fixture
import test_renderer_audition as cli_fixture
import test_book_voice_workflow as book_fixture


class PromptRefreshTests(unittest.TestCase):
    setUp = drafts_fixture.SampleDraftTests.setUp
    new_session = drafts_fixture.SampleDraftTests.new_session
    saved_voice = drafts_fixture.SampleDraftTests.saved_voice
    assert_no_models = drafts_fixture.SampleDraftTests.assert_no_models

    def helpers(self, session=None):
        session = session or self.session
        def descriptions(bundle, config_path, *, canonical_id, voice_ids, current_prompts, progress):
            return {"version": 1, "prompts": {
                canonical_id(voice_id): {"description": "A measured voice with clear consonants.",
                                         "reason": "Retains the intended identity."}
                for voice_id in voice_ids}}
        writer = self.enterContext(patch("epub_audiobook.voice_prompt_refresh.refresh_bundle_prompts",
                                        side_effect=descriptions))
        scripts = self.enterContext(patch("epub_audiobook.audition_scripts.ensure_bundle_scripts",
                                         return_value=session.bundle))
        return writer, scripts

    def test_replaces_drafts_keeps_recording_and_archives_previous_custom_edits(self):
        saved = self.saved_voice(text=drafts_fixture.EXCERPT, source={"kind": "user"})
        self.session.update_prompt(self.voice_id, prompt="My carefully edited gravelly voice.",
                                   reference_text="My previous personal audition text.")
        before = read_json(self.session._draft_path(self.voice_id))
        original = deepcopy(self.bundle)
        audio = Path(saved["reference_audio"]).read_bytes()
        writer, scripts = self.helpers()
        result = self.session.refresh_prompts(voice_ids=[self.voice_id], config_path=self.root / "global.toml")
        self.assertEqual(writer.call_args.kwargs["current_prompts"][self.voice_id], before["prompt"])
        self.assertTrue(scripts.call_args.kwargs["refresh"])
        refreshed = read_json(self.session._draft_path(self.voice_id))
        self.assertEqual(refreshed["prompt"], audiobook_voice_prompt("A measured voice with clear consonants."))
        self.assertEqual(refreshed["reference_text"], drafts_fixture.EXCERPT)
        history = read_json(result["history"])
        self.assertEqual(history["status"], "complete")
        self.assertEqual(history["previous_drafts"][self.voice_id], before)
        self.assertEqual(self.session.store.load(self.voice_id), saved)
        self.assertEqual(Path(saved["reference_audio"]).read_bytes(), audio)
        self.assertEqual(self.bundle, original)
        self.assertEqual(self.new_session().rows()[0]["prompt"], refreshed["prompt"])
        self.assert_no_models()

    def test_failed_model_refresh_keeps_every_draft_and_selection(self):
        saved = self.saved_voice()
        self.session.prepare_samples()
        before = {path: path.read_bytes() for path in self.session.store.directory.rglob("*.json")}
        writer, scripts = self.helpers()
        writer.side_effect = RuntimeError("Text model timeout")
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.session.refresh_prompts()
        scripts.assert_not_called()
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(self.session.store.load(self.voice_id), saved)
        self.assertFalse((self.session.store.directory / ".lock").exists())
        self.assert_no_models()

    def test_missing_script_keeps_edited_voice_direction_as_refresh_context(self):
        saved = self.saved_voice(text="Too short.", source={"kind": "book_excerpt"})
        for existing_draft in (False, True):
            with self.subTest(existing_draft=existing_draft):
                expected = saved["creation"]["requested_description"]
                if existing_draft:
                    expected = "My precise correction: a deep gravelly male voice."
                    self.session._save_draft(self.voice_id, {
                        "prompt": expected, "reference_text": "Too short.",
                        "sample_source": {"kind": "book_excerpt"}})
                self.helpers()
                with patch.object(self.session, "_sample_default", side_effect=MissingSampleScript("No script")), \
                        patch("epub_audiobook.voice_prompt_refresh.refresh_bundle_prompts",
                              side_effect=ProviderError("Stop after reading prompt")) as writer:
                    with self.assertRaises(ProviderError):
                        self.session.refresh_prompts(voice_ids=[self.voice_id])
                self.assertEqual(writer.call_args.kwargs["current_prompts"][self.voice_id], expected)

    def test_failed_second_draft_write_rolls_back_first_draft(self):
        self.session.prepare_samples()
        before = {path: read_json(path) for path in (self.session.store.directory / "casting-drafts").glob("*.json")}
        self.helpers()
        original_save = self.session._save_draft
        calls = 0
        def fail_second(voice_id, draft):
            nonlocal calls
            calls += 1
            original_save(voice_id, draft)
            if calls == 2:
                raise OSError("Interrupted after writing second draft")
        with patch.object(self.session, "_save_draft", side_effect=fail_second), self.assertRaises(OSError):
            self.session.refresh_prompts()
        self.assertEqual({path: read_json(path) for path in before}, before)
        self.assertFalse((self.session.store.directory / ".lock").exists())
        self.assert_no_models()

    def test_shared_roles_receive_one_new_draft_and_invalid_target_prevents_requests(self):
        bundle = deepcopy(self.bundle)
        alias = "shared-narrator"
        bundle["cast"]["voices"].append({"id": alias, "description": "Mara narrating.", "traits": []})
        bundle["cast"]["roles"][0]["voice_id"] = alias
        session = self.new_session(bundle)
        writer, _ = self.helpers(session)
        result = session.refresh_prompts(voice_ids=[alias, self.voice_id])
        self.assertEqual(result["voices"], [self.voice_id])
        self.assertEqual(session._draft_path(alias), session._draft_path(self.voice_id))
        self.assertEqual(set(writer.call_args.kwargs["current_prompts"]), {self.voice_id})
        writer.reset_mock()
        with self.assertRaisesRegex(ValueError, "Voice ID"):
            session.refresh_prompts(voice_ids=[self.voice_id, "unknown"])
        writer.assert_not_called()
        self.assert_no_models()

    def test_refresh_leaves_failed_bulk_marker_failed(self):
        self.session.begin_regeneration([self.voice_id])
        self.session.store.mark_regeneration_failed(self.voice_id, "Voice rejected")
        marker = self.session.store.regeneration_status(self.voice_id)
        self.helpers()
        self.session.refresh_prompts(voice_ids=[self.voice_id])
        self.assertEqual(self.session.store.regeneration_status(self.voice_id), marker)
        self.assertIsNone(self.session.store.load(self.voice_id))
        self.assert_no_models()


class RefreshCommandTests(unittest.TestCase):
    setUp = cli_fixture.CastingCLITests.setUp
    run_cli = cli_fixture.CastingCLITests.run_cli

    def test_cli_reports_text_provider_failure_and_cached_retry(self):
        session = cli_fixture.SessionDouble()
        session.close = Mock()
        session.refresh_prompts = Mock(side_effect=ProviderError("Model request timed out"))
        with patch("audiobook_renderer.casting.CastingSession", return_value=session):
            status, output, error = self.run_cli("--action", "refresh-prompts")
        self.assertEqual(status, 2)
        self.assertIn("Model request timed out", error)
        self.assertIn("remain cached", error)
        self.assertIn("Rerun the same command", error)
        self.assertEqual(session.events, [])

    def test_cli_refresh_all_or_one_only_updates_prompts(self):
        for voice_id in (None, "v1"):
            session = cli_fixture.SessionDouble()
            session.close = Mock()
            session.refresh_prompts = Mock(return_value={"voices": ["v1"], "history": "saved-history.json"})
            with patch("audiobook_renderer.casting.CastingSession", return_value=session), \
                    patch("audiobook_renderer.audition.AudioPlayer") as player:
                args = ["--action", "refresh-prompts"] + (["--voice-id", voice_id] if voice_id else [])
                status, output, error = self.run_cli(*args)
            self.assertEqual(status, 0, error)
            session.refresh_prompts.assert_called_once_with(
                voice_ids=[voice_id] if voice_id else None, config_path=self.config_path)
            self.assertIn("Recorded voices are unchanged", output)
            self.assertIn("1 distinct voice", output)
            player.return_value.start.assert_not_called()
            self.assertEqual(session.events, [])


class InheritedPromptRefreshTests(book_fixture.BookFixture):
    def test_later_book_refresh_writes_only_local_draft_and_keeps_inherited_binding(self):
        first = self.publish(1)
        second = self.publish(2, first)
        third = self.publish(3, second)
        voice_id = first["cast"]["characters"][0]["voice_id"]
        profile = next(item for item in first["cast"]["voices"] if item["id"] == voice_id)
        reference = self.root / "ancestor-reference.wav"
        reference.write_bytes(b"immutable ancestor recording, never decoded by text refresh")
        store = self.store(first)
        lock = store.lock()
        try:
            store.publish(voice_id, profile, reference, drafts_fixture.EXCERPT,
                          self.engine.identity, {"passed": True}, {
                              "description_applied": True,
                              "requested_description": "The established warm character voice.",
                              "sample_source": {"kind": "user"}})
        finally:
            lock.rmdir()
        with closing(self.session(first, models=False)) as ancestor:
            ancestor.update_prompt(voice_id, "An ancestor's custom casting draft.", drafts_fixture.EXCERPT)
        before = {book["directory"]: self.snapshot(Path(book["directory"])) for book in (first, second)}
        original_cast = deepcopy(third["cast"])
        with closing(self.session(third, models=False)) as session:
            inherited = session.store.load(voice_id)
            self.assertTrue(inherited["inherited"])
            self.assertFalse(session._draft_path(voice_id).exists())
            description = "A measured warm voice with clear consonants."
            prepared = deepcopy(session.bundle)
            with patch("epub_audiobook.voice_prompt_refresh.refresh_bundle_prompts", return_value={
                    "version": 1, "prompts": {voice_id: {
                        "description": description, "reason": "Preserves this character's vocal identity."}}}) as writer, \
                    patch("epub_audiobook.audition_scripts.ensure_bundle_scripts", return_value=prepared) as scripts:
                result = session.refresh_prompts(voice_ids=[voice_id])
            draft_path = session._draft_path(voice_id)
            self.assertTrue(draft_path.is_relative_to(Path(third["directory"]) / "voices"))
            self.assertEqual(read_json(draft_path)["prompt"], audiobook_voice_prompt(description))
            self.assertEqual(result["voices"], [voice_id])
            self.assertEqual(writer.call_args.kwargs["current_prompts"][voice_id],
                             "The established warm character voice.")
            self.assertTrue(scripts.call_args.kwargs["refresh"])
            self.assertEqual(session.store.load(voice_id), inherited)
            self.assertEqual(session.store.list_candidates(voice_id), [])
            self.assertEqual(session.bundle["cast"], original_cast)
        with closing(self.session(third, models=False)) as resumed:
            row = next(item for item in resumed.rows() if item["voice_id"] == voice_id)
            self.assertEqual(row["prompt"], audiobook_voice_prompt(description))
            self.assertEqual(row["selected"], inherited)
        self.assertEqual(self.engine.calls, [])
        for directory, snapshot in before.items():
            self.assertEqual(self.snapshot(Path(directory)), snapshot)


if __name__ == "__main__":
    unittest.main()
