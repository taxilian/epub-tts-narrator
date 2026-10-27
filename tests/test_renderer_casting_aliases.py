"""Casting duplicate choices reuse audio immediately without merging characters."""

from copy import deepcopy
from pathlib import Path
import unittest

from epub_tts_narrator.analysis_directory import continuation_registry, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.series import link_analysis
import test_book_voice_workflow as book_fixtures
from test_series import fixture, propose


class CastingAliasesTests(book_fixtures.BookFixture):
    def publish(self, number, parent=None):
        directory = self.root / f"book-{number:02}"
        document = fixture(sequel=parent is not None)
        document["source"]["book"].update(id=f"book-{number}", title=f"Journey {number}")
        document["cast"]["voices"].append({
            "id": "alternate-voice", "description": "An independent alternate narrator.", "traits": [],
        })
        document["cast"]["roles"].append({
            "id": "alternate-narrator", "kind": "narration", "character_id": None,
            "voice_id": "alternate-voice", "description": "The alternate narrator.",
        })
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
        narrator = next(role["voice_id"] for role in roles if role["id"] == "narrator_main")
        alternate = next(role["voice_id"] for role in roles if role["description"] == "The alternate narrator.")
        return narrator, bundle["cast"]["characters"][0]["voice_id"], alternate

    def open_session(self, bundle):
        session = self.session(bundle, models=False)
        self.addCleanup(session.close)
        return session

    def save_reference(self, session, voice_id):
        audio = self.root / (voice_id + ".wav")
        audio.write_bytes(("verified-reference-" + voice_id).encode())
        with session._locked():
            return session.store.publish(
                voice_id, session._profile(voice_id), audio,
                "An established reference for this character.", {"engine": "fixture"},
                {"passed": True}, {"description_applied": True})

    def test_share_reuses_saved_target_immediately_and_on_reopen_without_changing_source_assets(self):
        book = self.publish(1)
        narrator, character, _ = self.ids(book)
        session = self.open_session(book)
        source = self.save_reference(session, narrator)
        target = self.save_reference(session, character)
        cast_before = deepcopy(session.bundle["cast"])
        saved_before = self.snapshot(session.store.directory)
        session._sample_default(narrator)
        self.assertTrue(session._samples)

        result = session.share_voice(narrator, character)

        self.assertEqual(result["local"], {narrator: character})
        self.assertEqual(result["affected_voice_ids"], [narrator])
        self.assertEqual(session._samples, {})
        rows = {row["voice_id"]: row for row in session.rows()}
        self.assertEqual(rows[narrator]["canonical_voice_id"], character)
        self.assertEqual(rows[narrator]["selected"]["asset_hash"], target["asset_hash"])
        self.assertEqual(session.bundle["cast"], cast_before)
        after = self.snapshot(session.store.directory)
        self.assertTrue(all(after[path] == content for path, content in saved_before.items()
                            if path != ".lock.lease"))
        self.assertTrue(Path(source["reference_audio"]).is_file())
        reopened = self.open_session(load_bundle(book["directory"]))
        self.assertEqual(reopened.store.load(narrator)["asset_hash"], target["asset_hash"])
        self.assertEqual(self.engine.calls, [])

    def test_repointing_an_alias_leaves_its_former_target_and_recordings_unchanged(self):
        book = self.publish(1)
        narrator, character, alternate = self.ids(book)
        session = self.open_session(book)
        character_reference = self.save_reference(session, character)
        alternate_reference = self.save_reference(session, alternate)
        session.share_voice(narrator, character)

        result = session.share_voice(narrator, alternate)

        self.assertEqual(result["affected_voice_ids"], [narrator])
        self.assertEqual(result["local"], {narrator: alternate})
        self.assertEqual(session.store.load(character)["asset_hash"], character_reference["asset_hash"])
        self.assertEqual(session.store.load(narrator)["asset_hash"], alternate_reference["asset_hash"])
        self.assertEqual(session.store.canonical_id(character), character)

    def test_existing_dependents_follow_a_canonical_source_and_target_links_stay_live(self):
        book = self.publish(1)
        narrator, character, alternate = self.ids(book)
        session = self.open_session(book)
        session.share_voice(narrator, character)
        result = session.share_voice(character, alternate)
        self.assertEqual(set(result["affected_voice_ids"]), {narrator, character})
        self.assertEqual(result["local"], {narrator: character, character: alternate})
        self.assertEqual(session.store.canonical_id(narrator), alternate)

    def test_duplicate_voice_choice_keeps_distinct_character_records_and_role_assignments(self):
        book = self.publish(1)
        _, character, alternate = self.ids(book)
        other = deepcopy(book["cast"]["characters"][0])
        other.update(id="another-character", name="Another person", voice_id=alternate)
        book["cast"]["characters"].append(other)
        before = deepcopy(book["cast"])
        session = self.open_session(book)

        session.share_voice(character, alternate)

        self.assertEqual(session.bundle["cast"], before)
        self.assertEqual(session.store.canonical_id(character), alternate)
        self.assertNotEqual(session.bundle["cast"]["characters"][0]["id"], other["id"])
        self.assertEqual(book["cast"], before)

    def test_invalid_ids_self_links_and_alias_reversal_do_not_change_policy(self):
        book = self.publish(1)
        narrator, character, _ = self.ids(book)
        session = self.open_session(book)
        session.share_voice(narrator, character)
        path = Path(book["directory"]) / "voice-aliases.json"
        before = path.read_bytes()
        for source, target in ((narrator, narrator), ("missing", narrator),
                               (narrator, "missing"), (None, narrator),
                               (character, narrator), (narrator, character)):
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                session.share_voice(source, target)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(session.store.canonical_id(narrator), character)

    def test_local_override_is_live_in_descendants_and_never_changes_ancestors_or_siblings(self):
        first, second, third, sibling = self.lineage()
        narrator, character, _ = self.ids(first)
        _, _, alternate = self.ids(second)
        parent = self.open_session(first)
        child = self.open_session(second)
        later = self.open_session(third)
        other_branch = self.open_session(sibling)
        target = self.save_reference(parent, character)
        replacement = self.save_reference(child, alternate)
        parent.share_voice(narrator, character)
        self.assertEqual(later.store.load(narrator)["asset_hash"], target["asset_hash"])
        parent_before = self.snapshot(Path(first["directory"]))
        sibling_before = self.snapshot(Path(sibling["directory"]))

        child.share_voice(narrator, alternate)

        self.assertEqual(child.store.load(narrator)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(later.store.load(narrator)["asset_hash"], replacement["asset_hash"])
        self.assertEqual(other_branch.store.load(narrator)["asset_hash"], target["asset_hash"])
        self.assertEqual(parent.store.load(narrator)["asset_hash"], target["asset_hash"])
        self.assertEqual(self.snapshot(Path(first["directory"])), parent_before)
        self.assertEqual(self.snapshot(Path(sibling["directory"])), sibling_before)
        self.assertEqual(read_json(Path(second["directory"]) / "voice-aliases.json")["aliases"],
                         {narrator: alternate})
        self.assertFalse((Path(third["directory"]) / "voice-aliases.json").exists())

    def test_inherited_reverse_edge_is_rejected_before_writing_local_policy(self):
        first = self.publish(1)
        second = self.publish(2, first)
        narrator, character, _ = self.ids(first)
        parent = self.open_session(first)
        child = self.open_session(second)
        parent.share_voice(narrator, character)
        with self.assertRaisesRegex(ValueError, "cycle"):
            child.share_voice(character, narrator)
        self.assertFalse((Path(second["directory"]) / "voice-aliases.json").exists())
        self.assertEqual(child.store.canonical_id(narrator), character)


if __name__ == "__main__":
    unittest.main()
