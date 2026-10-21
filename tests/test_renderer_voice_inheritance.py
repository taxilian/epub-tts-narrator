"""Book-local voice overrides inherit live, immutable references from ancestors."""

from pathlib import Path
import tempfile
import unittest
import uuid

from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.voices import VoiceStore


class VoiceInheritanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.wav = self.root / "sample.wav"
        self.wav.write_bytes(b"verified-audio-fixture")
        self.one = self.store("book-01")
        self.two = self.store("book-02", parents=(self.one,))
        self.three = self.store("book-03", parents=(self.two,))

    def store(self, name, **kwargs):
        return VoiceStore(self.root / name / "voices", "series-one", **kwargs)

    def arguments(self, voice_id="voice-one", text="The original voice."):
        return (voice_id, {"description": "A clear voice."}, self.wav, text,
                {"engine": "fixture"}, {"passed": True}, {"description_applied": True})

    def candidate(self, store, text="A revised voice.", voice_id="voice-one", candidate_id=None):
        return store.publish_candidate(
            *self.arguments(voice_id, text), candidate_id=candidate_id or uuid.uuid4().hex)

    def snapshot(self, path):
        return {str(item.relative_to(path)): item.read_bytes()
                for item in path.rglob("*") if item.is_file()}

    @staticmethod
    def cast(*, alias=False, other=False):
        characters = [{"id": "character-one", "name": "Alex", "voice_id": "voice-one"}]
        roles = [{"id": "dialogue", "character_id": "character-one",
                  "voice_id": "voice-one"}]
        if alias:
            roles.append({"id": "narrator", "character_id": "character-one",
                          "voice_id": "voice-narrator"})
        if other:
            characters.append({"id": "character-two", "name": "Alex",
                               "voice_id": "voice-two"})
        return {"characters": characters, "roles": roles}

    def test_parent_created_after_children_is_found_without_copying(self):
        self.assertIsNone(self.three.load("voice-one"))
        first = self.one.publish(*self.arguments())
        before = self.snapshot(self.one.directory)
        for store in (self.two, self.three):
            inherited = store.load("voice-one")
            self.assertEqual(inherited["asset_hash"], first["asset_hash"])
            self.assertEqual(inherited["reference_audio"], first["reference_audio"])
            self.assertEqual(inherited["voice_directory"], str(self.one.directory))
            self.assertTrue(inherited["inherited"])
            self.assertEqual(store.list_candidates("voice-one"), [])
            self.assertFalse(store.directory.exists())
        self.assertEqual(self.snapshot(self.one.directory), before)

    def test_nearest_override_wins_and_earlier_books_stay_unchanged(self):
        original = self.one.publish(*self.arguments())
        original_files = self.snapshot(self.one.directory)
        replacement = self.candidate(self.two)
        self.two.select_candidate("voice-one", replacement["candidate_id"],
                                  expected_asset_hash=original["asset_hash"])
        self.assertEqual(self.two.load("voice-one")["asset_hash"], replacement["asset_hash"])
        self.assertFalse(self.two.load("voice-one")["inherited"])
        self.assertEqual(self.three.load("voice-one")["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.three.load("voice-one")["voice_directory"], str(self.two.directory))
        second_files = self.snapshot(self.two.directory)
        third = self.candidate(self.three, "A voice for the third book.")
        self.three.select_candidate("voice-one", third["candidate_id"],
                                    expected_asset_hash=replacement["asset_hash"])
        self.assertEqual(self.one.load("voice-one")["asset_hash"], original["asset_hash"])
        self.assertEqual(self.two.load("voice-one")["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.three.load("voice-one")["asset_hash"], third["asset_hash"])
        self.assertEqual(self.snapshot(self.one.directory), original_files)
        self.assertEqual(self.snapshot(self.two.directory), second_files)

    def test_import_can_publish_local_override_of_inherited_reference(self):
        first = self.one.publish(*self.arguments())
        local = self.two.publish(*self.arguments(text="A separately imported voice."))
        self.assertNotEqual(first["asset_hash"], local["asset_hash"])
        self.assertEqual(self.three.load("voice-one")["asset_hash"], local["asset_hash"])
        self.assertEqual(self.one.load("voice-one")["asset_hash"], first["asset_hash"])
        with self.assertRaisesRegex(ValueError, "never implicitly replaced"):
            self.two.publish(*self.arguments())

    def test_local_candidate_selection_never_targets_ancestor_revision(self):
        first = self.candidate(self.one)
        self.one.select_candidate("voice-one", first["candidate_id"])
        before = self.snapshot(self.one.directory)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.two.load_candidate("voice-one", first["candidate_id"])
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.two.select_candidate("voice-one", first["candidate_id"])
        self.assertEqual(self.two.list_candidates("voice-one"), [])
        self.assertEqual(self.snapshot(self.one.directory), before)
        self.assertFalse(self.two.directory.exists())

    def test_matching_candidate_ids_do_not_mark_unselected_local_revision_active(self):
        shared_id = "a" * 32
        parent = self.candidate(self.one, candidate_id=shared_id)
        self.one.select_candidate("voice-one", parent["candidate_id"])
        local = self.candidate(self.two, "A distinct local candidate.", candidate_id=shared_id)
        rows = self.two.list_candidates("voice-one")
        self.assertEqual([row["asset_hash"] for row in rows], [local["asset_hash"]])
        self.assertFalse(rows[0]["selected"])
        self.assertFalse(rows[0]["inherited"])
        self.assertTrue(self.two.load("voice-one")["inherited"])

    def test_selection_checks_current_inherited_asset_for_stale_audition(self):
        first = self.one.publish(*self.arguments())
        local = self.candidate(self.two)
        newer = self.candidate(self.one, "An updated earlier-book voice.")
        self.one.select_candidate("voice-one", newer["candidate_id"])
        with self.assertRaisesRegex(ValueError, "Active voice changed"):
            self.two.select_candidate("voice-one", local["candidate_id"],
                                      expected_asset_hash=first["asset_hash"])
        self.assertTrue(self.two.load("voice-one")["inherited"])
        self.assertEqual(self.two.load("voice-one")["asset_hash"], newer["asset_hash"])

    def test_corrupt_local_selection_never_falls_back(self):
        self.one.publish(*self.arguments())
        local = self.candidate(self.two)
        self.two.select_candidate("voice-one", local["candidate_id"])
        path = self.two._selection_path("voice-one")
        pointer = read_json(path)
        pointer["asset_hash"] = "0" * 64
        write_json(path, pointer)
        for store in (self.two, self.three):
            with self.assertRaisesRegex(ValueError, "selection changed"):
                store.load("voice-one")

    def test_corrupt_local_audio_never_falls_back(self):
        self.one.publish(*self.arguments())
        local = self.two.publish(*self.arguments(text="The override."))
        Path(local["reference_audio"]).write_bytes(b"changed")
        for store in (self.two, self.three):
            with self.assertRaisesRegex(ValueError, "reference changed"):
                store.load("voice-one")

    def test_incomplete_or_symlinked_local_override_never_falls_back(self):
        self.one.publish(*self.arguments())
        folder = self.two.folder("voice-one")
        folder.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "regular, nonsymlink"):
            self.three.load("voice-one")
        folder.rmdir()
        folder.symlink_to(self.root / "does-not-exist")
        with self.assertRaisesRegex(ValueError, "must not be symlinks"):
            self.three.load("voice-one")

    def test_alias_from_prior_cast_survives_absent_role_and_uses_local_override(self):
        first = self.store("aliases-01", cast=self.cast(alias=True))
        first.persist_aliases()
        original = first.publish(*self.arguments())
        following = self.store("aliases-02", cast={"characters": [], "roles": []}, parents=(first,))
        third = self.store("aliases-03", parents=(following,))
        self.assertEqual(following.canonical_id("voice-narrator"), "voice-one")
        self.assertEqual(third.load("voice-narrator")["asset_hash"], original["asset_hash"])
        before = self.snapshot(first.directory)
        candidate = self.candidate(following, voice_id="voice-narrator")
        following.select_candidate("voice-narrator", candidate["candidate_id"])
        following.persist_aliases()
        self.assertEqual(candidate["voice_id"], "voice-one")
        self.assertEqual(third.load("voice-narrator")["asset_hash"], candidate["asset_hash"])
        self.assertEqual(third.load("voice-one")["asset_hash"], candidate["asset_hash"])
        self.assertEqual(self.snapshot(first.directory), before)

    def test_explicit_alias_provenance_survives_later_book_without_configuration(self):
        first = self.store("explicit-01", cast=self.cast(other=True),
                           explicit_aliases={"voice-two": "voice-one"})
        original = first.publish(*self.arguments())
        first.persist_aliases()
        following = self.store("explicit-02", cast=self.cast(other=True), parents=(first,))
        self.assertEqual(following.canonical_id("voice-two"), "voice-one")
        self.assertEqual(following.load("voice-two")["asset_hash"], original["asset_hash"])
        following.persist_aliases()
        saved = read_json(following.directory / "voice-aliases.json")
        self.assertEqual(saved["explicit_aliases"], {"voice-two": "voice-one"})

    def test_matching_names_never_link_distinct_character_voices(self):
        first = self.store("names-01", cast=self.cast())
        first.publish(*self.arguments())
        following = self.store("names-02", cast=self.cast(other=True), parents=(first,))
        self.assertIsNone(following.load("voice-two"))
        self.assertEqual(following.canonical_id("voice-two"), "voice-two")

    def test_inherited_identity_collision_fails_even_without_saved_alias_file(self):
        first = self.store("conflict-01", cast=self.cast(alias=True))
        conflict = {"characters": [{"id": "different-person", "name": "Alex",
                                     "voice_id": "voice-narrator"}], "roles": []}
        with self.assertRaisesRegex(ValueError, "different character"):
            self.store("conflict-02", cast=conflict, parents=(first,))
        with self.assertRaisesRegex(ValueError, "canonical identity"):
            self.store("conflict-03", aliases={"voice-narrator": "voice-two"}, parents=(first,))

    def test_late_persisted_alias_is_resolved_on_existing_child_store(self):
        first = self.one.publish(*self.arguments())
        original = self.store("book-01", explicit_aliases={"voice-narrator": "voice-one"})
        original.persist_aliases()
        self.assertEqual(self.three.load("voice-narrator")["asset_hash"], first["asset_hash"])

    def test_origin_metadata_does_not_change_asset_hash_or_persisted_record(self):
        original = self.one.publish(*self.arguments())
        inherited = self.three.load("voice-one")
        record = read_json(self.one.folder("voice-one") / "voice.json")
        self.assertNotIn("inherited", record)
        self.assertNotIn("voice_directory", record)
        self.assertEqual(original["asset_hash"], digest(record))
        self.assertEqual(inherited["asset_hash"], digest(record))

    def test_lock_and_persisted_alias_writes_stay_in_child_directory(self):
        first = self.store("lock-01", cast=self.cast(alias=True))
        first.publish(*self.arguments())
        before = self.snapshot(first.directory)
        following = self.store("lock-02", parents=(first,))
        lock = following.lock()
        try:
            following.persist_aliases()
            self.assertEqual(lock.parent, following.directory)
            self.assertFalse((first.directory / ".lock").exists())
            self.assertTrue((following.directory / "voice-aliases.json").exists())
            self.assertEqual(following.list_candidates("voice-one"), [])
        finally:
            lock.rmdir()
        self.assertEqual(self.snapshot(first.directory), before)

    def test_recursive_and_flat_parent_lists_have_identical_precedence(self):
        self.one.publish(*self.arguments())
        second = self.two.publish(*self.arguments(text="Second book."))
        flat = self.store("book-flat", parents=(self.two, self.one))
        self.assertEqual(flat.parents, (self.two, self.one))
        self.assertEqual(flat.load("voice-one")["asset_hash"], second["asset_hash"])
        self.assertEqual(self.three.load("voice-one")["asset_hash"], second["asset_hash"])

    def test_wrong_series_and_self_inheritance_are_rejected(self):
        other = VoiceStore(self.root / "unrelated" / "voices", "other-series")
        with self.assertRaisesRegex(ValueError, "same series"):
            self.store("book-04", parents=(other,))
        with self.assertRaisesRegex(ValueError, "itself"):
            self.store("book-01", parents=(self.one,))


if __name__ == "__main__":
    unittest.main()
