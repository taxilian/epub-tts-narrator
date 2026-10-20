"""Editable book policy stays local while inherited voice lookup remains live."""

from pathlib import Path
import tempfile
import unittest

from audiobook_renderer.common import digest, read_json, write_json
from audiobook_renderer.voices import VoiceStore


class BookAliasStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sample = self.root / "sample.wav"
        self.sample.write_bytes(b"fixture-reference")

    @staticmethod
    def cast(*, narrator_target="a", include_narrator=True):
        return {
            "characters": [{"id": f"character-{value}", "name": "Alex",
                            "voice_id": value} for value in ("a", "b", "c")],
            "roles": ([{"id": "narrator", "voice_id": "narrator",
                        "character_id": f"character-{narrator_target}"}]
                      if include_narrator else []),
        }

    def store(self, name, policy, *, parents=(), cast=None, aliases=None):
        return VoiceStore(self.root / name / "voices", "series-one",
                          cast=self.cast() if cast is None else cast,
                          parents=parents, aliases=aliases,
                          alias_loader=lambda: dict(policy))

    def publish(self, store, voice_id):
        return store.publish(voice_id, {"description": "A clear voice."}, self.sample,
                             "A verified sample.", {"engine": "fixture"},
                             {"passed": True}, {"description_applied": True})

    def snapshot(self, store):
        return {str(path.relative_to(store.directory)): path.read_bytes()
                for path in store.directory.rglob("*") if path.is_file()}

    def test_parent_policy_added_after_child_creation_is_used_live(self):
        policy = {}
        first = self.store("one", policy)
        second = self.store("two", {}, parents=(first,))
        third = self.store("three", {}, parents=(second,))
        reference = self.publish(first, "b")
        self.assertIsNone(third.load("narrator"))
        policy["narrator"] = "b"
        self.assertEqual(third.canonical_id("narrator"), "b")
        self.assertEqual(third.load("narrator")["asset_hash"], reference["asset_hash"])
        self.assertFalse(second.directory.exists())
        self.assertFalse(third.directory.exists())

    def test_parent_explicit_policy_overrides_later_automatic_cast_edges(self):
        first = self.store("one", {"narrator": "b"})
        second = self.store("two", {}, parents=(first,))
        self.assertEqual(first.canonical_id("narrator"), "b")
        self.assertEqual(second.canonical_id("narrator"), "b")
        aliases, explicit, _ = second._alias_state()
        self.assertEqual(aliases["narrator"], explicit["narrator"])

    def test_nearest_explicit_override_affects_only_its_branch(self):
        first = self.store("one", {"narrator": "b"})
        second = self.store("two", {"narrator": "c"}, parents=(first,))
        third = self.store("three", {}, parents=(second,))
        sibling = self.store("sibling", {}, parents=(first,))
        b = self.publish(first, "b")
        c = self.publish(second, "c")
        before = self.snapshot(first)
        second.persist_aliases()
        self.assertEqual(first.load("narrator")["asset_hash"], b["asset_hash"])
        self.assertEqual(sibling.load("narrator")["asset_hash"], b["asset_hash"])
        self.assertEqual(second.load("narrator")["asset_hash"], c["asset_hash"])
        self.assertEqual(third.load("narrator")["asset_hash"], c["asset_hash"])
        self.assertEqual(self.snapshot(first), before)

    def test_child_cache_does_not_freeze_parent_policy(self):
        policy = {"narrator": "b"}
        first = self.store("one", policy)
        second = self.store("two", {}, parents=(first,))
        second.persist_aliases()
        saved = read_json(second.directory / "voice-aliases.json")
        self.assertEqual(saved["aliases"], {"narrator": "a"})
        self.assertEqual(saved["explicit_aliases"], {})
        policy["narrator"] = "c"
        self.assertEqual(second.canonical_id("narrator"), "c")
        policy.clear()
        self.assertEqual(second.canonical_id("narrator"), "a")
        self.assertEqual(read_json(second.directory / "voice-aliases.json"), saved)

    def test_local_policy_edit_and_removal_override_its_previous_cache(self):
        policy = {"narrator": "b"}
        first = self.store("one", policy)
        first.persist_aliases()
        policy["narrator"] = "c"
        self.assertEqual(first.canonical_id("narrator"), "c")
        first.persist_aliases()
        policy.clear()
        self.assertEqual(first.canonical_id("narrator"), "a")
        first.persist_aliases()
        saved = read_json(first.directory / "voice-aliases.json")
        self.assertEqual(saved["aliases"], {"narrator": "a"})
        self.assertEqual(saved["explicit_aliases"], {})

    def test_removing_local_override_restores_parent_policy(self):
        policy = {"narrator": "c"}
        first = self.store("one", {"narrator": "b"})
        second = self.store("two", policy, parents=(first,))
        second.persist_aliases()
        self.assertEqual(second.canonical_id("narrator"), "c")
        policy.clear()
        self.assertEqual(second.canonical_id("narrator"), "b")

    def test_old_inherited_snapshot_is_ignored_and_rewritten_as_local(self):
        first = self.store("one", {"narrator": "b"})
        empty_cast = {"characters": [], "roles": []}
        second = self.store("two", {}, parents=(first,), cast=empty_cast)
        second.directory.mkdir(parents=True)
        payload = {"format": "audiobook-voice-aliases", "version": 2,
                   "series_id": "series-one", "aliases": {"narrator": "c"},
                   "explicit_aliases": {"narrator": "c"},
                   "voice_owners": {"a": "character-a", "narrator": "character-a"}}
        write_json(second.directory / "voice-aliases.json", {**payload, "sha256": digest(payload)})
        self.assertEqual(second.canonical_id("narrator"), "b")
        second.persist_aliases()
        saved = read_json(second.directory / "voice-aliases.json")
        self.assertEqual(saved["aliases"], {})
        self.assertEqual(saved["explicit_aliases"], {})
        self.assertEqual(saved["voice_owners"], {})

    def test_current_explicit_decision_resolves_ancestor_automatic_conflict(self):
        empty_cast = {"characters": [], "roles": []}
        first = self.store("one", {}, cast=empty_cast, aliases={"narrator": "a"})
        second = self.store("two", {"narrator": "c"}, cast=empty_cast,
                            parents=(first,), aliases={"narrator": "b"})
        self.assertEqual(second.canonical_id("narrator"), "c")

    def test_unapproved_automatic_conflict_fails_closed(self):
        empty_cast = {"characters": [], "roles": []}
        first = self.store("one", {}, cast=empty_cast, aliases={"narrator": "a"})
        with self.assertRaisesRegex(ValueError, "canonical identity"):
            self.store("two", {}, cast=empty_cast, parents=(first,),
                       aliases={"narrator": "b"})

    def test_cycle_across_books_is_detected_including_late_edits(self):
        policy = {}
        first = self.store("one", policy)
        second = self.store("two", {"b": "a"}, parents=(first,))
        policy["a"] = "b"
        with self.assertRaisesRegex(ValueError, "cycles"):
            second.canonical_id("a")

    def test_explicit_override_can_remove_cycle_from_ancestor_edges(self):
        first = self.store("one", {"a": "b"})
        second = self.store("two", {"a": "c", "b": "a"}, parents=(first,))
        self.assertEqual(second.canonical_id("b"), "c")

    def test_alias_cache_checksum_still_checked_when_policy_is_authoritative(self):
        first = self.store("one", {})
        first.persist_aliases()
        path = first.directory / "voice-aliases.json"
        saved = read_json(path)
        saved["aliases"] = {"narrator": "b"}
        write_json(path, saved)
        with self.assertRaisesRegex(ValueError, "checksum"):
            first.canonical_id("narrator")

    def test_saved_ownership_collision_still_rejected(self):
        first = self.store("one", {})
        first.persist_aliases()
        different_cast = self.cast()
        different_cast["characters"][0]["id"] = "different-character"
        different_cast["roles"][0]["character_id"] = "different-character"
        with self.assertRaisesRegex(ValueError, "different character"):
            self.store("one", {}, cast=different_cast)

    def test_inherited_ownership_collision_still_rejected(self):
        first = self.store("one", {})
        different_cast = self.cast()
        different_cast["characters"][0]["id"] = "different-character"
        different_cast["roles"][0]["character_id"] = "different-character"
        with self.assertRaisesRegex(ValueError, "different character"):
            self.store("two", {}, parents=(first,), cast=different_cast)


if __name__ == "__main__":
    unittest.main()
