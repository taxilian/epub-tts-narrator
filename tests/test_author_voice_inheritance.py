"""Older per-book author voice IDs inherit without losing local overrides."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.book_aliases import alias_record
from audiobook_renderer.book_context import voice_store
from audiobook_renderer.common import digest, read_json, write_json
from audiobook_renderer.voices import VoiceStore


class AuthorVoiceInheritanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.wav = self.root / "sample.wav"
        self.wav.write_bytes(b"verified fixture audio")
        self.books = [self.book(index) for index in range(1, 7)]

    def book(self, number, *, author="Nathan Thompson", voice_id=None):
        voice_id = voice_id or f"author-{number}"
        return {"directory": str(self.root / f"book-{number}"),
                "source": {"book": {"id": f"book-{number}", "authors": [author]}},
                "cast": {"characters": [],
                         "voices": [{"id": voice_id}, {"id": f"narrator-{number}"}],
                         "roles": [
                             {"id": "narrator_main", "kind": "narration", "character_id": None,
                              "voice_id": f"narrator-{number}", "description": "Story narration."},
                             {"id": "author_afterword", "kind": "narration", "character_id": None,
                              "voice_id": voice_id, "description": "Author afterword and acknowledgments."}]}}

    def config(self, index):
        return {"voices": {"series_id": "one-series", "directory": str(Path(self.books[index]["directory"]) / "voices")},
                "_voice_parents": [book["directory"] for book in reversed(self.books[:index])]}

    def store(self, index, *, local_aliases=None):
        books = {book["directory"]: book for book in self.books}
        with patch("audiobook_renderer.bundle.load_bundle", side_effect=lambda path: books[str(path)]):
            return voice_store(self.books[index], self.config(index), local_aliases=local_aliases)

    def raw_store(self, index):
        return VoiceStore(Path(self.books[index]["directory"]) / "voices", "one-series",
                          cast=self.books[index]["cast"])

    def publish(self, store, voice_id, text="A saved author voice."):
        return store.publish(voice_id, {"description": "An audiobook author voice."}, self.wav, text,
                             {"engine": "fixture"}, {"passed": True}, {"description_applied": True})

    def snapshot(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def test_old_books_keep_local_voices_and_missing_books_inherit_nearest(self):
        original = [self.publish(self.raw_store(index), f"author-{index + 1}", f"Book {index + 1} voice.")
                    for index in range(4)]
        before = self.snapshot()
        for index in range(6):
            store = self.store(index)
            binding = store.load(f"author-{index + 1}")
            expected = original[min(index, 3)]
            self.assertEqual(binding["asset_hash"], expected["asset_hash"])
            self.assertEqual(binding["inherited"], index >= 4)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((Path(self.books[4]["directory"]) / "voices").exists())

    def test_ancestor_created_after_lookup_is_available_without_reopening(self):
        child = self.store(2)
        self.assertIsNone(child.load("author-3"))
        first = self.publish(self.store(0), "author-1")
        before = self.snapshot()
        inherited = child.load("author-3")
        self.assertEqual(inherited["asset_hash"], first["asset_hash"])
        self.assertTrue(inherited["inherited"])
        self.assertEqual(self.snapshot(), before)

    def test_local_regeneration_under_canonical_id_is_seen_by_descendants(self):
        original = self.publish(self.raw_store(0), "author-1")
        second = self.store(1)
        replacement = self.publish(second, "author-2", "A later-book override.")
        self.assertEqual(replacement["voice_id"], "author-1")
        self.assertEqual(self.store(2).load("author-3")["asset_hash"], replacement["asset_hash"])
        self.assertEqual(self.store(0).load("author-1")["asset_hash"], original["asset_hash"])

    def test_existing_selection_or_failed_marker_prevents_old_voice_fallback(self):
        self.publish(self.raw_store(0), "author-1")
        second = self.raw_store(1)
        second.invalidate("author-2")
        for state in ("pending", "failed"):
            if state == "failed":
                second.mark_regeneration_failed("author-2", "The replacement failed.")
            current, later = self.store(1), self.store(2)
            self.assertEqual(current.canonical_id("author-2"), "author-2")
            self.assertIsNone(current.load("author-2"))
            self.assertIsNone(later.load("author-3"))
            marker = later.regeneration_status("author-3")
            self.assertEqual(marker["state"], state)
            self.assertEqual(marker["voice_directory"], str(second.directory))

    def test_local_canonical_failure_also_blocks_ancestors(self):
        self.publish(self.raw_store(0), "author-1")
        second = self.store(1)
        second.invalidate("author-2")
        second.mark_regeneration_failed("author-2", "Failed under inherited identity.")
        self.assertIsNone(self.store(2).load("author-3"))
        self.assertEqual(self.store(2).regeneration_status("author-3")["state"], "failed")

    def test_local_binding_for_repeated_id_overrides_older_automatic_alias(self):
        self.books[2] = self.book(3, voice_id="author-2")
        self.publish(self.raw_store(0), "author-1")
        replacement = self.publish(self.raw_store(2), "author-2", "Third book's local voice.")
        store = self.store(2)
        self.assertEqual(store.canonical_id("author-2"), "author-2")
        self.assertEqual(store.load("author-2")["asset_hash"], replacement["asset_hash"])

    def test_explicit_policy_preserves_local_ancestor_id_binding_and_failed_marker(self):
        self.publish(self.raw_store(0), "author-1")
        third = self.raw_store(2)
        local = self.publish(third, "author-2", "Local override stored under an ancestor ID.")
        book = self.books[2]
        path = Path(book["directory"]) / "voice-aliases.json"
        write_json(path, alias_record(book, self.config(2), {"author-3": "author-2"}))
        following = self.store(3)
        self.assertEqual(following.canonical_id("author-4"), "author-2")
        self.assertEqual(following.load("author-4")["asset_hash"], local["asset_hash"])
        third.invalidate("author-2")
        third.mark_regeneration_failed("author-2", "The local replacement failed.")
        following = self.store(3)
        self.assertIsNone(following.load("author-4"))
        self.assertEqual(following.regeneration_status("author-4")["state"], "failed")

    def test_explicit_local_policy_wins_and_remains_live(self):
        self.publish(self.raw_store(0), "author-1")
        alternative = self.publish(self.raw_store(1), "narrator-2", "Explicit alternative.")
        local = {"author-2": "narrator-2"}
        book = self.books[1]
        path = Path(book["directory"]) / "voice-aliases.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, alias_record(book, self.config(1), local))
        store = self.store(2)
        self.assertEqual(store.load("author-3")["asset_hash"], alternative["asset_hash"])
        write_json(path, alias_record(book, self.config(1), {}))
        self.assertEqual(store.canonical_id("author-3"), "author-1")
        override = self.store(1, local_aliases=local)
        self.assertEqual(override.load("author-2")["asset_hash"], alternative["asset_hash"])

    def test_persisted_alias_cache_does_not_freeze_old_derived_edges(self):
        self.publish(self.raw_store(0), "author-1")
        prior = self.store(1)
        prior.persist_aliases()
        self.assertFalse((prior.directory / "voice-aliases.json").exists())
        # An older cached automatic alias is metadata, never live book policy.
        record = {"format": "audiobook-voice-aliases", "version": 2, "series_id": "one-series",
                  "aliases": {"author-2": "author-1"}, "explicit_aliases": {}, "voice_owners": {}}
        prior.directory.mkdir(parents=True, exist_ok=True)
        write_json(prior.directory / "voice-aliases.json", {**record, "sha256": digest(record)})
        # A pre-existing/newly imported local binding has explicit precedence
        # when the next invocation derives the live book policy.
        fresh_raw = VoiceStore(Path(self.books[1]["directory"]) / "voices", "one-series",
                               cast=self.books[1]["cast"], alias_loader=lambda: {})
        local = self.publish(fresh_raw, "author-2", "Local identity retained.")
        later = self.store(2)
        self.assertEqual(later.canonical_id("author-3"), "author-2")
        self.assertEqual(later.load("author-3")["asset_hash"], local["asset_hash"])

    def test_explicit_reverse_alias_overrides_inference_without_a_cycle(self):
        self.publish(self.raw_store(0), "author-1")
        policy = {"author-1": "author-2"}
        store = self.store(1, local_aliases=policy)
        self.assertEqual(store.canonical_id("author-1"), "author-2")
        self.assertEqual(store.canonical_id("author-2"), "author-2")
        self.assertIsNone(store.load("author-2"))
        local = self.publish(store, "author-2", "An explicitly selected replacement identity.")
        self.assertEqual(local["voice_id"], "author-2")

    def test_open_store_recomputes_inference_after_live_reverse_policy_edit(self):
        self.publish(self.raw_store(0), "author-1")
        store = self.store(2)
        self.assertEqual(store.canonical_id("author-3"), "author-1")
        book = self.books[2]
        path = Path(book["directory"]) / "voice-aliases.json"
        path.parent.mkdir(parents=True)
        write_json(path, alias_record(book, self.config(2), {"author-1": "author-3"}))
        self.assertEqual(store.canonical_id("author-1"), "author-3")
        self.assertEqual(store.canonical_id("author-3"), "author-3")
        self.assertIsNone(store.load("author-3"))
        path.unlink()
        self.assertEqual(store.canonical_id("author-3"), "author-1")

    def test_open_store_notices_later_local_ancestor_identity_override(self):
        original = self.publish(self.raw_store(0), "author-1")
        store = self.store(2)
        self.assertEqual(store.load("author-3")["asset_hash"], original["asset_hash"])
        replacement = self.publish(self.raw_store(1), "author-2", "New local ancestor identity.")
        self.assertEqual(store.load("author-3")["asset_hash"], replacement["asset_hash"])

    def test_automatic_loader_is_validated_and_runs_once_per_lookup(self):
        for loader in (True, {}, "not-callable"):
            with self.subTest(loader=loader), self.assertRaisesRegex(ValueError, "must be callable"):
                VoiceStore(self.root / "invalid", "one-series", automatic_alias_loader=loader)
        calls = []
        def loader(existing):
            calls.append(existing)
            return {"new": "old"}
        store = VoiceStore(self.root / "dynamic", "one-series", automatic_alias_loader=loader,
                           aliases={"explicit-source": "old"})
        calls.clear()
        self.assertEqual(store.canonical_id("new"), "old")
        self.assertEqual(calls, [{"explicit-source": "old"}])
        with self.assertRaisesRegex(ValueError, "Voice aliases must"):
            VoiceStore(self.root / "bad-result", "one-series", automatic_alias_loader=lambda existing: None)

    def test_different_author_generic_narrator_and_character_are_not_aliased(self):
        self.publish(self.raw_store(0), "author-1")
        for change in ("different_author", "generic_narrator", "main_narrator", "character"):
            with self.subTest(change=change):
                book = deepcopy(self.books[1])
                role = book["cast"]["roles"][1]
                if change == "different_author":
                    book["source"]["book"]["authors"] = ["Another Author"]
                elif change == "generic_narrator":
                    role["description"] = "An unidentified narrator."
                elif change == "main_narrator":
                    book["cast"]["roles"][0]["voice_id"] = "author-2"
                else:
                    book["cast"]["characters"] = [{"id": "char", "voice_id": "author-2"}]
                original = self.books[1]
                self.books[1] = book
                try:
                    store = self.store(1)
                    self.assertEqual(store.canonical_id("author-2"), "author-2")
                    self.assertIsNone(store.load("author-2"))
                finally:
                    self.books[1] = original

    def test_ambiguous_author_voices_block_automatic_older_choice(self):
        self.publish(self.raw_store(0), "author-1")
        book = self.books[1]
        other = deepcopy(book["cast"]["roles"][1])
        other.update(id="author_note", voice_id="author-note-2")
        book["cast"]["roles"].append(other)
        book["cast"]["voices"].append({"id": "author-note-2"})
        self.assertEqual(self.store(2).canonical_id("author-3"), "author-3")
        self.assertIsNone(self.store(2).load("author-3"))

    def test_corrupt_local_selection_and_symlink_are_not_hidden_by_inheritance(self):
        self.publish(self.raw_store(0), "author-1")
        raw = self.raw_store(1)
        raw.invalidate("author-2")
        path = raw._selection_path("author-2")
        pointer = read_json(path)
        pointer["selection_sha256"] = "0" * 64
        write_json(path, pointer)
        with self.assertRaisesRegex(ValueError, "selection changed"):
            self.store(2)
        path.unlink()
        raw.folder("author-2").symlink_to(self.root / "outside")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.store(2)


if __name__ == "__main__":
    unittest.main()
