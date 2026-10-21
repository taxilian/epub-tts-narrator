"""Recasting clears effective voices durably without altering reference history."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from epub_tts_narrator.renderer.common import digest, read_json, write_json
from epub_tts_narrator.renderer.voices import VoiceStore


class VoiceInvalidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.wav = self.root / "reference.wav"
        self.wav.write_bytes(b"verified-reference-fixture")
        self.one = self.store("one")
        self.two = self.store("two", parents=(self.one,))
        self.three = self.store("three", parents=(self.two,))

    def store(self, name, **kwargs):
        return VoiceStore(self.root / name / "voices", "series", **kwargs)

    def arguments(self, voice_id="voice-one", text="The sample text."):
        return (voice_id, {"name": "Hero"}, self.wav, text, {"engine": "fixture"},
                {"passed": True}, {"description_applied": True})

    def candidate(self, store, voice_id="voice-one"):
        return store.publish_candidate(*self.arguments(voice_id), candidate_id=uuid.uuid4().hex)

    @staticmethod
    def snapshot(path):
        return {str(item.relative_to(path)): item.read_bytes()
                for item in path.rglob("*") if item.is_file()}

    def test_clear_legacy_and_selected_revision_preserves_history(self):
        legacy = self.one.publish(*self.arguments())
        first = self.candidate(self.one)
        self.one.select_candidate("voice-one", first["candidate_id"])
        second = self.candidate(self.one)
        history = {item["candidate_id"]: self.snapshot(Path(item["reference_audio"]).parent)
                   for item in (legacy, first, second)}

        status = self.one.invalidate("voice-one")

        self.assertEqual(status["state"], "pending")
        self.assertIsNone(status["error"])
        self.assertFalse(status["inherited"])
        self.assertEqual(status["voice_directory"], str(self.one.directory))
        self.assertEqual(set(status["excluded_candidate_ids"]),
                         {"legacy", first["candidate_id"], second["candidate_id"]})
        self.assertIsNone(self.one.load("voice-one"))
        for candidate in self.one.list_candidates("voice-one"):
            self.assertFalse(candidate["selected"])
            self.assertEqual(self.snapshot(Path(candidate["reference_audio"]).parent),
                             history[candidate["candidate_id"]])

    def test_failed_recast_survives_restart_and_blocks_descendant_fallback(self):
        original = self.one.publish(*self.arguments())
        ancestor_before = self.snapshot(self.one.directory)
        self.two.invalidate("voice-one")
        self.two.mark_regeneration_failed("voice-one", "Provider returned HTTP 400: rejected")
        reopened = self.store("two", parents=(self.one,))
        descendant = self.store("three", parents=(reopened,))
        for store in (reopened, descendant):
            self.assertIsNone(store.load("voice-one"))
            status = store.regeneration_status("voice-one")
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["error"], "Provider returned HTTP 400: rejected")
            self.assertEqual(status["inherited"], store is descendant)
            self.assertEqual(status["voice_directory"], str(self.two.directory))
        self.assertEqual(self.one.load("voice-one")["asset_hash"], original["asset_hash"])
        self.assertEqual(self.snapshot(self.one.directory), ancestor_before)

    def test_pending_recast_blocks_legacy_and_ancestors_through_recursive_chain(self):
        self.one.publish(*self.arguments())
        local = self.two.publish(*self.arguments(text="Local override."))
        self.two.invalidate("voice-one")
        for store in (self.two, self.three):
            self.assertIsNone(store.load("voice-one"))
            self.assertEqual(store.regeneration_status("voice-one")["state"], "pending")
        self.assertEqual(self.two.load_candidate("voice-one", "legacy")["asset_hash"],
                         local["asset_hash"])

    def test_successful_selection_atomically_clears_failed_state(self):
        self.one.publish(*self.arguments())
        self.one.invalidate("voice-one")
        self.one.mark_regeneration_failed("voice-one", "No speech was generated")
        fresh = self.candidate(self.one)
        self.assertIsNone(self.one.load("voice-one"))
        self.assertEqual(self.one.regeneration_status("voice-one")["state"], "failed")

        self.one.select_candidate("voice-one", fresh["candidate_id"])

        self.assertIsNone(self.one.regeneration_status("voice-one"))
        self.assertEqual(self.one.load("voice-one")["asset_hash"], fresh["asset_hash"])
        self.assertIsNone(self.three.regeneration_status("voice-one"))
        self.assertEqual(self.three.load("voice-one")["asset_hash"], fresh["asset_hash"])
        pointer = read_json(self.one._selection_path("voice-one"))
        self.assertEqual(pointer["version"], 2)
        self.assertEqual(pointer["state"], "selected")
        self.assertNotIn("error", pointer)
        self.assertNotIn("excluded_candidate_ids", pointer)

    def test_explicit_historical_selection_can_restore_old_reference(self):
        old = self.one.publish(*self.arguments())
        self.one.invalidate("voice-one")
        self.one.mark_regeneration_failed("voice-one", "Rejected")
        self.one.select_candidate("voice-one", "legacy")
        self.assertEqual(self.one.load("voice-one"), old)
        self.assertIsNone(self.one.regeneration_status("voice-one"))

    def test_newer_book_voice_shadows_ancestor_failure(self):
        original = self.one.publish(*self.arguments())
        fresh = self.candidate(self.three)
        self.three.select_candidate("voice-one", fresh["candidate_id"])
        self.two.invalidate("voice-one")
        self.two.mark_regeneration_failed("voice-one", "Failed in book two")
        self.assertIsNone(self.three.regeneration_status("voice-one"))
        self.assertEqual(self.three.load("voice-one")["asset_hash"], fresh["asset_hash"])
        self.assertEqual(self.one.load("voice-one")["asset_hash"], original["asset_hash"])

    def test_aliases_share_canonical_invalidation_and_replacement(self):
        self.two = self.store("two", parents=(self.one,),
                              aliases={"narrator": "voice-one"})
        self.one.publish(*self.arguments())
        status = self.two.invalidate("narrator")
        self.assertEqual(status["voice_id"], "voice-one")
        self.assertIsNone(self.two.load("voice-one"))
        self.two.mark_regeneration_failed("narrator", "Rejected")
        self.assertEqual(self.two.regeneration_status("voice-one")["state"], "failed")
        candidate = self.candidate(self.two, "narrator")
        self.two.select_candidate("narrator", candidate["candidate_id"])
        self.assertIsNone(self.two.regeneration_status("narrator"))
        self.assertEqual(self.two.load("voice-one"), self.two.load("narrator"))

    def test_second_invalidation_captures_new_candidates_and_resets_failure(self):
        self.one.publish(*self.arguments())
        self.one.invalidate("voice-one")
        self.one.mark_regeneration_failed("voice-one", "Rejected")
        candidate = self.candidate(self.one)
        self.assertNotIn(candidate["candidate_id"],
                         self.one.regeneration_status("voice-one")["excluded_candidate_ids"])
        status = self.one.invalidate("voice-one")
        self.assertEqual(status["state"], "pending")
        self.assertIsNone(status["error"])
        self.assertEqual(set(status["excluded_candidate_ids"]), {"legacy", candidate["candidate_id"]})

    def test_missing_voice_can_be_invalidated_before_first_generation(self):
        self.assertIsNone(self.one.regeneration_status("voice-one"))
        status = self.one.invalidate("voice-one")
        self.assertEqual(status["excluded_candidate_ids"], [])
        self.assertIsNone(self.one.load("voice-one"))
        self.one.mark_regeneration_failed("voice-one", "Could not create voice")
        self.assertEqual(self.one.regeneration_status("voice-one")["state"], "failed")

    def test_generic_publish_cannot_bypass_local_or_inherited_marker(self):
        self.two.invalidate("voice-one")
        for store in (self.two, self.three):
            with self.assertRaisesRegex(ValueError, "explicitly recast and select"):
                store.publish(*self.arguments())
            self.assertFalse(store.folder("voice-one").exists())

    def test_cannot_mark_failure_without_local_invalidation_or_nonempty_reason(self):
        self.one.invalidate("voice-one")
        ancestor_before = self.snapshot(self.one.directory)
        with self.assertRaisesRegex(ValueError, "invalidated locally"):
            self.two.mark_regeneration_failed("voice-one", "Failure")
        for invalid in ("", " ", None, {}, 4):
            with self.subTest(error=invalid), self.assertRaisesRegex(ValueError, "error message"):
                self.one.mark_regeneration_failed("voice-one", invalid)
        self.assertEqual(self.snapshot(self.one.directory), ancestor_before)

    def test_legacy_v1_selection_remains_readable_and_can_be_invalidated(self):
        candidate = self.candidate(self.one)
        self.one.select_candidate("voice-one", candidate["candidate_id"])
        path = self.one._selection_path("voice-one")
        pointer = read_json(path)
        del pointer["state"]
        del pointer["selection_sha256"]
        pointer["version"] = 1
        write_json(path, {**pointer, "selection_sha256": digest(pointer)})
        self.assertEqual(self.one.load("voice-one"), candidate)
        self.assertIsNone(self.one.regeneration_status("voice-one"))
        self.one.invalidate("voice-one")
        self.assertIsNone(self.one.load("voice-one"))

    def test_tampered_marker_fails_closed_for_local_and_descendants(self):
        self.one.publish(*self.arguments())
        self.two.invalidate("voice-one")
        path = self.two._selection_path("voice-one")
        pointer = read_json(path)
        pointer["error"] = "Changed without checksum"
        write_json(path, pointer)
        for store in (self.two, self.three):
            with self.assertRaisesRegex(ValueError, "selection changed"):
                store.load("voice-one")
            with self.assertRaisesRegex(ValueError, "selection changed"):
                store.regeneration_status("voice-one")

    def test_checksummed_malformed_markers_are_rejected(self):
        self.one.publish(*self.arguments())
        self.two.invalidate("voice-one")
        path = self.two._selection_path("voice-one")
        original = read_json(path)
        original.pop("selection_sha256")
        for changes in ({"state": "unknown"}, {"state": "failed", "error": None},
                        {"error": "Pending cannot have an error"},
                        {"excluded_candidate_ids": ["../escape"]},
                        {"excluded_candidate_ids": ["legacy", "legacy"]},
                        {"excluded_candidate_ids": {}},
                        {"excluded_candidate_ids": [5]}, {"version": True}):
            with self.subTest(changes=changes):
                pointer = {**original, **changes}
                write_json(path, {**pointer, "selection_sha256": digest(pointer)})
                with self.assertRaises(ValueError):
                    self.three.load("voice-one")

    def test_failed_atomic_selection_write_keeps_marker_and_history(self):
        old = self.one.publish(*self.arguments())
        self.one.invalidate("voice-one")
        self.one.mark_regeneration_failed("voice-one", "Rejected")
        fresh = self.candidate(self.one)
        with patch("epub_tts_narrator.renderer.voices.write_json", side_effect=OSError("Disk full")):
            with self.assertRaisesRegex(OSError, "Disk full"):
                self.one.select_candidate("voice-one", fresh["candidate_id"])
        self.assertIsNone(self.one.load("voice-one"))
        self.assertEqual(self.one.regeneration_status("voice-one")["state"], "failed")
        self.assertEqual(self.one.load_candidate("voice-one", "legacy"), old)

    def test_symlinked_marker_never_follows_or_updates_external_file(self):
        self.one.publish(*self.arguments())
        self.two.invalidate("voice-one")
        path = self.two._selection_path("voice-one")
        target = self.root / "external.json"
        path.rename(target)
        path.symlink_to(target)
        before = target.read_bytes()
        for action in (lambda: self.three.load("voice-one"),
                       lambda: self.two.invalidate("voice-one"),
                       lambda: self.two.mark_regeneration_failed("voice-one", "Failure")):
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                action()
        self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
