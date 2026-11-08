"""Voice auditions preserve every reference and require explicit selection."""

from pathlib import Path
import os
import tempfile
import unittest
import uuid

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.voices import VoiceStore


class VoiceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.store = VoiceStore(self.root / "voices", "series-one")
        self.wav = self.root / "sample.wav"
        self.wav.write_bytes(b"verified-waveform-fixture")
        self.args = ("voice-one", {"name": "Hero"}, self.wav, "The exact sample.",
                     {"engine": "fixture", "model": "fixture"}, {"passed": True},
                     {"description_applied": False})

    def candidate(self):
        return self.store.publish_candidate(*self.args, candidate_id=uuid.uuid4().hex)

    def test_new_reference_records_descriptive_name_and_round_trips(self):
        # Given a verified voice owned by a named character.
        # When publishing the real immutable voice store artifact.
        binding = self.store.publish(*self.args)

        # Then its recorded path identifies the owner and remains checksum-bound.
        reference = Path(binding["reference_audio"])
        self.assertEqual(reference.name, "voice-one-Hero-reference.wav")
        record = read_json(reference.parent / "voice.json")
        self.assertEqual(record["reference_file"], reference.name)
        self.assertEqual(record["reference_sha256"], self.store.load("voice-one")["reference_sha256"])
        self.assertEqual(binding["asset_hash"], digest(record))

    def test_legacy_record_without_reference_file_uses_existing_generic_asset(self):
        # Given an old v1 record whose only audio is its established reference.wav.
        folder = self.store.folder("voice-legacy")
        folder.mkdir(parents=True)
        audio = folder / "reference.wav"
        audio.write_bytes(b"old verified waveform")
        record = {"format": "audiobook-voice", "version": 1, "kind": "reference",
                  "series_id": "series-one", "voice_id": "voice-legacy", "profile": {},
                  "reference_text": "Old reference.", "reference_sha256": file_hash(audio),
                  "created_with": {"engine": "old"}, "creation": {},
                  "verification": {"passed": True}, "description_applied": False}
        write_json(folder / "voice.json", record)
        (folder / "voice.sha256").write_text(file_hash(folder / "voice.json") + "\n")

        # When loading it through the public store.
        binding = self.store.load("voice-legacy")

        # Then no rename, regeneration, or metadata migration is needed.
        self.assertEqual(Path(binding["reference_audio"]), audio)
        self.assertEqual(read_json(folder / "voice.json"), record)

    def test_candidate_is_inactive_until_selected(self):
        candidate = self.candidate()
        self.assertIsNone(self.store.load("voice-one"))
        listed = self.store.list_candidates("voice-one")
        self.assertEqual(len(listed), 1)
        self.assertFalse(listed[0]["selected"])
        selected = self.store.select_candidate("voice-one", candidate["candidate_id"])
        self.assertEqual(selected, candidate)
        self.assertEqual(self.store.load("voice-one"), candidate)
        self.assertTrue(self.store.list_candidates("voice-one")[0]["selected"])

    def test_replace_and_restore_keeps_every_reference_and_legacy_hash(self):
        original = self.store.publish(*self.args)
        original_record = read_json(self.store.folder("voice-one") / "voice.json")
        self.assertEqual(original["asset_hash"], digest(original_record))
        original_files = {
            path.name: path.read_bytes() for path in self.store.folder("voice-one").iterdir()
        }
        candidate = self.candidate()
        self.assertEqual(self.store.load("voice-one"), original)
        self.store.select_candidate("voice-one", candidate["candidate_id"],
                                    expected_asset_hash=original["asset_hash"])
        self.assertEqual(self.store.load("voice-one"), candidate)
        restored = self.store.select_candidate(
            "voice-one", "legacy", expected_asset_hash=candidate["asset_hash"])
        self.assertEqual(restored, original)
        self.assertEqual(restored["asset_hash"], digest(original_record))
        for filename, data in original_files.items():
            self.assertEqual((self.store.folder("voice-one") / filename).read_bytes(), data)
        self.assertTrue(Path(candidate["reference_audio"]).exists())
        self.assertEqual([item["candidate_id"] for item in
                          self.store.list_candidates("voice-one") if item["selected"]],
                         ["legacy"])

    def test_legacy_publish_refuses_selected_candidate_without_legacy_folder(self):
        candidate = self.candidate()
        self.store.select_candidate("voice-one", candidate["candidate_id"])
        with self.assertRaisesRegex(ValueError, "never implicitly replaced"):
            self.store.publish(*self.args)
        self.assertFalse(self.store.folder("voice-one").exists())

    def test_candidate_id_cannot_traverse_or_overwrite(self):
        for invalid in ("../outside", "", "legacy", "0" * 31, "G" * 32, "A" * 32, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Candidate ID"):
                self.store.publish_candidate(*self.args, candidate_id=invalid)
        candidate = self.candidate()
        with self.assertRaisesRegex(ValueError, "never replaced"):
            self.store.publish_candidate(*self.args, candidate_id=candidate["candidate_id"])
        with self.assertRaisesRegex(ValueError, "Candidate ID"):
            self.store.load_candidate("voice-one", "../outside")

    def test_failed_verification_does_not_publish_candidate(self):
        args = list(self.args)
        args[5] = {"passed": False}
        with self.assertRaisesRegex(ValueError, "Only a verified"):
            self.store.publish_candidate(*args, candidate_id=uuid.uuid4().hex)
        self.assertEqual(self.store.list_candidates("voice-one"), [])

    def test_audio_and_metadata_tampering_block_read_and_selection(self):
        candidate = self.candidate()
        wav = Path(candidate["reference_audio"])
        wav.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "reference changed"):
            self.store.load_candidate("voice-one", candidate["candidate_id"])
        with self.assertRaisesRegex(ValueError, "reference changed"):
            self.store.select_candidate("voice-one", candidate["candidate_id"])
        wav.write_bytes(self.wav.read_bytes())
        path = wav.parent / "voice.json"
        record = read_json(path)
        record["reference_text"] = "Another transcript."
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            self.store.list_candidates("voice-one")

    def test_invalid_pointer_never_falls_back_to_legacy(self):
        self.store.publish(*self.args)
        candidate = self.candidate()
        self.store.select_candidate("voice-one", candidate["candidate_id"])
        pointer = self.store._selection_path("voice-one")
        record = read_json(pointer)
        record["candidate_id"] = "legacy"
        write_json(pointer, record)
        for action in (lambda: self.store.load("voice-one"),
                       lambda: self.store.list_candidates("voice-one"),
                       lambda: self.store.select_candidate("voice-one", "legacy")):
            with self.assertRaisesRegex(ValueError, "selection changed"):
                action()

    def test_missing_selected_candidate_never_falls_back_to_legacy(self):
        self.store.publish(*self.args)
        candidate = self.candidate()
        self.store.select_candidate("voice-one", candidate["candidate_id"])
        folder = Path(candidate["reference_audio"]).parent
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.store.load("voice-one")

    def test_outdated_selection_cannot_replace_newer_choice(self):
        original = self.store.publish(*self.args)
        first, second = self.candidate(), self.candidate()
        self.store.select_candidate("voice-one", first["candidate_id"],
                                    expected_asset_hash=original["asset_hash"])
        with self.assertRaisesRegex(ValueError, "Active voice changed"):
            self.store.select_candidate("voice-one", second["candidate_id"],
                                        expected_asset_hash=original["asset_hash"])
        self.assertEqual(self.store.load("voice-one"), first)

    def test_symlinks_cannot_redirect_voice_files_or_directories(self):
        candidate = self.candidate()
        folder = Path(candidate["reference_audio"]).parent
        for name in ("voice.json", "voice.sha256", Path(candidate["reference_audio"]).name):
            with self.subTest(name=name):
                path = folder / name
                data = path.read_bytes()
                target = self.root / ("external-" + name)
                target.write_bytes(data)
                path.unlink()
                path.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "nonsymlink"):
                    self.store.load_candidate("voice-one", candidate["candidate_id"])
                path.unlink()
                path.write_bytes(data)
        moved = folder.with_name("moved")
        folder.rename(moved)
        folder.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be symlinks"):
            self.store.load_candidate("voice-one", candidate["candidate_id"])

    def test_selection_file_symlink_and_dangling_link_fail_closed(self):
        candidate = self.candidate()
        self.store.select_candidate("voice-one", candidate["candidate_id"])
        pointer = self.store._selection_path("voice-one")
        target = self.root / "external-selection.json"
        pointer.rename(target)
        pointer.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "nonsymlink"):
            self.store.load("voice-one")
        target.unlink()
        with self.assertRaisesRegex(ValueError, "nonsymlink"):
            self.store.load("voice-one")

    def test_legacy_asset_hash_is_unchanged_when_returned_binding_gains_candidate_id(self):
        binding = self.store.publish(*self.args)
        record = read_json(self.store.folder("voice-one") / "voice.json")
        self.assertNotIn("candidate_id", record)
        self.assertEqual(binding["candidate_id"], "legacy")
        self.assertEqual(binding["asset_hash"], digest(record))
        self.assertEqual(file_hash(Path(binding["reference_audio"])),
                         record["reference_sha256"])

    def test_candidate_history_follows_creation_time_not_random_uuid_order(self):
        self.store.publish(*self.args)
        first_args = list(self.args)
        first_args[-1] = {"created_at": "2026-10-20T12:00:00+00:00"}
        first = self.store.publish_candidate(*first_args, candidate_id="f" * 32)
        second_args = list(self.args)
        second_args[-1] = {"created_at": "2026-10-20T13:00:00+00:00"}
        second = self.store.publish_candidate(*second_args, candidate_id="0" * 32)
        # Copying/restoring files can change mtimes; explicit creation wins.
        os.utime(Path(first["reference_audio"]).parent / "voice.json",
                 ns=(2_000_000_000_000_000_000,) * 2)
        os.utime(Path(second["reference_audio"]).parent / "voice.json",
                 ns=(1_000_000_000_000_000_000,) * 2)
        self.assertEqual([row["candidate_id"] for row in self.store.list_candidates("voice-one")],
                         ["legacy", "f" * 32, "0" * 32])

    def test_candidate_history_without_creation_timestamp_uses_publication_mtime(self):
        first = self.store.publish_candidate(*self.args, candidate_id="f" * 32)
        second = self.store.publish_candidate(*self.args, candidate_id="0" * 32)
        os.utime(Path(first["reference_audio"]).parent / "voice.json",
                 ns=(1_000_000_000_000_000_000,) * 2)
        os.utime(Path(second["reference_audio"]).parent / "voice.json",
                 ns=(2_000_000_000_000_000_000,) * 2)
        self.assertEqual([row["candidate_id"] for row in self.store.list_candidates("voice-one")],
                         ["f" * 32, "0" * 32])


if __name__ == "__main__":
    unittest.main()
