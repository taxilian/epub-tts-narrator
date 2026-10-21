"""Cleanup keeps resumable speech and all voice assets; deletion fails closed."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.cleanup import clean_take_intermediates, cleanup_work
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name) / "work"
        self.work.mkdir()

    def take(self, *, kind="speech", accepted=True, number=1, text="Hello there."):
        request = {"kind": kind, "text": text, "voice_asset_hash": "voice-identity"}
        directory = self.work / "clips" / digest(request)
        directory.mkdir(parents=True, exist_ok=True)
        wav = directory / f"attempt-{number:03d}.wav"
        wav.write_bytes(b"normalized speech bytes " + str(number).encode())
        raw = directory / f"attempt-{number:03d}-raw.wav"
        raw.write_bytes(b"redundant raw audio " * 10)
        record = {"request": request, "request_hash": digest(request), "attempt": number,
                  "audio_file": wav.name, "audio_sha256": file_hash(wav),
                  "report": {"passed": accepted}, "validation_key": "original policy"}
        write_json(directory / f"attempt-{number:03d}.json", record)
        if accepted:
            write_json(directory / "accepted.json", record)
        return directory, record

    def chapter(self, *, name="Chapter one", checkpoint=None, empty=False, context=None):
        identity = {"context": context or {"book": "a book", "voices": {"v": "asset"}},
                    "settings": {"chapter_format": "m4a"}, "chapter": {"title": name}}
        directory = self.work / "chapters" / digest(identity)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chapter.wav").write_bytes(b"chapter pcm")
        if not empty:
            (directory / "chapter.m4a").write_bytes(b"encoded chapter")
        record = {"format": "audiobook-renderer-chapter", "version": 1, "identity": identity,
                  "chapter": {"frames": 0 if empty else 100, "sample_rate": 44100},
                  "pcm_sha256": file_hash(directory / "chapter.wav"),
                  "encoded_sha256": None if empty else file_hash(directory / "chapter.m4a")}
        write_json(directory / "chapter.json", dict(record, record_sha256=digest(record)))
        if checkpoint:
            self.checkpoint(checkpoint, directory, identity["context"], empty=empty)
        return directory

    def checkpoint(self, name, directory, context, *, empty=False):
        record = {"format": "audiobook-chapter-checkpoint", "version": 1,
                  "cache_key": directory.name, "context": context, "acceptance": [],
                  "audio_file": None if empty else str(directory / "chapter.m4a")}
        path = self.work / "completed-chapters" / (name + ".json")
        write_json(path, dict(record, record_sha256=digest(record)))
        return path

    def snapshot(self):
        return {str(path.relative_to(self.work)): path.read_bytes()
                for path in self.work.rglob("*") if path.is_file() and not path.is_symlink()}

    def test_accepted_cleanup_preserves_exact_normalized_and_failed_takes(self):
        directory, accepted = self.take(number=2)
        _, failed = self.take(accepted=False, number=1)
        for suffix in ("-recovery-raw.wav", "-regenerated-0123456789abcdef.wav"):
            (directory / ("attempt-002" + suffix)).write_bytes(b"orphan winner source")
        # Unknown names, unfinished attempts, and normalized failures survive.
        for name in ("attempt-003-raw.wav", "debug.wav", "attempt-001-regenerated-nothex.wav"):
            (directory / name).write_bytes(b"keep")
        protected = {path.name: path.read_bytes() for path in directory.iterdir()
                     if path.suffix == ".json" or path.name in {
                         "attempt-001.wav", "attempt-002.wav", "attempt-003-raw.wav",
                         "debug.wav", "attempt-001-regenerated-nothex.wav"}}
        report = clean_take_intermediates(directory)
        self.assertEqual(report["files_removed"], 4)
        self.assertEqual(set(report["categories"]), {"speech_intermediates"})
        self.assertTrue(report["preserves_resumability"])
        self.assertEqual(protected, {path.name: path.read_bytes() for path in directory.iterdir()})
        self.assertEqual(read_json(directory / "accepted.json"), accepted)
        self.assertEqual(file_hash(directory / failed["audio_file"]), failed["audio_sha256"])

    def test_winner_acceptance_is_sufficient_without_attempt_record(self):
        directory, _ = self.take()
        (directory / "attempt-001.json").unlink()
        self.assertEqual(clean_take_intermediates(directory)["files_removed"], 1)
        self.assertTrue((directory / "accepted.json").is_file())

    def test_dry_run_is_exact_preview_and_repeat_is_noop(self):
        directory, _ = self.take()
        before = self.snapshot()
        preview = cleanup_work(self.work, dry_run=True)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(preview["files"], [{"path": f"clips/{directory.name}/attempt-001-raw.wav",
                                           "bytes": len(b"redundant raw audio " * 10),
                                           "category": "speech_intermediates"}])
        result = cleanup_work(self.work)
        self.assertEqual(preview["files"], result["files"])
        self.assertEqual(preview["bytes_removed"], result["bytes_removed"])
        self.assertEqual(cleanup_work(self.work)["files_removed"], 0)

    def test_never_rendered_work_is_noop_without_creating_it(self):
        missing = self.work / "not-yet-rendered"
        for discard in (False, True):
            for dry_run in (False, True):
                with self.subTest(discard=discard, dry_run=dry_run):
                    report = cleanup_work(missing, discard_render_cache=discard, dry_run=dry_run)
                    self.assertEqual(report["files_removed"], 0)
                    self.assertEqual(report["files"], [])
                    self.assertFalse(missing.exists())
        missing.write_bytes(b"not a directory")
        with self.assertRaises(ValueError):
            cleanup_work(missing)

    def test_work_cleanup_rejects_misplaced_request_but_standalone_helper_accepts_named_directory(self):
        directory, _ = self.take()
        misplaced = directory.with_name("0" * 64)
        directory.rename(misplaced)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "request identity"):
            cleanup_work(self.work)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(clean_take_intermediates(misplaced)["files_removed"], 1)

    def test_failed_and_pending_only_directories_are_left_untouched(self):
        directory, record = self.take(accepted=False)
        record["report"] = {"passed": False, "text_judge": {"decision": "unavailable"}}
        write_json(directory / "attempt-001.json", record)
        before = self.snapshot()
        self.assertEqual(clean_take_intermediates(directory)["files_removed"], 0)
        self.assertEqual(cleanup_work(self.work)["files_removed"], 0)
        self.assertEqual(before, self.snapshot())

    def test_reference_samples_are_never_pruned_even_in_clips(self):
        for kind in ("create_voice", "reference", "casting"):
            with self.subTest(kind=kind):
                directory, _ = self.take(kind=kind)
                before = self.snapshot()
                self.assertEqual(clean_take_intermediates(directory)["files_removed"], 0)
                self.assertEqual(before, self.snapshot())
        self.assertEqual(cleanup_work(self.work)["files_removed"], 0)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "voice/reference"):
            cleanup_work(self.work, discard_render_cache=True)
        self.assertEqual(before, self.snapshot())

    def test_all_voice_and_published_buckets_are_out_of_scope_in_both_modes(self):
        self.take()
        for folder in ("voices", "voice-candidates", "casting-candidates", "casting-drafts",
                       "reference-work", "sample-imports", "published", "runs"):
            path = self.work / folder / "voice" / "reference.wav"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"irreplaceable voice sample")
        (self.work / "status.json").write_text('{"status":"complete"}')
        protected = {key: value for key, value in self.snapshot().items() if not key.startswith("clips/")}
        cleanup_work(self.work)
        cleanup_work(self.work, discard_render_cache=True)
        self.assertEqual(self.snapshot(), protected)

    def test_request_acceptance_and_normalized_checksum_corruption_abort_before_deletion(self):
        directory, original = self.take(number=2)
        _, failed = self.take(accepted=False, number=1)
        for changes in ({"request_hash": "wrong"}, {"audio_sha256": "f" * 64},
                        {"audio_file": "../outside.wav"}, {"attempt": True},
                        {"request": None}, {"audio_sha256": None}):
            with self.subTest(changes=changes):
                write_json(directory / "accepted.json", {**original, **changes})
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    cleanup_work(self.work)
                self.assertEqual(before, self.snapshot())
        write_json(directory / "accepted.json", original)
        (directory / failed["audio_file"]).write_bytes(b"modified failure")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "checksum"):
            cleanup_work(self.work)
        self.assertEqual(before, self.snapshot())

    def test_missing_normalized_raw_is_retained_with_failed_generation_record(self):
        directory, accepted = self.take()
        record = dict(accepted, attempt=2, audio_file=None, audio_sha256=None,
                      report={"passed": False, "reasons": ["generation_error"]})
        write_json(directory / "attempt-002.json", record)
        raw = directory / "attempt-002-raw.wav"
        raw.write_bytes(b"potentially useful interrupted take")
        self.assertEqual(clean_take_intermediates(directory)["files_removed"], 1)
        self.assertTrue(raw.is_file())

    def test_wrong_attempt_request_aborts_instead_of_deleting_raw(self):
        directory, accepted = self.take(number=2)
        _, failed = self.take(accepted=False, number=1)
        failed["request"] = dict(failed["request"], kind="create_voice")
        failed["request_hash"] = digest(failed["request"])
        write_json(directory / "attempt-001.json", failed)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "reference"):
            cleanup_work(self.work)
        self.assertEqual(before, self.snapshot())

    def test_symlinked_raw_normalized_acceptance_and_attempt_metadata_are_rejected(self):
        directory, _ = self.take()
        for name in ("attempt-001-raw.wav", "attempt-001.wav", "accepted.json", "attempt-001.json"):
            with self.subTest(name=name):
                path = directory / name
                data = path.read_bytes()
                target = self.work / ("outside-" + name)
                target.write_bytes(data)
                path.unlink()
                path.symlink_to(target)
                with self.assertRaises(ValueError):
                    cleanup_work(self.work)
                self.assertTrue(path.is_symlink())
                self.assertEqual(target.read_bytes(), data)
                path.unlink()
                path.write_bytes(data)

    def test_symlinked_work_or_store_or_clip_directory_is_rejected(self):
        directory, _ = self.take()
        alias = self.work.parent / "alias"
        alias.symlink_to(self.work, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup_work(alias)
        alias.unlink()
        for path in (directory, self.work / "clips"):
            with self.subTest(path=path):
                target = self.work.parent / "relocated"
                path.rename(target)
                path.symlink_to(target, target_is_directory=True)
                with self.assertRaises(ValueError):
                    cleanup_work(self.work)
                path.unlink()
                target.rename(path)

    def test_every_checkpoint_protects_its_chapter_even_with_an_old_voice_context(self):
        current = self.chapter(checkpoint="current")
        old = self.chapter(name="Older voice version", checkpoint="old",
                           context={"voices": {"v": "old asset"}})
        obsolete = self.chapter(name="Unreferenced complete revision")
        empty = self.chapter(name="Empty unreferenced chapter", empty=True)
        protected = {key: value for key, value in self.snapshot().items()
                     if obsolete.name not in key and empty.name not in key}
        report = cleanup_work(self.work)
        self.assertEqual(report["files_removed"], 5)
        self.assertEqual(report["directories_removed"], 2)
        self.assertEqual(report["categories"]["obsolete_chapters"]["files_removed"], 5)
        self.assertFalse(obsolete.exists())
        self.assertFalse(empty.exists())
        self.assertTrue(current.is_dir())
        self.assertTrue(old.is_dir())
        self.assertEqual(protected, self.snapshot())

    def test_incomplete_chapters_and_staging_are_retained_by_default(self):
        for name in ("a" * 64, ".chapter-abcd1234"):
            folder = self.work / "chapters" / name
            folder.mkdir(parents=True)
            (folder / "chapter.wav").write_bytes(b"incomplete PCM")
        before = self.snapshot()
        self.assertEqual(cleanup_work(self.work)["files_removed"], 0)
        self.assertEqual(before, self.snapshot())

    def test_corrupt_or_missing_checkpoint_reference_blocks_all_cleanup(self):
        directory, _ = self.take()
        current = self.chapter(checkpoint="current")
        self.chapter(name="Obsolete")
        path = self.work / "completed-chapters" / "current.json"
        original = read_json(path)
        variants = []
        bad = deepcopy(original)
        bad["record_sha256"] = "0" * 64
        variants.append(bad)
        for changes in ({"cache_key": "0" * 64}, {"context": {"different": "context"}},
                        {"audio_file": "/some/other/place/chapter.m4a"}):
            bad = {key: value for key, value in {**original, **changes}.items() if key != "record_sha256"}
            variants.append(dict(bad, record_sha256=digest(bad)))
        for record in variants:
            with self.subTest(record=record):
                write_json(path, record)
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    cleanup_work(self.work)
                self.assertEqual(before, self.snapshot())
                self.assertTrue((directory / "attempt-001-raw.wav").is_file())
        write_json(path, original)
        (current / "chapter.m4a").unlink()
        with self.assertRaises((ValueError, OSError)):
            cleanup_work(self.work)
        self.assertTrue((directory / "attempt-001-raw.wav").is_file())

    def test_corrupt_obsolete_chapter_is_preserved_and_prevents_partial_cleanup(self):
        self.take()
        obsolete = self.chapter()
        (obsolete / "chapter.wav").write_bytes(b"altered")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "checksum"):
            cleanup_work(self.work)
        self.assertEqual(before, self.snapshot())

    def test_default_keeps_simultaneous_mixes(self):
        folder = self.work / "mixes" / ("f" * 64)
        folder.mkdir(parents=True)
        (folder / "mix.wav").write_bytes(b"simultaneous mix")
        before = self.snapshot()
        self.assertEqual(cleanup_work(self.work)["files_removed"], 0)
        self.assertEqual(before, self.snapshot())

    def test_finder_metadata_does_not_block_cleanup_and_is_deleted_only_with_its_cache(self):
        take, _ = self.take()
        current = self.chapter(checkpoint="current")
        obsolete = self.chapter(name="Obsolete")
        mixes = self.work / "mixes"
        mixes.mkdir()
        directories = [self.work, self.work / "clips", self.work / "chapters", mixes,
                       self.work / "completed-chapters", take, current, obsolete]
        for path in directories:
            (path / ".DS_Store").write_bytes(b"Finder metadata")
        preview = cleanup_work(self.work, dry_run=True)
        self.assertIn(f"chapters/{obsolete.name}/.DS_Store", [item["path"] for item in preview["files"]])
        cleanup_work(self.work)
        self.assertFalse(obsolete.exists())
        for path in directories[:-1]:
            self.assertTrue((path / ".DS_Store").is_file())
        cleanup_work(self.work, discard_render_cache=True)
        self.assertEqual(list(self.work.iterdir()), [self.work / ".DS_Store"])

    def test_symlinked_finder_metadata_is_not_followed_or_deleted(self):
        target = self.work / "voice.wav"
        target.write_bytes(b"protected reference")
        obsolete = self.chapter()
        link = obsolete / ".DS_Store"
        link.symlink_to(target)
        for discard in (False, True):
            with self.subTest(discard=discard), self.assertRaises(ValueError):
                cleanup_work(self.work, discard_render_cache=discard)
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_bytes(), b"protected reference")

    def test_explicit_discard_removes_only_four_stores_including_failed_speech(self):
        self.take(accepted=False)
        self.chapter(checkpoint="current")
        mixture = self.work / "mixes" / ("a" * 64)
        mixture.mkdir(parents=True)
        (mixture / "mix.wav").write_bytes(b"mix")
        (mixture / "mix.json").write_text("{}")
        partial = self.work / "chapters" / ".chapter-abcd1234" / "ready"
        partial.mkdir(parents=True)
        (partial / "normalized.wav").write_bytes(b"partial staging file")
        before = self.snapshot()
        preview = cleanup_work(self.work, discard_render_cache=True, dry_run=True)
        self.assertFalse(preview["preserves_resumability"])
        self.assertEqual(before, self.snapshot())
        result = cleanup_work(self.work, discard_render_cache=True)
        self.assertEqual(preview["files"], result["files"])
        self.assertEqual(result["files_removed"], len(before))
        self.assertEqual(list(self.work.iterdir()), [])

    def test_explicit_discard_refuses_unknown_files_before_deleting_anything(self):
        directory, _ = self.take()
        foreign = directory / "character-reference.wav"
        foreign.write_bytes(b"voice sample that must not be deleted")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            cleanup_work(self.work, discard_render_cache=True)
        self.assertEqual(before, self.snapshot())

    def test_explicit_discard_refuses_symlinks_in_every_store(self):
        target = self.work / "voices" / "reference.wav"
        target.parent.mkdir()
        target.write_bytes(b"protected voice")
        for store, relative in (("clips", "a" * 64 + "/attempt-001-raw.wav"),
                                ("chapters", "a" * 64 + "/chapter.wav"),
                                ("mixes", "a" * 64 + "/mix.wav"),
                                ("completed-chapters", "chapter.json")):
            with self.subTest(store=store):
                path = self.work / store / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(target)
                with self.assertRaises(ValueError):
                    cleanup_work(self.work, discard_render_cache=True)
                self.assertTrue(path.is_symlink())
                self.assertEqual(target.read_bytes(), b"protected voice")
                path.unlink()


if __name__ == "__main__":
    unittest.main()
