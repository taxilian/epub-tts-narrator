"""prune keeps what a book uses (current voices, newest chapters, reusable takes) and removes the rest."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer import prune
from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.config import load_config
from epub_tts_narrator.renderer.workflow import RenderBlocked
from audition_fixtures import prepare_document
from test_contracts import valid_document


class PruneFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config_path = self.root / "renderer.toml"
        self.config_path.write_text('[engine]\ncheckpoint="./unused-tts-model"\n'
                                    '[verification]\nmodel="./unused-asr-model"\n')
        self.book = self.root / "book"
        export_bundle(prepare_document(valid_document()), self.book)
        self.bundle = load_bundle(self.book)
        self.config = resolve_book_config(self.bundle, load_config(self.config_path))
        self.work = Path(self.config["render"]["work_dir"])
        self.work.mkdir()
        self.store = voice_store(self.bundle, self.config)
        self.voice_ids = [voice["id"] for voice in self.bundle["cast"]["voices"]]
        self.needed = set()      # Takes the next render reuses
        self.resumed = set()     # and take folders it resumes.
        # The finished book is never touched.
        self.audiobook = self.book / "audiobook" / "A Book.m4b"
        self.audiobook.parent.mkdir()
        self.audiobook.write_bytes(b"finished book")

    def prune(self, **options):
        options.setdefault("needed_takes", lambda bundle, config: (set(self.needed), set(self.resumed)))
        return prune.prune_book(self.bundle, self.config, **options)

    def files(self):
        # Lock leases stay by design; everything else is compared.
        return {path for path in self.root.rglob("*") if path.is_file() and path.suffix != ".lease"}

    # Voices ---------------------------------------------------------------------------------------

    def casting_take(self, voice_id, nonce, *, hosted=False):
        request = {"kind": "casting_candidate", "nonce": nonce, "series_id": self.store.series_id,
                   "voice_id": voice_id}
        directory = self.work / "casting-candidates" / digest(request)
        directory.mkdir(parents=True)
        write_json(directory / "casting-request.json", request)
        (directory / "attempt-001-casting_candidate-take.wav").write_bytes(b"casting take " + nonce.encode())
        if hosted:
            write_json(directory / "gemini-voice.json", {"voice": "hosted"})
        return directory

    def recording(self, voice_id, data, *, candidate=False):
        nonce = uuid.uuid4().hex
        take = self.casting_take(voice_id, nonce)
        sample = self.root / f"sample-{nonce}.wav"
        sample.write_bytes(data)
        arguments = (voice_id, {}, sample, "Reference words.", {"engine": "fixture"}, {"passed": True},
                     {"request_hash": take.name})
        if candidate:
            binding = self.store.publish_candidate(*arguments, candidate_id=nonce)
        else:
            binding = self.store.publish(*arguments)
        return binding, take

    def preview(self, source_sha256):
        identity = {"source_sha256": source_sha256, "settings": {"target_lufs": -20.0}}
        directory = self.work / "voice-previews" / digest(identity)
        directory.mkdir(parents=True)
        (directory / "voice-reference-loudness-sample.wav").write_bytes(b"preview")
        write_json(directory / "normalization.json", {"identity": identity})
        return directory

    # Chapters -------------------------------------------------------------------------------------

    def chapter(self, chapter_id, number, label, *, age, clips=()):
        identity = {"context": {"label": label}, "chapter": {"id": chapter_id, "number": number},
                    "settings": {"chapter_format": "m4a"}, "clips": list(clips)}
        directory = self.work / "chapters" / digest(identity)
        directory.mkdir(parents=True)
        for name in (f"chapter-{number:04d}.wav", f"chapter-{number:04d}.m4a"):
            (directory / name).write_bytes(b"chapter audio " + label.encode())
        record = {"format": "audiobook-renderer-chapter", "version": 1, "identity": identity,
                  "chapter": {"frames": 100}, "wav_file": f"chapter-{number:04d}.wav",
                  "encoded_file": f"chapter-{number:04d}.m4a"}
        write_json(directory / "chapter.json", dict(record, record_sha256=digest(record)))
        os.utime(directory / "chapter.json", ns=(age, age))
        return directory

    def checkpoint(self, chapter_id, directory, acceptance=()):
        record = {"format": "audiobook-chapter-checkpoint", "version": 1, "cache_key": directory.name,
                  "context": {}, "acceptance": list(acceptance), "audio_file": None}
        write_json(self.work / "completed-chapters" / f"{chapter_id}.json", dict(record, record_sha256=digest(record)))

    # Clips ----------------------------------------------------------------------------------------

    def take(self, text, *, attempts=2, passed=True, extra=None):
        request = {"kind": "speech", "text": text}
        key = digest(request)
        directory = self.work / "clips" / key
        directory.mkdir(parents=True)
        record = None
        for number in range(1, attempts + 1):
            audio = directory / f"attempt-{number:03d}-{key[:32]}-series_voice_a-take.wav"
            audio.write_bytes(f"{text} take {number}".encode())
            raw = directory / f"attempt-{number:03d}-{key[:32]}-series_voice_a-raw.wav"
            raw.write_bytes(b"raw")
            codes = directory / f"{digest([text, number])}.codes.json"
            codes.write_text("{}")
            record = {"request": request, "request_hash": key, "attempt": number,
                      "audio_file": audio.name, "raw_file": raw.name, "audio_sha256": file_hash(audio),
                      "engine_metadata": {"continuity_codes": {"file": codes.name}, **(extra or {})},
                      "report": {"passed": passed and number == attempts}}
            write_json(directory / f"attempt-{number:03d}.json", record)
        if passed:
            write_json(directory / "accepted.json", record)
        return directory, record

    def index(self, record):
        identity = {"kind": "speech", "text": record["request"]["text"]}
        entry = {"format": "audiobook-speech-cache", "version": 1, "identity": identity,
                 "request_hash": record["request_hash"], "accepted_sha256": "a" * 64,
                 "audio_sha256": record["audio_sha256"]}
        path = self.work / "speech-cache" / f"{digest(identity)}.json"
        write_json(path, dict(entry, record_sha256=digest(entry)))
        return path

    def timing(self, audio_sha256):
        binding = {"audio_sha256": audio_sha256, "version": 1}
        path = self.work / "pause-word-timings" / f"{digest(binding)}.json"
        write_json(path, {"format": "epub-tts-take-pause-words", "binding": binding, "words": []})
        return path


class VoicesGroupTests(PruneFixture):
    def test_current_choice_stays_and_superseded_recordings_takes_and_previews_go(self):
        # Given a voice recast twice, then set to its second revision.
        voice = self.voice_ids[0]
        first, first_take = self.recording(voice, b"first")
        old, old_take = self.recording(voice, b"old revision", candidate=True)
        chosen, chosen_take = self.recording(voice, b"chosen revision", candidate=True)
        self.store.select_candidate(voice, chosen["candidate_id"])
        failed_take = self.casting_take(voice, uuid.uuid4().hex)
        hosted_take = self.casting_take(voice, uuid.uuid4().hex, hosted=True)
        current_preview = self.preview(chosen["reference_sha256"])
        old_preview = self.preview(first["reference_sha256"])

        report = self.prune(groups=("voices",), dry_run=False)

        self.assertTrue(report["deleted"])
        self.assertEqual(self.store.load(voice)["asset_hash"], chosen["asset_hash"])
        self.assertTrue(Path(chosen["reference_audio"]).is_file())
        for path in (first["reference_audio"], old["reference_audio"]):
            self.assertFalse(Path(path).exists())
        self.assertFalse(self.store.folder(voice).exists())
        for gone in (first_take, old_take, failed_take, old_preview):
            self.assertFalse(gone.exists(), gone)
        for kept in (chosen_take, hosted_take, current_preview):
            self.assertTrue(kept.is_dir(), kept)
        self.assertEqual(set(report["groups"]), {"voices"})
        self.assertEqual(set(report["groups"]["voices"]["categories"]),
                         {"superseded_recordings", "casting_takes", "previews"})
        self.assertIn("its hosted voice is not released yet", {item["reason"] for item in report["retained"]})
        self.assertTrue(self.audiobook.is_file())

    def test_unsettled_voice_keeps_every_recording(self):
        # Given one voice whose recast is pending, and one never chosen here.
        pending, other = self.voice_ids[:2]
        first, first_take = self.recording(pending, b"first")
        revision, revision_take = self.recording(pending, b"revision", candidate=True)
        self.store.invalidate(pending)
        unchosen, unchosen_take = self.recording(other, b"unchosen", candidate=True)
        before = self.files()

        report = self.prune(groups=("voices",), dry_run=False)

        self.assertEqual(report["files_removed"], 0)
        self.assertEqual(self.files(), before)

    def test_legacy_choice_removes_its_whole_revisions_folder(self):
        voice = self.voice_ids[0]
        first, _ = self.recording(voice, b"first")
        revision, _ = self.recording(voice, b"never chosen", candidate=True)

        self.prune(groups=("voices",), dry_run=False)

        self.assertEqual(self.store.load(voice)["asset_hash"], first["asset_hash"])
        self.assertFalse((self.store.directory / "candidates" / self.store.folder(voice).name).exists())


class ChaptersGroupTests(PruneFixture):
    def test_newest_complete_version_and_checkpointed_version_stay(self):
        # Given chapter one with a checkpointed version and a newer unrecorded one, and
        # chapter two with no checkpoint (out of date) and two versions.
        oldest = self.chapter("chapter-0001", 1, "oldest", age=1_000)
        recorded = self.chapter("chapter-0001", 1, "recorded", age=2_000)
        newest = self.chapter("chapter-0001", 1, "newest", age=3_000)
        self.checkpoint("chapter-0001", recorded)
        stale_old = self.chapter("chapter-0002", 2, "stale old", age=1_000)
        stale_new = self.chapter("chapter-0002", 2, "stale new", age=2_000)
        staging = self.work / "chapters" / ".chapter-abcdefgh" / "ready"
        staging.mkdir(parents=True)
        (staging / "chapter-0003.wav").write_bytes(b"interrupted")

        report = self.prune(groups=("chapters",), dry_run=False)

        for kept in (recorded, newest, stale_new):
            self.assertTrue(kept.is_dir(), kept)
        for gone in (oldest, stale_old, staging.parent):
            self.assertFalse(gone.exists(), gone)
        self.assertEqual(report["groups"]["chapters"]["files"], 7)
        self.assertEqual(set(report["groups"]["chapters"]["categories"]), {"superseded", "interrupted_chapters"})

    def test_unrecognized_chapter_folder_is_kept_and_listed(self):
        odd = self.work / "chapters" / ("c" * 64)
        odd.mkdir(parents=True)
        (odd / "chapter-0001.m4a").write_bytes(b"no record")

        report = self.prune(groups=("chapters",), dry_run=False)

        self.assertTrue((odd / "chapter-0001.m4a").is_file())
        self.assertEqual([item["reason"] for item in report["retained"]], ["unrecognized chapter folder"])


class ClipsGroupTests(PruneFixture):
    def test_takes_the_book_uses_stay_and_everything_else_goes(self):
        # A take the next render reuses, with an earlier rejected attempt.
        reused, reused_record = self.take("reused line")
        # A recovery part the reused take's record names.
        part, _ = self.take("recovery part", attempts=1)
        record = dict(reused_record, engine_metadata=dict(reused_record["engine_metadata"], parts=[part.name]))
        write_json(reused / "accepted.json", record)
        # Takes a kept chapter (found by audio) and a superseded chapter were built from.
        in_chapter, chapter_record = self.take("chapter line")
        in_old_chapter, old_record = self.take("old chapter line")
        self.chapter("chapter-0001", 1, "old", age=1_000, clips=[{"audio_sha256": old_record["audio_sha256"]}])
        self.chapter("chapter-0001", 1, "new", age=2_000, clips=[{"audio_sha256": chapter_record["audio_sha256"]}])
        # A superseded take, a person's judged take, pending work, and abandoned failures.
        stale, stale_record = self.take("superseded line")
        judged, _ = self.take("judged line")
        write_json(self.work / "take-feedback.json",
                   {"format": "x", "version": 1, "takes": {f"{judged.name}:1": {"request_hash": judged.name}}})
        pending, _ = self.take("pending line", passed=False)
        abandoned, _ = self.take("abandoned line", passed=False)
        # A take the render resumes keeps its earlier attempts even after an acceptance.
        resumed, _ = self.take("resumed line")
        self.needed, self.resumed = {reused.name}, {pending.name, resumed.name}
        stale_index, reused_index = self.index(stale_record), self.index(reused_record)
        stale_timing, reused_timing = self.timing(stale_record["audio_sha256"]), self.timing(reused_record["audio_sha256"])
        older_run = self.work / "runs" / ("1" * 64) / "plan.json"
        newer_run = self.work / "runs" / ("2" * 64) / "plan.json"
        for age, path in ((1_000, older_run), (2_000, newer_run)):
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            os.utime(path, ns=(age, age))
        lease = self.work / ".render.lock.lease"

        # While the superseded chapter itself is kept, so is the take it was built from.
        clips_only = self.prune(groups=("clips",), dry_run=True)
        self.assertNotIn(str(in_old_chapter.relative_to(self.book)) + "/accepted.json",
                         {item["path"] for item in clips_only["files"]})

        report = self.prune(groups=("chapters", "clips"), dry_run=False)

        for gone in (stale, abandoned, in_old_chapter, stale_index, stale_timing, older_run.parent):
            self.assertFalse(gone.exists(), gone)
        for kept in (part, in_chapter, reused_index, reused_timing, newer_run, lease):
            self.assertTrue(kept.exists(), kept)
        # The reused take keeps its accepted audio, codes and every record; the rest goes.
        names = {path.name for path in reused.iterdir()}
        self.assertEqual(names, {"accepted.json", "attempt-001.json", "attempt-002.json",
                                 reused_record["audio_file"], reused_record["engine_metadata"]["continuity_codes"]["file"]})
        # Pending work and judged takes stay exactly as they were.
        self.assertEqual(len(list(pending.iterdir())), 8)
        self.assertEqual(len(list(resumed.iterdir())), 9)
        self.assertEqual(len(list(judged.iterdir())), 9)
        self.assertEqual(set(report["groups"]["clips"]["categories"]),
                         {"unused_takes", "rejected_takes", "rejected_codes", "intermediates", "index",
                          "pause_timings", "run_plans"})

    def test_clips_group_refuses_without_a_render_plan_it_can_trust(self):
        self.config["remote_chapters"] = {"url": "https://chapters.invalid"}
        stale, _ = self.take("superseded line")
        with self.assertRaisesRegex(ValueError, "--only voices,chapters"):
            prune.prune_book(self.bundle, self.config, groups=("clips",))
        self.assertTrue(stale.is_dir())


class PruneSafetyTests(PruneFixture):
    def stale_material(self):
        voice = self.voice_ids[0]
        self.recording(voice, b"first")
        chosen, _ = self.recording(voice, b"chosen", candidate=True)
        self.store.select_candidate(voice, chosen["candidate_id"])
        self.chapter("chapter-0001", 1, "old", age=1_000)
        self.chapter("chapter-0001", 1, "new", age=2_000)
        self.take("superseded line")

    def test_dry_run_and_declined_confirmation_delete_nothing(self):
        self.stale_material()
        before = self.files()

        report = self.prune(dry_run=True)
        self.assertEqual(self.files(), before)
        self.assertFalse(report["deleted"])
        self.assertEqual(set(report["groups"]), {"voices", "chapters", "clips"})
        for group in report["groups"].values():
            self.assertGreater(group["files"], 0)
            self.assertGreater(group["bytes"], 0)
        self.assertEqual(report["files_removed"], sum(group["files"] for group in report["groups"].values()))

        seen = []
        report = self.prune(confirm=lambda summary: seen.append(summary) or False)
        self.assertEqual(self.files(), before)
        self.assertFalse(report["deleted"])
        self.assertEqual(seen[0]["files_removed"], report["files_removed"])

    def test_held_render_or_voice_lock_refuses_and_leaves_everything(self):
        self.stale_material()
        for lock in (self.work / ".render.lock", self.book / ".audio-render.lock", self.store.directory / ".lock"):
            with self.subTest(lock=lock.name):
                lock.mkdir()
                self.addCleanup(lambda path=lock: path.rmdir() if path.is_dir() else None)
                before = self.files()
                with self.assertRaises(RenderBlocked):
                    self.prune(dry_run=False)
                self.assertEqual(self.files(), before)
                self.assertTrue(lock.is_dir())
                lock.rmdir()
                owned = {self.work / ".render.lock", self.book / ".audio-render.lock", self.store.directory / ".lock"}
                self.assertFalse([path for path in owned if path.is_dir()])

    def test_only_parses_group_names(self):
        self.assertEqual(prune.parse_groups(None), ("voices", "chapters", "clips"))
        self.assertEqual(prune.parse_groups("clips, voices"), ("voices", "clips"))
        for text in ("", "audio", "voices,books"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "voices, chapters, clips"):
                prune.parse_groups(text)


class PruneCommandTests(PruneFixture):
    def run_cli(self, *args, stdin=None):
        output, error = io.StringIO(), io.StringIO()
        with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(error),
              patch.object(prune, "plan_takes", lambda bundle, config: (set(self.needed), set(self.resumed))),
              patch("sys.stdin", stdin or io.StringIO())):
            status = main([str(arg) for arg in ("prune", self.book, "--config", self.config_path, *args)])
        return status, output.getvalue(), error.getvalue()

    def test_command_previews_asks_and_deletes(self):
        self.chapter("chapter-0001", 1, "old", age=1_000)
        newest = self.chapter("chapter-0001", 1, "new", age=2_000)
        before = self.files()

        status, output, error = self.run_cli("--dry-run")
        self.assertEqual(status, 0, error)
        self.assertIn("chapters", output)
        self.assertIn("Dry run", output)
        self.assertEqual(self.files(), before)

        # Without a terminal and without --yes, nothing is deleted.
        status, output, _ = self.run_cli("--only", "chapters")
        self.assertEqual(status, 0)
        self.assertIn("--yes", output)
        self.assertEqual(self.files(), before)

        terminal = io.StringIO("y\n")
        terminal.isatty = lambda: True
        status, output, _ = self.run_cli("--only", "chapters", stdin=terminal)
        self.assertEqual(status, 0)
        self.assertIn("Deleted 3 file(s)", output)
        self.assertEqual({path.parent for path in self.files() if "chapters" in path.parts}, {newest})
        self.assertTrue(self.audiobook.is_file())

    def test_json_output_is_only_the_report(self):
        self.chapter("chapter-0001", 1, "old", age=1_000)
        self.chapter("chapter-0001", 1, "new", age=2_000)
        status, output, error = self.run_cli("--dry-run", "--json")
        self.assertEqual(status, 0, error)
        report = json.loads(output)
        self.assertEqual(report["groups"]["chapters"]["files"], 3)
        self.assertIn("Dry run", error)

    def test_yes_deletes_without_asking_and_errors_are_per_book(self):
        self.chapter("chapter-0001", 1, "old", age=1_000)
        self.chapter("chapter-0001", 1, "new", age=2_000)
        status, output, error = self.run_cli("--only", "chapters", "--yes")
        self.assertEqual(status, 0, error)
        self.assertIn("Deleted 3 file(s)", output)

        status, _, error = self.run_cli("--only", "books")
        self.assertEqual(status, 2)
        self.assertIn("--only", error)


if __name__ == "__main__":
    unittest.main()
