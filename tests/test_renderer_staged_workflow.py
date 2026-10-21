"""Staged rendering retains verified chapters across independent runs and failures."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from audiobook_renderer.assembly import probe_audio
from audiobook_renderer.bundle import load_bundle, make_plan
from audiobook_renderer.common import digest, file_hash, read_json, write_json
from audiobook_renderer.config import DEFAULTS
from audiobook_renderer.quality import validate_policy
from audiobook_renderer.voices import VoiceStore
from audiobook_renderer.workflow import RenderBlocked, assemble, render
from epub_audiobook.cli import export_bundle
from test_chapter_export import chapter_document
from test_renderer_workflow import FixtureEngine, FixtureVerifier, FFMPEG, FFPROBE
from audition_fixtures import prepare_bundle


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class StagedWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The lantern shone above the quiet harbor.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("At dawn the boat returned safely to shore.", "narration", "p"),
        ])
        export_bundle(document, self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="staged-fixture")
        self.config["render"].update(
            work_dir=str(self.root / "work"), output_dir=str(self.root / "output"),
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.config["quality"] = validate_policy({})
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.messages = []

    def chapters(self):
        return render(self.bundle, self.config, self.engine, self.verifier,
                      chapters_only=True, notify=self.messages.append)

    def package(self):
        return assemble(self.bundle, self.config, notify=self.messages.append)

    def checkpoint(self, number=1):
        return self.root / "work" / "completed-chapters" / f"chapter-{number:04d}.json"

    def test_chapters_only_then_model_free_assembly_preserve_exact_audio_and_sync(self):
        result = self.chapters()
        self.assertEqual(result["status"], "chapters_complete")
        self.assertEqual(result["chapter_count"], 2)
        self.assertFalse((self.root / "output").exists())
        for chapter in result["chapters"]:
            self.assertTrue(Path(chapter["audio_file"]).is_file())
        count = len(self.engine.speech_calls), len(self.verifier.calls)
        with patch("audiobook_renderer.workflow.verified_take", side_effect=AssertionError("No ASR/TTS")):
            final = self.package()
        self.assertEqual(count, (len(self.engine.speech_calls), len(self.verifier.calls)))
        self.assertEqual(final["status"], "complete")
        self.assertFalse(final["partial_book"])
        for c in result["chapters"]:
            self.assertEqual(file_hash(c["audio_file"]),
                             file_hash(self.root / "output" / "chapters" / f"chapter-{c['number']:04d}.m4a"))
        output = self.root / "output"
        self.assertEqual(len(probe_audio(output / "book.m4b", ffprobe=FFPROBE)["chapters"]), 2)
        sync = read_json(output / "sync.json")
        self.assertEqual(len(sync["segments"]), 4)
        self.assertEqual(len(read_json(output / "verification.json")), 4)
        self.assertTrue(all(c["normalization"]["status"] == "normalized" for c in sync["chapters"]))

    def test_selected_chapters_from_separate_runs_assemble_with_original_numbers(self):
        self.config["render"]["chapter_numbers"] = [2]
        second = self.chapters()
        self.assertEqual(second["chapters"][0]["number"], 2)
        self.assertTrue(second["partial_book"])
        second_path = second["chapters"][0]["audio_file"]
        self.config["render"]["chapter_numbers"] = []
        with self.assertRaisesRegex(RenderBlocked, "have not finished rendering: 1"):
            self.package()
        self.config["render"]["chapter_numbers"] = [1]
        self.chapters()
        self.config["render"]["chapter_numbers"] = []
        with patch("audiobook_renderer.workflow.verified_take", side_effect=AssertionError("No generation")):
            result = self.chapters()
        self.assertEqual(result["chapters"][1]["audio_file"], second_path)
        final = self.package()
        self.assertFalse(final["partial_book"])
        self.assertEqual([c["number"] for c in final["audio"]["chapters"]], [1, 2])

    def test_partial_book_keeps_source_chapter_filenames(self):
        self.config["render"]["chapter_numbers"] = [2]
        self.chapters()
        final = self.package()
        self.assertTrue(final["partial_book"])
        self.assertTrue((self.root / "output" / "chapters" / "chapter-0002.m4a").is_file())
        self.assertFalse((self.root / "output" / "chapters" / "chapter-0001.m4a").exists())
        self.assertEqual(final["audio"]["chapters"][0]["ordinal"], 1)

    def test_assembly_can_make_mp3_chapters_from_saved_pcm_without_models(self):
        chapters = self.chapters()
        self.config["render"]["chapter_format"] = "mp3"
        with patch("audiobook_renderer.workflow.verified_take", side_effect=AssertionError("No ASR/TTS")):
            final = self.package()
        self.assertEqual(final["audio"]["chapter_format"], "mp3")
        for chapter in chapters["chapters"]:
            self.assertTrue(Path(chapter["audio_file"]).is_file())
            encoded = self.root / "output" / "chapters" / f"chapter-{chapter['number']:04d}.mp3"
            streams = probe_audio(encoded, ffprobe=FFPROBE)["streams"]
            self.assertEqual(next(s for s in streams if s["codec_type"] == "audio")["codec_name"], "mp3")
        self.assertEqual(final["audio"]["audiobook"]["codec"], "aac")

    def test_recasting_a_voice_blocks_stale_assembly_and_renders_new_speech(self):
        self.chapters()
        store = VoiceStore(self.config["voices"]["directory"], "staged-fixture")
        voice_id = make_plan(self.bundle, self.config)["voice_ids"][0]
        old = store.load(voice_id)
        reference = self.root / "replacement.wav"
        self.engine._write(old["reference_text"], reference, 1234)
        store.publish_candidate(voice_id, old["profile"], reference, old["reference_text"],
                                self.engine.identity, {"passed": True}, {"fixture": True},
                                candidate_id="a" * 32)
        store.select_candidate(voice_id, "a" * 32)
        self.assertNotEqual(old["asset_hash"], store.load(voice_id)["asset_hash"])
        with self.assertRaisesRegex(RenderBlocked, "is stale"):
            self.package()
        before = len(self.engine.speech_calls)
        self.chapters()
        self.assertEqual(len(self.engine.speech_calls) - before, 4)
        self.assertEqual(self.package()["status"], "complete")

    def test_later_interruption_preserves_first_chapter_and_skips_it_on_resume(self):
        self.engine.interrupt_speech_number = 3
        with self.assertRaises(KeyboardInterrupt):
            self.chapters()
        first = read_json(self.checkpoint())
        self.assertTrue(Path(first["audio_file"]).is_file())
        self.assertFalse(self.checkpoint(2).exists())
        checksum = file_hash(first["audio_file"])
        self.engine.interrupt_speech_number = None
        self.chapters()
        self.assertEqual(file_hash(first["audio_file"]), checksum)
        self.assertEqual(len(self.engine.speech_calls), 5)
        self.assertTrue(any("Reusing completed chapter-0001" in item for item in self.messages))
        self.assertFalse((self.root / "work" / ".render.lock").exists())

    def test_failed_final_encoding_preserves_all_completed_chapters_for_assembly(self):
        with patch("audiobook_renderer.workflow.package_audio", side_effect=ValueError("encoding failed")):
            with self.assertRaisesRegex(ValueError, "encoding failed"):
                render(self.bundle, self.config, self.engine, self.verifier, notify=self.messages.append)
        self.assertTrue(self.checkpoint(1).is_file())
        self.assertTrue(self.checkpoint(2).is_file())
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(self.package()["status"], "complete")

    def test_assembly_rejects_changed_generation_or_normalization_settings(self):
        self.chapters()
        for section, name, value in (
            ("render", "seed", 123), ("render", "paragraph_pause_ms", 700),
            ("render", "target_lufs", -22), ("engine", "temperature", 0.1),
            ("quality", "max_wer", 0.05), ("verification", "language", "fr"),
        ):
            with self.subTest(section=section, name=name):
                old = self.config[section].get(name)
                self.config[section][name] = value
                with self.assertRaisesRegex(RenderBlocked, "is stale"):
                    self.package()
                self.config[section][name] = old
        self.assertFalse((self.root / "output").exists())

    def test_changed_normalization_reuses_verified_clips_and_replaces_chapter_checkpoint(self):
        old = self.chapters()
        before = len(self.engine.speech_calls), len(self.verifier.calls)
        self.config["render"]["target_lufs"] = -22
        current = self.chapters()
        self.assertEqual(before, (len(self.engine.speech_calls), len(self.verifier.calls)))
        self.assertNotEqual(old["chapters"][0]["audio_file"], current["chapters"][0]["audio_file"])
        self.assertTrue(Path(old["chapters"][0]["audio_file"]).is_file())
        self.assertEqual(self.package()["status"], "complete")

    def test_corrupt_pcm_or_index_cannot_be_assembled_or_silently_regenerated(self):
        result = self.chapters()
        wav = Path(result["chapters"][0]["cache_directory"]) / "chapter.wav"
        original = wav.read_bytes()
        wav.write_bytes(original[:-2] + b"xx")
        for action in (self.package, self.chapters):
            with self.assertRaisesRegex(RenderBlocked, "PCM failed integrity"):
                action()
        wav.write_bytes(original)
        index = read_json(self.checkpoint())
        index["acceptance"][0]["report"]["passed"] = False
        write_json(self.checkpoint(), index)
        with self.assertRaisesRegex(RenderBlocked, "Checkpoint integrity"):
            self.package()

    def test_checkpoint_without_complete_verification_is_rejected_even_with_updated_checksum(self):
        self.chapters()
        record = read_json(self.checkpoint())
        del record["record_sha256"]
        record["acceptance"] = record["acceptance"][:-1]
        write_json(self.checkpoint(), dict(record, record_sha256=digest(record)))
        with self.assertRaisesRegex(RenderBlocked, "Incomplete chapter verification"):
            self.package()

    def test_atomic_publication_never_overwrites_existing_output(self):
        self.chapters()
        output = self.root / "output"
        output.mkdir()
        keep = output / "keep.txt"
        keep.write_text("keep")
        with self.assertRaisesRegex(RenderBlocked, "Output already exists"):
            self.package()
        self.assertEqual(keep.read_text(), "keep")
        # Chapter rendering can still use a different stage without touching a completed output.
        self.assertEqual(self.chapters()["status"], "chapters_complete")

    def test_failed_publication_leaves_no_partial_book_and_can_resume(self):
        self.chapters()
        with patch("audiobook_renderer.workflow.os.rename", side_effect=OSError("publication interrupted")):
            with self.assertRaisesRegex(OSError, "publication interrupted"):
                self.package()
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / ".output.publish.lock").exists())
        self.assertEqual(self.package()["status"], "complete")

    def test_malformed_checkpoint_reports_a_controlled_failure(self):
        self.chapters()
        for value in (None, "bad", []):
            with self.subTest(value=value):
                write_json(self.checkpoint(), value)
                with self.assertRaisesRegex(RenderBlocked, "Checkpoint must be a JSON object"):
                    self.package()


if __name__ == "__main__":
    unittest.main()
