"""Published books can be refreshed without discarding their verified audio."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.narration_boundaries import repair_narration_boundaries
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_text import trim_speech_text
from epub_tts_narrator.renderer.workflow import RenderBlocked, assemble, render
from audition_fixtures import prepare_document
from test_chapter_export import chapter_document
from test_renderer_workflow import FixtureEngine, FixtureVerifier, FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class IncrementalRendererTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "output"
        self.work = self.root / "work"
        self.bundle_dir = self.root / "bundle"
        self.document = prepare_document(chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("Let us leave before the morning bell.", "narration", "p"),
            ("The boat is already waiting for us.", "dialogue", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("At dawn the boat returned safely to shore.", "narration", "p"),
        ]))
        # Both voices are already in use before correcting the first passage.
        self.document["segments"][2].update(role_id="mara", voice_id="v1")
        self.save_revision()
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="incremental-fixture")
        self.config["render"].update(
            work_dir=str(self.work), output_dir=str(self.output),
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE)
        self.config["quality"] = validate_policy({})
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.messages = []

    def save_revision(self):
        # Export through the normal contract/checksum path, then install the
        # new fixture analysis at the same location as an analyzer revision.
        replacement = self.root / "next-bundle"
        export_bundle(self.document, replacement)
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)
        replacement.rename(self.bundle_dir)
        self.bundle = load_bundle(self.bundle_dir)

    def correct_speaker(self):
        self.document["segments"][1].update(kind="dialogue", role_id="mara", voice_id="v1")
        self.save_revision()

    def run_render(self, *, chapters_only=False):
        return render(self.bundle, self.config, self.engine, self.verifier,
                      chapters_only=chapters_only, notify=self.messages.append)

    def package(self):
        return assemble(self.bundle, self.config, notify=self.messages.append)

    def checkpoint(self, number):
        return self.work / "completed-chapters" / f"chapter-{number:04d}.json"

    def output_bytes(self):
        return {path.relative_to(self.output).as_posix(): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file()}

    def model_counts(self):
        return len(self.engine.voice_calls), len(self.engine.speech_calls), len(self.verifier.calls)

    def test_script_revision_replaces_book_and_only_generates_changed_passage(self):
        original = self.run_render()
        before = self.model_counts()
        old_book = file_hash(self.output / "book.m4b")
        old_manifest = (self.output / "audio-manifest.json").read_bytes()
        first_checkpoint = self.checkpoint(1).read_bytes()
        unaffected_checkpoint = self.checkpoint(2).read_bytes()
        unaffected_mtime = self.checkpoint(2).stat().st_mtime_ns
        original_clips = {path: path.read_bytes() for path in (self.work / "clips").rglob("accepted.json")}
        self.correct_speaker()

        current = self.run_render()

        self.assertEqual(current["status"], "complete")
        self.assertNotEqual(original["bundle_hash"], current["bundle_hash"])
        self.assertEqual(current["bundle_hash"], self.bundle["bundle_hash"])
        self.assertEqual(self.model_counts(), (before[0], before[1] + 1, before[2] + 1))
        self.assertEqual(self.engine.speech_calls[-1]["text"], self.document["segments"][1]["text"])
        self.assertNotEqual(old_book, file_hash(self.output / "book.m4b"))
        self.assertNotEqual(old_manifest, (self.output / "audio-manifest.json").read_bytes())
        self.assertNotEqual(first_checkpoint, self.checkpoint(1).read_bytes())
        self.assertEqual(unaffected_checkpoint, self.checkpoint(2).read_bytes())
        self.assertEqual(unaffected_mtime, self.checkpoint(2).stat().st_mtime_ns)
        self.assertTrue(any("Reusing completed chapter 2 / 2" in message for message in self.messages))
        for path, contents in original_clips.items():
            self.assertEqual(path.read_bytes(), contents)
        sync = read_json(self.output / "sync.json")
        corrected = next(segment for segment in sync["segments"] if segment["segment_id"] == "s2")
        self.assertEqual(corrected["role_id"], "mara")
        self.assertEqual(corrected["voice_id"], "v1")
        self.assertEqual(sync["audio_sha256"], file_hash(self.output / "book.m4b"))
        manifest = read_json(self.output / "audio-manifest.json")
        self.assertEqual(manifest["sync"]["sha256"], file_hash(self.output / "sync.json"))
        self.assertEqual(manifest["audiobook"]["sha256"], file_hash(self.output / "book.m4b"))

    def test_unchanged_published_book_is_reused_without_models_or_encoding(self):
        original = self.run_render()
        before = self.output_bytes()
        counts = self.model_counts()
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("No chapter encoding")), \
             patch("epub_tts_narrator.renderer.workflow.package_audio", side_effect=AssertionError("No book encoding")):
            current = self.run_render()
        self.assertEqual(current, original)
        self.assertEqual(before, self.output_bytes())
        self.assertEqual(counts, self.model_counts())

    def test_source_reidentification_reuses_speech_and_publishes_current_locations(self):
        self.run_render()
        counts = self.model_counts()
        # Re-extracting or moving source content can change every source ID,
        # without changing any of the requested performances.
        for block in self.document["source"]["blocks"]:
            block["id"] = "moved-" + block["id"]
        for chapter in self.document["source"]["chapters"]:
            chapter["block_ids"] = ["moved-" + identifier for identifier in chapter["block_ids"]]
        for segment in self.document["segments"]:
            segment["block_id"] = "moved-" + segment["block_id"]
            segment["id"] = "moved-" + segment["id"]
        self.save_revision()

        self.run_render()

        self.assertEqual(self.model_counts(), counts)
        sync = read_json(self.output / "sync.json")
        self.assertTrue(all(row["segment_id"].startswith("moved-") for row in sync["segments"]))
        self.assertTrue(all(row["block_id"].startswith("moved-") for row in sync["segments"]))
        # The new chapter checkpoint must also restore without synthesizing.
        self.run_render()
        self.assertEqual(self.model_counts(), counts)

    def test_attribution_explanation_edits_reuse_speech(self):
        self.run_render()
        counts = self.model_counts()
        self.document["segments"][1]["attribution"]["reason"] = "A corrected explanation only."
        self.save_revision()

        self.run_render()

        self.assertEqual(self.model_counts(), counts)

    def test_moving_a_passage_to_another_chapter_reuses_audio_and_rebuilds_timing(self):
        self.run_render()
        counts = self.model_counts()
        before = read_json(self.output / "sync.json")
        old = next(row for row in before["segments"] if row["segment_id"] == "s2")
        order = [0, 2, 3, 1, 4]
        self.document["source"]["blocks"] = [self.document["source"]["blocks"][i] for i in order]
        self.document["segments"] = [self.document["segments"][i] for i in order]
        self.document["source"]["chapters"][0]["block_ids"] = [
            row["id"] for row in self.document["source"]["blocks"]]
        self.save_revision()

        self.run_render()

        self.assertEqual(self.model_counts(), counts)
        after = read_json(self.output / "sync.json")
        moved = next(row for row in after["segments"] if row["segment_id"] == "s2")
        self.assertNotEqual(old["chapter_id"], moved["chapter_id"])
        self.assertNotEqual(old["book_start_frame"], moved["book_start_frame"])
        self.assertEqual(old["source_text"], moved["source_text"])
        self.run_render()
        self.assertEqual(self.model_counts(), counts)

    def test_splitting_a_dialogue_tag_does_not_regenerate_later_unchanged_speech(self):
        passage = '"Wow, this is amazing," she said, "Why should we do that again?"'
        self.document["source"]["blocks"][1]["text"] = passage
        self.document["segments"][1].update(
            text=passage, end=len(passage), kind="dialogue", role_id="mara", voice_id="v1")
        self.save_revision()
        self.run_render()
        counts = self.model_counts()
        unaffected_chapter = read_json(self.checkpoint(2))["audio_file"]
        unaffected_audio = file_hash(unaffected_chapter)

        self.document = repair_narration_boundaries(self.document)
        repaired = [segment for segment in self.document["segments"] if segment["block_id"] == "b2"]
        self.assertEqual([segment["role_id"] for segment in repaired], ["mara", "narrator", "mara"])
        self.save_revision()
        self.run_render()

        # Adding two segments shifts the export sequence for every later row;
        # that bookkeeping change must not be treated as different speech.
        self.assertEqual(self.model_counts(), (counts[0], counts[1] + 3, counts[2] + 3))
        self.assertEqual([call["text"] for call in self.engine.speech_calls[counts[1]:]],
                         [trim_speech_text(segment["text"]) for segment in repaired])
        self.assertEqual(read_json(self.checkpoint(2))["audio_file"], unaffected_chapter)
        self.assertEqual(file_hash(unaffected_chapter), unaffected_audio)

    def test_failed_updated_speech_keeps_previously_published_book_intact(self):
        self.run_render()
        before = self.output_bytes()
        self.correct_speaker()
        self.verifier.wrong_texts.add(self.document["segments"][1]["text"])
        counts = self.model_counts()
        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            self.run_render()
        self.assertEqual(before, self.output_bytes())
        self.assertEqual(len(self.engine.speech_calls), counts[1] + 2)
        self.assertFalse((self.work / ".render.lock").exists())
        self.assertFalse((self.root / ".output.publish.lock").exists())

    def test_failed_final_encoding_keeps_old_output_and_allows_model_free_retry(self):
        self.run_render()
        before = self.output_bytes()
        self.correct_speaker()
        with patch("epub_tts_narrator.renderer.workflow.package_audio", side_effect=ValueError("encoding failed")):
            with self.assertRaisesRegex(ValueError, "encoding failed"):
                self.run_render()
        self.assertEqual(before, self.output_bytes())
        counts = self.model_counts()
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            result = self.package()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["bundle_hash"], self.bundle["bundle_hash"])
        self.assertNotEqual(before["book.m4b"], (self.output / "book.m4b").read_bytes())
        self.assertEqual(counts, self.model_counts())

    def test_assemble_refreshes_existing_output_for_changed_encoding_without_models(self):
        self.run_render()
        counts = self.model_counts()
        self.config["render"]["chapter_format"] = "mp3"
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            result = self.package()
        self.assertEqual(result["audio"]["chapter_format"], "mp3")
        self.assertTrue((self.output / "chapters" / "chapter-0001.mp3").is_file())
        self.assertFalse(list((self.output / "chapters").glob("*.m4a")))
        self.assertEqual(counts, self.model_counts())

    def test_failed_replacement_publication_restores_complete_previous_output(self):
        self.run_render()
        before = self.output_bytes()
        self.correct_speaker()
        rename = os.rename

        def interrupt_new_publication(source, target, *args, **kwargs):
            if Path(source).name == "ready" and Path(target) == self.output:
                raise OSError("publication interrupted")
            return rename(source, target, *args, **kwargs)

        with patch("epub_tts_narrator.renderer.workflow.os.rename", side_effect=interrupt_new_publication):
            with self.assertRaisesRegex(OSError, "publication interrupted"):
                self.run_render()
        self.assertEqual(before, self.output_bytes())
        self.assertFalse((self.root / ".output.publish.lock").exists())
        counts = self.model_counts()
        self.assertEqual(self.package()["status"], "complete")
        self.assertEqual(counts, self.model_counts())

    def test_unmanaged_output_is_refused_before_generating_any_voices_or_speech(self):
        self.output.mkdir()
        (self.output / "notes.txt").write_text("This directory belongs to the user.")
        before = self.output_bytes()
        with self.assertRaises(RenderBlocked):
            self.run_render()
        self.assertEqual(self.model_counts(), (0, 0, 0))
        self.assertEqual(before, self.output_bytes())

    def test_output_from_another_book_is_refused_before_any_generation(self):
        self.run_render()
        before = self.output_bytes()
        counts = self.model_counts()
        self.document["source"]["book"].update(id="different-book", title="An Unrelated Story")
        self.save_revision()
        with self.assertRaises(RenderBlocked):
            self.run_render()
        self.assertEqual(counts, self.model_counts())
        self.assertEqual(before, self.output_bytes())

    def test_modified_published_audio_is_not_silently_overwritten(self):
        self.run_render()
        book = self.output / "book.m4b"
        book.write_bytes(book.read_bytes() + b"User changes")
        before = self.output_bytes()
        counts = self.model_counts()
        self.correct_speaker()
        with self.assertRaises(RenderBlocked):
            self.run_render()
        self.assertEqual(counts, self.model_counts())
        self.assertEqual(before, self.output_bytes())


if __name__ == "__main__":
    unittest.main()
