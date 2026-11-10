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
from epub_tts_narrator.renderer import chapter_cache
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.common import file_hash, read_json, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.speech_text import trim_speech_text
from epub_tts_narrator.renderer import workflow as render_module
from epub_tts_narrator.renderer.workflow import RenderBlocked, assemble, render
from audition_fixtures import prepare_document
from test_chapter_export import chapter_document
from shared_fixtures import rebind_audiobook_filename
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
            max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE, require_cover=False)
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

    def book_path(self, result=None):
        audio = result["audio"] if result is not None else read_json(self.output / "render.json")["audio"]
        return self.output / audio["audiobook"]["path"]

    def model_counts(self):
        return len(self.engine.voice_calls), len(self.engine.speech_calls), len(self.verifier.calls)

    def chapter_audio(self, number):
        return Path(read_json(self.checkpoint(number))["audio_file"])

    def recorded_chapters(self):
        entered = []
        real = render_module._render_chapter_with_recovery

        def spy(bundle, chapter, *args, **kwargs):
            entered.append(chapter["number"])
            return real(bundle, chapter, *args, **kwargs)
        return entered, spy

    def test_script_revision_replaces_book_and_only_generates_changed_passage(self):
        original = self.run_render()
        before = self.model_counts()
        old_book = file_hash(self.book_path(original))
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
        # Only the corrected passage is new. The following take heard the old
        # version as history; tolerant history reuse (the default) keeps it.
        self.assertEqual(self.model_counts(), (before[0], before[1] + 1, before[2] + 1))
        self.assertEqual([call["text"] for call in self.engine.speech_calls[-1:]],
                         [self.document["segments"][1]["text"]])
        self.assertNotEqual(old_book, file_hash(self.book_path()))
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
        self.assertEqual(sync["audio_sha256"], file_hash(self.book_path()))
        manifest = read_json(self.output / "audio-manifest.json")
        self.assertEqual(manifest["sync"]["sha256"], file_hash(self.output / "sync.json"))
        self.assertEqual(manifest["audiobook"]["sha256"], file_hash(self.book_path()))

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

    def completed_chapter_audio(self):
        return {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (self.work / "chapters").rglob("*.m4a")}

    def test_unchanged_rerender_never_enters_clip_execution_asr_or_encoding(self):
        self.run_render()
        audio = self.completed_chapter_audio()
        counts = self.model_counts()
        self.messages.clear()

        with patch("epub_tts_narrator.renderer.workflow._render_chapter_with_recovery",
                   side_effect=AssertionError("No clip execution")), \
             patch("epub_tts_narrator.renderer.workflow._render_chunk",
                   side_effect=AssertionError("No chunk synthesis")), \
             patch("epub_tts_narrator.renderer.workflow.verified_take",
                   side_effect=AssertionError("No TTS/ASR take")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("No ASR")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter",
                   side_effect=AssertionError("No chapter encoding")), \
             patch("epub_tts_narrator.renderer.workflow.package_audio",
                   side_effect=AssertionError("No book encoding")):
            current = self.run_render()

        self.assertEqual(current["status"], "complete")
        self.assertEqual(self.completed_chapter_audio(), audio,
                         "an unchanged rerender must not rewrite any chapter M4A")
        self.assertEqual(self.model_counts(), counts)
        for number in (1, 2):
            self.assertTrue(any(f"Reusing completed chapter {number} / 2" in message
                                for message in self.messages), self.messages)

    def test_edited_chapter_alone_rerenders_while_other_chapters_keep_their_audio(self):
        self.run_render()
        counts = self.model_counts()
        audio = self.completed_chapter_audio()
        saved_context = read_json(self.checkpoint(1))["context"]
        self.messages.clear()

        # A serial update pins the pre-update source fingerprint, then edits
        # only chapter two's text; chapter one's plan is unchanged.
        from epub_tts_narrator.pipeline import fingerprint
        self.document["provenance"].setdefault(
            "chunk_identity_source_fingerprint", fingerprint(self.document["source"]))
        new_text = "At dawn the boat returned safely to the harbor."
        self.document["source"]["blocks"][4]["text"] = new_text
        self.document["segments"][4]["text"] = new_text
        self.document["segments"][4]["end"] = len(new_text)
        self.save_revision()

        import epub_tts_narrator.renderer.workflow as workflow
        entered = []
        real_recovery = workflow._render_chapter_with_recovery
        real_prepare = workflow.prepare_chapter
        transcribe = self.verifier.transcribe
        asr_calls = []

        def recovery(bundle, chapter, *args, **kwargs):
            entered.append(("render", chapter["number"]))
            return real_recovery(bundle, chapter, *args, **kwargs)

        def prepare(chapter, *args, **kwargs):
            entered.append(("encode", chapter["number"]))
            return real_prepare(chapter, *args, **kwargs)

        def asr(path):
            asr_calls.append(path)
            return transcribe(path)

        with patch.object(workflow, "_render_chapter_with_recovery", side_effect=recovery), \
             patch.object(workflow, "prepare_chapter", side_effect=prepare), \
             patch.object(self.verifier, "transcribe", side_effect=asr):
            self.run_render()

        self.assertEqual([kind for kind, _ in entered], ["render", "encode"],
                         "only the edited chapter may enter clip execution and encoding")
        self.assertEqual({number for _, number in entered}, {2})
        self.assertEqual(len(asr_calls), 1, "only the edited chapter's new take is transcribed")
        self.assertEqual(self.model_counts(), (counts[0], counts[1] + 1, counts[2] + 1))
        for name, (saved_bytes, saved_mtime) in audio.items():
            if "chapter-0001" in name:
                current = self.completed_chapter_audio()[name]
                self.assertEqual(current[0], saved_bytes, f"{name} bytes changed")
                self.assertEqual(current[1], saved_mtime, f"{name} mtime changed")
        self.assertTrue(any("Reusing completed chapter 1 / 2" in message for message in self.messages),
                        self.messages)
        current_context = read_json(self.checkpoint(1))["context"]
        self.assertEqual({key: value for key, value in current_context.items()
                          if key not in {"bundle_hash"}},
                         {key: value for key, value in saved_context.items()
                          if key not in {"bundle_hash"}})

    def test_attribution_metadata_edit_reuses_chapter_audio_without_rendering(self):
        # Attribution prose and review flags never reach synthesis, composition,
        # or the published synchronization, so they must not rebuild the chapter.
        self.run_render()
        saved = {number: self.chapter_audio(number) for number in (1, 2)}
        before = {number: (saved[number].read_bytes(), saved[number].stat().st_mtime_ns)
                  for number in (1, 2)}
        checkpoints = {number: self.checkpoint(number).read_bytes() for number in (1, 2)}
        counts = self.model_counts()
        self.document["segments"][1]["attribution"]["reason"] = "A corrected explanation only."
        self.document["segments"][3]["attribution"]["review_required"] = True
        self.save_revision()

        with patch("epub_tts_narrator.renderer.workflow._render_chapter_with_recovery",
                   side_effect=AssertionError("clip render entered")), \
             patch("epub_tts_narrator.renderer.workflow.verified_take",
                   side_effect=AssertionError("TTS/ASR entered")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter",
                   side_effect=AssertionError("chapter encoding entered")):
            current = self.run_render()

        self.assertEqual(current["status"], "complete")
        self.assertEqual(self.model_counts(), counts)
        for number in (1, 2):
            self.assertEqual(self.chapter_audio(number).read_bytes(), before[number][0])
            self.assertEqual(self.chapter_audio(number).stat().st_mtime_ns, before[number][1])
            self.assertEqual(self.checkpoint(number).read_bytes(), checkpoints[number])
        self.assertTrue(any("Reusing completed chapter 1 / 2" in message
                            for message in self.messages), self.messages)

    def test_delivery_direction_change_rerenders_its_chapter(self):
        self.run_render()
        entered, spy = self.recorded_chapters()
        counts = self.model_counts()
        self.document["segments"][1].setdefault("delivery", {})["direction"] = "whisper"
        self.save_revision()

        with patch("epub_tts_narrator.renderer.workflow._render_chapter_with_recovery", spy):
            self.run_render()

        self.assertEqual(entered, [1])
        self.assertEqual(self.model_counts(), (counts[0], counts[1] + 1, counts[2] + 1))
        self.assertTrue(any("Reusing completed chapter 2 / 2" in message
                            for message in self.messages), self.messages)

    def test_custom_recorded_filename_survives_title_change_and_repackaging(self):
        # Given an intact custom-named publication needing a publication-key refresh.
        original = self.run_render()
        original_name = original["audio"]["audiobook"]["path"]
        rebind_audiobook_filename(self.output, "Existing Custom Name.m4b")
        previous = read_json(self.output / "render.json")
        previous.pop("publication_key")
        write_json(self.output / "render.json", previous)
        self.bundle["source"]["book"]["title"] = "Changed Fixture Title"

        # When repackaging without any new speech.
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            current = self.run_render()

        # Then every published reference retains the recorded name, not the new title or book.m4b.
        book = self.book_path(current)
        self.assertEqual(book.name, "Existing Custom Name.m4b")
        self.assertTrue(book.is_file())
        self.assertFalse((self.output / original_name).exists())
        self.assertFalse((self.output / "book.m4b").exists())
        self.assertEqual(current["audio"]["book"]["title"], "Changed Fixture Title")
        self.assertEqual(read_json(self.output / "sync.json")["audio_file"], book.name)
        self.assertEqual(current["files_sha256"][book.name], file_hash(book))

    def test_legacy_book_without_recorded_path_resumes_without_any_encoding(self):
        # Given an old complete publication with book.m4b and no audiobook.path field.
        self.run_render()
        rebind_audiobook_filename(self.output, "book.m4b", record_path=False)
        before = self.output_bytes()
        counts = self.model_counts()

        # When resuming the unchanged publication.
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("No chapter encoding")), \
             patch("epub_tts_narrator.renderer.workflow.package_audio", side_effect=AssertionError("No book encoding")):
            current = self.run_render()

        # Then the exact legacy publication is reused without adding a fresh descriptive output.
        self.assertEqual(current["status"], "complete")
        self.assertEqual(self.output_bytes(), before)
        self.assertEqual(self.model_counts(), counts)
        self.assertTrue((self.output / "book.m4b").is_file())

    def test_legacy_book_without_recorded_path_keeps_its_name_when_reassembled(self):
        # Given an old publication and a requested chapter-encoding change.
        original = self.run_render()
        original_name = original["audio"]["audiobook"]["path"]
        rebind_audiobook_filename(self.output, "book.m4b", record_path=False)
        self.config["render"]["chapter_format"] = "mp3"
        counts = self.model_counts()

        # When assembling the changed package without generating speech.
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            result = self.package()

        # Then the old manifest fallback keeps book.m4b and every reference agrees.
        self.assertEqual(result["audio"]["audiobook"]["path"], "book.m4b")
        self.assertTrue((self.output / "book.m4b").is_file())
        self.assertFalse((self.output / original_name).exists())
        self.assertEqual(read_json(self.output / "sync.json")["audio_file"], "book.m4b")
        self.assertEqual(result["files_sha256"]["book.m4b"], file_hash(self.output / "book.m4b"))
        self.assertEqual(self.model_counts(), counts)

    def test_runtime_update_reuses_published_book_without_model_alias(self):
        self.engine.identity = {
            "engine": "drama3_mlx", "model": "fishaudio/drama-3",
            "runtime": "Sogni-AI/drama3-mlx", "adapter_version": 1,
            "installed_version": "0.1.1", "installed_revision": "old-runtime",
            "checkpoint_files_sha256": {"model.safetensors": "a" * 64},
        }
        original = self.run_render()
        before = self.output_bytes()
        counts = self.model_counts()
        accepted = {path: path.read_bytes() for path in (self.work / "clips").rglob("accepted.json")}
        checkpoints = {number: self.checkpoint(number).read_bytes() for number in (1, 2)}
        self.engine.identity = dict(self.engine.identity,
            runtime="updated-runtime", adapter_version=2,
            installed_version="0.1.1+epub.1", installed_revision=None)

        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("No chapter encoding")), \
             patch("epub_tts_narrator.renderer.workflow.package_audio", side_effect=AssertionError("No book encoding")):
            current = self.run_render()

        self.assertEqual(current, original)
        self.assertEqual(before, self.output_bytes())
        self.assertEqual(counts, self.model_counts())
        for path, contents in accepted.items():
            self.assertEqual(path.read_bytes(), contents)
        for number, contents in checkpoints.items():
            self.assertEqual(self.checkpoint(number).read_bytes(), contents)

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

    def test_moving_a_passage_to_another_chapter_regenerates_changed_history_and_rebuilds_timing(self):
        self.run_render()
        counts = self.model_counts()
        texts = {call["text"]: call for call in self.engine.speech_calls}
        before = read_json(self.output / "sync.json")
        old = next(row for row in before["segments"] if row["segment_id"] == "s2")
        order = [0, 2, 3, 1, 4]
        self.document["source"]["blocks"] = [self.document["source"]["blocks"][i] for i in order]
        self.document["segments"] = [self.document["segments"][i] for i in order]
        self.document["source"]["chapters"][0]["block_ids"] = [
            row["id"] for row in self.document["source"]["blocks"]]
        self.save_revision()

        self.run_render()

        # History is same-voice. The moved narration now follows chapter two's
        # heading and the narration after it hears the new take, so both are
        # new; the other voice's dialogue left in chapter one heard neither.
        changed = [self.document["segments"][i]["text"] for i in (3, 4)]
        self.assertEqual(self.model_counts(), (counts[0], counts[1] + 2, counts[2] + 2))
        self.assertEqual([call["text"] for call in self.engine.speech_calls[counts[1]:]], changed)
        for call in self.engine.speech_calls[counts[1]:]:
            self.assertEqual(call["voice_hash"], texts[call["text"]]["voice_hash"])
        counts = self.model_counts()
        after = read_json(self.output / "sync.json")
        moved = next(row for row in after["segments"] if row["segment_id"] == "s2")
        self.assertNotEqual(old["chapter_id"], moved["chapter_id"])
        self.assertNotEqual(old["book_start_frame"], moved["book_start_frame"])
        self.assertEqual(old["source_text"], moved["source_text"])
        self.run_render()
        self.assertEqual(self.model_counts(), counts)

    def test_splitting_a_dialogue_tag_does_not_regenerate_unchanged_later_chapters(self):
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
        # that bookkeeping change must not be treated as different speech. The
        # following take heard the old passage as history; tolerant history
        # reuse (the default) keeps it, so only the new passages are generated.
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
        original = self.run_render()
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
        book_name = original["audio"]["audiobook"]["path"]
        self.assertNotEqual(before[book_name], (self.output / book_name).read_bytes())
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

    def append_chapter(self, texts):
        """Publish appended chapters as an analyzer book update does."""
        from epub_tts_narrator.pipeline import fingerprint
        provenance = self.document["provenance"]
        provenance.setdefault("chunk_identity_source_fingerprint", fingerprint(self.document["source"]))
        blocks, segments = self.document["source"]["blocks"], self.document["segments"]
        for text, kind in texts:
            number = len(blocks) + 1
            blocks.append(dict(deepcopy(blocks[-1]), id=f"b{number}", text=text))
            self.document["source"]["chapters"][0]["block_ids"].append(f"b{number}")
            segments.append(dict(deepcopy(segments[-1]), id=f"s{number}", block_id=f"b{number}",
                                 text=text, start=0, end=len(text), kind=kind))
        self.save_revision()

    def chapter_record(self, number):
        checkpoint = read_json(self.checkpoint(number))
        return checkpoint, read_json(self.work / "chapters" / checkpoint["cache_key"] / "chapter.json")

    def test_appended_chapter_renders_alone_and_existing_chapters_keep_their_audio(self):
        self.run_render()
        counts = self.model_counts()
        before = {number: self.chapter_record(number) for number in (1, 2)}
        old_ids = [[chunk["chunk_id"] for chunk in chapter["chunks"]]
                   for chapter in make_plan(self.bundle, self.config)["chapters"]]
        self.append_chapter([("CHAPTER THREE", "heading"), ("Night fell over the quiet harbor.", "narration")])

        plan = make_plan(self.bundle, self.config)
        self.assertEqual([[chunk["chunk_id"] for chunk in chapter["chunks"]] for chapter in plan["chapters"][:2]],
                         old_ids)
        with patch.object(chapter_cache, "compose_chapter", wraps=chapter_cache.compose_chapter) as composed:
            result = self.run_render()

        self.assertEqual(result["status"], "complete")
        # Only the appended chapter is synthesized, verified and composed.
        new_speech = self.engine.speech_calls[counts[1]:]
        self.assertEqual([call["text"] for call in new_speech],
                         ["CHAPTER THREE", "Night fell over the quiet harbor."])
        self.assertEqual(len(self.verifier.calls) - counts[2], 2)
        self.assertEqual(composed.call_count, 1)
        for number in (1, 2):
            (old_checkpoint, old_cache), (checkpoint, cache) = before[number], self.chapter_record(number)
            # Same verified takes and PCM; only the chapter count (track tag) is new.
            self.assertEqual(checkpoint["acceptance"], old_checkpoint["acceptance"])
            self.assertEqual(cache["pcm_sha256"], old_cache["pcm_sha256"])
            self.assertEqual(checkpoint["context"]["total_chapters"], 3)
            self.assertEqual(cache["identity"]["total_chapters"], 3)
            self.assertEqual({key: value for key, value in checkpoint["context"].items()
                              if key not in {"total_chapters", "bundle_hash"}},
                             {key: value for key, value in old_checkpoint["context"].items()
                              if key not in {"total_chapters", "bundle_hash"}})
        self.assertTrue(any("Re-encoding chapter 1 / 3" in message for message in self.messages), self.messages)
        # A rerun needs nothing at all.
        counts = self.model_counts()
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")), \
             patch("epub_tts_narrator.renderer.workflow.prepare_chapter", side_effect=AssertionError("No chapter encoding")):
            self.assertEqual(self.run_render()["status"], "complete")
        self.assertEqual(counts, self.model_counts())

    def test_assemble_retags_existing_chapters_for_an_appended_chapter_without_models(self):
        self.run_render()
        self.append_chapter([("CHAPTER THREE", "heading"), ("Night fell over the quiet harbor.", "narration")])
        # Render only the new chapter; assembly then retags the earlier ones.
        self.config["render"]["chapter_numbers"] = [3]
        self.run_render(chapters_only=True)
        self.config["render"]["chapter_numbers"] = []
        self.assertEqual(self.chapter_record(1)[1]["identity"]["total_chapters"], 2)
        counts = self.model_counts()
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            result = self.package()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(counts, self.model_counts())
        for number in (1, 2, 3):
            self.assertEqual(self.chapter_record(number)[1]["identity"]["total_chapters"], 3)

    def test_modified_published_audio_is_recreated_by_the_next_render(self):
        self.run_render()
        book = self.book_path()
        book.write_bytes(book.read_bytes() + b"User changes")
        self.correct_speaker()
        result = self.run_render()
        # Then the edited file does not block the render and the publication is recreated.
        self.assertEqual(result["status"], "complete")
        self.assertNotIn(b"User changes", book.read_bytes())
        self.assertEqual(file_hash(book), result["files_sha256"][book.name])


if __name__ == "__main__":
    unittest.main()
