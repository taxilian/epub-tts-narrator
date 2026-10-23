"""Visible render progress preserves truthful completion and accepted audio."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.progress import ChapterProgress
from epub_tts_narrator.renderer.workflow import RenderBlocked, render, verified_take
from shared_fixtures import renderer_config
from test_chapter_export import chapter_document
from test_renderer_paragraph_workflow import ParagraphFixtureEngine
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class RenderProgressWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = renderer_config(self.root, series_id="progress-fixture")
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.messages = []

    def bundle(self, document):
        export_bundle(document, self.root / "bundle")
        return prepare_bundle(load_bundle(self.root / "bundle"))

    def chapters(self, bundle):
        return render(bundle, self.config, self.engine, self.verifier,
                      chapters_only=True, notify=self.messages.append)

    def test_selected_chapter_reports_full_book_total_and_selected_run_count(self):
        bundle = self.bundle(chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The lantern shone above the quiet harbor.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("At dawn the boat returned safely to shore.", "narration", "p"),
        ]))
        self.config["render"]["chapter_numbers"] = [2]

        result = self.chapters(bundle)

        self.assertEqual([chapter["number"] for chapter in result["chapters"]], [2])
        self.assertTrue(any("Processing chapter 2 / 2:" in line for line in self.messages))
        completion = next(line for line in self.messages if "Completed chapter 2 / 2" in line)
        self.assertIn("run elapsed", completion)
        self.assertIn("1 / 1 selected chapters ready", completion)
        self.assertFalse(any("chapter 2 / 1" in line.lower() for line in self.messages))

        self.messages.clear()
        calls = len(self.engine.speech_calls), len(self.verifier.calls)
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated accepted audio")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified accepted audio")):
            resumed = self.chapters(bundle)
        self.assertEqual(resumed["chapters"], result["chapters"])
        self.assertEqual(calls, (len(self.engine.speech_calls), len(self.verifier.calls)))
        self.assertTrue(any("Reusing completed chapter 2 / 2:" in line for line in self.messages))
        self.assertTrue(any("Completed chapter 2 / 2" in line and "run elapsed" in line
                            for line in self.messages))

    def test_logs_distinguish_new_synthesis_from_reusing_clips_during_recomposition(self):
        bundle = self.bundle(read_json(EXAMPLE / "reviewed.json"))
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
        self.engine = ParagraphFixtureEngine()
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)

        self.chapters(bundle)

        generated = [line for line in self.messages if line.strip().startswith("Generating ")
                     and ", attempt " in line]
        self.assertEqual(len(generated), len(self.engine.speech_calls) + len(self.engine.dialogue_calls))
        self.assertTrue(any("Generating speech, attempt 1/2" in line for line in generated))
        self.assertTrue(any("Generating paragraph, attempt 1/2" in line for line in generated))

        self.config["render"]["sentence_pause_ms"] += 20
        self.messages.clear()
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated speech")), \
             patch.object(self.engine, "synthesize_dialogue", side_effect=AssertionError("regenerated paragraph")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified accepted audio")):
            result = self.chapters(bundle)

        self.assertEqual(result["status"], "chapters_complete")
        self.assertTrue(any("checking speech cache" in line for line in self.messages))
        self.assertTrue(any("checking paragraph cache" in line for line in self.messages))
        self.assertTrue(any("Reusing verified speech" in line for line in self.messages))
        self.assertFalse(any("Generating " in line or "Rendering chapter" in line
                             or "preparing speech" in line or "preparing paragraph" in line
                             for line in self.messages))

    def test_take_phase_matches_work_and_cached_audio_has_no_generation_phase(self):
        text = "The lantern shone above the quiet harbor."
        request = {"kind": "progress-fixture", "text": text, "engine": self.engine.identity}
        elapsed = [0.0]

        def generate(path, seed):
            progress.refresh()
            self.assertIn("generating", self.messages[-1])
            self.assertIn("Blocks 0 / 1 complete", self.messages[-1])
            elapsed[0] += 4.0
            return self.engine._write(text, path, seed)

        original_transcribe = self.verifier.transcribe

        def transcribe(path):
            progress.refresh()
            self.assertIn("verifying", self.messages[-1])
            self.assertIn("Blocks 0 / 1 complete", self.messages[-1])
            elapsed[0] += 2.0
            return original_transcribe(path)

        with ChapterProgress(1, 1, [{"spoken_text": text}], notify=self.messages.append,
                             clock=lambda: elapsed[0], heartbeat=False) as progress, \
             patch.object(self.verifier, "transcribe", side_effect=transcribe):
            with progress.task(0) as notify:
                wav, accepted = verified_take(
                    text, generate, self.root / "take", request, self.verifier,
                    self.config, notify=notify)
        self.assertTrue(accepted["report"]["passed"])
        self.assertIn("Blocks 1 / 1 complete", self.messages[-1])
        self.assertIn("elapsed 00:06", self.messages[-1])
        self.assertIn("speech complete", self.messages[-1])
        original = wav.read_bytes(), (self.root / "take" / "accepted.json").read_bytes()

        self.messages.clear()
        with ChapterProgress(1, 1, [{"spoken_text": text}], notify=self.messages.append,
                             clock=lambda: elapsed[0], heartbeat=False) as progress, \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified accepted audio")):
            with progress.task(0) as notify:
                resumed = verified_take(
                    text, lambda *_: self.fail("regenerated accepted audio"), self.root / "take",
                    request, self.verifier, self.config, notify=notify)
                progress.refresh()
                self.assertIn("checking saved audio", self.messages[-1])
        self.assertEqual(resumed, (wav, accepted))
        self.assertEqual(original, (wav.read_bytes(), (self.root / "take" / "accepted.json").read_bytes()))
        self.assertFalse(any("generating" in item or "verifying" in item for item in self.messages))
        self.assertIn("Blocks 1 / 1 complete", self.messages[-1])

    def test_rejected_take_never_advances_completed_count(self):
        text = "The lantern shone above the quiet harbor."
        self.verifier.wrong_texts.add(text)
        request = {"kind": "progress-fixture", "text": text, "engine": self.engine.identity}

        with self.assertRaisesRegex(RenderBlocked, "after 2 attempts"):
            with ChapterProgress(1, 1, [{"spoken_text": text}], notify=self.messages.append,
                                 heartbeat=False) as progress:
                with progress.task(0) as notify:
                    verified_take(
                        text, lambda path, seed: self.engine._write(text, path, seed),
                        self.root / "take", request, self.verifier, self.config, notify=notify)

        self.assertEqual(len(self.verifier.calls), 2)
        self.assertFalse((self.root / "take" / "accepted.json").exists())
        self.assertIn("Blocks 0 / 1 complete", self.messages[-1])
        self.assertIn("speech stopped", self.messages[-1])
        self.assertFalse(any("Blocks 1 / 1 complete" in item for item in self.messages))

    def test_interrupted_paragraph_counts_only_verified_work_and_resumes_exact_audio(self):
        bundle = self.bundle(read_json(EXAMPLE / "reviewed.json"))
        # Both targets have one role/voice so they still exercise grouped
        # generation; ordinary narrator/dialogue changes stay separate.
        for row in bundle["chapters"][0]["rows"]:
            if row["block_id"] == "b000003":
                row["role_id"] = "mara_dialogue"
            elif row["block_id"] == "b000004":
                row.update(role_id="lena_dialogue", voice_id="voice_lena")
        self.config["render"].update(paragraph_synthesis=True, max_paragraph_chars=1000)
        self.engine = ParagraphFixtureEngine()
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.engine.interrupt_dialogue_number = 2

        with self.assertRaisesRegex(KeyboardInterrupt, "paragraph interruption"):
            self.chapters(bundle)

        accepted = {path: path.read_bytes()
                    for path in (self.root / "work" / "clips").glob("*/accepted.json")}
        self.assertEqual(len(accepted), 3)
        stop = next(item for item in self.messages if "speech stopped" in item)
        self.assertIn("Blocks 3 / 6 complete", stop)
        self.assertFalse(any("Completed chapter" in item for item in self.messages))
        grouped_path = next(path for path in accepted
                            if read_json(path)["request"]["kind"] == "paragraph_speech")
        wav = grouped_path.parent / read_json(grouped_path)["audio_file"]
        checksum = file_hash(wav)
        verified_before = len(self.verifier.calls)

        self.messages.clear()
        self.engine.interrupt_dialogue_number = None
        result = self.chapters(bundle)

        self.assertEqual(result["status"], "chapters_complete")
        self.assertEqual(len(self.verifier.calls) - verified_before, 3)
        self.assertEqual(len(self.engine.dialogue_calls), 3)  # two accepted, one interrupted
        self.assertEqual(len(self.engine.speech_calls), 4)
        self.assertTrue(all(path.read_bytes() == content for path, content in accepted.items()))
        self.assertEqual(file_hash(wav), checksum)
        self.assertTrue(any("Blocks 6 / 6 complete" in item and "speech complete" in item
                            for item in self.messages))
        self.assertTrue(any("Completed chapter 1 / 1" in item for item in self.messages))


if __name__ == "__main__":
    unittest.main()
