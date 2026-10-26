"""A clean edition changes assembly only, including toggles of existing books."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.chapter_cache import load_chapter
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.workflow import assemble, render
from shared_fixtures import renderer_config
from test_chapter_export import chapter_document
from test_renderer_workflow import FixtureEngine, FixtureVerifier, FFMPEG, FFPROBE
from audition_fixtures import prepare_bundle


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class CensorshipWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        document = chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("Oh fuck this is still the same story.", "narration", "p"),
        ])
        export_bundle(document, self.root / "bundle")
        self.bundle = prepare_bundle(load_bundle(self.root / "bundle"))
        self.config = renderer_config(self.root, series_id="censorship-fixture",
                                      normalize_loudness=False, normalize_pauses=False)
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)

    def render(self):
        return render(self.bundle, self.config, self.engine, self.verifier,
                      chapters_only=True, notify=lambda _: None)

    def saved(self, result):
        return load_chapter(result["chapters"][0]["cache_directory"])

    def originals(self):
        return {str(p): file_hash(p) for p in (self.root / "work" / "clips").rglob("*") if p.is_file()}

    def test_assemble_on_and_off_retains_speech_pcm_timing_and_resumes_exact_edition(self):
        first = self.render()
        plain = self.saved(first)
        original, calls = self.originals(), (len(self.engine.speech_calls), len(self.verifier.calls))
        self.config["censorship"]["enabled"] = True
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No speech verification")), \
             patch("epub_tts_narrator.renderer.verification.create_verifier", side_effect=AssertionError("No ASR")), \
             patch("epub_tts_narrator.renderer.engines.create_engine", side_effect=AssertionError("No TTS")):
            second = assemble(self.bundle, self.config, chapters_only=True, notify=lambda _: None)
            clean = self.saved(second)
            self.assertNotEqual(clean["pcm_sha256"], plain["pcm_sha256"])
            self.assertEqual(clean["frames"], plain["frames"])
            self.assertEqual([(s["chapter_start_frame"], s["chapter_end_frame"]) for s in clean["segments"]],
                             [(s["chapter_start_frame"], s["chapter_end_frame"]) for s in plain["segments"]])
            self.assertEqual(sum(len(lane["audit"]["masks"]) for segment in clean["segments"]
                                 for lane in segment["censorship"]["lanes"]), 1)
            self.assertEqual(assemble(self.bundle, self.config, chapters_only=True, notify=lambda _: None), second)
            self.config["censorship"]["enabled"] = False
            restored = assemble(self.bundle, self.config, chapters_only=True, notify=lambda _: None)
        self.assertEqual(self.saved(restored)["pcm_sha256"], plain["pcm_sha256"])
        self.assertEqual(original, self.originals())
        self.assertEqual(calls, (len(self.engine.speech_calls), len(self.verifier.calls)))

    def test_render_policy_change_reuses_all_accepted_takes_without_tts_or_asr(self):
        first = self.render()
        original = self.originals()
        self.config["censorship"]["enabled"] = True
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("No TTS")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("No ASR")):
            second = self.render()
        self.assertEqual(original, self.originals())
        self.assertNotEqual(self.saved(first)["pcm_sha256"], self.saved(second)["pcm_sha256"])

    def test_censored_reassembly_can_publish_first_book_then_restore_original_edition(self):
        first = self.render()
        originals = self.originals()
        self.config["censorship"]["enabled"] = True
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            clean = assemble(self.bundle, self.config, notify=lambda _: None)
            self.assertEqual(clean["status"], "complete")
            sync = read_json(self.root / "output" / "sync.json")
            self.assertTrue(any(segment.get("censorship") for segment in sync["segments"]))
            self.config["censorship"]["enabled"] = False
            restored = assemble(self.bundle, self.config, notify=lambda _: None)
        self.assertEqual(restored["status"], "complete")
        sync = read_json(self.root / "output" / "sync.json")
        self.assertFalse(any(segment.get("censorship") for segment in sync["segments"]))
        self.assertEqual(self.saved(first)["frames"], sync["chapters"][0]["frames"])
        self.assertEqual(originals, self.originals())

    def test_alignment_failure_keeps_accepted_takes_and_resume_does_not_regenerate(self):
        from epub_tts_narrator.renderer.censorship_alignment import CensorshipAlignmentUnavailable
        # TTS/ASR fixture boundaries stay external; real verification accepts
        # exact speech even if the recognizer did not provide word timings.
        original_transcribe = self.verifier.transcribe
        def untimed(path):
            result = original_transcribe(path)
            for segment in result["segments"]:
                segment.pop("words", None)
            return result
        self.verifier.transcribe = untimed
        self.config["censorship"]["enabled"] = True
        with self.assertRaises(CensorshipAlignmentUnavailable):
            self.render()
        accepted = self.originals()
        self.assertTrue(any(name.endswith("accepted.json") for name in accepted))
        calls = len(self.engine.speech_calls)
        self.config["censorship"]["enabled"] = False
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("No TTS")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("No ASR")):
            result = self.render()
        self.assertEqual(result["status"], "chapters_complete")
        self.assertEqual(len(self.engine.speech_calls), calls)
        self.assertEqual(accepted, self.originals())
