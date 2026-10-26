"""Chapter-only assembly still materializes changed encoding settings."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.chapter_cache import load_chapter, reencode_chapter
from epub_tts_narrator.renderer.common import file_hash
from epub_tts_narrator.renderer.workflow import assemble, render
from shared_fixtures import renderer_config
from test_renderer_workflow import EXAMPLE, FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class CensorshipEncodingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = prepare_bundle(load_bundle(EXAMPLE / "bundle"))
        self.config = renderer_config(self.root, series_id="encoding-fixture", chapter_numbers=[1])
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)

    def test_chapter_only_encoding_change_creates_playable_file_without_regeneration(self):
        original = render(self.bundle, self.config, self.engine, self.verifier,
                          chapters_only=True, notify=lambda _: None)
        self.config["render"]["chapter_format"] = "mp3"
        with patch.object(self.engine, "synthesize", side_effect=AssertionError("regenerated speech")), \
             patch.object(self.verifier, "transcribe", side_effect=AssertionError("reverified speech")), \
             patch("epub_tts_narrator.renderer.chapter_cache.compose_chapter", side_effect=AssertionError("recomposed PCM")):
            updated = assemble(self.bundle, self.config, chapters_only=True, notify=lambda _: None)
            again = assemble(self.bundle, self.config, chapters_only=True, notify=lambda _: None)
        path = Path(updated["chapters"][0]["audio_file"])
        self.assertEqual(path.suffix, ".mp3")
        self.assertTrue(path.is_file())
        self.assertEqual(updated, again)
        before = load_chapter(original["chapters"][0]["cache_directory"])
        after = load_chapter(updated["chapters"][0]["cache_directory"])
        self.assertEqual(before["pcm_sha256"], after["pcm_sha256"])
        self.assertEqual(before["segments"], after["segments"])
        self.assertEqual(before["normalization"], after["normalization"])

    def test_reencode_helper_only_reads_chapter_pcm_and_can_resume_without_takes(self):
        original = render(self.bundle, self.config, self.engine, self.verifier,
                          chapters_only=True, notify=lambda _: None)
        before = load_chapter(original["chapters"][0]["cache_directory"])
        settings = dict(self.config["render"], chapter_format="mp3")
        context = deepcopy(before["context"])
        from epub_tts_narrator.renderer.chapter_cache import chapter_settings
        context["settings"] = chapter_settings(settings)
        with patch("epub_tts_narrator.renderer.chapter_cache.compose_chapter", side_effect=AssertionError("read original clips")):
            updated = reencode_chapter(before, self.root / "other-cache", book=self.bundle["source"]["book"],
                                       render_settings=settings, context=context, notify=lambda _: None)
            again = reencode_chapter(before, self.root / "other-cache", book=self.bundle["source"]["book"],
                                     render_settings=settings, context=context, notify=lambda _: None)
        self.assertEqual(updated, again)
        self.assertEqual(file_hash(before["wav_path"]), file_hash(updated["wav_path"]))
        with self.assertRaisesRegex(ValueError, "PCM settings"):
            reencode_chapter(before, self.root / "other-cache", book=self.bundle["source"]["book"],
                             render_settings=dict(settings, paragraph_pause_ms=123), context=context)
