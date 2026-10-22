"""Recovery suggestions must be executable by the active chapter coordinator."""

from copy import deepcopy
import unittest

from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
from epub_tts_narrator.renderer.common import digest, file_hash
from epub_tts_narrator.renderer.workflow import _render_chapter_with_recovery
import test_renderer_paragraph_pronunciation as fixtures
from test_renderer_workflow import FFMPEG, FFPROBE, write_tone


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
class ParagraphSplitCapabilityTests(unittest.TestCase):
    def setUp(self):
        fixtures.ParagraphPronunciationIntegrationTests.setUp(self)
        self.engine.capabilities = {"multi_speaker": True}
        reference = self.work / "reference.wav"
        write_tone(reference)
        self.voice.update(reference_audio=str(reference), reference_sha256=file_hash(reference))
        self.chapter = {"id": "chapter-0001", "number": 1, "title": "Fixture",
                        "chunks": [deepcopy(self.chunk)]}

    def test_remote_executor_never_offers_unavailable_paragraph_split(self):
        executor = ChapterExecutor(self.config, engine=self.engine, verifier=self.verifier)
        self.addCleanup(executor.close)
        spec = {"version": 1, "book": {"id": "fixture", "title": "Fixture"},
                "source_fingerprint": digest("source"), "chapter": self.chapter, "total_chapters": 1,
                "voice_bindings": {"voice": self.voice["asset_hash"]}, "render": {},
                "renderer_fingerprint": digest(executor.identity)}
        prepared = executor.synthesize(spec, {"voice": self.voice}, self.work, self.messages.append)
        self.assertTrue(prepared["acceptance"][0]["report"]["passed"])
        self.assertTrue(self.client.plans)
        for plan in self.client.plans:
            self.assertNotIn("split", plan["permitted_actions"])
            self.assertIn("rewrite", plan["permitted_actions"])

    def test_local_recovery_coordinator_still_offers_paragraph_split(self):
        _, _, acceptance = _render_chapter_with_recovery(
            {"chapters": [self.chapter]}, self.chapter, {"voice": self.voice}, self.engine,
            self.engine.identity, self.verifier, self.config, self.work, notify=self.messages.append)
        self.assertTrue(acceptance[0]["report"]["passed"])
        self.assertTrue(self.client.plans)
        for plan in self.client.plans:
            self.assertIn("split", plan["permitted_actions"])


if __name__ == "__main__":
    unittest.main()
