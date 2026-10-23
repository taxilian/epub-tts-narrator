from copy import deepcopy
import threading
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import workflow
import test_renderer_workflow as fixtures


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe required")
class LocalChapterOverlapTests(unittest.TestCase):
    def setUp(self):
        fixtures.RendererWorkflowTests.setUp(self)
        original = self.bundle["chapters"][0]
        self.bundle["chapters"] = []
        for number, rows in enumerate((original["rows"][:4], original["rows"][4:]), 1):
            entry = deepcopy(original["entry"])
            entry.update(title=f"Fixture chapter {number}", path=f"scripts/chapter-{number:04d}.jsonl")
            self.bundle["chapters"].append({"entry": entry, "rows": rows})

    def test_next_chapter_synthesizes_before_previous_chapter_finishes_encoding(self):
        assembling = threading.Event()
        following = threading.Event()
        original_prepare = workflow.prepare_chapter
        original_render = workflow._render_chapter_with_recovery
        observed = []

        def prepare(chapter, *args, **kwargs):
            if chapter["number"] == 1:
                assembling.set()
                observed.append(following.wait(5))
            return original_prepare(chapter, *args, **kwargs)

        def synthesize(bundle, chapter, *args, **kwargs):
            if chapter["number"] == 2:
                self.assertTrue(assembling.wait(5))
                following.set()
            return original_render(bundle, chapter, *args, **kwargs)

        with patch.object(workflow, "prepare_chapter", side_effect=prepare), \
                patch.object(workflow, "_render_chapter_with_recovery", side_effect=synthesize):
            result = workflow.render(self.bundle, self.config, self.engine, self.verifier,
                                     notify=lambda _: None, chapters_only=True)

        self.assertEqual(observed, [True])
        self.assertEqual([item["number"] for item in result["chapters"]],
                         list(range(1, len(self.bundle["chapters"]) + 1)))
        calls = len(self.engine.speech_calls)
        workflow.render(self.bundle, self.config, self.engine, self.verifier,
                        notify=lambda _: None, chapters_only=True)
        self.assertEqual(len(self.engine.speech_calls), calls)

    def test_failed_encoding_drains_before_run_locks_are_released(self):
        finished = threading.Event()

        def prepare(*args, **kwargs):
            finished.set()
            raise RuntimeError("fixture encoding failed")

        with patch.object(workflow, "prepare_chapter", side_effect=prepare):
            with self.assertRaisesRegex(RuntimeError, "fixture encoding failed"):
                workflow.render(self.bundle, self.config, self.engine, self.verifier,
                                notify=lambda _: None, chapters_only=True)
        self.assertTrue(finished.is_set())
        result = workflow.render(self.bundle, self.config, self.engine, self.verifier,
                                 notify=lambda _: None, chapters_only=True)
        self.assertEqual(result["status"], "chapters_complete")


if __name__ == "__main__":
    unittest.main()
