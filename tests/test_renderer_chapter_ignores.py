"""Chapter ignores through local render, local assemble, and the remote chapter client."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator import chapter_ignores
from epub_tts_narrator.renderer import chapter_client
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import file_hash, read_json
from epub_tts_narrator.renderer.workflow import RenderBlocked, assemble, render
from audition_fixtures import prepare_document
from shared_fixtures import renderer_config
import test_renderer_workflow as fixtures
from test_chapter_ignores import exported_book, multi_chapter_document


def notify_noop(_):
    return None


class ChapterIgnoreFixture(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp).resolve()
        self.config = renderer_config(self.root, series_id="fixture-series",
                                      checkpoint="unused-tts-model", asr_model="unused-asr-model")
        self.directory = Path(exported_book(prepare_document(multi_chapter_document()), self.root))
        self.bundle = load_bundle(self.directory)
        self.fixtures = {}
        self.engine = fixtures.FixtureEngine(self.fixtures)
        self.verifier = fixtures.FixtureVerifier(self.fixtures)

    def ignore(self, index, ignored=True):
        return chapter_ignores.set_ignored(self.directory, chapter_ignores.bundle_document(self.bundle),
                                           index, ignored, expected_manifest_hash=self.bundle["bundle_hash"])

    def render_chapters(self):
        return render(self.bundle, self.config, self.engine, self.verifier,
                      notify=notify_noop, chapters_only=True)

    def checkpoint(self, number):
        return Path(self.config["render"]["work_dir"]) / "completed-chapters" / f"chapter-{number:04d}.json"

    def audio_manifest(self):
        return read_json(Path(self.config["render"]["output_dir"]) / "audio-manifest.json")["chapters"]

    def manifest_records_by_number(self):
        return {record["number"]: record for record in self.audio_manifest()}


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe are required for workflow integration")
class LocalChapterIgnoreRenderTests(ChapterIgnoreFixture):
    def test_rendering_an_ignored_chapter_preserves_its_checkpoint_and_assembly_omits_it(self):
        self.render_chapters()
        speech_after_first = len(self.engine.speech_calls)
        saved_checkpoint = file_hash(self.checkpoint(2))
        self.ignore(1)
        result = self.render_chapters()
        self.assertEqual(len(self.engine.speech_calls), speech_after_first)
        self.assertEqual(file_hash(self.checkpoint(2)), saved_checkpoint)
        self.assertEqual([chapter["number"] for chapter in result["chapters"]], [1, 2, 3])
        published = assemble(self.bundle, self.config, notify=notify_noop)
        records = self.manifest_records_by_number()
        self.assertTrue(records[2]["skipped"])
        self.assertFalse(records[2]["path"])
        self.assertFalse(records[1]["skipped"] or records[3]["skipped"])
        self.assertGreater(records[1]["frames"] + records[3]["frames"], 0)
        self.assertEqual(published["engine"]["engine"], "fixture")
        # Unignoring restores the preserved checkpoint without new synthesis.
        self.ignore(1, False)
        self.render_chapters()
        self.assertEqual(len(self.engine.speech_calls), speech_after_first)

    def test_ignoring_before_the_first_render_never_synthesizes_or_checkpoints_the_chapter(self):
        self.ignore(1)
        self.render_chapters()
        self.assertFalse(self.checkpoint(2).exists())
        spoken = " ".join(call["text"] for call in self.engine.speech_calls)
        self.assertNotIn("tide is turning", spoken)
        self.assertTrue(self.checkpoint(1).exists() and self.checkpoint(3).exists())
        assemble(self.bundle, self.config, notify=notify_noop)
        records = self.manifest_records_by_number()
        self.assertTrue(records[2]["skipped"])
        self.assertFalse(records[1]["skipped"] or records[3]["skipped"])

    def test_assembling_a_fully_ignored_book_refuses_with_a_clear_message(self):
        for index in range(3):
            self.ignore(index)
        self.render_chapters()
        self.assertEqual(len(self.engine.speech_calls), 0)
        with self.assertRaisesRegex(RenderBlocked, "Every selected chapter is ignored"):
            assemble(self.bundle, self.config, notify=notify_noop)

    def test_corrupt_ignored_checkpoint_is_bypassed_until_unignored(self):
        self.render_chapters()
        checkpoint = self.checkpoint(2)
        checkpoint.write_text('{"corrupt": ')
        corrupted = checkpoint.read_bytes()
        self.ignore(1)
        lines = []
        result = render(self.bundle, self.config, self.engine, self.verifier,
                        notify=lines.append, chapters_only=True)
        self.assertEqual([chapter["number"] for chapter in result["chapters"]], [1, 2, 3])
        self.assertEqual(result["chapter_count"], 3)
        self.assertIn("Saved 2 chapter checkpoint(s)", "\n".join(lines))
        self.assertEqual(checkpoint.read_bytes(), corrupted)
        assemble(self.bundle, self.config, notify=notify_noop)
        self.assertTrue(self.manifest_records_by_number()[2]["skipped"])
        # Unignoring restores the ordinary integrity checks for the same bytes:
        # the plan-time synthesis-identity projection fails like any corrupt
        # checkpoint of an included chapter.
        self.ignore(1, False)
        with self.assertRaises(RenderBlocked) as raised:
            self.render_chapters()
        self.assertIn("Invalid saved synthesis identity", str(raised.exception))


@unittest.skipUnless(fixtures.FFMPEG and fixtures.FFPROBE, "FFmpeg and ffprobe are required for workflow integration")
class RemoteChapterIgnoreTests(ChapterIgnoreFixture):
    def setUp(self):
        super().setUp()
        self.render_chapters()

    def test_remote_paths_materialize_ignored_chapters_without_the_service(self):
        local_chapters = []
        original_local = chapter_client._local_chapter

        def remember_local(bundle, chapter, *args, **kwargs):
            local_chapters.append(chapter["number"])
            return original_local(bundle, chapter, *args, **kwargs)

        self.ignore(1)
        with patch.object(chapter_client, "ChapterClient", side_effect=AssertionError("service touched")), \
                patch.object(chapter_client, "_local_chapter", side_effect=remember_local):
            rendered = chapter_client.render_remote(self.bundle, self.config, notify=notify_noop, chapters_only=True)
            self.assertEqual([chapter["number"] for chapter in rendered["chapters"]], [1, 2, 3])
            self.assertEqual(local_chapters, [1, 3])
            published = chapter_client.assemble_remote(self.bundle, self.config, notify=notify_noop)
        records = self.manifest_records_by_number()
        self.assertTrue(records[2]["skipped"])
        self.assertFalse(records[1]["skipped"] or records[3]["skipped"])
        self.assertEqual(published["engine"]["engine"], "fixture")
        # The untouched checkpoint restores the chapter once the choice is cleared.
        self.ignore(1, False)
        with patch.object(chapter_client, "ChapterClient", side_effect=AssertionError("service touched")):
            chapter_client.assemble_remote(self.bundle, self.config, notify=notify_noop)
        records = self.manifest_records_by_number()
        self.assertFalse(any(record["skipped"] for record in records.values()))

    def test_remote_assembly_of_a_fully_ignored_book_refuses_without_the_service(self):
        with patch.object(chapter_client, "ChapterClient", side_effect=AssertionError("service touched")):
            for index in range(3):
                self.ignore(index)
            with self.assertRaisesRegex(RenderBlocked, "Every selected chapter is ignored"):
                chapter_client.assemble_remote(self.bundle, self.config, notify=notify_noop)


if __name__ == "__main__":
    unittest.main()
