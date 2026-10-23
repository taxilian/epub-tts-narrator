"""Replacing a published audiobook must preserve files the renderer does not own."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.assembly import package_audio
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.publication import check_selection, published_book, publish
from epub_tts_narrator.renderer.workflow import render
from audition_fixtures import prepare_document
from shared_fixtures import renderer_config
from test_chapter_export import chapter_document
from test_renderer_workflow import FixtureEngine, FixtureVerifier, FFMPEG, FFPROBE


def managed_output(directory, *, chapter_ids=("chapter-0001", "chapter-0002"), partial=False):
    """Metadata/file fixtures for publication ownership checks, not playable audio."""
    directory.mkdir()
    book = {"id": "fixture-book", "title": "Fixture"}
    audio = {"book": book, "chapters": [{"id": identifier} for identifier in chapter_ids]}
    write_json(directory / "source.json", {"book": book})
    write_json(directory / "cast.json", {})
    write_json(directory / "verification.json", [])
    write_json(directory / "audio-manifest.json", audio)
    write_json(directory / "sync.json", {})
    (directory / "book.m4b").write_bytes(b"A publication test fixture, not audio.")
    result = {
        "format": "audiobook-render", "version": 1, "status": "complete",
        "partial_book": partial, "audio": audio,
        "files_sha256": {path.name: file_hash(path) for path in directory.iterdir()},
    }
    write_json(directory / "render.json", result)
    return result


class PublicationSafetyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.previous = managed_output(self.output)

    def test_untracked_files_and_even_empty_directories_block_replacement(self):
        for name, is_directory in (("notes.txt", False), ("personal-notes", True)):
            with self.subTest(name=name):
                unknown = self.output / name
                unknown.mkdir() if is_directory else unknown.write_text("Do not delete")
                with self.assertRaisesRegex(ValueError, "untracked"):
                    published_book(self.output, "fixture-book")
                self.assertTrue(unknown.exists())
                unknown.rmdir() if is_directory else unknown.unlink()

    def test_symlinked_output_or_child_is_never_followed_for_replacement(self):
        link = self.root / "linked-output"
        link.symlink_to(self.output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "regular directory"):
            published_book(link, "fixture-book")
        saved = self.output / "book.m4b"
        original = saved.read_bytes()
        external = self.root / "external.m4b"
        external.write_bytes(original)
        saved.unlink()
        saved.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            published_book(self.output, "fixture-book")
        self.assertEqual(external.read_bytes(), original)

    def test_partial_render_cannot_replace_a_complete_or_different_subset(self):
        with self.assertRaisesRegex(ValueError, "different chapters"):
            check_selection(self.previous, ["chapter-0002"], partial=True)
        partial = deepcopy(self.previous)
        partial["partial_book"] = True
        partial["audio"]["chapters"] = [{"id": "chapter-0001"}]
        with self.assertRaisesRegex(ValueError, "different chapters"):
            check_selection(partial, ["chapter-0002"], partial=True)
        check_selection(partial, ["chapter-0001"], partial=True)
        check_selection(partial, ["chapter-0001", "chapter-0002"], partial=False)

    def test_publish_checks_partial_selection_again_after_staging(self):
        stage = self.root / "ready"
        managed_output(stage, chapter_ids=("chapter-0002",), partial=True)
        with self.assertRaisesRegex(ValueError, "different chapters"):
            publish(stage, self.output, expected=digest(self.previous))
        self.assertEqual(read_json(self.output / "render.json"), self.previous)
        self.assertTrue(stage.is_dir())
        self.assertFalse((self.root / ".output.publish.lock").exists())

    def test_output_revised_during_render_is_not_overwritten(self):
        stage = self.root / "ready"
        managed_output(stage)
        changed = dict(self.previous, publication_key="changed-after-preflight")
        write_json(self.output / "render.json", changed)
        with self.assertRaisesRegex(ValueError, "changed during rendering"):
            publish(stage, self.output, expected=digest(self.previous))
        self.assertEqual(read_json(self.output / "render.json"), changed)
        self.assertTrue(stage.is_dir())
        self.assertFalse(list(self.root.glob(".output.previous-*")))

    def test_output_removed_during_render_is_not_recreated_without_a_fresh_run(self):
        stage = self.root / "ready"
        managed_output(stage)
        shutil.rmtree(self.output)
        with self.assertRaisesRegex(ValueError, "removed during rendering"):
            publish(stage, self.output, expected=digest(self.previous))
        self.assertFalse(self.output.exists())
        self.assertTrue(stage.is_dir())

    def test_output_that_appeared_after_preflight_is_not_overwritten(self):
        stage = self.root / "ready"
        managed_output(stage)
        with self.assertRaisesRegex(ValueError, "changed during rendering"):
            publish(stage, self.output, expected=None)
        self.assertEqual(read_json(self.output / "render.json"), self.previous)

    def test_malformed_chapter_metadata_is_a_controlled_validation_failure(self):
        for chapters in (None, "chapters", [None], [{}], [{"id": 1}]):
            with self.subTest(chapters=chapters):
                malformed = deepcopy(self.previous)
                malformed["audio"]["chapters"] = chapters
                write_json(self.output / "audio-manifest.json", malformed["audio"])
                malformed["files_sha256"]["audio-manifest.json"] = file_hash(self.output / "audio-manifest.json")
                write_json(self.output / "render.json", malformed)
                with self.assertRaises(ValueError):
                    published_book(self.output, "fixture-book")


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class LegacyPublicationTests(unittest.TestCase):
    def test_legacy_render_without_publication_key_refreshes_once_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            document = prepare_document(chapter_document([
                ("CHAPTER ONE", "heading", "p"),
                ("The lantern shone above the quiet harbor.", "narration", "p"),
            ]))
            export_bundle(document, root / "bundle")
            bundle = load_bundle(root / "bundle")
            config = renderer_config(root, series_id="legacy-fixture")
            engine = FixtureEngine({})
            verifier = FixtureVerifier(engine.fixture_transcripts)

            def run():
                return render(bundle, config, engine, verifier, notify=lambda _: None)

            run()
            record = read_json(root / "output" / "render.json")
            del record["publication_key"]
            write_json(root / "output" / "render.json", record)
            counts = len(engine.voice_calls), len(engine.speech_calls), len(verifier.calls)
            with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")), \
                 patch("epub_tts_narrator.renderer.workflow.package_audio", wraps=package_audio) as package:
                refreshed = run()
                self.assertEqual(package.call_count, 1)
                self.assertIn("publication_key", refreshed)
                self.assertEqual(run(), refreshed)
                self.assertEqual(package.call_count, 1)
            self.assertEqual(counts, (len(engine.voice_calls), len(engine.speech_calls), len(verifier.calls)))


if __name__ == "__main__":
    unittest.main()
