"""Replacing a published audiobook must preserve files the renderer does not own."""

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.assembly import package_audio
from epub_tts_narrator.renderer.book_output import resolve_audiobook_filename
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.publication import (check_selection, published_book, published_gaps,
                                                   publish)
from epub_tts_narrator.renderer.workflow import render
from audition_fixtures import prepare_document
from shared_fixtures import rebind_audiobook_filename, renderer_config
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
    (directory / "book.m4b").write_bytes(b"A publication test fixture, not audio.")
    audio["audiobook"] = {"path": "book.m4b", "sha256": file_hash(directory / "book.m4b")}
    write_json(directory / "audio-manifest.json", audio)
    write_json(directory / "sync.json", {})
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

    def test_custom_recorded_filename_is_accepted_as_the_owned_book(self):
        # Given an intact existing publication whose manifest records a custom basename.
        self.previous = rebind_audiobook_filename(self.output, "Existing Custom Name.m4b")

        # When validating publication ownership.
        published = published_book(self.output, "fixture-book")

        # Then the recorded file, not book.m4b, is the required owned audiobook.
        self.assertEqual(published, self.previous)
        self.assertFalse((self.output / "book.m4b").exists())

    def test_unsafe_recorded_filename_is_refused_before_replacement(self):
        # Given a manifest whose persisted name attempts to escape the publication.
        malformed = deepcopy(self.previous)
        malformed["audio"]["audiobook"] = {"path": "../outside.m4b"}
        write_json(self.output / "audio-manifest.json", malformed["audio"])
        malformed["files_sha256"]["audio-manifest.json"] = file_hash(self.output / "audio-manifest.json")
        write_json(self.output / "render.json", malformed)

        # When validating it, then the existing output remains intact and no outside file is created.
        with self.assertRaisesRegex(ValueError, "audio.audiobook.path"):
            published_book(self.output, "fixture-book")
        self.assertTrue((self.output / "book.m4b").is_file())
        self.assertFalse((self.root / "outside.m4b").exists())

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

    def test_abandoned_publish_lock_is_reclaimed_after_a_crash(self):
        stage = self.root / "ready"
        managed_output(stage)
        lock = self.root / ".output.publish.lock"
        crashed = subprocess.run([
            sys.executable, "-c",
            "import os, sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock; "
            "acquire_directory_lock(sys.argv[1]); os._exit(130)",
            str(lock),
        ])
        self.assertEqual(crashed.returncode, 130)
        self.assertTrue((lock / "owner.json").is_file())
        publish(stage, self.output)
        self.assertFalse(lock.exists())
        self.assertEqual(read_json(self.output / "render.json")["status"], "complete")

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

    def test_renderer_owned_chapter_links_are_accepted_for_replacement(self):
        links = self.output / "chapter-links"
        links.mkdir()
        link = links / "chapter-01.m4a"
        link.symlink_to(self.output / "book.m4b")
        self.assertEqual(published_book(self.output, "fixture-book"), self.previous)
        stray = self.output / "chapter-02.m4a"
        stray.symlink_to(self.output / "book.m4b")
        with self.assertRaisesRegex(ValueError, "symlink"):
            published_book(self.output, "fixture-book")
        stray.unlink()
        real = links / "chapter-02.m4a"
        real.write_bytes(b"Not a link")
        with self.assertRaisesRegex(ValueError, "chapter-links entries must be symlinks"):
            published_book(self.output, "fixture-book")
        real.unlink()
        self.assertEqual(published_book(self.output, "fixture-book"), self.previous)

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

    def test_deleted_published_files_are_recreated_instead_of_blocking_the_render(self):
        # Given a publication whose published files were deleted.
        (self.output / "book.m4b").unlink()
        (self.output / "cast.json").unlink()

        # When validating publication ownership.
        published = published_book(self.output, "fixture-book")

        # Then the deleted files are reported for recreation instead of refused.
        self.assertEqual(published, self.previous)
        missing, modified = published_gaps(self.output, published)
        self.assertEqual(missing, ["book.m4b", "cast.json"])
        self.assertEqual(modified, [])

    def test_modified_published_file_is_recreated_instead_of_blocking_the_render(self):
        # Given a published file whose content changed after publication.
        (self.output / "cast.json").write_text("{}")

        # When validating publication ownership.
        published = published_book(self.output, "fixture-book")

        # Then the changed file is reported for recreation instead of refused.
        missing, modified = published_gaps(self.output, published)
        self.assertEqual(missing, [])
        self.assertEqual(modified, ["cast.json"])

    def test_renamed_published_audiobook_keeps_its_name_and_is_not_untracked(self):
        # Given a published audiobook that was renamed outside the renderer.
        renamed = self.output / "Quest Academy.m4b"
        (self.output / "book.m4b").rename(renamed)

        # When validating publication ownership and resolving the output name.
        published = published_book(self.output, "fixture-book")

        # Then the renamed file is the owned audiobook and keeps its name.
        self.assertEqual(published, self.previous)
        self.assertEqual(resolve_audiobook_filename(previous_audio=published["audio"],
                                                     output_dir=self.output), "Quest Academy.m4b")

    def test_foreign_audiobook_file_is_not_adopted_as_the_published_book(self):
        # Given an unrelated audiobook file beside the published one.
        (self.output / "other.m4b").write_bytes(b"Not the published audiobook.")

        # When validating publication ownership, then the foreign file still blocks replacement.
        with self.assertRaisesRegex(ValueError, "untracked"):
            published_book(self.output, "fixture-book")


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class RecreatedPublicationTests(unittest.TestCase):
    def test_missing_published_audiobook_is_recreated_instead_of_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            document = prepare_document(chapter_document([
                ("CHAPTER ONE", "heading", "p"),
                ("The lantern shone above the quiet harbor.", "narration", "p"),
            ]))
            export_bundle(document, root / "bundle")
            bundle = load_bundle(root / "bundle")
            config = renderer_config(root, series_id="recreate-fixture")
            engine = FixtureEngine({})
            verifier = FixtureVerifier(engine.fixture_transcripts)

            def run():
                return render(bundle, config, engine, verifier, notify=lambda _: None)

            run()
            output = root / "output"
            record = read_json(output / "render.json")
            audiobook = next(name for name in record["files_sha256"] if name.endswith(".m4b"))
            (output / audiobook).unlink()
            speech_calls = len(engine.speech_calls)

            with patch("epub_tts_narrator.renderer.workflow.package_audio", wraps=package_audio) as package:
                run()

            # Then the audiobook is packaged again and the deleted file exists.
            self.assertEqual(package.call_count, 1)
            self.assertTrue((output / audiobook).is_file())
            # And the verified speech is reused instead of synthesized again.
            self.assertEqual(len(engine.speech_calls), speech_calls)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class VerifierProvenanceReuseTests(unittest.TestCase):
    """A verifier update is provenance, not a package input: restored chapters
    keep their accepted takes and reports, so verification-side changes must
    not re-encode a completed book."""

    def completed_book(self, root, verifier_version=1, title="Fixture"):
        document = prepare_document(chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("The lantern shone above the quiet harbor.", "narration", "p"),
        ]))
        document["source"]["book"]["title"] = title
        export_bundle(document, root / "bundle")
        return self.render_bundle(root, verifier_version)

    def render_bundle(self, root, verifier_version=1):
        bundle = load_bundle(root / "bundle")
        config = renderer_config(root, series_id="verifier-provenance")
        engine = FixtureEngine({})
        verifier = FixtureVerifier(engine.fixture_transcripts, version=verifier_version)
        render(bundle, config, engine, verifier, notify=lambda _: None)
        return bundle, config, engine

    def test_verifier_update_alone_reuses_the_completed_audiobook(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bundle, config, engine = self.completed_book(root, verifier_version=1)
            output = root / "output"
            audiobook = next(output.glob("*.m4b"))
            saved_bytes = audiobook.read_bytes()
            saved_mtime = audiobook.stat().st_mtime_ns
            speech_calls = len(engine.speech_calls)
            messages = []
            verifier = FixtureVerifier(engine.fixture_transcripts, version=2)

            with patch("epub_tts_narrator.renderer.workflow.package_audio", wraps=package_audio) as package:
                result = render(bundle, config, engine, verifier, notify=messages.append)

            # Then the completed publication is reused without repackaging.
            self.assertEqual(package.call_count, 0)
            self.assertEqual(len(engine.speech_calls), speech_calls)
            self.assertEqual(audiobook.read_bytes(), saved_bytes)
            self.assertEqual(audiobook.stat().st_mtime_ns, saved_mtime)
            self.assertTrue(any("reusing" in message for message in messages))
            self.assertEqual(result["verifier"], {"backend": "fixture-verifier", "version": 1})

    def test_verifier_update_with_changed_bundle_still_repackages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            bundle, config, engine = self.completed_book(root, verifier_version=1)
            output = root / "output"
            audiobook = next(output.glob("*.m4b"))
            saved_mtime = audiobook.stat().st_mtime_ns
            # A real source change: export the renamed book and render it
            # against the same work and output directories.
            document = prepare_document(chapter_document([
                ("CHAPTER ONE", "heading", "p"),
                ("The lantern shone above the quiet harbor.", "narration", "p"),
            ]))
            document["source"]["book"]["title"] = "Renamed Title"
            export_bundle(document, root / "bundle-renamed")
            bundle = load_bundle(root / "bundle-renamed")
            config = renderer_config(root, series_id="verifier-provenance")
            engine = FixtureEngine({})
            verifier = FixtureVerifier(engine.fixture_transcripts, version=2)

            with patch("epub_tts_narrator.renderer.workflow.package_audio", wraps=package_audio) as package:
                render(bundle, config, engine, verifier, notify=lambda _: None)

            self.assertEqual(package.call_count, 1)
            self.assertNotEqual(audiobook.stat().st_mtime_ns, saved_mtime)


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
