"""Published artwork repair preserves accepted audio, timing, and ownership."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from epub_tts_narrator.renderer.assembly import compose_chapter, package_audio, probe_audio
from epub_tts_narrator.renderer.cli import main
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.publication import published_book, publish
from epub_tts_narrator.renderer.refresh_metadata import _audio_payload_hash, refresh_metadata
from epub_tts_narrator.source_epub import source_epub_hint
from shared_fixtures import FFMPEG, FFPROBE, renderer_config, snapshot
import test_renderer_assembly as audio_fixtures


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg/FFprobe required")
class RefreshMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="metadata refresh é ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.epub, _ = audio_fixtures.CoverExtractionTests.epub(self)
        with zipfile.ZipFile(self.epub) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        files["OPS/book.opf"] = files["OPS/book.opf"].replace(b"<metadata>", b'''<metadata
            xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>The Book &amp; the Sea</dc:title>
            <dc:creator>Author One</dc:creator><dc:creator>Author Two</dc:creator>
            <dc:language>en-US</dc:language><dc:date>2024-04-03</dc:date>
            <dc:publisher>Fixture Press</dc:publisher><dc:rights>Copyright Authors</dc:rights>
            <dc:description>A description from the EPUB.</dc:description>
            <dc:identifier>urn:isbn:9780000000000</dc:identifier>
            <meta name="calibre:series" content="Fixture Series"/>
            <meta name="calibre:series_index" content="4"/>
        ''')
        with zipfile.ZipFile(self.epub, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        self.book_id = file_hash(self.epub)
        self.directory = self.root / "bundle"
        self.directory.mkdir()
        self.source = {"book": {"id": self.book_id, "title": "Legacy title",
                                "authors": ["Legacy author"]}}
        self.manifest = {"format": "refresh-test", "version": 1}
        write_json(self.directory / "manifest.json", self.manifest)
        self.bundle = {"directory": self.directory, "source": self.source,
                       "manifest": self.manifest, "bundle_hash": digest(self.manifest)}
        self.config = renderer_config(epub=str(self.epub))
        self.output = self.directory / "audiobook"

    def publish_original(self):
        wav = audio_fixtures.make_wav(self.root / "sample.wav", frames=11025)
        chapters = []
        for number in (1, 2):
            chapter = compose_chapter(
                [audio_fixtures.clip(wav, identifier=f"segment-{number}")],
                self.root / f"chapter-{number}.wav")
            chapter.update(id=f"chapter-{number}", title=f"Chapter {number}",
                           script_path=f"scripts/chapter-{number:04d}.jsonl")
            chapter["segments"][0]["words"] = [
                {"text": "Hello", "start_frame": 0, "end_frame": 11025,
                 "source_start": 0, "source_end": 5}]
            chapters.append(chapter)
        audio = package_audio(chapters, self.output, book=self.source["book"],
                              source_fingerprint="fixture-source", ffmpeg=FFMPEG, ffprobe=FFPROBE)
        write_json(self.output / "source.json", self.source)
        write_json(self.output / "cast.json", {"voices": [{"id": "voice_1"}]})
        write_json(self.output / "verification.json", {"passed": True, "fixture": "retained"})
        result = {"format": "audiobook-render", "version": 1, "status": "complete",
                  "partial_book": False, "publication_key": "previous-publication-key",
                  "audio": audio,
                  "files_sha256": {path.relative_to(self.output).as_posix(): file_hash(path)
                                   for path in self.output.rglob("*") if path.is_file()}}
        write_json(self.output / "render.json", result)
        self.assertEqual(published_book(self.output, self.book_id), result)
        return result

    def assert_no_staging(self):
        self.assertFalse(list(self.directory.glob(".audiobook-metadata-*")))
        self.assertFalse(list(self.directory.glob(".audiobook.previous-*")))
        self.assertFalse((self.directory / ".audiobook.publish.lock").exists())
        self.assertFalse((self.directory / ".audio-render.lock").exists())

    def test_refresh_adds_square_cover_and_epub_tags_without_changing_audio_or_timing(self):
        previous = self.publish_original()
        links = self.output / "chapter-links"
        links.mkdir()
        for number in (1, 2):
            (links / f"chapter-{number:02d}.m4a").symlink_to(
                f"../chapters/chapter-{number:04d}.m4a")
        saved = snapshot(self.output)
        original_sync = read_json(self.output / "sync.json")
        original_probe = probe_audio(self.output / "book.m4b", ffprobe=FFPROBE)
        original_payload = _audio_payload_hash(self.output / "book.m4b", FFMPEG)
        config_before = deepcopy(self.config)
        result = refresh_metadata(self.bundle, self.config, notify=lambda message: None)

        self.assertEqual(self.config, config_before)
        self.assertEqual(_audio_payload_hash(self.output / "book.m4b", FFMPEG), original_payload)
        self.assertEqual(result, published_book(self.output, self.book_id))
        self.assertEqual(result["audio"]["chapters"], previous["audio"]["chapters"])
        for number in (1, 2):
            link = links / f"chapter-{number:02d}.m4a"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), self.output / "chapters" / f"chapter-{number:04d}.m4a")
            self.assertNotIn(link.relative_to(self.output).as_posix(), result["files_sha256"])
        self.assertNotIn("publication_key", result)
        checksum = file_hash(self.output / "book.m4b")
        self.assertEqual(read_json(self.output / "sync.json"),
                         dict(original_sync, audio_sha256=checksum))
        self.assertEqual(result["audio"]["audiobook"]["sha256"], checksum)
        self.assertEqual(result["audio"]["sync"]["sha256"], file_hash(self.output / "sync.json"))
        self.assertTrue(result["audio"]["audiobook"]["cover_embedded"])
        self.assertNotEqual(saved["book.m4b"], (self.output / "book.m4b").read_bytes())
        for name, data in saved.items():
            if name not in {"book.m4b", "sync.json", "audio-manifest.json", "render.json"}:
                self.assertEqual((self.output / name).read_bytes(), data, name)
        inspection = probe_audio(self.output / "book.m4b", ffprobe=FFPROBE)
        self.assertEqual(inspection["chapters"], original_probe["chapters"])
        covers = [stream for stream in inspection["streams"]
                  if stream.get("disposition", {}).get("attached_pic")]
        self.assertEqual(len(covers), 1)
        self.assertEqual((covers[0]["width"], covers[0]["height"]), (1600, 1600))
        tags = inspection["format"]["tags"]
        for key, value in {"title": "The Book & the Sea", "album": "The Book & the Sea",
                           "artist": "Author One; Author Two", "album_artist": "Author One; Author Two",
                           "date": "2024-04-03", "copyright": "Copyright Authors",
                           "description": "A description from the EPUB.",
                           "grouping": "Fixture Series", "media_type": "2"}.items():
            self.assertEqual(tags.get(key), value, key)
        self.assertEqual(result["audio"]["book"]["publisher"], "Fixture Press")
        self.assertEqual(result["audio"]["book"]["identifiers"], ["urn:isbn:9780000000000"])
        self.assertEqual(source_epub_hint(self.directory, self.book_id), self.epub)
        for name in ("artwork/cover.jpg", "artwork/original-cover.png"):
            self.assertIn(name, result["files_sha256"])
        self.assert_no_staging()

    def test_remux_failure_preserves_the_complete_previous_publication(self):
        self.publish_original()
        before = snapshot(self.output)
        with patch("epub_tts_narrator.renderer.refresh_metadata._run",
                   side_effect=ValueError("FFmpeg conversion failed")):
            with self.assertRaisesRegex(ValueError, "FFmpeg conversion failed"):
                refresh_metadata(self.bundle, self.config, notify=lambda message: None)
        self.assertEqual(snapshot(self.output), before)
        self.assertIsNotNone(published_book(self.output, self.book_id))
        self.assertIsNone(source_epub_hint(self.directory, self.book_id))
        self.assert_no_staging()

    def test_concurrent_publication_change_is_not_overwritten_by_the_repair(self):
        previous = self.publish_original()
        before = snapshot(self.output)
        changed = dict(previous, publication_key="another-render-finished")

        def concurrently_publish(stage, destination, *, expected):
            write_json(self.output / "render.json", changed)
            return publish(stage, destination, expected=expected)

        with patch("epub_tts_narrator.renderer.refresh_metadata.publish", side_effect=concurrently_publish):
            with self.assertRaisesRegex(ValueError, "changed during rendering"):
                refresh_metadata(self.bundle, self.config, notify=lambda message: None)
        self.assertEqual(published_book(self.output, self.book_id), changed)
        before["render.json"] = (self.output / "render.json").read_bytes()
        self.assertEqual(snapshot(self.output), before)
        self.assert_no_staging()

    def test_unexpected_compressed_audio_change_blocks_publication(self):
        self.publish_original()
        before = snapshot(self.output)
        with patch("epub_tts_narrator.renderer.refresh_metadata._audio_payload_hash",
                   side_effect=["SHA256=original", "SHA256=changed"]):
            with self.assertRaisesRegex(ValueError, "changed the encoded audio"):
                refresh_metadata(self.bundle, self.config, notify=lambda message: None)
        self.assertEqual(snapshot(self.output), before)
        self.assertIsNotNone(published_book(self.output, self.book_id))
        self.assert_no_staging()

    def test_missing_and_wrong_epub_stop_before_creating_output_or_render_locks(self):
        for path, error in ((None, "original EPUB"),
                            (str(self.root / "missing.epub"), "no longer available"),
                            (str(self.epub), "does not match")):
            with self.subTest(epub=path):
                config = deepcopy(self.config)
                config["render"]["epub"] = path
                bundle = deepcopy(self.bundle)
                if path == str(self.epub):
                    bundle["source"]["book"]["id"] = "0" * 64
                before = snapshot(self.directory)
                with self.assertRaisesRegex(ValueError, error):
                    refresh_metadata(bundle, config, notify=lambda message: None)
                self.assertEqual(snapshot(self.directory), before)
                self.assertFalse(self.output.exists())
                self.assert_no_staging()

    def test_missing_published_book_does_not_create_an_audiobook(self):
        with self.assertRaisesRegex(ValueError, "No completed audiobook"):
            refresh_metadata(self.bundle, self.config, notify=lambda message: None)
        self.assertFalse(self.output.exists())
        self.assert_no_staging()

    def test_cli_repairs_existing_book_with_no_models_or_voice_store(self):
        self.publish_original()
        config_file = self.root / "global.toml"
        config_file.write_text("", encoding="utf-8")
        output, error = StringIO(), StringIO()
        with (patch("epub_tts_narrator.renderer.cli.load_bundle", return_value=self.bundle),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure") as setup,
              patch("epub_tts_narrator.renderer.engines.create_engine") as engine,
              patch("epub_tts_narrator.renderer.verification.create_verifier") as verifier,
              patch("epub_tts_narrator.renderer.cli.voice_store") as store,
              redirect_stdout(output), redirect_stderr(error)):
            code = main(["refresh-metadata", str(self.directory), "--config", str(config_file),
                         "--epub", str(self.epub)])
        self.assertEqual(code, 0, error.getvalue())
        self.assertIn("Updated audiobook", output.getvalue())
        for untouched in (setup, engine, verifier, store):
            untouched.assert_not_called()
        self.assertTrue(published_book(self.output, self.book_id)["audio"]["audiobook"]["cover_embedded"])


if __name__ == "__main__":
    unittest.main()
