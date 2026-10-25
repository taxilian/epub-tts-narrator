"""Artwork preflight and embedded book tags need no TTS or external services."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from epub_tts_narrator.renderer.assembly import compose_chapter, package_audio, probe_audio
from epub_tts_narrator.renderer.book_metadata import (
    book_tags, check_book_tags, check_publication_source, metadata_arguments, publication_assets,
)
from epub_tts_narrator.renderer.config import DEFAULTS
import test_renderer_assembly as audio_fixtures


class PublicationMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = deepcopy(DEFAULTS)
        self.bundle = {"source": {"book": {"id": "a" * 64}}}

    def test_missing_original_epub_is_blocked_by_default(self):
        with self.assertRaisesRegex(ValueError, "original EPUB.*cover and metadata"):
            check_publication_source(self.bundle, self.config)
        self.config["render"].pop("require_cover")
        with self.assertRaisesRegex(ValueError, "original EPUB"):
            check_publication_source(self.bundle, self.config)

    def test_missing_cover_requires_explicit_opt_out(self):
        self.config["render"]["require_cover"] = False
        check_publication_source(self.bundle, self.config)
        epub, book_id = audio_fixtures.CoverExtractionTests.epub(self, cover=False)
        self.bundle["source"]["book"]["id"] = book_id
        self.config["render"]["epub"] = str(epub)
        check_publication_source(self.bundle, self.config)
        self.config["render"]["require_cover"] = True
        with self.assertRaisesRegex(ValueError, "no supported.*cover"):
            check_publication_source(self.bundle, self.config)

    def test_opt_out_does_not_allow_wrong_or_missing_epub(self):
        self.config["render"]["require_cover"] = False
        self.config["render"]["epub"] = str(self.root / "missing.epub")
        with self.assertRaisesRegex(ValueError, "no longer available"):
            check_publication_source(self.bundle, self.config)
        epub, _ = audio_fixtures.CoverExtractionTests.epub(self)
        self.config["render"]["epub"] = str(epub)
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_publication_source(self.bundle, self.config)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
    def test_valid_declared_cover_passes_read_only_preflight(self):
        epub, book_id = audio_fixtures.CoverExtractionTests.epub(self)
        self.bundle["source"]["book"]["id"] = book_id
        self.config["render"]["epub"] = str(epub)
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        check_publication_source(self.bundle, self.config)
        self.assertEqual({path.name: path.read_bytes() for path in self.root.iterdir()}, before)

    def test_optional_missing_cover_still_reads_complete_epub_metadata(self):
        epub = self.root / "without-cover.epub"
        with zipfile.ZipFile(epub, "w") as archive:
            archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile '
                             'full-path="book.opf" media-type="application/oebps-package+xml"/>'
                             '</rootfiles></container>')
            archive.writestr("book.opf", '<package xmlns="http://www.idpf.org/2007/opf" '
                             'xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata>'
                             '<dc:title>Updated Book Title</dc:title><dc:creator>Book Author</dc:creator>'
                             '<dc:publisher>Small Press</dc:publisher><dc:language>en-US</dc:language>'
                             '<dc:date>2024-01-09</dc:date><dc:rights>Copyright 2024</dc:rights>'
                             '<dc:description>Full EPUB description.</dc:description>'
                             '</metadata><manifest/></package>')
        self.bundle["source"]["book"] = {"id": hashlib.sha256(epub.read_bytes()).hexdigest(),
                                            "title": "Old title", "authors": [], "language": "und"}
        original = deepcopy(self.bundle)
        self.config["render"].update(epub=str(epub), require_cover=False)
        book, cover, artwork = publication_assets(self.bundle, self.config, self.root / "artwork")
        self.assertIsNone(cover)
        self.assertIsNone(artwork)
        self.assertEqual(book["title"], "Updated Book Title")
        self.assertEqual(book["authors"], ["Book Author"])
        self.assertEqual(book["publisher"], "Small Press")
        self.assertEqual(book["date"], "2024-01-09")
        self.assertEqual(book["rights"], "Copyright 2024")
        self.assertEqual(book["description"], "Full EPUB description.")
        self.assertEqual(self.bundle, original)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
    def test_corrupt_image_is_rejected_before_speech_even_with_valid_signature(self):
        epub, book_id = audio_fixtures.CoverExtractionTests.epub(self, image=b"\x89PNG\r\n\x1a\ncorrupt")
        self.bundle["source"]["book"]["id"] = book_id
        self.config["render"]["epub"] = str(epub)
        with self.assertRaisesRegex(ValueError, "cover"):
            check_publication_source(self.bundle, self.config)

    @staticmethod
    def book():
        return {"id": "b" * 64, "title": "A book = #;\\ é", "authors": ["Author One", "Author Two"],
                "language": "en-US", "date": "2024-01-09", "publisher": "Small Press",
                "rights": "Copyright © 2024 Authors", "description": "First line.\nSecond line = #;\\!",
                "series": "The Fixture Books", "series_position": "2", "subjects": ["Fantasy", "Adventure"],
                "identifiers": ["urn:isbn:9780000000000"]}

    def test_complete_fields_map_to_standard_audio_tags_and_language(self):
        book = self.book()
        tags = book_tags(book)
        self.assertEqual(tags, {"title": book["title"], "album": book["title"],
            "artist": "Author One; Author Two", "album_artist": "Author One; Author Two",
            "genre": "Audiobook", "media_type": "2", "date": book["date"],
            "publisher": book["publisher"], "copyright": book["rights"],
            "description": book["description"], "synopsis": book["description"], "grouping": book["series"]})
        arguments = metadata_arguments(book)
        for name, value in tags.items():
            self.assertIn(f"{name}={value}", arguments)
        self.assertEqual(arguments[-2:], ["-metadata:s:a:0", "language=eng"])
        self.assertEqual(book, self.book())

    def test_metadata_validation_rejects_missing_tags_and_nul(self):
        book = self.book()
        tags = book_tags(book)
        check_book_tags({"format": {"tags": tags}}, book)
        for key in ("title", "album", "artist", "album_artist", "date", "copyright",
                    "description", "synopsis", "grouping", "media_type"):
            with self.subTest(tag=key), self.assertRaisesRegex(ValueError, key):
                check_book_tags({"format": {"tags": {name: value for name, value in tags.items() if name != key}}}, book)
        with self.assertRaisesRegex(ValueError, "NUL"):
            metadata_arguments(dict(book, title="bad\x00title"))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
class PublicationMetadataEncodingTests(unittest.TestCase):
    def test_real_m4b_keeps_cover_chapters_language_and_epub_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cover = root / "cover.png"
            cover.write_bytes(audio_fixtures.tiny_png())
            clip = audio_fixtures.make_wav(root / "clip.wav")
            chapter = compose_chapter([audio_fixtures.clip(clip)], root / "chapter.wav")
            chapter.update(id="ch0001", title="Chapter One", script_path="scripts/chapter-0001.jsonl")
            book = PublicationMetadataTests.book()
            result = package_audio([chapter], root / "out", book=book, cover_path=cover)
            inspection = probe_audio(root / "out" / "book.m4b")
            actual = inspection["format"]["tags"]
            for key, value in book_tags(book).items():
                if key != "publisher":  # FFmpeg's iTunes muxer may omit this nonstandard tag.
                    self.assertEqual(actual.get(key), value, key)
            audio = next(stream for stream in inspection["streams"] if stream["codec_type"] == "audio")
            self.assertEqual(audio["tags"]["language"], "eng")
            self.assertEqual(len(inspection["chapters"]), 1)
            self.assertTrue(any(stream.get("disposition", {}).get("attached_pic") == 1
                                for stream in inspection["streams"]))
            self.assertEqual(result["book"], book)
            manifest = json.loads((root / "out" / "audio-manifest.json").read_text())
            self.assertEqual(manifest["book"]["publisher"], "Small Press")
            self.assertEqual(manifest["book"]["identifiers"], book["identifiers"])
