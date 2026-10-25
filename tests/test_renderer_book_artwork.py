"""Publication metadata fidelity and lossless-source square cover preparation."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from epub_tts_narrator.renderer.book_artwork import _image_info, read_publication_metadata, prepare_book_artwork
import test_renderer_assembly as assembly_fixtures


class _ArtworkFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="book artwork é ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def book(self, metadata="", *, image=None, cover=True, kind="image/png", extras=None):
        fixture = assembly_fixtures.CoverExtractionTests()
        fixture.root = self.root
        path, _ = fixture.epub(image=image, cover=cover)
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        package = files["OPS/book.opf"].decode().replace(
            "<metadata>", '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:opf="http://www.idpf.org/2007/opf">' + metadata).replace("image/png", kind)
        files["OPS/book.opf"] = package.encode()
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in {**files, **(extras or {})}.items():
                archive.writestr(name, data)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()


class BookArtworkTests(_ArtworkFixture, unittest.TestCase):
    def test_metadata_keeps_publication_fields_and_authors_without_illustrators(self):
        path, digest = self.book('''
            <dc:title id="short">A subtitle</dc:title>
            <dc:title id="main">The Book &amp; the World</dc:title>
            <meta refines="#main" property="title-type">main</meta>
            <dc:creator opf:role="aut">Author One</dc:creator>
            <dc:creator id="writer">Author Two</dc:creator>
            <meta refines="#writer" property="role" scheme="marc:relators">aut</meta>
            <dc:creator opf:role="ill">Illustrator</dc:creator>
            <dc:contributor>Editor</dc:contributor>
            <dc:language>en-US</dc:language>
            <dc:publisher>Publisher &amp; Sons</dc:publisher>
            <dc:date opf:event="modification">2025-01-01</dc:date>
            <dc:date opf:event="publication">2020-04-13</dc:date>
            <dc:description>&lt;p&gt;First &lt;b&gt;bold&lt;/b&gt; paragraph.&lt;/p&gt;&lt;p&gt;Second &amp;amp; last.&lt;/p&gt;&lt;script&gt;hidden&lt;/script&gt;</dc:description>
            <dc:rights>Copyright Author One</dc:rights>
            <dc:identifier>urn:isbn:12345</dc:identifier>
            <dc:identifier>urn:uuid:1234</dc:identifier>
            <dc:subject>Fantasy</dc:subject><dc:subject>Adventure</dc:subject>
            <meta property="belongs-to-collection" id="series">Great Series</meta>
            <meta refines="#series" property="collection-type">series</meta>
            <meta refines="#series" property="group-position">2.5</meta>
        ''')
        metadata = read_publication_metadata(path, digest)
        self.assertEqual(metadata, {"id": digest, "title": "The Book & the World",
                                   "authors": ["Author One", "Author Two"], "language": "en-US",
                                   "publisher": "Publisher & Sons", "date": "2020-04-13",
                                   "description": "First bold paragraph. Second & last.",
                                   "rights": "Copyright Author One", "identifiers": ["urn:isbn:12345", "urn:uuid:1234"],
                                   "subjects": ["Fantasy", "Adventure"], "contributors": ["Editor", "Illustrator"],
                                   "series": "Great Series", "series_index": "2.5"})

    def test_calibre_series_and_sparse_metadata_keep_extractor_fallbacks(self):
        path, digest = self.book('<meta name="calibre:series" content="A Series"/>'
                                 '<meta name="calibre:series_index" content="3"/>')
        metadata = read_publication_metadata(path, digest)
        self.assertEqual(metadata["title"], "book")
        self.assertEqual(metadata["authors"], [])
        self.assertEqual(metadata["language"], "und")
        self.assertEqual(metadata["series"], "A Series")
        self.assertEqual(metadata["series_index"], "3")

    def test_wrong_book_rejected_before_any_output_is_created(self):
        path, digest = self.book()
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare_book_artwork(path, self.root / "out", "0" * 64)
        self.assertFalse((self.root / "out").exists())

    def test_missing_and_unsupported_covers_fail_explicitly(self):
        for cover, kind in ((False, "image/png"), (True, "image/svg+xml")):
            with self.subTest(cover=cover, kind=kind):
                path, digest = self.book(cover=cover, kind=kind)
                with self.assertRaisesRegex(ValueError, "JPEG or PNG"):
                    prepare_book_artwork(path, self.root / "out", digest)
                self.assertFalse((self.root / "out" / "cover.jpg").exists())

    def test_unsafe_archive_and_encrypted_assets_are_rejected(self):
        for extras, error in (({"../outside": b"bad"}, "Unsafe"),
                              ({"META-INF/encryption.xml": '<encryption><CipherReference URI="OPS/book.opf"/></encryption>'}, "Encrypted")):
            with self.subTest(error=error):
                path, digest = self.book(extras=extras)
                with self.assertRaisesRegex(ValueError, error):
                    prepare_book_artwork(path, self.root / "out", digest)

    def test_invalid_cover_signature_is_rejected(self):
        path, digest = self.book(image=b"not a PNG")
        with self.assertRaisesRegex(ValueError, "signature"):
            prepare_book_artwork(path, self.root / "out", digest)

    def test_invalid_image_dimensions_fail_before_artwork_conversion(self):
        for width, height in ((0, 1600), (20000, 1000), (10000, 10000)):
            with self.subTest(width=width, height=height):
                result = json.dumps({"streams": [{"codec_name": "png", "width": width, "height": height}]}).encode()
                with patch("epub_tts_narrator.renderer.book_artwork._command", return_value=result):
                    with self.assertRaisesRegex(ValueError, "too large"):
                        _image_info(self.root / "cover.png", "ffprobe")

    def test_symlinked_output_directory_is_rejected(self):
        path, digest = self.book()
        target = self.root / "real"
        target.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            prepare_book_artwork(path, alias, digest)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class BookArtworkIntegrationTests(_ArtworkFixture, unittest.TestCase):
    def test_portrait_landscape_and_square_preserve_complete_foreground_and_original_bytes(self):
        for width, height in ((320, 640), (640, 320), (320, 320)):
            with self.subTest(width=width, height=height):
                source = self.root / "source.png"
                if width < height:
                    bars = "drawbox=x=0:y=0:w=iw:h=40:color=red:t=fill,drawbox=x=0:y=ih-40:w=iw:h=40:color=blue:t=fill"
                    red, blue, side = (800, 30), (800, 1570), (20, 800)
                else:
                    bars = "drawbox=x=0:y=0:w=40:h=ih:color=red:t=fill,drawbox=x=iw-40:y=0:w=40:h=ih:color=blue:t=fill"
                    red, blue, side = (30, 800), (1570, 800), (800, 20)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=green:s={width}x{height}",
                                "-vf", bars, "-frames:v", "1", "-update", "1", str(source)], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                path, digest = self.book(image=source.read_bytes())
                output = self.root / f"out-{width}-{height}"
                prepared = prepare_book_artwork(path, output, digest)
                self.assertEqual(prepared["cover_path"], output / "cover.jpg")
                artwork = prepared["artwork"]
                self.assertEqual((artwork["cover"]["width"], artwork["cover"]["height"]), (1600, 1600))
                self.assertEqual((output / artwork["source_image"]["path"]).read_bytes(), source.read_bytes())
                self.assertEqual(artwork["epub_sha256"], digest)
                raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(prepared["cover_path"]), "-f", "rawvideo", "-pix_fmt", "rgb24", "-frames:v", "1", "pipe:1"],
                                     check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
                def pixel(point):
                    x, y = point
                    offset = (y * 1600 + x) * 3
                    return tuple(raw[offset:offset + 3])
                self.assertGreater(pixel(red)[0], 200)
                self.assertLess(pixel(red)[2], 30)
                self.assertGreater(pixel(blue)[2], 200)
                self.assertLess(pixel(blue)[0], 30)
                self.assertGreater(pixel(side)[1], 25)  # Background is cover-derived green, never plain black.
                # A repeated preparation may reuse identical files, never alter them.
                repeated = prepare_book_artwork(path, output, digest)
                self.assertEqual(prepared, repeated)

    def test_corrupt_image_fails_without_publishing_any_artwork(self):
        path, digest = self.book(image=b"\x89PNG\r\n\x1a\ncorrupt")
        with self.assertRaises(ValueError):
            prepare_book_artwork(path, self.root / "out", digest)
        self.assertEqual(list((self.root / "out").iterdir()), [])

    def test_existing_artwork_symlink_is_never_followed(self):
        path, digest = self.book()
        output = self.root / "out"
        output.mkdir()
        unrelated = self.root / "unrelated.jpg"
        unrelated.write_bytes(b"keep me")
        (output / "cover.jpg").symlink_to(unrelated)
        with self.assertRaisesRegex(ValueError, "regular file"):
            prepare_book_artwork(path, output, digest)
        self.assertEqual(unrelated.read_bytes(), b"keep me")
