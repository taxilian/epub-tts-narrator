"""Descriptive fresh audiobook names and safe persisted-name compatibility."""

from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.book_output import (
    default_audiobook_filename,
    resolve_audiobook_filename,
    validate_audiobook_filename,
)


class AudiobookFilenameTests(unittest.TestCase):
    def test_fresh_filename_preserves_unicode_and_removes_path_syntax(self):
        # Given an untrusted title containing separators, controls, and Unicode.
        book = {"title": "  明の本 / \\:\x00  "}

        # When deriving a fresh output name.
        filename = default_audiobook_filename(book)

        # Then it is one descriptive basename, not a path or generic book.m4b.
        self.assertEqual(filename, "明の本.m4b")
        self.assertEqual(Path(filename).name, filename)
        self.assertNotEqual(filename, "book.m4b")

    def test_fresh_filename_bounds_utf8_length_without_losing_unicode(self):
        # Given a title whose UTF-8 representation exceeds filesystem limits.
        book = {"title": "界" * 100}

        # When deriving the name.
        filename = default_audiobook_filename(book)

        # Then the complete suffix-bearing basename remains bounded and valid Unicode.
        self.assertLessEqual(len(filename.encode("utf-8")), 184)
        self.assertTrue(filename.startswith("界"))
        self.assertTrue(filename.endswith(".m4b"))

    def test_fresh_reserved_or_generic_titles_still_get_descriptive_names(self):
        for title in ("CON", "Book"):
            with self.subTest(title=title):
                # Given a title that is unsafe or generic as a bare basename.
                # When deriving the name.
                filename = default_audiobook_filename({"title": title})

                # Then the title remains visible without using the unsafe/generic bare name.
                self.assertEqual(filename, f"{title} audiobook.m4b")
                self.assertEqual(validate_audiobook_filename(filename), filename)

    def test_punctuation_only_title_has_a_stable_descriptive_fallback(self):
        # Given a title with no characters that can remain in a safe basename.
        book = {"title": "///"}

        # When deriving the name twice.
        first = default_audiobook_filename(book)
        second = default_audiobook_filename(book)

        # Then a stable nonempty identifier avoids both generic and empty names.
        self.assertEqual(first, second)
        self.assertRegex(first, r"^audiobook-[0-9a-f]{16}\.m4b$")
        self.assertEqual(validate_audiobook_filename(first), first)

    def test_custom_recorded_name_wins_over_a_changed_book_title(self):
        # Given an existing manifest with a custom name and different current metadata.
        previous_audio = {"audiobook": {"path": "Existing 名.m4b"}}

        # When resolving the name for a subsequent publication.
        filename = resolve_audiobook_filename(
            {"title": "Replacement Title"}, previous_audio, "/unused/output")

        # Then the persisted name is authoritative and needs no legacy file.
        self.assertEqual(filename, "Existing 名.m4b")

    def test_explicit_legacy_recorded_name_remains_supported(self):
        # Given an existing manifest explicitly recording the legacy basename.
        previous_audio = {"audiobook": {"path": "book.m4b"}}

        # When resolving without any filesystem fallback.
        filename = resolve_audiobook_filename(previous_audio=previous_audio)

        # Then the existing book keeps its established name.
        self.assertEqual(filename, "book.m4b")

    def test_old_manifest_without_path_uses_only_an_existing_legacy_book(self):
        with tempfile.TemporaryDirectory() as temporary:
            # Given an old manifest with no path field and an intact legacy output.
            root = Path(temporary)
            (root / "book.m4b").write_bytes(b"legacy fixture")

            # When resolving the old publication.
            filename = resolve_audiobook_filename(previous_audio={}, output_dir=root)

            # Then the legacy file supplies the compatibility fallback.
            self.assertEqual(filename, "book.m4b")

    def test_old_manifest_without_path_or_legacy_file_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            # Given no recorded path and no recognizable legacy output.
            # When resolving, then the missing contract is reported rather than inventing book.m4b.
            with self.assertRaisesRegex(ValueError, "audio.audiobook.path"):
                resolve_audiobook_filename(previous_audio={}, output_dir=Path(temporary))

    def test_unsafe_or_non_m4b_persisted_names_are_refused(self):
        invalid = (
            "../outside.m4b", "/tmp/outside.m4b", "..\\outside.m4b",
            "bad\x00name.m4b", "chapter.wav", ".hidden.m4b", "CON.m4b",
            "trailing .m4b", "é" * 200 + ".m4b", "", None, 42,
        )
        for value in invalid:
            with self.subTest(value=value):
                # Given an unsafe persisted override.
                # When validating it, then no path is returned for later replacement or cleanup.
                with self.assertRaisesRegex(ValueError, "audio.audiobook.path"):
                    validate_audiobook_filename(value)


if __name__ == "__main__":
    unittest.main()
