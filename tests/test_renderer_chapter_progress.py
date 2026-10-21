"""Chapter progress retains book positions across partial rendering and reuse."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import assembly, chapter_cache
from epub_tts_narrator.renderer.common import file_hash
from test_renderer_assembly import clip, make_wav


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
class ChapterProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clips = [clip(make_wav(self.root / "speech.wav"))]
        self.chapter = {
            "id": "chapter-0006", "number": 6, "title": "A Later Chapter",
            "script_path": "chapters/0006.jsonl", "_ordinal": 6, "_total_chapters": 35,
        }
        self.book = {"id": "book", "title": "A Book", "authors": ["An Author"]}
        self.messages = []

    def prepare(self, **kwargs):
        return chapter_cache.prepare_chapter(
            self.chapter, self.clips, self.root / "work" / "chapters",
            book=self.book, render_settings={"normalize_loudness": False},
            context=kwargs.get("context", {}), notify=self.messages.append,
        )

    def test_partial_chapter_build_and_cached_reuse_show_full_book_count(self):
        original = deepcopy(self.chapter)
        first = self.prepare()
        self.assertIn("Assembling chapter 6 / 35: A Later Chapter", self.messages)
        self.assertIn("Encoding chapter 6 / 35: A Later Chapter", self.messages)
        self.assertIn(f"Chapter 6 / 35 ready: {first['encoded_path']}", self.messages)
        checksum = file_hash(first["encoded_path"])
        self.messages.clear()
        with patch.object(chapter_cache, "compose_chapter", side_effect=AssertionError("recomposed")), \
             patch.object(chapter_cache, "encode_chapter", side_effect=AssertionError("reencoded")):
            second = self.prepare()
        self.assertEqual(first, second)
        self.assertEqual(file_hash(second["encoded_path"]), checksum)
        self.assertEqual(self.chapter, original)
        self.assertEqual(self.messages, ["Reusing completed chapter 6 / 35: A Later Chapter"])

    def test_empty_chapter_uses_context_total_without_changing_metadata(self):
        del self.chapter["_total_chapters"]
        self.clips = []
        result = self.prepare(context={"total_chapters": 35})
        self.assertIsNone(result["encoded_path"])
        self.assertIn("Assembling chapter 6 / 35: A Later Chapter", self.messages)
        self.assertIn("Chapter 6 / 35 is empty; recorded as skipped", self.messages)
        self.assertFalse(any(message.startswith("Encoding") for message in self.messages))

    def test_packaging_cached_partial_chapter_keeps_full_book_count(self):
        chapter = self.prepare()
        self.messages.clear()
        with patch.object(assembly, "encode_chapter", side_effect=AssertionError("reencoded")):
            packaged = assembly.package_audio(
                [chapter], self.root / "output", book=self.book, notify=self.messages.append)
        self.assertIn("Reusing chapter audio 6 / 35: A Later Chapter", self.messages)
        self.assertEqual(packaged["chapters"][0]["number"], 6)
        self.assertEqual(packaged["chapters"][0]["ordinal"], 1)
        self.assertEqual(file_hash(self.root / "output" / "chapters" / "chapter-0006.m4a"),
                         file_hash(chapter["encoded_path"]))

    def test_packaging_uncached_chapters_uses_available_book_total_or_record_count(self):
        for declared_total in (35, None):
            with self.subTest(declared_total=declared_total):
                chapter = assembly.compose_chapter(self.clips, self.root / f"chapter-{declared_total}.wav")
                number = 6 if declared_total else 1
                chapter.update(id=f"chapter-{number}", number=number, title="A Chapter")
                if declared_total:
                    chapter["total_chapters"] = declared_total
                before = deepcopy(chapter)
                self.messages.clear()
                assembly.package_audio(
                    [chapter], self.root / f"output-{declared_total}", book=self.book,
                    notify=self.messages.append)
                self.assertIn(f"Encoding chapter {number} / {declared_total or 1}: A Chapter", self.messages)
                self.assertEqual(chapter, before)
