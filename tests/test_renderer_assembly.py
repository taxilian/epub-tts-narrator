"""Exact PCM timelines and real FFmpeg audiobook/container integration."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import wave
import zipfile
import zlib

from epub_tts_narrator.renderer.assembly import (
    compose_chapter, extract_cover, normalize_clip, package_audio, probe_audio,
)


def make_wav(path, *, frames=4410, rate=44100, channels=1):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        data = b"".join(struct.pack("<h", round(12000 * math.sin(2 * math.pi * 440 * i / rate)))
                        * channels for i in range(frames))
        audio.writeframes(data)
    return path


def tiny_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" + b"\xff\0\0" * 2 + b"\0" + b"\0\xff\0" * 2))
            + chunk(b"IEND", b""))


def clip(path, *, text="Hello.", start=0, end=None, identifier="b1:0:6", **changes):
    end = len(text) if end is None else end
    row = {"id": identifier, "block_id": "b1", "start": 0, "end": len(text),
           "text": text, "voice_id": "voice_1", "role_id": "narrator",
           "break_after": "paragraph"}
    row.update(changes)
    spoken = row.get("spoken_text", text)
    return {"wav_path": path, "chunk_id": f"{identifier}:{start}:{end}", "segment": row,
            "spoken_text": spoken[start:end], "spoken_start": start, "spoken_end": end,
            "source_ranges": [{"start": start, "end": min(end, len(text))}]}


class ChapterCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.wav = make_wav(self.root / "clip.wav")

    def test_split_segment_gets_one_paragraph_pause_and_exact_ranges(self):
        result = compose_chapter([
            clip(self.wav, text="Hello world.", start=0, end=6),
            clip(self.wav, text="Hello world.", start=6, end=12),
        ], self.root / "chapter.wav")
        self.assertEqual(result["frames"], 4410 * 2 + 11025)
        self.assertEqual([(row["chapter_start_frame"], row["chapter_end_frame"])
                          for row in result["segments"]], [(0, 4410), (4410, 8820)])
        self.assertEqual(result["segments"][1]["spoken_text"], "world.")
        self.assertEqual(result["segments"][1]["source_ranges"], [{"start": 6, "end": 12}])

    def test_blank_pause_does_not_add_to_preceding_paragraph_pause(self):
        result = compose_chapter([
            clip(self.wav),
            clip(self.wav, identifier="b2:0:6", break_before="blank_paragraph"),
        ], self.root / "chapter.wav")
        self.assertEqual(result["segments"][1]["chapter_start_frame"], 4410 + 22050)
        self.assertEqual(result["frames"], 4410 * 2 + 22050 + 11025)

    def test_empty_and_no_speech_controls_do_not_manufacture_audio(self):
        empty = compose_chapter([], self.root / "empty.wav")
        control = compose_chapter([clip(None, text="[]", break_before="blank_paragraph")],
                                  self.root / "control.wav")
        self.assertEqual(empty["frames"], 0)
        self.assertEqual(control["frames"], 0)
        self.assertEqual(control["segments"][0]["chapter_end_frame"], 0)
        self.assertTrue(control["segments"][0]["no_speech"])

    def test_spoken_text_offsets_are_separate_from_bracketed_source(self):
        item = clip(self.wav, text="[Fireball]", end=8, spoken_text="Fireball")
        item["source_ranges"] = [{"start": 1, "end": 9}]
        result = compose_chapter([item], self.root / "chapter.wav")
        row = result["segments"][0]
        self.assertEqual(row["source_text"], "[Fireball]")
        self.assertEqual(row["spoken_text"], "Fireball")
        self.assertEqual(row["source_ranges"], [{"start": 1, "end": 9}])
        self.assertEqual(row["source_end"], 10)

    def test_bracket_group_pauses_only_at_outer_edges_across_rows_and_chunks(self):
        opening = {"kind": "bracketed", "group_id": "g1", "enter": True, "exit": False}
        closing = {**opening, "enter": False, "exit": True}
        result = compose_chapter([
            clip(self.wav, text="First words", start=0, end=6,
                 performance=opening, break_after="none"),
            clip(self.wav, text="First words", start=6, end=11,
                 performance=opening, break_after="none"),
            clip(self.wav, text="last words", identifier="b2", performance=closing, break_after="none"),
            clip(self.wav, text="After.", identifier="b3", break_after="none"),
        ], self.root / "bracket.wav")
        pause = 5292  # 120 ms at 44100 Hz.
        self.assertEqual([(item["chapter_start_frame"], item["chapter_end_frame"])
                          for item in result["segments"]],
                         [(pause, pause + 4410),
                          (pause + 4410, pause + 8820),
                          (pause + 8820, pause + 13230),
                          (pause * 2 + 13230, pause * 2 + 17640)])
        self.assertEqual(result["frames"], pause * 2 + 17640)
        self.assertEqual(result["segments"][0]["performance"], opening)
        self.assertEqual({item["voice_id"] for item in result["segments"]}, {"voice_1"})

    def test_bracket_pause_does_not_stack_with_blank_or_paragraph(self):
        performance = {"kind": "bracketed", "group_id": "g1", "enter": True, "exit": True}
        result = compose_chapter([
            clip(self.wav, performance=performance, break_before="blank_paragraph"),
            clip(self.wav, identifier="b2", break_after="none"),
        ], self.root / "bracket.wav")
        self.assertEqual(result["segments"][0]["chapter_start_frame"], 22050)
        self.assertEqual(result["segments"][1]["chapter_start_frame"], 22050 + 4410 + 11025)

    def test_invalid_offsets_rates_and_overwrites_fail(self):
        item = clip(self.wav)
        item["spoken_text"] = "Unrelated."
        with self.assertRaisesRegex(ValueError, "spoken-text"):
            compose_chapter([item], self.root / "wrong.wav")
        other = make_wav(self.root / "other.wav", rate=24000)
        with self.assertRaisesRegex(ValueError, "sample rate"):
            compose_chapter([clip(other)], self.root / "wrong.wav")
        target = self.root / "existing.wav"
        target.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            compose_chapter([clip(self.wav)], target)
        self.assertEqual(target.read_bytes(), b"keep")
        with self.assertRaisesRegex(ValueError, "Pauses"):
            compose_chapter([], self.root / "wrong.wav", paragraph_pause_ms=float("nan"))

    def test_truncated_input_is_rejected_without_publishing(self):
        self.wav.write_bytes(self.wav.read_bytes()[:-10])
        target = self.root / "chapter.wav"
        with self.assertRaisesRegex(ValueError, "Truncated"):
            compose_chapter([clip(self.wav)], target)
        self.assertFalse(target.exists())


class CoverExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def epub(self, *, modern=True, href="art/cover.png", image=None, cover=True):
        path = self.root / "book.epub"
        data = image if image is not None else tiny_png()
        metadata = '<meta name="cover" content="art"/>' if cover and not modern else ""
        properties = ' properties="cover-image"' if cover and modern else ""
        package = (f'<package xmlns="http://www.idpf.org/2007/opf"><metadata>{metadata}</metadata>'
                   f'<manifest><item id="art" href="{href}" media-type="image/png"{properties}/>'
                   '</manifest></package>')
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile '
                             'full-path="OPS/book.opf" media-type="application/oebps-package+xml"/>'
                             '</rootfiles></container>')
            archive.writestr("OPS/book.opf", package)
            archive.writestr("OPS/art/cover.png", data)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_epub3_and_epub2_declared_covers_are_copied_byte_exact(self):
        for modern in (True, False):
            with self.subTest(modern=modern):
                path, digest = self.epub(modern=modern)
                target = extract_cover(path, self.root / str(modern), digest)
                self.assertEqual(target.read_bytes(), tiny_png())

    def test_missing_cover_is_explicit_and_wrong_epub_is_rejected(self):
        path, digest = self.epub(cover=False)
        self.assertIsNone(extract_cover(path, self.root / "out", digest))
        with self.assertRaisesRegex(ValueError, "does not match"):
            extract_cover(path, self.root / "out", "0" * 64)

    def test_cover_path_traversal_and_invalid_signature_are_rejected(self):
        path, digest = self.epub(href="../../outside.png")
        with self.assertRaisesRegex(ValueError, "escapes"):
            extract_cover(path, self.root / "out", digest)
        path, digest = self.epub(image=b"not a png")
        with self.assertRaisesRegex(ValueError, "signature"):
            extract_cover(path, self.root / "out", digest)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg integration")
class EncodingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="audio ' UTF8 é ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.wav = make_wav(self.root / "clip.wav", frames=11025)

    def chapters(self):
        chapters = []
        for index, title in enumerate(("第一章 = #;\\ title", "Second chapter"), 1):
            chapter = compose_chapter([clip(self.wav, identifier=f"s{index}")],
                                      self.root / f"chapter-{index}.wav")
            chapter.update(id=f"ch{index}", title=title, script_path=f"scripts/chapter-{index:04d}.jsonl")
            chapters.append(chapter)
        empty = compose_chapter([], self.root / "empty.wav")
        empty.update(id="empty", title="Empty", script_path="scripts/empty.jsonl")
        return [chapters[0], empty, chapters[1]]

    def test_m4a_chapters_and_m4b_cover_metadata_and_pcm_sync(self):
        cover = self.root / "cover.png"
        cover.write_bytes(tiny_png())
        chapters = self.chapters()
        output = self.root / "output"
        manifest = package_audio(chapters, output, book={
            "id": "a" * 64, "title": "Book = #;\\ é", "authors": ["Author 🚀"]},
            cover_path=cover, source_fingerprint="b" * 64)
        sync = json.loads((output / "sync.json").read_text())
        self.assertEqual(manifest["frames"], 44100)
        self.assertEqual(sync["sample_rate"], 44100)
        self.assertEqual(sync["segments"][1]["book_start_frame"], 22050)
        self.assertEqual(sync["segments"][1]["book_end_frame"], 33075)
        self.assertTrue(manifest["chapters"][1]["skipped"])
        self.assertIsNone(manifest["chapters"][1]["path"])
        self.assertFalse((output / "chapters/chapter-0002.m4a").exists())
        inspection = probe_audio(output / "book.m4b")
        self.assertEqual(len(inspection["chapters"]), 2)
        self.assertEqual(inspection["chapters"][0]["tags"]["title"], chapters[0]["title"])
        self.assertEqual(inspection["format"]["tags"]["title"], "Book = #;\\ é")
        self.assertTrue(any(stream.get("disposition", {}).get("attached_pic")
                            for stream in inspection["streams"]))
        self.assertEqual(sync["audio_sha256"], hashlib.sha256((output / "book.m4b").read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, "already exists"):
            package_audio(chapters, output, book={})

    def test_mp3_chapters_still_produce_aac_book_from_pcm(self):
        output = self.root / "output"
        manifest = package_audio(self.chapters(), output, book={"title": "Book"}, chapter_format="mp3")
        chapter = probe_audio(output / "chapters/chapter-0001.mp3")
        book = probe_audio(output / "book.m4b")
        self.assertEqual(chapter["streams"][0]["codec_name"], "mp3")
        self.assertEqual(book["streams"][0]["codec_name"], "aac")
        self.assertEqual(manifest["frames"], 44100)

    def test_normalization_resamples_stereo_and_rejects_overwrite(self):
        stereo = make_wav(self.root / "stereo.wav", frames=2400, rate=24000, channels=2)
        target = self.root / "normalized.wav"
        result = normalize_clip(stereo, target)
        self.assertEqual(result["sample_rate"], 44100)
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["frames"], 4410)
        with self.assertRaisesRegex(ValueError, "already exists"):
            normalize_clip(stereo, target)

    def test_all_empty_book_fails(self):
        empty = compose_chapter([], self.root / "empty.wav")
        empty.update(id="empty", title="Empty")
        with self.assertRaisesRegex(ValueError, "no speech"):
            package_audio([empty], self.root / "output", book={})


if __name__ == "__main__":
    unittest.main()
