from pathlib import Path
import unittest

from epub_tts_narrator.renderer.cleanup import cleanup_work
from epub_tts_narrator.renderer.common import file_hash
import test_renderer_continuity as fixtures
from test_renderer_workflow import FFMPEG, FFPROBE


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg required")
class ContinuityCleanupTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ContinuityTests("runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.run_chapter(self.fixture.chunks[:2])
        self.work = self.fixture.root / "work"
        self.codes = list(self.work.rglob("*.codes.json"))
        self.reference = Path(self.fixture.voices["v1"]["reference_audio"])

    def test_explicit_discard_reports_and_removes_continuity_artifacts(self):
        original = {path: file_hash(path) for path in self.work.rglob("*") if path.is_file()}
        reference_hash = file_hash(self.reference)

        preview = cleanup_work(self.work, dry_run=True, discard_render_cache=True)

        self.assertEqual(original, {path: file_hash(path) for path in original})
        listed = {item["path"] for item in preview["files"]}
        self.assertTrue(self.codes)
        self.assertTrue(all(str(path.relative_to(self.work)) in listed for path in self.codes))
        result = cleanup_work(self.work, discard_render_cache=True)
        self.assertEqual(result["files"], preview["files"])
        self.assertFalse(result["preserves_resumability"])
        self.assertFalse((self.work / "clips").exists())
        self.assertFalse((self.work / "speech-cache").exists())
        self.assertEqual(file_hash(self.reference), reference_hash)

    def test_normal_cleanup_preserves_continuity_artifacts_byte_for_byte(self):
        original = {path: file_hash(path) for path in self.codes}

        result = cleanup_work(self.work)

        self.assertTrue(result["preserves_resumability"])
        self.assertEqual(original, {path: file_hash(path) for path in self.codes})

    def test_discard_rejects_unrecognized_codec_names_before_deleting_anything(self):
        directory = self.codes[0].parent
        for name in ("notes.codes.json", "a" * 63 + ".codes.json", "A" * 64 + ".codes.json"):
            with self.subTest(name=name):
                unexpected = directory / name
                unexpected.write_text("unrelated content")
                try:
                    original = {path: file_hash(path) for path in self.work.rglob("*") if path.is_file()}
                    with self.assertRaisesRegex(ValueError, "unrecognized speech cache file"):
                        cleanup_work(self.work, discard_render_cache=True)
                    self.assertEqual(original, {path: file_hash(path) for path in original})
                finally:
                    unexpected.unlink()

    def test_discard_rejects_codec_symlink_without_touching_reference_or_cache(self):
        original = {path: file_hash(path) for path in self.work.rglob("*") if path.is_file()}
        link = self.codes[0].parent / ("0" * 64 + ".codes.json")
        link.symlink_to(self.reference)
        reference_hash = file_hash(self.reference)

        with self.assertRaises(ValueError):
            cleanup_work(self.work, discard_render_cache=True)

        self.assertTrue(link.is_symlink())
        self.assertEqual(file_hash(self.reference), reference_hash)
        self.assertEqual(original, {path: file_hash(path) for path in original})

    def test_discard_rejects_codec_checksum_mismatch_before_deleting_anything(self):
        self.codes[0].write_text("changed codec artifact")
        original = {path: file_hash(path) for path in self.work.rglob("*") if path.is_file()}

        with self.assertRaisesRegex(ValueError, "checksum"):
            cleanup_work(self.work, discard_render_cache=True)

        self.assertEqual(original, {path: file_hash(path) for path in original})
