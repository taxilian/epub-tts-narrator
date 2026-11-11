"""Gemini casting concurrency is bounded file configuration."""

from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.config import load_config


class CastingConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "renderer.toml"
        self.base = '[engine]\ncheckpoint="./tts-model"\n[verification]\nmodel="./asr-model"\n'

    def load(self, text=""):
        self.path.write_text(self.base + text)
        return load_config(self.path)

    def test_gemini_concurrency_defaults_to_four(self):
        self.assertEqual(self.load()["casting"]["gemini_concurrency"], 4)

    def test_gemini_concurrency_accepts_integer_bounds(self):
        for value in (1, 2, 8):
            with self.subTest(value=value):
                config = self.load(f"[casting]\ngemini_concurrency={value}\n")
                self.assertEqual(config["casting"]["gemini_concurrency"], value)

    def test_gemini_concurrency_rejects_nonintegers_and_out_of_range_values(self):
        for value in (0, -1, 9, "true", "1.0", '"4"', "[]"):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "casting.gemini_concurrency"):
                self.load(f"[casting]\ngemini_concurrency={value}\n")

    def test_reference_clips_default_to_three_and_accept_one_through_three(self):
        self.assertEqual(self.load()["casting"]["reference_clips"], 3)
        for value in (1, 2, 3):
            with self.subTest(value=value):
                self.assertEqual(self.load(f"[casting]\nreference_clips={value}\n")["casting"]["reference_clips"], value)

    def test_reference_clips_reject_nonintegers_and_out_of_range_values(self):
        for value in (0, 4, "true", "2.0", '"3"'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "casting.reference_clips"):
                self.load(f"[casting]\nreference_clips={value}\n")


if __name__ == "__main__":
    unittest.main()
