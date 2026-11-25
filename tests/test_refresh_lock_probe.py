"""A voice-store lock left by a killed casting session never blocks saving a book; a live one does."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.run_locks import acquire_directory_lock
from epub_tts_narrator.series import link_analysis
from epub_tts_narrator.voice_profile_guard import refresh_locks
from test_character_merge import AUDITION, build

ROOT = Path(__file__).resolve().parents[1]


class RefreshLockProbeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"
        document = build("locks", "Locks", [("mara", "Mara", ())], [[("“Hello,” ", "mara"), ("she said.", None)]])
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)
        self.voice_lock = self.path / "voices" / "series" / ".lock"
        self.voice_lock.parent.mkdir(parents=True)

    def test_a_lock_a_killed_casting_session_left_does_not_block(self):
        subprocess.run([sys.executable, "-c",
                        "import os, sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock; "
                        "acquire_directory_lock(sys.argv[1]); os._exit(130)", str(self.voice_lock)],
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
        self.assertTrue(self.voice_lock.exists(), "the killed session left its lock behind")
        with refresh_locks(self.path):
            pass
        self.assertTrue(self.voice_lock.exists(), "the probe never removes a lock")

    def test_a_live_casting_session_still_blocks_and_is_named(self):
        lock = acquire_directory_lock(self.voice_lock)
        self.addCleanup(lock.rmdir)
        with self.assertRaises(ValueError) as caught:
            with refresh_locks(self.path):
                pass
        self.assertIn(str(self.voice_lock), str(caught.exception))
        self.assertIn("Let it finish, then rerun", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
