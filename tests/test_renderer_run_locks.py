"""Kernel leases recover killed renders without guessing at legacy locks."""

import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer.run_locks import acquire_directory_lock, directory_lock_status
from epub_tts_narrator.renderer.voices import VoiceStore


class DirectoryLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / ".render.lock"

    def child(self, code, *arguments):
        return subprocess.run([sys.executable, "-c", code, *map(str, arguments)],
                              capture_output=True, text=True, check=False, timeout=15)

    def abandoned(self):
        result = self.child(
            "import os, sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock; "
            "lock = acquire_directory_lock(sys.argv[1]); os._exit(130)", self.path)
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertTrue(self.path.is_dir())

    def test_release_removes_sentinel_but_preserves_private_lease_inode(self):
        lock = acquire_directory_lock(self.path)
        lease = self.path.with_name(self.path.name + ".lease")
        inode = lease.stat().st_ino
        self.assertEqual(stat.S_IMODE(lease.stat().st_mode), 0o600)
        self.assertTrue(self.path.is_dir())
        self.assertEqual(Path(lock), self.path)
        self.assertEqual(lock.parent, self.root)
        lock.rmdir()
        lock.rmdir()  # Harmless when cleanup is repeated.
        self.assertFalse(self.path.exists())
        following = acquire_directory_lock(self.path)
        following.rmdir()
        self.assertEqual(lease.stat().st_ino, inode)
        self.assertEqual(lease.read_bytes(), b"")

    def test_live_owner_refuses_another_handle_and_another_process(self):
        lock = acquire_directory_lock(self.path)
        self.addCleanup(lock.rmdir)
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        result = self.child(
            "import sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock\n"
            "try: acquire_directory_lock(sys.argv[1])\n"
            "except FileExistsError: sys.exit(17)\n"
            "sys.exit(99)", self.path)
        self.assertEqual(result.returncode, 17, result.stderr)

    def test_hard_exit_can_be_reacquired_without_manual_lock_removal(self):
        self.abandoned()
        old = json.loads((self.path / "owner.json").read_text())
        lock = acquire_directory_lock(self.path)
        new = json.loads((self.path / "owner.json").read_text())
        self.assertNotEqual(old["token"], new["token"])
        lock.rmdir()
        self.assertFalse(self.path.exists())

    def test_second_interrupt_during_marker_creation_finishes_ownership_record(self):
        result = self.child(
            "import os, signal, sys\n"
            "from unittest.mock import patch\n"
            "from epub_tts_narrator.renderer import run_locks\n"
            "from epub_tts_narrator.renderer.interrupts import RenderInterrupts\n"
            "original = run_locks.write_json\n"
            "def interrupted_write(path, value):\n"
            "    os.kill(os.getpid(), signal.SIGINT)\n"
            "    os.kill(os.getpid(), signal.SIGINT)\n"
            "    original(path, value)\n"
            "with RenderInterrupts(), patch.object(run_locks, 'write_json', interrupted_write):\n"
            "    run_locks.acquire_directory_lock(sys.argv[1])\n", self.path)
        self.assertEqual(result.returncode, 130, result.stderr)
        lock = acquire_directory_lock(self.path)
        lock.rmdir()

    def test_second_interrupt_during_release_finishes_removing_owned_sentinel(self):
        result = self.child(
            "import os, signal, sys\n"
            "from pathlib import Path\n"
            "from unittest.mock import patch\n"
            "from epub_tts_narrator.renderer.run_locks import acquire_directory_lock\n"
            "from epub_tts_narrator.renderer.interrupts import RenderInterrupts\n"
            "original = Path.unlink\n"
            "def interrupted_unlink(path, *args, **kwargs):\n"
            "    if path.name == 'owner.json':\n"
            "        os.kill(os.getpid(), signal.SIGINT)\n"
            "        os.kill(os.getpid(), signal.SIGINT)\n"
            "    return original(path, *args, **kwargs)\n"
            "lock = acquire_directory_lock(sys.argv[1])\n"
            "with RenderInterrupts(), patch.object(Path, 'unlink', interrupted_unlink):\n"
            "    lock.rmdir()\n", self.path)
        self.assertEqual(result.returncode, 130, result.stderr)
        self.assertFalse(self.path.exists())
        lock = acquire_directory_lock(self.path)
        lock.rmdir()

    def test_legacy_empty_directory_is_never_reclaimed(self):
        self.path.mkdir()
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        self.assertEqual(list(self.path.iterdir()), [])

    def test_abandoned_directory_with_extra_content_is_not_reclaimed(self):
        self.abandoned()
        extra = self.path / "unrelated.txt"
        extra.write_text("keep")
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        self.assertEqual(extra.read_text(), "keep")
        self.assertTrue((self.path / "owner.json").exists())

    def test_status_probe_keeps_an_abandoned_marker_and_its_ownership_record(self):
        self.abandoned()
        lease = self.path.with_name(self.path.name + ".lease")
        inode = lease.stat().st_ino
        owner = (self.path / "owner.json").read_bytes()
        self.assertIsNone(directory_lock_status(self.path))
        self.assertTrue(self.path.is_dir())
        self.assertEqual((self.path / "owner.json").read_bytes(), owner)
        self.assertEqual(lease.stat().st_ino, inode)
        self.assertEqual(lease.read_bytes(), b"")

    def test_status_probe_reports_a_live_lease_until_the_owner_releases_it(self):
        held, release = self.root / "held", self.root / "release"
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import sys, time\n"
             "from pathlib import Path\n"
             "from epub_tts_narrator.renderer.run_locks import acquire_directory_lock\n"
             "lock = acquire_directory_lock(sys.argv[1])\n"
             "Path(sys.argv[2]).write_text('held')\n"
             "while not Path(sys.argv[3]).exists():\n"
             "    time.sleep(0.01)\n",
             str(self.path), str(held), str(release)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        try:
            while not held.exists():
                self.assertIsNone(child.poll())
                self.assertGreater(deadline, time.monotonic(), "lock holder never signaled")
            reason = directory_lock_status(self.path)
            self.assertIsNotNone(reason)
            self.assertIn("live process", reason)
            self.assertTrue(self.path.is_dir())
            self.assertTrue((self.path / "owner.json").is_file())
        finally:
            release.write_text("go")
            self.assertEqual(child.wait(timeout=15), 0)
        self.assertIsNone(directory_lock_status(self.path))
        self.assertTrue(self.path.is_dir())
        self.assertTrue((self.path / "owner.json").is_file())

    def test_status_probe_reports_a_lease_held_before_the_directory_exists(self):
        lease = self.path.with_name(self.path.name + ".lease")
        held, release = self.root / "held", self.root / "release"
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import fcntl, os, sys, time\n"
             "from pathlib import Path\n"
             "descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
             "fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
             "Path(sys.argv[2]).write_text('held')\n"
             "while not Path(sys.argv[3]).exists():\n"
             "    time.sleep(0.01)\n",
             str(lease), str(held), str(release)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        try:
            while not held.exists():
                self.assertIsNone(child.poll())
                self.assertGreater(deadline, time.monotonic(), "lease holder never signaled")
            reason = directory_lock_status(self.path)
            self.assertIsNotNone(reason)
            self.assertIn("live process", reason)
            self.assertFalse(self.path.exists())
        finally:
            release.write_text("go")
            self.assertEqual(child.wait(timeout=15), 0)
        self.assertIsNone(directory_lock_status(self.path))

    def test_status_probe_leaves_legacy_and_unverifiable_locks_untouched(self):
        self.path.mkdir()
        self.assertIn("manually", directory_lock_status(self.path))
        self.assertEqual(list(self.path.iterdir()), [])
        self.path.rmdir()
        self.abandoned()
        (self.path / "unrelated.txt").write_text("keep")
        self.assertIn("manually", directory_lock_status(self.path))
        self.assertTrue((self.path / "owner.json").is_file())
        (self.path / "unrelated.txt").unlink()
        (self.path / "owner.json").write_text("{not json")
        self.assertIn("manually", directory_lock_status(self.path))
        lease = self.path.with_name(self.path.name + ".lease")
        lease.rename(self.root / "old-lease")
        self.assertIn("manually", directory_lock_status(self.path))
        self.assertTrue(self.path.is_dir())
        target = self.root / "target"
        target.mkdir()
        (self.path / "owner.json").rename(target / "owner.json")
        (self.path / "owner.json").symlink_to(target / "owner.json")
        self.assertIn("manually", directory_lock_status(self.path))
        self.assertTrue((self.path / "owner.json").is_symlink())

    def test_status_probe_without_any_lock_reports_none(self):
        self.assertIsNone(directory_lock_status(self.path))

    def test_status_probe_never_reports_idle_for_an_existing_unverifiable_lease(self):
        lease = self.path.with_name(self.path.name + ".lease")
        target = self.root / "keep"
        target.write_text("keep")
        with self.subTest(lease="symlink"):
            lease.symlink_to(target)
            self.assertIn("manually", directory_lock_status(self.path))
            lease.unlink()
        with self.subTest(lease="directory"):
            lease.mkdir()
            self.assertIn("manually", directory_lock_status(self.path))
            lease.rmdir()
        with self.subTest(lease="unreadable"):
            lease.write_bytes(b"")
            lease.chmod(0o000)
            self.assertIn("manually", directory_lock_status(self.path))
            lease.chmod(0o600)
            lease.unlink()
        with self.subTest(lease="nonempty"):
            lease.write_bytes(b"junk")
            self.assertIn("manually", directory_lock_status(self.path))
            lease.unlink()
        with self.subTest(lease="group-readable"):
            lease.write_bytes(b"")
            lease.chmod(0o640)
            self.assertIn("manually", directory_lock_status(self.path))
            lease.unlink()
        self.assertEqual(target.read_text(), "keep")
        self.assertIsNone(directory_lock_status(self.path))

    def test_replaced_directory_cannot_inherit_copied_ownership(self):
        self.abandoned()
        self.path.rename(self.root / "original")
        self.path.mkdir()
        (self.path / "owner.json").write_bytes((self.root / "original/owner.json").read_bytes())
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)

    def test_replaced_lease_does_not_prove_abandoned_ownership(self):
        self.abandoned()
        lease = self.path.with_name(self.path.name + ".lease")
        lease.rename(self.root / "old-lease")
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        self.assertTrue(self.path.is_dir())

    def test_symlinked_sentinel_marker_or_lease_is_refused(self):
        target = self.root / "target"
        target.mkdir()
        self.path.symlink_to(target, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        self.path.unlink()
        self.abandoned()
        owner = self.path / "owner.json"
        owner.rename(target / "owner.json")
        owner.symlink_to(target / "owner.json")
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        lease = self.path.with_name(self.path.name + ".lease")
        lease.unlink()
        data = self.root / "keep"
        data.write_text("keep")
        lease.symlink_to(data)
        with self.assertRaises(FileExistsError):
            acquire_directory_lock(self.path)
        self.assertEqual(data.read_text(), "keep")

    def test_release_does_not_remove_a_replacement_directory(self):
        lock = acquire_directory_lock(self.path)
        self.path.rename(self.root / "original")
        self.path.mkdir()
        with self.assertRaises(FileExistsError):
            lock.rmdir()
        self.assertTrue(self.path.is_dir())
        self.assertTrue((self.root / "original/owner.json").exists())

    def test_marker_write_failure_releases_lease_and_empty_sentinel(self):
        with patch("epub_tts_narrator.renderer.run_locks.write_json", side_effect=OSError("full")):
            with self.assertRaisesRegex(OSError, "full"):
                acquire_directory_lock(self.path)
        self.assertFalse(self.path.exists())
        lock = acquire_directory_lock(self.path)
        lock.rmdir()

    def test_voice_store_retains_path_contract_and_recovers_after_exit(self):
        directory = self.root / "voices"
        store = VoiceStore(directory, "test-series")
        result = self.child(
            "import os, sys; from epub_tts_narrator.renderer.voices import VoiceStore; "
            "store = VoiceStore(sys.argv[1], 'test-series'); lock = store.lock(); os._exit(130)", directory)
        self.assertEqual(result.returncode, 130, result.stderr)
        lock = store.lock()
        self.assertEqual(lock.parent, store.directory)
        self.assertTrue((store.directory / ".lock").exists())
        lock.rmdir()
        self.assertFalse((store.directory / ".lock").exists())


if __name__ == "__main__":
    unittest.main()
