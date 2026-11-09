"""Smoke-test the real review-ui command through a POSIX pseudo-terminal."""

import fcntl
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time
from typing import Any
import unittest

try:
    import pty
except ImportError:
    pty = None

from tests.test_review_tui import publish_book


@unittest.skipUnless(os.name == "posix" and pty is not None, "POSIX required")
class ReviewTerminalTests(unittest.TestCase):
    root: Path = Path()
    directory: Path = Path()
    master: int = -1
    process: Any = None
    pending: bytes = b""
    transcript: bytes = b""

    def setUp(self):
        if pty is None:
            self.fail("POSIX PTYs required")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory, _ = publish_book(self.root, two_chapters=True)
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 110, 0, 0))
        environment = dict(os.environ, TERM="xterm-256color", LINES="24", COLUMNS="110")
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "epub_tts_narrator", "review-ui", str(self.directory)],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment, stdin=slave, stdout=slave, stderr=slave,
            start_new_session=True,
        )
        os.close(slave)
        self.pending = b""
        self.transcript = b""
        self.addCleanup(self.close_child)

    def close_child(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        os.close(self.master)

    def read_more(self, timeout):
        if not select.select([self.master], [], [], timeout)[0]:
            return False
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            chunk = b""
        if not chunk:
            return False
        self.pending += chunk
        self.transcript += chunk
        return True

    def wait_for(self, marker):
        deadline = time.monotonic() + 5
        while marker not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.read_more(remaining):
                self.fail(f"Timed out waiting for {marker!r}: {self.transcript!r}")
        self.pending = self.pending.split(marker, 1)[1]

    def finish(self):
        deadline = time.monotonic() + 5
        while self.process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fail(f"review-ui did not exit: {self.transcript!r}")
            self.read_more(min(0.1, remaining))
        self.process.wait(timeout=1)
        self.assertEqual(self.process.returncode, 0, self.transcript.decode(errors="replace"))

    def test_real_terminal_opens_toc_navigates_to_passages_and_quits(self):
        self.wait_for(b"Review chapters 1/2")
        self.wait_for(b"Chapter Two")

        os.write(self.master, b"\x1bOB")
        self.wait_for(b"> Chapter Two")

        os.write(self.master, b"\r")
        self.wait_for(b"Passage 1/1")
        self.wait_for(b"Later, Lena waited.")

        os.write(self.master, b"q")
        self.finish()


if __name__ == "__main__":
    unittest.main()
