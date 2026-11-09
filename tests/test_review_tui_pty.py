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

from tests.test_review_tui import add_render_work, publish_book


_REVIEW_WITH_FAKE_TAKES = r"""
import curses
from pathlib import Path
import sys

from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.analysis_revision import related_directories, revise_directories
from epub_tts_narrator.review_tui import CursesScreen, ReviewApp, _drive
from epub_tts_narrator.review_tui_model import palette


def list_takes(clips_directory, request_hash):
    return [{"attempt": 1, "audio_path": None, "passed": True, "reasons": [],
             "text_confirmation_method": "verbatim", "duration_seconds": 1.0,
             "transcript": "DO NOT DISPLAY", "echo": False, "eligible": True,
             "ineligible_reason": None, "accepted": True}]


directory = Path(sys.argv[1]).resolve()
document = _completed_directory(directory)[0]
app = ReviewApp(directory, document, revise_directories=revise_directories,
                related_directories=related_directories, player=object(),
                play_sample=lambda player, path: False, list_takes=list_takes,
                select_take=lambda *args, **kwargs: {})


def wrapped(stdscr):
    return _drive(app, CursesScreen(curses, stdscr, palette(len(app.roles))))


curses.wrapper(wrapped)
"""


@unittest.skipUnless(os.name == "posix" and pty is not None, "POSIX required")
class ReviewTerminalTests(unittest.TestCase):
    root: Path = Path()
    directory: Path = Path()
    master: int = -1
    slave: int = -1
    process: Any = None
    environment: dict[str, str] = {}
    pending: bytes = b""
    transcript: bytes = b""

    def setUp(self):
        if pty is None:
            self.fail("POSIX PTYs required")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory, _ = publish_book(self.root, two_chapters=True)
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 110, 0, 0))
        self.environment = dict(os.environ, TERM="xterm-256color", LINES="24", COLUMNS="110")
        self.pending = b""
        self.transcript = b""
        self.addCleanup(self.close_child)

    def start(self, command=None):
        command = command or [sys.executable, "-u", "-m", "epub_tts_narrator", "review-ui", str(self.directory)]
        self.process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parent.parent,
            env=self.environment, stdin=self.slave, stdout=self.slave, stderr=self.slave,
            start_new_session=True,
        )
        os.close(self.slave)
        self.slave = -1

    def close_child(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.slave >= 0:
            os.close(self.slave)
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
        self.start()
        self.wait_for(b"Review chapters 1/2")
        self.wait_for(b"Chapter Two")

        os.write(self.master, b"\x1bOB")
        self.wait_for(b"> Chapter Two")

        os.write(self.master, b"\r")
        self.wait_for(b"Passage 1/1")
        self.wait_for(b"Later, Lena waited.")

        os.write(self.master, b"q")
        self.finish()

    def test_real_terminal_marks_failed_chapter_jumps_and_edits_without_audio(self):
        from tests.test_review_tui import add_failure_status
        add_failure_status(self.directory, 2, error="Verification exhausted", segment_ids=["s4"])
        self.start()
        self.wait_for(b"Review chapters 1/2")
        self.wait_for(b"FAILED SpeechVerificationFailed")

        os.write(self.master, b"\x1bOB")
        self.wait_for(b"> FAILED SpeechVerificationFailed")
        os.write(self.master, b"f")
        self.wait_for(b"Verification exhausted")
        self.wait_for(b"[failed]")

        os.write(self.master, b"\r")
        self.wait_for(b"Passage metadata")
        self.wait_for(b"Render: FAILED")
        os.write(self.master, b"\r")
        self.wait_for(b"Choose existing speaker(s)")
        os.write(self.master, b"\x1b")
        self.wait_for(b"Passage metadata")
        os.write(self.master, b"q")
        self.finish()
        self._save_evidence("failed-chapter-flow")

    def test_real_terminal_failure_markers_stay_visible_at_80_and_120_columns(self):
        from tests.test_review_tui import add_failure_status, review_document
        from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
        from epub_tts_narrator.cli import export_bundle
        from epub_tts_narrator.series import link_analysis
        for cols in (80, 120):
            with self.subTest(cols=cols):
                document = review_document(long_passage=True)
                document["source"]["chapters"][0]["title"] = (
                    "Chapter One: The Extremely Long Chapter Title That Would Push "
                    "Any Trailing Failure Marker Right Off The Narrow Terminal")
                directory = self.root / f"wide-book-{cols}"
                registry = fresh_registry(directory)
                linked = link_analysis(document, registry)
                publish_directory(linked, registry, directory, export_bundle)
                add_failure_status(directory, 1, error="Verification exhausted", segment_ids=["s3"])
                self._resize(cols)
                self.start([sys.executable, "-u", "-m", "epub_tts_narrator", "review-ui",
                            str(directory)])
                self.wait_for(b"FAILED SpeechVerificationFailed")
                os.write(self.master, b"\r")
                # The long title legitimately pushes the error text off the
                # narrow header; the marker before the passage text is the
                # guaranteed-visible evidence.
                self.wait_for(b"[failed]")
                os.write(self.master, b"q")
                self.finish()
                self._save_evidence(f"failure-markers-{cols}-cols")

    def _resize(self, cols):
        if self.slave >= 0:
            os.close(self.slave)
        os.close(self.master)
        self.master, self.slave = pty.openpty()
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, cols, 0, 0))
        self.pending = b""
        self.transcript = b""

    def _save_evidence(self, name):
        evidence = (Path(__file__).resolve().parent.parent / ".omo" / "evidence"
                    / "review-ui-failures")
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / f"{name}.txt").write_bytes(self.transcript)

    def test_real_terminal_opens_takes_with_injected_audio_and_take_boundaries(self):
        add_render_work(self.directory)
        self.start([sys.executable, "-u", "-c", _REVIEW_WITH_FAKE_TAKES, str(self.directory)])
        self.wait_for(b"Review chapters 1/2")

        os.write(self.master, b"\r")
        self.wait_for(b"Passage 1/3")
        os.write(self.master, b"\r")
        self.wait_for(b"Takes: view and play rendered attempts")
        os.write(self.master, b"t")

        self.wait_for(b"Rendered takes")
        self.wait_for(b"* Take 1")
        os.write(self.master, b"\x1b")
        self.wait_for(b"Passage metadata")
        os.write(self.master, b"q")
        self.finish()


if __name__ == "__main__":
    unittest.main()
