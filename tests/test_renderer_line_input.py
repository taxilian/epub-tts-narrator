"""Exercise casting input as typed in a real terminal, without audio or models."""

import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.renderer import terminal

try:
    import pty
    import termios
except ImportError:
    pty = termios = None


CHILD = r'''
import fcntl
import json
import sys
import termios
from epub_tts_narrator.renderer.audition import run_casting
from epub_tts_narrator.renderer import terminal

fcntl.ioctl(0, termios.TIOCSCTTY, 0)

# Select a deterministic editor mode in this fixture process without changing
# the real application behavior or replacing the user's HOME environment.
enable_editor = terminal.enable_line_editing
def fixture_editor():
    enabled = enable_editor()
    if enabled:
        editor = sys.modules["readline"]
        libedit = "libedit" in (editor.__doc__ or "")
        editor.parse_and_bind("bind -e" if libedit else "set editing-mode emacs")
    return enabled
terminal.enable_line_editing = fixture_editor

class Session:
    def __init__(self):
        self.edits = []
        self.voice = {"reference_audio": "fixture.wav"}
        self.row = {
            "voice_id": "v1", "characters": ["Mara"], "roles": ["mara"],
            "prompt": "Original prompt.", "reference_text": "Original script.",
            "selected": self.voice, "candidates": [],
        }

    def rows(self):
        return [self.row.copy()]

    def update_prompt(self, voice_id, **values):
        assert voice_id == "v1"
        self.edits.append(values)
        self.row.update(values)

    def regenerate(self, voice_id, *, voice_designer=None):
        assert voice_id == "v1"
        assert voice_designer == "local"
        return self.voice

class Player:
    def __init__(self):
        self.active = False

    def start(self, path):
        self.active = True
        print("PLAYBACK_READY", flush=True)

    def poll(self):
        return self.active

    def stop(self):
        self.active = False

session = Session()
run_casting(session, Player())
print(json.dumps(session.edits), file=sys.stderr, flush=True)
'''


class OptionalLineEditingTests(unittest.TestCase):
    def test_python_without_readline_can_still_use_casting(self):
        with patch.dict(sys.modules, {"readline": None}):
            self.assertFalse(terminal.enable_line_editing())


@unittest.skipUnless(
    pty is not None and termios is not None and importlib.util.find_spec("readline"),
    "Requires POSIX PTYs and Python readline support",
)
class CastingLineInputTests(unittest.TestCase):
    def setUp(self):
        master, slave = pty.openpty()
        self.master = master
        # Isolate readline/libedit from personal vi-mode or custom bindings.
        environment = dict(os.environ, INPUTRC=os.devnull, TERM="xterm")
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            stdin=slave,
            stdout=slave,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        os.close(slave)
        self.pending = b""
        self.transcript = b""
        self.addCleanup(self.close_child)
        self.wait_for(b"> ")

    def close_child(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.process.stderr.close()
        os.close(self.master)

    def wait_for(self, marker):
        deadline = time.monotonic() + 5
        while marker not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.master], [], [], remaining)[0]:
                self.fail(f"Timed out waiting for {marker!r}: {self.transcript!r}")
            try:
                chunk = os.read(self.master, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                self.fail(f"Child closed before {marker!r}: {self.transcript!r}")
            self.pending += chunk
            self.transcript += chunk
        self.pending = self.pending.split(marker, 1)[1]

    def type(self, value):
        os.write(self.master, value)

    def edit(self, prompt, script, *, command=b"g 1\r"):
        self.type(command)
        self.wait_for(b"New prompt (Enter keeps current): ")
        self.type(prompt)
        self.wait_for(b"New script (Enter keeps current): ")
        self.type(script)
        self.wait_for(b"New vocal traits (Enter keeps current; - clears): ")
        self.type(b"\r")
        self.wait_for(b"PLAYBACK_READY")

    def stop_playback(self):
        # An arrow can arrive across multiple reads while playback is raw.
        for part in (b"\x1b", b"[", b"D"):
            self.type(part)
            time.sleep(0.01)
        self.wait_for(b"> ")

    def finish(self):
        self.type(b"q\r")
        # On macOS libedit can wait for terminal output to drain before input()
        # returns. communicate() alone only services stderr, leaving it stuck.
        deadline = time.monotonic() + 5
        while self.process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.fail(f"Casting did not exit: {self.transcript!r}")
            if select.select([self.master], [], [], min(0.1, remaining))[0]:
                try:
                    self.transcript += os.read(self.master, 65536)
                except OSError:
                    self.process.wait(timeout=remaining)
        _, stderr = self.process.communicate(timeout=5)
        self.assertEqual(self.process.returncode, 0, stderr.decode())
        return json.loads(stderr)

    def test_arrows_edit_prompt_and_up_arrow_recalls_it_for_script(self):
        self.edit(
            b"Warm vice." + b"\x1b[D" * 5 + b"\x1b[C" + b"o\r",
            b"\x1b[A\x05 Again.\r",
        )
        self.stop_playback()
        self.assertEqual(self.finish(), [{
            "prompt": "Warm voice.",
            "reference_text": "Warm voice. Again.",
        }])

    def test_control_keys_move_cursor_and_clear_line_or_suffix(self):
        self.edit(
            b"Warm voice.\x01A \x05 Nice.\r",
            b"discarded script\x01\x0bHello there.\r",
            command=b"mistyped command\x15g 1\r",
        )
        self.stop_playback()
        self.assertEqual(self.finish(), [{
            "prompt": "A Warm voice. Nice.",
            "reference_text": "Hello there.",
        }])

    def test_line_editing_still_works_after_stopping_playback(self):
        self.type(b"1\r")
        self.wait_for(b"PLAYBACK_READY")
        self.stop_playback()
        self.edit(
            b"discarded prompt\x15Warm vice." + b"\x1b[D" * 4 + b"o\r",
            b"Hello there.\x01Well, \x05 Goodbye.\r",
        )
        self.stop_playback()
        self.assertEqual(self.finish(), [{
            "prompt": "Warm voice.",
            "reference_text": "Well, Hello there. Goodbye.",
        }])


if __name__ == "__main__":
    unittest.main()
