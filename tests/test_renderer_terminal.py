import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import unittest

from audiobook_renderer.terminal import play_sample

try:
    import pty
    import termios
except ImportError:
    pty = termios = None


CHILD = r'''
import fcntl
import json
import os
import select
import sys
import termios
import time
from audiobook_renderer.terminal import play_sample

fcntl.ioctl(0, termios.TIOCSCTTY, 0)
original = termios.tcgetattr(0)
mode = sys.argv[1]

class Player:
    def __init__(self):
        self.active = False
        self.stops = 0
        self.started = None

    def start(self, path):
        current = termios.tcgetattr(0)
        assert not current[3] & (termios.ECHO | termios.ICANON)
        assert current[3] & termios.ISIG == original[3] & termios.ISIG
        self.active = True
        self.started = time.monotonic()
        print("READY", flush=True)
        if mode == "start-error":
            raise RuntimeError("start failed")

    def poll(self):
        elapsed = time.monotonic() - self.started
        if mode == "poll-error" and elapsed >= 0.10:
            raise RuntimeError("poll failed")
        if mode == "natural" and elapsed >= 0.15:
            self.active = False
        return self.active

    def stop(self):
        if self.active:
            self.stops += 1
        self.active = False
        if mode == "stop-error":
            raise RuntimeError("stop failed")

    def wait(self):
        raise AssertionError("TTY should poll")

player = Player()
failure = None
try:
    play_sample(player, "sample.wav")
except (KeyboardInterrupt, RuntimeError) as exc:
    failure = type(exc).__name__
restored_attributes = termios.tcgetattr(0)
# Darwin sets PENDIN when canonical mode is restored, to reprocess any queued
# input. It is kernel bookkeeping rather than a changed user terminal mode.
restored_attributes[3] &= ~getattr(termios, "PENDIN", 0)
expected_attributes = original[:]
expected_attributes[3] &= ~getattr(termios, "PENDIN", 0)
restored = restored_attributes == expected_attributes
# Inspect bytes still queued after restoration, without blocking in canonical
# mode: an unconsumed arrow-key suffix would otherwise hide until Enter.
probe = termios.tcgetattr(0)
probe[3] &= ~(termios.ECHO | termios.ICANON)
probe[6][termios.VMIN] = 1
probe[6][termios.VTIME] = 0
termios.tcsetattr(0, termios.TCSANOW, probe)
pending = os.read(0, 4096) if select.select([0], [], [], 0.08)[0] else b""
termios.tcsetattr(0, termios.TCSANOW, original)
print(json.dumps({
    "restored": restored,
    "stops": player.stops,
    "active": player.active,
    "failure": failure,
    "pending": pending.hex(),
}), flush=True)
'''


class FakePlayer:
    def __init__(self, duration=0.10):
        self.duration = duration
        self.started = None
        self.stopped = False
        self.waited = False

    def start(self, path):
        self.started = time.monotonic()

    def poll(self):
        return time.monotonic() - self.started < self.duration

    def wait(self):
        self.waited = True

    def stop(self):
        self.stopped = True


@unittest.skipUnless(pty is not None and termios is not None, "Requires POSIX PTYs")
class TerminalPlaybackTests(unittest.TestCase):
    def run_child(self, mode="key", writes=()):
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", CHILD, mode],
            cwd=Path(__file__).resolve().parent.parent,
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        os.close(slave)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 3)[0])
            self.assertEqual(process.stdout.readline(), b"READY\n")
            for delay, value in writes:
                if delay:
                    time.sleep(delay)
                os.write(master, value)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0, stderr.decode())
            result = json.loads(stdout)
            self.assertTrue(result["restored"])
            self.assertFalse(result["active"])
            self.assertEqual(result["pending"], "")
            return result
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()
            os.close(master)

    def test_key_without_enter_stops_playback(self):
        result = self.run_child(writes=[(0, b"x")])
        self.assertEqual(result["stops"], 1)
        self.assertIsNone(result["failure"])

    def test_natural_completion_returns_without_input_and_restores_terminal(self):
        result = self.run_child("natural")
        self.assertIsNone(result["failure"])

    def test_split_arrow_key_does_not_leak_into_menu(self):
        result = self.run_child(writes=[(0, b"\x1b"), (0.01, b"["), (0.01, b"A")])
        self.assertEqual(result["stops"], 1)

    def test_split_crlf_does_not_leak_into_menu(self):
        self.run_child(writes=[(0, b"\r"), (0.01, b"\n")])

    def test_utf8_key_does_not_leak_into_menu(self):
        self.run_child(writes=[(0, "é".encode("utf-8"))])

    def test_ctrl_c_propagates_after_cleanup(self):
        result = self.run_child(writes=[(0, b"\x03")])
        self.assertEqual(result["failure"], "KeyboardInterrupt")
        self.assertEqual(result["stops"], 1)

    def test_start_failure_restores_terminal(self):
        result = self.run_child("start-error")
        self.assertEqual(result["failure"], "RuntimeError")

    def test_poll_failure_stops_playback_and_restores_terminal(self):
        result = self.run_child("poll-error")
        self.assertEqual(result["failure"], "RuntimeError")
        self.assertEqual(result["stops"], 1)

    def test_stop_failure_still_restores_terminal(self):
        result = self.run_child("stop-error", writes=[(0, b"x")])
        self.assertEqual(result["failure"], "RuntimeError")


class RedirectedPlaybackTests(unittest.TestCase):
    def test_stream_without_descriptor_waits_for_audio(self):
        player = FakePlayer()
        play_sample(player, "sample.wav", stream=io.StringIO(""))
        self.assertTrue(player.waited)
        self.assertTrue(player.stopped)

    def test_pipe_without_input_returns_at_natural_completion(self):
        reader, writer = os.pipe()
        try:
            with os.fdopen(reader, "rb", buffering=0) as stream:
                player = FakePlayer()
                started = time.monotonic()
                play_sample(player, "sample.wav", stream=stream)
                self.assertLess(time.monotonic() - started, 1)
                self.assertTrue(player.stopped)
                self.assertFalse(player.waited)
        finally:
            os.close(writer)

    def test_pipe_eof_stops_playback(self):
        reader, writer = os.pipe()
        os.close(writer)
        with os.fdopen(reader, "rb", buffering=0) as stream:
            player = FakePlayer(duration=10)
            started = time.monotonic()
            play_sample(player, "sample.wav", stream=stream)
            self.assertLess(time.monotonic() - started, 1)
            self.assertTrue(player.stopped)

    def test_pipe_input_stops_playback(self):
        reader, writer = os.pipe()
        try:
            os.write(writer, b" ")
            with os.fdopen(reader, "rb", buffering=0) as stream:
                player = FakePlayer(duration=10)
                play_sample(player, "sample.wav", stream=stream)
                self.assertTrue(player.stopped)
        finally:
            os.close(writer)


if __name__ == "__main__":
    unittest.main()
