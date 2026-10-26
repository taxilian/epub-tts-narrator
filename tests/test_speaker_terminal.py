"""Exercise actual terminal input when a shell pipeline owns standard input."""

import errno
import os
from pathlib import Path
import select
import signal
import sys
import time
import unittest

try:
    import pty
except ImportError:
    pty = None


@unittest.skipUnless(pty is not None and hasattr(os, "fork"), "POSIX terminal required")
class SpeakerTerminalTests(unittest.TestCase):
    def test_redirected_stdin_prompts_waits_and_reads_the_controlling_terminal(self):
        program = """
import os, sys
sys.path[:0] = [sys.argv[1], sys.argv[1] + '/tests']
from test_speaker_decisions import missing
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from epub_tts_narrator.speaker_completion import unassigned_segments
try:
    with open('/dev/tty', 'r'), open('/dev/tty', 'w'):
        pass
except PermissionError:
    print('TERMINAL_ACCESS_BLOCKED', flush=True)
    sys.exit(77)
with open(os.devnull, 'r') as redirected:
    os.dup2(redirected.fileno(), 0)
assert not sys.stdin.isatty()
result = prompt_speaker_decisions(missing(), book_path='saved-book')
assert not unassigned_segments(result)
assert result['segments'][1]['role_id'] == 'mara'
print('TERMINAL_SELECTION_SAVED', flush=True)
"""
        self.run_terminal_program(program, reprompt=True)

    def test_saved_analysis_with_truncated_jury_reaches_controlling_terminal(self):
        program = """
import os, sys
sys.path[:0] = [sys.argv[1], sys.argv[1] + '/tests']
from unittest.mock import patch
from test_analyze_resume import AnalyzeResumeTests, TruncatedCompletionClient
from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.cli import main
from epub_tts_narrator.speaker_completion import unassigned_segments
try:
    with open('/dev/tty', 'r'), open('/dev/tty', 'w'):
        pass
except PermissionError:
    print('TERMINAL_ACCESS_BLOCKED', flush=True)
    sys.exit(77)
fixture = AnalyzeResumeTests()
fixture.setUp()
try:
    _, path, original = fixture.seed()
    with open(os.devnull, 'r') as redirected:
        os.dup2(redirected.fileno(), 0)
    assert not sys.stdin.isatty()
    client = TruncatedCompletionClient()
    with patch('epub_tts_narrator.cli.OpenAICompatibleClient', return_value=client):
        status = main(['analyze', str(path), '--config', str(fixture.config)])
    assert status == 0, status
    assert len(client.calls) == 4, len(client.calls)
    result = _completed_directory(path)[0]
    assert not unassigned_segments(result)
    assert any(row['attribution']['basis'] == 'user' for row in result['segments'])
    assert result['source'] == original['source']
    print('TERMINAL_SELECTION_SAVED', flush=True)
finally:
    fixture.doCleanups()
"""
        self.run_terminal_program(program)

    def run_terminal_program(self, program, *, reprompt=False):
        root = Path(__file__).resolve().parents[1]
        pid, descriptor = pty.fork()
        if pid == 0:
            os.execv(sys.executable, [sys.executable, "-u", "-c", program, str(root)])
        exited = False
        received = b""

        def until(marker):
            nonlocal received
            deadline = time.monotonic() + 15
            while marker not in received:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                    self.fail("Terminal did not reach expected prompt: " + received.decode(errors="replace"))
                try:
                    chunk = os.read(descriptor, 65536)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if not chunk:
                    if b"TERMINAL_ACCESS_BLOCKED" in received:
                        self.skipTest("Sandbox prevents the child from opening /dev/tty")
                    self.fail("Terminal closed before selection: " + received.decode(errors="replace"))
                received += chunk
            before, received = received.split(marker, 1)
            return before

        try:
            prompt = b"Speaker number(s), Enter = 1, O = Other, N = new voice, Q = save and stop: "
            until(prompt)
            finished, _ = os.waitpid(pid, os.WNOHANG)
            self.assertEqual(finished, 0, "The child must wait for a speaker choice")
            if reprompt:
                os.write(descriptor, b"o\n")
                expanded_prompt = b"Speaker number(s), B = back, N = new voice, Q = save and stop: "
                self.assertIn(b"All voices", until(expanded_prompt))
                os.write(descriptor, b"\n")
                self.assertIn(b"Enter selects a voice only", until(expanded_prompt))
                os.write(descriptor, b"b\n")
                self.assertIn(b"Most likely speakers", until(prompt))
            os.write(descriptor, b"\n")
            until(b"TERMINAL_SELECTION_SAVED")
            _, status = os.waitpid(pid, 0)
            exited = True
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        finally:
            if not exited:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
            os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
