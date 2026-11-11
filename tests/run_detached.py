"""Run unittest discovery without a terminal, so no test can stop to wait for input.

Several commands prompt whenever standard input or ``/dev/tty`` is a terminal.
Run from an interactive shell, a test that reaches one of those prompts waits
forever. This starts discovery in its own session (no controlling terminal) with
standard input from ``/dev/null``; output still goes to the caller's terminal,
and Ctrl-C is passed on to the tests.

    .venv-renderer/bin/python tests/run_detached.py [unittest discover options]
"""

from pathlib import Path
import signal
import subprocess
import sys


def main(arguments):
    root = Path(__file__).resolve().parent.parent
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q", *arguments]
    child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, start_new_session=True)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            # The child is in its own session, so the terminal's Ctrl-C reaches only us.
            child.send_signal(signal.SIGINT)


if __name__ == "__main__":
    status = main(sys.argv[1:])
    sys.exit(130 if status == -signal.SIGINT else (status if status >= 0 else 1))
