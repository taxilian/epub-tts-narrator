"""Decide when a command uses its full-screen terminal UI (stdlib only).

Full-screen UIs are built with Textual. The normal setup installs it
(``renderer.requirements.txt``, or the analyzer's ``.[tui]`` extra) while the
core package keeps no required dependencies. A command uses one only on an interactive terminal, when Textual
is installed, and without ``--plain``; otherwise it keeps its line-based
output exactly, so scripts, pipes and logs see what they always have.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import os
import sys


INSTALL_HINT = ("The full-screen interface needs Textual, which this environment lacks; rerun the "
                "install steps (uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt, "
                'or pip install -e ".[tui]" for the analyzer). Using the line interface.')


def textual_available():
    try:
        return importlib.util.find_spec("textual") is not None
    except (ImportError, ValueError):
        return False


def add_plain_option(parser):
    parser.add_argument("--plain", action="store_true",
                        help="Use line-based output instead of the full-screen interface")
    return parser


def interactive_terminal(stdin=None, stdout=None, environ=None):
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    environ = os.environ if environ is None else environ
    try:
        interactive = stdin.isatty() and stdout.isatty()
    except (AttributeError, ValueError):
        return False
    return bool(interactive and environ.get("TERM") not in (None, "", "dumb"))


def wants_tui(args=None, *, stdin=None, stdout=None, environ=None, available=None, warn=None):
    """Whether to open the full-screen UI for this invocation.

    On an interactive terminal without Textual, ``warn`` (default: stderr)
    receives one hint and the line interface is used.
    """
    if getattr(args, "plain", False) or not interactive_terminal(stdin, stdout, environ):
        return False
    if available is None:
        available = textual_available()
    if not available:
        (warn or (lambda text: print(text, file=sys.stderr)))(INSTALL_HINT)
        return False
    return True


_PROMPT_STREAMS = None


@contextmanager
def prompt_streams(input_stream, output_stream):
    """While a full-screen UI is open, terminal questions use these streams.

    Code that asks a person something on the terminal (model download
    confirmation) reads ``active_prompt_streams()`` before touching the real
    terminal, which the UI owns. Process-wide, because the asking code runs
    on a worker thread.
    """
    global _PROMPT_STREAMS
    previous, _PROMPT_STREAMS = _PROMPT_STREAMS, (input_stream, output_stream)
    try:
        yield
    finally:
        _PROMPT_STREAMS = previous


def active_prompt_streams():
    """``(input, output)`` of the open full-screen UI, or ``None``."""
    return _PROMPT_STREAMS


_SCREEN_HOST = None


@contextmanager
def screen_host(show):
    """While a full-screen UI is open, ``show(make_screen)`` opens a screen and waits for its result.

    Code on the command's worker thread that has a screen of its own for a
    question (the speaker chooser) asks through ``active_screen_host()``
    instead of a line prompt. ``make_screen`` runs on the UI thread.
    """
    global _SCREEN_HOST
    previous, _SCREEN_HOST = _SCREEN_HOST, show
    try:
        yield
    finally:
        _SCREEN_HOST = previous


def active_screen_host():
    """The open full-screen UI's ``show(make_screen)``, or ``None``."""
    return _SCREEN_HOST
