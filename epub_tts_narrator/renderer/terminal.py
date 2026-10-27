"""Terminal line editing and interruptible sample playback."""

from __future__ import annotations

import io
import os
import select
import sys
import time

try:
    import termios
except ImportError:  # A redirected/non-POSIX session can still play samples.
    termios = None


def enable_line_editing():
    """Give input() the platform's line editor and existing user key bindings.

    Python scripts do not automatically load readline as the interactive Python
    shell does. Importing it enables arrows, editing shortcuts, and session
    history using GNU readline or macOS libedit. Do not override the user's
    editing mode or write a history file.
    """
    try:
        import readline  # noqa: F401 -- registers Python's input() editor
    except ImportError:
        return False  # Redirected input and platforms without readline still work.
    return True


def _input_fd(stream):
    try:
        descriptor = stream.fileno()
    except (AttributeError, io.UnsupportedOperation, OSError, ValueError):
        return None
    return descriptor if type(descriptor) is int and descriptor >= 0 else None


def _discard_key_tail(descriptor):
    """Consume a terminal key's escape/UTF-8 sequence and any queued newline.

    Terminal emulators can deliver an arrow key or CRLF in several writes.
    Wait for a short quiet interval, with a hard bound even during key repeat.
    Anything entered before the menu reappears belongs to stopping playback.
    """
    deadline = time.monotonic() + 0.20
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        ready, _, _ = select.select([descriptor], [], [], min(0.04, remaining))
        if not ready or not os.read(descriptor, 4096):
            return


def play_sample(player, path, *, stream=None):
    """Play until any key, input EOF, or natural completion.

    Returns True only when the player confirmed natural completion, and False
    when a key or input EOF stopped playback early; callers that auto-accept
    listened audio must check this value. A player that reports failure is
    never treated as heard: ``AudioPlayer.poll()``/``wait()`` raise on a
    nonzero player exit, and that error propagates out of this function.
    ``player.poll()`` reports whether audio is still playing. TTY input becomes
    noncanonical only for playback, preserving signal keys such as Ctrl-C.
    Redirected input is polled without a blocking readline; streams without a
    file descriptor simply wait for the audio to finish.
    """
    stream = sys.stdin if stream is None else stream
    descriptor = _input_fd(stream)
    original = None
    terminal = (
        descriptor is not None and termios is not None and os.isatty(descriptor)
    )
    try:
        if terminal:
            original = termios.tcgetattr(descriptor)
            changed = original[:]
            changed[6] = original[6][:]
            changed[3] &= ~(termios.ECHO | termios.ICANON)
            changed[6][termios.VMIN] = 1
            changed[6][termios.VTIME] = 0
            termios.tcsetattr(descriptor, termios.TCSANOW, changed)
        player.start(path)
        if descriptor is None:
            player.wait()
            return True
        while player.poll():
            ready, _, _ = select.select([descriptor], [], [], 0.05)
            if ready:
                key = os.read(descriptor, 1)
                # Stop immediately, before collecting a split escape sequence.
                player.stop()
                if terminal and key:
                    _discard_key_tail(descriptor)
                    termios.tcflush(descriptor, termios.TCIFLUSH)
                return False
        return True
    finally:
        try:
            player.stop()
        finally:
            if original is not None:
                termios.tcsetattr(descriptor, termios.TCSANOW, original)


def wait_key(seconds, *, stream=None):
    """Wait up to ``seconds`` for a single key; never blocks past the wait.

    Returns ``"key"`` when a key arrived (the key's escape/UTF-8 tail and any
    queued newline are consumed), ``"eof"`` when input reached EOF, and None
    on timeout. TTY input is noncanonical only for the wait; redirected input
    is polled so a closed pipe ends the wait instead of hanging.
    """
    stream = sys.stdin if stream is None else stream
    descriptor = _input_fd(stream)
    if descriptor is None:
        time.sleep(seconds)
        return None
    original = None
    terminal = termios is not None and os.isatty(descriptor)
    try:
        if terminal:
            original = termios.tcgetattr(descriptor)
            changed = original[:]
            changed[6] = original[6][:]
            changed[3] &= ~(termios.ECHO | termios.ICANON)
            changed[6][termios.VMIN] = 1
            changed[6][termios.VTIME] = 0
            termios.tcsetattr(descriptor, termios.TCSANOW, changed)
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([descriptor], [], [], min(0.05, remaining))
            if not ready:
                continue
            key = os.read(descriptor, 1)
            if not key:
                return "eof"
            if terminal:
                _discard_key_tail(descriptor)
                termios.tcflush(descriptor, termios.TCIFLUSH)
            return "key"
    finally:
        if original is not None:
            termios.tcsetattr(descriptor, termios.TCSANOW, original)
