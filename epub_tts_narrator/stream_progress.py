"""TTY-only live display for concurrent model streams."""

from __future__ import annotations

import os
from threading import Lock, Thread
from time import monotonic, sleep
import unicodedata


def _seconds(value):
    value = max(0, int(value))
    return f"{value // 60}:{value % 60:02d}"


def _size(value):
    return f"{value / 1024:.1f} KiB"


# The console of the running command, for work that reports its own batches (``step_progress``).
_ACTIVE = None


def activate(console):
    """Make ``console`` the one ``step_progress`` reports to; returns the previous one."""
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, console
    return previous


def step_progress(done, total, what=""):
    """Report how much of the running check is done ("12 of 40 scan batches"); nothing without a console."""
    console = _ACTIVE
    if console is not None:
        console.set_step_progress(done, total, what)


def time_left(elapsed, fraction):
    """An estimate of the time left from the time taken so far, or None while too little is done to tell."""
    if fraction < 0.03 or fraction >= 1 or elapsed < 15:
        return None
    return elapsed * (1 - fraction) / fraction


def _clean(text):
    text = "".join(" " if unicodedata.category(char).startswith("C") else char
                   for char in text)
    return " ".join(text.split())


class StreamConsole:
    """Two continuously updated status lines over ordinary progress output."""

    _STEPS = ((0.50, "Casting"), (0.90, "Attribution"),
              (0.99, "Review and completion"), (1.00, "Validation and publication"))

    def __init__(self, stream, *, width=120, draw_interval=0.25):
        self.stream = stream
        self.enabled = bool(getattr(stream, "isatty", lambda: False)())
        self.width = max(40, int(width))
        self.draw_interval = max(0.0, float(draw_interval))
        self._lock = Lock()
        self._active = {}
        self._preview = ""
        self._notice = ""
        self._drawn = False
        self._last_draw = 0.0
        self._thread = None
        self._pending = ""
        self._overall = 0.0
        self._stage = ""
        # (number, count, name) while the saved book's checks run (``set_check``), else None.
        self._check = None
        # (done, total, what) the running check reported (``step_progress``), else None.
        self._check_progress = None
        # When the shown step (or the part of a check reporting progress) started, for the time left.
        self._step_key, self._step_started = None, monotonic()

    def set_progress(self, fraction, stage):
        if not self.enabled:
            return
        try:
            fraction = min(1.0, max(0.0, float(fraction)))
        except (TypeError, ValueError):
            return
        with self._lock:
            self._overall = max(self._overall, fraction)
            self._stage = str(stage)
            if self._active:
                self._draw_locked(monotonic())

    def set_check(self, number, count, name):
        """Show the saved book's checks as the step: "step 3/9 Checking the book 22% · each character's gender"."""
        if not self.enabled:
            return
        with self._lock:
            self._check = (number, count, name)
            self._check_progress = None
            self._stage = name
            self._after_check_locked()

    def set_step_progress(self, done, total, what=""):
        if not self.enabled or not self._check or type(total) is not int or total < 1:
            return
        with self._lock:
            self._check_progress = (min(max(0, done), total), total, str(what))
            name = self._check[2]
            self._stage = f"{name} · {done}/{total} {what}".rstrip()
            self._after_check_locked()

    def end_checks(self):
        if not self.enabled:
            return
        with self._lock:
            self._check = None
            self._check_progress = None

    def _after_check_locked(self):
        if self._active:
            self._draw_locked(monotonic())

    def close(self):
        if not self.enabled:
            return
        with self._lock:
            self._active.clear()
            self._preview = ""
            self._notice = ""
            self._clear_locked()

    def write(self, text):
        if not self.enabled:
            return self.stream.write(text)
        with self._lock:
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self._message_locked(line)
        return len(text)

    def flush(self):
        if not self.enabled:
            return self.stream.flush()
        with self._lock:
            if self._pending:
                self._message_locked(self._pending)
                self._pending = ""
            self.stream.flush()

    def progress(self, message):
        if not self.enabled:
            print(message, file=self.stream, flush=True)
            return
        with self._lock:
            self._message_locked(str(message))

    def _message_locked(self, message):
        self._clear_locked()
        self.stream.write(message + "\n")
        self.stream.flush()
        if self._active:
            self._draw_locked(monotonic())

    def observe(self, event):
        if not self.enabled or not isinstance(event, dict):
            return
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            return
        now = monotonic()
        with self._lock:
            phase = event.get("phase")
            if phase == "started":
                self._active[request_id] = {
                    "started": now,
                    "last": now,
                    "received": 0,
                    "events": 0,
                    "reasoning": 0,
                    "content": 0,
                }
                self._notice = "Waiting for first provider event"
                self._ensure_thread_locked()
            elif phase == "streaming":
                state = self._active.get(request_id)
                if state is None:
                    return
                state["last"] = now
                state["received"] = max(state["received"], int(event.get("received", 0) or 0))
                state["events"] = max(state["events"], int(event.get("events", 0) or 0))
                state["reasoning"] = max(state["reasoning"], int(event.get("reasoning_chars", 0) or 0))
                state["content"] = max(state["content"], int(event.get("content_chars", 0) or 0))
                if event.get("kind") == "reasoning" and isinstance(event.get("text"), str):
                    addition = _clean(event["text"])
                    if addition:
                        separator = " " if self._preview else ""
                        self._preview = (self._preview + separator + addition)[-240:]
                self._notice = ""
            elif phase == "retry":
                if request_id in self._active:
                    self._notice = (f"Retrying provider request {event.get('attempt', '?')} "
                                    f"in {float(event.get('delay', 0)):.1f}s")
            elif phase in {"completed", "failed"}:
                self._active.pop(request_id, None)
                self._notice = ""
                if not self._active:
                    self._preview = ""
                    self._clear_locked()
                    return
            if now - self._last_draw >= self.draw_interval:
                self._draw_locked(now)

    def _ensure_thread_locked(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = Thread(target=self._heartbeat, name="llm-stream-status", daemon=True)
            self._thread.start()

    def _heartbeat(self):
        while True:
            with self._lock:
                if not self._active:
                    self._thread = None
                    return
                self._draw_locked(monotonic())
            sleep(1.0)

    def _step_status(self):
        if self._check is not None:
            number, count, name = self._check
            if self._check_progress is not None:
                done, total, _ = self._check_progress
                return number, count, "Checking the book", done / total
            return number, count, "Checking the book", 0.0
        start = 0.0
        for index, (end, name) in enumerate(self._STEPS, 1):
            if self._overall <= end or end == self._STEPS[-1][0]:
                fraction = 0.0 if end == start else (self._overall - start) / (end - start)
                return index, len(self._STEPS), name, min(1.0, max(0.0, fraction))
            start = end
        return len(self._STEPS), len(self._STEPS), self._STEPS[-1][1], 1.0

    def _time_left(self, now):
        """ "(~4:10 left in this step)" once enough of the step is done to estimate it, else "". """
        step, _, name, fraction = self._step_status()
        what = self._check_progress[2] if self._check is not None and self._check_progress else ""
        key = (step, name, what)
        if key != self._step_key:
            self._step_key, self._step_started = key, now
        left = time_left(now - self._step_started, fraction)
        return "" if left is None else f"(~{_seconds(left)} left in this step)"

    def _terminal_width(self):
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            return self.width

    def _draw_locked(self, now):
        lines = self._status_lines(now)
        if lines is None:
            self._clear_locked()
            return
        first, second = lines
        width = max(40, self._terminal_width())
        self._clear_locked()
        self.stream.write(first[:max(1, width - 1)] + "\n")
        self.stream.write(second[:max(1, width - 1)] + "\n")
        self.stream.flush()
        self._drawn = True
        self._last_draw = now

    def _status_lines(self, now):
        """The two live lines (model activity, then totals), or None with no active stream."""
        states = list(self._active.values())
        if not states:
            return None
        oldest = min(state["started"] for state in states)
        received = sum(state["received"] for state in states)
        events = sum(state["events"] for state in states)
        reasoning = sum(state["reasoning"] for state in states)
        content = sum(state["content"] for state in states)
        quiet = now - max(state["last"] for state in states)
        if self._preview:
            first = "Thinking: " + self._preview
        elif self._notice:
            first = self._notice
        elif received:
            first = "Streaming response; provider exposes no reasoning text"
        else:
            first = "Waiting for provider response"
        if quiet >= 1:
            first += f" · quiet {_seconds(quiet)}"
        step, step_count, step_name, step_fraction = self._step_status()
        left = self._time_left(now)
        second = (f"LLM {len(states)} active | step {step}/{step_count} {step_name} "
                  f"{step_fraction:.0%}{' ' + left if left else ''} · {self._stage} | elapsed {_seconds(now - oldest)} | "
                  f"{events} events | thought {reasoning:,} chars | "
                  f"answer {content:,} chars | {_size(received)} received")
        return first, second

    def _clear_locked(self):
        if not self._drawn:
            return
        self.stream.write("\x1b[2A\x1b[J")
        self.stream.flush()
        self._drawn = False


class BusStreamConsole(StreamConsole):
    """``StreamConsole`` for a full-screen run shell: the same status, as run events.

    Messages become log events; the two live lines (and the overall stage
    fraction) become a ``progress`` event the shell shows in its status view.
    Nothing is drawn on a terminal.
    """

    def __init__(self, bus, *, draw_interval=0.5):
        self.bus = bus
        super().__init__(_NoStream(), draw_interval=draw_interval)
        self.enabled = True

    def write(self, text):
        with self._lock:
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self.bus.log(line)
        return len(text)

    def flush(self):
        with self._lock:
            if self._pending:
                self.bus.log(self._pending)
                self._pending = ""

    def _message_locked(self, message):
        self.bus.log(message)

    def _after_check_locked(self):
        self._draw_locked(monotonic())

    def set_progress(self, fraction, stage):
        super().set_progress(fraction, stage)
        with self._lock:
            if not self._active:
                # The stage fraction moves between model requests too.
                self._draw_locked(monotonic())

    def _draw_locked(self, now):
        lines = self._status_lines(now)
        step, step_count, step_name, step_fraction = self._step_status()
        left = self._time_left(now)
        self.bus.publish("progress", data={
            "overall": self._overall, "stage": self._stage, "step": step, "steps": step_count,
            "step_name": step_name, "step_fraction": step_fraction, "left": left[1:-1] if left else "",
            "streams": len(self._active),
            "received": sum(state["received"] for state in self._active.values()),
            "activity": lines[0] if lines else "", "totals": lines[1] if lines else ""})
        self._last_draw = now

    def _clear_locked(self):
        pass


class _NoStream:
    def isatty(self):
        return False

    def write(self, text):
        return len(text)

    def flush(self):
        pass
