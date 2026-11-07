"""TTY-only live display for concurrent model streams."""

from __future__ import annotations

from threading import Lock, Thread
from time import monotonic, sleep
import unicodedata


def _seconds(value):
    value = max(0, int(value))
    return f"{value // 60}:{value % 60:02d}"


def _size(value):
    return f"{value / 1024:.1f} KiB"


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
        start = 0.0
        for index, (end, name) in enumerate(self._STEPS, 1):
            if self._overall <= end or end == self._STEPS[-1][0]:
                fraction = 0.0 if end == start else (self._overall - start) / (end - start)
                return index, len(self._STEPS), name, min(1.0, max(0.0, fraction))
            start = end
        return len(self._STEPS), len(self._STEPS), self._STEPS[-1][1], 1.0

    def _draw_locked(self, now):
        states = list(self._active.values())
        if not states:
            self._clear_locked()
            return
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
        second = (f"LLM {len(states)} active | step {step}/{step_count} {step_name} "
                  f"{step_fraction:.0%} · {self._stage} | elapsed {_seconds(now - oldest)} | "
                  f"{events} events | thought {reasoning:,} chars | "
                  f"answer {content:,} chars | {_size(received)} received")
        self._clear_locked()
        self.stream.write(first[:max(1, self.width - 1)] + "\n")
        self.stream.write(second[:max(1, self.width - 1)] + "\n")
        self.stream.flush()
        self._drawn = True
        self._last_draw = now

    def _clear_locked(self):
        if not self._drawn:
            return
        self.stream.write("\x1b[2A\x1b[J")
        self.stream.flush()
        self._drawn = False
