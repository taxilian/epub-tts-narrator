"""Structured run events for full-screen interfaces (stdlib only).

Commands report progress through ``notify`` callbacks and, for rendering, a
live board (``renderer/progress.py``). When a full-screen run shell is active,
those same calls become ``Event`` records on an ``EventBus`` instead of
terminal writes; nothing changes for plain output, where no bus exists.

Events carry only what the line output already shows (status text, IDs,
numbers, sanitized errors), never extra book text or credentials.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
import threading
import time
from typing import Any


LOG, STATUS, STATUS_REMOVED, OUTCOME, FAILURE, FINISHED, PROGRESS, HOLD, BOOK, REJECTED, REBUILD = (
    "log", "status", "status_removed", "outcome", "failure", "finished", "progress", "hold", "book",
    "rejected", "rebuild")
SEVERITIES = ("debug", "info", "warning", "error")


@dataclass
class Event:
    kind: str
    time: float
    text: str = ""
    severity: str = "info"
    scope: dict = field(default_factory=dict)
    data: Any = None


# Markers the existing line output already uses for problems.
_ERROR = re.compile(r"(^=== Failed|(?<!\b0 )\bfailed\b|^Error\b|^Renderer stopped|\bblocked\b|Traceback)", re.IGNORECASE)
_WARNING = re.compile(r"(\bwarning\b|\bcould not\b|\bretry(ing)?\b|\brecheck|\bstopping\b|\binterrupted\b)",
                      re.IGNORECASE)
_BOOK = re.compile(r"^\[(?P<book>[^\]\n]{1,200})\] ")
_CHAPTER = re.compile(r"^(?:Ch|Chapter) (?P<chapter>\d+)(?:/\d+| / \d+)?: ")
_CHUNK = re.compile(r"\bChunk (?P<chunk>\d+)\b")
_WORKER = re.compile(r"\((?:worker|audio from) (?P<worker>[A-Za-z0-9][A-Za-z0-9_.-]{0,63})\)|"
                     r"^\s*(?P<worker2>[A-Za-z0-9][A-Za-z0-9_.-]{0,63}): speech generation")


def classify(text):
    """Severity for one line of existing output, from the markers it already uses."""
    if _ERROR.search(text):
        return "error"
    if _WARNING.search(text):
        return "warning"
    return "info"


def parse_scope(text, scope=None):
    """``(scope, remaining_text)`` from the book/chapter prefixes the board adds."""
    scope = dict(scope or {})
    match = _BOOK.match(text)
    if match:
        scope.setdefault("book", match.group("book"))
        text = text[match.end():]
    match = _CHAPTER.match(text)
    if match:
        scope.setdefault("chapter", int(match.group("chapter")))
        text = text[match.end():]
    match = _CHUNK.search(text)
    if match:
        scope.setdefault("chunk", int(match.group("chunk")))
    match = _WORKER.search(text)
    if match:
        scope.setdefault("worker", match.group("worker") or match.group("worker2"))
    return scope, text


class EventBus:
    """Thread-safe fan-out of events to subscribers, with a bounded history."""

    def __init__(self, *, history=20000, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._subscribers = []
        self.history = deque(maxlen=history)

    def subscribe(self, callback):
        with self._lock:
            self._subscribers.append(callback)
            return list(self.history)

    def publish(self, kind, text="", *, severity=None, scope=None, data=None):
        event = Event(kind, self._clock(), str(text), severity or "info", dict(scope or {}), data)
        with self._lock:
            if kind not in (STATUS, STATUS_REMOVED, PROGRESS):
                # Live rows refresh every second; only lasting events are replayed.
                self.history.append(event)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                # A closing screen must never stop the work it displays.
                pass
        return event

    def log(self, text, *, scope=None, severity=None):
        """One line of existing output; scope and severity come from its own prefixes."""
        text = str(text).rstrip("\n")
        events = []
        for line in text.split("\n") or [""]:
            if not line.strip():
                continue
            parsed, body = parse_scope(line, scope)
            events.append(self.publish(LOG, body, severity=severity or classify(body), scope=parsed))
        return events


_ACTIVE_BUS = None


def set_active_bus(bus):
    """The run shell's bus while it is open (process-wide; commands run on a worker thread)."""
    global _ACTIVE_BUS
    previous, _ACTIVE_BUS = _ACTIVE_BUS, bus
    return previous


def active_bus():
    return _ACTIVE_BUS
