"""State of a full-screen run shell, built from run events (stdlib only, no UI).

``RunModel`` subscribes to an ``EventBus`` and keeps what the screen shows:
live chapter rows with their numbers, per-book outcomes, every failure with
its saved record, the log, worker activity derived from the rows, and the
person's filters. It is thread-safe: events arrive from command threads while
the screen reads snapshots.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time

from .run_events import FAILURE, FINISHED, HOLD, LOG, OUTCOME, PROGRESS, SEVERITIES, STATUS, STATUS_REMOVED


SEVERITY_LEVELS = ("info", "warning", "error", "debug")  # cycle order; "debug" shows everything


@dataclass
class RunFilter:
    book: str | None = None
    chapter: int | None = None
    worker: str | None = None
    voice: str | None = None
    severity: str = "info"
    text: str = ""

    def active(self):
        return any((self.book, self.chapter, self.worker, self.voice, self.text)) or self.severity != "info"

    def describe(self):
        parts = [f"{name}={value}" for name, value in (("book", self.book), ("chapter", self.chapter),
                                                       ("worker", self.worker), ("voice", self.voice))
                 if value is not None]
        if self.severity != "info":
            parts.append("all messages" if self.severity == "debug" else f"{self.severity} and worse")
        if self.text:
            parts.append(repr(self.text))
        return ", ".join(parts)

    def matches(self, scope, *, severity=None, text=""):
        # Unscoped lines (run-wide messages) stay visible with a book filter.
        if self.book is not None and scope.get("book") not in (None, self.book):
            return False
        if self.chapter is not None and scope.get("chapter") != self.chapter:
            return False
        if self.worker is not None and scope.get("worker") != self.worker:
            return False
        if self.voice is not None and scope.get("voice") not in (None, self.voice):
            return False
        if severity is not None and SEVERITIES.index(severity) < SEVERITIES.index(self.severity):
            return False
        words = self.text.casefold().split()
        return all(word in text.casefold() for word in words)


def next_severity(current):
    index = SEVERITY_LEVELS.index(current) if current in SEVERITY_LEVELS else 0
    return SEVERITY_LEVELS[(index + 1) % len(SEVERITY_LEVELS)]


class RunModel:
    """Everything the run shell displays, updated from events on any thread."""

    def __init__(self, title, *, clock=time.monotonic, log_limit=20000):
        self.title = title
        self._clock = clock
        self._lock = threading.RLock()
        self.started = clock()
        self.logs = deque(maxlen=log_limit)  # (sequence, event)
        self.log_total = 0
        self.rows = {}            # board key -> {"scope", "snapshot", "text", "updated"}
        self.completed = {}       # book (or None) -> chapters whose row closed
        self.books = {}           # book -> True/False final outcome
        self.failures = []        # {"kind", "scope", "error", "record", "attempt"}
        self.attempt = 1
        self.state = "running"    # running | finished | failed | interrupted
        self.exit_code = None
        self.error = None
        self.finished_at = None
        self.filter = RunFilter()
        self.seen = {"book": set(), "chapter": set(), "worker": set(), "voice": set()}
        self.stopping = False
        self.progress = None      # analysis stage and model-stream status (PROGRESS events)
        self.holds = {}           # book (or None) -> release callable while a render waits for retries
        self.changed = threading.Event()

    # Events --------------------------------------------------------------------

    def on_event(self, event):
        with self._lock:
            self._remember(event.scope)
            if event.kind == LOG:
                self.log_total += 1
                self.logs.append((self.log_total, event))
            elif event.kind == STATUS:
                key = event.data.get("key")
                row = self.rows.setdefault(key, {"scope": {}, "snapshot": None, "text": ""})
                row["scope"] = {**row["scope"], **event.scope}
                if event.data.get("snapshot") is not None:
                    row["snapshot"] = event.data["snapshot"]
                    for task in row["snapshot"].get("active", []):
                        if task.get("worker"):
                            self.seen["worker"].add(task["worker"])
                        self.seen["voice"].update(task.get("voices") or ())
                if event.text:
                    row["text"] = event.text
                row["updated"] = event.time
            elif event.kind == STATUS_REMOVED:
                row = self.rows.pop(event.data.get("key"), None)
                if row is not None:
                    book = row["scope"].get("book")
                    self.completed[book] = self.completed.get(book, 0) + 1
            elif event.kind == OUTCOME:
                book = event.scope.get("book")
                self.books[book] = bool(event.data.get("succeeded"))
                if not self.books[book]:
                    last = next((item.text for _, item in reversed(self.logs)
                                 if item.severity == "error" and item.scope.get("book") == book), "")
                    self.failures.append({"kind": "book", "scope": {"book": book}, "error": last,
                                          "record": {"book": event.data.get("key")}, "attempt": self.attempt})
            elif event.kind == FAILURE:
                data = dict(event.data or {})
                self.failures.append({"kind": "chapter", "scope": dict(event.scope), "error": event.text,
                                      "record": dict(data.get("record") or {}), "retry": data.get("retry"),
                                      "state": "failed", "attempt": self.attempt})
            elif event.kind == HOLD:
                book = event.scope.get("book")
                if event.data.get("release") is None:
                    self.holds.pop(book, None)
                else:
                    self.holds[book] = event.data["release"]
            elif event.kind == PROGRESS:
                self.progress = dict(event.data or {})
            elif event.kind == FINISHED:
                self.finish(event.data.get("exit_code"), event.data.get("error"))
        self.changed.set()

    def _remember(self, scope):
        for key in ("book", "chapter", "worker", "voice"):
            if scope.get(key) is not None:
                self.seen[key].add(scope[key])

    # Lifecycle -------------------------------------------------------------------

    def finish(self, exit_code, error=None, *, interrupted=False):
        with self._lock:
            self.exit_code = exit_code
            self.error = error
            self.finished_at = self._clock()
            self.rows.clear()
            if interrupted or exit_code == 130:
                self.state = "interrupted"
            elif exit_code == 0 and error is None and not self.current_failures():
                self.state = "finished"
            else:
                self.state = "failed"
        self.changed.set()

    def restart(self):
        """A resume run of the same command: earlier failures stay listed as an earlier attempt."""
        with self._lock:
            self.attempt += 1
            self.state, self.exit_code, self.error, self.finished_at = "running", None, None, None
            self.books.clear()
            self.completed.clear()
            self.rows.clear()
            self.stopping = False
            self.progress = None
            self.holds.clear()
        self.changed.set()

    @property
    def running(self):
        return self.state == "running"

    def should_hold(self):
        """Stay open after the command ends: anything failed, or it did not finish cleanly."""
        return self.state in ("failed", "interrupted")

    # Reading ---------------------------------------------------------------------

    def current_failures(self):
        """This attempt's failures that have not been sent for a retry."""
        with self._lock:
            return [item for item in self.failures
                    if item["attempt"] == self.attempt and item.get("state", "failed") == "failed"]

    def retry(self, item):
        """Render one failed chapter again in the running render; False when it no longer can."""
        callback = item.get("retry")
        if not callable(callback) or item.get("state", "failed") != "failed" or not self.running:
            return False
        if not callback():
            return False
        with self._lock:
            item["state"] = "retrying"
        self.changed.set()
        return True

    def release_holds(self):
        """Finish every render that waits for retries; it then reports what still failed."""
        with self._lock:
            releases, self.holds = list(self.holds.values()), {}
        for release in releases:
            release()
        self.changed.set()
        return len(releases)

    def visible_failures(self):
        with self._lock:
            return [item for item in self.failures
                    if self.filter.matches(item["scope"], text=item["error"])]

    def visible_logs(self, after=0):
        """``[(sequence, event)]`` passing the filter, newer than ``after``."""
        with self._lock:
            return [(sequence, event) for sequence, event in self.logs
                    if sequence > after
                    and self.filter.matches(event.scope, severity=event.severity, text=event.text)]

    def visible_rows(self):
        """``[(key, scope, snapshot, text)]`` for live rows, ordered by book then chapter."""
        with self._lock:
            items = [(key, row["scope"], row["snapshot"], row["text"]) for key, row in self.rows.items()]
        flt = self.filter
        items = [item for item in items
                 if (flt.book is None or item[1].get("book") == flt.book)
                 and (flt.chapter is None or item[1].get("chapter") == flt.chapter)
                 and (flt.worker is None or any(task.get("worker") == flt.worker
                                                for task in (item[2] or {}).get("active", [])))
                 and (flt.voice is None or any(flt.voice in (task.get("voices") or ())
                                               for task in (item[2] or {}).get("active", [])))]
        return sorted(items, key=lambda item: (str(item[1].get("book") or ""), item[1].get("chapter") or 0))

    def workers(self):
        """``{worker: [(book, chapter, block, phase, elapsed)]}`` from the live rows."""
        result = {}
        with self._lock:
            rows = list(self.rows.values())
        for row in rows:
            for task in (row["snapshot"] or {}).get("active", []):
                name = task.get("worker") or "local"
                result.setdefault(name, []).append((row["scope"].get("book"), row["scope"].get("chapter"),
                                                    task.get("block"), task.get("phase"), task.get("elapsed", 0)))
        return result

    def choices(self, dimension):
        with self._lock:
            return sorted(self.seen[dimension], key=lambda value: (str(type(value)), value))

    def summary(self, now=None):
        now = self._clock() if now is None else now
        with self._lock:
            end = self.finished_at if self.finished_at is not None else now
            parts = [_duration(end - self.started)]
            remaining = [row["snapshot"].get("remaining") for row in self.rows.values()
                         if row["snapshot"] and row["snapshot"].get("remaining")]
            if remaining and self.running:
                parts.append(f"~{_duration(max(remaining))} left in active chapters")
            progress = self.progress
            if progress and self.running:
                parts.append(f"step {progress['step']}/{progress['steps']} {progress['step_name']} "
                             f"{progress['step_fraction']:.0%}")
                if progress.get("streams"):
                    parts.append(f"{progress['streams']} model request(s) active")
            if self.rows:
                parts.append(f"{len(self.rows)} chapter(s) active")
            done = sum(self.completed.values())
            if done:
                parts.append(f"{done} chapter(s) finished")
            if self.books:
                ok = sum(self.books.values())
                parts.append(f"books {ok} ok" + (f", {len(self.books) - ok} failed" if len(self.books) > ok else ""))
            failures = len(self.current_failures())
            if failures:
                parts.append(f"{failures} failure(s)")
            if self.holds and self.running:
                parts.append("waiting for retries (r retries a chapter, F finishes)")
            parts.append({"running": "stopping…" if self.stopping else "running", "finished": "finished",
                          "failed": "finished with errors", "interrupted": "interrupted"}[self.state])
            if self.attempt > 1:
                parts.append(f"attempt {self.attempt}")
        return " · ".join(parts)

    def closing_lines(self, limit=12):
        """Lines to print on the restored terminal after the screen closes."""
        with self._lock:
            errors = [event for _, event in self.logs if event.severity == "error"]
            tail = [event for _, event in list(self.logs)[-limit:]]
        lines = []
        for event in errors[-limit:]:
            if event not in tail:
                lines.append(_line(event))
        lines.extend(_line(event) for event in tail)
        return lines


def _line(event):
    prefix = ""
    if event.scope.get("book"):
        prefix += f"[{event.scope['book']}] "
    if event.scope.get("chapter") is not None:
        prefix += f"Ch {event.scope['chapter']}: "
    return prefix + event.text


def _duration(seconds):
    seconds = max(0, int(seconds or 0))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
