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
import re
import threading
import time

from .run_events import (BOOK, FAILURE, FINISHED, HOLD, LOG, OUTCOME, PROGRESS, REBUILD, REJECTED, SEVERITIES,
                         STATUS, STATUS_REMOVED)


SEVERITY_LEVELS = ("info", "warning", "error", "debug")  # cycle order; "debug" shows everything
RATE_WINDOW = 300.0         # seconds of recent history behind "per minute" figures
STALL_SECONDS = 900.0       # running chapters with no accepted block for this long look stalled
_READY = re.compile(r"\((\d+) / (\d+) selected chapters ready\)")
_REUSED = re.compile(r"^Reusing completed chapter (\d+) / (\d+)")
_REJECTED = re.compile(r"\brejected\b", re.IGNORECASE)
_RETRYING = re.compile(r"\b(retrying|retry \d|attempt \d+/\d+)\b", re.IGNORECASE)
FINAL_BOOK_STATES = ("done", "failed", "stopped")


def repair_hint(item):
    """One plain-language suggestion for a failure, from its saved type and message."""
    text = f"{item['record'].get('error_type') or ''} {item.get('error') or ''}".casefold()
    if "history echo" in text or "history_echo" in text:
        return "A take repeated an earlier line. Retrying usually succeeds with another layout; or listen and accept a good take."
    if "verificationunavailable" in text or "unavailable" in text:
        return "The verifier or its model service was unavailable. Check it, then resume."
    if re.search(r"\block(ed)?\b|\blease\b", text) or "another render" in text:
        return "Another render, assembly or cleanup holds this book's lock. Resume when it has finished."
    if "voice" in text and ("missing" in text or "no active" in text):
        return "Choose the missing voices with cast for this book, then resume."
    if "pronunciation" in text or "untagged" in text or "unsure" in text:
        return "Confirm the listed words with pronounce, then resume."
    if item.get("repair") is not None:
        return ("No take passed verification. Listen to the takes (t) and accept one that sounds right, "
                "or retry the chapter (r).")
    if item["kind"] == "chapter":
        return "Retry the chapter (r); the retry reuses every verified clip."
    return "Resume this book (R) after the run; completed work is reused."


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
        self.book_states = {}     # book -> {"state", "ready", "total", "key"}
        self.sched = None         # latest render_schedule_status snapshot (polled by the screen)
        self._done_by_key = {}    # board key -> blocks done last seen
        self.accepted = deque()   # (time, blocks) accepted while chapters ran
        self.accepted_total = 0
        self.rejected = deque()   # times of rejected takes
        self.rejected_total = 0
        self.last_progress = clock()
        self.rejections = deque(maxlen=5000)   # rejected takes, newest last
        self._rejected_keys = set()
        self.rebuild_handlers = {}             # book -> handler(chapter number) while it renders
        self.pending_rebuilds = set()          # (book, chapter) to rebuild once its chapter finishes
        self.changed = threading.Event()

    # Events --------------------------------------------------------------------

    def on_event(self, event):
        with self._lock:
            if ("book" not in event.scope and len(self.book_states) == 1
                    and event.kind in (LOG, STATUS, FAILURE, REJECTED, REBUILD)):
                # A single-book render's lines carry no book prefix; they are that book's.
                event.scope["book"] = next(iter(self.book_states))
            self._remember(event.scope)
            if event.kind == LOG:
                self.log_total += 1
                self.logs.append((self.log_total, event))
                self._count_log(event)
            elif event.kind == STATUS:
                key = event.data.get("key")
                row = self.rows.setdefault(key, {"scope": {}, "snapshot": None, "text": ""})
                row["scope"] = {**row["scope"], **event.scope}
                if event.data.get("snapshot") is not None:
                    row["snapshot"] = event.data["snapshot"]
                    done = row["snapshot"].get("done", 0)
                    gained = done - self._done_by_key.get(key, done)
                    self._done_by_key[key] = done
                    if gained > 0:
                        self.accepted.append((event.time, gained))
                        self.accepted_total += gained
                        self.last_progress = event.time
                    book = row["scope"].get("book")
                    state = self._book(book)
                    state["total"] = row["snapshot"].get("total_chapters") or state["total"]
                    if state["state"] in ("queued", "preparing", None):
                        state["state"] = "rendering"
                    for task in row["snapshot"].get("active", []):
                        if task.get("worker"):
                            self.seen["worker"].add(task["worker"])
                        self.seen["voice"].update(task.get("voices") or ())
                if event.text:
                    row["text"] = event.text
                row["updated"] = event.time
            elif event.kind == REJECTED:
                data = dict(event.data or {})
                key = (data.get("request_hash"), data.get("attempt"))
                if key not in self._rejected_keys:
                    self._rejected_keys.add(key)
                    self.rejections.append({"scope": dict(event.scope), "time": event.time,
                                            "request_hash": data.get("request_hash"), "attempt": data.get("attempt"),
                                            "reasons": data.get("reasons") or [], "detail": data.get("detail") or "",
                                            "review": data.get("review")})
            elif event.kind == REBUILD:
                self.rebuild_handlers[event.scope.get("book")] = event.data.get("handler")
            elif event.kind == STATUS_REMOVED:
                self._done_by_key.pop(event.data.get("key"), None)
                row = self.rows.pop(event.data.get("key"), None)
                if row is not None:
                    place = (row["scope"].get("book"), row["scope"].get("chapter"))
                    if place in self.pending_rebuilds:
                        # The chapter that was rendering when its take changed has finished.
                        self.pending_rebuilds.discard(place)
                        handler = self.rebuild_handlers.get(place[0])
                        if callable(handler):
                            handler(place[1])
                if row is not None:
                    book = row["scope"].get("book")
                    self.completed[book] = self.completed.get(book, 0) + 1
            elif event.kind == OUTCOME:
                book = event.scope.get("book")
                self.books[book] = bool(event.data.get("succeeded"))
                self._book(book, key=event.data.get("key"))["state"] = "done" if self.books[book] else "failed"
                if not self.books[book]:
                    self._book_failure(book, event.data.get("key"))
                else:
                    for item in self.failures:
                        # A retried book that now rendered is no longer a failure.
                        if item["kind"] == "book" and item["scope"].get("book") == book and item.get("state") == "retrying":
                            item["state"] = "retried"
            elif event.kind == BOOK:
                book = event.scope.get("book")
                state = event.data.get("state")
                self._book(book, key=event.data.get("key"))["state"] = state
                if state == "failed":
                    self._book_failure(book, event.data.get("key"), retry=event.data.get("retry"))
            elif event.kind == FAILURE:
                data = dict(event.data or {})
                self.failures.append({"kind": "chapter", "scope": dict(event.scope), "error": event.text,
                                      "record": dict(data.get("record") or {}), "retry": data.get("retry"),
                                      "repair": data.get("repair"), "state": "failed", "attempt": self.attempt})
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

    def _book(self, book, *, key=None):
        state = self.book_states.setdefault(book, {"state": None, "ready": 0, "total": 0, "key": key})
        if key and not state["key"]:
            state["key"] = key
        return state

    def _book_failure(self, book, key, retry=None):
        """One book-level failure per book and attempt, carrying its last error and any retry.

        A book that fails again after a retry updates the same entry, which
        becomes a failure again with its newest error.
        """
        last = next((item.text for _, item in reversed(self.logs)
                     if item.severity == "error" and item.scope.get("book") in (book, None)), "")
        existing = next((item for item in self.failures if item["kind"] == "book"
                         and item["scope"].get("book") == book and item["attempt"] == self.attempt), None)
        if existing is not None:
            if retry is not None or existing.get("state") == "retrying":
                existing.update(state="failed", error=last or existing["error"],
                                retry=retry if retry is not None else existing.get("retry"))
            return
        self.failures.append({"kind": "book", "scope": {"book": book}, "error": last,
                              "record": {"book": key}, "retry": retry, "state": "failed", "attempt": self.attempt})

    def _count_log(self, event):
        match = _READY.search(event.text) or _REUSED.match(event.text)
        if match:
            state = self._book(event.scope.get("book"))
            ready, total = int(match.group(1)), int(match.group(2))
            if _READY.search(event.text):
                state["ready"] = max(state["ready"], ready)
            else:
                state["ready"] += 1
            state["total"] = total
            self.last_progress = event.time
        if _REJECTED.search(event.text):
            self.rejected.append(event.time)
            self.rejected_total += 1

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
                # Whatever had not finished was stopped by the person, not failed.
                for state in self.book_states.values():
                    if state["state"] not in ("done", "failed"):
                        state["state"] = "stopped"
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
            self.rebuild_handlers.clear()
            self.pending_rebuilds.clear()
            self.book_states = {book: dict(state, state="queued") for book, state in self.book_states.items()
                                if state["state"] != "done"}
            self._done_by_key.clear()
            self.last_progress = self._clock()
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

    def request_rebuild(self, book, chapter):
        """Rebuild one chapter with a newly chosen take: ``"now"``, ``"after"`` it finishes, or ``None``."""
        with self._lock:
            handler = self.rebuild_handlers.get(book)
            if not callable(handler) or not self.running or chapter is None:
                return None
            if any(row["scope"].get("book") == book and row["scope"].get("chapter") == chapter
                   for row in self.rows.values()):
                self.pending_rebuilds.add((book, chapter))
                return "after"
        return "now" if handler(chapter) else None

    def visible_rejections(self):
        """Rejected takes passing the book/chapter/text filters, newest first."""
        with self._lock:
            items = list(self.rejections)
        flt = self.filter
        return [item for item in reversed(items)
                if flt.matches(item["scope"], text=" ".join(item["reasons"]) + " " + item["detail"])]

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

    def _recent(self, now):
        cutoff = now - RATE_WINDOW
        while self.accepted and self.accepted[0][0] < cutoff:
            self.accepted.popleft()
        while self.rejected and self.rejected[0] < cutoff:
            self.rejected.popleft()

    def activity(self):
        """Blocks by what they are doing now, from every live chapter row."""
        counts = {"generating": 0, "verifying": 0, "waiting": 0, "other": 0}
        with self._lock:
            tasks = [task for row in self.rows.values() for task in (row["snapshot"] or {}).get("active", [])]
        for task in tasks:
            phase = (task.get("phase") or "").casefold()
            if task.get("worker_stage") == "waiting" or phase.startswith("waiting"):
                counts["waiting"] += 1
            elif task.get("worker_stage") == "generating" or phase.startswith(("gen", "prepare")):
                counts["generating"] += 1
            elif phase.startswith(("verify", "normalize", "cache")):
                counts["verifying"] += 1
            else:
                counts["other"] += 1
        return counts

    def health(self, now=None):
        """``(level, text)``: ok, busy, warning or error, for the at-a-glance view."""
        now = self._clock() if now is None else now
        with self._lock:
            failures = len(self.current_failures())
            if self.state == "finished":
                return "ok", "Finished"
            if self.state == "interrupted":
                return "warning", "Stopped — R resumes from where it stopped"
            if self.state == "failed":
                return "error", f"Ended with {failures or 'an'} error{'s' if failures != 1 else ''} — R resumes"
            if self.stopping:
                return "warning", "Stopping at safe boundaries"
            if self.holds:
                return "error", f"Waiting for retries: {failures} failure(s)"
            if self.rows and now - self.last_progress > STALL_SECONDS:
                return "warning", f"No block accepted for {_duration(now - self.last_progress)}"
            if failures:
                return "warning", f"Running with {failures} failure(s)"
            return "ok", "Running normally"

    def glance(self, now=None):
        """The at-a-glance figures, as ``{name: (value, detail, level)}``."""
        now = self._clock() if now is None else now
        with self._lock:
            self._recent(now)
            states = [state["state"] for state in self.book_states.values()]
            per_minute = sum(blocks for _, blocks in self.accepted) * 60 / RATE_WINDOW
            rejected_recent = len(self.rejected)
            failures = len(self.current_failures())
            remaining = [row["snapshot"].get("remaining") for row in self.rows.values()
                         if row["snapshot"] and row["snapshot"].get("remaining")]
            stages = (self.sched or {}).get("stages") or {}
        activity = self.activity()
        level, text = self.health(now)
        books = len(states)
        book_detail = ", ".join(f"{states.count(name)} {name}" for name in
                                ("rendering", "assembling", "preparing", "queued", "done", "stopped", "failed")
                                if states.count(name))
        result = {
            "Health": (text, f"elapsed {_duration((self.finished_at or now) - self.started)}", level),
            "Books": (str(books) if books else "—", book_detail,
                      "error" if states.count("failed") else "ok"),
            "Generating": (str(activity["generating"]), f"{activity['waiting']} waiting for a worker", "busy"),
            "Verifying": (str(activity["verifying"]), "normalize, check and transcribe", "busy"),
            "Accepted": (str(self.accepted_total), f"{per_minute:.1f} blocks/min (last 5 min)", "ok"),
            "Rejected takes": (str(self.rejected_total), f"{rejected_recent} in the last 5 min",
                               "warning" if rejected_recent else "ok"),
            "Failures": (str(failures), "r retries · t listens to takes" if failures else "none",
                         "error" if failures else "ok"),
            "Left": (f"~{_duration(max(remaining))}" if remaining and self.running else "—",
                     f"{len(self.rows)} chapter(s) active", "ok"),
        }
        for stage, label in (("synthesis", "Generating"), ("recognition", "Verifying")):
            gauges = stages.get(stage)
            if gauges:
                value, detail, level = result[label]
                result[label] = (value, detail + f" · pipeline {gauges.get('active_now', 0)} active, "
                                 f"{gauges.get('queued_now', 0)} queued", level)
        return result

    def task_glance(self, now=None):
        """At-a-glance figures for a command that is not a render: ``{name: (value, detail, level)}``."""
        now = self._clock() if now is None else now
        with self._lock:
            progress = dict(self.progress or {}) if self.running else {}
            events = [event for _, event in self.logs]
            failures = len(self.current_failures())
        level, text = self.health(now)
        warnings = [event for event in events if event.severity == "warning"]
        errors = [event for event in events if event.severity == "error"]
        step = "—"
        detail = ""
        if progress.get("step_name"):
            step = f"{progress['step_name']} {progress.get('step_fraction', 0):.0%}"
            detail = f"step {progress.get('step')} of {progress.get('steps')}" + (
                f" · {progress['stage']}" if progress.get("stage") else "")
        requests = progress.get("streams") or 0
        received = progress.get("received")
        return {
            "Health": (text, "", level),
            "Step": (step, detail, "busy" if self.running else "ok"),
            "Model requests": (str(requests), (f"{received / 1024:.1f} KiB received" if received else
                                               "none in progress" if not requests else ""), "busy"),
            "Warnings": (str(len(warnings)), _short(warnings[-1].text) if warnings else "none",
                         "warning" if warnings else "ok"),
            "Errors": (str(len(errors) + failures), _short(errors[-1].text) if errors else "none",
                       "error" if errors or failures else "ok"),
            "Elapsed": (_duration((self.finished_at or now) - self.started), "", "ok"),
        }

    def book_rows(self):
        """``[(book, percent or None, state, ready, total, active chapters, blocks done/total, failures)]``.

        The percentage counts finished chapters plus the accepted share of
        each active chapter's blocks; unknown until a chapter total is seen.
        """
        with self._lock:
            rows = []
            for book, state in self.book_states.items():
                live = [row for row in self.rows.values() if row["scope"].get("book") == book and row["snapshot"]]
                done = sum(row["snapshot"]["done"] for row in live)
                blocks = sum(row["snapshot"]["blocks"] for row in live)
                failed = sum(1 for item in self.current_failures() if item["scope"].get("book") == book)
                percent = None
                if state["state"] == "done":
                    percent = 100
                elif state["total"]:
                    partial = sum(row["snapshot"]["done"] / max(1, row["snapshot"]["blocks"]) for row in live)
                    percent = min(99, int(100 * (state["ready"] + partial) / state["total"]))
                rows.append((book, percent, state["state"] or "starting", state["ready"], state["total"], len(live),
                             f"{done}/{blocks}" if live else "", failed))
        return rows

    def node_rows(self, nodes):
        """Per node and role, one row per slot: ``(node, role, slot, job or None)``; idle slots too."""
        rows = []
        for node in nodes or []:
            jobs = list(node["jobs"])
            for slot in range(max(node["slots"], len(jobs))):
                rows.append((node["name"], node["role"], slot + 1, jobs[slot] if slot < len(jobs) else None))
        return rows

    def node_work_rows(self, nodes):
        """Per node: ``(node, rows, decoding, prefilling, waiting, requests, sent)`` from its own report.

        ``sent`` is this render's requests in flight to the node; the node's
        own numbers count every client. ``None`` where a node does not say.
        """
        result = []
        for node in nodes or []:
            if node["role"] != "speech":
                continue
            work = node.get("work") or {}
            result.append((node["name"], work.get("rows"), work.get("decoding"), work.get("prefilling"),
                           work.get("waiting"), work.get("requests"), len(node["jobs"])))
        return result

    def node_summary(self, nodes):
        """``{node: {role: (busy, slots)}}`` for the at-a-glance line."""
        result = {}
        for node in nodes or []:
            result.setdefault(node["name"], {})[node["role"]] = (len(node["jobs"]), node["slots"])
        return result

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
                parts.append("waiting for retries (r retries, F finishes)")
            parts.append({"running": "stopping…" if self.stopping else "running", "finished": "finished",
                          "failed": "finished with errors", "interrupted": "stopped"}[self.state])
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


def _short(text, limit=60):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
