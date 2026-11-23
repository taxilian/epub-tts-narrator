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
import itertools
from pathlib import Path
import re
import shlex
import threading
import time

from .run_events import (ADDER, BOOK, CHAPTER, CHAPTERS, FAILURE, FINISHED, HOLD, LOG, OUTCOME, PROGRESS, REBUILD,
                         REJECTED, SEVERITIES, STATUS, STATUS_REMOVED)


SEVERITY_LEVELS = ("info", "warning", "error", "debug")  # cycle order; "debug" shows everything
RATE_WINDOW = 300.0         # seconds of recent history behind "per minute" figures
STALL_SECONDS = 900.0       # running chapters with no accepted block for this long look stalled
_READY = re.compile(r"\((\d+) / (\d+) selected chapters ready\)")
_REUSED = re.compile(r"^Reusing completed chapter (\d+) / (\d+)")
_REJECTED = re.compile(r"\brejected\b", re.IGNORECASE)
_RETRYING = re.compile(r"\b(retrying|retry \d|attempt \d+/\d+)\b", re.IGNORECASE)
FINAL_BOOK_STATES = ("done", "failed", "stopped")
# A chapter's state in the Chapters view, in breakdown order. "not started" is a
# chapter of a book that is not rendering yet; "queued" waits inside a rendering
# book; "saved" has a checkpoint from an earlier run, checked when it is reached.
CHAPTER_STATES = ("rendering", "assembling", "retrying", "queued", "not started", "saved", "done", "failed",
                  "blocked", "stopped")
_UNFINISHED_CHAPTER = ("rendering", "assembling", "retrying")


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
    return "Fix what stopped it, then resume the book (Enter on it in Summary, or r in Errors); completed work is reused."


# What a failed attempt did, in plain words, by its rejection reason.
_REASON_WORDS = {
    "text_mismatch": "was heard saying different words", "audio_too_short": "was too short",
    "audio_too_long": "ran too long", "silent_audio": "was silent", "audio_too_quiet": "was too quiet",
    "excessive_silence": "had too much silence", "internal_silence_stall": "stalled in a long pause",
    "excessive_clipping": "was distorted (clipped)", "generation_incomplete": "never finished",
}


def _attempt_cause(row):
    error = (row.get("error") or "").casefold()
    if "did not finish the text" in error and "silence" in error:
        return "Drama 3 went silent partway through and stopped"
    if row.get("echo"):
        return "repeated an earlier line instead (history echo)"
    reasons = [reason for reason in row.get("reasons") or [] if reason != "generation_or_verification_error"]
    if reasons:
        return " and ".join(_REASON_WORDS.get(reason, reason.replace("_", " ")) for reason in reasons)
    if row.get("error"):
        return "stopped with an error and made no audio"
    return "was rejected"


def plain_failure(rows):
    """One plain sentence for what a failed passage's attempts did, e.g. 'Drama 3 went silent … on all 3 attempts'."""
    rows = [row for row in rows or [] if not row.get("passed")]
    if not rows:
        return ""
    causes = {}
    for row in rows:
        causes.setdefault(_attempt_cause(row), []).append(row)
    if len(causes) == 1:
        (cause, same), = causes.items()
        if cause.startswith("Drama 3 went silent"):
            return f"Drama 3 went silent partway through this passage and stopped, on {'all ' if len(same) > 1 else ''}{len(same)} attempt{'s' if len(same) > 1 else ''}; no audio was kept."
        which = f"all {len(same)} attempts" if len(same) > 1 else "its one attempt"
        return f"No take passed: {which} {cause}."
    parts = [f"{len(same)} {cause}" for cause, same in sorted(causes.items(), key=lambda pair: -len(pair[1]))]
    return f"No take passed in {len(rows)} attempts: " + "; ".join(parts) + "."


@dataclass
class RunFilter:
    book: str | None = None
    chapter: int | None = None
    worker: str | None = None
    voice: str | None = None
    severity: str = "info"
    text: str = ""
    state: str | None = None  # a chapter state; narrows the Chapters view only

    def active(self):
        return any((self.book, self.chapter, self.worker, self.voice, self.text, self.state)) or self.severity != "info"

    def describe(self):
        parts = [f"{name}={value}" for name, value in (("book", self.book), ("chapter", self.chapter),
                                                       ("worker", self.worker), ("voice", self.voice),
                                                       ("state", self.state))
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


def book_paths(text):
    """The book folders in what a person typed or dropped: one path (spaces and all) or several, shell-quoted."""
    text = (text or "").strip()
    if not text:
        return []
    whole = text.strip("'\"")
    if Path(whole).expanduser().exists():
        return [whole]
    try:
        return shlex.split(text)
    except ValueError:
        return [whole]


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
        self.failures = []        # {"id", "kind", "scope", "error", "record", "attempt"}
        self._failure_ids = itertools.count(1)  # a failure's stable id: the Errors table's row key
        self.attempt = 1
        self.state = "running"    # running | finished | failed | interrupted
        self.exit_code = None
        self.error = None
        self.finished_at = None
        self.filter = RunFilter()
        self.seen = {"book": set(), "chapter": set(), "worker": set(), "voice": set()}
        self.stopping = False
        self.paused = False       # a person paused the render's network work (``renderer.render_pause``)
        self.progress = None      # analysis stage and model-stream status (PROGRESS events)
        self.holds = {}           # book (or None) -> release callable while a render waits for retries
        self.hold_failed = {}     # book (or None) -> failures it waits on; 0: a queue waiting for more books
        self.add_book = None      # the running render queue's add(path) -> (accepted, message), else None
        self.dismissed = 0        # finished books a person cleared from the lists
        self.book_states = {}     # book -> {"state", "ready", "total", "key"}
        self.chapter_plans = {}   # book -> {number: {"number", "title", "blocks", "state", "done", ...}}
        self._outlines = set()    # books whose chapters are still a queued outline
        self._hidden = {}         # book -> chapters proven to have no planned speech work (blocks == 0)
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
        self._feedback = {}                    # render work -> {(request_hash, attempt): verdict}
        self._rejected_attempts = {}           # (render work, request_hash) -> rejected attempts with audio
        # book -> chapter numbers a person asked to retry when no render of that
        # book could take them yet; its next failed state (in a queue) re-admits
        # the book, and a run that ends resumes it.
        self.wanted_retries = {}
        self.notices = deque(maxlen=50)  # one-time messages for the person (a retried chapter failed again)
        self.changed = threading.Event()

    # Events --------------------------------------------------------------------

    def on_event(self, event):
        deferred = None
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
                    entry = self._chapter(book, row["scope"].get("chapter"), create=False)
                    if entry is not None:
                        entry["done"], entry["blocks"] = done, row["snapshot"].get("blocks", entry["blocks"])
                        if entry["state"] not in ("rendering", "assembling"):
                            entry["state"], entry["error"], entry["error_type"] = "rendering", None, None
                    for task in row["snapshot"].get("active", []):
                        if task.get("worker"):
                            self.seen["worker"].add(task["worker"])
                        self.seen["voice"].update(task.get("voices") or ())
                if event.text:
                    row["text"] = event.text
                row["updated"] = event.time
            elif event.kind == CHAPTERS:
                self._plan_chapters(event.scope.get("book"), (event.data or {}).get("chapters") or [],
                                    outline=bool((event.data or {}).get("outline")))
            elif event.kind == CHAPTER:
                data = event.data or {}
                entry = self._chapter(event.scope.get("book"), event.scope.get("chapter"))
                if entry is not None and data.get("state") in CHAPTER_STATES:
                    entry["state"] = data["state"]
                    entry["reused"] = bool(data.get("reused"))
                    if data["state"] == "rendering":
                        entry["error"] = entry["error_type"] = None
                    elif data["state"] == "done" and entry["blocks"] is not None:
                        entry["done"] = entry["blocks"]
                if data.get("state") == "done":
                    for item in self.failures:
                        # A chapter that rendered after a retry is no longer failing.
                        if (item["kind"] == "chapter" and item.get("state") == "retrying"
                                and item["scope"].get("book") == event.scope.get("book")
                                and item["scope"].get("chapter") == event.scope.get("chapter")):
                            item["state"] = "retried"
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
                    if not self._hidden_chapter(place[0], place[1]):
                        book = row["scope"].get("book")
                        self.completed[book] = self.completed.get(book, 0) + 1
            elif event.kind == OUTCOME:
                book = event.scope.get("book")
                self.books[book] = bool(event.data.get("succeeded"))
                self._book(book, key=event.data.get("key"))["state"] = "done" if self.books[book] else "failed"
                self._stop_chapters(book)
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
                if state in FINAL_BOOK_STATES:
                    self._stop_chapters(book)
                if state == "failed":
                    self._book_failure(book, event.data.get("key"), retry=event.data.get("retry"),
                                       error=event.data.get("error"))
                    if self.wanted_retries.get(book) and callable(event.data.get("retry")):
                        # A chapter retry asked for while the book could not take it: re-admit the book.
                        deferred = (book, event.data["retry"])
            elif event.kind == FAILURE:
                data = dict(event.data or {})
                retried = None
                for item in self.failures:
                    # The chapter failed again: its newest failure replaces the older ones.
                    if (item["kind"] == "chapter" and item["scope"].get("book") == event.scope.get("book")
                            and item["scope"].get("chapter") == event.scope.get("chapter")
                            and item.get("state", "failed") in ("failed", "retrying")):
                        if item.get("state") == "retrying":
                            retried = item
                        item["state"] = "superseded"
                after_retry = 0
                if retried is not None:
                    after_retry = retried.get("after_retry", 0) + 1
                    self.notices.append(self._failed_again(event, retried, after_retry))
                self.failures.append({"id": next(self._failure_ids), "after_retry": after_retry,
                                      "kind": "chapter", "scope": dict(event.scope), "error": event.text,
                                      "record": dict(data.get("record") or {}), "retry": data.get("retry"),
                                      "repair": data.get("repair"), "passage": data.get("passage"),
                                      "state": "failed", "attempt": self.attempt})
                entry = self._chapter(event.scope.get("book"), event.scope.get("chapter"), create=False)
                if entry is not None:
                    error_type = (data.get("record") or {}).get("error_type")
                    # No take passed verification ("failed"), or something must be fixed first ("blocked").
                    entry["state"] = "failed" if error_type in (None, "SpeechVerificationFailed") else "blocked"
                    entry["error"], entry["error_type"] = event.text, error_type
            elif event.kind == HOLD:
                book = event.scope.get("book")
                if event.data.get("release") is None:
                    self.holds.pop(book, None)
                    self.hold_failed.pop(book, None)
                else:
                    self.holds[book] = event.data["release"]
                    self.hold_failed[book] = event.data.get("failed") or 0
            elif event.kind == ADDER:
                add = (event.data or {}).get("add")
                self.add_book = add if callable(add) else None
            elif event.kind == PROGRESS:
                self.progress = dict(event.data or {})
            elif event.kind == FINISHED:
                self.finish(event.data.get("exit_code"), event.data.get("error"))
        if deferred is not None:
            book, retry = deferred
            if retry():
                with self._lock:
                    chapters = self.wanted_retries.pop(book, set())
                    for item in self.failures:
                        if item["scope"].get("book") == book and (
                                item["kind"] == "book" and item.get("state") == "failed"
                                or item["kind"] == "chapter" and item.get("wanted")):
                            item["state"], item["wanted"] = "retrying", False
                    for number in chapters:
                        entry = self._chapter(book, number, create=False)
                        if entry is not None:
                            entry["state"] = "retrying"
        self.changed.set()

    def _book(self, book, *, key=None):
        state = self.book_states.setdefault(book, {"state": None, "ready": 0, "total": 0, "key": key})
        if key and not state["key"]:
            state["key"] = key
        return state

    def _plan_chapters(self, book, chapters, *, outline):
        """A book's chapter list: its render's plan replaces a queued outline, never the reverse."""
        if outline and book in self.chapter_plans and book not in self._outlines:
            return
        self._book(book)
        plan, hidden = {}, set()
        for item in chapters:
            number = item.get("number")
            if type(number) is not int:
                continue
            blocks = item.get("blocks") if type(item.get("blocks")) is int else None
            if blocks == 0:
                # A chapter with no planned speech work (unlike one holding only
                # silence controls, which has blocks) renders nothing; it stays
                # out of every list and count, like the M4B it never joins.
                hidden.add(number)
                continue
            plan[number] = {"number": number, "title": str(item.get("title") or ""), "blocks": blocks,
                            "state": "saved" if item.get("saved") else "queued", "done": 0, "reused": False,
                            "error": None, "error_type": None}
        self.chapter_plans[book] = dict(sorted(plan.items()))
        self._hidden[book] = hidden
        if outline:
            self._outlines.add(book)
        else:
            self._outlines.discard(book)

    def _hidden_chapter(self, book, number):
        """Whether a chapter is proven to have no planned speech work and stays hidden."""
        return number in self._hidden.get(book, ())

    def _chapter(self, book, number, *, create=True):
        """A planned chapter's entry; a state for an unplanned chapter of a planned book adds it."""
        plan = self.chapter_plans.get(book)
        if plan is None or type(number) is not int or self._hidden_chapter(book, number):
            return None
        if number not in plan and create:
            plan[number] = {"number": number, "title": "", "blocks": None, "state": "queued", "done": 0,
                            "reused": False, "error": None, "error_type": None}
            self.chapter_plans[book] = dict(sorted(plan.items()))
        return plan.get(number)

    def _stop_chapters(self, book=None):
        """Chapters still in progress when their book (or the run) ended were stopped, not finished."""
        for name, plan in self.chapter_plans.items():
            if book is None or name == book:
                for entry in plan.values():
                    if entry["state"] in _UNFINISHED_CHAPTER:
                        entry["state"] = "stopped"

    def _book_failure(self, book, key, retry=None, error=None):
        """One book-level failure per book and attempt, carrying its error and any retry.

        The error is the one the failed state reported, else the book's last
        error line. A book that fails again after a retry updates the same
        entry, which becomes a failure again with its newest error.
        """
        last = error or next((item.text for _, item in reversed(self.logs)
                              if item.severity == "error" and item.scope.get("book") in (book, None)), "")
        existing = next((item for item in self.failures if item["kind"] == "book"
                         and item["scope"].get("book") == book and item["attempt"] == self.attempt), None)
        if existing is not None:
            if retry is not None or error or existing.get("state") == "retrying":
                existing.update(state="failed", error=last or existing["error"],
                                retry=retry if retry is not None else existing.get("retry"))
            return
        self.failures.append({"id": next(self._failure_ids), "kind": "book", "scope": {"book": book}, "error": last,
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
            self.add_book = None
            self._stop_chapters()
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
            self.hold_failed.clear()
            self.add_book = None
            self.wanted_retries.clear()
            self.rebuild_handlers.clear()
            self.pending_rebuilds.clear()
            self.book_states = {book: dict(state, state="queued") for book, state in self.book_states.items()
                                if state["state"] != "done"}
            for book, plan in self.chapter_plans.items():
                if book in self.book_states:
                    for entry in plan.values():
                        if entry["state"] != "done":
                            entry.update(state="queued", error=None, error_type=None)
            self._done_by_key.clear()
            self.last_progress = self._clock()
        self.changed.set()

    @property
    def running(self):
        return self.state == "running"

    def should_hold(self):
        """Stay open after the command ends: anything failed, it did not finish cleanly, or a
        rejected take is still waiting for a person to judge it."""
        return self.state in ("failed", "interrupted") or bool(self.unjudged())

    # A person's judgement of rejected takes ------------------------------------------

    def _verdicts(self, review):
        work = str(review.work)
        if work not in self._feedback:
            try:
                self._feedback[work] = review.feedback()
            except (AttributeError, ValueError, OSError):
                self._feedback[work] = None  # Unreadable: shown as unknown, never holds the screen.
        return self._feedback[work]

    def take_verdict(self, item):
        """A rejected take's verdict: ``"good"``, ``"bad"``, ``"truncated"``, ``None`` (not judged) or ``"?"`` (unknown)."""
        review = item.get("review")
        verdicts = self._verdicts(review) if hasattr(review, "work") else None
        if verdicts is None:
            return "?"
        return verdicts.get((item["request_hash"], item["attempt"]))

    def failure_takes(self, item):
        """``(rejected, judged)`` takes of a failed passage, or None when it has no takes to review."""
        review = item.get("repair")
        if review is None or not hasattr(review, "work"):
            return None
        key = (str(review.work), review.request_hash)
        if key not in self._rejected_attempts:
            try:
                self._rejected_attempts[key] = list(review.rejected_attempts())
            except (AttributeError, ValueError, OSError):
                self._rejected_attempts[key] = None
        attempts, verdicts = self._rejected_attempts[key], self._verdicts(review)
        if attempts is None or verdicts is None:
            return None
        return len(attempts), sum(1 for attempt in attempts if verdicts.get((review.request_hash, attempt)))

    def unjudged(self):
        """Every rejected take (listed this run, or of a current failed passage) not judged yet."""
        with self._lock:
            rejections, failures = list(self.rejections), self.current_failures()
        waiting = set()
        for item in rejections:
            if hasattr(item.get("review"), "work") and self.take_verdict(item) is None:
                waiting.add((item["request_hash"], item["attempt"]))
        for item in failures:
            review = item.get("repair")
            if self.failure_takes(item) is None:
                continue
            verdicts = self._verdicts(review)
            for attempt in self._rejected_attempts[(str(review.work), review.request_hash)]:
                if not verdicts.get((review.request_hash, attempt)):
                    waiting.add((review.request_hash, attempt))
        return waiting

    def judgements_changed(self):
        """A person judged or used takes: read the saved verdicts and takes again."""
        self._feedback.clear()
        self._rejected_attempts.clear()
        self.changed.set()

    # Reading ---------------------------------------------------------------------

    def current_failures(self):
        """This attempt's failures that have not been sent for a retry."""
        with self._lock:
            return [item for item in self.failures
                    if item["attempt"] == self.attempt and item.get("state", "failed") == "failed"]

    def failure_by_id(self, failure_id):
        """The failure with this id (the Errors table's row key), or None."""
        with self._lock:
            return next((item for item in self.failures if item.get("id") == failure_id), None)

    def chapter_name(self, book, number):
        """How a notice names a chapter: its number, the same one every view shows, and its title when known."""
        with self._lock:
            title = ((self.chapter_plans.get(book) or {}).get(number) or {}).get("title") or ""
        name = f"chapter {number}" if number is not None else "this chapter"
        return f"{name} ({title})" if title and title.casefold() != name.casefold() else name

    def current_failure(self, item):
        """The newest failure of the same chapter (an older one was replaced when the chapter failed again)."""
        if item["kind"] != "chapter" or item.get("state") != "superseded":
            return item
        return self.chapter_failure(item["scope"].get("book"), item["scope"].get("chapter")) or item

    def retry(self, item):
        """Render a failed chapter or book again while the run is alive; how, or False once the run has ended.

        ``"chapter"``: its book's render retries it now. ``"book"``: its book's
        render had ended, so the book goes back in the queue (finished chapters
        are restored, the failed one renders with fresh takes). ``"later"``: no
        render can take it yet; the book is re-admitted when its render ends
        in a queue, and resumed when the run ends. False when the run is over
        (resume it) or the chapter is already being rendered again.
        """
        item = self.current_failure(item)
        if not self.running or item.get("state", "failed") not in ("failed",):
            return False
        book = item["scope"].get("book")
        route = None
        callback = item.get("retry")
        if callable(callback) and callback():
            route = item["kind"]
        elif item["kind"] == "chapter":
            failure, _ = self.book_failure(book)
            if (failure is not None and failure.get("state", "failed") == "failed"
                    and callable(failure.get("retry")) and failure["retry"]()):
                route = "book"
                with self._lock:
                    failure["state"] = "retrying"
            else:
                route = "later"
                with self._lock:
                    self.wanted_retries.setdefault(book, set()).add(item["scope"].get("chapter"))
        if route is None:
            return False
        with self._lock:
            item["state"] = "retrying" if route != "later" else "failed"
            item["wanted"] = route == "later"
            # Takes the retry makes are counted from here (``_failed_again``).
            item["rejections_before"] = len(self.rejections)
            entry = self._chapter(book, item["scope"].get("chapter"), create=False)
            if item["kind"] == "chapter" and entry is not None and route != "later":
                entry["state"] = "retrying"
        self.changed.set()
        return route

    def _failed_again(self, event, retried, count):
        """The notice for a chapter whose retry failed again: what the retry did, in words."""
        book, chapter = event.scope.get("book"), event.scope.get("chapter")
        made = sum(1 for item in list(self.rejections)[retried.get("rejections_before", len(self.rejections)):]
                   if item["scope"].get("book") == book and item["scope"].get("chapter") == chapter)
        name = self.chapter_name(book, chapter)
        what = (f"its {made} new take{'s were' if made != 1 else ' was'} rejected" if made
                else "it made no new take that was rejected")
        reason = (event.text or "").split(";")[0].strip()
        return (f"{name[0].upper()}{name[1:]}" + (f" of {book}" if book else "") + " failed again after "
                + ("your retry" if count == 1 else f"retry {count}") + f": {what}. {reason}"
                + (" Errors (5) has the passage, its takes and other ways to fix it." if reason else ""))

    def pop_notices(self):
        """One-time messages for the person, oldest first (each is returned once)."""
        with self._lock:
            items, self.notices = list(self.notices), deque(maxlen=50)
        return items

    def wanted_books(self):
        """Books with chapter retries still waiting for a render to take them (resumed when the run ends)."""
        with self._lock:
            return [book for book, chapters in self.wanted_retries.items() if chapters]

    def book_failure(self, book):
        """The book's newest book-level failure this attempt (failed or sent for a retry) and its failed chapters.

        ``(failure or None, [chapter failures])``; the chapters are this
        attempt's failures not yet sent for a retry.
        """
        with self._lock:
            mine = [item for item in self.failures if item["attempt"] == self.attempt
                    and item["scope"].get("book") == book]
            failure = next((item for item in reversed(mine) if item["kind"] == "book"
                            and item.get("state", "failed") in ("failed", "retrying")), None)
            chapters = [item for item in mine if item["kind"] == "chapter" and item.get("state", "failed") == "failed"]
        return failure, chapters

    def chapter_failure(self, book, chapter):
        """This attempt's newest failure of one chapter (failed or being retried), or None."""
        with self._lock:
            return next((item for item in reversed(self.failures)
                         if item["kind"] == "chapter" and item["attempt"] == self.attempt
                         and item["scope"].get("book") == book and item["scope"].get("chapter") == chapter
                         and item.get("state", "failed") in ("failed", "retrying")), None)

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

    def idle(self):
        """The render queue has finished every book and waits for more (``a`` adds one, ``q`` closes)."""
        with self._lock:
            return (self.running and bool(self.holds) and not any(self.hold_failed.values())
                    and not self.rows and not self.current_failures())

    def add_books(self, paths):
        """Hand each path to the running render queue: ``[(path, accepted, message)]``."""
        add = self.add_book
        results = []
        for path in paths:
            if not callable(add):
                results.append((path, False, "This render takes no more books now"))
                continue
            try:
                accepted, message = add(path)
            except Exception as error:  # noqa: BLE001 - a person's typo must never stop the render
                accepted, message = False, f"{type(error).__name__}: {error}"
            results.append((path, bool(accepted), str(message)))
        self.changed.set()
        return results

    def dismiss_done(self):
        """Clear every finished book from the lists (its audiobook stays); returns their names.

        Its chapters, rejected takes and past failures leave the screen with
        it; the log keeps its messages. Adding the book again lists it anew.
        """
        with self._lock:
            done = [book for book, state in self.book_states.items()
                    if book is not None and state["state"] == "done"]
            gone = set(done)
            for book in done:
                for table in (self.book_states, self.chapter_plans, self._hidden, self.books, self.completed,
                              self.wanted_retries, self.rebuild_handlers):
                    table.pop(book, None)
                self._outlines.discard(book)
                self.seen["book"].discard(book)
            self.failures = [item for item in self.failures if item["scope"].get("book") not in gone]
            self.rejections = deque((item for item in self.rejections if item["scope"].get("book") not in gone),
                                    maxlen=self.rejections.maxlen)
            self.pending_rebuilds = {place for place in self.pending_rebuilds if place[0] not in gone}
            if self.filter.book in gone:
                self.filter.book = None
            self.dismissed += len(done)
        self.changed.set()
        return done

    def release_holds(self):
        """Finish every render that waits for retries; it then reports what still failed."""
        with self._lock:
            releases, self.holds, self.hold_failed = list(self.holds.values()), {}, {}
        for release in releases:
            release()
        self.changed.set()
        return len(releases)

    def visible_failures(self):
        """Every failure passing the filter; a chapter's older failures it has since replaced are not listed."""
        with self._lock:
            return [item for item in self.failures if item.get("state") != "superseded"
                    and self.filter.matches(item["scope"], text=item["error"])]

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
                 if not self._hidden_chapter(item[1].get("book"), item[1].get("chapter"))
                 and (flt.book is None or item[1].get("book") == flt.book)
                 and (flt.chapter is None or item[1].get("chapter") == flt.chapter)
                 and (flt.worker is None or any(task.get("worker") == flt.worker
                                                for task in (item[2] or {}).get("active", [])))
                 and (flt.voice is None or any(flt.voice in (task.get("voices") or ())
                                               for task in (item[2] or {}).get("active", [])))]
        return sorted(items, key=lambda item: (str(item[1].get("book") or ""), item[1].get("chapter") or 0))

    def _book_order(self):
        order = list(self.book_states)
        order += [book for book in self.chapter_plans if book not in self.book_states]
        order += sorted({row["scope"].get("book") for row in self.rows.values()} - set(order), key=str)
        return order

    def _chapters(self):
        """Every chapter row, in book then chapter order, before filters (call holding the lock).

        Planned chapters come first; a live row the plan does not know (a
        command that reports no plan) is listed under its book as well.
        """
        live = {}
        unplanned = []
        for key, row in self.rows.items():
            place = (row["scope"].get("book"), row["scope"].get("chapter"))
            if self._hidden_chapter(place[0], place[1]):
                continue
            if place[0] in self.chapter_plans and place[1] in self.chapter_plans[place[0]]:
                live[place] = (key, row)
            else:
                unplanned.append((key, row))
        rejected = {}
        for item in self.rejections:
            place = (item["scope"].get("book"), item["scope"].get("chapter"))
            rejected[place] = rejected.get(place, 0) + 1
        result = []
        for book in self._book_order():
            book_state = (self.book_states.get(book) or {}).get("state")
            for number, entry in (self.chapter_plans.get(book) or {}).items():
                key, row = live.get((book, number), (None, None))
                snapshot = row["snapshot"] if row else None
                state = entry["state"]
                if row is not None and state not in ("assembling", "done"):
                    state = "rendering"
                elif state == "queued" and book_state not in ("rendering", "assembling"):
                    state = "not started"
                result.append((f"{book}\x1f{number}", {
                    "book": book, "chapter": number, "title": entry["title"], "state": state,
                    "reused": entry["reused"], "blocks": snapshot["blocks"] if snapshot else entry["blocks"],
                    "done": snapshot["done"] if snapshot else entry["done"],
                    "rejected": rejected.get((book, number), 0), "error": entry["error"],
                    "error_type": entry["error_type"], "snapshot": snapshot, "text": row["text"] if row else "",
                    "row_key": key}))
            for key, row in sorted((item for item in unplanned if item[1]["scope"].get("book") == book),
                                   key=lambda item: item[1]["scope"].get("chapter") or 0):
                snapshot = row["snapshot"]
                number = row["scope"].get("chapter")
                result.append((f"live\x1f{key}", {
                    "book": book, "chapter": number, "title": "", "state": "rendering", "reused": False,
                    "blocks": snapshot["blocks"] if snapshot else None, "done": snapshot["done"] if snapshot else 0,
                    "rejected": rejected.get((book, number), 0), "error": None, "error_type": None,
                    "snapshot": snapshot, "text": row["text"], "row_key": key}))
        return result

    def visible_chapters(self):
        """``[(key, chapter)]`` for the Chapters view: every planned chapter of the shown books.

        Keys are strings, stable while the chapter list stays the same, so
        the table updates in place. The book, chapter and state filters
        apply; a worker or voice filter keeps chapters with matching active work.
        """
        with self._lock:
            items = self._chapters()
        flt = self.filter

        def tasks(chapter):
            return (chapter["snapshot"] or {}).get("active", [])
        return [(key, chapter) for key, chapter in items
                if (flt.book is None or chapter["book"] == flt.book)
                and (flt.chapter is None or chapter["chapter"] == flt.chapter)
                and (flt.state is None or chapter["state"] == flt.state)
                and (flt.worker is None or any(task.get("worker") == flt.worker for task in tasks(chapter)))
                and (flt.voice is None or any(flt.voice in (task.get("voices") or ()) for task in tasks(chapter)))]

    def chapter_breakdown(self):
        """``[(book, total, {state: count}, reused)]`` per shown book (the book filter applies, not the others).

        ``reused`` counts done chapters restored from an earlier run.
        """
        with self._lock:
            items = self._chapters()
        books = {}
        for _, chapter in items:
            if self.filter.book is not None and chapter["book"] != self.filter.book:
                continue
            tally = books.setdefault(chapter["book"], {"total": 0, "counts": {}, "reused": 0})
            tally["total"] += 1
            tally["counts"][chapter["state"]] = tally["counts"].get(chapter["state"], 0) + 1
            tally["reused"] += chapter["state"] == "done" and chapter["reused"]
        return [(book, tally["total"], tally["counts"], tally["reused"]) for book, tally in books.items()]

    def chapter_states(self):
        """The chapter states present now, in breakdown order, for the state filter."""
        with self._lock:
            present = {chapter["state"] for _, chapter in self._chapters()}
        return [state for state in CHAPTER_STATES if state in present]

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

    def set_paused(self, paused):
        """Record a pause or resume; the time paused never counts as a stall."""
        with self._lock:
            self.paused = bool(paused)
            if not paused:
                self.last_progress = self._clock()
        self.changed.set()

    def health(self, now=None):
        """``(level, text)``: ok, busy, warning or error, for the at-a-glance view."""
        now = self._clock() if now is None else now
        with self._lock:
            failures = len(self.current_failures())
            if self.state == "finished":
                waiting = len(self.unjudged())
                if waiting:
                    return "warning", f"Finished — {waiting} rejected take(s) not judged yet (Rejected, 6)"
                return "ok", "Finished"
            if self.state == "interrupted":
                return "warning", "Stopped — R resumes from where it stopped"
            if self.state == "failed":
                return "error", f"Ended with {failures or 'an'} error{'s' if failures != 1 else ''} — R resumes"
            if self.stopping:
                return "warning", "Stopping at safe boundaries"
            if self.paused:
                return "warning", "Paused — no new requests start; running ones finish (p resumes)"
            if self.holds and not any(self.hold_failed.values()) and not failures and not self.rows:
                return "ok", "Every book finished — a adds a book, q closes"
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
            active_rows = sum(1 for row in self.rows.values()
                              if not self._hidden_chapter(row["scope"].get("book"), row["scope"].get("chapter")))
            stages = (self.sched or {}).get("stages") or {}
        activity = self.activity()
        level, text = self.health(now)
        books = len(states)
        book_detail = ", ".join(f"{states.count(name)} {name}" for name in
                                ("rendering", "assembling", "preparing", "queued", "done", "stopped", "failed")
                                if states.count(name))
        if self.dismissed:
            book_detail = ", ".join(part for part in (book_detail, f"{self.dismissed} dismissed") if part)
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
                     f"{active_rows} chapter(s) active", "ok"),
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
        each active chapter's blocks. Chapters proven to have no planned
        speech work are hidden, so their book's totals come from its visible
        plan rather than the render's raw log totals; unknown until then.
        """
        with self._lock:
            rows = []
            for book, state in self.book_states.items():
                live = [row for row in self.rows.values()
                        if row["scope"].get("book") == book and row["snapshot"]
                        and not self._hidden_chapter(book, row["scope"].get("chapter"))]
                done = sum(row["snapshot"]["done"] for row in live)
                blocks = sum(row["snapshot"]["blocks"] for row in live)
                failed = sum(1 for item in self.current_failures() if item["scope"].get("book") == book)
                plan = self.chapter_plans.get(book)
                if plan is not None:
                    total = len(plan)
                    ready = sum(1 for entry in plan.values() if entry["state"] == "done")
                else:
                    total, ready = state["total"], state["ready"]
                percent = None
                if state["state"] == "done":
                    percent = 100
                elif total:
                    if plan is not None:
                        # Every unfinished chapter keeps its accepted share, not only the ones
                        # rendering now: a failed, blocked or stopped chapter's saved takes and
                        # an assembling chapter's finished blocks never drop out of the book.
                        partial = sum(_chapter_share(entry) for entry in plan.values() if entry["state"] != "done")
                    else:
                        partial = sum(row["snapshot"]["done"] / max(1, row["snapshot"]["blocks"]) for row in live)
                    percent = min(99, int(100 * (ready + partial) / total))
                rows.append((book, percent, state["state"] or "starting", ready, total, len(live),
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
        """Per node: ``(node, rows, decoding, prefilling, waiting, requests, sent, speed)`` from its own report.

        ``sent`` is this render's requests in flight to the node; the node's
        own numbers count every client. ``None`` where a node does not say.
        ``speed`` is how fast this render's audio came from it lately
        (``node_activity.NodeActivity.speeds``), or ``None`` before any.
        """
        result = []
        for node in nodes or []:
            if node["role"] != "speech":
                continue
            work = node.get("work") or {}
            result.append((node["name"], work.get("rows"), work.get("decoding"), work.get("prefilling"),
                           work.get("waiting"), work.get("requests"), len(node["jobs"]), node.get("speed")))
        return result

    @staticmethod
    def overall_speed(nodes):
        """Every speech node's speed together: ``{"realtime", "fps", "audio_seconds", "span"}`` or ``None``."""
        speeds = [node.get("speed") for node in nodes or [] if node["role"] == "speech" and node.get("speed")]
        if not speeds:
            return None
        realtime = sum(speed["realtime"] for speed in speeds)
        return {"realtime": round(realtime, 2), "fps": round(sum(speed["fps"] for speed in speeds), 1),
                "audio_seconds": round(sum(speed["audio_seconds"] for speed in speeds), 1),
                "span": max(speed["span"] for speed in speeds)}

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
            active = sum(1 for row in self.rows.values()
                         if not self._hidden_chapter(row["scope"].get("book"), row["scope"].get("chapter")))
            if active:
                parts.append(f"{active} chapter(s) active")
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
                parts.append("waiting for more books (a adds, q closes)"
                             if not any(self.hold_failed.values()) and not failures
                             else "waiting for retries (r retries, F finishes)")
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


def _chapter_share(entry):
    """How much of an unfinished chapter is done: all of it while assembling, else its accepted blocks."""
    if entry["state"] == "assembling":
        return 1
    blocks = entry.get("blocks")
    return min(1, entry.get("done", 0) / blocks) if blocks else 0


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
