"""Display rendering progress without changing synthesis or cache identities."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import re
import shutil
import sys
import threading
import time

from .render_metrics import METRICS, STAGES, classify_outcome
from .render_schedule_status import scheduling_status


# Fixed stage abbreviations for the compact status line; stage names come
# only from the closed render_metrics.STAGES set. The chapter window's own
# bookkeeping gauges (chapter, chapter_clips, clip) are excluded: the line
# already reports this chapter's blocks, and the segment stays about the
# shared cross-chapter resources.
_DISPLAY_STAGES = STAGES - {"chapter", "chapter_clips", "clip"}

_STAGE_SHORT = {
    "admission": "admit", "synthesis_slot": "slot", "synthesis": "gen",
    "recognition": "asr", "review": "review", "preparation": "prep",
    "review_coordination": "revcoord", "model_thread": "model",
}


def _scheduling_segment():
    """One numeric ``sched`` segment for the status line, or empty.

    Read-only: observing never creates or starts anything. A broken or
    absent scheduling runtime leaves the status line unchanged, and the
    segment carries only fixed labels and numbers.
    """
    try:
        status = scheduling_status()
    except Exception:
        return ""
    if not status:
        return ""
    parts = []
    for stage, gauges in sorted((status.get("stages") or {}).items()):
        if stage not in _DISPLAY_STAGES:
            continue
        parts.append(f"{_STAGE_SHORT.get(stage, stage)} "
                     f"{gauges.get('queued_now', 0)}q/{gauges.get('active_now', 0)}a")
    logical = status.get("logical")
    if logical and ("inflight" in logical or "pending" in logical):
        parts.append(f"logical {logical.get('pending', 0)}/{logical.get('global_limit', 0)}"
                     f" active {logical.get('inflight', 0)} owners {logical.get('owners', 0)}")
    backlog = status.get("backlog")
    if backlog:
        parts.append(f"backlog {backlog.get('pending_count', 0)} pend"
                     f" (peak {backlog.get('peak_pending_count', 0)})")
        if "materialized_bytes" in backlog:
            mib = 1024 * 1024
            parts.append(f"audio {backlog['materialized_bytes'] / mib:.1f}"
                         f"/{backlog.get('max_materialized_bytes', 0) / mib:.1f} MiB"
                         f" (peak {backlog.get('peak_materialized_bytes', 0) / mib:.1f})")
        if backlog.get("waiters"):
            parts.append(f"wait {backlog['waiters']}")
        if backlog.get("new_generation_blocked"):
            parts.append("blocked")
    if not parts:
        return ""
    return "sched " + " ".join(parts)


class TerminalBoard:
    """Live status rows pinned to the bottom of a terminal; logs scroll above.

    One row per active chapter (registered by ``ChapterProgress``) plus a
    scheduler footer. ``log`` is print-compatible: it erases the rows, writes
    the message where they were, and redraws them below it, so history
    scrolls up while the bottom lines stay current. Rows are truncated to the
    terminal width so the erase count is exact. The board never calls back
    into its producers, so producers may hold their own locks while updating.
    """

    def __init__(self, stream, *, width=None):
        self._stream = stream
        self._width = width
        self._rows = {}
        self._book_outcomes = {}
        self._drawn = 0
        self._lock = threading.RLock()
        self._failed = False
        self._closed = False

    def log(self, *values, sep=" ", end="\n", file=None, flush=False):
        if file is not None and file is not self._stream:
            print(*values, sep=sep, end=end, file=file, flush=flush)
            return
        text = sep.join(str(value) for value in values)
        if not text.endswith("\n"):
            text += "\n"
        with self._lock:
            self._write(self._erase() + text + self._render())

    def set_row(self, key, order, text):
        with self._lock:
            if self._closed:
                return
            self._rows[key] = (order, text)
            self._write(self._erase() + self._render())

    def remove_row(self, key):
        with self._lock:
            if self._rows.pop(key, None) is not None:
                self._write(self._erase() + self._render())

    def book_outcome(self, key, succeeded):
        """Record one book's final outcome for the pinned books line.

        The queue reports each book's final status exactly once, keyed by an
        identity unique within the run, so re-reporting a key never double
        counts. Thread-safe like every other board mutation.
        """
        with self._lock:
            if self._closed:
                return
            self._book_outcomes[key] = bool(succeeded)
            self._write(self._erase() + self._render())

    def close(self):
        with self._lock:
            self._closed = True
            self._rows.clear()
            self._write(self._erase())

    def _columns(self):
        if self._width is not None:
            return self._width
        return shutil.get_terminal_size(fallback=(120, 24)).columns

    def _erase(self):
        if not self._drawn:
            return ""
        erase = "\r\033[2K" + "\033[1A\033[2K" * (self._drawn - 1)
        self._drawn = 0
        return erase

    def _render(self):
        if self._closed or (not self._rows and not self._book_outcomes):
            return ""
        width = max(10, self._columns() - 1)
        lines = [text for _, text in sorted(self._rows.values(), key=lambda row: row[0])]
        segment = _scheduling_segment()
        if segment:
            lines.append(segment)
        books = _books_line(self._book_outcomes, width)
        if books:
            lines.append(books)
        lines = [_fit(line, width) for line in lines]
        if books:
            # Color after fitting: _fit counts printable cells, and the
            # escapes add none.
            lines[-1] = _colorize_books(lines[-1])
        self._drawn = len(lines)
        return "\n".join(lines)

    def _write(self, text):
        if self._failed or not text:
            return
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):
            self._failed = True


def _fit(text, width):
    text = text.replace("\n", " ").replace("\r", " ")
    return text if len(text) <= width else text[:max(0, width - 1)] + "…"


_GREEN, _RED, _RESET = "\033[32m", "\033[31m", "\033[0m"


def _books_line(outcomes, width):
    """Plain books tally at the widest tier that fits ``width`` printable cells.

    Sufficient width keeps ``Books: N succeeded | N failed``; a narrower
    terminal switches to ``OK N | FAIL N`` and then to ``OKN FAILN``, so both
    nonzero counts keep their outcome labels at any width (a failure must
    never clip away). Zero counts are omitted, so the line appears once a
    book finishes and then keeps every finished book's outcome while chapter
    rows come and go. The returned text is always plain; only the tightest
    tier is ever handed to ``_fit`` for clipping.
    """
    succeeded = sum(1 for ok in outcomes.values() if ok)
    failed = len(outcomes) - succeeded
    if not succeeded and not failed:
        return ""
    tiers = (
        ("Books: ", " | ", "{} succeeded", "{} failed"),
        ("", " | ", "OK {}", "FAIL {}"),
        ("", " ", "OK{}", "FAIL{}"),
    )
    lines = []
    for prefix, separator, success_format, failure_format in tiers:
        parts = []
        if succeeded:
            parts.append(success_format.format(succeeded))
        if failed:
            parts.append(failure_format.format(failed))
        lines.append(prefix + separator.join(parts))
    for line in lines:
        if len(line) <= width:
            return line
    return lines[-1]


def _colorize_books(line):
    """Color the counts of an already fitted plain books line.

    Every tier's count token takes its color (green succeeded, red failed);
    the labels stay plain so the outcome reads without color, and every
    escape is closed so adjacent text never inherits it. A line clipped
    mid-part simply stays plain.
    """

    def compact(match):
        color = _GREEN if match.group(1).startswith("OK") else _RED
        return f"{match.group(1)}{color}{match.group(2)}{_RESET}"

    line = re.sub(r"(\d+) succeeded",
                  lambda match: f"{_GREEN}{match.group(1)}{_RESET} succeeded", line)
    line = re.sub(r"(\d+) failed",
                  lambda match: f"{_RED}{match.group(1)}{_RESET} failed", line)
    return re.sub(r"(OK ?|FAIL ?)(\d+)", compact, line)


_BOARD = None
_SHELL_BOARD = None


class BusBoard(TerminalBoard):
    """The board a full-screen run shell installs: every update becomes an event.

    It keeps the ``TerminalBoard`` interface (so ``ChapterProgress`` and the
    render queue treat it as the run's board) but draws nothing; the shell
    renders the events instead. ``set_status`` carries each chapter row's
    numbers, and ``chapter_failure`` each chapter-scoped failure record.
    """

    structured_status = True

    def __init__(self, bus):
        super().__init__(stream=None)
        self.bus = bus

    def log(self, *values, sep=" ", end="\n", file=None, flush=False):
        self.bus.log(sep.join(str(value) for value in values))

    def set_row(self, key, order, text):
        scope, body = _row_scope(order, text)
        self.bus.publish("status", body, scope=scope, data={"key": key})

    def set_status(self, key, order, snapshot, *, book=None):
        scope = {"chapter": snapshot.get("chapter")}
        if book:
            scope["book"] = book
        self.bus.publish("status", "", scope=scope, data={"key": key, "snapshot": snapshot})

    def remove_row(self, key):
        self.bus.publish("status_removed", data={"key": key})

    def book_outcome(self, key, succeeded):
        self.bus.publish("outcome", severity="info" if succeeded else "error",
                         scope={"book": _book_label(key)}, data={"key": key, "succeeded": bool(succeeded)})

    def chapter_failure(self, failure, *, book=None, retry=None, repair=None):
        scope = {"chapter": failure.get("number")}
        if book:
            scope["book"] = book
        self.bus.publish("failure", failure.get("error", ""), severity="error", scope=scope,
                         data={"record": dict(failure), "retry": retry, "repair": repair})

    def book_state(self, key, state):
        """One book's lifecycle: queued, preparing, rendering, assembling, done, failed, interrupted."""
        self.bus.publish("book", state, scope={"book": _book_label(key)}, data={"key": str(key), "state": state})

    def retry_hold(self, release, failed, *, book=None):
        """The run waits for retries (``release`` finishes it), or stopped waiting (``None``)."""
        self.bus.publish("hold", scope={"book": book} if book else {},
                         data={"release": release, "failed": failed})

    def close(self):
        self._closed = True


def _book_label(key):
    from pathlib import Path
    return Path(str(key)).name or str(key)


def _prefix_book(prefix):
    prefix = str(prefix or "").strip()
    return prefix[1:-1] if prefix.startswith("[") and prefix.endswith("]") else (prefix or None)


def _row_scope(order, text):
    scope = {}
    if isinstance(order, tuple) and order and isinstance(order[0], str):
        book = _prefix_book(order[0])
        if book:
            scope["book"] = book
            text = text[len(order[0]):] if text.startswith(order[0]) else text
    return scope, text


@contextmanager
def shell_board(bus):
    """Route this process's render boards to ``bus`` while a run shell is open."""
    global _SHELL_BOARD
    previous, board = _SHELL_BOARD, BusBoard(bus)
    _SHELL_BOARD = board
    try:
        yield board
    finally:
        _SHELL_BOARD = previous


@contextmanager
def live_board(notify, stream=None):
    """Use a terminal board for ``print`` output on an interactive terminal.

    Yields the notify callable to use: the board's print-compatible ``log``
    when output is a terminal, otherwise the original callable unchanged.
    While a run shell is open (``shell_board``) ``print`` goes to its board.
    """
    global _BOARD
    if _SHELL_BOARD is not None and notify is print:
        yield _SHELL_BOARD.log
        return
    stream = sys.stdout if stream is None else stream
    interactive = notify is print and _BOARD is None
    try:
        interactive = interactive and stream.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        yield notify
        return
    board = _BOARD = TerminalBoard(stream)
    try:
        yield board.log
    finally:
        _BOARD = None
        board.close()


class PrefixedBoardNotify:
    """Keep a book's logs and status rows on the queue's shared board."""

    def __init__(self, board, prefix):
        self.board = board
        self.prefix = prefix

    def __call__(self, message):
        self.board.log(self.prefix + str(message))

    log = __call__

    def set_row(self, key, order, text):
        self.board.set_row(key, (self.prefix, *order), self.prefix + text)

    def remove_row(self, key):
        self.board.remove_row(key)

    def book_outcome(self, key, succeeded):
        self.board.book_outcome(key, succeeded)

    @property
    def structured_status(self):
        return getattr(self.board, "structured_status", False)

    def set_status(self, key, order, snapshot):
        self.board.set_status(key, (self.prefix, *order), snapshot, book=_prefix_book(self.prefix))

    def chapter_failure(self, failure, retry=None, repair=None):
        if getattr(self.board, "structured_status", False):
            self.board.chapter_failure(failure, book=_prefix_book(self.prefix), retry=retry, repair=repair)

    def book_state(self, key, state):
        if getattr(self.board, "structured_status", False):
            self.board.book_state(key, state)

    def retry_hold(self, release, failed):
        if getattr(self.board, "structured_status", False):
            self.board.retry_hold(release, failed, book=_prefix_book(self.prefix))


def _board_of(notify):
    if isinstance(notify, PrefixedBoardNotify):
        return notify
    owner = getattr(notify, "__self__", None)
    return owner if isinstance(owner, TerminalBoard) else None


def offers_chapter_retries(notify):
    """Whether this run's output is a run shell that can ask for failed chapters again."""
    owner = _board_of(notify)
    return owner is not None and bool(getattr(owner, "structured_status", False))


def report_book_state(notify, key, state):
    """A book's lifecycle state for a run shell; plain, terminal and custom output ignore it."""
    if offers_chapter_retries(notify):
        _board_of(notify).book_state(key, state)


def report_chapter_failure(notify, failure, retry=None, repair=None):
    """Hand one chapter-scoped failure record (and its retry callback) to a run shell's board.

    The line output already carries the failure message; plain, terminal
    boards and custom notify callbacks ignore this.
    """
    if offers_chapter_retries(notify):
        _board_of(notify).chapter_failure(failure, retry=retry, repair=repair)


def report_retry_hold(notify, release, failed):
    """Tell a run shell the run is waiting for retries (``release`` finishes it); ``None`` ends the wait."""
    if offers_chapter_retries(notify):
        _board_of(notify).retry_hold(release, failed)


def report_book_outcome(notify, key, succeeded):
    """Record one finished book's outcome on the queue's shared board, if any.

    Plain and custom notify callbacks have no board: non-TTY output and test
    notifies stay exactly as they were, with no colors and no extra lines.
    """
    owner = _board_of(notify)
    if owner is not None:
        owner.book_outcome(key, succeeded)


_SHORT_PHASES = (
    ("generating speech, attempt ", "gen "), ("generating paragraph, attempt ", "gen "),
    ("verifying speech, attempt ", "verify "), ("verifying saved take ", "verify saved "),
    ("normalizing speech", "normalize"), ("checking saved audio", "cache"),
    ("verifying saved audio", "verify cache"), ("preparing paragraph", "prepare"),
    ("preparing speech", "prepare"),
)


def _short_phase(phase):
    phase = phase.replace("\n", " ").replace("\r", " ")
    for long, short in _SHORT_PHASES:
        phase = phase.replace(long, short)
    return phase


def format_duration(seconds):
    """Format a monotonic duration, including hours for long book renders."""
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


@dataclass
class _Task:
    phase: str
    phase_started: float
    generated_at: float | None = None
    worker: str | None = None
    worker_stage: str | None = None


class _TaskNotifier:
    def __init__(self, progress, index):
        self._progress = progress
        self._index = index

    def __call__(self, message):
        self._progress._task_message(self._index, message)

    def phase(self, phase, *, generated=False):
        self._progress._set_phase(self._index, phase, generated=generated)

    def worker_activity(self, event, worker=None, elapsed=None):
        self._progress._worker_activity(self._index, event, worker, elapsed)


def set_phase(notify, phase, *, generated=False):
    """Update a tracked task; ordinary print/log callbacks remain compatible."""
    if isinstance(notify, _TaskNotifier):
        notify.phase(phase, generated=generated)


_GENERATION_NOTIFY = ContextVar("audiobook_generation_notify", default=None)

# Console-only display policy (ChapterProgress with notify=print): these lines
# are already represented by the rewritten live status line. Custom notify
# callbacks (tests, integrations) always receive every message.
_FIRST_ATTEMPT = re.compile(r"\s*Generating (?:speech|paragraph), attempt 1/\d+\s*$")
_CACHE_CHECK = re.compile(r"\s*Chunk \d+/\d+: checking (?:speech|paragraph) cache\b")


@contextmanager
def generation_activity(notify):
    """Bind progress to one take without adding UI arguments to every engine."""
    token = _GENERATION_NOTIFY.set(notify)
    try:
        yield
    finally:
        _GENERATION_NOTIFY.reset(token)


def generation_notifier():
    return _GENERATION_NOTIFY.get()


def _worker_message(event, worker, elapsed):
    if event == "started":
        return f"Generating speech on {worker}"
    if event in {"completed", "failed"}:
        duration = format_duration(elapsed or 0)
        return f"{worker}: speech generation {event} in {duration}"
    return None


def report_worker_activity(notify, event, worker=None, elapsed=None):
    if event == "source" and (not isinstance(worker, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", worker)):
        worker = None
    try:
        if isinstance(notify, _TaskNotifier):
            notify.worker_activity(event, worker, elapsed)
        elif notify is not None:
            message = _worker_message(event, worker, elapsed)
            if message:
                notify("  " + message)
    except (OSError, ValueError):
        # An unavailable output stream must not turn synthesized audio into a
        # failed take, or stop a selected worker before its slot is returned.
        pass


def _block_voices(chunk):
    """Voice IDs speaking one block (IDs only), for the run shell's voice filter."""
    from .bundle import chunk_speakers
    try:
        return [speaker["voice_id"] for speaker in chunk_speakers(chunk) if speaker.get("voice_id")]
    except (KeyError, TypeError, ValueError, AttributeError):
        return []


class ChapterProgress:
    """Count successful blocks and show real elapsed time during blocking TTS.

    Task indices are zero-based source positions; completed tasks may arrive in
    any order. The estimate covers speech generation and verification only. It
    uses character-weighted timings of at least three newly generated blocks,
    never fast cache hits. Unvisited blocks are provisionally treated as work.

    A daemon heartbeat refreshes standard terminal output once a second and
    emits plain redirected-output updates every thirty seconds. Custom notify
    callbacks receive synchronous start/completion/final messages by default;
    tests or integrations can explicitly enable the heartbeat. No model work
    is moved to the UI thread. ``refresh`` allows deterministic clock tests.
    """

    def __init__(self, number, total_chapters, chunks, *, notify=print, parallel=1,
                 clock=time.monotonic, stream=None, heartbeat=None,
                 refresh_interval=1.0, log_interval=30.0):
        self.number = number
        self.total_chapters = total_chapters
        self._weights = [0 if chunk.get("silent_control") else len(chunk.get("spoken_text", ""))
                         for chunk in chunks]
        self._chunks = chunks
        self._notify = notify
        # On an interactive terminal the run's board gives this chapter one
        # persistent row; its messages scroll above all chapters' rows.
        self._board = _board_of(notify)
        self._custom_notify = notify is not print and self._board is None
        self._stream = stream if stream is not None else (sys.stdout if notify is print else None)
        self._tty = self._board is not None or bool(self._stream is not None and self._stream.isatty())
        self._clock = clock
        self._parallel = max(1, parallel)
        self._heartbeat = not self._custom_notify if heartbeat is None else heartbeat
        self._refresh_interval = max(0.01, refresh_interval)
        self._log_interval = max(0.01, log_interval)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._started = None
        self._closed = False
        self._last_display = None
        self._line_visible = False
        self._output_failed = False
        self._active = {}
        self._done = set()
        self._generated_samples = []
        self._clip_operation = None

    def __enter__(self):
        with self._lock:
            if self._started is not None:
                raise RuntimeError("Chapter progress cannot be started twice")
            self._started = self._clock()
            self._show(force=True)
            if self._heartbeat:
                self._thread = threading.Thread(target=self._tick,
                                                name="audiobook-progress", daemon=True)
                self._thread.start()
            # Started last so nothing after it can fail inside __enter__ and
            # leak the chapter-clips gauges.
            self._clip_operation = METRICS.operation("chapter_clips")
            self._clip_operation.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            # Stop refreshing before clearing the live line and writing a summary.
            self._thread.join(timeout=2.0)
        with self._lock:
            self._closed = True
            self._clear_line()
            state = "complete" if exc_type is None and len(self._done) == len(self._weights) else "stopped"
            if self._board is not None:
                self._board.remove_row(id(self))
                self._emit(f"speech {state}: {len(self._done)}/{len(self._weights)} blocks in "
                           f"{format_duration(self._clock() - self._started)}")
            else:
                self._emit(self._status(final=state))
            operation, self._clip_operation = self._clip_operation, None
        if operation is not None:
            operation.end(classify_outcome(exc))
        return False

    def __call__(self, message):
        if not self._custom_notify and _CACHE_CHECK.match(message):
            return
        with self._lock:
            self._clear_line()
            self._emit(message)
            if self._tty and not self._closed:
                self._show(force=True)

    def _task_message(self, index, message):
        if not self._custom_notify and _FIRST_ATTEMPT.match(message):
            return
        with self._lock:
            task = self._active.get(index)
            if task and task.worker and not message.lstrip().startswith("Chunk "):
                origin = "worker" if task.worker_stage else "audio from"
                message = f"  Chunk {index + 1} ({origin} {task.worker}): {message.lstrip()}"
            self(message)

    def _worker_activity(self, index, event, worker, elapsed):
        with self._lock:
            task = self._active.get(index)
            if task is None or self._closed:
                return
            if event == "waiting":
                # Pinned chapters name the machine they are queued for.
                task.worker, task.worker_stage = worker, "waiting"
            elif event == "assigned":
                task.worker, task.worker_stage = worker, "waiting"
            elif event == "started":
                task.worker, task.worker_stage = worker, "generating"
            elif event == "source":
                task.worker, task.worker_stage = worker, None
            elif event in {"completed", "failed"}:
                task.worker_stage = "failed" if event == "failed" else None
                task.phase = "speech generated" if event == "completed" else "generation failed"
            if event != "source":
                task.phase_started = self._clock()
            message = _worker_message(event, worker, elapsed)
            if not self._custom_notify and event == "started":
                message = None
            if not self._custom_notify and event == "completed" and elapsed is not None:
                message = f"{worker} done in {format_duration(elapsed)}"
            if self._board is not None and event == "completed":
                # The chapter's live row already shows routine generation.
                message = None
            if message:
                self(f"  Chunk {index + 1}: {message}")
            elif self._tty:
                self._show(force=True)

    @contextmanager
    def task(self, index):
        with self._lock:
            if self._started is None or self._closed:
                raise RuntimeError("Chapter progress must be active before starting a block")
            if not 0 <= index < len(self._weights):
                raise IndexError("Progress block index is outside the chapter")
            if index in self._active or index in self._done:
                raise RuntimeError("Progress block is already running or complete")
            self._active[index] = _Task("checking saved audio", self._clock())
            if self._tty:
                self._show(force=True)
        with METRICS.active("clip"):
            try:
                yield _TaskNotifier(self, index)
            except BaseException:
                with self._lock:
                    self._active.pop(index, None)
                raise
            with self._lock:
                task = self._active.pop(index)
                self._done.add(index)
                if task.generated_at is not None and self._weights[index] > 0:
                    self._generated_samples.append((self._weights[index],
                                                    max(0.0, self._clock() - task.generated_at)))
                self._show(force=self._custom_notify or self._tty)

    def _set_phase(self, index, phase, *, generated):
        with self._lock:
            task = self._active.get(index)
            if task is None or self._closed:
                return
            now = self._clock()
            if phase != task.phase:
                task.phase, task.phase_started = phase, now
            if generated and task.generated_at is None:
                task.generated_at = now
            if generated or phase == "checking saved audio":
                task.worker, task.worker_stage = None, None
            if self._tty:
                self._show(force=True)

    def refresh(self):
        """Display the current state immediately, without advancing any task."""
        with self._lock:
            if self._started is not None and not self._closed:
                self._show(force=True)

    def _remaining(self, now):
        if len(self._generated_samples) < 3:
            return None
        duration = sum(sample[1] for sample in self._generated_samples)
        weight = sum(sample[0] for sample in self._generated_samples)
        if duration <= 0 or weight <= 0:
            return None
        seconds_per_character = duration / weight
        remaining = []
        for index, chars in enumerate(self._weights):
            if index in self._done or chars <= 0:
                continue
            estimate = chars * seconds_per_character
            task = self._active.get(index)
            if task is not None and task.generated_at is not None:
                estimate -= now - task.generated_at
                if estimate <= 0:
                    # This block has exceeded the observed rate. Do not count
                    # down to zero while synthesis is visibly still running.
                    return None
            remaining.append(estimate)
        if not remaining:
            return None
        # A large block cannot be divided between workers; account for that tail.
        return max(max(remaining), sum(remaining) / min(self._parallel, len(remaining)))

    def snapshot(self):
        """This chapter's numbers for a run shell: blocks, timing, and each active block."""
        now = self._clock()
        remaining = self._remaining(now)
        active = []
        for index, task in sorted(self._active.items()):
            if task.worker_stage == "generating":
                phase = _short_phase(task.phase)
            elif task.worker_stage == "waiting":
                phase = "waiting for a worker"
            elif task.worker_stage == "failed":
                phase = "generation failed"
            else:
                phase = _short_phase(task.phase)
            active.append({"block": index + 1, "phase": phase, "worker": task.worker,
                           "worker_stage": task.worker_stage, "elapsed": now - task.phase_started,
                           "voices": _block_voices(self._chunks[index])})
        return {"chapter": self.number, "total_chapters": self.total_chapters,
                "done": len(self._done), "blocks": len(self._weights),
                "elapsed": now - self._started, "remaining": remaining,
                "active": active, "row": self._row()}

    def _row(self):
        """This chapter's live board row: no book percentage or shared segment."""
        now = self._clock()
        parts = [f"Ch {self.number}/{self.total_chapters}",
                 f"{len(self._done)}/{len(self._weights)} blocks "
                 f"({round(100 * len(self._done) / max(1, len(self._weights)))}%)",
                 format_duration(now - self._started)]
        active = []
        for index, task in sorted(self._active.items()):
            if task.worker_stage == "generating":
                phase = f"{task.worker} {_short_phase(task.phase)}"
            elif task.worker_stage == "waiting":
                phase = f"waiting for {task.worker or 'a worker'}"
            elif task.worker_stage == "failed":
                phase = f"{task.worker} failed"
            else:
                phase = _short_phase(task.phase)
                if task.worker and phase.startswith(("normalize", "verify")):
                    phase += f" ({task.worker} audio)"
            active.append(f"#{index + 1} {phase} {format_duration(now - task.phase_started)}")
        parts.append("; ".join(active) if active else "idle")
        remaining = self._remaining(now)
        if remaining is not None:
            parts.append(f"~{format_duration(math.ceil(remaining))} left")
        return " | ".join(parts)

    def _status(self, *, final=None, compact=False):
        now = self._clock()
        elapsed = format_duration(now - self._started)
        chapter_percent = round(100 * len(self._done) / max(1, len(self._weights)))
        book_percent = round(100 * ((self.number - 1)
                                    + len(self._done) / max(1, len(self._weights)))
                             / max(1, self.total_chapters))
        if compact and final is None and any(task.worker for task in self._active.values()):
            # Put machine activity before totals so an 80-column terminal shows
            # remote generation even while an earlier block verifies locally.
            active = []
            tasks = sorted(self._active.items(), key=lambda item: (
                item[1].worker_stage != "generating", item[0]))
            for index, task in tasks:
                if task.worker_stage == "generating":
                    label = f"{task.worker} gen"
                elif task.worker_stage == "waiting":
                    label = f"{task.worker or 'renderer'} wait"
                elif task.phase.startswith("verifying"):
                    label = "local verify"
                elif task.phase.startswith("normalizing"):
                    label = "local normalize"
                elif task.worker_stage == "failed":
                    label = f"{task.worker} failed"
                else:
                    label = "cache" if "saved" in task.phase else "ready"
                active.append(f"{label} #{index + 1}")
            parts = [f"Ch {self.number}/{self.total_chapters} ({book_percent}%) | " + "; ".join(active),
                     f"{len(self._done)}/{len(self._weights)} blocks ({chapter_percent}%)",
                     f"{elapsed} elapsed"]
            segment = _scheduling_segment()
            if segment:
                parts.append(segment)
            return " | ".join(parts)
        if compact:
            parts = [f"Ch {self.number}/{self.total_chapters} ({book_percent}%)",
                     f"{len(self._done)}/{len(self._weights)} blocks ({chapter_percent}%)",
                     f"{elapsed} elapsed"]
        else:
            parts = [f"Chapter {self.number} / {self.total_chapters} (book {book_percent}%)",
                     f"Blocks {len(self._done)} / {len(self._weights)} complete ({chapter_percent}%)",
                     f"elapsed {elapsed}"]
        if final is not None:
            parts.append(f"speech {final}")
        remaining = self._remaining(now) if final is None else None
        if compact and remaining is not None:
            parts.append(f"~{format_duration(math.ceil(remaining))} left")
        if final is None and self._active:
            active = []
            for index, task in sorted(self._active.items()):
                phase = task.phase.replace("\n", " ").replace("\r", " ")
                if task.worker_stage == "waiting":
                    phase = f"waiting for {task.worker or 'renderer'}"
                elif task.worker_stage in {"generating", "failed"}:
                    phase = f"{task.worker}: {phase}"
                elif task.worker:
                    if phase.startswith(("normalizing", "verifying")):
                        phase += " locally"
                    phase += f" (audio: {task.worker})"
                if compact:
                    phase = phase.replace("generating speech, attempt ", "gen ")
                    phase = phase.replace("generating paragraph, attempt ", "gen ")
                    phase = phase.replace("verifying speech, attempt ", "verify ")
                    phase = phase.replace("normalizing speech", "normalize")
                    phase = phase.replace("checking saved audio", "cache")
                    phase = phase.replace("verifying saved audio", "verify cache")
                    phase = phase.replace("preparing paragraph", "prepare")
                    phase = phase.replace("preparing speech", "prepare")
                active.append(f"#{index + 1} {phase} {format_duration(now - task.phase_started)}")
                if compact:
                    if len(self._active) > 1:
                        active[-1] += f" +{len(self._active) - 1}"
                    break
            parts.append("; ".join(active))
        if final is None:
            segment = _scheduling_segment()
            if segment:
                parts.append(segment)
        if not compact and remaining is not None:
            parts.append(f"approx speech left {format_duration(math.ceil(remaining))}")
        return " | ".join(parts)

    def _emit(self, message):
        if self._board is not None:
            self._board.log(f"Ch {self.number}: {message.strip()}")
            return
        if self._custom_notify:
            self._notify(message)
        elif not self._output_failed:
            try:
                print(message, file=self._stream, flush=True)
            except (OSError, ValueError):
                self._output_failed = True

    def _write_terminal(self, text):
        if self._output_failed:
            return
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):
            self._output_failed = True

    def _clear_line(self):
        if self._line_visible:
            self._write_terminal("\r\033[2K")
            self._line_visible = False

    def _show(self, *, force=False):
        now = self._clock()
        if self._board is not None:
            if getattr(self._board, "structured_status", False):
                self._board.set_status(id(self), (self.number, id(self)), self.snapshot())
            else:
                self._board.set_row(id(self), (self.number, id(self)), self._row())
            self._last_display = now
            return
        if not force and self._last_display is not None and now - self._last_display < self._log_interval:
            return
        status = self._status()
        if self._tty:
            width = max(10, shutil.get_terminal_size(fallback=(120, 24)).columns - 1)
            if len(status) > width:
                status = self._status(compact=True)
            if len(status) > width:
                status = status[:max(0, width - 1)] + "…"
            self._write_terminal("\r\033[2K" + status)
            self._line_visible = not self._output_failed
        else:
            self._emit(status)
        self._last_display = now

    def _tick(self):
        while not self._stop.wait(self._refresh_interval):
            try:
                with self._lock:
                    if self._stop.is_set() or self._closed:
                        return
                    self._show(force=self._tty)
            except (OSError, ValueError):
                # A closed terminal should not crash or hang the rendering worker.
                return
