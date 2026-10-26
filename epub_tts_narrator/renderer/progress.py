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
        self._notify = notify
        self._custom_notify = notify is not print
        self._stream = stream if stream is not None else (sys.stdout if notify is print else None)
        self._tty = bool(self._stream is not None and self._stream.isatty())
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
                task.worker, task.worker_stage = None, "waiting"
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
