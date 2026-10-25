"""Payload-free resource-stage metrics for the renderer.

Bounded, thread-safe aggregates for renderer resource stages: how much work is
queued or active at each stage, how long operations waited and ran (fixed
histogram buckets), and how operations ended. Stage names are a closed set
(``STAGES``); snapshots contain only stage names and numbers, never text,
prompts, transcripts, URLs, keys, or unbounded user labels. There is no
durable per-clip event history: every observation lands in the aggregate and
is discarded on ``reset``.

The API for later scheduling stages:

* ``metrics.operation(stage)`` returns a ``StageOperation`` token at enqueue
  time; ``token.start()`` when the work leaves the queue, ``token.end(outcome)``
  (or ``token.cancel()``) when it leaves the stage. Queue-backed lanes should
  carry the token with the job so cancellation of still-queued work can
  release the queued gauge.
* ``metrics.waiting(stage)`` observes a pure wait (enqueue to release).
* ``metrics.active(stage)`` observes an execution with ``completed`` /
  ``failed`` / ``cancelled`` classified from the exception that leaves the
  block: ``KeyboardInterrupt``, ``SystemExit``, and ``RenderPaused`` count as
  cancelled; every other exception counts as failed.

Stages are independent observations and intentionally overlap when nested (a
synthesis contains a model-thread segment). New stages are added to ``STAGES``
by an explicit reviewed change, never by callers. ``gpu_telemetry`` in every
snapshot is a constant: GPU utilization samples are simply absent until a
collector exists, and reports must say so instead of implying idleness.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from time import monotonic

from .interrupts import RenderPaused


STAGES = frozenset({
    "admission",       # whole-chunk admission window (RenderPool.admit_chunk)
    "synthesis_slot",  # waiting for a TTS worker slot (RenderPool._execute)
    "synthesis",       # active synthesis on a slot
    "recognition",     # active serialized model-affine verification
    "review",          # active text-only network review lane
    "preparation",     # bounded audio preparation lane (normalization, metrics)
    "review_coordination",  # blind-review coordination executor (not HTTP itself)
    "model_thread",    # persistent local inference thread (LocalRuntime)
    "chapter",         # chapter coordination (ChapterExecutor.synthesize)
    "chapter_clips",   # chapter clip window (ChapterProgress lifecycle)
    "clip",            # one chunk transaction inside a chapter window
})

OUTCOMES = ("completed", "failed", "cancelled")

# Fixed duration buckets in seconds; names name the inclusive lower and
# exclusive upper edge. Bounded memory: one counter per bucket per stage.
_BUCKET_BOUNDS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
BUCKET_NAMES = ("under_1ms", "1_10ms", "10_100ms", "100ms_1s",
                "1_10s", "10_100s", "100_1000s", "over_1000s")


def bucket_index(seconds):
    for index, bound in enumerate(_BUCKET_BOUNDS):
        if seconds < bound:
            return index
    return len(BUCKET_NAMES) - 1


def classify_outcome(error):
    """Classify how an operation ended, without recording the exception itself."""
    if error is None:
        return "completed"
    if isinstance(error, (KeyboardInterrupt, SystemExit, RenderPaused)):
        return "cancelled"
    return "failed"


def _fresh_state():
    return {stage: {"queued": 0, "active": 0,
                    "wait_seconds": 0.0, "wait_count": 0,
                    "active_seconds": 0.0, "active_count": 0,
                    "wait_buckets": [0] * len(BUCKET_NAMES),
                    "active_buckets": [0] * len(BUCKET_NAMES),
                    "outcomes": dict.fromkeys(OUTCOMES, 0)}
            for stage in STAGES}


class StageOperation:
    """One observed pass through a stage: enqueue, then start, then end."""

    __slots__ = ("_metrics", "_stage", "_enqueued", "_started", "_ended")

    def __init__(self, metrics, stage):
        self._metrics = metrics
        self._stage = stage
        self._enqueued = monotonic()
        self._started = None
        self._ended = False

    def start(self):
        with self._metrics._lock:
            if self._ended:
                raise RuntimeError("A stage operation cannot start after it ended")
            if self._started is not None:
                raise RuntimeError("A stage operation can only start once")
            state = self._metrics._state[self._stage]
            state["queued"] -= 1
            self._started = monotonic()
            waited = self._started - self._enqueued
            state["wait_seconds"] += waited
            state["wait_count"] += 1
            state["wait_buckets"][bucket_index(waited)] += 1
            state["active"] += 1

    def end(self, outcome="completed"):
        if outcome not in OUTCOMES:
            raise ValueError("outcome must be one of render_metrics.OUTCOMES")
        with self._metrics._lock:
            if self._ended:
                raise RuntimeError("A stage operation can only end once")
            state = self._metrics._state[self._stage]
            self._ended = True
            if self._started is None:
                waited = monotonic() - self._enqueued
                state["wait_seconds"] += waited
                state["wait_count"] += 1
                state["wait_buckets"][bucket_index(waited)] += 1
                state["queued"] -= 1
            else:
                ran = monotonic() - self._started
                state["active_seconds"] += ran
                state["active_count"] += 1
                state["active_buckets"][bucket_index(ran)] += 1
                state["active"] -= 1
            state["outcomes"][outcome] += 1

    def cancel(self):
        self.end("cancelled")


class RenderMetrics:
    """Thread-safe bounded aggregate counters for the closed stage set."""

    def __init__(self):
        self._lock = Lock()
        self._state = _fresh_state()

    def _stage_state(self, stage):
        if not isinstance(stage, str) or stage not in STAGES:
            # Deliberately never echo the rejected value: it may be payload.
            raise ValueError("stage must be one of the render_metrics.STAGES names")
        return self._state[stage]

    def operation(self, stage):
        """Record an enqueue and return the token that owns start/end."""
        with self._lock:
            self._stage_state(stage)["queued"] += 1
        return StageOperation(self, stage)

    @contextmanager
    def waiting(self, stage):
        """Observe a pure wait: enqueue at entry, wait bucket at exit."""
        operation = self.operation(stage)
        try:
            yield operation
        except BaseException as error:
            operation.end(classify_outcome(error))
            raise
        operation.end("completed")

    @contextmanager
    def active(self, stage):
        """Observe an execution; the outcome is classified from its exception."""
        operation = self.operation(stage)
        operation.start()
        try:
            yield operation
        except BaseException as error:
            operation.end(classify_outcome(error))
            raise
        operation.end("completed")

    def snapshot(self):
        """A JSON-serializable counter report; no per-event history exists."""
        with self._lock:
            report = {"gpu_telemetry": "unavailable"}
            for stage in sorted(self._state):
                state = self._state[stage]
                report[stage] = {
                    "queued_now": state["queued"],
                    "active_now": state["active"],
                    "wait_seconds": state["wait_seconds"],
                    "wait_count": state["wait_count"],
                    "active_seconds": state["active_seconds"],
                    "active_count": state["active_count"],
                    "wait_buckets": dict(zip(BUCKET_NAMES, state["wait_buckets"])),
                    "active_buckets": dict(zip(BUCKET_NAMES, state["active_buckets"])),
                    "outcomes": dict(state["outcomes"]),
                }
            return report

    def reset(self):
        with self._lock:
            self._state = _fresh_state()


METRICS = RenderMetrics()
