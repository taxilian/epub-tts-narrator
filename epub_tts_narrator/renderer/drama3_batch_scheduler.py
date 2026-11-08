"""Continuous batching of independent Drama 3 decodes on one model.

Each submitted request is one passage: its own prompt (references, chapter
history and text), sampling parameters and seed. Up to ``backend.rows``
requests decode together, one frame per step; a finished row is released at
once and the next waiting request takes its slot. Requests never share
prompts, caches or sampling state, so batching changes only how steps are
computed, never what a request conditions on or returns.

The backend owns the model and its per-row caches; this module is pure
Python so the admission and completion rules are testable without a GPU.
Only the scheduler thread calls the backend, so model work is never
concurrent with itself.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
import time
from typing import Any


class SchedulerClosed(RuntimeError):
    pass


class RequestCancelled(RuntimeError):
    """The request's cancel signal was set (its client went away); its row was freed."""


@dataclass
class BatchRequest:
    prompt: Any
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    # Optional per-request stop check: ``observe(codes) -> bool`` per frame
    # (see silence_stop.SilenceWatch); True ends the request as "silence".
    watch: Any = None
    # Optional prompt rows where a shared start ends (references, pinned history),
    # which a backend may cache for later requests with the same start.
    prefix_hints: tuple = ()
    # Optional threading.Event: once set, the request leaves the batch at the
    # next step (or is skipped before it starts) and fails RequestCancelled.
    cancel: Any = None
    future: Future = field(default_factory=Future)
    submitted: float = field(default_factory=time.perf_counter)
    # Payload-free per-request batching facts, delivered as ``future.batch_stats``.
    stats: dict = field(default_factory=dict)


@dataclass
class _Row:
    request: BatchRequest
    frames: list = field(default_factory=list)
    steps: int = 0
    row_sum: int = 0
    peak: int = 0


class Drama3BatchScheduler:
    """One decode thread; up to ``backend.rows`` passages per step.

    The backend provides ``rows``, ``eos_id``, ``prefill(slot, request)``,
    ``step(slots) -> [(frame, finite), ...]`` in slot order, and
    ``release(slot)``. A frame is ``[main token, code 0, ..., code N-1]``.
    Results are ``(frames, finish_reason)`` exactly like a single-request
    decode: frames exclude the end-of-speech token, ``"length"`` means the
    request's ``max_new_tokens`` ran out first, and ``"silence"`` means its
    watch ended it after the audio went silent.
    """

    def __init__(self, backend, *, name="drama3-batch"):
        if type(backend.rows) is not int or backend.rows < 1:
            raise ValueError("The batch backend needs at least one row")
        self.backend = backend
        self._queue: deque[BatchRequest] = deque()
        self._active: dict[int, _Row] = {}
        self._admitting: list[BatchRequest] = []
        self._condition = threading.Condition()
        self._closed = False
        self._failure: BaseException | None = None
        self.steps = 0
        self.row_steps = 0
        self.peak_active = 0
        # Payload-free efficiency counters (seconds on the decode thread).
        self.occupancy = Counter()   # active rows per step -> steps
        self.prefills = 0
        self.cancelled = 0
        self.prefill_seconds = 0.0
        self.step_seconds = 0.0
        self.observe_seconds = 0.0
        self.idle_seconds = 0.0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, prompt, max_new_tokens, *, temperature, top_k, top_p, seed, watch=None,
               prefix_hints=(), cancel=None) -> Future:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if watch is not None and not callable(getattr(watch, "observe", None)):
            raise ValueError("watch must provide observe(codes)")
        if cancel is not None and not callable(getattr(cancel, "is_set", None)):
            raise ValueError("cancel must provide is_set()")
        request = BatchRequest(prompt, max_new_tokens, temperature, top_k, top_p, seed, watch,
                               tuple(prefix_hints), cancel)
        with self._condition:
            if self._failure is not None:
                raise SchedulerClosed("The Drama 3 batch scheduler stopped after an internal error") from self._failure
            if self._closed:
                raise SchedulerClosed("The Drama 3 batch scheduler is closed")
            self._queue.append(request)
            self._condition.notify_all()
        return request.future

    def stats(self):
        """A snapshot of the counters: rows per step, and where the decode thread's time went."""
        with self._condition:
            queued = len(self._queue)
        busy = self.prefill_seconds + self.step_seconds + self.observe_seconds
        return {"rows": self.backend.rows, "steps": self.steps, "row_steps": self.row_steps,
                "mean_rows": self.row_steps / self.steps if self.steps else 0.0,
                "occupancy": dict(sorted(self.occupancy.items())), "peak_active": self.peak_active,
                "prefills": self.prefills, "prefill_seconds": self.prefill_seconds,
                "step_seconds": self.step_seconds, "observe_seconds": self.observe_seconds,
                "idle_seconds": self.idle_seconds, "busy_seconds": busy, "queued": queued,
                **({"prefix_cache": self.backend.prefixes.stats()} if hasattr(self.backend, "prefixes") else {})}

    def close(self, timeout=60):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout)
        with self._condition:
            waiting = list(self._queue)
            self._queue.clear()
        for request in waiting:
            if not request.future.done():
                request.future.set_exception(SchedulerClosed("The Drama 3 batch scheduler closed"))

    def _free_slots(self):
        return [slot for slot in range(self.backend.rows) if slot not in self._active]

    def _finish(self, slot, *, result=None, error=None):
        # Deliver first: a request's outcome never waits on releasing its slot.
        row = self._active.pop(slot)
        if error is not None:
            row.request.future.set_exception(error)
        else:
            stats = row.request.stats
            stats.update(steps=row.steps, peak_rows=row.peak,
                         mean_rows=round(row.row_sum / row.steps, 2) if row.steps else 0.0)
            row.request.future.batch_stats = stats
            row.request.future.set_result(result)
        self.backend.release(slot)

    def _run(self):
        try:
            self._loop()
        except BaseException as exc:
            # The backend's slot bookkeeping is no longer trustworthy. Fail
            # every waiting request now instead of stranding it, and refuse
            # new work.
            with self._condition:
                self._failure = exc
                self._closed = True
                waiting = list(self._queue)
                self._queue.clear()
            active = [row.request for row in self._active.values()]
            admitting, self._admitting = self._admitting, []
            self._active.clear()
            for request in (*active, *admitting, *waiting):
                if not request.future.done():
                    request.future.set_exception(SchedulerClosed(
                        "The Drama 3 batch scheduler stopped after an internal error"))

    def _loop(self):
        while True:
            with self._condition:
                waited = time.perf_counter()
                while not self._closed and not self._queue and not self._active:
                    self._condition.wait()
                self.idle_seconds += time.perf_counter() - waited
                if self._closed:
                    break
                admissions = []
                for slot in self._free_slots():
                    if not self._queue:
                        break
                    admissions.append((slot, self._queue.popleft()))
                self._admitting = [request for _slot, request in admissions]
            for slot, request in admissions:
                self._admitting.remove(request)
                if not request.future.set_running_or_notify_cancel():
                    continue
                if request.cancel is not None and request.cancel.is_set():
                    request.future.set_exception(RequestCancelled("The request was cancelled before it started"))
                    continue
                began = time.perf_counter()
                request.stats.update(waited_seconds=round(began - request.submitted, 3),
                                     joined_active=len(self._active), rows=self.backend.rows)
                self._active[slot] = _Row(request)
                try:
                    self.backend.prefill(slot, request)
                except BaseException as exc:  # one bad prompt must not stop the others
                    self._finish(slot, error=exc)
                finally:
                    self.prefills += 1
                    self.prefill_seconds += time.perf_counter() - began
                    request.stats["prefill_seconds"] = round(time.perf_counter() - began, 3)
                    request.stats.update(getattr(self.backend, "last_prefill", None) or {})
            for slot in [slot for slot, row in self._active.items()
                         if row.request.cancel is not None and row.request.cancel.is_set()]:
                self.cancelled += 1
                self._finish(slot, error=RequestCancelled("The request was cancelled; its row was freed"))
            if not self._active:
                continue
            slots = sorted(self._active)
            self.peak_active = max(self.peak_active, len(slots))
            began = time.perf_counter()
            try:
                results = self.backend.step(slots)
                if len(results) != len(slots):
                    raise RuntimeError("The batch backend returned the wrong number of rows")
            except BaseException as exc:
                # The step's state is unknown, so every active row fails; the
                # next prefill resets whichever slot it uses.
                for slot in slots:
                    self._finish(slot, error=exc)
                continue
            self.step_seconds += time.perf_counter() - began
            self.steps += 1
            self.row_steps += len(slots)
            self.occupancy[len(slots)] += 1
            for slot in slots:
                row = self._active[slot]
                row.steps += 1
                row.row_sum += len(slots)
                row.peak = max(row.peak, len(slots))
            for slot, (frame, finite) in zip(slots, results):
                row = self._active[slot]
                if not finite:
                    self._finish(slot, error=FloatingPointError(
                        "non-finite logits; refusing to sample invalid audio"))
                elif frame[0] == self.backend.eos_id:
                    self._finish(slot, result=(row.frames, "eos"))
                else:
                    row.frames.append(list(frame))
                    watch = row.request.watch
                    began = time.perf_counter()
                    try:
                        silent = watch is not None and watch.observe(frame[1:])
                    except BaseException as exc:  # a failed check fails only its own request
                        self._finish(slot, error=exc)
                        continue
                    finally:
                        self.observe_seconds += time.perf_counter() - began
                    if silent:
                        self._finish(slot, result=(row.frames, "silence"))
                    elif len(row.frames) >= row.request.max_new_tokens:
                        self._finish(slot, result=(row.frames, "length"))
        for slot in list(self._active):
            self._finish(slot, error=SchedulerClosed("The Drama 3 batch scheduler closed"))
