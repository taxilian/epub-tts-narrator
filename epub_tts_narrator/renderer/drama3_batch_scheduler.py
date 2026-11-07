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

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
from typing import Any


class SchedulerClosed(RuntimeError):
    pass


@dataclass
class BatchRequest:
    prompt: Any
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    future: Future = field(default_factory=Future)


@dataclass
class _Row:
    request: BatchRequest
    frames: list = field(default_factory=list)


class Drama3BatchScheduler:
    """One decode thread; up to ``backend.rows`` passages per step.

    The backend provides ``rows``, ``eos_id``, ``prefill(slot, request)``,
    ``step(slots) -> [(frame, finite), ...]`` in slot order, and
    ``release(slot)``. A frame is ``[main token, code 0, ..., code N-1]``.
    Results are ``(frames, finish_reason)`` exactly like a single-request
    decode: frames exclude the end-of-speech token, and ``"length"`` means the
    request's ``max_new_tokens`` ran out first.
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
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, prompt, max_new_tokens, *, temperature, top_k, top_p, seed) -> Future:
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        request = BatchRequest(prompt, max_new_tokens, temperature, top_k, top_p, seed)
        with self._condition:
            if self._failure is not None:
                raise SchedulerClosed("The Drama 3 batch scheduler stopped after an internal error") from self._failure
            if self._closed:
                raise SchedulerClosed("The Drama 3 batch scheduler is closed")
            self._queue.append(request)
            self._condition.notify_all()
        return request.future

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
                while not self._closed and not self._queue and not self._active:
                    self._condition.wait()
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
                self._active[slot] = _Row(request)
                try:
                    self.backend.prefill(slot, request)
                except BaseException as exc:  # one bad prompt must not stop the others
                    self._finish(slot, error=exc)
            if not self._active:
                continue
            slots = sorted(self._active)
            self.peak_active = max(self.peak_active, len(slots))
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
            self.steps += 1
            self.row_steps += len(slots)
            for slot, (frame, finite) in zip(slots, results):
                row = self._active[slot]
                if not finite:
                    self._finish(slot, error=FloatingPointError(
                        "non-finite logits; refusing to sample invalid audio"))
                elif frame[0] == self.backend.eos_id:
                    self._finish(slot, result=(row.frames, "eos"))
                else:
                    row.frames.append(list(frame))
                    if len(row.frames) >= row.request.max_new_tokens:
                        self._finish(slot, result=(row.frames, "length"))
        for slot in list(self._active):
            self._finish(slot, error=SchedulerClosed("The Drama 3 batch scheduler closed"))
