"""Bounded, cancellable resource lanes for pooled rendering.

A ``ResourceLane`` is a long-lived bounded queue with a fixed pool of worker
threads: one lane per resource class, instantiated with the declared
concurrency of its endpoint (TTS slots, the single ASR model, review HTTP
calls, the two-worker audio-preparation lane, and the
``2 * review_concurrency`` blind-review coordination executor). Lanes never
execute model work themselves and never infer hardware: the caller decides
the worker count, and model-affine callables keep delegating to
``LocalRuntime``, which retains its own persistent inference thread, inline
same-thread reentrancy, and drain-before-close semantics.

Public API for scheduling integration:

* ``ResourceLane(name=..., workers=N, stage=..., max_pending=M)`` starts N
  workers immediately. ``stage`` must be a ``render_metrics.STAGES`` name:
  the lane's queued gauge covers submit-to-dequeue and its active gauge
  covers execution on the lane worker.
* ``submit(function, *args, owner=(run_key, chapter_key), **kwargs)``
  returns a ``concurrent.futures.Future``. Every submission captures
  ``contextvars.copy_context()`` at submit time, so scoped stop conditions
  (owner loss, session epochs) travel with the job. Pending work is bounded:
  beyond ``max_pending`` submissions raise ``LaneFull`` and the caller owns
  backpressure. Owners are served round-robin one job at a time, FIFO within
  an owner.
* ``run(...)`` is the synchronous adapter: submit, then ``future.result()``.
* At dequeue every job checks its copied context with
  ``interrupts.stop_if_requested()`` before running; a stopped job or one
  whose owner/epoch is gone never executes and reports the stop exception on
  its future.
* Cancelling a still-queued future wakes the queue promptly: the job never
  executes and acquires no resource. Exceptions propagate as the original
  safe exception on the future; there is no automatic stage retry.
* ``close(drain=False)`` cancels all queued work and joins the workers;
  ``close(drain=True)`` lets queued work finish first. Close is idempotent
  and must run on a coordinator thread, never on one of the lane's own
  workers. A worker waiting on its own lane is refused by the same guard;
  cross-lane waits, such as the coordination executor awaiting the HTTP
  lane, remain allowed.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from contextvars import copy_context
from threading import Condition, Thread, get_ident

from .interrupts import stop_if_requested
from .render_metrics import METRICS, STAGES, classify_outcome


class LaneFull(RuntimeError):
    """The lane's bounded pending queue is full; the caller owns backpressure."""


class _Job:
    __slots__ = ("context", "function", "args", "kwargs", "future", "operation")

    def __init__(self, context, function, args, kwargs, future, operation):
        self.context = context
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.operation = operation


class ResourceLane:
    """Fixed-worker bounded queue with owner fairness and scoped cancellation."""

    def __init__(self, *, name, workers, stage, max_pending):
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty resource lane name")
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if not isinstance(stage, str) or stage not in STAGES:
            # Never echo the rejected value: stage labels may carry payload.
            raise ValueError("stage must be one of the render_metrics.STAGES names")
        self._name = name
        self._stage = stage
        self._max_pending = max_pending
        self._cond = Condition()
        self._owners = {}
        self._cursor = None
        self._pending = 0
        self._closed = False
        self._drain = False
        self._worker_idents = set()
        self._threads = []
        for index in range(workers):
            thread = Thread(target=self._work, name=f"{name}-{index}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def submit(self, function, *args, owner=(None, None), **kwargs):
        """Queue the callable and return its Future; raises LaneFull when full."""
        if get_ident() in self._worker_idents:
            raise RuntimeError("A lane worker must not submit work to its own lane")
        if not callable(function):
            raise TypeError("submit requires a callable")
        with self._cond:
            if self._closed:
                raise RuntimeError("The resource lane is closed")
            if self._pending >= self._max_pending:
                raise LaneFull("The resource lane pending queue is full")
            operation = METRICS.operation(self._stage)
            job = _Job(copy_context(), function, args, kwargs, Future(), operation)
            self._owners.setdefault(owner, deque()).append(job)
            self._pending += 1
            self._cond.notify()
        return job.future

    def run(self, function, *args, owner=(None, None), **kwargs):
        """Submit and wait synchronously; return the value or raise the error."""
        return self.submit(function, *args, owner=owner, **kwargs).result()

    def close(self, *, drain=False):
        """Stop submissions; cancel queued work (default) or drain it, then join."""
        if get_ident() in self._worker_idents:
            raise RuntimeError("Close a resource lane from its coordinator, "
                               "never from one of its workers")
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._drain = bool(drain)
            if not drain:
                for queue in self._owners.values():
                    for job in queue:
                        self._cancel(job)
                self._owners.clear()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join()

    def _cancel(self, job):
        # The lane condition is held: one funnel keeps futures and gauges equal.
        if job.future.cancel():
            job.operation.cancel()

    def _take(self):
        # The lane condition is held: one job at a time, rotating fair owners.
        owners = self._owners
        if not owners:
            return None
        keys = list(owners)
        if self._cursor in owners:
            start = keys.index(self._cursor)
            keys = keys[start + 1:] + keys[:start + 1]
        for key in keys:
            queue = owners[key]
            job = queue.popleft()
            if not queue:
                del owners[key]
            self._cursor = key
            self._pending -= 1
            return job
        return None

    def _work(self):
        self._worker_idents.add(get_ident())
        try:
            while True:
                with self._cond:
                    job = self._take()
                    if job is None:
                        if self._closed:
                            return
                        self._cond.wait()
                        continue
                    if self._closed and not self._drain:
                        self._cancel(job)
                        continue
                    if not job.future.set_running_or_notify_cancel():
                        job.operation.cancel()
                        continue
                    job.operation.start()
                try:
                    job.context.run(stop_if_requested)
                except BaseException as stop:
                    job.operation.end(classify_outcome(stop))
                    job.future.set_exception(stop)
                    continue
                try:
                    result = job.context.run(job.function, *job.args, **job.kwargs)
                except BaseException as error:
                    job.operation.end(classify_outcome(error))
                    job.future.set_exception(error)
                    continue
                job.operation.end("completed")
                job.future.set_result(result)
        finally:
            self._worker_idents.discard(get_ident())
