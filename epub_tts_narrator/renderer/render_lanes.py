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
  backpressure. Jobs are considered in global submission order.
* ``submit(..., reserve=callable)`` gates dequeue on an external resource:
  the lane calls ``reserve()`` under its own lock in submission order;
  ``None`` leaves that job queued while a later job with an available
  resource may run, and anything else is a release callback the lane calls
  exactly once after the job finishes or is dropped, then wakes its workers.
  Same-resource work therefore stays FIFO without blocking another eligible
  resource. A job that cannot start never occupies a lane thread.
  ``reserve`` must not block or call back into the lane.
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

from .interrupts import stop_if_owner_requested, stop_if_requested
from .render_metrics import METRICS, STAGES, classify_outcome


class LaneFull(RuntimeError):
    """The lane's bounded pending queue is full; the caller owns backpressure."""


class _Job:
    __slots__ = ("context", "function", "args", "kwargs", "future", "operation",
                 "reserve", "release")

    def __init__(self, context, function, args, kwargs, future, operation, reserve=None):
        self.reserve = reserve
        self.release = None
        self.context = context
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.operation = operation


class ResourceLane:
    """Fixed-worker bounded FIFO queue with scoped cancellation."""

    def __init__(self, *, name, workers, stage, max_pending, stop_at_dequeue=True):
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
        # New expensive work (synthesis) refuses to start after any stop;
        # lanes finishing an already-generated take drain through a global
        # Ctrl-C but still drop work whose scoped owner was lost.
        self._stop_at_dequeue = stop_at_dequeue
        self._max_pending = max_pending
        self._cond = Condition()
        self._jobs = deque()
        self._capacity_waiters = deque()
        self._pending = 0
        self._closed = False
        self._drain = False
        self._worker_idents = set()
        self._threads = []
        for index in range(workers):
            thread = Thread(target=self._work, name=f"{name}-{index}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def _enqueue(self, function, args, kwargs, reserve):
        """Append one job while holding the lane condition."""
        operation = METRICS.operation(self._stage)
        future = Future()

        def wake_cancelled(done):
            if done.cancelled():
                with self._cond:
                    self._cond.notify_all()

        future.add_done_callback(wake_cancelled)
        job = _Job(copy_context(), function, args, kwargs, future, operation, reserve)
        self._jobs.append(job)
        self._pending += 1
        self._cond.notify()
        return job.future

    def submit(self, function, *args, owner=(None, None), reserve=None, **kwargs):
        """Queue the callable and return its Future; raises LaneFull when full."""
        if get_ident() in self._worker_idents:
            raise RuntimeError("A lane worker must not submit work to its own lane")
        if not callable(function):
            raise TypeError("submit requires a callable")
        if reserve is not None and not callable(reserve):
            raise TypeError("reserve must be callable")
        with self._cond:
            if self._closed:
                raise RuntimeError("The resource lane is closed")
            if self._capacity_waiters or self._pending >= self._max_pending:
                raise LaneFull("The resource lane pending queue is full")
            return self._enqueue(function, args, kwargs, reserve)

    def run(self, function, *args, owner=(None, None), reserve=None, **kwargs):
        """Wait in FIFO order for capacity, then wait for the job's result."""
        from .interrupts import stop_if_requested

        if get_ident() in self._worker_idents:
            raise RuntimeError("A lane worker must not submit work to its own lane")
        if not callable(function):
            raise TypeError("run requires a callable")
        if reserve is not None and not callable(reserve):
            raise TypeError("reserve must be callable")
        marker = object()
        with self._cond:
            self._capacity_waiters.append(marker)
            try:
                while True:
                    stop_if_requested()
                    if self._closed:
                        raise RuntimeError("The resource lane is closed")
                    if self._capacity_waiters[0] is marker and self._pending < self._max_pending:
                        self._capacity_waiters.popleft()
                        self._cond.notify_all()
                        future = self._enqueue(function, args, kwargs, reserve)
                        break
                    self._cond.wait(0.1)
            finally:
                if marker in self._capacity_waiters:
                    self._capacity_waiters.remove(marker)
                    self._cond.notify_all()
        return future.result()

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
                for job in self._jobs:
                    self._cancel(job)
                self._jobs.clear()
                # Cancelled queued jobs never reach _take; restore the count.
                self._pending = 0
            self._cond.notify_all()
        for thread in self._threads:
            thread.join()

    def _cancel(self, job):
        # The lane condition is held: one funnel keeps futures and gauges equal.
        # A submitter may have cancelled the future first; its token must end
        # exactly once either way, or the queued gauge leaks after close.
        if job.future.cancel() or job.future.cancelled():
            job.operation.cancel()

    def _take(self):
        # The lane condition is held: consider jobs in global submission order.
        for job in self._jobs:
            if job.reserve is not None and (not self._closed or self._drain) and not job.future.cancelled():
                release = job.reserve()
                if release is None:
                    continue
                job.release = release
            self._jobs.remove(job)
            self._pending -= 1
            self._cond.notify_all()
            return job
        return None

    def _work(self):
        self._worker_idents.add(get_ident())
        try:
            while True:
                with self._cond:
                    job = self._take()
                    if job is None:
                        # Reserved-resource jobs may still be queued behind a
                        # running job's release; wait for it even when closed.
                        if self._closed and not self._jobs:
                            return
                        self._cond.wait()
                        continue
                try:
                    self._run(job)
                finally:
                    if job.release is not None:
                        release, job.release = job.release, None
                        try:
                            release()
                        finally:
                            with self._cond:
                                self._cond.notify_all()
        finally:
            self._worker_idents.discard(get_ident())

    def _run(self, job):
        with self._cond:
            if self._closed and not self._drain:
                self._cancel(job)
                return
            if not job.future.set_running_or_notify_cancel():
                job.operation.cancel()
                return
            job.operation.start()
        try:
            job.context.run(stop_if_requested if self._stop_at_dequeue else stop_if_owner_requested)
        except BaseException as stop:
            job.operation.end(classify_outcome(stop))
            job.future.set_exception(stop)
            return
        try:
            result = job.context.run(job.function, *job.args, **job.kwargs)
        except BaseException as error:
            job.operation.end(classify_outcome(error))
            job.future.set_exception(error)
            return
        job.operation.end("completed")
        job.future.set_result(result)
