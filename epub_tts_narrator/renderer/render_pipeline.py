"""Shared FIFO logical-clip scheduling for pooled rendering.

One bounded global executor runs logical clip transactions across all
chapter coordinators in submission order. Owner metadata remains only for
cancellation and retirement; it never changes dispatch order. Pending
submissions also wait in FIFO order for capacity.

Chapter coordinators never run on this executor: it executes only
functions submitted through :meth:`LogicalClipScheduler.submit`, and
``on_logical_executor()`` lets a coordinator assert it is not running
clip-execution context. Every submission captures a fresh
``contextvars.copy_context()`` so scoped stop conditions travel to the
worker thread, and every dequeued transaction checks
``interrupts.stop_if_requested()`` before its body starts. Errors reach
the future unchanged; they never stop the scheduler. Queued work can be
cancelled per owner (a failed chapter's siblings) or all at once by
closing; already-started transactions always drain.

Task 6 consumes this scheduler in place of per-chapter executors; task 4
reads effective values from :mod:`render_pipeline_config`. Nothing here
touches engine/verifier identity, prompts, or acceptance policy.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from threading import Condition, local

from .interrupts import stop_if_requested


def chapter_share(global_limit, chapter_concurrency):
    """Non-borrowable logical runners one chapter may occupy at once."""
    if type(global_limit) is not int or global_limit < 1:
        raise ValueError("global_limit must be a positive integer")
    if type(chapter_concurrency) is not int or chapter_concurrency < 1:
        raise ValueError("chapter_concurrency must be a positive integer")
    return max(1, global_limit // chapter_concurrency)


class AdmissionClosed(RuntimeError):
    """The owner's chapter admission closed after a terminal transaction."""


def _checked(function, args, kwargs):
    stop_if_requested()
    return function(*args, **kwargs)


class _PriorityMarker:
    priority = True


class LogicalClipScheduler:
    """Globally bounded FIFO executor for logical clip transactions."""

    def __init__(self, global_limit, chapter_concurrency, *, pending_limit=None,
                 name="render-pipeline-logical"):
        self._share = chapter_share(global_limit, chapter_concurrency)
        self._limit = global_limit
        if pending_limit is None:
            pending_limit = 2 * global_limit
        if type(pending_limit) is not int or pending_limit < 1:
            raise ValueError("pending_limit must be a positive integer")
        self._pending_limit = pending_limit
        self._executor = ThreadPoolExecutor(max_workers=global_limit, thread_name_prefix=name)
        self._cond = Condition()
        self._owners = {}
        self._order = deque()
        # Owners a person asked to render again go to the front: their
        # transactions (still in their own order) dispatch before any other.
        self._priority_owners = set()
        self._priority_order = deque()
        self._submit_waiters = deque()
        self._inflight = 0
        self._pending = 0
        self._closed = False
        self._threads = local()

    @property
    def global_limit(self):
        return self._limit

    @property
    def share(self):
        return self._share

    def on_logical_executor(self):
        return getattr(self._threads, "logical", False)

    def set_priority(self, owner, high=True):
        """Put ``owner``'s transactions ahead of every other owner's (a retried chapter)."""
        with self._cond:
            if high:
                self._priority_owners.add(owner)
            else:
                self._priority_owners.discard(owner)
            self._cond.notify_all()

    def submit(self, owner, function, *args, **kwargs):
        """Queue one logical transaction for ``owner``; blocks while full."""
        if not callable(function):
            raise ValueError("function must be callable")
        context = copy_context()
        future = Future()
        marker = object()
        with self._cond:
            if owner in self._priority_owners:
                # A retried chapter also waits ahead of other submitters.
                position = sum(1 for waiter in self._submit_waiters if getattr(waiter, "priority", False))
                marker = _PriorityMarker()
                self._submit_waiters.insert(position, marker)
            else:
                self._submit_waiters.append(marker)
            try:
                while True:
                    if self._closed:
                        raise RuntimeError("The logical clip scheduler is closed")
                    if self._submit_waiters[0] is marker and self._pending < self._pending_limit:
                        self._submit_waiters.popleft()
                        self._cond.notify_all()
                        break
                    self._cond.wait()
                entry = self._owners.get(owner)
                if entry is not None and entry.get("closed"):
                    # A terminal transaction closed this chapter's admission; a
                    # fresh generation must retire the drained owner first.
                    raise AdmissionClosed(
                        "The logical clip admission for this owner is closed "
                        "after a terminal failure")
                if entry is None:
                    entry = {"queue": deque(), "active": 0}
                    self._owners[owner] = entry
                entry["queue"].append((future, function, args, kwargs, context))
                (self._priority_order if owner in self._priority_owners else self._order).append((owner, future))
                self._pending += 1
                self._pump()
            finally:
                if marker in self._submit_waiters:
                    self._submit_waiters.remove(marker)
                    self._cond.notify_all()
        return future

    def collect(self, futures):
        """Results in submission (source) order, not completion order."""
        return [future.result() for future in futures]

    def snapshot(self):
        """Read-only numeric scheduling gauges; no per-run history."""
        with self._cond:
            return {"pending": self._pending, "inflight": self._inflight,
                    "owners": len(self._owners), "global_limit": self._limit,
                    "chapter_share": self._share, "closed": self._closed}

    def cancel_owner(self, owner):
        """Cancel every queued transaction of one owner; started work drains.

        Future cancellation fires done-callbacks, so no scheduler lock is
        held while cancelling: queued items are popped under the condition
        and cancelled after it is released.
        """
        with self._cond:
            entry = self._owners.get(owner)
            cancelled, popped = 0, []
            while entry is not None and entry["queue"]:
                popped.append(entry["queue"].popleft()[0])
                self._pending -= 1
            self._cond.notify_all()
        for future in popped:
            if future.cancel():
                cancelled += 1
        return cancelled

    def retire_owner(self, owner):
        """Drop one owner's entry after its work fully drained.

        Bounded memory for a warm server: finished chapter owners leave no
        scheduling residue. The entry is removed only when its queue is
        empty and no transaction is active; retiring a closed entry is how
        a fresh scheduling generation reopens the owner.
        """
        with self._cond:
            entry = self._owners.get(owner)
            if entry is None or entry["queue"] or entry["active"]:
                return False
            del self._owners[owner]
            return True

    def close(self, *, wait=True):
        """Cancel queued work, drain started transactions, stop submission."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            popped = []
            for entry in self._owners.values():
                while entry["queue"]:
                    popped.append(entry["queue"].popleft()[0])
                    self._pending -= 1
            self._owners.clear()
            self._order.clear()
            self._priority_order.clear()
            self._cond.notify_all()
        for future in popped:
            future.cancel()
        self._executor.shutdown(wait=wait)

    def _pump(self):
        # Caller holds self._cond: dispatch globally in submission order,
        # prioritized (retried) owners first.
        while self._inflight < self._limit and (self._priority_order or self._order):
            owner, future = (self._priority_order or self._order).popleft()
            entry = self._owners.get(owner)
            if entry is None or not entry["queue"] or entry["queue"][0][0] is not future:
                continue
            self._dispatch(owner, entry)

    def _close_owner_admission_locked(self, entry):
        """Pop and return queued work of a closed owner. Caller holds _cond."""
        popped = []
        while entry["queue"]:
            popped.append(entry["queue"].popleft()[0])
            self._pending -= 1
        return popped

    def _dispatch(self, owner, entry):
        future, function, args, kwargs, context = entry["queue"].popleft()
        self._pending -= 1
        if not future.set_running_or_notify_cancel():
            return
        entry["active"] += 1
        self._inflight += 1

        def run():
            previous = getattr(self._threads, "logical", False)
            self._threads.logical = True
            try:
                result = context.run(_checked, function, args, kwargs)
            except BaseException as error:
                result = None
                # The chapter's admission closes at the failure instant:
                # queued siblings are popped before any pump can dispatch
                # them. Collectors treat their later cancellation as drain
                # outcome, never as the chapter's failure, so the original
                # error below is the one that surfaces.
                with self._cond:
                    entry["active"] -= 1
                    self._inflight -= 1
                    entry["closed"] = True
                    popped = self._close_owner_admission_locked(entry)
                    self._pump()
                for sibling in popped:
                    sibling.cancel()
                future.set_exception(error)
            else:
                with self._cond:
                    entry["active"] -= 1
                    self._inflight -= 1
                    self._pump()
                # Accounting precedes resolution, so a collector that saw
                # this future finish can immediately retire the owner.
                future.set_result(result)
            self._threads.logical = previous
            with self._cond:
                self._cond.notify_all()

        self._executor.submit(run)
