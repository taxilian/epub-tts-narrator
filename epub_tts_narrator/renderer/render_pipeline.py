"""Shared fair logical-clip scheduling for pooled rendering.

One bounded global executor runs logical clip transactions across all
chapter coordinators. Ready work is kept in one FIFO per
``(run/submission, chapter)`` owner and dispatched round-robin, one
transaction at a time, so a long chapter cannot starve later chapters.

A chapter occupies at most ``chapter_share(global_limit, chapters) =
max(1, floor(global_limit / configured_chapter_concurrency))`` logical
runners at once. The share comes from the *configured* chapter count, so
the reservation is not borrowed merely because a later chapter has not
registered work yet. Physical stage lanes may still borrow all unused
capacity; this executor only bounds logical transactions.

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
from threading import Condition, Lock, local

from .interrupts import stop_if_requested


def chapter_share(global_limit, chapter_concurrency):
    """Non-borrowable logical runners one chapter may occupy at once."""
    if type(global_limit) is not int or global_limit < 1:
        raise ValueError("global_limit must be a positive integer")
    if type(chapter_concurrency) is not int or chapter_concurrency < 1:
        raise ValueError("chapter_concurrency must be a positive integer")
    return max(1, global_limit // chapter_concurrency)


def _checked(function, args, kwargs):
    stop_if_requested()
    return function(*args, **kwargs)


class LogicalClipScheduler:
    """Globally bounded, chapter-fair executor for logical clip transactions."""

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
        self._cursor = 0
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

    def submit(self, owner, function, *args, **kwargs):
        """Queue one logical transaction for ``owner``; blocks while full."""
        if not callable(function):
            raise ValueError("function must be callable")
        context = copy_context()
        future = Future()
        with self._cond:
            if self._closed:
                raise RuntimeError("The logical clip scheduler is closed")
            while self._pending >= self._pending_limit:
                self._cond.wait()
                if self._closed:
                    raise RuntimeError("The logical clip scheduler is closed")
            entry = self._owners.get(owner)
            if entry is None:
                entry = {"queue": deque(), "active": 0}
                self._owners[owner] = entry
            entry["queue"].append((future, function, args, kwargs, context))
            self._pending += 1
            self._pump()
        return future

    def collect(self, futures):
        """Results in submission (source) order, not completion order."""
        return [future.result() for future in futures]

    def cancel_owner(self, owner):
        """Cancel every queued transaction of one owner; started work drains."""
        with self._cond:
            entry = self._owners.get(owner)
            cancelled = 0
            while entry is not None and entry["queue"]:
                future = entry["queue"].popleft()[0]
                self._pending -= 1
                if future.cancel():
                    cancelled += 1
            self._cond.notify_all()
        return cancelled

    def close(self, *, wait=True):
        """Cancel queued work, drain started transactions, stop submission."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            for entry in self._owners.values():
                while entry["queue"]:
                    self._pending -= 1
                    entry["queue"].popleft()[0].cancel()
            self._owners.clear()
            self._cond.notify_all()
        self._executor.shutdown(wait=wait)

    def _pump(self):
        # Caller holds self._cond. Round-robin: each pass serves at most one
        # transaction per owner, resuming after the last owner served.
        while self._inflight < self._limit:
            keys = list(self._owners)
            count = len(keys)
            if not count:
                return
            last_served = None
            for offset in range(count):
                index = (self._cursor + offset) % count
                entry = self._owners[keys[index]]
                if (entry["queue"] and entry["active"] < self._share
                        and self._inflight < self._limit):
                    self._dispatch(entry)
                    last_served = index
                    if self._inflight >= self._limit:
                        break
            if last_served is None:
                return
            self._cursor = (last_served + 1) % count

    def _dispatch(self, entry):
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
                future.set_result(context.run(_checked, function, args, kwargs))
            except BaseException as error:
                future.set_exception(error)
            finally:
                self._threads.logical = previous
                with self._cond:
                    entry["active"] -= 1
                    self._inflight -= 1
                    self._pump()
                    self._cond.notify_all()

        self._executor.submit(run)
