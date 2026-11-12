"""Shared FIFO chapter orchestration for pipeline-scheduled rendering.

One ``LogicalClipScheduler`` per pipeline ``RenderPool`` runs every chapter's
logical clip transactions. This module owns the two coordination shapes built
on it:

* :func:`collect_chapter` — one chapter coordinator: bounded submissions (the
  chapter's bounded outstanding work), source-ordered results, and fail-fast
  admission on the first terminal clip failure (queued siblings cancelled
  before any new transaction starts, started ones drained).
* :class:`ChapterRunAdmission` + :func:`run_units` — book-level bounded
  chapter coordinators: chapters start in manifest order up to the configured
  concurrency, the first terminal failure (or stop) closes the run's
  admission, started chapters drain, and the first error is re-raised.

Chapter coordinators run on caller threads; only per-chunk transactions run
on the logical executor. Paragraph-fallback replans simply call
:func:`collect_chapter` again for the same owner — a fresh scheduling
generation with no cancelled residue, and :func:`retire_owner` drops the
owner entry once a chapter finishes so a warm server accumulates no
per-run scheduling history.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError
from threading import Condition, Lock

from .render_pipeline import AdmissionClosed


class ChapterRunAdmission:
    """Fail-fast admission gate shared by a render's chapter coordinators."""

    def __init__(self):
        self._lock = Lock()
        self._failure = None

    def enter(self):
        """Raise the first recorded failure once admission has closed."""
        with self._lock:
            if self._failure is not None:
                raise self._failure

    def fail(self, error):
        """Record the first terminal failure and close admission."""
        with self._lock:
            if self._failure is None:
                self._failure = error

    @property
    def failure(self):
        with self._lock:
            return self._failure


def collect_chapter(scheduler, owner, count, run_one, *, sequential=False, dependencies=None):
    """Run one chapter's transactions on the shared logical scheduler.

    Returns results in source order. At most ``scheduler.share``
    transactions are outstanding for this owner. With ``dependencies``
    (for each index, the earlier indices it must wait for), a transaction is
    submitted once all of its dependencies have finished, earliest first;
    independent transactions run concurrently. The first terminal
    failure closes the chapter's admission: no further submissions, queued
    siblings cancelled, already-started transactions drained, and the
    original failure re-raised after the drain.

    Completion is collected through per-future done-callbacks, never
    ``concurrent.futures.wait``: a queued future that the scheduler
    cancels becomes ``CANCELLED`` on this CPython without waking wait()
    registrants, which would deadlock the coordinator forever. A mid-loop
    ``AdmissionClosed`` refusal stops submissions so the drain below can
    retire the owner and surface the original failure.
    """
    if dependencies is not None and len(dependencies) != count:
        raise ValueError("collect_chapter needs one dependency set per transaction")
    share = 1 if sequential and dependencies is None else max(1, scheduler.share)
    futures = {}
    results = [None] * count
    finished = deque()
    finished_now = Condition()
    next_index = 0
    failure = None
    done = set()
    waiting = list(range(count))

    def on_done(future):
        with finished_now:
            finished.append(future)
            finished_now.notify_all()

    def submit(index):
        try:
            future = scheduler.submit(owner, run_one, index)
        except AdmissionClosed:
            return False  # a terminal failure closed the chapter; drain below
        futures[future] = index
        future.add_done_callback(on_done)
        return True

    def submit_more():
        nonlocal next_index
        if dependencies is None:
            while next_index < count and len(futures) < share:
                if not submit(next_index):
                    return
                next_index += 1
            return
        for index in list(waiting):
            if len(futures) >= share:
                return
            if dependencies[index] <= done:
                if not submit(index):
                    return
                waiting.remove(index)

    def collect_finished():
        nonlocal failure
        with finished_now:
            while not finished:
                finished_now.wait()
            batch = list(finished)
            finished.clear()
        for future in batch:
            index = futures.pop(future)
            done.add(index)
            try:
                results[index] = future.result()
            except BaseException as error:  # noqa: BLE001 - first failure wins
                # Admission-close cancellations are expected drain outcomes,
                # never the chapter's failure.
                if future.cancelled() and isinstance(error, CancelledError):
                    continue
                if failure is None:
                    failure = error

    submit_more()
    while futures and failure is None:
        collect_finished()
        if failure is None:
            submit_more()
    if failure is None:
        return results
    # Close the chapter's admission: cancel queued siblings, drain every
    # outstanding future (started transactions and cancellations), then
    # surface the first failure. The drained owner is retired so a
    # paragraph-fallback replan opens a fresh scheduling generation.
    scheduler.cancel_owner(owner)
    while futures:
        collect_finished()
    scheduler.retire_owner(owner)
    raise failure


def retire_owner(scheduler, owner):
    """Drop the owner's scheduling entry once its work fully drained."""
    return scheduler.retire_owner(owner)


def run_units(units, limit, admission, *, coordinator_pool=None, priority=0, on_submitted=None):
    """Run chapter units on bounded coordinator threads, in order.

    ``units`` are index-callables in manifest order. At most ``limit``
    coordinators run; each pulls the next unit only when admission is still
    open. The first terminal failure closes admission: no further unit
    starts, running units drain, and the first failure is re-raised after
    every coordinator finishes.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Coordinator limit must be a positive integer")
    if coordinator_pool is not None:
        return coordinator_pool.run(units, admission, priority=priority, on_submitted=on_submitted)
    results = [None] * len(units)
    if limit == 1 or len(units) <= 1:
        for index, unit in enumerate(units):
            admission.enter()
            try:
                results[index] = unit(index)
            except BaseException as error:
                admission.fail(error)
                raise
        return results
    from threading import Thread

    cursor = iter(range(len(units)))
    cursor_lock = Lock()

    def coordinator():
        while True:
            with cursor_lock:
                index = next(cursor, None)
            if index is None:
                return
            try:
                admission.enter()
            except BaseException:
                return
            try:
                results[index] = units[index](index)
            except BaseException as error:
                admission.fail(error)

    threads = [Thread(target=coordinator, name=f"chapter-coordinator-{number + 1}")
               for number in range(min(limit, len(units)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failure = admission.failure
    if failure is not None:
        raise failure
    return results


class ChapterRetries:
    """Failed chapters a person asked to render again during this run.

    A run shell offers each chapter-scoped failure a ``request`` callback;
    one retry worker takes requests in order and re-runs those chapters
    while the other chapters keep rendering. After the main pass a held run
    waits in ``wait_for_request`` until the person retries more chapters or
    releases it. Plain runs never create requests, so nothing here runs.
    """

    def __init__(self):
        self._condition = Condition()
        self._queue = deque()
        self._closed = False
        self._released = False
        self._busy = 0

    def request(self, index):
        """Queue one chapter (manifest index); False once the run stopped taking retries."""
        with self._condition:
            if self._closed:
                return False
            if index not in self._queue:
                self._queue.append(index)
            self._condition.notify_all()
            return True

    def take(self):
        """The next requested chapter, or None once closed and empty (marks the worker busy)."""
        with self._condition:
            while not self._queue and not self._closed:
                self._condition.wait()
            if not self._queue:
                return None
            self._busy += 1
            return self._queue.popleft()

    def peek(self):
        """The next requested unit without taking it, or None."""
        with self._condition:
            return self._queue[0] if self._queue else None

    def take_nowait(self):
        """The next requested chapter without waiting, or None (marks the caller busy when one is returned)."""
        with self._condition:
            if not self._queue:
                return None
            self._busy += 1
            return self._queue.popleft()

    def done(self):
        with self._condition:
            self._busy -= 1
            self._condition.notify_all()

    def idle(self):
        with self._condition:
            return not self._queue and not self._busy

    def wait_idle(self, *, stop_check=None, poll=0.2):
        """Block until no retry is queued or running; ``stop_check`` may raise to stop."""
        with self._condition:
            while self._queue or self._busy:
                self._condition.wait(poll)
                if stop_check is not None:
                    self._condition.release()
                    try:
                        stop_check()
                    finally:
                        self._condition.acquire()

    def release(self):
        """The person is done retrying: let the held run finish."""
        with self._condition:
            self._released = True
            self._condition.notify_all()

    def wait_for_request(self, *, stop_check=None, poll=0.2):
        """Block until a retry is requested (True) or the run is released (False)."""
        with self._condition:
            while not self._queue and not self._busy and not self._released and not self._closed:
                self._condition.wait(poll)
                if stop_check is not None:
                    self._condition.release()
                    try:
                        stop_check()
                    finally:
                        self._condition.acquire()
            return bool(self._queue or self._busy) and not self._released

    def close(self):
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()
