"""Shared fair chapter orchestration for pipeline-scheduled rendering.

One ``LogicalClipScheduler`` per pipeline ``RenderPool`` runs every chapter's
logical clip transactions. This module owns the two coordination shapes built
on it:

* :func:`collect_chapter` — one chapter coordinator: bounded submissions (the
  chapter's non-borrowable share), source-ordered results, and fail-fast
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


def collect_chapter(scheduler, owner, count, run_one):
    """Run one chapter's transactions on the shared logical scheduler.

    Returns results in source order. At most ``scheduler.share``
    transactions are outstanding for this owner. The first terminal
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
    share = max(1, scheduler.share)
    futures = {}
    results = [None] * count
    finished = deque()
    finished_now = Condition()
    next_index = 0
    failure = None

    def on_done(future):
        with finished_now:
            finished.append(future)
            finished_now.notify_all()

    def submit_more():
        nonlocal next_index
        while next_index < count and len(futures) < share:
            try:
                future = scheduler.submit(owner, run_one, next_index)
            except AdmissionClosed:
                return  # a terminal failure closed the chapter; drain below
            futures[future] = next_index
            future.add_done_callback(on_done)
            next_index += 1

    def collect_finished():
        nonlocal failure
        with finished_now:
            while not finished:
                finished_now.wait()
            batch = list(finished)
            finished.clear()
        for future in batch:
            index = futures.pop(future)
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


def run_units(units, limit, admission):
    """Run chapter units on bounded coordinator threads, in order.

    ``units`` are index-callables in manifest order. At most ``limit``
    coordinators run; each pulls the next unit only when admission is still
    open. The first terminal failure closes admission: no further unit
    starts, running units drain, and the first failure is re-raised after
    every coordinator finishes.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Coordinator limit must be a positive integer")
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
