"""A serial background job queue for terminal UIs (stdlib only, UI-independent).

Interactive screens hand slow work (voice generation, verification, saves that
take a lock) to one worker thread so the screen stays responsive. Jobs run one
at a time in submission order, which also serializes every write the work
makes. A job may instead name a *lane* (``add_lane``): consecutive jobs of one
lane run together, up to the lane's limit, and never beside any other job; the
lane's ``opening`` runs before its first job and ``closing`` after its last, so
the lane can hold a shared resource (a lock, a pool of workers) while it runs.
An *independent* lane instead runs its jobs one at a time on a worker of its
own, beside everything else (quick saves that need not wait for long work).
Each job records its state, timing, and a sanitized error, so a screen
can list failures and retry one while later jobs keep running. A job never
decides anything on a person's behalf: it only runs what was submitted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import itertools
import threading
import time
from typing import Any, Callable


QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"


@dataclass
class Job:
    """One unit of background work and its outcome."""

    job_id: int
    label: str
    run: Callable[[], Any]
    scope: dict = field(default_factory=dict)
    state: str = QUEUED
    submitted: float = 0.0
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str | None = None
    error_type: str | None = None
    retry_of: int | None = None
    lane: str | None = None

    @property
    def active(self):
        return self.state in (QUEUED, RUNNING)

    def elapsed(self, now):
        if self.started is None:
            return 0.0
        return (self.finished if self.finished is not None else now) - self.started


class JobQueue:
    """Run submitted jobs one at a time on a daemon worker thread.

    ``on_change(job)`` is called from the worker thread (and from the
    submitting thread for queue changes) after every state change; UI code
    must marshal it onto its own thread. ``describe_error`` turns an exception
    into display text, so callers can strip secrets before anything is shown.
    """

    def __init__(self, *, on_change=None, describe_error=None, clock=time.monotonic, history=200):
        self._on_change = on_change or (lambda job: None)
        self._describe = describe_error or (lambda error: str(error) or type(error).__name__)
        self._clock = clock
        self._ids = itertools.count(1)
        self._lock = threading.Condition()
        self._pending: deque[Job] = deque()
        self._jobs: dict[int, Job] = {}
        self._order: deque[int] = deque()
        self._history = history
        self._closed = False
        self._thread = None
        self._lanes = {}
        self._open_lane = None  # the lane whose ``opening`` ran and whose ``closing`` has not
        self._lane_running = 0
        self._side = {}  # independent lane -> its own pending jobs
        self._side_threads = {}

    def add_independent_lane(self, name):
        """Let jobs submitted with ``lane=name`` run in order on their own worker, never waiting for other jobs."""
        with self._lock:
            self._side.setdefault(name, deque())

    def add_lane(self, name, limit, *, opening=None, closing=None):
        """Let jobs submitted with ``lane=name`` run up to ``limit`` at once.

        ``opening()`` runs on the queue's worker before the lane's first job of
        a run of consecutive lane jobs, and ``closing()`` after the last one
        finishes. A failed ``opening`` fails the job waiting to start, and a
        failed ``closing`` is recorded as a failed job of its own.
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("A lane needs a limit of at least 1")
        with self._lock:
            self._lanes[name] = (limit, opening or (lambda: None), closing or (lambda: None))

    def submit(self, label, run, *, scope=None, retry_of=None, lane=None):
        if lane is not None and lane not in self._lanes and lane not in self._side:
            raise ValueError(f"Unknown job lane: {lane}")
        job = Job(next(self._ids), label, run, dict(scope or {}), submitted=self._clock(), retry_of=retry_of,
                  lane=lane)
        with self._lock:
            if self._closed:
                raise RuntimeError("The job queue is closed")
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._trim()
            if lane in self._side:
                self._side[lane].append(job)
                if self._side_threads.get(lane) is None:
                    self._side_threads[lane] = threading.Thread(target=self._side_work, args=(lane,),
                                                                name=f"tui-jobs-{lane}", daemon=True)
                    self._side_threads[lane].start()
                self._lock.notify_all()
                pending_main = False
            else:
                self._pending.append(job)
                pending_main = True
            if pending_main and self._thread is None:
                self._thread = threading.Thread(target=self._work, name="tui-jobs", daemon=True)
                self._thread.start()
            self._lock.notify_all()
        self._changed(job)
        return job

    def retry(self, job_id):
        """Submit a failed or cancelled job's work again as a new job."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.state not in (FAILED, CANCELLED) or job.run is None:
            raise ValueError("Only a failed or cancelled job can be retried")
        return self.submit(job.label, job.run, scope=job.scope, retry_of=job.job_id, lane=job.lane)

    def cancel_queued(self, predicate=lambda job: True):
        """Cancel jobs that have not started; a running job always finishes."""
        cancelled = []
        with self._lock:
            for pending in (self._pending, *self._side.values()):
                for job in list(pending):
                    if predicate(job):
                        pending.remove(job)
                        job.state, job.finished = CANCELLED, self._clock()
                        cancelled.append(job)
        for job in cancelled:
            self._changed(job)
        return cancelled

    def jobs(self):
        """Snapshot of known jobs, oldest first (finished ones beyond history dropped)."""
        with self._lock:
            return [self._jobs[job_id] for job_id in self._order]

    def running(self):
        with self._lock:
            return next((self._jobs[job_id] for job_id in self._order
                         if self._jobs[job_id].state == RUNNING), None)

    def active_scopes(self, key):
        """``{value: state}`` for queued/running jobs' ``scope[key]`` (running wins)."""
        result = {}
        with self._lock:
            for job_id in self._order:
                job = self._jobs[job_id]
                if job.active:
                    for value in _scope_values(job.scope.get(key)):
                        if result.get(value) != RUNNING:
                            result[value] = job.state
        return result

    def failures(self):
        """Failed jobs not yet superseded by a later retry of the same work."""
        jobs = self.jobs()
        retried = {job.retry_of for job in jobs if job.retry_of is not None}
        return [job for job in jobs if job.state == FAILED and job.job_id not in retried]

    def wait_idle(self, timeout=None):
        """Block until nothing is queued or running (for tests and orderly exit)."""
        deadline = None if timeout is None else self._clock() + timeout
        with self._lock:
            while (self._pending or self._open_lane is not None or any(self._side.values())
                   or any(self._jobs[job_id].state == RUNNING for job_id in self._order)):
                remaining = None if deadline is None else deadline - self._clock()
                if remaining is not None and remaining <= 0:
                    return False
                self._lock.wait(remaining)
            return True

    def close(self):
        """Cancel queued jobs and stop accepting new ones; a running job finishes."""
        self.cancel_queued()
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    def _trim(self):
        while len(self._order) > self._history:
            oldest = self._jobs[self._order[0]]
            if oldest.active:
                return
            self._order.popleft()
            del self._jobs[oldest.job_id]

    def _changed(self, job):
        try:
            self._on_change(job)
        except Exception:
            # A closing screen must not stop the work or lose its outcome.
            pass

    def _work(self):
        while True:
            with self._lock:
                action, job = self._next()
                if action == "stop":
                    return
                if action in ("run", "lane"):
                    self._pending.popleft()
                    job.state, job.started = RUNNING, self._clock()
                    if action == "lane":
                        self._lane_running += 1
            if action == "open":
                self._open(job)
            elif action == "close":
                self._close_lane()
            elif action == "lane":
                self._changed(job)
                threading.Thread(target=self._run_lane_job, args=(job,), name=f"tui-jobs-{job.lane}",
                                 daemon=True).start()
            else:
                self._changed(job)
                if not self._execute(job):
                    # Interruption and exit still stop the worker after recording;
                    # a later submission starts a fresh worker.
                    with self._lock:
                        self._thread = None
                    return

    def _next(self):
        """What the worker does next (called holding the lock): wait until there is something."""
        while True:
            head = self._pending[0] if self._pending else None
            if self._open_lane is not None and (head is None or head.lane != self._open_lane):
                if self._lane_running:
                    self._lock.wait()
                    continue
                return "close", None
            if head is None:
                if self._closed:
                    return "stop", None
                self._lock.wait()
                continue
            if head.lane is None:
                return "run", head
            if self._open_lane is None:
                return "open", head
            if self._lane_running < self._lanes[head.lane][0]:
                return "lane", head
            self._lock.wait()

    def _open(self, job):
        opening = self._lanes[job.lane][1]
        try:
            opening()
        except BaseException as error:  # noqa: BLE001 - the waiting job carries the failure
            with self._lock:
                if self._pending and self._pending[0] is job:
                    self._pending.popleft()
                job.started = job.started or self._clock()
                job.state, job.finished = FAILED, self._clock()
                job.error, job.error_type = self._describe(error), type(error).__name__
                self._lock.notify_all()
            self._changed(job)
            return
        with self._lock:
            self._open_lane = job.lane
            self._lock.notify_all()

    def _close_lane(self):
        with self._lock:
            name = self._open_lane
        try:
            self._lanes[name][2]()
        except BaseException as error:  # noqa: BLE001 - recorded as its own failed job
            now = self._clock()
            failed = Job(next(self._ids), f"Finish {name}", None, submitted=now, state=FAILED, started=now,
                         finished=now, error=self._describe(error), error_type=type(error).__name__)
            with self._lock:
                self._jobs[failed.job_id] = failed
                self._order.append(failed.job_id)
                self._trim()
            self._changed(failed)
        with self._lock:
            self._open_lane = None
            self._lock.notify_all()

    def _side_work(self, lane):
        while True:
            with self._lock:
                while not self._side[lane] and not self._closed:
                    self._lock.wait()
                if not self._side[lane]:
                    self._side_threads[lane] = None
                    return
                job = self._side[lane].popleft()
                job.state, job.started = RUNNING, self._clock()
            self._changed(job)
            if not self._execute(job):
                with self._lock:
                    self._side_threads[lane] = None
                return

    def _run_lane_job(self, job):
        try:
            self._execute(job)
        finally:
            with self._lock:
                self._lane_running -= 1
                self._lock.notify_all()

    def _execute(self, job):
        """Run one job and record its outcome; False when it was interrupted."""
        try:
            result = job.run()
        except BaseException as error:  # noqa: BLE001 - every outcome is recorded
            with self._lock:
                job.state, job.finished = FAILED, self._clock()
                job.error, job.error_type = self._describe(error), type(error).__name__
                self._lock.notify_all()
            self._changed(job)
            return isinstance(error, Exception)
        with self._lock:
            job.state, job.finished, job.result = DONE, self._clock(), result
            self._lock.notify_all()
        self._changed(job)
        return True


def _scope_values(value):
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)
