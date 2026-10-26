"""Attempt-scoped backpressure for untranscribed rendered audio.

A ``BacklogBudget`` bounds how many take attempts may hold untranscribed
audio at once (active generations, saved-but-unverified takes, and joined
candidates share one exact pending bound) and how many materialized bytes
they may account. New generation claims stop at or above the soft byte
high-water mark (256 MiB default) and resume only below half of it;
already-running outputs may land and overshoot, and neither byte accounting
nor release ever blocks a worker handing off a completed file.
``reserve(new_generation=False)`` accounts a joined-audio ASR check or a
resumed saved-but-unverified take without spending a generation start or
waiting on the high-water. Cached-accepted work reserves nothing.

Tickets record absolute sizes via ``materialize`` / ``update_bytes``;
``release`` returns the slot and bytes idempotently from any thread. Call
it at primary-ASR return or any earlier take exit — always before JEV,
planners, reviews, child recovery, or split/join recursion; the outer
finally releases the captured ticket even when the wave file stays on
disk. ``cancel``/``phase_ok`` are cheap caller predicates rechecked before
waiting and on every wakeup; they run under the budget lock. ``close()``
wakes every waiter with ``BacklogClosed``; granted tickets release
normally. ``snapshot()`` reports fixed numeric gauges only, never paths,
text, secrets, labels, or history.

Pipeline glue: ``begin_attempt_ticket(new_generation, initial_bytes=0)``
acquires from the runtime scoped by ``render_resource_context`` (None and
the legacy direct path outside one; adopt mode also None when a ticket is
already held, so an ASR boundary never double-counts). Blocked waits wake
on render stop or runtime close, surfacing the canonical interrupt. The
reservation context is single-use; its ContextVar token resets before the
outer finally releases the captured ticket, so a nested attempt can never
release another attempt's ticket. No file locks and no retry/identity
policy: reservation order relative to take locks is the integrator's
concern. Full hook map: ``.omo/evidence/.../task-5-api.md``.
"""

from __future__ import annotations

from contextvars import ContextVar
from threading import Condition

DEFAULT_MAX_MATERIALIZED_BYTES = 256 * 1024 * 1024
_WAKE_POLL_SECONDS = 0.05  # bound for noticing cancel/phase predicate flips


class BacklogError(RuntimeError):
    """Base class for render_backlog contract errors."""


class BacklogClosed(BacklogError):
    """The budget was closed before or while waiting for a ticket."""


class BacklogCancelled(BacklogError):
    """A caller cancel or phase predicate fired before a ticket was granted."""

    def __init__(self, reason):
        super().__init__(f"The backlog reservation was abandoned: {reason}")
        self.reason = reason


class BacklogReleased(BacklogError):
    """The ticket was already released; only release may run again."""


class BacklogTicket:
    """One attempt's counted reservation; carries no payload or history."""

    __slots__ = ("_budget", "_bytes", "_released")

    def __init__(self, budget, nbytes=0):
        self._budget = budget
        self._bytes = nbytes
        self._released = False

    @property
    def accounted_bytes(self):
        return self._bytes

    @property
    def released(self):
        return self._released

    def materialize(self, nbytes):
        """Record the attempt's absolute materialized size in bytes."""
        self._account(nbytes)

    def update_bytes(self, nbytes):
        """Re-record the absolute size after a join, trim, or rewrite."""
        self._account(nbytes)

    def _account(self, nbytes):
        if not isinstance(nbytes, int) or nbytes < 0:
            raise ValueError("nbytes must be a nonnegative integer")
        with self._budget._cond:
            if self._released:
                raise BacklogReleased("The ticket was already released")
            delta = nbytes - self._bytes
            if delta:
                self._bytes = nbytes
                self._budget._apply_bytes(delta)

    def release(self):
        """Return the slot and its accounted bytes; idempotent, any thread."""
        with self._budget._cond:
            if self._released:
                return
            self._released = True
            self._budget._pending -= 1
            self._budget._apply_bytes(-self._bytes)
            self._bytes = 0
            self._budget._cond.notify_all()


class BacklogBudget:
    """Exact pending-take bound with soft byte high-water and hysteresis."""

    def __init__(self, *, max_pending,
                 max_materialized_bytes=DEFAULT_MAX_MATERIALIZED_BYTES):
        if not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if (not isinstance(max_materialized_bytes, int)
                or max_materialized_bytes < 1):
            raise ValueError(
                "max_materialized_bytes must be a positive integer")
        self._max_pending = max_pending
        self._max_bytes = max_materialized_bytes
        self._low_water = max(1, max_materialized_bytes // 2)
        self._cond = Condition()
        self._pending = 0
        self._bytes = 0
        self._blocked = False
        self._closed = False
        self._waiters = 0

    def reserve(self, *, new_generation=True, initial_bytes=0, cancel=None,
                phase_ok=None):
        """Return a single-use context manager that yields the ticket."""
        return _Reservation(self, new_generation, initial_bytes, cancel,
                            phase_ok)

    def acquire(self, *, new_generation=True, initial_bytes=0, cancel=None,
                phase_ok=None):
        """Grant one ticket immediately for an explicit-transfer caller."""
        return self._grant(new_generation=new_generation,
                           initial_bytes=initial_bytes,
                           cancel=cancel, phase_ok=phase_ok)

    def snapshot(self):
        """Payload-free numeric gauges for progress and tests."""
        with self._cond:
            return {
                "pending_count": self._pending,
                "materialized_bytes": self._bytes,
                "max_pending": self._max_pending,
                "max_materialized_bytes": self._max_bytes,
                "low_water_bytes": self._low_water,
                "new_generation_blocked": self._blocked,
                "waiters": self._waiters,
                "closed": self._closed,
            }

    def close(self):
        """Wake all waiters with BacklogClosed; idempotent."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _apply_bytes(self, delta):
        # The budget lock is held by every caller.
        self._bytes += delta
        if self._bytes >= self._max_bytes:
            self._blocked = True
        elif self._blocked and self._bytes < self._low_water:
            self._blocked = False
            self._cond.notify_all()

    def _grant(self, *, new_generation, initial_bytes, cancel, phase_ok):
        if not isinstance(initial_bytes, int) or initial_bytes < 0:
            raise ValueError("initial_bytes must be a nonnegative integer")
        with self._cond:
            while True:
                if self._closed:
                    raise BacklogClosed("The backlog budget is closed")
                if cancel is not None and cancel():
                    raise BacklogCancelled("cancel")
                if phase_ok is not None and not phase_ok():
                    raise BacklogCancelled("phase")
                if (self._pending < self._max_pending
                        and not (new_generation and self._blocked)):
                    self._pending += 1
                    self._apply_bytes(initial_bytes)
                    return BacklogTicket(self, initial_bytes)
                self._waiters += 1
                try:
                    self._cond.wait(_WAKE_POLL_SECONDS)
                finally:
                    self._waiters -= 1


class _Reservation:
    """Single-use context manager granting one attempt ticket."""

    __slots__ = ("_budget", "_new_generation", "_initial_bytes", "_cancel",
                 "_phase_ok", "_ticket", "_token", "_entered")

    def __init__(self, budget, new_generation, initial_bytes, cancel,
                 phase_ok):
        self._budget = budget
        self._new_generation = new_generation
        self._initial_bytes = initial_bytes
        self._cancel = cancel
        self._phase_ok = phase_ok
        self._ticket = None
        self._token = None
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise BacklogError("A reservation context is single-use")
        self._entered = True
        self._ticket = self._budget._grant(
            new_generation=self._new_generation,
            initial_bytes=self._initial_bytes,
            cancel=self._cancel,
            phase_ok=self._phase_ok,
        )
        self._token = _current_ticket.set(self._ticket)
        return self._ticket

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            token, self._token = self._token, None
            _current_ticket.reset(token)
        # Captured-ticket outer finally: releases exactly this attempt's
        # ticket, never one from a nested attempt. Idempotent after an
        # explicit primary-ASR release.
        self._ticket.release()
        return False


_current_ticket: ContextVar = ContextVar(
    "render_backlog_current_ticket", default=None)


def current_ticket():
    """The innermost live ticket in this context, or None (read-only)."""
    return _current_ticket.get()


def begin_attempt_ticket(new_generation, initial_bytes=0):
    """Acquire one attempt ticket from the pipeline backlog in scope.

    Returns None outside a pipeline transaction: the caller keeps the
    legacy direct path with no budget. Adopt mode (``new_generation=False``)
    also returns None when this context already holds an attempt ticket;
    the held ticket already accounts this operation's bytes. A fresh
    generation always acquires its own per-attempt ticket, nested safely.

    A blocked wait wakes when the render stop controller requests a stop
    or the runtime closes; a stop raises the canonical interrupt from
    ``interrupts`` instead of ``BacklogCancelled``.
    """
    from .interrupts import stop_if_requested, stop_requested
    from .render_resource_context import current_resources

    resources = current_resources()
    if resources is None:
        return None
    if not new_generation and _current_ticket.get() is not None:
        return None
    try:
        ticket = resources.backlog.acquire(
            new_generation=new_generation, initial_bytes=initial_bytes,
            cancel=stop_requested)
    except BacklogCancelled:
        stop_if_requested()
        raise
    return AttemptTicket(ticket, _current_ticket.set(ticket))


class AttemptTicket:
    """Explicit transfer of one granted ticket plus its context binding.

    The token resets in ``release`` on the acquiring thread (the caller's
    outer primary-ASR finally); ticket release stays idempotent, any thread.
    """

    __slots__ = ("ticket", "_token")

    def __init__(self, ticket, token):
        self.ticket = ticket
        self._token = token

    def materialize(self, nbytes):
        """Record the attempt's absolute materialized size in bytes."""
        self.ticket.materialize(nbytes)

    def update_bytes(self, nbytes):
        """Re-record the absolute size after a join, trim, or rewrite."""
        self.ticket.update_bytes(nbytes)

    @property
    def released(self):
        return self.ticket.released

    def release(self):
        token, self._token = self._token, None
        if token is not None:
            _current_ticket.reset(token)
        self.ticket.release()
