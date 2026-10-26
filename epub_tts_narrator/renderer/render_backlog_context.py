"""Runtime acquisition and context transfer of attempt backlog tickets.

The context-side half of :mod:`render_backlog`: ``begin_attempt_ticket``
acquires from the pipeline backlog scoped by ``render_resource_context``
(None and the legacy direct path outside one; adopt mode also None when a
ticket is already held, so an ASR boundary never double-counts), and
``AttemptTicket`` carries the granted ticket plus its context binding so the
acquiring thread's outer primary-ASR ``release`` resets exactly this
attempt's binding. Blocked waits wake on render stop or runtime close,
surfacing the canonical interrupt. Everything here is payload-free; the
module keeps no history and grows no registry.

``render_backlog`` re-exports ``begin_attempt_ticket``, ``AttemptTicket``,
and ``current_ticket`` so existing import sites stay unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar


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
    from .render_backlog import BacklogCancelled
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
