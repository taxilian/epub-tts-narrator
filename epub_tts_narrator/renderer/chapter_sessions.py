"""Ephemeral connected-client rendering sessions.

A session is a server-side memory lease that proves a rendering client is
actively connected right now. Only an explicitly opened, live session can
grant execution: queued work without one stays durable but paused. Restarts
clear every session, so server startup never resumes another client's book.
Observation endpoints never create, renew, or require a session, and session
identifiers are hashes that must never enter logs.
"""

from __future__ import annotations

import threading
import time

from .chapter_protocol import require_hash


def require_session_id(value):
    return require_hash(value, "Client session")


class ClientSessions:
    """Thread-safe in-memory session registry with injectable monotonic time.

    Heartbeats renew only an already-live session; they never recreate an
    expired or revoked one. Expired entries are pruned lazily, so the registry
    stays bounded without a background timer thread.
    """

    def __init__(self, ttl=30.0, *, clock=None):
        if type(ttl) not in (int, float) or not 0 < ttl <= 3600:
            raise ValueError("Client session TTL must be a positive number of at most 3600 seconds")
        self._ttl = float(ttl)
        self._clock = time.monotonic if clock is None else clock
        self._lock = threading.Lock()
        self._deadlines = {}

    @property
    def ttl(self):
        return self._ttl

    def open(self, session_id):
        """Explicitly open (or reopen) one client's lease; nothing else may."""
        require_session_id(session_id)
        with self._lock:
            now = self._clock()
            for key in [key for key, deadline in self._deadlines.items() if deadline <= now]:
                del self._deadlines[key]
            self._deadlines[session_id] = now + self._ttl

    def heartbeat(self, session_id):
        """Renew only an existing live session; an expired one stays closed."""
        require_session_id(session_id)
        with self._lock:
            deadline = self._deadlines.get(session_id)
            if deadline is None or deadline <= self._clock():
                self._deadlines.pop(session_id, None)
                raise KeyError("Unknown or expired client session; open a new session")
            self._deadlines[session_id] = self._clock() + self._ttl

    def revoke(self, session_id):
        """Drop the lease immediately; active grants observe the loss."""
        require_session_id(session_id)
        with self._lock:
            return self._deadlines.pop(session_id, None) is not None

    def live(self, session_id):
        if not isinstance(session_id, str):
            return False
        with self._lock:
            deadline = self._deadlines.get(session_id)
            return deadline is not None and deadline > self._clock()

    def any_live(self):
        with self._lock:
            now = self._clock()
            return any(deadline > now for deadline in self._deadlines.values())
