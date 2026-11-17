"""Pause a render's network work, for a laptop changing location.

While paused, no speech request (to any TTS worker) and no network review
request (JEV, the recovery model) starts; local work on takes already made
keeps going. Requests already running finish. One that fails because the
connection went away (no answer at all: a transport failure or a timeout,
never an HTTP answer) is sent again, unchanged, after resume instead of
counting as a failed take or a missing verifier. The same holds for a
connection failure within ``RESUME_GRACE_SECONDS`` of resuming, while the
network comes back (retried every ``RECONNECT_RETRY_SECONDS``).

A pause never changes a request, its seed or its identity: the request
sent again is the same request. Stopping the render still works while
paused (``interrupts.stop_if_requested``). The run shell's ``p`` toggles
``PAUSE``; it is process-wide, so it covers every book of a render queue.
"""

from __future__ import annotations

from threading import Condition
import time

RESUME_GRACE_SECONDS = 120.0
RECONNECT_RETRY_SECONDS = 5.0
_WAIT_SECONDS = 0.2


def network_failure(error):
    """Whether ``error`` is a request that got no answer (connection lost or timed out), never an HTTP answer."""
    if getattr(error, "status", None) is not None or getattr(error, "status_code", None) is not None:
        return False
    if getattr(error, "transport_error", False) is True:  # a TTS worker's FishHTTPError
        return True
    if getattr(error, "error_code", None) == "transport_error":  # JevError
        return True
    from ..provider import ProviderError
    # The provider reports transport and stalled-stream failures as retryable
    # with no status; a malformed answer is not retryable.
    return isinstance(error, ProviderError) and getattr(error, "retryable", False) is True


class PauseGate:
    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self._cond = Condition()
        self._paused = False
        self._epoch = 0  # counts pauses, so a request knows one began while it ran
        self._resumed_at = None
        self._clock = clock
        self._sleep = sleep
        self._listeners = []

    def paused(self):
        with self._cond:
            return self._paused

    def listen(self, callback):
        """Call ``callback(paused)`` on every change (from the thread making it)."""
        with self._cond:
            self._listeners.append(callback)

    def _changed(self, paused):
        with self._cond:
            listeners = list(self._listeners)
        for callback in listeners:
            callback(paused)

    def pause(self):
        with self._cond:
            if self._paused:
                return
            self._paused = True
            self._epoch += 1
        self._changed(True)

    def resume(self):
        with self._cond:
            if not self._paused:
                return
            self._paused = False
            self._resumed_at = self._clock()
            self._cond.notify_all()
        self._changed(False)

    def toggle(self):
        """Pause or resume; returns whether it is now paused."""
        if self.paused():
            self.resume()
            return False
        self.pause()
        return True

    def wait(self):
        """Return once not paused; a stop requested while paused still stops (never checked otherwise)."""
        from .interrupts import stop_if_requested
        with self._cond:
            while self._paused:
                self._cond.release()
                try:
                    stop_if_requested()
                finally:
                    self._cond.acquire()
                self._cond.wait(_WAIT_SECONDS)

    def _excused(self, epoch):
        """Whether a request started at ``epoch`` that lost its connection is the pause's doing."""
        with self._cond:
            if self._paused or self._epoch != epoch:
                return True
            return self._resumed_at is not None and self._clock() - self._resumed_at < RESUME_GRACE_SECONDS

    def call(self, function, *args, **kwargs):
        """``function(*args, **kwargs)`` once not paused, sent again when the pause lost its connection."""
        while True:
            self.wait()
            with self._cond:
                epoch = self._epoch
            try:
                return function(*args, **kwargs)
            except Exception as error:
                if not network_failure(error) or not self._excused(epoch):
                    raise
                if not self.paused():
                    self._sleep(RECONNECT_RETRY_SECONDS)  # just resumed: the network may still be coming back


PAUSE = PauseGate()
