"""Finish active cache transactions on Ctrl-C; a second Ctrl-C exits promptly.

Only the rendering command installs this controller. The shared state is visible
to its worker threads, but only the main thread replaces Python's SIGINT handler.
Safe task boundaries, not the signal handler, decide when a graceful stop raises.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import os
import signal
import subprocess
import threading


class RenderInterrupted(KeyboardInterrupt):
    """Rendering stopped at a boundary after active cache writes completed."""


class RenderPaused(Exception):
    """A scoped stop fired at a durable boundary; active work drains losslessly.

    The chapter service raises this when a job's connected client disappeared:
    the job returns to the queue with its budget, cache, and history intact,
    never marked failed or cancelled.
    """


_active: RenderInterrupts | None = None

# Scoped stop conditions carry owner-loss stops into worker threads. Executor
# submissions propagate them with contextvars.copy_context(); a copied context
# is never reused concurrently and each future receives its own fresh copy.
_scoped_stop: contextvars.ContextVar = contextvars.ContextVar("renderer_scoped_stop", default=None)


class RenderInterrupts:
    """Temporarily handle render interruption without interrupting active takes.

    ``force_exit`` is injectable for tests. Production uses ``os._exit`` so a
    blocked worker, executor shutdown, or model destructor cannot delay the
    second interrupt. Tracked child process groups are killed first.
    """

    def __init__(self, notify=None, *, force_exit=None):
        self.requested = False
        self.forced = False
        self._notify = notify
        self._exit = os._exit if force_exit is None else force_exit
        self._installed = False
        self._exiting = False
        self._previous = None
        self._previous_handler = None
        self._children = {}
        self._defer_count = 0
        self._defer_lock = threading.Lock()

    def __enter__(self):
        global _active
        # Library callers may render on a background thread. Signals belong to
        # the process's main thread; do not alter another command's controller.
        if threading.current_thread() is not threading.main_thread():
            return self
        if self._installed:
            raise RuntimeError("A render interrupt controller cannot be entered twice")
        self._previous = _active
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        _active = self
        self._installed = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        global _active
        if self._installed:
            signal.signal(signal.SIGINT, self._previous_handler)
            _active = self._previous
            self._installed = False

    def _message(self, message):
        # The default avoids print/progress locks, which SIGINT could interrupt
        # while the same thread owns them. No book content or credentials appear.
        try:
            if self._notify is None:
                os.write(2, ("\n" + message + "\n").encode())
            else:
                self._notify(message)
        except Exception:
            pass  # A closed output stream must not prevent stopping.

    def _handle(self, signum, frame):
        if not self.requested:
            self.requested = True
            self._message("Stopping after active tasks finish verification and save to cache; "
                          "press Ctrl-C again to stop immediately.")
            return
        self.forced = True
        self._message("Stopping immediately. Completed cache entries remain reusable.")
        self._force_if_ready()

    @staticmethod
    def _kill(process):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass  # Already exited; avoid arbitrary cleanup inside the handler.

    def _force_if_ready(self):
        # Do not acquire a lock in the signal handler. Counter updates happen
        # under the GIL; the lock serializes increments from worker threads.
        if not self.forced or self._defer_count or self._exiting:
            return
        self._exiting = True
        for process in tuple(self._children.values()):
            self._kill(process)
        self._exit(130)

    @contextmanager
    def defer_force_stop(self):
        """Protect a tiny ownership-publication transaction, never model work.

        A second interrupt is remembered and acted on as soon as all such
        transactions end. This closes the Popen-to-child-registration window
        and permits a filesystem lock to publish its recoverable owner record.
        """
        if self.forced:
            self._force_if_ready()
            raise RenderInterrupted()
        with self._defer_lock:
            self._defer_count += 1
        try:
            yield
        finally:
            with self._defer_lock:
                self._defer_count -= 1
            self._force_if_ready()


def stop_requested() -> bool:
    """Whether this render should stop dispatching new tasks."""
    if _active is not None and _active.requested:
        return True
    check = _scoped_stop.get()
    return check is not None and check() is not None


def stop_if_requested() -> None:
    """Raise only when called at a completed, durably cached task boundary."""
    if _active is not None and _active.requested:
        raise RenderInterrupted()
    check = _scoped_stop.get()
    if check is not None:
        error = check()
        if error is not None:
            raise error


@contextmanager
def scoped_stop_condition(predicate, error):
    """Raise ``error()`` at safe task boundaries while ``predicate()`` holds.

    Scopes one owner's stop condition to work running for that owner, such as
    a chapter job whose rendering client disconnected. Nested scopes chain, so
    an inherited condition keeps applying inside an inner scope.
    """
    outer = _scoped_stop.get()

    def check():
        if predicate():
            return error() if callable(error) and not isinstance(error, BaseException) else error
        return None if outer is None else outer()

    token = _scoped_stop.set(check)
    try:
        yield
    finally:
        _scoped_stop.reset(token)


@contextmanager
def defer_force_stop():
    """Protect a short ownership transaction when render SIGINT handling is on."""
    controller = _active
    if controller is None:
        yield
    else:
        with controller.defer_force_stop():
            yield


def run_subprocess(*popenargs, input=None, capture_output=False, timeout=None,
                   check=False, **kwargs):
    """Run a child that survives the first Ctrl-C and dies with a forced stop.

    Normal subprocess.run behavior is retained outside the render controller,
    including existing test/mocking boundaries. Active FFmpeg children get a
    separate session so terminal process-group SIGINT cannot corrupt their
    in-progress normalization before the resulting clip is verified and cached.
    """
    controller = _active
    if controller is None:
        return subprocess.run(*popenargs, input=input, capture_output=capture_output,
                              timeout=timeout, check=check, **kwargs)
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    if os.name == "posix":
        kwargs["start_new_session"] = True
    # A force-stop arriving during Popen must wait for us to register the child
    # before it kills the process group and exits. No request/model work belongs
    # in this small deferral window.
    with controller.defer_force_stop():
        process = subprocess.Popen(*popenargs, **kwargs)
        controller._children[process.pid] = process
    try:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            controller._kill(process)
            stdout, stderr = process.communicate()
            exc.output, exc.stderr = stdout, stderr
            raise
        except BaseException:
            controller._kill(process)
            process.wait()
            raise
        returncode = process.poll()
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, process.args,
                                                output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(process.args, returncode, stdout, stderr)
    finally:
        controller._children.pop(process.pid, None)
