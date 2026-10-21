"""Keep lazy MLX arrays and their streams on one persistent inference thread."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock, get_ident


class LocalRuntime:
    """Serial execution with thread affinity, lazy streams, and reentrant calls.

    Production MLX operations request stream initialization explicitly. Test
    doubles and non-MLX plugins never import or allocate a GPU runtime here.
    """

    def __init__(self, *, name="audiobook-local-mlx"):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._thread_id = None
        self._streams = None
        self._state = Lock()
        self._closed = False

    def initialize_mlx(self):
        """Create this worker's CPU/GPU streams before its first model load."""
        if get_ident() != self._thread_id:
            raise RuntimeError("MLX streams must be initialized on their inference thread")
        if self._streams is None:
            import mlx.core as mx
            streams = [mx.new_stream(mx.cpu), mx.new_stream(mx.gpu)]
            for stream in streams:
                mx.set_default_stream(stream)
            self._streams = streams

    def _invoke(self, function, args, kwargs, initialize_mlx):
        self._thread_id = get_ident()
        if initialize_mlx:
            self.initialize_mlx()
        return function(*args, **kwargs)

    def run(self, function, *args, initialize_mlx=False, **kwargs):
        with self._state:
            if self._closed:
                raise RuntimeError("The local inference runtime is closed")
            reentrant = get_ident() == self._thread_id
            future = None if reentrant else self._executor.submit(
                self._invoke, function, args, kwargs, initialize_mlx)
        if reentrant:
            # Model wrappers can delegate to another operation on the same lane.
            return self._invoke(function, args, kwargs, initialize_mlx)
        try:
            return future.result()
        except BaseException:
            # A main-thread Ctrl-C interrupts the wait, not the queued model
            # operation. Keep the caller's book/cache locks alive until that
            # operation is actually finished, then propagate the interruption.
            while not future.done():
                try:
                    wait((future,), timeout=0.25)
                except KeyboardInterrupt:
                    continue
            raise

    def close(self):
        if get_ident() == self._thread_id:
            raise RuntimeError("Close the local inference runtime from its owner")
        with self._state:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)


def requires_mlx(component):
    """Recognize built-in lazy MLX components without looking at mock identities."""
    from .engines import Drama3MLXEngine
    from .quality import MlxWhisperVerifier
    from .reference_asr import MlxParakeetVerifier
    from .verification import ReferenceCorroboratingVerifier, ReferenceTextJudgingVerifier

    if type(component) in (Drama3MLXEngine, MlxWhisperVerifier, MlxParakeetVerifier):
        return True
    if type(component) is ReferenceCorroboratingVerifier:
        return requires_mlx(component.primary) or requires_mlx(component.reference)
    if type(component) is ReferenceTextJudgingVerifier:
        return requires_mlx(component.verifier)
    return False
