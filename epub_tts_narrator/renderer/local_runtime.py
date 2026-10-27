"""Keep lazy MLX arrays and their streams on one persistent inference thread."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import copy_context
from threading import Lock, get_ident

from .render_metrics import METRICS, STAGES, classify_outcome
from .render_resource_context import suppress_resource_routing


class LocalRuntime:
    """Serial execution with thread affinity, lazy streams, and reentrant calls.

    Production MLX operations request stream initialization explicitly. Test
    doubles and non-MLX plugins never import or allocate a GPU runtime here.
    ``metrics_stage`` opts a runtime into payload-free resource metrics; the
    queued gauge covers dispatch-to-thread wait and the active gauge covers
    execution on the persistent thread. Reentrant same-thread delegation stays
    part of the enclosing operation and is never counted twice. The model
    function runs with resource routing suppressed: bounded lanes block on
    this thread, so nested leaves direct-call instead of re-entering a lane
    whose worker is waiting for this very operation.
    """

    def __init__(self, *, name="audiobook-local-mlx", metrics_stage=None):
        if metrics_stage is not None and metrics_stage not in STAGES:
            raise ValueError("metrics_stage must be one of the render_metrics.STAGES names")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._thread_id = None
        self._streams = None
        self._state = Lock()
        self._closed = False
        self._metrics_stage = metrics_stage

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

    def _invoke(self, function, args, kwargs, initialize_mlx, operation=None):
        self._thread_id = get_ident()
        if operation is not None:
            operation.start()
        try:
            if initialize_mlx:
                self.initialize_mlx()
            with suppress_resource_routing():
                result = function(*args, **kwargs)
        except BaseException as error:
            if operation is not None:
                operation.end(classify_outcome(error))
            raise
        if operation is not None:
            operation.end("completed")
        return result

    def run(self, function, *args, initialize_mlx=False, **kwargs):
        with self._state:
            if self._closed:
                raise RuntimeError("The local inference runtime is closed")
            reentrant = get_ident() == self._thread_id
            if not reentrant:
                # A fresh context copy per submitted operation carries scoped
                # stop conditions onto the persistent inference thread; a
                # Context object is never reused concurrently.
                context = copy_context()
                operation = (METRICS.operation(self._metrics_stage)
                             if self._metrics_stage is not None else None)
                try:
                    future = self._executor.submit(
                        lambda: context.run(self._invoke, function, args, kwargs,
                                            initialize_mlx, operation))
                except BaseException:
                    if operation is not None:
                        operation.cancel()
                    raise
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
    from .mlx_audio_engine import MLXAudioDrama3Engine
    from .quality import MlxWhisperVerifier
    from .reference_asr import MlxParakeetVerifier
    from .verification import ReferenceCorroboratingVerifier, ReferenceTextJudgingVerifier

    if type(component) in (Drama3MLXEngine, MLXAudioDrama3Engine, MlxWhisperVerifier, MlxParakeetVerifier):
        return True
    if type(component) is ReferenceCorroboratingVerifier:
        return requires_mlx(component.primary) or requires_mlx(component.reference)
    if type(component) is ReferenceTextJudgingVerifier:
        return requires_mlx(component.verifier)
    return False
