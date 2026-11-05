"""Bounded Gemini casting with private workers and model-affine verification."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from functools import wraps
from threading import Lock

from .gemini_voice_design import GeminiVoiceError
from .local_runtime import LocalRuntime, requires_mlx
from .model_setup import ModelSetupError
from .workflow import RenderBlocked, SpeechVerificationFailed, VerificationUnavailable


_ORDINARY_ERRORS = (
    RenderBlocked, SpeechVerificationFailed, GeminiVoiceError,
    OSError, ValueError, RuntimeError,
)


def ordinary_error(error):
    """Classify per-voice failures without hiding batch-fatal prerequisites."""
    return (not isinstance(error, (ModelSetupError, VerificationUnavailable))
            and isinstance(error, _ORDINARY_ERRORS))


class BatchLease:
    """Authorize worker lock contexts only while the coordinator owns the store lease."""

    def __init__(self):
        self._lock = Lock()
        self._active = True

    def require_active(self):
        with self._lock:
            if not self._active:
                raise RuntimeError("The casting batch lease is no longer active")

    def revoke(self):
        with self._lock:
            self._active = False


class _Fatal:
    def __init__(self):
        self._lock = Lock()
        self._error = None

    @property
    def error(self):
        with self._lock:
            return self._error

    def record(self, error):
        with self._lock:
            if self._error is None:
                self._error = error

    def submit(self, executor, function, *args):
        # Submission and the fatal check share one linearization point. A task is
        # either admitted before the first fatal or never admitted after it.
        with self._lock:
            if self._error is not None:
                return None
            return executor.submit(function, *args)


class _BatchEngine:
    def __init__(self, engine):
        self.identity = deepcopy(engine.identity)
        self.voice_binding_identity = deepcopy(
            getattr(engine, "voice_binding_identity", engine.identity))
        self.capabilities = deepcopy(engine.capabilities)


class _BatchVerifier:
    """Route every verifier method through the batch's one inference thread."""

    def __init__(self, verifier, runtime, fatal):
        self._verifier = verifier
        self._runtime = runtime
        self._fatal = fatal
        self._initialize_mlx = requires_mlx(verifier)

    def __getattr__(self, name):
        try:
            value = self._runtime.run(
                getattr, self._verifier, name, initialize_mlx=self._initialize_mlx)
        except AttributeError:
            # Optional verifier capabilities use getattr(..., default) probes.
            raise
        except BaseException as error:
            self._fatal.record(error)
            raise
        if not callable(value):
            return value

        @wraps(value)
        def serialized(*args, **kwargs):
            try:
                return self._runtime.run(
                    value, *args, initialize_mlx=self._initialize_mlx, **kwargs)
            except BaseException as error:
                # quality.validate_audio converts these failures into ordinary
                # reports. Preserve their fatal origin for the batch scheduler.
                self._fatal.record(error)
                raise

        return serialized


def _close_resource(resource):
    close = getattr(resource, "close", None)
    if callable(close):
        close()


class _Resources:
    def __init__(self, session, fatal):
        self.session = session
        self.fatal = fatal
        self.runtime = LocalRuntime(name="audiobook-casting-verification")
        self.actual_engine = None
        self.engine = None
        self.verifier = None
        self.closed = False

    def start(self):
        def construct():
            engine_factory = self.session._engine_factory
            if engine_factory is None:
                from .engines import create_engine
                engine_factory = create_engine
            self.actual_engine = engine_factory(deepcopy(self.session.config["engine"]))
            self.engine = _BatchEngine(self.actual_engine)

        self.runtime.run(construct)
        self.verifier = self.session._batch_verifier(self.fatal)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.actual_engine is not None:
            try:
                self.runtime.run(_close_resource, self.actual_engine)
            except BaseException as error:
                self.fatal.record(error)
        try:
            self.runtime.close()
        except BaseException as error:
            self.fatal.record(error)


def _invoke(worker, voice_id, regenerate, fatal):
    try:
        if regenerate:
            return worker.regenerate(voice_id)
        return worker.ensure_ready(voice_id)
    except BaseException as error:
        if not ordinary_error(error):
            fatal.record(error)
        raise


def _schedule(workers, representatives, regenerate, limit, fatal):
    results = [{
        "voice_id": row["voice_id"], "canonical_voice_id": row["canonical_voice_id"],
        "selected": None, "error": None,
    } for row in representatives]
    executor = ThreadPoolExecutor(max_workers=limit, thread_name_prefix="audiobook-gemini-casting")
    pending = {}
    free_slots = list(range(len(workers)))
    next_index = 0
    try:
        while next_index < len(representatives) or pending:
            while next_index < len(representatives) and len(pending) < limit and free_slots:
                slot = free_slots.pop(0)
                future = fatal.submit(
                    executor, _invoke, workers[slot],
                    representatives[next_index]["voice_id"], regenerate, fatal)
                if future is None:
                    free_slots.append(slot)
                    break
                pending[future] = next_index, slot
                next_index += 1
            if not pending:
                break
            try:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            except KeyboardInterrupt as error:
                fatal.record(error)
                continue
            for future in done:
                index, slot = pending.pop(future)
                try:
                    selected = future.result()
                except BaseException as error:
                    if ordinary_error(error):
                        results[index]["error"] = error
                    else:
                        fatal.record(error)
                else:
                    results[index]["selected"] = selected
                free_slots.append(slot)
    finally:
        while True:
            try:
                executor.shutdown(wait=True)
                break
            except KeyboardInterrupt as error:
                # Repeated Ctrl-C must not abandon started voice work or release
                # the outer store lease while a worker is still active.
                fatal.record(error)
    return results


def run_gemini_batch(session, representatives, *, regenerate, worker_factory):
    """Run one preflighted Gemini batch under the caller's outer store lease."""
    lease = BatchLease()
    fatal = _Fatal()
    resources = _Resources(session, fatal)
    workers = []
    results = []
    try:
        session.prepare_models(generation=True)
        session._retry_gemini_cleanup()
        resources.start()
        limit = min(session.config["casting"]["gemini_concurrency"], len(representatives))
        for _ in range(limit):
            worker = worker_factory(lease, resources)
            workers.append(worker)
            worker._cleanup_attempted = set(session._cleanup_attempted)
            worker._get_designer()
        if regenerate:
            session._invalidate_regeneration_targets([row["voice_id"] for row in representatives])
        results = _schedule(workers, representatives, regenerate, limit, fatal)
    except BaseException as error:
        fatal.record(error)
    finally:
        for worker in workers:
            try:
                worker.close()
            except BaseException as error:
                fatal.record(error)
        resources.close()
        lease.revoke()
    if fatal.error is not None:
        raise fatal.error
    return results
