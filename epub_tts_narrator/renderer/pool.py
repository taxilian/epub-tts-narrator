"""Concurrent TTS workers with one coordinator-owned verification and cache store."""

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from queue import Queue
from threading import RLock
from time import monotonic

from .engines import EngineError, create_engine
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
from .local_runtime import LocalRuntime, requires_mlx
from .progress import generation_notifier, report_worker_activity


_SAMPLING_DEFAULTS = {
    "experts": "q8", "temperature": 1.0, "top_p": 0.9, "top_k": 30,
    "max_new_tokens": 2048, "min_audio_seconds": 0.1,
}


def pool_sampling(config):
    """Audio-affecting settings, without the coordinator's model location."""
    settings = config.get("engine", {})
    return {key: settings.get(key, value) for key, value in _SAMPLING_DEFAULTS.items()}


class _SerializedVerifier:
    """Protect optional ASR/judge methods without inventing unsupported methods."""

    def __init__(self, verifier, lock, runtime):
        self._verifier = verifier
        self._lock = lock
        self._runtime = runtime

    def __getattr__(self, name):
        value = getattr(self._verifier, name)
        if not callable(value):
            return value

        @wraps(value)
        def serialized(*args, **kwargs):
            def run():
                with self._lock:
                    return value(*args, **kwargs)
            return self._runtime.run(run, initialize_mlx=name != "close" and requires_mlx(self._verifier))
        return serialized


class _Worker:
    def __init__(self, settings, engine_settings, sampling):
        self.settings = deepcopy(settings)
        self.engine_settings = deepcopy(engine_settings)
        self.sampling = deepcopy(sampling)
        self._engine = None
        self._initialization = RLock()

    @property
    def local(self):
        return self.settings["type"] == "local"

    def engine(self):
        with self._initialization:
            if self._engine is None:
                if self.local:
                    self._engine = create_engine(self.engine_settings)
                else:
                    from .fish_http import FishHTTPEngine
                    self._engine = FishHTTPEngine(self.settings, sampling=self.sampling)
            return self._engine

    def close(self):
        if self._engine is not None:
            close = getattr(self._engine, "close", None)
            if callable(close):
                close()


class RenderPool:
    """A work-conserving pool; transport details do not change saved take identity.

    The configured model revision is the operator's compatibility contract for
    these workers. The actual backend and worker remain recorded on every take.
    References are prepared once on the coordinator, never redesigned remotely.
    """

    voice_binding_identity = {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    def __init__(self, config):
        settings = config["render_pool"]
        selected = get_drama3_model(config["engine"].get("model", DEFAULT_DRAMA3_MODEL))
        self.sampling = pool_sampling(config)
        self.local_lock = RLock()
        self._local_runtime = LocalRuntime()
        self._closed = False
        self._workers = [_Worker(worker, config["engine"], self.sampling)
                         for worker in settings["workers"]]
        if not self._workers:
            raise EngineError("A render pool needs at least one worker")
        self._slots = Queue()
        for worker in self._workers:
            for _ in range(worker.settings.get("concurrency", 1)):
                self._slots.put(worker)
        self.max_parallel = self._slots.qsize()
        # One verification/normalization job may overlap the occupied TTS slots.
        self.max_inflight = self.max_parallel + 1
        self._identity = {
            "engine": "fish_tts_pool", "model": selected.repo, "contract_version": 1,
            "model_revision": settings.get("model_revision", selected.revision),
            "sampling": self.sampling,
        }
        self.capabilities = {
            "reference_voice": True,
            "voice_design": selected.runtime == "sogni" and any(w.local for w in self._workers),
            "preset_voice": False, "delivery_control": True,
            "multi_speaker": all(w.local or w.settings.get("multi_speaker", True)
                                 for w in self._workers),
            "unconditioned_voice_creation": False,
        }

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def voice_design_identity(self):
        worker = next((worker for worker in self._workers if worker.local), None)
        if worker is None:
            return self.identity
        def identity():
            with self.local_lock:
                engine = worker.engine()
                return deepcopy(getattr(engine, "voice_design_identity", engine.identity))
        return self._local_runtime.run(identity)

    def wrap_verifier(self, verifier):
        if isinstance(verifier, _SerializedVerifier) and verifier._lock is self.local_lock:
            return verifier
        return _SerializedVerifier(verifier, self.local_lock, self._local_runtime)

    def _execute(self, method, *args, **kwargs):
        if self._closed:
            raise EngineError("The render pool is closed")
        notify = generation_notifier()
        report_worker_activity(notify, "waiting")
        worker = self._slots.get()
        try:
            name = worker.settings["name"]
            report_worker_activity(notify, "assigned", name)
            def run():
                started = monotonic()
                report_worker_activity(notify, "started", name)
                try:
                    engine = worker.engine()
                    if worker.local and requires_mlx(engine):
                        self._local_runtime.initialize_mlx()
                    metadata = getattr(engine, method)(*args, **kwargs)
                    if not isinstance(metadata, dict):
                        raise EngineError("TTS worker returned invalid generation metadata")
                    result = {**metadata, "render_worker": name, "worker_engine": engine.identity}
                except BaseException:
                    report_worker_activity(notify, "failed", name, monotonic() - started)
                    raise
                report_worker_activity(notify, "completed", name, monotonic() - started)
                return result
            if worker.local:
                # All model loading, arrays and default streams belong to one
                # persistent thread, shared with local speech verification.
                def local_run():
                    with self.local_lock:
                        return run()
                return self._local_runtime.run(local_run)
            return run()
        finally:
            self._slots.put(worker)

    def synthesize(self, text, voice, output_path, *, seed, direction=""):
        return self._execute("synthesize", text, voice, output_path, seed=seed, direction=direction)

    def synthesize_dialogue(self, turns, output_path, *, seed):
        if not self.capabilities["multi_speaker"]:
            raise EngineError("Every render worker must support multi-speaker paragraphs")
        return self._execute("synthesize_dialogue", turns, output_path, seed=seed)

    def create_voice(self, description, text, output_path, *, seed):
        worker = next((worker for worker in self._workers if worker.local), None)
        if worker is None:
            raise EngineError("Remote workers reuse saved voices. Configure a voice designer or add a local worker to create missing references.")
        def create():
            with self.local_lock:
                engine = worker.engine()
                if requires_mlx(engine):
                    self._local_runtime.initialize_mlx()
                metadata = engine.create_voice(description, text, output_path, seed=seed)
                return {**metadata, "render_worker": worker.settings["name"], "worker_engine": engine.identity}
        return self._local_runtime.run(create)

    def close(self):
        # The workflow drains its executor before the CLI closes model instances.
        if self._closed:
            return
        self._closed = True
        def close_local():
            with self.local_lock:
                for worker in self._workers:
                    if worker.local:
                        worker.close()
        try:
            self._local_runtime.run(close_local)
            for worker in self._workers:
                if not worker.local:
                    worker.close()
        finally:
            self._local_runtime.close()


def create_render_engine(config):
    """Keep the original single-engine path unless workers are configured."""
    if not config.get("render_pool", {}).get("workers"):
        return create_engine(config["engine"])
    return RenderPool(config)
