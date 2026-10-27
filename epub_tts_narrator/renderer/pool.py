"""Concurrent TTS workers with one coordinator-owned verification and cache store."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from queue import Queue
from threading import BoundedSemaphore, Condition, RLock, get_ident
from time import monotonic, sleep

from .config import SAMPLING_DEFAULTS, default_multi_speaker
from .engines import EngineError, create_engine
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
from .local_runtime import LocalRuntime, requires_mlx
from .progress import generation_notifier, report_worker_activity
from .render_metrics import METRICS
from .render_resource_context import current_owner


_SAMPLING_DEFAULTS = {**SAMPLING_DEFAULTS, "min_audio_seconds": 0.1}

# Bounded backpressure poll while the synthesis lane's declared backlog is full.
_LANE_RETRY_SECONDS = 0.005


def pool_sampling(config):
    """Audio-affecting settings, without the coordinator's model location."""
    settings = config.get("engine", {})
    return {key: settings.get(key, value) for key, value in _SAMPLING_DEFAULTS.items()}


def _text_review_method(name, method):
    """Recognize our text-only adapters, never infer safety from a plugin name."""
    if name not in {"judge_reference", "review_phonetics"}:
        return False
    from epub_tts_narrator.jev import JevTranscriptJudge
    from .phonetic_review import PronunciationReviewer
    from .recovery_client import RecoveryModelClient
    from .verification import ReferenceTextJudgingVerifier

    owner, function = getattr(method, "__self__", None), getattr(method, "__func__", None)
    if type(owner) is not ReferenceTextJudgingVerifier:
        return False
    if name == "judge_reference":
        return (function is ReferenceTextJudgingVerifier.judge_reference
                and type(owner.judge) is JevTranscriptJudge)
    if name == "review_phonetics":
        reviewer = owner.phonetic_reviewer
        return (function is ReferenceTextJudgingVerifier.review_phonetics
                and type(reviewer) is PronunciationReviewer
                and type(reviewer._client) is RecoveryModelClient)
    return False


class _SerializedVerifier:
    """Keep recognition model-affine while known text reviews use their own lane."""

    def __init__(self, verifier, lock, runtime, review_runtime, resources=None):
        self._verifier = verifier
        self._lock = lock
        self._runtime = runtime
        self._review_runtime = review_runtime
        self._resources = resources
        self._state = Condition(RLock())
        self._active = 0
        self._closing = self._closed = False

    def _call(self, name, method, args, kwargs):
        with self._state:
            if self._closing or self._closed:
                raise RuntimeError("The pooled verifier is closed")
            self._active += 1
        try:
            if _text_review_method(name, method):
                if self._resources is not None:
                    # A parent review routine never executes on the HTTP lane
                    # it waits on; its leaf requests use the shared review lane.
                    return self._resources.run_review_parent(
                        method, *args, owner=current_owner())
                # Keep the reusable HTTP clients serialized, but do not occupy
                # the inference thread or its lock during a network request.
                with METRICS.active("review"):
                    return self._review_runtime.run(method, *args, **kwargs)

            if self._resources is not None:
                def recognized():
                    with self._lock:
                        return method(*args, **kwargs)
                return self._resources.run_recognition(
                    lambda: self._runtime.run(recognized,
                                              initialize_mlx=requires_mlx(self._verifier)),
                    owner=current_owner())

            def run():
                with self._lock:
                    return method(*args, **kwargs)
            with METRICS.active("recognition"):
                return self._runtime.run(run, initialize_mlx=requires_mlx(self._verifier))
        finally:
            with self._state:
                self._active -= 1
                self._state.notify_all()

    def _close(self, method, args, kwargs):
        if get_ident() in {self._runtime._thread_id, self._review_runtime._thread_id}:
            raise RuntimeError("Close the pooled verifier from its owner")
        interrupted = None
        with self._state:
            if self._closing:
                while not self._closed:
                    try:
                        self._state.wait()
                    except KeyboardInterrupt as error:
                        interrupted = error
                if interrupted is not None:
                    raise interrupted
                return
            if self._closed:
                return
            self._closing = True
            while self._active:
                try:
                    self._state.wait()
                except KeyboardInterrupt as error:
                    # Finish draining both lanes before releasing the caller's
                    # render/cache locks, just as LocalRuntime.run does.
                    interrupted = error
        try:
            def run():
                with self._lock:
                    return method(*args, **kwargs)
            result = self._runtime.run(run)
        finally:
            with self._state:
                self._closed = True
                self._state.notify_all()
        if interrupted is not None:
            raise interrupted
        return result

    def __getattr__(self, name):
        value = getattr(self._verifier, name)
        if not callable(value):
            return value

        @wraps(value)
        def serialized(*args, **kwargs):
            if name == "close":
                return self._close(value, args, kwargs)
            return self._call(name, value, args, kwargs)
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
    This preserves already verified speech when switching native/SGLang workers;
    it does not promise identical samples. SGLang uses its stock uncapped main
    tokens and capped residual codebooks, recorded in worker sampling provenance.
    References are prepared once on the coordinator, never redesigned remotely.
    """

    voice_binding_identity = {"engine": "drama3_mlx", "model": DEFAULT_DRAMA3_MODEL}

    def __init__(self, config):
        settings = config["render_pool"]
        max_inflight = settings.get("max_inflight", 0)
        if type(max_inflight) is not int or not 0 <= max_inflight <= 256:
            raise EngineError("render_pool.max_inflight must be 0 (automatic) or an integer from 1 through 256")
        if max_inflight and max_inflight < sum(worker.get("concurrency", 1) for worker in settings["workers"]):
            raise EngineError("render_pool.max_inflight must be at least the total worker concurrency")
        selected = get_drama3_model(config["engine"].get("model", DEFAULT_DRAMA3_MODEL))
        self.sampling = pool_sampling(config)
        self.local_lock = RLock()
        self._local_runtime = LocalRuntime(metrics_stage="model_thread")
        self._review_runtime = LocalRuntime(name="audiobook-text-review")
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
        from .render_pipeline_config import effective_scheduling
        self._effective = effective_scheduling(config)
        # Automatic mode retains the original per-chapter window. An explicit
        # window is shared by every chapter using this pool, including review.
        self.max_inflight = max_inflight or self.max_parallel + 1
        if self._effective["scheduler"] == "pipeline":
            # Every pipeline runtime view reports the same whole-transaction
            # cap the logical scheduler admits; the legacy window keeps its
            # own slots+1 automatic meaning for legacy pools.
            self.max_inflight = self._effective["max_inflight"]
        self._admission = BoundedSemaphore(max_inflight) if max_inflight else None
        self._resources = self._build_resources(config)
        self._logical = None
        if self._resources is not None:
            # One shared, chapter-fair logical clip scheduler for this pool's
            # whole lifetime; a warm server keeps it across books.
            from .render_pipeline import LogicalClipScheduler
            self._logical = LogicalClipScheduler(
                self._effective["max_inflight"], self._effective["chapter_concurrency"],
                name="render-pool-logical")
        self._identity = {
            "engine": "fish_tts_pool", "model": selected.repo, "contract_version": 1,
            "model_revision": settings.get("model_revision", selected.revision),
            "sampling": self.sampling,
        }
        self.capabilities = {
            "reference_voice": True,
            "voice_design": selected.runtime == "sogni" and any(w.local for w in self._workers),
            "preset_voice": False, "delivery_control": True,
            "multi_speaker": all(w.local or w.settings.get(
                "multi_speaker", default_multi_speaker(w.settings.get("protocol")))
                                 for w in self._workers),
            "single_speaker_paragraph": all(
                w.local or w.settings.get("multi_speaker", True)
                or w.settings.get("protocol") == "fish_sglang_drama3" for w in self._workers),
            "unconditioned_voice_creation": False,
            "structured_performance": True,
            "chapter_continuity": all(w.local for w in self._workers),
        }

    def check_continuity(self):
        from .continuity import require_support
        def check():
            for worker in self._workers:
                if not worker.local:
                    raise EngineError("Chapter continuity requires local pool workers")
                require_support({"render": {"continuity": "chapter"}}, worker.engine())
        self._local_runtime.run(check)

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _build_resources(self, config):
        """Pipeline lanes when configured; None keeps the legacy serial paths.

        Lane pending bounds follow the SAME effective max_inflight the
        logical scheduler admits, so a saturated lane backpressures
        transaction runners instead of failing leaves with LaneFull; the
        legacy ``max_inflight`` attribute keeps its per-chapter meaning.
        """
        from .render_resources import RenderResources

        if self._effective["scheduler"] != "pipeline":
            return None
        return RenderResources(
            tts_slots=self.max_parallel,
            synthesis_backlog=self._effective["synthesis_backlog"],
            review_concurrency=self._effective["review_concurrency"],
            max_inflight=self._effective["max_inflight"],
            pending_audio_mb=self._effective["pending_audio_mb"],
        )

    @property
    def resources(self):
        """The pipeline resource runtime, or None in legacy scheduling mode."""
        return self._resources

    @property
    def logical(self):
        """The shared logical clip scheduler, or None in legacy mode."""
        return self._logical

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
        return _SerializedVerifier(verifier, self.local_lock, self._local_runtime,
                                   self._review_runtime, self._resources)

    @contextmanager
    def admit_chunk(self):
        """Bound whole chunk transactions before take locks or inference slots.

        Retries and recovery parts use the parent's admission. Acquiring again
        inside synthesis could deadlock a full window of retrying chunks.
        Pipeline scheduling owns admission through the shared logical clip
        scheduler, so the legacy whole-transaction window does not double-cap it.
        """
        from .interrupts import stop_if_requested

        def check():
            stop_if_requested()
            if self._closed:
                raise EngineError("The render pool is closed")

        if self._logical is not None:
            check()
            yield
            return
        if self._admission is None:
            check()
            yield
            return
        acquired = False
        try:
            with METRICS.waiting("admission"):
                while not acquired:
                    check()
                    acquired = self._admission.acquire(timeout=0.1)
                check()
            yield
        finally:
            if acquired:
                self._admission.release()

    def _execute(self, method, *args, **kwargs):
        from .interrupts import stop_if_requested
        stop_if_requested()
        if self._closed:
            raise EngineError("The render pool is closed")
        if self._resources is not None:
            return self._execute_on_lane(method, args, kwargs)
        notify = generation_notifier()
        report_worker_activity(notify, "waiting")
        with METRICS.waiting("synthesis_slot"):
            worker = self._slots.get()
        try:
            name = worker.settings["name"]
            report_worker_activity(notify, "assigned", name)
            def run():
                # A job may have waited for a worker or the local model thread.
                # Do not start it after cancellation; already-running jobs drain.
                stop_if_requested()
                started = monotonic()
                report_worker_activity(notify, "started", name)
                with METRICS.active("synthesis"):
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

    def _execute_on_lane(self, method, args, kwargs):
        """Pipeline mode: one bounded lane job per declared endpoint slot.

        The lane owns the synthesis stage gauges, so the job does not wrap
        them again. Worker selection stays inside the job: a queued
        transaction must never hold an endpoint slot while it waits for a
        lane thread. A full declared synthesis backlog makes the caller wait
        here instead of hiding an unbounded work queue.
        """
        from .interrupts import stop_if_requested
        from .render_lanes import LaneFull

        notify = generation_notifier()
        owner = current_owner()

        def run_on_lane():
            report_worker_activity(notify, "waiting")
            with METRICS.waiting("synthesis_slot"):
                worker = self._slots.get()
            try:
                name = worker.settings["name"]
                report_worker_activity(notify, "assigned", name)
                def run():
                    # A job may have waited for the endpoint slot or the local
                    # model thread. Do not start it after cancellation;
                    # already-running jobs drain.
                    stop_if_requested()
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
                    def local_run():
                        with self.local_lock:
                            return run()
                    return self._local_runtime.run(local_run)
                return run()
            finally:
                self._slots.put(worker)

        while True:
            try:
                future = self._resources.tts.submit(run_on_lane, owner=owner)
            except LaneFull:
                # Queue-overflow backpressure only: the job was never queued,
                # so a retry cannot rerun started synthesis.
                stop_if_requested()
                if self._closed:
                    raise EngineError("The render pool is closed") from None
                sleep(_LANE_RETRY_SECONDS)
                continue
            return future.result()

    def synthesize(self, text, voice, output_path, *, seed, direction="", performance=None, traits=""):
        options = {"performance": performance, "traits": traits} if performance is not None else {}
        return self._execute("synthesize", text, voice, output_path, seed=seed, direction=direction, **options)

    def synthesize_dialogue(self, turns, output_path, *, seed):
        if not self.capabilities["multi_speaker"]:
            if not self.capabilities["single_speaker_paragraph"]:
                raise EngineError("Every render worker must support multi-speaker paragraphs")
            from .engines import format_dialogue
            _, voices, _ = format_dialogue(turns)
            if len(voices) != 1:
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
        try:
            if self._logical is not None:
                # Stop logical admission, cancel queued transactions, and let
                # started takes finish while every lane and model they may
                # wait on is still open.
                self._logical.close()
        finally:
            # Text reviewers never enter the model lane. Drain them before closing
            # engine resources or the shared model runtime, including during error
            # cleanup after a render coordinator has stopped dispatching work.
            def close_local():
                with self.local_lock:
                    for worker in self._workers:
                        if worker.local:
                            worker.close()
            try:
                # Shutdown runs off the calling thread so an interrupted wait is
                # drained by LocalRuntime before engine resources are released.
                self._local_runtime.run(self._review_runtime.close)
            finally:
                try:
                    if self._resources is not None:
                        # Drains started-take verification leaves, cancels queued
                        # synthesis, and joins every lane worker before the
                        # engines they may still be using are released.
                        self._resources.close()
                    self._local_runtime.run(close_local)
                    for worker in self._workers:
                        if not worker.local:
                            worker.close()
                finally:
                    self._local_runtime.close()


def create_render_engine(config, *, shared_inference=False):
    """Keep the original single-engine path unless workers are configured.

    With shared_inference, a drama3_mlx coordinator without configured workers
    renders through a one-slot local pool so concurrent chapter jobs share one
    persistent MLX inference thread, including lazy model initialization.
    """
    workers = config.get("render_pool", {}).get("workers")
    if (not workers and shared_inference
            and config.get("engine", {}).get("type") == "drama3_mlx"):
        config = {**config, "render_pool": {**config.get("render_pool", {}),
                                            "workers": [{"name": "local", "type": "local", "concurrency": 1}]}}
        workers = config["render_pool"]["workers"]
    if not workers:
        return create_engine(config["engine"])
    return RenderPool(config)


def bind_engine_verifier(engine, verifier):
    """Apply the engine's verifier wrapping when the engine provides one."""
    wrap = getattr(engine, "wrap_verifier", None)
    if callable(wrap):
        return wrap(verifier)
    return verifier


def create_render_components(config, *, verifier_factory):
    """Create the configured render engine and verifier, then bind them."""
    engine = create_render_engine(config)
    verifier = verifier_factory(config["verification"], config.get("jev"), config.get("provider"))
    return engine, bind_engine_verifier(engine, verifier)
