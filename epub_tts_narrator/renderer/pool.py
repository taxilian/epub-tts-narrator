"""Concurrent TTS workers with one coordinator-owned verification and cache store."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from threading import Condition, RLock, get_ident
from time import monotonic

from .config import SAMPLING_DEFAULTS, default_multi_speaker
from .engines import EngineError, create_engine
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
from .local_runtime import LocalRuntime, requires_mlx
from .progress import generation_notifier, report_worker_activity
from .render_metrics import METRICS, classify_outcome
from .render_resource_context import current_owner


_SAMPLING_DEFAULTS = {**SAMPLING_DEFAULTS, "min_audio_seconds": 0.1}

class _Slots:
    """Endpoint slots with FIFO waiters and first-free claims."""

    def __init__(self, workers):
        self._free = {id(worker): worker.settings.get("concurrency", 1) for worker in workers}
        self._workers = list(workers)
        self._waiting = deque()
        self._ready = Condition()

    def qsize(self):
        with self._ready:
            return sum(self._free.values())

    def _claim(self):
        worker = next((worker for worker in self._workers if self._free[id(worker)]), None)
        if worker is not None:
            self._free[id(worker)] -= 1
        return worker

    def try_acquire(self):
        """Claim the first free slot, or return None without blocking."""
        with self._ready:
            return None if self._waiting else self._claim()

    def acquire(self):
        return self.wait_acquire()

    def wait_acquire(self):
        """Wait in FIFO order for the first free slot."""
        marker = object()
        with self._ready:
            self._waiting.append(marker)
            try:
                while True:
                    if self._waiting[0] is marker:
                        worker = self._claim()
                        if worker is not None:
                            self._waiting.popleft()
                            self._ready.notify_all()
                            return worker
                    self._ready.wait()
            finally:
                if marker in self._waiting:
                    self._waiting.remove(marker)
                    self._ready.notify_all()

    def release(self, worker):
        with self._ready:
            self._free[id(worker)] += 1
            self._ready.notify_all()


class _FifoAdmission:
    """Bounded legacy admission with FIFO waiters and timeout-safe markers."""

    def __init__(self, limit):
        self._available = limit
        self._waiting = deque()
        self._markers = {}
        self._cond = Condition()

    def acquire(self, timeout=None):
        deadline = None if timeout is None else monotonic() + timeout
        ident = get_ident()
        with self._cond:
            marker = self._markers.get(ident)
            if marker is None:
                marker = object()
                self._markers[ident] = marker
                self._waiting.append(marker)
            try:
                while self._waiting[0] is not marker or not self._available:
                    if deadline is None:
                        notified = self._cond.wait()
                    else:
                        remaining = deadline - monotonic()
                        if remaining <= 0:
                            return False
                        notified = self._cond.wait(remaining)
                    if not notified:
                        return False
                self._waiting.popleft()
                del self._markers[ident]
                self._available -= 1
                self._cond.notify_all()
                return True
            except BaseException:
                self._remove(ident, marker)
                raise

    def release(self):
        with self._cond:
            self._available += 1
            self._cond.notify_all()

    def cancel(self):
        with self._cond:
            ident = get_ident()
            marker = self._markers.get(ident)
            if marker is not None:
                self._remove(ident, marker)

    def _remove(self, ident, marker):
        self._markers.pop(ident, None)
        if marker in self._waiting:
            self._waiting.remove(marker)
            self._cond.notify_all()


def pool_sampling(config):
    """Audio-affecting settings, without the coordinator's model location."""
    settings = config.get("engine", {})
    return {key: settings.get(key, value) for key, value in _SAMPLING_DEFAULTS.items()}


def _text_review_method(name, method):
    """Recognize our text-only adapters, never infer safety from a plugin name."""
    if name not in {"judge_reference", "judge_skipped_word", "review_phonetics", "review_reference_phonetics"}:
        return False
    from epub_tts_narrator.jev import JevTranscriptJudge
    from .phonetic_review import PronunciationReviewer
    from .recovery_client import RecoveryModelClient
    from .verification import ReferenceTextJudgingVerifier

    owner, function = getattr(method, "__self__", None), getattr(method, "__func__", None)
    if type(owner) is not ReferenceTextJudgingVerifier:
        return False
    if name in {"judge_reference", "judge_skipped_word"}:
        return (function is getattr(ReferenceTextJudgingVerifier, name)
                and type(owner.judge) is JevTranscriptJudge)
    if name in {"review_phonetics", "review_reference_phonetics"}:
        if name == "review_phonetics" and owner.judge is None:
            return False
        reviewer = owner.phonetic_reviewer
        return (function is getattr(ReferenceTextJudgingVerifier, name)
                and type(reviewer) is PronunciationReviewer
                and type(reviewer._client) is RecoveryModelClient)
    return False


class _SerializedVerifier:
    """Keep recognition model-affine while known text reviews use their own lane.

    With a ``recognize`` dispatcher (a RenderPool), each recognizer model call
    (``RECOGNITION_CALLS``) runs on whichever recognition slot is free: the
    controller's own recognition thread or a matching remote ``serve`` worker.
    """

    def __init__(self, verifier, lock, runtime, review_runtime, resources=None, recognize=None):
        self._recognize = recognize
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

            from .recognition_workers import RECOGNITION_CALLS
            if self._recognize is not None and name in RECOGNITION_CALLS:
                return self._recognize(name, method, args, kwargs,
                                       initialize_mlx=requires_mlx(self._verifier))

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

    @property
    def batched(self):
        """A local engine that decodes this worker's concurrent slots together on its own MLX thread.

        Only an engine that reports several ``concurrent_requests`` leaves the
        shared model thread; any other local engine stays serialized on it.
        """
        if not self.local or self.settings.get("concurrency", 1) <= 1:
            return False
        return getattr(self.engine(), "concurrent_requests", 1) > 1

    def engine(self):
        with self._initialization:
            if self._engine is None:
                if self.local:
                    engine = create_engine(self.engine_settings)
                    rows = self.settings.get("concurrency", 1)
                    if rows > 1:
                        # A local engine serving several slots decodes them together.
                        if not callable(getattr(engine, "set_batch_rows", None)):
                            raise EngineError("A local renderer with concurrency above 1 needs an engine that decodes batches (drama3_mlx)")
                        engine.set_batch_rows(rows)
                    self._engine = engine
                else:
                    from .fish_http import FishHTTPEngine
                    self._engine = FishHTTPEngine(self.settings, sampling=self.sampling)
            return self._engine

    def close(self):
        if self._engine is not None:
            close = getattr(self._engine, "close", None)
            if callable(close):
                close()


class _Recognizer:
    """One recognition endpoint: the controller's own thread (client None) or a serve worker."""

    def __init__(self, name, concurrency, client=None):
        self.settings = {"name": name, "concurrency": concurrency}
        self.client = client

    @property
    def local(self):
        return self.client is None


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
        # Local speech recognition has its own MLX thread (own streams), so it
        # runs beside local generation instead of pausing it.
        self._recognition_lock = RLock()
        self._recognition_runtime = LocalRuntime(name="audiobook-recognition-mlx")
        self._closed = False
        self._workers = [_Worker(worker, config["engine"], self.sampling)
                         for worker in settings["workers"] if "tts" in worker.get("roles", ["tts"])]
        if not self._workers:
            raise EngineError("A render pool needs at least one TTS worker")
        from .recognition_workers import RemoteRecognizer
        self._recognizers = [_Recognizer("local", 1)] + [
            _Recognizer(worker["name"], worker.get("recognition_concurrency", 1), RemoteRecognizer(worker))
            for worker in settings["workers"] if "recognition" in worker.get("roles", ())]
        self._recognition_slots = _Slots(self._recognizers)
        self._recognition_verified = None
        self._slots = _Slots(self._workers)
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
        self._admission = _FifoAdmission(max_inflight) if max_inflight else None
        self._resources = self._build_resources(config)
        self._logical = None
        if self._resources is not None:
            # One shared FIFO logical clip scheduler for this pool's
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
            "chapter_continuity": True,
        }
        self._continuity_checked = False

    def check_continuity(self):
        """Every worker must carry chapter history; checked once per pool."""
        if self._continuity_checked:
            return
        from .continuity import require_support
        def check():
            for worker in self._workers:
                if worker.local:
                    require_support(worker.engine())
        self._local_runtime.run(check)
        for worker in self._workers:
            if not worker.local:
                require_support(worker.engine())
        self._continuity_checked = True

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
            recognition_slots=self._recognition_slots.qsize(),
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
        if isinstance(verifier, _SerializedVerifier) and verifier._lock is self._recognition_lock:
            return verifier
        self.check_recognition(verifier)
        return _SerializedVerifier(verifier, self._recognition_lock, self._recognition_runtime,
                                   self._review_runtime, self._resources, recognize=self._recognize)

    def check_recognition(self, verifier):
        """Remote recognizers must run exactly this verifier's models; checked once per verifier."""
        remote = [recognizer for recognizer in self._recognizers if not recognizer.local]
        if not remote:
            return
        from .recognition_workers import identity_digest, recognition_identity, recognizers
        expected = recognition_identity(*recognizers(verifier))
        digest = identity_digest(expected)
        if self._recognition_verified == digest:
            return
        for recognizer in remote:
            recognizer.client.verify(expected)
        self._recognition_verified = digest

    def _recognize(self, name, method, args, kwargs, *, initialize_mlx):
        """One recognizer model call in FIFO order on the first free slot."""
        def run_on(recognizer):
            if recognizer.local or kwargs or len(args) != 1:
                def local():
                    with self._recognition_lock:
                        return method(*args, **kwargs)
                result = self._recognition_runtime.run(local, initialize_mlx=initialize_mlx)
            else:
                result = recognizer.client.recognize(name, args[0])
            if isinstance(result, dict) and len(self._recognizers) > 1:
                # Provenance only, once remote recognizers share the work.
                result = {**result, "recognized_by": recognizer.settings["name"]}
            return result

        # No stop check: a started take's verification drains past owner loss.
        if self._closed:
            raise EngineError("The render pool is closed")
        if self._resources is not None:
            claimed = []

            def reserve():
                # Under the recognition lane's lock: claim and dequeue are one
                # step, and the reserving thread runs the call at once.
                recognizer = self._recognition_slots.try_acquire()
                if recognizer is None:
                    return None
                claimed.append(recognizer)
                return lambda: self._recognition_slots.release(recognizer)

            return self._resources.run_recognition(lambda: run_on(claimed[-1]),
                                                   owner=current_owner(), reserve=reserve)
        recognizer = self._recognition_slots.wait_acquire()
        try:
            with METRICS.active("recognition"):
                return run_on(recognizer)
        finally:
            self._recognition_slots.release(recognizer)

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
            else:
                self._admission.cancel()

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
            worker = self._slots.acquire()
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
                        if worker.local and not worker.batched and requires_mlx(engine):
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
            if worker.local and not worker.batched:
                # All model loading, arrays and default streams belong to one
                # persistent thread, shared with local speech verification.
                # A batched local engine has its own MLX thread and decode loop.
                def local_run():
                    with self.local_lock:
                        return run()
                return self._local_runtime.run(local_run)
            return run()
        finally:
            self._slots.release(worker)

    def _execute_on_lane(self, method, args, kwargs):
        """Pipeline mode: one bounded FIFO job per declared endpoint slot.

        The lane owns the synthesis stage gauges, so the job does not wrap
        them again. Each take reserves the first free worker only as it reaches
        the head of the FIFO queue, so waiting takes hold neither an endpoint
        slot nor a lane thread. A full declared synthesis backlog makes the
        caller wait here instead of hiding an unbounded work queue.
        """
        from .interrupts import stop_if_requested

        notify = generation_notifier()
        claimed = []
        # Queued time until a worker slot is reserved. Ended once: an
        # interrupted caller may race a later reservation of its queued take.
        slot_wait = METRICS.operation("synthesis_slot")
        slot_wait_open = [True]
        slot_wait_lock = RLock()

        def end_slot_wait(outcome):
            with slot_wait_lock:
                if slot_wait_open[0]:
                    slot_wait_open[0] = False
                    slot_wait.end(outcome)

        def reserve():
            # Claim and dequeue are one atomic step. The reserving lane thread
            # runs the take at once or releases the slot exactly once.
            worker = self._slots.try_acquire()
            if worker is None:
                return None
            claimed.append(worker)
            end_slot_wait("completed")
            return lambda: self._slots.release(worker)

        def run_on_lane():
            worker = claimed[-1]
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
                    if worker.local and not worker.batched and requires_mlx(engine):
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
            if worker.local and not worker.batched:
                def local_run():
                    with self.local_lock:
                        return run()
                return self._local_runtime.run(local_run)
            return run()

        report_worker_activity(notify, "waiting")
        try:
            return self._resources.tts.run(run_on_lane, reserve=reserve)
        except BaseException as error:
            if not claimed:
                end_slot_wait(classify_outcome(error))
            raise

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
                    try:
                        self._local_runtime.close()
                    finally:
                        self._recognition_runtime.close()


def create_render_engine(config, *, shared_inference=True):
    """Render Drama 3 through a pool, even with no configured workers.

    A drama3_mlx coordinator without configured workers renders through a
    one-slot local pool: several chapter coordinators queue their next take
    for one persistent MLX inference thread while others verify, so the model
    stays busy although a chapter's takes of one voice are sequential. Plugins keep their own
    single-engine path.
    """
    workers = config.get("render_pool", {}).get("workers")
    if (not workers and shared_inference
            and config.get("engine", {}).get("type") == "drama3_mlx"):
        config = {**config, "render_pool": {**config.get("render_pool", {}),
                                            "workers": [{"name": "local", "type": "local", "concurrency": 1}]}}
        workers = config["render_pool"]["workers"]
    if not workers:
        return create_engine(config["engine"])
    from .auto_serve import shared_worker_config
    config = shared_worker_config(config)
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
