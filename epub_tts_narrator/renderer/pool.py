"""Concurrent TTS workers with one coordinator-owned verification and cache store."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from threading import Condition, Event, RLock, Thread, get_ident
from time import monotonic

from .config import SAMPLING_DEFAULTS, default_multi_speaker
from .engines import EngineError, create_engine
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
from .local_runtime import LocalRuntime, requires_mlx
from .progress import generation_notifier, report_worker_activity
from .render_metrics import METRICS, classify_outcome
from .render_resource_context import current_owner


_SAMPLING_DEFAULTS = {**SAMPLING_DEFAULTS, "min_audio_seconds": 0.1}


def _audio_seconds(metadata):
    """Seconds of audio a generation made, from its metadata (``None`` when it does not say)."""
    seconds = metadata.get("duration_seconds")
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
        return seconds
    count, rate = metadata.get("sample_count"), metadata.get("sample_rate")
    if isinstance(count, int) and isinstance(rate, int) and rate > 0:
        return count / rate
    return None

class _Slots:
    """Endpoint slots with FIFO waiters and first-free claims.

    A worker can be set aside (its free slots are no longer offered; ones
    already claimed are returned as usual) and later restored in its place.
    """

    def __init__(self, workers):
        self._capacity = {id(worker): worker.settings.get("concurrency", 1) for worker in workers}
        self._claimed = {id(worker): 0 for worker in workers}
        self._order = list(workers)
        self._workers = list(workers)  # the workers offered now, in configured order
        self._waiting = deque()
        self._ready = Condition()

    def _free(self, worker):
        return self._capacity[id(worker)] - self._claimed[id(worker)]

    def qsize(self):
        with self._ready:
            return sum(self._free(worker) for worker in self._workers)

    def _claim(self):
        worker = next((worker for worker in self._workers if self._free(worker) > 0), None)
        if worker is not None:
            self._claimed[id(worker)] += 1
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
            self._claimed[id(worker)] -= 1
            self._ready.notify_all()

    def offered(self):
        with self._ready:
            return list(self._workers)

    def set_aside(self, worker):
        """Offer none of ``worker``'s slots until it is restored."""
        with self._ready:
            if worker in self._workers:
                self._workers.remove(worker)
            self._ready.notify_all()

    def restore(self, worker):
        with self._ready:
            if worker not in self._workers:
                self._workers = [known for known in self._order if known in self._workers or known is worker]
            self._ready.notify_all()

    def wait_idle(self, worker, stop=None, poll=0.5):
        """Wait until none of ``worker``'s slots is claimed (its takes have finished); False if ``stop()`` first."""
        with self._ready:
            while self._claimed[id(worker)] > 0:
                if stop is not None and stop():
                    return False
                self._ready.wait(poll)
            return True


class _WorkerGone(Exception):
    """A take's remote worker has gone and was set aside; the pool sends the take to another worker."""


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

    def work_status(self):
        """What this node reports doing now (rows decoding/prefilling/waiting), or None.

        Observing never creates an engine or loads a model: a worker whose
        engine does not exist yet reports nothing.
        """
        engine = self._engine
        if engine is None:
            return None
        if self.local:
            from .drama3_batch_scheduler import engine_batch_status
            return engine_batch_status(engine)
        probe = getattr(engine, "work_status", None)
        return probe() if callable(probe) else None


class _Recognizer:
    """One recognition endpoint: the controller's own thread (client None) or a serve worker."""

    def __init__(self, name, concurrency, client=None):
        self.settings = {"name": name, "concurrency": concurrency}
        self.client = client

    @property
    def local(self):
        return self.client is None


MAX_SERVER_RESTARTS = 3  # starts of the auto-started local server per render before the render stops
REPROBE_SECONDS = 30.0  # how often a set-aside worker is asked whether it is back


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
        # The auto-started loopback server behind routed local slots (auto_serve.shared_worker_config).
        self._auto_serve = config.get("_auto_serve")
        self._server_epoch = 0  # bumped each time that server is started again
        self._server_restarts = 0
        self._revive_lock = RLock()
        self._workers = [_Worker(worker, config["engine"], self.sampling)
                         for worker in settings["workers"] if "tts" in worker.get("roles", ["tts"])]
        if not self._workers:
            raise EngineError("A render pool needs at least one TTS worker")
        # Speech nodes a person turned off (run shell ``o``): id(worker) -> "draining" (finishing its
        # takes), "stopping" (shutting down), "off" or "starting"; a node not listed is on.
        self._switched = {}
        self._switch_lock = RLock()
        from .recognition_workers import RemoteRecognizer
        self._recognizers = [_Recognizer("local", 1)] + [
            _Recognizer(worker["name"], worker.get("recognition_concurrency", 1), RemoteRecognizer(worker))
            for worker in settings["workers"] if "recognition" in worker.get("roles", ())]
        self._recognition_slots = _Slots(self._recognizers)
        from .node_activity import ACTIVITY
        for worker in self._workers:
            ACTIVITY.register(self, worker.settings["name"], "speech", worker.settings.get("concurrency", 1),
                              local=worker.local or bool(worker.settings.get("auto_serve")),
                              probe=worker.work_status,
                              switch=lambda on, notify=None, worker=worker: self.switch_worker(worker, on, notify),
                              state=lambda worker=worker: self.worker_state(worker))
        for recognizer in self._recognizers:
            ACTIVITY.register(self, recognizer.settings["name"], "verify", recognizer.settings["concurrency"],
                              local=recognizer.local)
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
        # Remote workers that gave no answer, rendering without them until they answer again.
        self._away = {}  # id(worker) -> (worker, notify)
        self._away_lock = RLock()
        self._probe_stop = Event()
        self._prober = None

    def check_continuity(self, notify=None):
        """Every worker must carry chapter history; checked once per pool.

        A remote worker that gives no answer at all (switched off, asleep,
        or off the network) is set aside, and the render carries on with the
        others; it is asked every ``REPROBE_SECONDS`` whether it is back. A
        worker that answers wrongly (an old protocol, a refused key) still
        stops the render, and so does finding no worker that answers.
        """
        if self._continuity_checked:
            return
        from .continuity import require_support
        from .fish_http import FishHTTPError
        def check():
            for worker in self._workers:
                if worker.local:
                    require_support(worker.engine())
        self._local_runtime.run(check)
        unreachable = []
        for worker in self._workers:
            if worker.local or worker in self._away_workers():
                continue
            try:
                require_support(worker.engine())
            except FishHTTPError as error:
                if error.status is not None:
                    raise
                unreachable.append((worker, error))
        if unreachable and len(unreachable) == len(self._workers):
            raise unreachable[0][1]
        for worker, error in unreachable:
            self._set_aside(worker, f"does not answer ({error})", notify)
        self._continuity_checked = True

    @property
    def set_aside_workers(self):
        """Names of the remote workers this render is going on without, in configured order."""
        away = self._away_workers()
        return [worker.settings["name"] for worker in self._workers if worker in away]

    def _away_workers(self):
        with self._away_lock:
            return [worker for worker, _ in self._away.values()]

    def _set_aside(self, worker, reason, notify=None):
        say = notify or generation_notifier() or (lambda message: None)
        with self._away_lock:
            if id(worker) in self._away or self._switched_state(worker) is not None:
                return
            self._slots.set_aside(worker)
            self._away[id(worker)] = (worker, say)
            if self._prober is None or not self._prober.is_alive():
                self._prober = Thread(target=self._probe_away, name="render-pool-worker-probe", daemon=True)
                self._prober.start()
        say(f"Render worker {worker.settings['name']} {reason}; rendering without it, and checking every "
            f"{REPROBE_SECONDS:g}s whether it is back.")

    def _worker_gone(self, worker, error, notify):
        """Whether ``error`` means ``worker`` has gone; if so it is set aside and the take goes elsewhere.

        Only a remote worker's transport failure with no HTTP answer counts,
        and only when a short ``/health`` request gets no answer either: a
        worker that answers is slow or refusing, an ordinary failed attempt.
        The last offered worker is never set aside (its failures stay
        ordinary), so a render never waits on nobody.
        """
        if (worker.local or worker.settings.get("auto_serve") or not getattr(error, "transport_error", False)
                or getattr(error, "status", None) is not None):
            return False
        with self._away_lock:
            if id(worker) in self._away:
                return True  # another take found it gone already
        if worker not in self._slots.offered() or len(self._slots.offered()) <= 1:
            return False
        reachable = getattr(worker.engine(), "reachable", None)
        if not callable(reachable) or reachable():
            return False
        with self._away_lock:
            if id(worker) in self._away:
                return True
            if len(self._slots.offered()) <= 1:
                return False
            self._set_aside(worker, f"stopped answering ({error})", notify)
        return True

    def _probe_away(self):
        """Ask each set-aside worker every ``REPROBE_SECONDS`` whether it is back; restore those that are."""
        from .continuity import require_support
        while not self._probe_stop.wait(REPROBE_SECONDS):
            with self._away_lock:
                away = list(self._away.values())
                if not away:
                    self._prober = None
                    return
            for worker, say in away:
                if self._probe_stop.is_set():
                    return
                engine = worker.engine()
                try:
                    recheck = getattr(engine, "recheck_continuity", None)
                    recheck() if callable(recheck) else require_support(engine)
                except Exception:
                    continue  # still away (or not yet right): ask again later
                with self._away_lock:
                    if self._away.pop(id(worker), None) is None or self._switched_state(worker) is not None:
                        continue  # turned off (or on) by a person meanwhile
                    self._slots.restore(worker)
                if self._resources is not None:
                    self._resources.tts.wake()
                say(f"Render worker {worker.settings['name']} answers again; sending it speech again.")

    def _switched_state(self, worker):
        # Read without the switch lock (writes hold it): callers hold the away or slot locks, which
        # ``switch_worker`` takes inside the switch lock.
        return self._switched.get(id(worker))

    def worker_state(self, worker):
        """``on``, ``away`` (gave no answer), ``turning off``, ``shutting down``, ``off`` or ``turning on``."""
        state = self._switched_state(worker)
        if state is not None:
            return {"draining": "turning off", "stopping": "shutting down", "starting": "turning on"}.get(state, state)
        with self._away_lock:
            return "away" if id(worker) in self._away else "on"

    def switch_worker(self, worker, on, notify=None):
        """Turn a speech node off or back on for the rest of this render; returns what happens, in plain words.

        Off: no new take goes to it, and takes already on it finish. Then the
        node gives back what it holds: the auto-started local speech server is
        stopped (its model's memory freed; another program using it may start
        it again), and an in-process local model is unloaded. A remote node
        stays running for its other clients. With every node off, the render
        waits for one to be turned on. On: the local server is started again
        (loading its model takes a while) or a remote node is asked whether
        it answers, and takes go to it again. Nothing about a take changes.
        """
        say = notify or generation_notifier() or (lambda message: None)
        name = worker.settings["name"]
        if self._closed:
            return f"{name}: this render has ended"
        with self._switch_lock:
            state = self._switched.get(id(worker))
            if not on:
                if state is not None:
                    return f"{name} is {self.worker_state(worker)} already" if state != "starting" else (
                        f"{name} is turning on; turn it off again once it is on")
                self._switched[id(worker)] = "draining"
                with self._away_lock:
                    self._away.pop(id(worker), None)
                self._slots.set_aside(worker)
                busy = self._slots._claimed[id(worker)]
            else:
                if state is None:
                    return f"{name} is on already"
                if state in ("stopping", "starting"):
                    return f"{name} is {self.worker_state(worker)}; try again in a moment"
                if state == "draining":
                    # Not shut down yet: it simply takes speech again.
                    del self._switched[id(worker)]
                    self._restore(worker)
                    return f"{name} stays on"
                self._switched[id(worker)] = "starting"
        if not on:
            Thread(target=self._turn_off, args=(worker, say), name=f"render-node-off-{name}", daemon=True).start()
            what = ("then the local speech server stops" if worker.settings.get("auto_serve")
                    else "then its model is unloaded" if worker.local else "it keeps running for other clients")
            return (f"Turning {name} off: no new speech goes to it"
                    + (f"; its {busy} take(s) in progress finish first" if busy else "") + f", {what}.")
        Thread(target=self._turn_on, args=(worker, say), name=f"render-node-on-{name}", daemon=True).start()
        return (f"Turning {name} on: starting the local speech server (loading its model can take a few minutes)."
                if worker.settings.get("auto_serve") else f"Turning {name} on.")

    def _restore(self, worker):
        self._slots.restore(worker)
        if self._resources is not None:
            self._resources.tts.wake()

    def _turn_off(self, worker, say):
        name = worker.settings["name"]
        if not self._slots.wait_idle(worker, stop=lambda: self._closed or self._switched_state(worker) != "draining"):
            return
        with self._switch_lock:
            if self._switched.get(id(worker)) != "draining":
                return  # turned on again while its takes finished
            self._switched[id(worker)] = "stopping"
        detail = ""
        try:
            detail = self._release_node(worker)
        except Exception as error:  # noqa: BLE001 - reported; the node stays off either way
            detail = f"; it could not give back its resources ({error})"
        with self._switch_lock:
            self._switched[id(worker)] = "off"
        say(f"Render node {name} is off{detail}. o turns it on again.")
        if not self._slots.offered():
            say("Every speech node is off: the render waits until one is turned on (o).")

    def _release_node(self, worker):
        """Give back what a node that is off holds; returns how, as the end of a sentence."""
        if worker.settings.get("auto_serve") and self._auto_serve is not None:
            others = [other for other in self._workers if other is not worker and other.settings.get("auto_serve")
                      and self._switched_state(other) not in ("stopping", "off")]
            if others:
                return "; the local speech server keeps running for " + ", ".join(
                    other.settings["name"] for other in others)
            from .auto_serve import AutoServeError, stop_server
            with self._revive_lock:
                try:
                    stopped = stop_server(self._auto_serve)
                except AutoServeError as error:
                    return f"; the local speech server is still running ({error})"
            return ("; the local speech server has stopped and its memory is free" if stopped
                    else "; the local speech server was not running")
        if worker.local:
            def unload():
                with self.local_lock:
                    worker.close()
                    worker._engine = None
            self._local_runtime.run(unload)
            return "; its model is unloaded and its memory is free"
        return " (it keeps running for other clients)"

    def _turn_on(self, worker, say):
        name = worker.settings["name"]
        try:
            if worker.settings.get("auto_serve") and self._auto_serve is not None:
                from .auto_serve import resolve_small_job_engine
                with self._revive_lock:
                    resolve_small_job_engine(self._auto_serve, notify=say)
                    self._server_epoch += 1
            if not worker.local:
                engine = worker.engine()
                reachable = getattr(engine, "reachable", None)
                if callable(reachable) and not reachable():
                    with self._switch_lock:
                        self._switched.pop(id(worker), None)
                    self._set_aside(worker, "does not answer", say)
                    return
                recheck = getattr(engine, "recheck_continuity", None)
                if callable(recheck):
                    recheck()
        except Exception as error:  # noqa: BLE001 - reported; the node stays off
            with self._switch_lock:
                self._switched[id(worker)] = "off"
            say(f"Render node {name} could not be turned on ({error}); it stays off.")
            return
        with self._switch_lock:
            if self._closed:
                return
            self._switched.pop(id(worker), None)
            self._restore(worker)
        say(f"Render node {name} is on again; sending it speech"
            + (" (its model loads with its first take)." if worker.local else "."))

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
        from .node_activity import ACTIVITY, current_job
        job = current_job()

        def run_on(recognizer):
            with ACTIVITY.busy(recognizer.settings["name"], "verify", job):
                return transcribe_on(recognizer)

        def transcribe_on(recognizer):
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

    def _generate(self, worker, engine, method, args, kwargs, notify):
        """Call the engine; if the render's own local server has gone, start it again and retry once.

        Only a slot routed to the auto-started loopback server qualifies, and
        only a transport failure with no HTTP answer at all. The server is
        started again (at most ``MAX_SERVER_RESTARTS`` times per render) only
        when nothing listens on its port; a server that answered, is busy,
        or is loading keeps the failure an ordinary failed attempt. When it
        cannot be started, the render stops (``RenderServiceUnavailable``)
        instead of failing every later take the same way.
        """
        from .render_pause import PAUSE
        epoch = self._server_epoch
        try:
            # A paused render starts no request, and one whose connection the
            # pause lost is sent again on resume (``render_pause``).
            return PAUSE.call(getattr(engine, method), *args, **kwargs)
        except EngineError as error:
            if not (self._auto_serve is not None and worker.settings.get("auto_serve")
                    and getattr(error, "transport_error", False) and getattr(error, "status", None) is None):
                raise
            self._revive_server(epoch, error, notify)
        from .interrupts import stop_if_requested
        stop_if_requested()
        return getattr(engine, method)(*args, **kwargs)

    def _revive_server(self, epoch, error, notify):
        from .auto_serve import AutoServeError, resolve_small_job_engine, server_absent
        from .workflow import RenderServiceUnavailable
        say = notify or (lambda message: None)
        advice = ("macOS may have stopped it for lack of memory: close other programs that load models (a cast "
                  "session, another render), then run the render again; finished chapters are kept")
        with self._revive_lock:
            if self._server_epoch != epoch:
                return  # another slot already started it again
            try:
                absent = server_absent(self._auto_serve)
            except AutoServeError as exc:
                raise RenderServiceUnavailable(f"The local speech server stopped answering: {exc}") from error
            if not absent:
                raise error  # it is running: an ordinary failed request
            if self._server_restarts >= MAX_SERVER_RESTARTS:
                raise RenderServiceUnavailable(
                    f"The local speech server stopped again after {self._server_restarts} restart(s) in this "
                    f"render; {advice}.") from error
            self._server_restarts += 1
            say("The local speech server stopped (nothing answers on its port); starting it again and retrying "
                "its requests.")
            try:
                resolve_small_job_engine(self._auto_serve, notify=say)
            except AutoServeError as exc:
                raise RenderServiceUnavailable(
                    f"The local speech server stopped and could not be started again ({exc}); {advice}.") from error
            self._server_epoch += 1

    def _execute(self, method, *args, **kwargs):
        """Run one take on the first free worker; a take whose worker has gone is sent to another."""
        while True:
            try:
                return self._execute_once(method, args, kwargs)
            except _WorkerGone:
                continue  # that worker is set aside now; nothing was generated

    def _execute_once(self, method, args, kwargs):
        from .interrupts import stop_if_requested
        stop_if_requested()
        if self._closed:
            raise EngineError("The render pool is closed")
        if self._resources is not None:
            return self._execute_on_lane(method, args, kwargs)
        notify = generation_notifier()
        from .node_activity import ACTIVITY, current_job
        job = current_job()
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
                with METRICS.active("synthesis"), ACTIVITY.busy(name, "speech", job):
                    try:
                        engine = worker.engine()
                        if worker.local and not worker.batched and requires_mlx(engine):
                            self._local_runtime.initialize_mlx()
                        metadata = self._generate(worker, engine, method, args, kwargs, notify)
                        if not isinstance(metadata, dict):
                            raise EngineError("TTS worker returned invalid generation metadata")
                        result = {**metadata, "render_worker": name, "worker_engine": engine.identity}
                    except BaseException as error:
                        report_worker_activity(notify, "failed", name, monotonic() - started)
                        if self._worker_gone(worker, error, notify):
                            raise _WorkerGone() from error
                        raise
                    report_worker_activity(notify, "completed", name, monotonic() - started)
                    ACTIVITY.made(name, _audio_seconds(result))
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
        from .node_activity import ACTIVITY, current_job
        job = current_job()
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
                    with ACTIVITY.busy(name, "speech", job):
                        engine = worker.engine()
                        if worker.local and not worker.batched and requires_mlx(engine):
                            self._local_runtime.initialize_mlx()
                        metadata = self._generate(worker, engine, method, args, kwargs, notify)
                    if not isinstance(metadata, dict):
                        raise EngineError("TTS worker returned invalid generation metadata")
                    result = {**metadata, "render_worker": name, "worker_engine": engine.identity}
                except BaseException as error:
                    report_worker_activity(notify, "failed", name, monotonic() - started)
                    if self._worker_gone(worker, error, notify):
                        raise _WorkerGone() from error
                    raise
                report_worker_activity(notify, "completed", name, monotonic() - started)
                ACTIVITY.made(name, _audio_seconds(result))
                return result
            if worker.local and not worker.batched:
                def local_run():
                    with self.local_lock:
                        return run()
                return self._local_runtime.run(local_run)
            return run()

        report_worker_activity(notify, "waiting")
        from .render_pause import PAUSE
        PAUSE.wait()  # a paused take waits here, holding no lane thread or worker slot
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
        if self._switched_state(worker) is not None:
            raise EngineError(f"Render node {worker.settings['name']} is turned off; turn it on (o) to create voices with it.")
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
        self._probe_stop.set()
        from .node_activity import ACTIVITY
        ACTIVITY.unregister(self)
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
