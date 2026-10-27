"""Production chapter service and real client delivery for the benchmark run.

Starts one in-process ChapterJobs service (real ChapterExecutor, real HTTP
routes, sessions, ACK lifecycle) bound to loopback with an ephemeral port, then
drives it with the real ChapterClient: capability negotiation, lossless voice
registration, run-scoped idempotent submissions, heartbeat, event watching,
verified downloads and acknowledgement. Nothing here replaces the chapter loop
with a microbenchmark worker; only the two model factories can be faked.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.renderer.bundle import chunk_speakers
from epub_tts_narrator.renderer.chapter_client import (
    ChapterClient, ClientHeartbeat, _cached_artifacts_present, _read_cache,
    _register_voices, _submit_chapter, _voice_assets, _wait_jobs,
)
from epub_tts_narrator.renderer.chapter_executor import ChapterExecutor
from epub_tts_narrator.renderer.chapter_jobs import ChapterJobs
from epub_tts_narrator.renderer.chapter_protocol import PROTOCOL, validate_spec
from epub_tts_narrator.renderer.chapter_server import create_app
from epub_tts_narrator.renderer.common import digest

TOTAL_CHAPTERS = 5  # one warmup chapter plus four measured benchmark groups


class ChapterService:
    """One live production chapter service on loopback; teardown joins workers."""

    def __init__(self, config, *, fake_engine=False):
        self.config = config
        self._fake_engine = fake_engine
        self._restore_models = None
        self.jobs = None
        self._server = self._thread = None

    def start(self):
        if self._fake_engine:
            from scripts.chapter_benchmark_fakeengine import install_fake_models
            self._restore_models = install_fake_models()
        settings = dict(self.config["chapter_server"])
        executor = ChapterExecutor(self.config)
        job_settings = {**settings, "ffprobe": self.config["render"]["ffprobe"]}
        self.jobs = ChapterJobs(settings["data_dir"], executor, job_settings)
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(create_app(self.jobs, settings),
                                               host="127.0.0.1", port=0, log_level="warning"))
        self._server = server
        self._thread = threading.Thread(target=server.run, name="benchmark-chapter-server", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            if not self._thread.is_alive():
                raise RuntimeError("The benchmark chapter service failed to start")
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError("The benchmark chapter service did not start in time")
        port = server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def close(self):
        try:
            if self._server is not None:
                self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=60)
            # Lifespan close normally joins the workers; this is a safety net
            # for a failed transport so no coordinator thread outlives the run.
            if self.jobs is not None and self._thread is not None and self._thread.is_alive():
                self.jobs.close()
        finally:
            if self._restore_models is not None:
                self._restore_models()
                self._restore_models = None


def connect(config, base_url):
    """Real capability negotiation against the production HTTP routes."""
    client = ChapterClient({**config["remote_chapters"], "url": base_url, "websocket": False})
    capabilities = client.request("GET", "v1/capabilities")
    if capabilities.get("protocol") != PROTOCOL:
        raise ValueError("Chapter service uses an incompatible protocol")
    if capabilities.get("client_sessions") != 1 or capabilities.get("ephemeral_runs") != 1:
        raise ValueError("The chapter service lacks connected ephemeral runs; update the worker")
    return client, capabilities


def benchmark_chapter(corpus, group):
    """The benchmark-only chapter for a group; group None is the warmup chapter."""
    if group is None:
        chunks = [entry["chunk"] for entry in corpus["warmup"]]
        number, title = 1, "Warmup"
    else:
        chunks = [entry["chunk"] for entry in corpus["cases"] if entry["group"] == group]
        number, title = group + 2, f"Benchmark group {group + 1}"
    return {"id": f"chapter-{number:04d}", "number": number, "title": title, "chunks": chunks}


def build_selection(corpus, group, bindings):
    chapter = benchmark_chapter(corpus, group)
    used = set()
    for chunk in chapter["chunks"]:
        used.update(speaker["voice_id"] for speaker in chunk_speakers(chunk))
    return {"version": 1, "book": deepcopy(corpus["book"]),
            "source_fingerprint": corpus["source_fingerprint"], "chapter": chapter,
            "total_chapters": TOTAL_CHAPTERS,
            "voice_bindings": {voice_id: bindings[voice_id] for voice_id in sorted(used)},
            "render": deepcopy(corpus["render"])}


def group_folder(work, selection):
    """Same stable selection-bound folder identity the production client uses."""
    return Path(work) / "remote-chapters" / digest(selection)


def cached_record(folder, selection):
    """A validated downloaded-chapter checkpoint, or None; corruption fails closed."""
    descriptor = _read_cache(folder, selection)
    if descriptor is not None and _cached_artifacts_present(folder, descriptor):
        return descriptor
    return None


def telemetry_snapshot(pool):
    """Payload-free lane/backlog/logical gauges while the pool is still alive.

    The backlog snapshot is captured wholesale, so peak gauges the core adds
    later flow into reports automatically. Legacy scheduling has no lane
    runtime or backlog budget; the caller reports those as unavailable.
    """
    if pool is None or pool.resources is None or pool.logical is None:
        return None
    lanes = {}
    for name, lane in (("tts", pool.resources.tts), ("preparation", pool.resources.preparation),
                       ("recognition", pool.resources.recognition), ("review", pool.resources.review),
                       ("review_coordination", pool.resources.review_coordination)):
        lanes[name] = {"workers": len(lane._threads), "max_pending": lane._max_pending,
                       "pending_now": lane._pending}
    return {"lanes": lanes, "backlog": pool.resources.backlog.snapshot(),
            "logical": {"global_limit": pool.logical.global_limit, "chapter_share": pool.logical.share},
            "runtime_limits": {"max_inflight": pool.max_inflight, "max_parallel": pool.max_parallel}}


class Delivery:
    """The production client session: submission budget, heartbeat, ACK waits."""

    def __init__(self, client, capabilities, work, notify):
        self.client, self.capabilities, self.notify = client, capabilities, notify
        self.work = Path(work)
        self.submission = os.urandom(32).hex() if capabilities.get("submission_budget") is True else None
        self.session = os.urandom(32).hex()
        client.request("POST", "v1/sessions", {"id": self.session})
        ttl = capabilities.get("session_ttl_seconds")
        interval = max(1.0, min(10.0, ttl / 6)) if type(ttl) in (int, float) and ttl > 0 else 5.0
        self.heartbeat = ClientHeartbeat(client, self.session, interval)
        self.heartbeat.start()

    def register_voices(self, voices):
        """Lossless FLAC upload through the production ensure/register routes."""
        bindings, assets = _voice_assets(voices, self.work, None)
        _register_voices(self.client, assets, self.notify)
        return bindings

    def submit(self, corpus, group, bindings):
        """One run-scoped idempotent submission through the production route."""
        selection = build_selection(corpus, group, bindings)
        spec = deepcopy(selection)
        spec["renderer_fingerprint"] = self.capabilities["renderer_fingerprint"]
        validate_spec(spec)
        folder = group_folder(self.work, selection)
        record, _status = _submit_chapter(self.client, selection, spec, submission=self.submission,
                                          session=self.session, notify=self.notify)
        return selection, folder, record

    def wait(self, pending):
        """Real event watching, verified downloads, and acknowledged delivery."""
        _wait_jobs(self.client, pending, self.notify, include_masters=True,
                   session=self.session, watchdog=self.heartbeat)

    def close(self):
        self.heartbeat.stop()
        try:
            self.client.request("DELETE", f"v1/sessions/{self.session}")
        except Exception:
            pass
