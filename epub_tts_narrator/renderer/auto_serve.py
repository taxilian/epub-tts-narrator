"""Discover or spawn a local ``serve`` worker for small interactive jobs.

Pronunciation previews and ``pronounce-name`` auditions synthesize a few short
clips. Loading a private Drama 3 MLX engine per process can exhaust memory
while a render is already running, so with ``server.auto_start`` these jobs
reuse a running local ``serve`` worker on ``server.port`` and, when none is
running, start one as a detached child that exits itself after
``server.idle_timeout_seconds`` (see ``server._idle_watchdog``).

Caveat: ``/health`` advertises the wire protocol and the model family but not
the full engine identity, so protocol + model is the accepted compatibility
check here. A port owned by any other process fails closed; nothing is ever
killed.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .engines import EngineError


SPAWN_READY_SECONDS = 300.0  # a cold model load and quantization can take minutes
SPAWN_POLL_SECONDS = 1.0
_PROBE_TIMEOUT_SECONDS = 5.0
_MAX_HEALTH_BYTES = 1024 * 1024
LOG_TAIL_CHARACTERS = 2000


class _NoRedirects(HTTPRedirectHandler):
    """Never follow a health redirect; the probe must not forward its key."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_PROBE_OPENER = build_opener(_NoRedirects())

# Probe outcomes for a port without a usable health answer.
_ABSENT = "absent"    # nothing is listening (connection refused)
_WARMING = "warming"  # something accepts connections but not health yet (model loading)


class AutoServeError(EngineError):
    """The local auto-serve worker could not be resolved; fail closed."""


def _cache_root():
    return Path.home() / ".cache" / "epub-tts-narrator"


def _probe(port, *, api_key="", timeout=_PROBE_TIMEOUT_SECONDS):
    """One ``GET /health`` probe: a health dict, ``_ABSENT`` or ``_WARMING``.

    Any HTTP answer that is not our health object raises: the port belongs to
    a foreign process and spawning would be wrong.
    """
    request = Request(f"http://127.0.0.1:{port}/health")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _PROBE_OPENER.open(request, timeout=timeout) as response:
            body = response.read(_MAX_HEALTH_BYTES)
    except HTTPError as exc:
        code = exc.code
        exc.close()
        if code == 401:
            raise AutoServeError(f"A renderer serve worker on port {port} uses a different server.api_key; "
                                 "run this command with the server's config, or free the port or change server.port") from None
        raise AutoServeError(f"Port {port} answers but is not a renderer serve worker (HTTP {code}); "
                             "free the port or change server.port") from None
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError):
            return _WARMING
        if isinstance(reason, OSError):
            return _ABSENT
        raise AutoServeError(f"Port {port} could not be probed ({type(reason).__name__})") from exc
    except TimeoutError:
        return _WARMING
    except OSError:
        return _ABSENT
    try:
        health = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise AutoServeError(f"Port {port} answers but did not return a health object; "
                             "free the port or change server.port") from None
    if not isinstance(health, dict):
        raise AutoServeError(f"Port {port} answers but did not return a health object; "
                             "free the port or change server.port")
    return health


def _expected_models(config):
    """The /health model values this configuration's engine accepts."""
    from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
    selected = (config.get("engine") or {}).get("model", DEFAULT_DRAMA3_MODEL)
    repo = get_drama3_model(selected).repo
    return ({"drama-3", DEFAULT_DRAMA3_MODEL} if repo == DEFAULT_DRAMA3_MODEL else {repo})


def _client_model(config):
    from .drama3_models import DEFAULT_DRAMA3_MODEL
    selected = (config.get("engine") or {}).get("model", DEFAULT_DRAMA3_MODEL)
    return "drama-3" if selected == DEFAULT_DRAMA3_MODEL else selected


def _client_sampling(config):
    from .config import SAMPLING_DEFAULTS
    engine = config.get("engine") or {}
    return {key: engine.get(key, value) for key, value in SAMPLING_DEFAULTS.items()}


def _verified_client(config, port, health, *, api_key=""):
    """Build the pool-style client after the health body proves compatibility."""
    from .fish_http import PROTOCOL, FishHTTPEngine
    if health.get("protocol") != PROTOCOL:
        raise AutoServeError(f"Port {port} answers but is not a renderer serve worker "
                             f"(protocol {health.get('protocol')!r}); free the port or change server.port")
    expected = _expected_models(config)
    if health.get("model") not in expected:
        raise AutoServeError(f"The serve worker on port {port} provides model {health.get('model')!r} but this "
                             f"configuration uses {sorted(expected)}; change server.port or the worker's engine.model")
    settings = {"url": f"http://127.0.0.1:{port}/v1/tts", "protocol": "fish_inline", "name": "local-serve",
                "model": _client_model(config), "api_key": api_key,
                "timeout_seconds": (config.get("server") or {}).get("request_timeout_seconds", 600)}
    return FishHTTPEngine(settings, sampling=_client_sampling(config))


@contextmanager
def _exclusive_spawn_lock(port):
    """Serialize discover-or-spawn across processes; the lock file is never deleted."""
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    handle = open(root / f"auto-serve-{port}.lock", "a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        handle.close()


def _wait_ready(port, budget, *, child=None, api_key=""):
    """Poll /health until a serve worker answers, the budget ends, or a spawned child exits."""
    deadline = time.monotonic() + budget
    while True:
        health = _probe(port, api_key=api_key)
        if isinstance(health, dict):
            return health
        if child is not None and child.poll() is not None:
            return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(SPAWN_POLL_SECONDS)


def _log_tail(log_path, limit=LOG_TAIL_CHARACTERS):
    try:
        data = Path(log_path).read_bytes()[-limit:]
    except OSError:
        return "(the log could not be read)"
    return data.decode("utf-8", errors="replace").strip() or "(the log is empty)"


def _write_state_note(root, port, pid):
    note = {"pid": pid, "port": port,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    path = root / f"auto-serve-{port}.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _check_model_available(config):
    checkpoint = (config.get("engine") or {}).get("checkpoint")
    if not checkpoint or not Path(str(checkpoint)).exists():
        raise AutoServeError("server.auto_start needs the local Drama 3 model: set engine.checkpoint to the "
                             "downloaded model directory (audiobook-render doctor checks it)")


def _spawn_and_connect(config, port, notify=None):
    config_path = config.get("_config_path")
    if not config_path:
        raise AutoServeError("server.auto_start did not record the configuration file path; start the server "
                             "manually with audiobook-render serve")
    _check_model_available(config)
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f"serve-{port}.log"
    from .server_config import ensure_server_api_key
    try:
        # The spawned serve generates and saves a key when missing (the serve
        # CLI does this unconditionally); ensure it here so the readiness
        # probes and this client speak the same key the child will use.
        api_key, _ = ensure_server_api_key(Path(config_path))
    except (OSError, ValueError) as exc:
        raise AutoServeError(f"Could not prepare server.api_key for the local serve worker "
                             f"({type(exc).__name__}); start the server manually with audiobook-render serve") from exc
    command = [sys.executable, "-m", "epub_tts_narrator.renderer", "serve",
               "--config", str(config_path), "--port", str(port)]
    if notify is not None:
        notify(f"No local serve worker on port {port}; starting one (log: {log_path}).")
    try:
        with open(log_path, "ab") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True)
    except OSError as exc:
        raise AutoServeError(f"Could not start the local serve worker on port {port} "
                             f"({type(exc).__name__}); its log is {log_path}") from exc
    _write_state_note(root, port, child.pid)
    health = _wait_ready(port, SPAWN_READY_SECONDS, child=child, api_key=api_key)
    if isinstance(health, dict):
        if notify is not None:
            notify(f"Local serve worker on port {port} is ready.")
        return _verified_client(config, port, health, api_key=api_key)
    raise AutoServeError(f"The local serve worker on port {port} did not become ready within "
                         f"{SPAWN_READY_SECONDS:g}s; check its log {log_path}. Last lines:\n{_log_tail(log_path)}")


def resolve_small_job_engine(config, notify=None):
    """Return a client engine for the local serve worker, starting one if needed.

    Probes ``http://127.0.0.1:{server.port}/health``; a compatible worker
    becomes a ``FishHTTPEngine`` client. Otherwise, under an exclusive lock,
    the port is re-probed (another client may have spawned a worker), a
    warming worker is waited for, and only a truly absent port spawns a
    detached ``audiobook-render serve`` child from ``config["_config_path"]``.
    Raises ``AutoServeError`` with the serve log path when spawning or
    readiness fails, and never kills or reconfigures a foreign process.
    """
    if (config.get("engine") or {}).get("type", "drama3_mlx") != "drama3_mlx":
        raise AutoServeError("server.auto_start reuses the local Drama 3 serve worker, so it requires "
                             "engine.type = drama3_mlx")
    server = config.get("server") or {}
    port = int(server.get("port", 8080))
    api_key = server.get("api_key", "")
    health = _probe(port, api_key=api_key)
    if isinstance(health, dict):
        return _verified_client(config, port, health, api_key=api_key)
    with _exclusive_spawn_lock(port):
        health = _probe(port, api_key=api_key)
        if isinstance(health, dict):
            return _verified_client(config, port, health, api_key=api_key)
        if health == _WARMING:
            # Another client (or an operator) started a worker that is still
            # loading its model; wait for it instead of spawning a second one.
            health = _wait_ready(port, SPAWN_READY_SECONDS, api_key=api_key)
            if isinstance(health, dict):
                return _verified_client(config, port, health, api_key=api_key)
            raise AutoServeError(f"The serve worker on port {port} did not become ready within "
                                 f"{SPAWN_READY_SECONDS:g}s; start it manually with audiobook-render serve "
                                 "to see its startup error")
        return _spawn_and_connect(config, port, notify=notify)
