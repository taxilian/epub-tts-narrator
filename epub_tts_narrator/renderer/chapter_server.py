"""Authenticated HTTP and reconnectable WebSocket transport for chapter jobs.

The job service owns all work. Connections only submit work or observe durable
state, so losing a socket cannot cancel a chapter. FastAPI and uvicorn are
optional dependencies, imported only when this server is started.
"""

import asyncio
import base64
import binascii
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
import hmac
import ipaddress
import json
import logging
import math
from pathlib import Path
import re
from urllib.parse import quote


LOGGER = logging.getLogger(__name__)
_ID = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TERMINAL = {"complete", "failed", "cancelled"}
_ARTIFACT_TYPES = {
    ".m4a": "audio/mp4", ".m4b": "audio/mp4", ".mp3": "audio/mpeg",
    ".flac": "audio/flac", ".json": "application/json",
}


def _require_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Identifier must be a lowercase SHA-256 hash")
    return value


def _require_name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("Invalid artifact name")
    if Path(value).suffix.lower() not in _ARTIFACT_TYPES:
        raise ValueError("Only compressed audio and JSON artifacts can be downloaded")
    return value


def _is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _validate_settings(settings):
    host = settings.get("host", "127.0.0.1")
    key = settings.get("api_key", "")
    if not isinstance(host, str) or not host:
        raise ValueError("chapter_server.host must be a nonempty host")
    if not isinstance(key, str) or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValueError("chapter_server.api_key must not contain whitespace or control characters")
    if not _is_loopback(host) and not key:
        raise ValueError("A public chapter server requires chapter_server.api_key")
    if type(settings.get("max_request_bytes", 100 * 1024 * 1024)) is not int or settings.get("max_request_bytes", 100 * 1024 * 1024) <= 0:
        raise ValueError("chapter_server.max_request_bytes must be a positive integer")
    return host, key


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON objects must not contain duplicate keys")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("JSON must not contain non-finite numbers")


def create_app(jobs, settings=None, *, shutdown_callback=None):
    """Create an app around an already configured persistent ChapterJobs service.

    ``shutdown_callback`` is a provider lifecycle operation, never an OS shutdown
    or process exit. Core admission must close atomically before that callback
    can run. With no callback the server never automatically shuts down.
    """
    settings = dict(settings or {})
    _, api_key = _validate_settings(settings)
    try:
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, JSONResponse
        from starlette.concurrency import run_in_threadpool
    except ImportError as exc:
        raise RuntimeError("Chapter serving requires the chapter-server optional dependencies") from exc

    max_body = settings.get("max_request_bytes", 100 * 1024 * 1024)
    request_timeout = float(settings.get("request_timeout_seconds", 60))
    event_poll = float(settings.get("event_poll_seconds", 0.25))
    idle_timeout = float(settings.get("idle_timeout_seconds", 600))
    idle_poll = min(5.0, max(0.01, idle_timeout / 10))
    active_uploads = 0
    if (not all(math.isfinite(value) for value in (request_timeout, event_poll, idle_timeout))
            or request_timeout <= 0 or event_poll <= 0 or idle_timeout < 0):
        raise ValueError("Chapter server timeouts must be positive (idle timeout may be zero)")

    async def call(method, *args):
        try:
            return await run_in_threadpool(method, *args)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(404, "Requested chapter, voice, or artifact was not found") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            # Storage/provider exceptions can contain credentials or signed
            # URLs. A failed observation does not change the durable job.
            LOGGER.error("Chapter service operation failed (%s)", type(exc).__name__)
            raise HTTPException(503, "Chapter service operation failed; retry this request") from None

    async def idle_monitor():
        while True:
            await asyncio.sleep(idle_poll)
            if active_uploads:
                continue
            try:
                ready = await run_in_threadpool(jobs.begin_shutdown_if_idle, idle_timeout)
                if not ready:
                    continue
                await run_in_threadpool(shutdown_callback)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                # Provider exceptions can include request credentials. Do not
                # interpolate them into logs returned to users or operators.
                LOGGER.error("Idle shutdown failed; the chapter server is still running")
                reset = getattr(jobs, "cancel_shutdown", None)
                if reset is not None:
                    await run_in_threadpool(reset)

    @asynccontextmanager
    async def lifespan(app):
        monitor = None
        try:
            await run_in_threadpool(jobs.start)
            if shutdown_callback is not None and idle_timeout > 0:
                monitor = asyncio.create_task(idle_monitor())
            yield
        finally:
            if monitor is not None:
                monitor.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor
            # close waits for running work; queued work remains durable for
            # restart. A transport exit never deletes accepted jobs.
            await run_in_threadpool(jobs.close)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.chapter_jobs = jobs

    def authenticated(headers):
        if not api_key:
            return True
        scheme, _, token = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(
            token.encode("utf-8"), api_key.encode("utf-8"),
        )

    @app.middleware("http")
    async def authentication(request: Request, next_handler):
        nonlocal active_uploads
        if not authenticated(request.headers):
            return JSONResponse(
                {"detail": "Bearer authentication required"}, status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        uploading = request.method in {"POST", "PUT"}
        if uploading:
            active_uploads += 1
        try:
            return await next_handler(request)
        finally:
            if uploading:
                active_uploads -= 1

    async def body(request):
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(415, "Use Content-Type application/json")
        if request.headers.get("content-encoding", "identity").lower() != "identity":
            raise HTTPException(415, "Compressed HTTP request bodies are unsupported; upload FLAC audio")
        length = request.headers.get("content-length")
        if length is not None:
            try:
                length = int(length)
            except ValueError as exc:
                raise HTTPException(400, "Invalid Content-Length") from exc
            if length < 0:
                raise HTTPException(400, "Invalid Content-Length")
            if length > max_body:
                raise HTTPException(413, "Request body is too large")
        data = bytearray()
        try:
            async with asyncio.timeout(request_timeout):
                async for part in request.stream():
                    if len(data) + len(part) > max_body:
                        raise HTTPException(413, "Request body is too large")
                    data.extend(part)
        except TimeoutError as exc:
            raise HTTPException(408, "Timed out receiving request body") from exc
        try:
            value = json.loads(data, object_pairs_hook=_json_object, parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise HTTPException(400, "Request body must be valid JSON without duplicate keys") from exc
        if not isinstance(value, dict):
            raise HTTPException(400, "Request body must be a JSON object")
        return value

    def fields(value, allowed, required=()):
        if set(value) - set(allowed) or set(required) - set(value):
            raise HTTPException(400, "Unexpected or missing request fields")

    def with_artifact_urls(value):
        value = deepcopy(value)
        job_id = value.get("id")
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            return value
        for artifact in value.get("artifacts", []):
            if not artifact.get("url"):
                name = _require_name(artifact["name"])
                artifact["url"] = f"/v1/jobs/{job_id}/artifacts/{quote(name, safe='')}"
        return value

    async def get_status(job_id):
        _require_id(job_id)
        return with_artifact_urls(await call(jobs.status, job_id))

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/v1/capabilities")
    async def capabilities():
        return await call(jobs.capabilities)

    @app.post("/v1/voices/ensure")
    async def ensure_voices(request: Request):
        data = await body(request)
        fields(data, {"voices"}, {"voices"})
        voices = data["voices"]
        if not isinstance(voices, list) or len(voices) > 2048:
            raise HTTPException(400, "voices must be an array of at most 2048 hashes")
        for voice in voices:
            _require_id(voice)
        return {"missing": await call(jobs.ensure_voices, list(dict.fromkeys(voices)))}

    @app.put("/v1/voices/{asset_id}")
    async def register_voice(asset_id: str, request: Request):
        _require_id(asset_id)
        data = await body(request)
        fields(data, {"manifest", "audio_base64"}, {"manifest", "audio_base64"})
        manifest, encoded = data["manifest"], data["audio_base64"]
        if not isinstance(manifest, dict) or not isinstance(encoded, str):
            raise HTTPException(400, "Provide a voice manifest and base64 FLAC audio")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(400, "audio_base64 must be valid base64") from exc
        if not audio.startswith(b"fLaC"):
            raise HTTPException(400, "Voice uploads must be FLAC audio")
        return await call(jobs.register_voice, asset_id, manifest, audio)

    @app.post("/v1/jobs", status_code=202)
    async def submit_job(request: Request):
        data = await body(request)
        fields(data, {"spec", "idempotency_key"}, {"spec", "idempotency_key"})
        if not isinstance(data["spec"], dict):
            raise HTTPException(400, "spec must be a chapter specification object")
        key = _require_id(data["idempotency_key"])
        return with_artifact_urls(await call(jobs.submit, data["spec"], key))

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str):
        return await get_status(job_id)

    @app.post("/v1/jobs/{job_id}/retry", status_code=202)
    async def retry_job(job_id: str):
        _require_id(job_id)
        return with_artifact_urls(await call(jobs.retry, job_id))

    def parse_after(value):
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("after must be a nonnegative event sequence") from exc
        if result < 0:
            raise ValueError("after must be a nonnegative event sequence")
        return result

    async def event_batch(job_id, after):
        # Read status before events. A completion between these reads leaves
        # status behind, never ahead of the last event delivered to a client.
        status = await get_status(job_id)
        events = await call(jobs.events, job_id, after)
        return {"events": events, "job": status}

    @app.get("/v1/jobs/{job_id}/events")
    async def poll_events(job_id: str, after: str = "0"):
        return await event_batch(job_id, parse_after(after))

    @app.websocket("/v1/jobs/{job_id}/events")
    async def websocket_events(websocket: WebSocket, job_id: str):
        # The transport is a CLI API. Browsers do not apply CORS to WebSockets,
        # so an unrelated web page must not access an unauthenticated local Pod.
        if not api_key and websocket.headers.get("origin"):
            await websocket.close(code=1008, reason="Browser connections require authentication")
            return
        if not authenticated(websocket.headers):
            await websocket.close(code=1008, reason="Bearer authentication required")
            return
        try:
            _require_id(job_id)
            after = parse_after(websocket.query_params.get("after", "0"))
            initial = await event_batch(job_id, after)
        except (ValueError, HTTPException):
            await websocket.close(code=1008, reason="Invalid or unknown chapter job")
            return
        await websocket.accept()
        receive = asyncio.create_task(websocket.receive())
        try:
            batch = initial
            while True:
                await websocket.send_json(batch)
                for event in batch["events"]:
                    after = max(after, event.get("sequence", event.get("seq", after)))
                caught_up = after >= batch["job"].get("last_event", after)
                if batch["job"]["state"] in _TERMINAL and caught_up:
                    await websocket.close(code=1000)
                    return
                # A terminal job can still have many pages of retained logs.
                # Drain that backlog before closing; socket backpressure limits
                # delivery without retaining a database lock during the send.
                done, _ = await asyncio.wait({receive}, timeout=event_poll if caught_up else 0)
                if done:
                    message = receive.result()
                    if message["type"] == "websocket.disconnect":
                        return
                    receive = asyncio.create_task(websocket.receive())
                batch = await event_batch(job_id, after)
        except (WebSocketDisconnect, OSError):
            return
        except HTTPException:
            await websocket.close(code=1011, reason="Unable to read chapter progress")
        finally:
            receive.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect):
                await receive

    @app.get("/v1/jobs/{job_id}/artifacts/{name}")
    async def download_artifact(job_id: str, name: str):
        _require_id(job_id)
        _require_name(name)
        path = Path(await call(jobs.artifact, job_id, name))
        if path.suffix.lower() not in _ARTIFACT_TYPES:
            raise HTTPException(400, "Uncompressed audio downloads are unsupported")
        return FileResponse(path, filename=name, media_type=_ARTIFACT_TYPES[path.suffix.lower()])

    return app


def serve(config):
    """Start one transport process; GPU and CPU scheduling belong to ChapterJobs."""
    settings = config["chapter_server"]
    _validate_settings(settings)
    verification = config["verification"]
    if not isinstance(verification.get("model"), str) or not verification["model"].strip():
        raise ValueError("serve-chapters requires verification.model to identify a local ASR model; remote_chapters.url only configures the client")
    workers = config.get("render_pool", {}).get("workers", [])
    needs_local_engine = not workers or any(worker["type"] == "local" for worker in workers)
    if (needs_local_engine and config["engine"]["type"] == "drama3_mlx"
            and not config["engine"].get("checkpoint")):
        raise ValueError("serve-chapters requires engine.checkpoint for local Drama 3, or configured remote render_pool workers")
    if verification.get("type") == "faster_whisper":
        from .model_setup import ModelSetup
        # This backend only validates already installed CT2 files; it never
        # downloads or prompts when the service is starting unattended.
        ModelSetup(config).ensure("verification")
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Chapter serving requires the chapter-server optional dependencies") from exc
    from .chapter_executor import ChapterExecutor
    from .chapter_jobs import ChapterJobs
    from .chapter_storage import S3ArtifactPublisher, stop_runpod

    storage = config.get("chapter_storage", {})
    publisher = S3ArtifactPublisher(storage) if storage.get("bucket") else None
    shutdown_callback = None
    if settings.get("idle_action", "none") == "runpod_stop":
        if not settings.get("persistent_data") or publisher is None:
            raise ValueError("Runpod idle shutdown requires persistent data and independent artifact storage")
        shutdown_callback = lambda: stop_runpod(settings)
    executor = ChapterExecutor(config)
    job_settings = {**settings, "ffprobe": config["render"]["ffprobe"]}
    jobs = ChapterJobs(settings["data_dir"], executor, job_settings, publisher=publisher)
    app = create_app(jobs, settings, shutdown_callback=shutdown_callback)
    uvicorn.run(
        app, host=settings.get("host", "127.0.0.1"), port=settings.get("port", 8081),
        workers=1, access_log=False,
    )
