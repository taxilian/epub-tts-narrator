"""A bounded rendering endpoint for chapter-history Drama 3 requests (protocol v2).

Clients send the composed request: tagged text, speaker-tagged references and
the chapter's history turns with their codec tokens. The response is one
MessagePack object with the WAV, the generated codes and the history used.

With a ``[verification]`` model configured the server is also a recognition
worker: ``POST /v1/recognize`` transcribes one WAV on its own model thread,
beside synthesis, and ``/health`` reports the recognizers' identity so the
controller can require exactly its own models before using them.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import secrets
import select
import socket
import tempfile
import threading
import time

from .audio_naming import descriptive_audio_filename
from .chapter_protocol import require_hash
from .engines import EngineError, _text, create_engine
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
from .fish_http import MAX_AUDIO_BYTES, PROTOCOL, SPEAKER_TAG, inspect_wav, redact, validate_wire_text
from .local_runtime import LocalRuntime, requires_mlx
from .recognition_workers import (
    RECOGNITION_CALLS, RECOGNITION_PROTOCOL, build_recognizers, identity_digest, plain_transcript,
    recognition_identity,
)
from .reference_cache import UNKNOWN_REFERENCE_MESSAGE, ReferenceCache


def batch_summary(batch):
    """``; batch: ...`` for a batched take's log line (numbers only), else empty."""
    if not isinstance(batch, dict) or "steps" not in batch:
        return ""
    cached = batch.get("cached_rows", 0)
    return (f"; batch: joined {batch.get('joined_active', 0)} running of {batch.get('rows', '?')} rows "
            f"after {batch.get('waited_seconds', 0):.2f}s, {batch.get('mean_rows', 0):.1f} rows on average "
            f"(peak {batch.get('peak_rows', 0)}) over {batch['steps']} steps, prompt {batch.get('prompt_rows', '?')} rows"
            + (f" ({cached} cached)" if cached else "")
            + f", prefill {batch.get('prefill_seconds', 0):.2f}s")


class RequestError(ValueError):
    pass


def _print_activity(message):
    print(message, flush=True)


def _parse_request(data, content_type, references_cache=None):
    try:
        if content_type == "application/msgpack":
            try:
                import msgpack
            except ImportError as exc:
                raise RequestError("MessagePack support requires installing renderer.requirements.txt") from exc
            body = msgpack.unpackb(data, raw=False, strict_map_key=True)
        elif content_type == "application/json":
            body = json.loads(data)
        else:
            raise RequestError("Use Content-Type application/msgpack or application/json")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise RequestError(f"Invalid request body: {exc}") from exc
    if not isinstance(body, dict):
        raise RequestError("Request body must be an object")
    allowed = {"text", "references", "history", "format", "normalize", "temperature", "top_p", "max_new_tokens",
               "layout"}
    unsupported = set(body) - allowed
    if unsupported:
        raise RequestError("Unsupported Fish request option(s): " + ", ".join(sorted(map(str, unsupported))))
    if body.get("format", "wav") != "wav":
        raise RequestError("This server supports format='wav' only")
    if body.get("normalize", False) is not False:
        raise RequestError("This server requires normalize=false to preserve audiobook wording")
    if body.get("layout", "directing") not in {"directing", "streaming"}:
        raise RequestError("layout must be directing or streaming")
    references = body.get("references")
    # Zero references is the deployment smoke probe's unconditioned voice.
    if not isinstance(references, list) or not 0 <= len(references) <= 16:
        raise RequestError("Provide at most 16 inline audio references; remote audio paths are unsupported")
    normalized = []
    for index, reference in enumerate(references):
        if (not isinstance(reference, dict)
                or set(reference) - {"audio", "text", "speaker", "reference_id"}):
            raise RequestError(f"references[{index}] must contain inline audio or a registered "
                               f"reference_id, with text and optional speaker")
        reference_id = reference.get("reference_id")
        if reference_id is not None:
            try:
                require_hash(reference_id, "reference_id")
            except ValueError as exc:
                raise RequestError(f"references[{index}].{exc}") from exc
        audio = reference.get("audio")
        cache_hit = False
        if audio is None:
            if reference_id is None:
                raise RequestError(f"references[{index}].audio must contain inline audio bytes")
            audio = references_cache.load(reference_id) if references_cache else None
            if audio is None:
                raise RequestError(f"references[{index}] {UNKNOWN_REFERENCE_MESSAGE}")
            cache_hit = True
        if content_type == "application/json" and isinstance(audio, str):
            try:
                audio = base64.b64decode(audio, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise RequestError(f"references[{index}].audio must be base64 audio") from exc
        if not isinstance(audio, bytes) or not audio or len(audio) > MAX_AUDIO_BYTES:
            raise RequestError(f"references[{index}].audio must contain inline audio bytes")
        if not cache_hit and reference_id is not None and hashlib.sha256(audio).hexdigest() != reference_id:
            raise RequestError(f"references[{index}].reference_id does not match its inline audio")
        text = reference.get("text")
        if not isinstance(text, str):
            raise RequestError(f"references[{index}].text must be an exact transcript")
        speaker = reference.get("speaker")
        if speaker is not None and (type(speaker) is not int or not 0 <= speaker <= 15):
            raise RequestError("reference speaker must be an integer from 0 through 15")
        match = SPEAKER_TAG.match(text.lstrip())
        if match:
            tagged = text.lstrip()
            tagged_speaker = int(match.group(1))
            if speaker is not None and speaker != tagged_speaker:
                raise RequestError("reference speaker conflicts with its transcript speaker tag")
            speaker = tagged_speaker
            if speaker > 15:
                raise RequestError("reference speaker must be from 0 through 15")
            # Drama 3's speaker token separator is one space, not transcript text.
            text = tagged[match.end():]
            if text.startswith(" "):
                text = text[1:]
        _text(text, f"references[{index}].text")
        entry = {"audio": audio, "text": text, "speaker": speaker}
        if cache_hit:
            # Bytes from the reference cache already passed inspection when
            # they were registered; inline bytes are validated by the caller.
            entry["cache_hit"] = True
        if reference_id is not None:
            entry["reference_id"] = reference_id
        normalized.append(entry)
    speakers = [reference["speaker"] for reference in normalized]
    if len(speakers) > 1 and (None in speakers or len(set(speakers)) != len(speakers)):
        raise RequestError("Multiple references must each have a distinct explicit speaker tag or speaker ID")
    if any(speaker is not None for speaker in speakers):
        text = body.get("text")
        if not isinstance(text, str) or not SPEAKER_TAG.match(text.lstrip()):
            raise RequestError("Text using tagged references must begin with a matching speaker tag")
    speaker_ids = {reference["speaker"] for reference in normalized}
    validate_wire_text(body.get("text"), speaker_ids)
    history = body.get("history", [])
    if not isinstance(history, list) or len(history) > 2:
        raise RequestError("history must list at most two prior turns")
    from .continuity_codes import validate_codes
    turns = []
    for index, turn in enumerate(history):
        if not isinstance(turn, dict) or set(turn) != {"text", "codes"}:
            raise RequestError(f"history[{index}] must contain text and codes")
        if not isinstance(turn["text"], str) or not SPEAKER_TAG.match(turn["text"]):
            raise RequestError(f"history[{index}].text must begin with a speaker tag")
        try:
            validate_wire_text(turn["text"], speaker_ids)
            # Codebook bounds are checked against the loaded model by the engine.
            validate_codes(turn["codes"], 65536)
        except (EngineError, ValueError) as exc:
            raise RequestError(f"history[{index}]: {exc}") from exc
        turns.append((turn["text"], turn["codes"]))
    return body, normalized, turns


def _sampling(body, headers):
    """The controlling client's sampling and seed; the worker's own config never applies.

    Every history request must carry temperature, top_p and max_new_tokens in
    the body and the seed and top-k as headers. The response echoes the values
    actually applied so the client can prove the take used its settings.
    """
    settings = {}
    for key in ("temperature", "top_p"):
        if key not in body:
            raise RequestError(f"{key} is required; the controlling client owns sampling")
        value = body[key]
        upper = 2 if key == "temperature" else 1
        if type(value) not in (int, float) or not math.isfinite(value) or not 0.000001 <= value <= upper:
            raise RequestError(f"{key} must be between 0.000001 and {upper}")
        settings[key] = float(value)
    value = body.get("max_new_tokens")
    if type(value) is not int or not 1 <= value <= 2048:
        raise RequestError("max_new_tokens is required as an integer between 1 and 2048")
    settings["max_new_tokens"] = value
    try:
        seed = int(headers["X-Audiobook-Seed"])
        top_k = int(headers["X-Audiobook-Top-K"])
        if not 0 <= seed <= 0xFFFFFFFF or not 1 <= top_k <= 1000:
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise RequestError("X-Audiobook-Seed (uint32) and X-Audiobook-Top-K (1–1000) are required") from exc
    settings["top_k"] = top_k
    return seed, settings


class _RenderServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False

    def __init__(self, address, config, engine, *, recognizers=None, notify=_print_activity):
        self.settings = config["server"]
        self.engine = engine
        # (primary, reference-or-None) speech recognizers, or None for TTS only.
        self.recognizers = recognizers
        self.recognition = (recognition_identity(*recognizers) if recognizers is not None else None)
        self.recognition_digest = identity_digest(self.recognition) if self.recognition is not None else None
        # Recognition has its own model thread so it never waits behind synthesis.
        self.recognition_runtime = LocalRuntime(name="recognition-inference")
        self.notify = notify
        self.references = ReferenceCache()
        self._activity_lock = threading.Lock()
        self._request_sequence = 0
        self.model_id = get_drama3_model(
            config["engine"].get("model", DEFAULT_DRAMA3_MODEL)).repo
        self.accepted_models = ({"drama-3", DEFAULT_DRAMA3_MODEL}
                                if self.model_id == DEFAULT_DRAMA3_MODEL else {self.model_id})
        self.inference_runtime = LocalRuntime(name="drama3-inference")
        self._inference_closed = False
        self._close_lock = threading.Lock()
        self.admission = threading.BoundedSemaphore(self.settings.get("max_pending_requests", 4) + 1)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        try:
            super().__init__(address, _Handler)
        except BaseException:
            self.server_close()
            raise

    def next_request_id(self):
        with self._activity_lock:
            self._request_sequence += 1
            return f"request-{self._request_sequence:06d}"

    def activity(self, request_id, message):
        # Callers supply only metadata and fixed status messages. In particular,
        # exception text can contain book/reference text and is not logged here.
        with self._activity_lock:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                self.notify(f"[{timestamp}] {request_id}: {message}")
            except (OSError, ValueError):
                # A closed terminal/pipe must not fail synthesis or prevent the
                # request handler from releasing its admission slot.
                pass

    def _infer(self, text, voices, history, output, seed, sampling, request_id=None, queued_at=None,
               layout="directing"):
        if request_id:
            self.activity(request_id, f"inference started (queued {time.monotonic() - queued_at:.2f}s)")
        previous = {key: getattr(self.engine, key) for key in sampling}
        try:
            for key, value in sampling.items():
                setattr(self.engine, key, value)
            options = {"layout": layout} if layout != "directing" else {}
            return self.engine.synthesize_fish(text, voices, output, seed=seed, history=history, **options)
        finally:
            for key, value in previous.items():
                setattr(self.engine, key, value)

    def _infer_concurrent(self, text, voices, history, output, seed, sampling, request_id=None,
                          queued_at=None, layout="directing", connection=None):
        """A batching engine decodes many requests at once; sampling travels per request.

        While it decodes, the client's connection is watched: once the client
        has gone (a restarted controller, a timeout), the request's cancel
        signal frees its batch row instead of finishing audio nobody receives.
        """
        from .request_sampling import request_cancellation, request_sampling
        if request_id:
            self.activity(request_id, f"inference started (queued {time.monotonic() - queued_at:.2f}s)")
        options = {"layout": layout} if layout != "directing" else {}
        cancel, finished = threading.Event(), threading.Event()
        watcher = None
        if connection is not None:
            watcher = threading.Thread(target=self._watch_client, args=(connection, cancel, finished, request_id),
                                       name="serve-client-watch", daemon=True)
            watcher.start()
        try:
            with request_sampling(sampling), request_cancellation(cancel):
                return self.engine.synthesize_fish(text, voices, output, seed=seed, history=history, **options)
        finally:
            finished.set()
            if watcher is not None:
                watcher.join(2)

    def _watch_client(self, connection, cancel, finished, request_id, interval=0.5):
        """Set ``cancel`` once the client closes its connection (it sends nothing while it waits)."""
        while not finished.is_set():
            try:
                readable, _, _ = select.select([connection], [], [], interval)
                if not readable:
                    continue
                if connection.recv(1, socket.MSG_PEEK) != b"":
                    finished.wait(interval)  # unexpected bytes: keep watching for a close
                    continue
            except (OSError, ValueError):
                pass  # the socket is unusable: the client cannot receive the result either
            if not finished.is_set():
                cancel.set()
                if request_id:
                    self.activity(request_id, "client disconnected; its batched decode was stopped")
            return

    def synthesize(self, text, voices, output, seed, sampling, *, history=(), layout="directing",
                   request_id=None, queued_at=None, connection=None):
        if getattr(self.engine, "concurrent_requests", 1) > 1:
            return self._infer_concurrent(text, voices, list(history), output, seed, sampling,
                                          request_id, queued_at, layout, connection=connection)
        return self.inference_runtime.run(
            self._infer, text, voices, list(history), output, seed, sampling, request_id, queued_at, layout,
            initialize_mlx=requires_mlx(self.engine),
        )

    def recognize(self, call, path, *, request_id=None):
        part, method = RECOGNITION_CALLS[call]
        recognizer = self.recognizers[0] if part == "primary" else self.recognizers[1]
        if recognizer is None:
            raise RequestError("This server has no reference recognizer; set verification.reference_model")
        if request_id:
            self.activity(request_id, f"recognition started ({call})")
        result = self.recognition_runtime.run(getattr(recognizer, method), path,
                                              initialize_mlx=requires_mlx(recognizer))
        return plain_transcript(result)

    def _warmup(self):
        warmup = getattr(self.engine, "warmup", None)
        if warmup:
            warmup()

    def warmup(self):
        self.inference_runtime.run(self._warmup, initialize_mlx=requires_mlx(self.engine))

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(self.settings.get("request_timeout_seconds", 300))
        return request, address

    def process_request(self, request, client_address):
        if not self.admission.acquire(blocking=False):
            self.activity(self.next_request_id(), "rejected: renderer queue is full (HTTP 503)")
            data = b'{"error":{"message":"Renderer queue is full; retry later"}}'
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                                b"Connection: close\r\nRetry-After: 2\r\nContent-Length: "
                                + str(len(data)).encode() + b"\r\n\r\n" + data)
                # An early response must not reset a client still uploading its
                # request body, or the client may never see the rejection.
                request.shutdown(socket.SHUT_WR)
                deadline = time.monotonic() + 1.0
                remaining = self.settings.get("max_request_bytes", MAX_AUDIO_BYTES) + 65536
                while remaining > 0:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    request.settimeout(timeout)
                    chunk = request.recv(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.admission.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.admission.release()

    def server_close(self):
        try:
            if hasattr(self, "socket"):
                # Non-daemon HTTP threads finish their submitted inference
                # before the model is released or its executor is stopped.
                super().server_close()
        finally:
            with self._close_lock:
                if not self._inference_closed:
                    self._inference_closed = True
                    try:
                        close = getattr(self.engine, "close", None)
                        if close:
                            self.inference_runtime.run(close)
                    finally:
                        try:
                            self.inference_runtime.close()
                        finally:
                            for recognizer in self.recognizers or ():
                                close = getattr(recognizer, "close", None)
                                if callable(close):
                                    self.recognition_runtime.run(close)
                            self.recognition_runtime.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Book text, reference transcripts, and credentials never enter access logs.
        pass

    def _json(self, status, body):
        if 400 <= status < 500 and getattr(self, "request_id", None):
            self.server.activity(self.request_id, f"rejected (HTTP {status})")
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)

    def _authorized(self):
        key = self.server.settings.get("api_key", "")
        if key and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {key}".encode()):
            self._json(401, {"error": {"message": "Missing or invalid bearer API key"}})
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path != "/health":
            self._json(404, {"error": {"message": "Use GET /health, POST /v1/tts or POST /v1/recognize"}})
            return
        model = "drama-3" if self.server.model_id == DEFAULT_DRAMA3_MODEL else self.server.model_id
        health = {"status": "ready", "model": model, "protocol": PROTOCOL, "reference_reuse": True,
                  "history_layouts": sorted(getattr(self.server.engine, "history_layouts", {"directing"})),
                  "roles": ["tts"] + (["recognition"] if self.server.recognizers is not None else []),
                  # Requests this worker decodes at once (server.batch_rows on a
                  # batching engine); a controller's worker concurrency should match.
                  "concurrent_requests": getattr(self.server.engine, "concurrent_requests", 1)}
        if self.server.recognizers is not None:
            health.update(recognition_protocol=RECOGNITION_PROTOCOL, recognition=self.server.recognition)
        self._json(200, health)

    def do_POST(self):
        self.request_id = self.server.next_request_id()
        self.request_started = time.monotonic()
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.server.activity(self.request_id, "client disconnected or timed out; response not delivered")
            self.close_connection = True

    def _post(self):
        if not self._authorized():
            return
        if self.path == "/v1/recognize":
            self._recognize()
            return
        if self.path != "/v1/tts":
            self._json(404, {"error": {"message": "Use POST /v1/tts or POST /v1/recognize"}})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise RequestError("Chunked request bodies are unsupported; send Content-Length")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                raise RequestError("Provide one numeric Content-Length")
            length = int(lengths[0])
            if not 0 < length <= self.server.settings.get("max_request_bytes", MAX_AUDIO_BYTES):
                self._json(413, {"error": {"message": "Request body exceeds the configured size limit"}})
                return
            if self.headers.get("model", self.server.model_id) not in self.server.accepted_models:
                raise RequestError(f"This server provides model {self.server.model_id} only")
            data = self.rfile.read(length)
            if len(data) != length:
                raise RequestError("Incomplete request body")
            body, references, history = _parse_request(data, self.headers.get_content_type(),
                                                       self.server.references)
            seed, sampling = _sampling(body, self.headers)
            layout = body.get("layout", "directing")
            if layout not in getattr(self.server.engine, "history_layouts", {"directing"}):
                raise RequestError(f"This server's engine does not build the {layout!r} history layout")
        except (RequestError, EngineError) as exc:
            self._json(400, {"error": {"message": str(exc)}})
            return
        try:
            with tempfile.TemporaryDirectory(prefix="audiobook-tts-") as directory:
                root = Path(directory)
                voices = []
                for index, reference in enumerate(references):
                    owner = (f"speaker-{reference['speaker']}" if reference.get("speaker") is not None
                             else f"index-{index}")
                    path = root / descriptive_audio_filename("reference", self.request_id, owner,
                                                             suffix=".wav")
                    path.write_bytes(reference["audio"])
                    if not reference.get("cache_hit"):
                        try:
                            inspect_wav(path)
                        except EngineError as exc:
                            self._json(400, {"error": {"message": f"Invalid reference {index}: {exc}"}})
                            return
                        self.server.references.store(reference["audio"],
                                                     reference.get("reference_id"))
                    voices.append({"kind": "reference", "reference_audio": str(path),
                                   "reference_text": reference["text"], "speaker": reference["speaker"],
                                   "reference_sha256": hashlib.sha256(reference["audio"]).hexdigest()})
                output = root / descriptive_audio_filename("synthesis-take", self.request_id,
                                                           suffix=".wav")
                reused = sum(1 for reference in references if reference.get("cache_hit"))
                suffix = f", {reused} reused" if reused else ""
                queued_at = time.monotonic()
                self.server.activity(self.request_id, f"queued: {len(body['text'])} characters, "
                                                     f"{len(voices)} reference(s), {len(history)} history turn(s){suffix}")
                generated = self.server.synthesize(body["text"], voices, output, seed, sampling, history=history,
                                                   layout=layout, request_id=self.request_id, queued_at=queued_at,
                                                   connection=self.connection)
                audio = inspect_wav(output)
                if output.stat().st_size > MAX_AUDIO_BYTES:
                    raise EngineError("Generated WAV exceeds the 100 MiB response limit")
                import msgpack
                payload = msgpack.packb({
                    "audio": output.read_bytes(), "codes": generated["codes"],
                    "codebook_size": generated["codebook_size"],
                    "history_turns": generated["history_turns"],
                    "history_sha256": generated["history_sha256"],
                    "history_layout": generated.get("history_layout", "directing"),
                    "sampling": sampling, "seed": seed}, use_bin_type=True)
                size = len(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/msgpack")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Audiobook-Seed-Applied", str(seed))
                self.send_header("X-Audiobook-Reference-Reuse", "sha256-v1")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                self.wfile.write(payload)
                self.wfile.flush()
                self.server.activity(
                    self.request_id,
                    f"completed in {time.monotonic() - self.request_started:.2f}s: "
                    f"{audio['duration_seconds']:.2f}s audio, {audio['sample_rate']} Hz, {size} bytes"
                    + batch_summary(generated.get("batch")),
                )
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            raise
        except Exception as exc:
            self.server.activity(self.request_id, f"failed ({type(exc).__name__}); returning HTTP 500")
            self._json(500, {"error": {"message": redact(f"{type(exc).__name__}: {exc}", self.server.settings.get("api_key", ""))}})


    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise RequestError("Chunked request bodies are unsupported; send Content-Length")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise RequestError("Provide one numeric Content-Length")
        length = int(lengths[0])
        if not 0 < length <= self.server.settings.get("max_request_bytes", MAX_AUDIO_BYTES):
            return None
        data = self.rfile.read(length)
        if len(data) != length:
            raise RequestError("Incomplete request body")
        return data

    def _recognize(self):
        if self.server.recognizers is None:
            self._json(404, {"error": {"message": "This server does not offer recognition; set its [verification] model"}})
            return
        try:
            data = self._body()
            if data is None:
                self._json(413, {"error": {"message": "Request body exceeds the configured size limit"}})
                return
            if self.headers.get_content_type() != "application/msgpack":
                raise RequestError("Recognition requests must be MessagePack")
            import msgpack
            try:
                body = msgpack.unpackb(data, raw=False, strict_map_key=True)
            except (ValueError, TypeError) as exc:
                raise RequestError("Invalid MessagePack body") from exc
            if (not isinstance(body, dict) or set(body) != {"call", "audio"}
                    or body["call"] not in RECOGNITION_CALLS or not isinstance(body["audio"], bytes)
                    or not body["audio"]):
                raise RequestError("Send {call, audio} with call transcribe, transcribe_reference or corroborate_reference")
        except RequestError as exc:
            self._json(400, {"error": {"message": str(exc)}})
            return
        try:
            with tempfile.TemporaryDirectory(prefix="audiobook-asr-") as directory:
                path = Path(directory) / descriptive_audio_filename(
                    "recognition-input", self.request_id, body["call"], suffix=".wav")
                path.write_bytes(body["audio"])
                try:
                    inspect_wav(path)
                except EngineError as exc:
                    self._json(400, {"error": {"message": f"Invalid recognition audio: {exc}"}})
                    return
                self.server.activity(self.request_id, f"recognition queued ({body['call']})")
                transcript = self.server.recognize(body["call"], path, request_id=self.request_id)
            import msgpack
            payload = msgpack.packb({"call": body["call"], "transcript": transcript,
                                     "identity_sha256": self.server.recognition_digest}, use_bin_type=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/msgpack")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(payload)
            self.wfile.flush()
            self.server.activity(self.request_id, f"recognition completed in {time.monotonic() - self.request_started:.2f}s")
        except RequestError as exc:
            self._json(400, {"error": {"message": str(exc)}})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            raise
        except Exception as exc:
            # Recognizer errors may quote audio-derived text; log only the type.
            self.server.activity(self.request_id, f"recognition failed ({type(exc).__name__}); returning HTTP 500")
            self._json(500, {"error": {"message": redact(f"{type(exc).__name__}: {exc}", self.server.settings.get("api_key", ""))}})


def make_server(config, engine=None, *, recognizers=None, notify=_print_activity):
    """Bind an endpoint; an injected engine enables hardware-free integration tests."""
    settings = config["server"]
    host = settings.get("host", "127.0.0.1")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.casefold() == "localhost"
    if not loopback and not settings.get("api_key"):
        raise EngineError("Set server.api_key before listening beyond localhost")
    if engine is None:
        if config["engine"].get("type", "drama3_mlx") not in {"drama3_mlx", "fish_speech_cuda"}:
            raise EngineError("The renderer server requires engine.type='drama3_mlx' or 'fish_speech_cuda'")
        engine = create_engine(config["engine"])
    rows = settings.get("batch_rows", 1)
    if rows > 1:
        if not callable(getattr(engine, "set_batch_rows", None)):
            raise EngineError("server.batch_rows above 1 requires an engine that decodes batches (drama3_mlx or fish_speech_cuda)")
        if settings.get("max_pending_requests", 4) < rows:
            raise EngineError("server.max_pending_requests must be at least server.batch_rows")
        engine.set_batch_rows(rows)
    if recognizers is None:
        # A configured [verification] model makes this a recognition worker too.
        # An unusable one leaves it TTS-only; controllers that ask this worker
        # for recognition are then refused at their startup check.
        try:
            recognizers = build_recognizers(config.get("verification"))
        except (ValueError, OSError, EngineError, ImportError) as exc:
            recognizers = None
            notify(f"Speech recognition is not offered: {exc}")
    return _RenderServer((host, settings.get("port", 8080)), config, engine,
                         recognizers=recognizers, notify=notify)


def serve(config, notify=_print_activity):
    server = make_server(config, notify=notify)
    host, port = server.server_address[:2]
    rendered_host = f"[{host}]" if ":" in host else host
    try:
        notify("Loading Drama 3 before accepting rendering requests…")
        server.warmup()
        notify(f"Drama 3 renderer listening at http://{rendered_host}:{port}/v1/tts")
        notify("One inference request runs at a time; the model stays loaded between requests.")
        if server.recognizers is not None:
            notify(f"Speech recognition offered at http://{rendered_host}:{port}/v1/recognize, "
                   "on its own model thread beside synthesis.")
        server.serve_forever()
    except KeyboardInterrupt:
        notify("Stopping renderer server.")
    finally:
        server.server_close()
