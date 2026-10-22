"""A bounded Fish-compatible rendering endpoint for the local Drama 3 engine."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import secrets
import shutil
import socket
import tempfile
import threading

from .engines import EngineError, _text, create_engine
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
from .fish_http import MAX_AUDIO_BYTES, SPEAKER_TAG, inspect_wav, redact, validate_wire_text
from .local_runtime import LocalRuntime, requires_mlx


class RequestError(ValueError):
    pass


def _parse_request(data, content_type):
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
    allowed = {"text", "references", "format", "normalize", "temperature", "top_p", "max_new_tokens"}
    unsupported = set(body) - allowed
    if unsupported:
        raise RequestError("Unsupported Fish request option(s): " + ", ".join(sorted(map(str, unsupported))))
    if body.get("format", "wav") != "wav":
        raise RequestError("This server supports format='wav' only")
    if body.get("normalize", False) is not False:
        raise RequestError("This server requires normalize=false to preserve audiobook wording")
    references = body.get("references")
    if not isinstance(references, list) or not 1 <= len(references) <= 16:
        raise RequestError("Provide 1–16 inline audio references; reference_id and remote audio paths are unsupported")
    normalized = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict) or set(reference) - {"audio", "text", "speaker"}:
            raise RequestError(f"references[{index}] must contain inline audio and text, with optional speaker")
        audio = reference.get("audio")
        if content_type == "application/json" and isinstance(audio, str):
            try:
                audio = base64.b64decode(audio, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise RequestError(f"references[{index}].audio must be base64 audio") from exc
        if not isinstance(audio, bytes) or not audio or len(audio) > MAX_AUDIO_BYTES:
            raise RequestError(f"references[{index}].audio must contain inline audio bytes")
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
        normalized.append({"audio": audio, "text": text, "speaker": speaker})
    speakers = [reference["speaker"] for reference in normalized]
    if len(speakers) > 1 and (None in speakers or len(set(speakers)) != len(speakers)):
        raise RequestError("Multiple references must each have a distinct explicit speaker tag or speaker ID")
    if any(speaker is not None for speaker in speakers):
        text = body.get("text")
        if not isinstance(text, str) or not SPEAKER_TAG.match(text.lstrip()):
            raise RequestError("Text using tagged references must begin with a matching speaker tag")
    validate_wire_text(body.get("text"), {reference["speaker"] for reference in normalized})
    return body, normalized


def _sampling(body, headers):
    settings = {}
    for key in ("temperature", "top_p"):
        if key in body:
            value = body[key]
            lower = 0.000001
            upper = 2 if key == "temperature" else 1
            if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
                raise RequestError(f"{key} must be between {lower} and {upper}")
            settings[key] = value
    if "max_new_tokens" in body:
        value = body["max_new_tokens"]
        if type(value) is not int or not 1 <= value <= 2048:
            raise RequestError("max_new_tokens must be an integer between 1 and 2048")
        settings["max_new_tokens"] = value
    try:
        seed = int(headers.get("X-Audiobook-Seed", str(secrets.randbits(32))))
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError()
        top_k = headers.get("X-Audiobook-Top-K")
        if top_k is not None:
            top_k = int(top_k)
            if not 1 <= top_k <= 1000:
                raise ValueError()
            settings["top_k"] = top_k
    except ValueError as exc:
        raise RequestError("X-Audiobook-Seed must be uint32 and X-Audiobook-Top-K must be 1–1000") from exc
    return seed, settings


class _RenderServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False

    def __init__(self, address, config, engine):
        self.settings = config["server"]
        self.engine = engine
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

    def _infer(self, text, voices, output, seed, sampling):
        previous = {key: getattr(self.engine, key) for key in sampling}
        try:
            for key, value in sampling.items():
                setattr(self.engine, key, value)
            return self.engine.synthesize_fish(text, voices, output, seed=seed)
        finally:
            for key, value in previous.items():
                setattr(self.engine, key, value)

    def synthesize(self, text, voices, output, seed, sampling):
        return self.inference_runtime.run(
            self._infer, text, voices, output, seed, sampling,
            initialize_mlx=requires_mlx(self.engine),
        )

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
            data = b'{"error":{"message":"Renderer queue is full; retry later"}}'
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                                b"Connection: close\r\nRetry-After: 2\r\nContent-Length: "
                                + str(len(data)).encode() + b"\r\n\r\n" + data)
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
                        self.inference_runtime.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Book text, reference transcripts, and credentials never enter access logs.
        pass

    def _json(self, status, body):
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
            self._json(404, {"error": {"message": "Use GET /health or POST /v1/tts"}})
            return
        model = "drama-3" if self.server.model_id == DEFAULT_DRAMA3_MODEL else self.server.model_id
        self._json(200, {"status": "ready", "model": model, "protocol": "fish-inline-reference-wav-v1"})

    def do_POST(self):
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def _post(self):
        if not self._authorized():
            return
        if self.path != "/v1/tts":
            self._json(404, {"error": {"message": "Use POST /v1/tts"}})
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
            body, references = _parse_request(data, self.headers.get_content_type())
            seed, sampling = _sampling(body, self.headers)
        except (RequestError, EngineError) as exc:
            self._json(400, {"error": {"message": str(exc)}})
            return
        try:
            with tempfile.TemporaryDirectory(prefix="audiobook-tts-") as directory:
                root = Path(directory)
                voices = []
                for index, reference in enumerate(references):
                    path = root / f"reference-{index}.wav"
                    path.write_bytes(reference["audio"])
                    try:
                        inspect_wav(path)
                    except EngineError as exc:
                        self._json(400, {"error": {"message": f"Invalid reference {index}: {exc}"}})
                        return
                    voices.append({"kind": "reference", "reference_audio": str(path),
                                   "reference_text": reference["text"], "speaker": reference["speaker"],
                                   "reference_sha256": hashlib.sha256(reference["audio"]).hexdigest()})
                output = root / "take.wav"
                self.server.synthesize(body["text"], voices, output, seed, sampling)
                inspect_wav(output)
                size = output.stat().st_size
                if size > MAX_AUDIO_BYTES:
                    raise EngineError("Generated WAV exceeds the 100 MiB response limit")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Audiobook-Seed-Applied", str(seed))
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                with output.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            self._json(500, {"error": {"message": redact(f"{type(exc).__name__}: {exc}", self.server.settings.get("api_key", ""))}})


def make_server(config, engine=None):
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
        if config["engine"].get("type", "drama3_mlx") != "drama3_mlx":
            raise EngineError("The local server requires engine.type='drama3_mlx'")
        engine = create_engine(config["engine"])
    return _RenderServer((host, settings.get("port", 8080)), config, engine)


def serve(config, notify=print):
    server = make_server(config)
    host, port = server.server_address[:2]
    rendered_host = f"[{host}]" if ":" in host else host
    try:
        notify("Loading Drama 3 before accepting rendering requests…")
        server.warmup()
        notify(f"Drama 3 renderer listening at http://{rendered_host}:{port}/v1/tts")
        notify("One inference request runs at a time; the model stays loaded between requests.")
        server.serve_forever()
    except KeyboardInterrupt:
        notify("Stopping renderer server.")
    finally:
        server.server_close()
