"""Speech-recognition workers: the controller's own MLX thread or remote ``serve`` recognizers.

A rendered take's verification makes exactly one model call per lane job:
``transcribe``/``transcribe_reference`` on the primary recognizer or
``corroborate_reference`` on the reference recognizer. Every other step
(acoustic checks, word comparison, JEV and phonetic review) stays on the
controller. A remote recognizer is used only when it proves, by its reported
identity, that it runs the same model files, packages and options as the
controller; its transcripts then stand in for local ones without changing any
saved verification identity, and each one names the worker that produced it.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import http.client
import json
import socket
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from .engines import EngineError

RECOGNITION_PROTOCOL = "audiobook-recognition-v1"
# Verifier method → (recognizer, recognizer method). Nothing else is remote.
RECOGNITION_CALLS = {
    "transcribe": ("primary", "transcribe"),
    "transcribe_reference": ("primary", "transcribe_reference"),
    "corroborate_reference": ("reference", "transcribe_reference"),
}
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024


def recognizers(verifier):
    """The (primary, reference-or-None) recognizers inside the verifier wrappers."""
    from .verification import ReferenceCorroboratingVerifier

    current, seen = verifier, set()
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ReferenceCorroboratingVerifier):
            return current.primary, current.reference
        inner = vars(current).get("verifier") if hasattr(current, "__dict__") else None
        if inner is None:
            return current, None
        current = inner
    raise ValueError("Verifier wrappers form a cycle")


def comparable_identity(value):
    """An identity without machine-local model locations; everything else must match."""
    if isinstance(value, dict):
        return {key: comparable_identity(item) for key, item in value.items() if key != "model_path"}
    if isinstance(value, list):
        return [comparable_identity(item) for item in value]
    return deepcopy(value)


def recognition_identity(primary, reference=None):
    identity = getattr(primary, "identity", None)
    if not isinstance(identity, dict):
        raise EngineError("The speech recognizer has no identity to compare with recognition workers")
    return {"primary": comparable_identity(identity),
            "reference": None if reference is None else comparable_identity(reference.identity)}


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity_differences(expected, actual):
    """Field names (never values) where a worker's recognizers differ."""
    if not isinstance(actual, dict):
        return ["recognition"]
    differences = []
    for part in ("primary", "reference"):
        mine, theirs = expected.get(part), actual.get(part)
        if mine == theirs or (part == "reference" and mine is None):
            # A controller without a reference model never asks for one.
            continue
        if not isinstance(mine, dict) or not isinstance(theirs, dict):
            differences.append(part)
            continue
        differences.extend(f"{part}.{key}" for key in sorted(set(mine) | set(theirs))
                           if mine.get(key) != theirs.get(key))
    return differences


def build_recognizers(settings):
    """The serve process's own recognizers from its ``[verification]`` table, or None."""
    if not settings or not settings.get("model"):
        return None
    kind = settings.get("type", "mlx_whisper")
    if kind == "mlx_whisper":
        from .quality import MlxWhisperVerifier
        primary = MlxWhisperVerifier(settings["model"], language=settings.get("language", "en"))
        reference = None
        if settings.get("reference_model"):
            from .reference_asr import MlxParakeetVerifier
            reference = MlxParakeetVerifier(settings["reference_model"])
        return primary, reference
    if kind == "faster_whisper":
        from .chapter_verification import FasterWhisperVerifier

        def create(model):
            return FasterWhisperVerifier(
                model, language=settings.get("language", "en"), device=settings.get("device", "cpu"),
                compute_type=settings.get("compute_type", "int8"), cpu_threads=settings.get("cpu_threads", 4))
        return create(settings["model"]), (create(settings["reference_model"])
                                           if settings.get("reference_model") else None)
    raise EngineError("verification.type must be mlx_whisper or faster_whisper to offer recognition")


def plain_transcript(value):
    """A JSON-plain copy of a recognizer result (numpy scalars become numbers)."""
    def default(item):
        convert = getattr(item, "item", None) or getattr(item, "tolist", None)
        if callable(convert):
            return convert()
        raise TypeError(f"unserializable recognizer value {type(item).__name__}")
    return json.loads(json.dumps(value, default=default, allow_nan=False))


class RemoteRecognizer:
    """Client for one ``serve`` worker's ``POST /v1/recognize``; thread-safe and stateless."""

    def __init__(self, settings: Mapping[str, Any]):
        from .fish_http import FishHTTPEngine
        # Reuse the TTS client's URL, name, key and timeout validation.
        transport = FishHTTPEngine(settings)
        self.name = transport.name
        self.api_key = transport.api_key
        self.timeout = transport.timeout
        self.base = transport.url.rsplit("/v1/", 1)[0]
        self.expected = None

    def _headers(self, content_type=None):
        headers = {"Accept": "application/msgpack, application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _error(self, message, cause):
        from .fish_http import FishHTTPError, redact
        raise FishHTTPError(redact(f"Recognition worker {self.name}: {message}: {cause}", self.api_key)) from cause

    def verify(self, expected):
        """Require the worker's recognizers to equal ``expected`` (fail closed, names only)."""
        from .fish_http import _NoRedirect
        request = Request(self.base + "/health", headers=self._headers(), method="GET")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=min(self.timeout, 60)) as response:
                health = json.loads(response.read(1024 * 1024))
        except HTTPError as exc:
            exc.close()
            self._error("health check failed", exc)
        except (URLError, TimeoutError, socket.timeout, http.client.HTTPException, OSError, ValueError) as exc:
            self._error("health check failed", exc)
        if not isinstance(health, dict) or health.get("recognition_protocol") != RECOGNITION_PROTOCOL:
            raise EngineError(f"Recognition worker {self.name} does not offer speech recognition; update "
                              "audiobook-render serve there and set its [verification] model")
        differences = identity_differences(expected, health.get("recognition"))
        if differences:
            raise EngineError(f"Recognition worker {self.name} uses different recognizers than this machine "
                              f"({', '.join(differences)}); install the same models and package versions there, "
                              "or remove recognition from its roles")
        self.expected = identity_digest(expected)

    def recognize(self, call, path):
        if self.expected is None:
            raise EngineError(f"Recognition worker {self.name} has not been verified")
        if call not in RECOGNITION_CALLS:
            raise EngineError("Unsupported recognition call")
        from .fish_http import FishHTTPError, _NoRedirect, redact
        import msgpack
        audio = Path(path).read_bytes()
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise EngineError("Recognition audio must be nonempty and at most 100 MiB")
        payload = msgpack.packb({"call": call, "audio": audio}, use_bin_type=True)
        request = Request(self.base + "/v1/recognize", data=payload,
                          headers=self._headers("application/msgpack"), method="POST")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                if response.headers.get_content_type() != "application/msgpack":
                    raise FishHTTPError(f"Recognition worker {self.name} returned an unexpected response type")
                blob = response.read(MAX_TRANSCRIPT_BYTES + 1)
        except HTTPError as exc:
            try:
                detail = exc.read(64 * 1024).decode("utf-8", errors="replace")
            except (http.client.HTTPException, URLError, TimeoutError, OSError):
                detail = ""
            finally:
                exc.close()
            raise FishHTTPError(redact(f"Recognition worker {self.name} HTTP {exc.code}: {detail or exc.reason}",
                                       self.api_key), retryable=exc.code in {408, 425, 429} or exc.code >= 500,
                                status=exc.code) from exc
        except (URLError, TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
            self._error("request failed", exc)
        if len(blob) > MAX_TRANSCRIPT_BYTES:
            raise FishHTTPError(f"Recognition worker {self.name} returned an oversized transcript")
        try:
            result = msgpack.unpackb(blob, raw=False, strict_map_key=True)
        except (ValueError, TypeError) as exc:
            raise FishHTTPError(f"Recognition worker {self.name} returned an invalid response body") from exc
        if not isinstance(result, dict) or result.get("call") != call or not isinstance(result.get("transcript"), dict):
            raise FishHTTPError(f"Recognition worker {self.name} returned an invalid transcript")
        if result.get("identity_sha256") != self.expected:
            # The worker restarted with other models since it was checked.
            raise EngineError(f"Recognition worker {self.name} changed its recognizers during this render; "
                              "restart the render after installing matching models", )
        return {**result["transcript"], "recognized_by": self.name}
