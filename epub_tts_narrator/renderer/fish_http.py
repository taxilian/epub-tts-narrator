"""HTTP transport to this project's ``serve`` workers without model imports.

Every request carries the chapter history composed on the coordinator and
every response returns the generated codec tokens with the WAV, so remote
workers (another Mac, or the Runpod inference service) continue a chapter's
performance exactly like the local runtime. Stock Fish and SGLang servers
cannot accept history turns or return codes and are therefore not supported.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..spoken_text import MAX_DELIVERY_CHARACTERS
from .config import SAMPLING_DEFAULTS, default_multi_speaker
from .engines import EngineError, HistorySynthesis, _seed, _text
from .performance_compiler import validate_compiled_text
from .diagnostics import safe_error_text
from .reference_cache import UNKNOWN_REFERENCE_MARKER, ReferenceCache
from .speech_text import trim_speech_text


MAX_AUDIO_BYTES = 100 * 1024 * 1024
PROTOCOL = "audiobook-history-wav-v2"
PROTOCOLS = {"fish_inline"}
SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
STYLE_CONTROL = rf"\[[^\[\]<>\r\n]{{1,{MAX_DELIVERY_CHARACTERS}}}\]"
SUPPORTED_CONTROLS = re.compile(r"</?(?:whisper|emphasis|soft|fast|slow|stress|forceful)>|<\|phoneme_(?:start|end)\|>|" + STYLE_CONTROL)


class FishHTTPError(EngineError):
    """A transport failure, distinct from rejected speech quality."""

    transport_error = True

    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def redact(message: str, key: str) -> str:
    return safe_error_text(message, secrets=(key,) if key else ())


def validate_wire_text(text: str, speakers: set[int | None]) -> None:
    """Accept compiler controls, balanced within each referenced speaker turn."""
    if not isinstance(text, str) or len(text) > 100_000:
        raise EngineError("text must be a string of at most 100000 characters")
    for match in SPEAKER_TAG.finditer(text):
        speaker = int(match.group(1))
        if speaker > 15 or speaker not in speakers:
            raise EngineError(f"Speaker {speaker} has no matching inline reference")
    bodies = []
    previous = 0
    for match in SPEAKER_TAG.finditer(text):
        body = text[previous:match.start()]
        if previous or body.strip():
            bodies.append(body)
        previous = match.end()
    bodies.append(text[previous:])
    for body in bodies:
        try:
            validate_compiled_text(body)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        _text(SUPPORTED_CONTROLS.sub("", body), "text")


def trim_wire_text(text: str, speakers: set[int | None]) -> str:
    """Trim validated Fish turn bodies without removing speaker/delivery tags."""
    validate_wire_text(text, speakers)

    def body(value, depth=0):
        if depth > 32:
            raise EngineError("Fish delivery wrappers are nested too deeply")
        value = trim_speech_text(value)
        prefix = re.match(r"(?:" + STYLE_CONTROL + r"\s*)+", value)
        if prefix:
            return prefix.group().rstrip() + " " + body(value[prefix.end():], depth + 1)
        # Only a whole-turn wrapper encloses removable outer formatting. An
        # inline scope can contain the sole whitespace between lexical words;
        # trimming its interior would change remote input relative to local.
        wrapper = re.fullmatch(r"<(whisper|emphasis|soft|fast|slow|stress|forceful)>(.*)</\1>", value, flags=re.S)
        if wrapper:
            return f"<{wrapper.group(1)}>" + body(wrapper.group(2), depth + 1) + f"</{wrapper.group(1)}>"
        return value

    parts, previous, tag = [], 0, ""
    for match in SPEAKER_TAG.finditer(text):
        target = body(text[previous:match.start()])
        if target or tag:
            _text(SUPPORTED_CONTROLS.sub("", target), "text")
            parts.append((tag + " " + target).strip())
        tag, previous = match.group(), match.end()
    target = body(text[previous:])
    if target or tag:
        _text(SUPPORTED_CONTROLS.sub("", target), "text")
        parts.append((tag + " " + target).strip())
    result = "\n".join(parts)
    validate_wire_text(result, speakers)
    return result


def read_reference(voice: Mapping[str, Any]) -> tuple[bytes, str]:
    if voice.get("kind") != "reference":
        raise EngineError("Fish HTTP requires a saved reference voice")
    value = voice.get("reference_audio")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise EngineError("reference_audio must be an absolute local file path")
    transcript = _text(voice.get("reference_text"), "reference_text")
    try:
        with Path(value).open("rb") as stream:
            blob = stream.read(MAX_AUDIO_BYTES + 1)
    except OSError as exc:
        raise EngineError(f"Could not read saved reference audio: {exc}") from exc
    if not blob or len(blob) > MAX_AUDIO_BYTES:
        raise EngineError("Saved reference audio must contain at most 100 MiB")
    checksum = voice.get("reference_sha256", voice.get("reference_audio_sha256"))
    if checksum is not None and hashlib.sha256(blob).hexdigest() != checksum:
        raise EngineError("Saved reference audio checksum does not match")
    return blob, transcript


def wire_reference_audio(blob: bytes) -> bytes:
    """The ``serve`` protocol carries WAV references only.

    A chapter service stores every registered voice as lossless FLAC, so a
    FLAC reference is decoded exactly to mono 16-bit PCM WAV before it is
    sent. Anything other than lossless mono 16-bit FLAC is refused rather
    than resampled or downmixed. Voice and speech identities stay bound to
    the stored asset; only the transfer encoding changes.
    """
    if not blob.startswith(b"fLaC"):
        return blob
    import soundfile as sf
    try:
        info = sf.info(io.BytesIO(blob))
        if info.format != "FLAC" or info.channels != 1 or info.subtype != "PCM_16":
            raise EngineError("FLAC voice references must be lossless mono 16-bit audio")
        samples, rate = sf.read(io.BytesIO(blob), dtype="int16", always_2d=True)
    except EngineError:
        raise
    except (RuntimeError, ValueError, TypeError) as exc:
        raise EngineError("Saved FLAC reference audio could not be decoded") from exc
    if not len(samples):
        raise EngineError("Saved FLAC reference audio is empty")
    output = io.BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
    data = output.getvalue()
    if len(data) > MAX_AUDIO_BYTES:
        raise EngineError("Decoded reference audio exceeds the 100 MiB request limit")
    return data


def inspect_wav(path: Path) -> dict[str, Any]:
    """Check the entire received WAV before it can become a rendering asset."""
    try:
        import numpy as np
        import soundfile as sf
        with sf.SoundFile(path) as audio:
            if (audio.format not in {"WAV", "WAVEX", "RF64"} or audio.frames <= 0
                    or not 8000 <= audio.samplerate <= 192000 or not 1 <= audio.channels <= 2):
                raise ValueError("expected nonempty mono/stereo WAV at 8000–192000 Hz")
            frames = 0
            for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError("non-finite audio samples")
                frames += len(block)
            if frames != audio.frames:
                raise ValueError("truncated WAV data")
            return {"sample_rate": audio.samplerate, "sample_count": frames,
                    "duration_seconds": frames / audio.samplerate}
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise FishHTTPError(f"Fish HTTP returned invalid WAV audio: {exc}") from exc


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a bearer key or reference voice to a redirected host.
        return None


class FishHTTPEngine(HistorySynthesis):
    """Thread-safe transport; every request contains its immutable voice bytes."""

    def __init__(self, settings: Mapping[str, Any], sampling: Mapping[str, Any] | None = None):
        self.url = settings.get("url", "")
        parsed = urlsplit(self.url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise EngineError("Fish HTTP url must be an http(s) endpoint without credentials, query, or fragment")
        self.name = settings.get("name", "remote")
        self.model = settings.get("model", "drama-3")
        self.api_key = settings.get("api_key", "")
        self.timeout = settings.get("timeout_seconds", 600)
        self.protocol = settings.get("protocol", "fish_inline")
        if not isinstance(self.protocol, str) or self.protocol not in PROTOCOLS:
            raise EngineError("Render workers must run audiobook-render serve (protocol fish_inline); stock Fish "
                              "and SGLang servers cannot carry chapter history")
        self.multi_speaker = settings.get("multi_speaker", default_multi_speaker(self.protocol))
        for name, value in (("name", self.name), ("model", self.model), ("api_key", self.api_key)):
            if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise EngineError(f"Fish HTTP {name} must be plain text")
        if not self.model or type(self.multi_speaker) is not bool:
            raise EngineError("Fish HTTP model must be nonempty and multi_speaker must be boolean")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise EngineError("Fish HTTP timeout_seconds must be positive")
        values = {**settings, **(sampling or {})}
        self.sampling = {key: values.get(key, SAMPLING_DEFAULTS[key])
                         for key in ("temperature", "top_p", "top_k", "max_new_tokens")}
        self._reuse_lock = threading.Lock()
        self._reuse_supported = False
        self._registered_references: set[str] = set()
        self._history_checked = False
        # Layouts the worker's /health advertises; older servers build only Fish's.
        self._history_layouts = frozenset({"directing"})

    @property
    def history_layouts(self) -> frozenset[str]:
        return getattr(self, "_history_layouts", frozenset({"directing"}))

    @property
    def identity(self) -> dict[str, Any]:
        return {"engine": "fish_http", "model": self.model, "protocol": PROTOCOL,
                "sampling": dict(self.sampling), "multi_speaker": self.multi_speaker}

    @property
    def voice_binding_identity(self) -> dict[str, Any]:
        return {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"reference_voice": True, "voice_design": False, "preset_voice": False,
                "delivery_control": True, "multi_speaker": self.multi_speaker,
                "structured_performance": True, "single_speaker_paragraph": self.multi_speaker,
                "unconditioned_voice_creation": False, "chapter_continuity": True}

    def check_continuity(self) -> None:
        """Confirm once that the worker is a history-capable ``serve`` process."""
        if self._history_checked:
            return
        health = self._get_json(self.url.rsplit("/v1/", 1)[0] + "/health")
        if health.get("protocol") != PROTOCOL:
            raise EngineError(f"Render worker {self.name} does not support chapter history; update and "
                              "restart audiobook-render serve on that machine")
        layouts = health.get("history_layouts")
        if isinstance(layouts, list) and all(isinstance(layout, str) for layout in layouts):
            self._history_layouts = frozenset({"directing"} | ({"streaming"} & set(layouts)))
        self._history_checked = True

    def close(self) -> None:
        pass

    def create_voice(self, description, text, output_path, *, seed):
        raise EngineError("Fish HTTP rendering uses saved reference voices; create voices with the cast tool")

    def synthesize_dialogue(self, turns, output_path, *, seed):
        if not self.multi_speaker:
            raise EngineError("This Fish HTTP endpoint is configured without multi-speaker support")
        return super().synthesize_dialogue(turns, output_path, seed=seed)

    def _generate_wire(self, text, references, history, output_path, *, seed, layout="directing"):
        _seed(seed)
        output = Path(output_path)
        if output.exists():
            raise EngineError(f"Audio output already exists: {output}")
        if layout not in self.history_layouts:
            raise EngineError(f"Render worker {self.name} does not build the {layout!r} history layout")
        wire = []
        for voice in references:
            blob, transcript = read_reference(voice)
            wire.append({"audio": wire_reference_audio(blob), "text": transcript, "speaker": voice["speaker"]})
        turns = [{"text": turn_text, "codes": codes} for turn_text, codes in history]
        metadata, codes, codebook_size = self._request(text, wire, turns, output, seed, layout)
        return {**metadata, "synthesis_text": text}, codes, codebook_size

    def _references_for_wire(self, references):
        """Swap repeated reference audio for its registered content id."""
        wire, seeded, reused = [], [], []
        with self._reuse_lock:
            registered = set(self._registered_references)
            supported = self._reuse_supported
        for reference in references:
            content_id = ReferenceCache.content_id(reference["audio"])
            if supported and content_id in registered:
                wire.append({"reference_id": content_id, "text": reference["text"],
                             "speaker": reference["speaker"]})
                reused.append(content_id)
            else:
                wire.append(reference)
                seeded.append(content_id)
        return wire, seeded, reused

    def _mark_registered(self, content_ids, supported) -> None:
        with self._reuse_lock:
            self._reuse_supported = supported
            if supported:
                self._registered_references.update(content_ids)
            else:
                self._registered_references.clear()

    def _forget_registered(self, content_ids) -> None:
        with self._reuse_lock:
            self._registered_references.difference_update(content_ids)

    def _request(self, text, references, history, output, seed, layout="directing"):
        wire, seeded, reused = self._references_for_wire(references)
        try:
            return self._transmit(text, wire, history, output, seed, seeded + reused, layout)
        except FishHTTPError as exc:
            if not reused or exc.status != 400 or UNKNOWN_REFERENCE_MARKER not in str(exc):
                raise
            # The server restarted or evicted an entry, so registration state
            # is stale. The rejected attempt never reached synthesis because
            # unknown ids fail at parse time; resend the audio inline once.
            self._forget_registered(seeded + reused)
        return self._transmit(text, references, history, output, seed, seeded + reused, layout)

    def _headers(self, content_type=None):
        headers = {"Accept": "application/msgpack", "model": self.model}
        if content_type:
            headers["Content-Type"] = content_type
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_json(self, url):
        request = Request(url, headers=self._headers(), method="GET")
        try:
            with build_opener(_NoRedirect()).open(request, timeout=min(self.timeout, 60)) as response:
                body = json.loads(response.read(1024 * 1024))
        except HTTPError as exc:
            exc.close()
            raise FishHTTPError(redact(f"Fish HTTP {exc.code} ({self.name}; endpoint={url})", self.api_key),
                                status=exc.code) from exc
        except (URLError, TimeoutError, socket.timeout, http.client.HTTPException, OSError, ValueError) as exc:
            raise FishHTTPError(redact(f"Fish HTTP health check failed ({self.name}; endpoint={url}): {exc}",
                                       self.api_key)) from exc
        if not isinstance(body, dict):
            raise FishHTTPError(f"Fish HTTP health check returned invalid JSON ({self.name})")
        return body

    def _transmit(self, text, references, history, output, seed, seeded_ids, layout="directing"):
        body = {"text": text, "references": references, "history": history, "format": "wav",
                "normalize": False, "temperature": float(self.sampling["temperature"]),
                "top_p": float(self.sampling["top_p"]), "max_new_tokens": self.sampling["max_new_tokens"]}
        if layout != "directing":
            # Only sent to workers whose /health advertises the layout.
            body["layout"] = layout
        try:
            import msgpack
        except ImportError as exc:
            raise FishHTTPError("Fish HTTP requires msgpack; install renderer.requirements.txt", retryable=False) from exc
        payload = msgpack.packb(body, use_bin_type=True)
        if len(payload) > MAX_AUDIO_BYTES:
            raise EngineError("Fish HTTP request is larger than 100 MiB")
        headers = {**self._headers("application/msgpack"),
                   "X-Audiobook-Seed": str(seed), "X-Audiobook-Top-K": str(self.sampling["top_k"])}
        request = Request(self.url, data=payload, headers=headers, method="POST")
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise FishHTTPError(f"Fish HTTP returned unexpected status {response.status}")
                if response.headers.get_content_type() != "application/msgpack":
                    raise FishHTTPError(f"Render worker {self.name} did not return chapter-history output; "
                                        "update and restart audiobook-render serve on that machine")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None:
                    if not declared_length.isdigit() or int(declared_length) > MAX_AUDIO_BYTES:
                        raise FishHTTPError("Fish HTTP returned an invalid or oversized Content-Length")
                    declared_length = int(declared_length)
                blob = response.read(MAX_AUDIO_BYTES + 1)
                if len(blob) > MAX_AUDIO_BYTES:
                    raise FishHTTPError("Fish HTTP response exceeded the 100 MiB limit")
                if declared_length is not None and len(blob) != declared_length:
                    raise FishHTTPError(
                        f"Fish HTTP response was truncated: expected {declared_length} bytes, received {len(blob)}")
                try:
                    result = msgpack.unpackb(blob, raw=False, strict_map_key=True)
                except (ValueError, TypeError) as exc:
                    raise FishHTTPError(f"Fish HTTP returned an invalid response body: {exc}") from exc
                if (not isinstance(result, dict) or not isinstance(result.get("audio"), bytes)
                        or not result["audio"]):
                    raise FishHTTPError("Fish HTTP response must contain WAV audio and generated codes")
                from .continuity_codes import validate_codes
                try:
                    validate_codes(result.get("codes"), result.get("codebook_size"))
                except ValueError as exc:
                    raise FishHTTPError(f"Fish HTTP returned invalid generated codes: {exc}") from exc
                turns = result.get("history_turns")
                if type(turns) is not int or not 0 <= turns <= len(history):
                    raise FishHTTPError("Fish HTTP response must report the history turns it used")
                if result.get("history_layout", "directing") != layout:
                    raise FishHTTPError(f"Render worker {self.name} did not apply the requested "
                                        f"{layout!r} history layout", retryable=False)
                with tempfile.NamedTemporaryFile(prefix=".fish-", suffix=".wav", dir=output.parent, delete=False) as stream:
                    staged = Path(stream.name)
                    stream.write(result["audio"])
                metadata = inspect_wav(staged)
                # The controlling system owns sampling: the worker must prove it
                # applied exactly this client's settings and seed, never its own.
                expected = {"temperature": float(self.sampling["temperature"]),
                            "top_p": float(self.sampling["top_p"]),
                            "top_k": self.sampling["top_k"], "max_new_tokens": self.sampling["max_new_tokens"]}
                if result.get("sampling") != expected or result.get("seed") != seed:
                    raise FishHTTPError(f"Render worker {self.name} did not apply the controlling sampling "
                                        "settings and seed; update and restart audiobook-render serve", retryable=False)
                os.link(staged, output)
                self._mark_registered(seeded_ids, response.headers.get("X-Audiobook-Reference-Reuse") == "sha256-v1")
                metadata.update(backend="fish_http", worker=self.name, model=self.model, seed=seed,
                                seed_requested=seed,
                                seed_applied=True,
                                history_turns=turns, history_available=len(history),
                                history_sha256=result.get("history_sha256"))
                if layout != "directing":
                    metadata["history_layout"] = layout
                return metadata, result["codes"], result["codebook_size"]
        except HTTPError as exc:
            try:
                try:
                    detail = exc.read(1024 * 1024).decode("utf-8", errors="replace")
                except (http.client.HTTPException, URLError, TimeoutError, OSError) as body_error:
                    # Exceptions raised inside this HTTPError handler do not
                    # reach the sibling transport handlers below. Preserve the
                    # HTTP status/retry policy even if its error body is broken.
                    partial = getattr(body_error, "partial", b"")
                    detail = partial.decode("utf-8", errors="replace") if isinstance(partial, bytes) else ""
                    detail += f"\n[Error response body could not be read completely: {body_error}]"
            finally:
                exc.close()
            raise FishHTTPError(
                redact(f"Fish HTTP {exc.code} ({self.name}; endpoint={self.url}): {detail or exc.reason}", self.api_key),
                retryable=exc.code in {408, 425, 429} or exc.code >= 500,
                status=exc.code,
            ) from exc
        except (URLError, TimeoutError, socket.timeout, http.client.HTTPException) as exc:
            raise FishHTTPError(redact(f"Fish HTTP request failed ({self.name}; endpoint={self.url}): {exc}", self.api_key)) from exc
        except OSError as exc:
            raise FishHTTPError(redact(f"Could not receive Fish HTTP audio ({self.name}; endpoint={self.url}): {exc}", self.api_key)) from exc
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
