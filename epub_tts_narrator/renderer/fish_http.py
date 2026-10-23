"""Native Fish and stock SGLang-Omni reference transports without model imports."""

from __future__ import annotations

from collections.abc import Mapping
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..spoken_text import MAX_DELIVERY_CHARACTERS
from .engines import EngineError, _directed_text, _plain_direction, _seed, _text, format_dialogue
from .diagnostics import safe_error_text
from .speech_text import trim_speech_text


MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_SGLANG_REFERENCE_BYTES = 10 * 1024 * 1024
SGLANG_MAIN_TOP_K = (1 << 31) - 1
PROTOCOLS = {"fish_inline", "fish_native_drama3", "fish_sglang_drama3"}
SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
STYLE_CONTROL = rf"\[[^\[\]<>\r\n]{{1,{MAX_DELIVERY_CHARACTERS}}}\]"
SUPPORTED_CONTROLS = re.compile(r"</?whisper>|" + STYLE_CONTROL)


class FishHTTPError(EngineError):
    """A transport failure, distinct from rejected speech quality."""

    transport_error = True

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def redact(message: str, key: str) -> str:
    return safe_error_text(message, secrets=(key,) if key else ())


def validate_wire_text(text: str, speakers: set[int | None]) -> None:
    """Accept bounded, non-nested style controls emitted by the adapter."""
    if not isinstance(text, str) or len(text) > 100_000:
        raise EngineError("text must be a string of at most 100000 characters")
    for match in SPEAKER_TAG.finditer(text):
        speaker = int(match.group(1))
        if speaker > 15 or speaker not in speakers:
            raise EngineError(f"Speaker {speaker} has no matching inline reference")
    stripped = SPEAKER_TAG.sub("", text)
    for match in re.finditer(STYLE_CONTROL, stripped):
        if not _plain_direction(match.group()[1:-1]):
            raise EngineError("Style controls must contain a direction")
    _text(SUPPORTED_CONTROLS.sub("", stripped), "text")
    depth = 0
    for match in re.finditer(r"</?whisper>", text):
        depth += -1 if match.group() == "</whisper>" else 1
        if depth not in (0, 1):
            raise EngineError("whisper controls must be paired and not nested")
    if depth:
        raise EngineError("whisper controls must be paired")


def trim_wire_text(text: str, speakers: set[int | None]) -> str:
    """Trim validated Fish turn bodies without removing speaker/delivery tags."""
    validate_wire_text(text, speakers)

    def body(value):
        value = trim_speech_text(value)
        value = re.sub(r"<whisper>(.*?)</whisper>",
                       lambda match: "<whisper>" + trim_speech_text(match.group(1)) + "</whisper>",
                       value, flags=re.S)
        prefix = re.match(r"(?:" + STYLE_CONTROL + r"\s*)+", value)
        if prefix:
            value = prefix.group().rstrip() + " " + trim_speech_text(value[prefix.end():])
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


class FishHTTPEngine:
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
            raise EngineError("Fish HTTP protocol must be fish_inline, fish_native_drama3 or fish_sglang_drama3")
        self.multi_speaker = settings.get("multi_speaker", self.protocol != "fish_sglang_drama3")
        for name, value in (("name", self.name), ("model", self.model), ("api_key", self.api_key)):
            if not isinstance(value, str) or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise EngineError(f"Fish HTTP {name} must be plain text")
        if not self.model or type(self.multi_speaker) is not bool:
            raise EngineError("Fish HTTP model must be nonempty and multi_speaker must be boolean")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise EngineError("Fish HTTP timeout_seconds must be positive")
        values = {**settings, **(sampling or {})}
        self.sampling = {key: values.get(key, default) for key, default in (
            ("temperature", 1.0), ("top_p", 0.9), ("top_k", 30), ("max_new_tokens", 2048),
        )}
        if self.protocol == "fish_native_drama3":
            # Fish Speech 5415669 accepts these fields in the body, has a fixed
            # top-k of 30, and seeds its one shared inference queue globally.
            if type(settings.get("concurrency", 1)) is not int or settings.get("concurrency", 1) != 1:
                raise EngineError("fish_native_drama3 requires concurrency = 1")
            if type(self.sampling["top_k"]) is not int or self.sampling["top_k"] != 30:
                raise EngineError("fish_native_drama3 requires top_k = 30; the native server fixes this value")
            for key in ("temperature", "top_p"):
                value = self.sampling[key]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0.1 <= value <= 1:
                    raise EngineError(f"fish_native_drama3 {key} must be from 0.1 through 1")
            if (type(self.sampling["max_new_tokens"]) is not int
                    or not 1 <= self.sampling["max_new_tokens"] <= 2048):
                raise EngineError("fish_native_drama3 max_new_tokens must be from 1 through 2048")
        elif self.protocol == "fish_sglang_drama3":
            # Stock SGLang-Omni 35e973d's speech_request.py rejects speaker IDs
            # other than zero and per-request seeds. Its engine batches requests
            # using one model; do not reset global RNG state around requests.
            if self.multi_speaker:
                raise EngineError("fish_sglang_drama3 supports one reference voice per request; multi_speaker must be false")
            concurrency = settings.get("concurrency", 1)
            if type(concurrency) is not int or not 1 <= concurrency <= 16:
                raise EngineError("fish_sglang_drama3 concurrency must be from 1 through 16")
            for key, maximum in (("temperature", 2), ("top_p", 1)):
                value = self.sampling[key]
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                    raise EngineError(f"fish_sglang_drama3 {key} must be positive and at most {maximum}")
            if type(self.sampling["top_k"]) is not int or self.sampling["top_k"] != 30:
                raise EngineError("fish_sglang_drama3 requires top_k = 30 for its residual codebooks; main tokens use the stock uncapped profile")
            for key, maximum in (("max_new_tokens", 2048),):
                if type(self.sampling[key]) is not int or not 1 <= self.sampling[key] <= maximum:
                    raise EngineError(f"fish_sglang_drama3 {key} must be from 1 through {maximum}")

    @property
    def sglang_sampling(self) -> dict[str, Any]:
        # Stock 35e973d speech_request.py deliberately leaves main-token top-k
        # uncapped: its source warns that 30 can fall silent after a phrase.
        # audio_decoder.py independently caps residual codebooks at 30. The
        # native backend applies 30 to both, so retain both requested and actual
        # profiles in provenance instead of claiming identical sampling.
        return {**{key: value for key, value in self.sampling.items() if key != "top_k"},
                "main_token_top_k": SGLANG_MAIN_TOP_K, "residual_codebook_top_k": 30}

    @property
    def identity(self) -> dict[str, Any]:
        identity = {"engine": "fish_http", "model": self.model,
                    "protocol": "fish-inline-reference-wav-v1", "sampling": dict(self.sampling),
                    "multi_speaker": self.multi_speaker}
        if self.protocol == "fish_native_drama3":
            identity.update(protocol="fish-native-drama3-wav-v1",
                            request_options={"single_chunk": True, "use_memory_cache": "on",
                                             "seed_transport": "body", "fixed_top_k": 30})
        elif self.protocol == "fish_sglang_drama3":
            identity.update(protocol="fish-sglang-drama3-wav-v1",
                            sampling_effective=self.sglang_sampling,
                            request_options={"request_format": "openai-speech-json",
                                             "reference_encoding_cache": "content",
                                             "seed_transport": "unsupported"})
        return identity

    @property
    def voice_binding_identity(self) -> dict[str, Any]:
        return {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"reference_voice": True, "voice_design": False, "preset_voice": False,
                "delivery_control": True, "multi_speaker": self.multi_speaker,
                "single_speaker_paragraph": self.multi_speaker or self.protocol == "fish_sglang_drama3",
                "unconditioned_voice_creation": False}

    def close(self) -> None:
        pass

    def create_voice(self, description, text, output_path, *, seed):
        raise EngineError("Fish HTTP rendering uses saved reference voices; create voices with the cast tool")

    def synthesize(self, text, voice, output_path, *, seed, direction=""):
        text = _text(trim_speech_text(_text(text, "text")), "text")
        directed, metadata = _directed_text(text, direction)
        blob, transcript = read_reference(voice)
        return {**self._request(directed, [{"audio": blob, "text": transcript}], output_path, seed),
                "synthesis_text": directed,
                **metadata}

    def synthesize_dialogue(self, turns, output_path, *, seed):
        if not self.multi_speaker and self.protocol != "fish_sglang_drama3":
            raise EngineError("This Fish HTTP endpoint is configured without multi-speaker support")
        text, voices, metadata = format_dialogue(turns)
        if self.protocol == "fish_sglang_drama3" and len(voices) != 1:
            raise EngineError("fish_sglang_drama3 supports one reference voice per request")
        references = []
        for index, voice in enumerate(voices):
            blob, transcript = read_reference(voice)
            references.append({"audio": blob, "text": f"<|speaker:{index}|> {transcript}"})
        return {**self._request(text, references, output_path, seed),
                "synthesis_text": text, **metadata}

    def _request(self, text, references, output_path, seed):
        _seed(seed)
        output = Path(output_path)
        if output.exists():
            raise EngineError(f"Audio output already exists: {output}")
        body = {"text": text, "references": references, "format": "wav", "normalize": False,
                **{key: value for key, value in self.sampling.items() if key != "top_k"}}
        effective_seed = seed
        if self.protocol == "fish_native_drama3":
            # Fish Speech 5415669's fish_speech/utils/utils.py:set_seed clamps
            # every seed above 2**31 to the same value. Fold our uint32 attempt
            # seeds into 31 bits so consecutive retries remain distinct, including
            # across the boundary. Keep the original seed and cache identity.
            effective_seed = seed % (1 << 31)
            # Preserve the paragraph as one take and reuse native encoded voice
            # references. The chapter API already registers the FLAC bytes once;
            # this cache prevents re-encoding them on each loopback request.
            body.update(seed=effective_seed, single_chunk=True, use_memory_cache="on")
        if self.protocol == "fish_sglang_drama3":
            # Stock /v1/audio/speech accepts inline base64 references and caches
            # their encoded content. No native-Fish-only fields or unsupported
            # seed are sent; each request keeps its own immutable input.
            encoded_references = []
            for reference in references:
                blob = reference["audio"]
                if len(blob) > MAX_SGLANG_REFERENCE_BYTES:
                    raise EngineError("SGLang reference audio must contain at most 10 MiB")
                encoded_references.append({
                    "data": base64.b64encode(blob).decode("ascii"),
                    "media_type": "audio/flac" if blob.startswith(b"fLaC") else "audio/wav",
                    "text": reference["text"],
                })
            body = {"input": text, "references": encoded_references,
                    "response_format": "wav", "stream": False, **self.sampling,
                    "top_k": SGLANG_MAIN_TOP_K}
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            content_type = "application/json"
        else:
            try:
                import msgpack
            except ImportError as exc:
                raise FishHTTPError("Fish HTTP requires msgpack; install renderer.requirements.txt", retryable=False) from exc
            payload = msgpack.packb(body, use_bin_type=True)
            content_type = "application/msgpack"
        if len(payload) > MAX_AUDIO_BYTES:
            raise EngineError("Fish HTTP request is larger than 100 MiB")
        headers = {"Content-Type": content_type, "Accept": "audio/wav", "model": self.model}
        if self.protocol == "fish_inline":
            headers.update({"X-Audiobook-Seed": str(seed), "X-Audiobook-Top-K": str(self.sampling["top_k"])})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=payload, headers=headers, method="POST")
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = None
        try:
            with build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise FishHTTPError(f"Fish HTTP returned unexpected status {response.status}")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None:
                    if not declared_length.isdigit() or int(declared_length) > MAX_AUDIO_BYTES:
                        raise FishHTTPError("Fish HTTP returned an invalid or oversized Content-Length")
                    declared_length = int(declared_length)
                with tempfile.NamedTemporaryFile(prefix=".fish-", suffix=".wav", dir=output.parent, delete=False) as stream:
                    staged = Path(stream.name)
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_AUDIO_BYTES:
                            raise FishHTTPError("Fish HTTP response exceeded the 100 MiB limit")
                        stream.write(chunk)
                if declared_length is not None and size != declared_length:
                    raise FishHTTPError(
                        f"Fish HTTP audio was truncated: expected {declared_length} bytes, received {size}"
                    )
                metadata = inspect_wav(staged)
                os.link(staged, output)
                if self.protocol == "fish_native_drama3":
                    metadata["seed_effective"] = effective_seed
                elif self.protocol == "fish_sglang_drama3":
                    metadata.update(seed_effective=None, seed_mode="unsupported",
                                    sampling_requested=dict(self.sampling),
                                    sampling_effective=self.sglang_sampling)
                return {**metadata, "backend": "fish_http", "worker": self.name,
                        "model": self.model, "seed_requested": seed,
                        "seed_applied": (self.protocol != "fish_sglang_drama3"
                                         and response.headers.get("X-Audiobook-Seed-Applied") == str(effective_seed))}
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
                redact(f"Fish HTTP {exc.code} ({self.name}): {detail or exc.reason}", self.api_key),
                retryable=exc.code in {408, 425, 429} or exc.code >= 500,
            ) from exc
        except (URLError, TimeoutError, socket.timeout, http.client.HTTPException) as exc:
            raise FishHTTPError(redact(f"Fish HTTP request failed ({self.name}): {exc}", self.api_key)) from exc
        except OSError as exc:
            raise FishHTTPError(redact(f"Could not receive Fish HTTP audio ({self.name}): {exc}", self.api_key)) from exc
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
