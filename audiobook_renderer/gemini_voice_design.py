"""Create persistent Gemini voices and exact-script local cloning references.

The hosted designer is deliberately separate from the local rendering engine.
Only the verified reference WAV is needed after casting. The remote voice ID is
saved before sample synthesis so retries cannot silently change the persona.
"""

from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request
import wave

from .engines import EngineError, _seed


GEMINI_DESIGN_MODEL = "gemini-3.8-flash-tts"
_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_STATE_VERSION = 1
_VOICE_ID = re.compile(r"voice_[A-Za-z0-9_-]{1,200}\Z")


class GeminiVoiceError(EngineError):
    """A credential-free diagnostic safe to show in casting progress."""


class _RequestFailure(GeminiVoiceError):
    def __init__(self, message, *, definite_rejection=False):
        super().__init__(message)
        self.definite_rejection = definite_rejection


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the API credential to a redirect destination."""

    def redirect_request(self, request, response, code, message, headers, url):
        return None


def _fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _write_exclusive(path, content):
    """Publish only complete files, refusing to replace any existing path."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".gemini-", dir=path.parent, delete=False,
        ) as stage:
            temporary = Path(stage.name)
            stage.write(content)
            stage.flush()
            os.fsync(stage.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_state(path, value):
    _write_exclusive(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _read_state(path, fingerprint):
    try:
        if path.is_symlink() or path.stat().st_size > 8192:
            raise ValueError
        result = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or result.get("version") != _STATE_VERSION
                or result.get("request_fingerprint") != fingerprint):
            raise ValueError
    except (OSError, ValueError, UnicodeError):
        raise GeminiVoiceError(f"Invalid or mismatched Gemini voice state: {path}") from None
    return result


def _wav_details(audio):
    """Check the complete RIFF container before the shared acoustic/ASR gate."""
    if (len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE"
            or int.from_bytes(audio[4:8], "little") + 8 != len(audio)):
        raise GeminiVoiceError("Gemini returned a malformed or truncated WAV")
    offset, counts = 12, {b"fmt ": 0, b"data": 0}
    while offset < len(audio):
        if offset + 8 > len(audio):
            raise GeminiVoiceError("Gemini returned a malformed or truncated WAV")
        kind = audio[offset:offset + 4]
        size = int.from_bytes(audio[offset + 4:offset + 8], "little")
        if kind in counts:
            counts[kind] += 1
        offset += 8 + size + size % 2
        if offset > len(audio):
            raise GeminiVoiceError("Gemini returned a malformed or truncated WAV")
    if counts != {b"fmt ": 1, b"data": 1}:
        raise GeminiVoiceError("Gemini WAV must contain exactly one PCM audio stream")
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            rate, frames = source.getframerate(), source.getnframes()
            channels, width = source.getnchannels(), source.getsampwidth()
            if (source.getcomptype() != "NONE" or channels not in (1, 2)
                    or width not in (1, 2, 3, 4) or not 8000 <= rate <= 192000
                    or not 0.1 <= frames / rate <= 180):
                raise GeminiVoiceError("Gemini returned unsupported or empty WAV audio")
            if len(source.readframes(frames + 1)) != frames * channels * width:
                raise GeminiVoiceError("Gemini returned truncated WAV samples")
    except (wave.Error, EOFError, OSError):
        raise GeminiVoiceError("Gemini returned an invalid PCM WAV") from None
    return {"sample_rate": rate, "sample_count": frames,
            "duration_seconds": frames / rate, "channels": channels,
            "sample_width_bytes": width}


class GeminiVoiceDesigner:
    """Design remotely, retain the ID, then synthesize the requested script."""

    capabilities = {"voice_design": True}

    def __init__(self, api_key, *, model=GEMINI_DESIGN_MODEL,
                 language_code="en-US", timeout_seconds=180):
        if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096
                or not api_key.isascii() or any(char.isspace() or ord(char) < 32
                                               or ord(char) == 127 for char in api_key)):
            raise GeminiVoiceError("gemini.api_key must be configured in the local config file")
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model):
            raise GeminiVoiceError("gemini.model must be a model ID")
        if (not isinstance(language_code, str)
                or not re.fullmatch(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*", language_code)):
            raise GeminiVoiceError("gemini.language_code must be a language code such as en-US")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600):
            raise GeminiVoiceError("gemini.timeout_seconds must be from 1 through 600")
        self._api_key = api_key
        self.model = model
        self.language_code = language_code
        self.timeout_seconds = float(timeout_seconds)
        self._opener = urllib.request.build_opener(_NoRedirect())

    @property
    def identity(self):
        # Neither credentials nor a key-derived fingerprint belong in bundles.
        return deepcopy({
            "engine": "gemini_voice_design", "adapter_version": 1,
            "model": self.model, "api_version": "v1beta",
            "language_code": self.language_code, "output_encoding": "wav-pcm",
        })

    def _post(self, endpoint, payload):
        request = urllib.request.Request(
            _API_ROOT + endpoint, method="POST",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "x-goog-api-key": self._api_key},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise _RequestFailure("Gemini API response exceeded the size limit")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise _RequestFailure("Gemini API returned an invalid response")
            return result
        except urllib.error.HTTPError as exc:
            # Remote bodies, URLs, exception strings and headers may contain
            # credentials or echoed private text. Report only the status code.
            code = exc.code
            exc.close()
            hints = {
                400: "Check the configured Gemini model and request settings.",
                401: "Check gemini.api_key in the local config file.",
                403: "Check Gemini API access and project permissions.",
                404: "Check model availability and the stored Gemini voice ID.",
                429: "The Gemini quota or rate limit was reached; try again later.",
            }
            hint = hints.get(code, "Try again after checking Gemini service availability.")
            raise _RequestFailure(
                f"Gemini API request failed (HTTP {code}). {hint}",
                definite_rejection=400 <= code < 500 and code != 408,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise _RequestFailure("Gemini API request failed or timed out") from None
        except (ValueError, UnicodeError):
            raise _RequestFailure("Gemini API returned invalid JSON") from None

    @staticmethod
    def _pending_message(path, display_name):
        return (
            "A Gemini voice creation request may already have succeeded. "
            f"Check Google AI Studio for {display_name!r}; if it exists, add its "
            f"ID as voice_id in {path}. Remove that pending file only after "
            "confirming no voice was created. No duplicate voice was requested."
        )

    def _remote_voice(self, description, directory):
        fingerprint = _fingerprint({"designer": self.identity, "description": description})
        state_path = directory / "gemini-voice.json"
        pending_path = directory / "gemini-voice-pending.json"
        display_name = "Audiobook voice " + _fingerprint(str(directory.resolve()))[:16]
        if state_path.exists() or state_path.is_symlink():
            state = _read_state(state_path, fingerprint)
            if not isinstance(state.get("voice_id"), str) or not _VOICE_ID.fullmatch(state["voice_id"]):
                raise GeminiVoiceError(f"Invalid stored Gemini voice ID: {state_path}")
            return state["voice_id"]
        if pending_path.exists() or pending_path.is_symlink():
            pending = _read_state(pending_path, fingerprint)
            # An explicit, local repair can reconnect a creation whose HTTP
            # response was lost. We never guess an identity from similar names.
            recovered_id = pending.get("voice_id")
            if isinstance(recovered_id, str) and _VOICE_ID.fullmatch(recovered_id):
                _write_state(state_path, {"version": _STATE_VERSION,
                             "request_fingerprint": fingerprint, "voice_id": recovered_id,
                             "display_name": display_name, "recovered_from_pending": True})
                pending_path.unlink()
                return recovered_id
            raise GeminiVoiceError(self._pending_message(pending_path, display_name))
        pending = {"version": _STATE_VERSION, "request_fingerprint": fingerprint,
                   "display_name": display_name}
        try:
            _write_state(pending_path, pending)
        except FileExistsError:
            raise GeminiVoiceError(self._pending_message(pending_path, display_name)) from None
        try:
            created = self._post("/voices", {
                "store": True, "voice": {
                    "model": self.model, "type": "prompted", "display_name": display_name,
                    "language_code": self.language_code, "prompted": {"input": description},
                },
            })
        except _RequestFailure as exc:
            if exc.definite_rejection:
                pending_path.unlink()
                raise GeminiVoiceError(str(exc)) from None
            raise GeminiVoiceError(str(exc) + ". " + self._pending_message(pending_path, display_name)) from None
        voice_id = created.get("id")
        if not isinstance(voice_id, str) or not _VOICE_ID.fullmatch(voice_id):
            raise GeminiVoiceError("Gemini did not return a valid voice ID. "
                              + self._pending_message(pending_path, display_name))
        state = {**pending, "voice_id": voice_id}
        try:
            _write_state(state_path, state)
        except OSError:
            raise GeminiVoiceError(
                f"Gemini created {voice_id}, but its state could not be saved. "
                f"Add this ID as voice_id in {pending_path} before retrying."
            ) from None
        pending_path.unlink()
        return voice_id

    def create_voice(self, description, text, output_path, *, seed):
        for value, label, limit in ((description, "Voice description", 4000),
                                    (text, "Sample text", 1000)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise GeminiVoiceError(f"{label} must contain 1 through {limit} characters")
            if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
                raise GeminiVoiceError(f"{label} contains control characters")
            if any(marker in value for marker in ("<|", "|>")):
                raise GeminiVoiceError(f"{label} contains model control syntax")
        if re.search(r"<[^>\n]+>", text):
            raise GeminiVoiceError("Sample text contains Gemini inline control tags; provide literal spoken text")
        _seed(seed)
        # The common renderer uses unsigned 32-bit seeds. Gemini's API accepts
        # signed 32-bit seeds; retain both values so the saved take is auditable.
        synthesis_seed = seed % (2 ** 31)
        output_path = Path(output_path)
        if output_path.exists() or output_path.is_symlink():
            raise GeminiVoiceError(f"Audio output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice_id = self._remote_voice(description, output_path.parent)
        response = self._post("/interactions", {
            "model": self.model, "store": False, "stream": False,
            "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            "response_format": {"type": "audio", "mime_type": "audio/wav"},
            "generation_config": {"seed": synthesis_seed, "speech_config": [{"voice": voice_id}]},
        })
        if response.get("status") != "completed" or response.get("error"):
            raise GeminiVoiceError("Gemini sample synthesis did not complete successfully")
        steps = response.get("steps")
        if not isinstance(steps, list):
            raise GeminiVoiceError("Gemini returned invalid audio output")
        audio_outputs = []
        for step in steps:
            if not isinstance(step, dict):
                raise GeminiVoiceError("Gemini returned invalid audio output")
            if step.get("type") != "model_output":
                continue
            content = step.get("content")
            if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
                raise GeminiVoiceError("Gemini returned invalid audio output")
            audio_outputs.extend(item for item in content if item.get("type") == "audio")
        if len(audio_outputs) != 1:
            raise GeminiVoiceError("Gemini must return exactly one audio segment")
        output = audio_outputs[0]
        if output.get("mime_type") != "audio/wav" or not isinstance(output.get("data"), str):
            raise GeminiVoiceError("Gemini must return inline WAV audio")
        if len(output["data"]) > _MAX_RESPONSE_BYTES:
            raise GeminiVoiceError("Gemini audio exceeded the size limit")
        try:
            audio = base64.b64decode(output["data"], validate=True)
        except (ValueError, binascii.Error):
            raise GeminiVoiceError("Gemini returned invalid base64 audio") from None
        details = _wav_details(audio)
        try:
            _write_exclusive(output_path, audio)
        except OSError:
            raise GeminiVoiceError(f"Could not publish Gemini audio without overwriting: {output_path}") from None
        return {
            "creation_method": "prompt_conditioned_voice_design", "description_applied": True,
            "requested_description": description, "designer": self.identity,
            "reference_text": text, "remote_voice_id": voice_id,
            "seed": seed, "synthesis_seed": synthesis_seed,
            "seed_applies_to": "sample_synthesis", **details,
        }

    def close(self):
        pass
