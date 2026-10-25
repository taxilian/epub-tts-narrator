"""Create temporary Gemini voices and exact-script local cloning references.

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
from urllib.parse import urlencode
import wave

from .engines import EngineError, _seed
from .diagnostics import safe_error_text


GEMINI_DESIGN_MODEL = "gemini-3.8-flash-tts"
_API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_STATE_VERSION = 1
_VOICE_ID = re.compile(r"voice_[A-Za-z0-9_-]{1,200}\Z")


class GeminiVoiceError(EngineError):
    """A complete provider diagnostic with configured credentials removed."""


class _RequestFailure(GeminiVoiceError):
    def __init__(self, message, *, definite_rejection=False, status_code=None):
        super().__init__(message)
        self.definite_rejection = definite_rejection
        self.status_code = status_code


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

    def _error_detail(self, value):
        """Keep provider messages and structured details, masking only the key."""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
            try:
                value = json.loads(value)
            except ValueError:
                pass
        if not isinstance(value, str):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        return safe_error_text(value, secrets=(self._api_key,))

    def _post(self, endpoint, payload):
        return self._request("POST", endpoint, payload)

    def _request(self, method, endpoint, payload=None):
        request = urllib.request.Request(
            _API_ROOT + endpoint, method=method,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "x-goog-api-key": self._api_key},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            code = exc.code
            reason = self._error_detail(str(exc.reason))
            try:
                raw = exc.read(_MAX_RESPONSE_BYTES + 1)
                detail = self._error_detail(raw[:_MAX_RESPONSE_BYTES]) or "(empty response body)"
                if len(raw) > _MAX_RESPONSE_BYTES:
                    detail += f"\n[Response exceeds {_MAX_RESPONSE_BYTES} bytes; remaining bytes omitted.]"
            except (OSError, ValueError) as read_error:
                detail = "Could not read the error response: " + self._error_detail(str(read_error))
            finally:
                exc.close()
            raise _RequestFailure(
                f"Gemini API request failed (HTTP {code}): {reason}\n"
                f"{method} {endpoint}; model {self.model}\n{detail}",
                definite_rejection=400 <= code < 500 and code != 408,
                status_code=code,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise _RequestFailure(
                f"Gemini API request failed or timed out ({method} {endpoint}; model {self.model}):\n"
                f"{type(exc).__name__}: {self._error_detail(str(exc))}",
            ) from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise _RequestFailure("Gemini API response exceeded the size limit")
        if method == "DELETE" and not raw.strip():
            return {}
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise _RequestFailure(
                f"Gemini API returned invalid JSON ({method} {endpoint}; model {self.model}):\n"
                f"{self._error_detail(str(exc))}\n{self._error_detail(raw)}",
            ) from None
        if not isinstance(result, dict):
            raise _RequestFailure("Gemini API returned an invalid response:\n" + self._error_detail(result))
        return result

    def list_voices(self, *, search=None):
        """List prompted voices in this API project, following every page."""
        if search is not None and (not isinstance(search, str) or not search or len(search) > 300):
            raise GeminiVoiceError("Gemini voice search must be a nonempty string of at most 300 characters")
        voices, seen_ids, seen_tokens = [], set(), set()
        token = None
        for _ in range(1000):
            query = {"type": "prompted", "page_size": 100}
            if search is not None:
                query["search"] = search
            if token:
                query["page_token"] = token
            response = self._request("GET", "/voices?" + urlencode(query))
            page = response.get("voices", [])
            if not isinstance(page, list):
                raise GeminiVoiceError("Gemini returned an invalid voices list")
            for voice in page:
                if (not isinstance(voice, dict) or not isinstance(voice.get("id"), str)
                        or not _VOICE_ID.fullmatch(voice["id"])):
                    raise GeminiVoiceError("Gemini returned an invalid stored voice ID")
                if voice["id"] not in seen_ids:
                    seen_ids.add(voice["id"])
                    voices.append(voice)
            token = response.get("next_page_token")
            if token is None or token == "":
                return voices
            if not isinstance(token, str) or len(token) > 8192 or token in seen_tokens:
                raise GeminiVoiceError("Gemini returned an invalid or repeated voice pagination token")
            seen_tokens.add(token)
        raise GeminiVoiceError("Gemini voice listing exceeded the pagination limit")

    def _find_created_voices(self, display_name, model):
        """Hosted creations attributable to one exact request, never similar names."""
        return [voice["id"] for voice in self.list_voices(search=display_name)
                if voice.get("display_name") == display_name
                and voice.get("model") in (model, "models/" + model)
                and voice.get("type") == "prompted"]

    def delete_voice(self, voice_id):
        """Delete a stored voice; an already absent voice is a successful cleanup."""
        if not isinstance(voice_id, str) or not _VOICE_ID.fullmatch(voice_id):
            raise GeminiVoiceError("Invalid Gemini voice ID for deletion")
        try:
            self._request("DELETE", "/voices/" + voice_id)
        except _RequestFailure as exc:
            if exc.status_code == 404:
                return False
            raise
        return True

    def release_voice(self, directory):
        """Remove only this creation request's hosted voices, retaining local audio.

        Unknown POST outcomes are reconciled by the exact saved display name.
        Failed deletions leave state intact so the caller can retry cleanup.
        """
        directory = Path(directory)
        if directory.is_symlink():
            raise GeminiVoiceError(f"Gemini cleanup directory must not be a symlink: {directory}")
        state_path = directory / "gemini-voice.json"
        pending_path = directory / "gemini-voice-pending.json"
        cleanup_path = directory / "gemini-voice-cleanup.json"
        cleanup_request_path = directory / "gemini-voice-cleanup-request.json"
        path = state_path if state_path.exists() or state_path.is_symlink() else pending_path
        if not path.exists() and not path.is_symlink():
            return []
        try:
            if path.is_symlink() or path.stat().st_size > 8192:
                raise ValueError
            saved = json.loads(path.read_text(encoding="utf-8"))
            fingerprint = saved.get("request_fingerprint")
            if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                raise ValueError
        except (OSError, ValueError, UnicodeError, AttributeError):
            raise GeminiVoiceError(f"Invalid Gemini voice state for cleanup: {path}") from None
        state = _read_state(path, fingerprint)
        if cleanup_path.exists() or cleanup_path.is_symlink():
            cleanup = _read_state(cleanup_path, fingerprint)
            ids = cleanup.get("voice_ids")
            if (cleanup.get("remote_deleted") is not True or not isinstance(ids, list)
                    or not ids and cleanup.get("confirmed_absent") is not True
                    or any(not isinstance(item, str) or not _VOICE_ID.fullmatch(item) for item in ids)
                    or state.get("voice_id") is not None and state["voice_id"] not in ids):
                raise GeminiVoiceError(f"Invalid Gemini voice cleanup record: {cleanup_path}")
            return ids
        voice_id = state.get("voice_id")
        if cleanup_request_path.exists() or cleanup_request_path.is_symlink():
            cleanup_request = _read_state(cleanup_request_path, fingerprint)
            ids = cleanup_request.get("voice_ids")
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(item, str) or not _VOICE_ID.fullmatch(item) for item in ids)
                    or voice_id is not None and voice_id not in ids):
                raise GeminiVoiceError(f"Invalid Gemini cleanup request: {cleanup_request_path}")
        elif voice_id is not None:
            if not isinstance(voice_id, str) or not _VOICE_ID.fullmatch(voice_id):
                raise GeminiVoiceError(f"Invalid stored Gemini voice ID: {path}")
            ids = [voice_id]
        else:
            display_name = state.get("display_name")
            if not isinstance(display_name, str) or not re.fullmatch(r"Audiobook voice [0-9a-f]{16}", display_name):
                raise GeminiVoiceError(f"Cannot reconcile the Gemini creation without its saved display name: {path}")
            original_model = state.get("model", self.model)
            if not isinstance(original_model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", original_model):
                raise GeminiVoiceError(f"Invalid Gemini model in cleanup state: {path}")
            ids = self._find_created_voices(display_name, original_model)
            if not ids:
                # The first empty listing only starts reconciliation: the caller
                # records a failed cleanup attempt, leaving the pending state
                # repairable. A repeated empty listing on a later attempt is
                # conclusive, so confirm absence instead of blocking forever.
                marker_path = directory / "gemini-cleanup-pending.json"
                if marker_path.exists() and not marker_path.is_symlink():
                    _write_state(cleanup_path, {"version": _STATE_VERSION,
                                 "request_fingerprint": fingerprint, "voice_ids": [],
                                 "remote_deleted": True, "confirmed_absent": True,
                                 "display_name": display_name})
                    return []
                raise GeminiVoiceError(
                    f"No Gemini voice named {display_name!r} is visible in this API project, so the "
                    f"creation outcome is still unknown. Rerun the command: a repeated empty listing "
                    "confirms the voice was never created and clears this state automatically. To "
                    "resolve it now, check Google AI Studio: if the voice exists, add its ID as "
                    f"voice_id in {path}; if it does not exist, remove that file.")
        # Persist discovered IDs BEFORE deletion. If the response or final marker
        # is lost, a retry can confirm absence with DELETE/404 without depending
        # on a list response that can no longer find the removed resource.
        if not cleanup_request_path.exists():
            _write_state(cleanup_request_path, {"version": _STATE_VERSION,
                         "request_fingerprint": fingerprint, "voice_ids": ids})
        for identifier in ids:
            self.delete_voice(identifier)
        _write_state(cleanup_path, {"version": _STATE_VERSION, "request_fingerprint": fingerprint,
                                   "voice_ids": ids, "remote_deleted": True,
                                   "display_name": state.get("display_name")})
        return ids

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
        cleanup_path = directory / "gemini-voice-cleanup.json"
        cleanup_request_path = directory / "gemini-voice-cleanup-request.json"
        display_name = "Audiobook voice " + _fingerprint(str(directory.resolve()))[:16]
        if cleanup_path.exists() or cleanup_path.is_symlink():
            _read_state(cleanup_path, fingerprint)
            raise GeminiVoiceError("This request's hosted Gemini voice has already been cleaned up. "
                                   "Use a new voice-creation request to generate another take; saved local samples remain usable.")
        if cleanup_request_path.exists() or cleanup_request_path.is_symlink():
            _read_state(cleanup_request_path, fingerprint)
            raise GeminiVoiceError("Cleanup has begun for this request's hosted Gemini voice. "
                                   "Finish cleanup and use a new voice-creation request; saved local samples remain usable.")
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
                   "display_name": display_name, "model": self.model}
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
            if exc.status_code is not None:
                # An answered request was processed and rejected, unlike a lost
                # connection, so its outcome is knowable by listing: adopt a
                # voice created despite the error, or prove nothing was made.
                try:
                    found = self._find_created_voices(display_name, self.model)
                except _RequestFailure:
                    found = None
                if found is not None:
                    if len(found) == 1:
                        _write_state(state_path, {**pending, "voice_id": found[0],
                                                  "recovered_after_error": True})
                        pending_path.unlink()
                        return found[0]
                    if not found:
                        pending_path.unlink()
                        remedy = (
                            "Rerun to retry. If the same voice description keeps failing with a 5xx "
                            "response, the request itself is crashing the Gemini voice service: reword "
                            "the description (cast --action edit --prompt-file) or choose another "
                            "gemini.model."
                            if exc.status_code >= 500 else
                            "Rerun to retry once the service accepts requests again.")
                        raise GeminiVoiceError(
                            str(exc) + f". The service answered, and no voice named {display_name!r} "
                            "exists in this API project, so nothing was created and no cleanup is "
                            "pending. " + remedy) from None
            raise GeminiVoiceError(str(exc) + ". " + self._pending_message(pending_path, display_name)) from None
        voice_id = created.get("id")
        if not isinstance(voice_id, str) or not _VOICE_ID.fullmatch(voice_id):
            raise GeminiVoiceError("Gemini did not return a valid voice ID. "
                              + self._pending_message(pending_path, display_name))
        state = {**pending, "voice_id": voice_id}
        try:
            _write_state(state_path, state)
        except OSError as exc:
            raise GeminiVoiceError(
                f"Gemini created {voice_id}, but its state could not be saved. "
                f"Add this ID as voice_id in {pending_path} before retrying.\n"
                f"{type(exc).__name__}: {self._error_detail(str(exc))}"
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
            detail = self._error_detail({key: response[key] for key in ("id", "status", "error")
                                         if key in response})
            raise GeminiVoiceError("Gemini sample synthesis did not complete successfully:\n" + detail)
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
        except OSError as exc:
            raise GeminiVoiceError(
                f"Could not publish Gemini audio without overwriting: {output_path}\n"
                f"{type(exc).__name__}: {self._error_detail(str(exc))}"
            ) from None
        return {
            "creation_method": "prompt_conditioned_voice_design", "description_applied": True,
            "requested_description": description, "designer": self.identity,
            "reference_text": text, "remote_voice_id": voice_id,
            "seed": seed, "synthesis_seed": synthesis_seed,
            "seed_applies_to": "sample_synthesis", **details,
        }

    def close(self):
        pass
