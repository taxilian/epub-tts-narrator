"""ElevenLabs Voice Design previews saved as local cloning references.

Only the preview endpoint is used: no permanent remote voice or voice slot is
created. Returned alternatives are cached and consumed across quality retries.
"""

from __future__ import annotations

import base64
import binascii
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

from .common import digest, read_json, regular_file
from .diagnostics import safe_error_text
from .engines import EngineError, _seed


_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-voice/design?output_format=pcm_24000"
_MAX_RESPONSE = 64 * 1024 * 1024
_RATE = 24000
MODELS = {"eleven_ttv_v3", "eleven_multilingual_ttv_v2"}


class ElevenLabsVoiceError(EngineError):
    """An actionable provider error with credentials removed."""

    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def _publish(path, content):
    """Publish a whole file exclusively; never replace an existing candidate."""
    with tempfile.NamedTemporaryFile(prefix=".elevenlabs-", dir=path.parent) as stage:
        stage.write(content)
        stage.flush()
        os.fsync(stage.fileno())
        os.link(stage.name, path)


def _save_json(path, value):
    record = {**value, "record_sha256": digest(value)}
    _publish(path, (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode())


def _load_json(path):
    try:
        if regular_file(path).stat().st_size > _MAX_RESPONSE:
            raise ValueError("oversized cache")
        record = read_json(path)
        checksum = record.pop("record_sha256")
        if checksum != digest(record):
            raise ValueError("checksum mismatch")
        return record
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ElevenLabsVoiceError(f"Invalid ElevenLabs preview cache: {path}") from None


def _audio(preview):
    if not isinstance(preview, dict):
        raise ElevenLabsVoiceError("ElevenLabs returned an invalid preview")
    media = preview.get("media_type")
    if not isinstance(media, str):
        raise ElevenLabsVoiceError("ElevenLabs returned an invalid preview media type")
    media = media.split(";", 1)[0].lower()
    if media not in {"audio/pcm", "audio/x-pcm", "audio/l16", "application/octet-stream"}:
        raise ElevenLabsVoiceError(f"ElevenLabs returned {media!r}; expected requested PCM audio")
    try:
        data = base64.b64decode(preview["audio_base_64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise ElevenLabsVoiceError("ElevenLabs returned invalid base64 audio") from None
    if len(data) % 2 or not .1 <= len(data) / (2 * _RATE) <= 180:
        raise ElevenLabsVoiceError("ElevenLabs returned empty, truncated, or oversized PCM audio")
    if data.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC")):
        raise ElevenLabsVoiceError("ElevenLabs audio does not match the requested raw PCM format")
    voice_id = preview.get("generated_voice_id")
    if not isinstance(voice_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", voice_id):
        raise ElevenLabsVoiceError("ElevenLabs returned an invalid generated voice ID")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((1, 2, _RATE, 0, "NONE", "not compressed"))
        output.writeframes(data)
    return buffer.getvalue(), len(data) // 2


class ElevenLabsVoiceDesigner:
    capabilities = {"voice_design": True}

    def __init__(self, api_key, *, model="eleven_ttv_v3", timeout_seconds=180,
                 guidance_scale=5.0, loudness=0.0):
        if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096
                or not api_key.isascii() or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in api_key)):
            raise ElevenLabsVoiceError("elevenlabs.api_key must be configured in the local config file")
        if not isinstance(model, str) or model not in MODELS:
            raise ElevenLabsVoiceError("elevenlabs.model must be eleven_ttv_v3 or eleven_multilingual_ttv_v2")
        for name, value, low, high in (("timeout_seconds", timeout_seconds, 1, 600),
                                      ("guidance_scale", guidance_scale, 0, 100),
                                      ("loudness", loudness, -1, 1)):
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ElevenLabsVoiceError(f"elevenlabs.{name} must be from {low} through {high}")
        self._api_key = api_key
        self.model, self.timeout_seconds = model, float(timeout_seconds)
        self.guidance_scale, self.loudness = float(guidance_scale), float(loudness)
        self._opener = urllib.request.build_opener(_NoRedirect())

    @property
    def identity(self):
        return {"engine": "elevenlabs_voice_design", "adapter_version": 1,
                "model": self.model, "output_format": "pcm_24000",
                "guidance_scale": self.guidance_scale, "loudness": self.loudness,
                "auto_generate_text": False, "should_enhance": False}

    def _detail(self, value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return safe_error_text(value, secrets=(self._api_key,))

    def _post(self, payload):
        request = urllib.request.Request(_ENDPOINT, method="POST",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "xi-api-key": self._api_key})
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(_MAX_RESPONSE + 1)
        except urllib.error.HTTPError as exc:
            try:
                detail = self._detail(exc.read(_MAX_RESPONSE))
            except OSError as read_error:
                detail = self._detail(read_error)
            finally:
                exc.close()
            raise ElevenLabsVoiceError(
                f"ElevenLabs voice design failed (HTTP {exc.code}; model {self.model}):\n{detail}",
                retryable=exc.code >= 500 or exc.code in {408, 429}) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ElevenLabsVoiceError("ElevenLabs request failed or timed out: " + self._detail(exc),
                                       retryable=True) from None
        if len(raw) > _MAX_RESPONSE:
            raise ElevenLabsVoiceError("ElevenLabs response exceeds the size limit")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError):
            raise ElevenLabsVoiceError("ElevenLabs returned invalid JSON") from None
        if not isinstance(result, dict):
            raise ElevenLabsVoiceError("ElevenLabs returned an invalid response")
        return result

    def _batch(self, directory, fingerprint, description, text, seed, index):
        path = directory / f"elevenlabs-previews-{index:03d}.json"
        if path.exists() or path.is_symlink():
            record = _load_json(path)
            if (record.get("version") != 1 or record.get("fingerprint") != fingerprint
                    or type(record.get("design_seed")) is not int
                    or not 0 <= record["design_seed"] < 2 ** 31
                    or not isinstance(record.get("previews"), list)
                    or not 1 <= len(record["previews"]) <= 10):
                raise ElevenLabsVoiceError("ElevenLabs preview cache belongs to a different prompt or script")
            return record
        result = self._post({"voice_description": description, "text": text,
            "model_id": self.model, "seed": seed % (2 ** 31),
            "auto_generate_text": False, "stream_previews": False, "should_enhance": False,
            "guidance_scale": self.guidance_scale, "loudness": self.loudness})
        if result.get("text") != text:
            raise ElevenLabsVoiceError("ElevenLabs changed the requested preview script; no voice was saved")
        previews = result.get("previews")
        if not isinstance(previews, list) or not 1 <= len(previews) <= 10:
            raise ElevenLabsVoiceError("ElevenLabs returned no usable voice previews")
        kept = []
        for preview in previews:
            _audio(preview)
            kept.append({key: preview[key] for key in ("audio_base_64", "media_type", "generated_voice_id")})
        record = {"version": 1, "fingerprint": fingerprint, "design_seed": seed % (2 ** 31),
                  "text": text, "previews": kept}
        _save_json(path, record)
        return record

    def create_voice(self, description, text, output_path, *, seed):
        for value, label, low in ((description, "Voice description", 20), (text, "Sample script", 100)):
            if not isinstance(value, str) or not low <= len(value) <= 1000 or not value.strip():
                raise ElevenLabsVoiceError(f"ElevenLabs {label.lower()} must contain {low}–1000 characters; edit it and retry G11")
            if any(ord(c) < 32 and c not in "\n\r\t" for c in value) or "<|" in value or "|>" in value:
                raise ElevenLabsVoiceError(f"{label} contains model control syntax")
        if re.search(r"\[[^\]]*\]|<[^>\n]*>", text):
            raise ElevenLabsVoiceError("Sample script must be literal spoken text without inline control tags")
        _seed(seed)
        output = Path(output_path)
        if output.exists() or output.is_symlink():
            raise ElevenLabsVoiceError(f"Audio output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = digest({"designer": self.identity, "description": description, "text": text})
        selection_path = output.with_name(output.name + ".elevenlabs.json")
        if selection_path.exists() or selection_path.is_symlink():
            selection = _load_json(selection_path)
            if (selection.get("fingerprint") != fingerprint
                    or type(selection.get("batch")) is not int or not 1 <= selection["batch"] <= 10
                    or type(selection.get("preview")) is not int or not 0 <= selection["preview"] < 10):
                raise ElevenLabsVoiceError("Saved ElevenLabs selection belongs to a different request")
            index, chosen = selection["batch"], selection["preview"]
            if not (output.parent / f"elevenlabs-previews-{index:03d}.json").exists():
                raise ElevenLabsVoiceError("Saved ElevenLabs preview batch is missing; inspect the candidate directory")
            batch = self._batch(output.parent, fingerprint, description, text, seed, index)
        else:
            used = set()
            for path in output.parent.glob("*.elevenlabs.json"):
                previous = _load_json(path)
                if (previous.get("fingerprint") != fingerprint
                        or type(previous.get("batch")) is not int or not 1 <= previous["batch"] <= 10
                        or type(previous.get("preview")) is not int or not 0 <= previous["preview"] < 10):
                    raise ElevenLabsVoiceError("Conflicting ElevenLabs voice requests share a candidate directory")
                used.add((previous["batch"], previous["preview"]))
            for index in range(1, 11):
                batch = self._batch(output.parent, fingerprint, description, text, seed, index)
                available = [i for i in range(len(batch["previews"])) if (index, i) not in used]
                if available:
                    chosen = available[0]
                    break
            else:
                raise ElevenLabsVoiceError("ElevenLabs preview retry limit reached; use G11 for a fresh design")
            selection = {"version": 1, "fingerprint": fingerprint, "batch": index, "preview": chosen}
            _save_json(selection_path, selection)
        if chosen >= len(batch["previews"]):
            raise ElevenLabsVoiceError("Saved ElevenLabs preview selection is out of range")
        audio, frames = _audio(batch["previews"][chosen])
        _publish(output, audio)
        return {"creation_method": "prompt_conditioned_voice_design", "description_applied": True,
                "requested_description": description, "reference_text": text,
                "designer": self.identity, "seed": seed, "design_seed": batch["design_seed"],
                "seed_applies_to": "preview_batch_design", "preview_index": chosen,
                "generated_voice_id": batch["previews"][chosen]["generated_voice_id"],
                "remote_voice_created": False, "sample_rate": _RATE, "sample_count": frames,
                "duration_seconds": frames / _RATE, "channels": 1, "sample_width_bytes": 2}

    def close(self):
        pass
