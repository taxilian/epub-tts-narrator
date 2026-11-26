"""ElevenLabs Voice Design previews saved as local cloning references.

The first reference clip is a Voice Design preview: no remote voice is
created for it. Returned alternatives are cached and consumed across quality
retries. Further clips of the same voice (``reference_clips``) need the chosen
preview saved as a temporary voice in the ElevenLabs account, because a preview
can only speak its own script and can be saved only once: that voice speaks the
other scripts through text to speech and is deleted when the clips are made.
Its state is kept in the creation folder, so an interrupted creation resumes
with the same voice, and a leftover from a crash is deleted by a later creation
in the same book.
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

import time
import urllib.parse

from .audio_naming import temporary_audio_prefix
from .common import digest, read_json, regular_file
from .diagnostics import safe_error_text
from .engines import EngineError, _seed


_API = "https://api.elevenlabs.io"
_ENDPOINT = _API + "/v1/text-to-voice/design?output_format=pcm_24000"
_MAX_RESPONSE = 64 * 1024 * 1024
_RATE = 24000
MODELS = {"eleven_ttv_v3", "eleven_multilingual_ttv_v2"}
# The speech model that speaks a saved voice designed by each design model.
SPEECH_MODELS = {"eleven_ttv_v3": "eleven_v3", "eleven_multilingual_ttv_v2": "eleven_multilingual_v2"}
VOICE_STATE = "elevenlabs-voice.json"
VOICE_PENDING = "elevenlabs-voice-pending.json"
VOICE_DELETED = "elevenlabs-voice-cleanup.json"
# A temporary voice left this long by a creation that never finished is a crash's leftover.
LEFTOVER_SECONDS = 3600
_PERMISSIONS = ("The ElevenLabs API key needs the voices_read and voices_write permissions to make extra "
                "reference clips: they save the chosen preview as a temporary voice and delete it afterwards. "
                "Edit the key's permissions at elevenlabs.io (Developers, API keys)")


class ElevenLabsVoiceError(EngineError):
    """An actionable provider error with credentials removed."""

    def __init__(self, message, *, retryable=False, status=None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


def _missing_permission(error):
    return error.status in {401, 403} and "permission" in str(error).lower()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def _publish(path, content):
    """Publish a whole file exclusively; never replace an existing candidate."""
    with tempfile.NamedTemporaryFile(prefix=temporary_audio_prefix(path, "elevenlabs-design"),
                                     dir=path.parent) as stage:
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
    voice_id = preview.get("generated_voice_id")
    if not isinstance(voice_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", voice_id):
        raise ElevenLabsVoiceError("ElevenLabs returned an invalid generated voice ID")
    return _wav(data)


def _wav(data):
    """Requested 16-bit mono PCM at 24 kHz as a WAV, with its frame count."""
    if len(data) % 2 or not .1 <= len(data) / (2 * _RATE) <= 180:
        raise ElevenLabsVoiceError("ElevenLabs returned empty, truncated, or oversized PCM audio")
    if data.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC", b"{")):
        raise ElevenLabsVoiceError("ElevenLabs audio does not match the requested raw PCM format")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((1, 2, _RATE, 0, "NONE", "not compressed"))
        output.writeframes(data)
    return buffer.getvalue(), len(data) // 2


def _same_script(requested, returned):
    """The returned script is the requested one, apart from spacing.

    ElevenLabs echoes the script back trimmed, so a script with leading,
    trailing or repeated spaces comes back re-spaced. Any other change means
    the previews may speak different words, so they are refused.
    """
    return isinstance(returned, str) and requested.split() == returned.split()


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
        return self._send("POST", _ENDPOINT, payload, action="voice design")

    def _send(self, method, url, payload=None, *, action, audio=False):
        """One API request: JSON back, or raw bytes when ``audio``."""
        request = urllib.request.Request(url, method=method,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
            headers={"Content-Type": "application/json",
                     "Accept": "audio/pcm" if audio else "application/json",
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
                f"ElevenLabs {action} failed (HTTP {exc.code}; model {self.model}):\n{detail}",
                retryable=exc.code >= 500 or exc.code in {408, 429}, status=exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ElevenLabsVoiceError("ElevenLabs request failed or timed out: " + self._detail(exc),
                                       retryable=True) from None
        if len(raw) > _MAX_RESPONSE:
            raise ElevenLabsVoiceError("ElevenLabs response exceeds the size limit")
        if audio:
            return raw
        if method == "DELETE" and not raw.strip():
            return {}
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
        if not _same_script(text, result.get("text")):
            raise ElevenLabsVoiceError("ElevenLabs changed the words of the requested preview script; no voice was saved")
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

    # Further reference clips of the designed voice (reference_clips.add_clips).

    def check_clip_voice(self):
        """Refuse before saving a temporary voice that this key could not delete again.

        Deleting a voice that does not exist answers "not found" with the
        permission and "missing permission" without it, and changes nothing.
        """
        if getattr(self, "_clip_voice_checked", False):
            return
        try:
            self._send("DELETE", _API + "/v1/voices/audiobook-permission-check", action="permission check")
        except ElevenLabsVoiceError as exc:
            if _missing_permission(exc):
                raise ElevenLabsVoiceError(_PERMISSIONS) from None
            if exc.status is None or exc.status >= 500 or exc.status in {408, 429}:
                raise
        self._clip_voice_checked = True

    @staticmethod
    def _voice_name(directory):
        return "Audiobook voice " + digest(str(Path(directory).resolve()))[:16]

    def _find_voices(self, name):
        """IDs of this account's voices named exactly ``name``."""
        query = urllib.parse.urlencode({"search": name, "page_size": 100})
        try:
            listing = self._send("GET", _API + "/v2/voices?" + query, action="voice listing")
        except ElevenLabsVoiceError as exc:
            if _missing_permission(exc):
                raise ElevenLabsVoiceError(_PERMISSIONS) from None
            raise
        voices = listing.get("voices")
        if not isinstance(voices, list):
            raise ElevenLabsVoiceError("ElevenLabs returned an invalid voice listing")
        return [voice["voice_id"] for voice in voices if isinstance(voice, dict)
                and voice.get("name") == name and isinstance(voice.get("voice_id"), str)
                and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", voice["voice_id"])]

    def _clip_voice(self, directory, description, generated_voice_id):
        """The temporary voice saved from the chosen preview, created once per creation folder."""
        directory = Path(directory)
        fingerprint = digest({"designer": self.identity, "description": description,
                              "generated_voice_id": generated_voice_id})
        state_path, pending_path = directory / VOICE_STATE, directory / VOICE_PENDING
        if (directory / VOICE_DELETED).exists() or (directory / VOICE_DELETED).is_symlink():
            raise ElevenLabsVoiceError("This creation's temporary ElevenLabs voice was already deleted")
        if state_path.exists() or state_path.is_symlink():
            state = _load_json(state_path)
            if state.get("fingerprint") != fingerprint or not isinstance(state.get("voice_id"), str):
                raise ElevenLabsVoiceError(f"Saved temporary ElevenLabs voice belongs to another request: {state_path}")
            return state["voice_id"]
        self._delete_leftovers(directory.parent, directory)
        name = self._voice_name(directory)
        resumed = pending_path.exists() or pending_path.is_symlink()
        if resumed:
            pending = _load_json(pending_path)
            if pending.get("fingerprint") != fingerprint:
                raise ElevenLabsVoiceError(f"Pending temporary ElevenLabs voice belongs to another request: {pending_path}")
        else:
            _save_json(pending_path, {"version": 1, "fingerprint": fingerprint, "voice_name": name,
                                      "generated_voice_id": generated_voice_id})
        try:
            created = self._send("POST", _API + "/v1/text-to-voice", {
                "voice_name": name, "voice_description": description,
                "generated_voice_id": generated_voice_id,
                "labels": {"created_by": "epub-tts-narrator", "use": "temporary reference clips"}},
                action="saving the chosen preview as a temporary voice")
        except ElevenLabsVoiceError as exc:
            if exc.status is None or exc.status >= 500 or exc.status in {408, 429}:
                raise  # The outcome is unknown: the pending record lets a retry find the voice.
            found = self._find_voices(name) if resumed else []
            if len(found) != 1:
                if not found:
                    pending_path.unlink()
                raise
            created = {"voice_id": found[0]}
        voice_id = created.get("voice_id")
        if not isinstance(voice_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", voice_id):
            raise ElevenLabsVoiceError("ElevenLabs returned no valid voice ID for the saved preview")
        _save_json(state_path, {"version": 1, "fingerprint": fingerprint, "voice_name": name,
                                "generated_voice_id": generated_voice_id, "voice_id": voice_id})
        pending_path.unlink(missing_ok=True)
        return voice_id

    def speak_clip(self, description, text, output_path, *, seed, first):
        """Another reference clip: the first clip's voice speaking ``text``.

        ``first`` is the first clip's creation metadata, naming the preview it used.
        """
        generated = (first or {}).get("generated_voice_id")
        if not isinstance(generated, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", generated):
            raise ElevenLabsVoiceError("The first reference clip names no ElevenLabs preview to continue")
        if not isinstance(text, str) or not text.strip() or len(text) > 5000:
            raise ElevenLabsVoiceError("Reference clip script must contain 1–5000 characters")
        if any(ord(c) < 32 and c not in "\n\r\t" for c in text) or re.search(r"\[[^\]]*\]|<[^>\n]*>", text):
            raise ElevenLabsVoiceError("Reference clip script must be literal spoken text without control tags")
        _seed(seed)
        output = Path(output_path)
        if output.exists() or output.is_symlink():
            raise ElevenLabsVoiceError(f"Audio output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.check_clip_voice()
        voice_id = self._clip_voice(output.parent, description, generated)
        speech_model = SPEECH_MODELS[self.model]
        raw = self._send("POST", f"{_API}/v1/text-to-speech/{voice_id}?output_format=pcm_24000",
                         {"text": text, "model_id": speech_model, "seed": seed},
                         action="speaking a reference clip", audio=True)
        audio, frames = _wav(raw)
        _publish(output, audio)
        return {"creation_method": "prompt_conditioned_voice_design", "description_applied": True,
                "requested_description": description, "reference_text": text,
                "designer": self.identity, "seed": seed, "speech_model": speech_model,
                "seed_applies_to": "clip_speech", "generated_voice_id": generated,
                "remote_voice_created": True, "remote_voice_deleted_after_clips": True,
                "sample_rate": _RATE, "sample_count": frames, "duration_seconds": frames / _RATE,
                "channels": 1, "sample_width_bytes": 2}

    def release_clip_voice(self, directory):
        """Delete this creation's temporary voice; returns its ID, or None when there was none."""
        directory = Path(directory)
        state_path, pending_path = directory / VOICE_STATE, directory / VOICE_PENDING
        deleted_path = directory / VOICE_DELETED
        if deleted_path.exists() or deleted_path.is_symlink():
            return None
        if state_path.exists() or state_path.is_symlink():
            ids = [_load_json(state_path)["voice_id"]]
        elif pending_path.exists() or pending_path.is_symlink():
            ids = self._find_voices(_load_json(pending_path)["voice_name"])
        else:
            return None
        for voice_id in ids:
            try:
                self._send("DELETE", f"{_API}/v1/voices/{voice_id}", action="deleting the temporary voice")
            except ElevenLabsVoiceError as exc:
                if _missing_permission(exc):
                    raise ElevenLabsVoiceError(_PERMISSIONS + f"; delete the voice named "
                                               f"{self._voice_name(directory)!r} there yourself") from None
                if exc.status not in {400, 404}:
                    raise
        _save_json(deleted_path, {"version": 1, "voice_ids": ids})
        return ids[0] if ids else None

    def _delete_leftovers(self, root, current):
        """Delete temporary voices that crashed creations in this book left behind."""
        now = time.time()
        for state_path in sorted(Path(root).glob("*/" + VOICE_STATE)):
            directory = state_path.parent
            try:
                if (directory == current or (directory / VOICE_DELETED).exists()
                        or now - state_path.stat().st_mtime < LEFTOVER_SECONDS):
                    continue
                self.release_clip_voice(directory)
            except (ElevenLabsVoiceError, OSError, KeyError):
                continue

    def close(self):
        pass
