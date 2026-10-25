"""Durable, input-bound proposal checkpoints; prepared profiles remain invalid."""

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import tempfile
from threading import Lock

from .pipeline import fingerprint
from .voice_prompt_refresh import _provider_identity


FILENAME = ".voice-profile-refresh.json"
LOCKNAME = ".voice-profile-refresh.lock"


def read_state(directory):
    path = Path(directory) / FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Voice profile refresh state must be a regular unlinked file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("version") != 1
            or value.get("phase") not in {"pending", "publishing"}
            or not isinstance(value.get("voices"), dict)
            or not isinstance(value.get("binding"), dict)):
        raise ValueError("Invalid voice profile refresh state; refusing to discard progress")
    return value


def save_state(directory, value):
    path = Path(directory) / FILENAME
    read_state(directory)
    descriptor, temporary = tempfile.mkstemp(prefix=".profile-state-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def clear_state(directory):
    read_state(directory)
    (Path(directory) / FILENAME).unlink()
    _sync_directory(directory)


class ProfileCheckpoint:
    """Serialize worker results before returning them to the proposal collector."""

    def __init__(self, directory, requests, binding, *, resume_command=None):
        self.directory, self.requests, self.binding = Path(directory), requests, binding
        self.client = requests.client
        self.lock = Lock()
        previous = read_state(directory)
        self.state = previous if previous and previous["binding"] == binding else {
            "version": 1, "phase": "pending", "binding": deepcopy(binding),
            "voices": {voice: {"status": "invalid"} for voice in binding["voice_ids"]}}
        self.state["resume_command"] = resume_command
        save_state(directory, self.state)

    def run(self, system, payload, validate, *, cancelled=None):
        voice = payload["voice"]["id"]
        request_hash = fingerprint({"system": system, "payload": payload,
                                    "provider": _provider_identity(self)})
        with self.lock:
            saved = self.state["voices"][voice]
            if saved.get("request_fingerprint") == request_hash and saved.get("status") == "prepared":
                return validate(deepcopy(saved["proposal"]))
            self.state["voices"][voice] = {"status": "invalid", "request_fingerprint": request_hash}
            save_state(self.directory, self.state)
        result = validate(self.requests.run(system, payload, validate, cancelled=cancelled))
        with self.lock:
            self.state["voices"][voice] = {"status": "prepared", "request_fingerprint": request_hash,
                                           "proposal": deepcopy(result)}
            save_state(self.directory, self.state)
        return result

    def publishing(self, revised, sidecars=None):
        self.state.update(phase="publishing", expected_after=fingerprint(revised["cast"]),
                          expected_audit=deepcopy(revised["provenance"].get("voice_profile_refresh")),
                          expected_sidecars={name: fingerprint(value) for name, value in (sidecars or {}).items()})
        save_state(self.directory, self.state)
