"""An index of successful speech, independent of its place in the book.

Take directories retain the original full request and attempt diagnostics. Only
accepted, byte-verified audio is published to this synthesis-content index.
"""

from pathlib import Path
import re
from threading import RLock

from .common import digest, file_hash, read_json, regular_file, write_json
from .speech_identity import generation_identity


class SpeechCache:
    def __init__(self, work, voices):
        self.work = Path(work)
        self.voices = {voice["asset_hash"]: voice for voice in voices.values()}
        self.entries = {}
        self.lock = RLock()
        self.loaded = False

    def identity(self, request, *, engine_metadata=None):
        return generation_identity(request, self.voices, engine_metadata=engine_metadata)

    def _accepted(self, key):
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Invalid cached speech request identifier")
        directory = self.work / "clips" / key
        if directory.parent.is_symlink() or directory.is_symlink():
            raise ValueError("Cached speech directories must not use symlinks")
        path = directory / "accepted.json"
        if not path.exists() and not path.is_symlink():
            return None
        record = read_json(regular_file(path))
        if not isinstance(record, dict):
            raise ValueError("Invalid cached speech acceptance")
        if not isinstance(record.get("report"), dict) or record["report"].get("passed") is not True:
            return None
        if not isinstance(record.get("validation_key"), str) or re.fullmatch(
                r"[0-9a-f]{64}", record["validation_key"]) is None:
            raise ValueError("Cached speech lacks successful verification provenance")
        request = record.get("request")
        if (not isinstance(request, dict) or record.get("request_hash") != key
                or digest(request) != key):
            raise ValueError("Cached speech request integrity check failed")
        name = record.get("audio_file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
            raise ValueError("Invalid cached speech audio path")
        audio = directory / name
        if not audio.exists() and not audio.is_symlink():
            return None
        regular_file(audio)
        return path, audio, record

    def _load(self):
        if self.loaded:
            return
        index = self.work / "speech-cache"
        if index.is_symlink():
            raise ValueError("Speech cache index must not use symlinks")
        indexed = set()
        for path in sorted(index.glob("*.json")):
            entry = self._read_index(path)
            saved = self._accepted(entry.get("request_hash"))
            indexed.add(entry["request_hash"])
            if saved is None or file_hash(saved[0]) != entry.get("accepted_sha256"):
                continue
            self.entries[path.stem] = entry
        # Import successful legacy takes without copying audio or changing
        # their provenance. Unresolvable historical voices cannot be guessed.
        for path in sorted((self.work / "clips").glob("*/accepted.json")):
            if path.parent.name in indexed:
                continue
            saved = self._accepted(path.parent.name)
            if saved is None:
                continue
            _, _, record = saved
            if record["request"].get("kind") not in {"speech", "paragraph_speech"}:
                continue
            try:
                identity = self.identity(record["request"], engine_metadata=record.get("engine_metadata"))
            except ValueError:
                continue
            self.entries.setdefault(digest(identity), self._entry(identity, saved))
        self.loaded = True

    @staticmethod
    def _read_index(path):
        if path.parent.is_symlink():
            raise ValueError("Speech cache index must not use symlinks")
        entry = read_json(regular_file(path))
        if not isinstance(entry, dict):
            raise ValueError("Invalid speech cache index record")
        checksum = entry.pop("record_sha256", None)
        if (checksum != digest(entry) or entry.get("version") != 1
                or entry.get("format") != "audiobook-speech-cache"
                or not isinstance(entry.get("identity"), dict)
                or digest(entry["identity"]) != path.stem):
            raise ValueError("Speech cache index integrity check failed")
        return entry

    def _current_entry(self, key):
        # Chapter workers have separate index objects but share take locks and
        # storage. Observe publications made while waiting for the content lock.
        path = self.work / "speech-cache" / (key + ".json")
        if path.exists() or path.is_symlink():
            self.entries[key] = self._read_index(path)
        return self.entries.get(key)

    @staticmethod
    def _entry(identity, saved):
        path, _, record = saved
        return {"format": "audiobook-speech-cache", "version": 1,
                "identity": identity, "request_hash": record["request_hash"],
                "accepted_sha256": file_hash(path), "audio_sha256": record["audio_sha256"]}

    def _publish(self, key, entry):
        directory = self.work / "speech-cache"
        if directory.is_symlink():
            raise ValueError("Speech cache index must not use symlinks")
        path = directory / (key + ".json")
        if path.is_symlink():
            raise ValueError("Speech cache entries must not use symlinks")
        if not path.exists():
            write_json(path, dict(entry, record_sha256=digest(entry)))
        else:
            existing = read_json(regular_file(path))
            if existing != dict(entry, record_sha256=digest(entry)):
                write_json(path, dict(entry, record_sha256=digest(entry)))
        self.entries[key] = entry

    def find(self, identity):
        with self.lock:
            self._load()
            key = digest(identity)
            entry = self._current_entry(key)
            if entry is None:
                return None
            saved = self._accepted(entry["request_hash"])
            if saved is None or file_hash(saved[0]) != entry["accepted_sha256"]:
                self.entries.pop(key, None)
                return None
            _, audio, record = saved
            if file_hash(audio) != record["audio_sha256"] or record["audio_sha256"] != entry["audio_sha256"]:
                raise ValueError("Cached speech audio changed on disk")
            self._publish(key, entry)
            return audio, record

    def remember(self, identity, record):
        with self.lock:
            if record.get("report", {}).get("passed") is not True:
                raise ValueError("Only successful speech can enter the cache")
            saved = self._accepted(record["request_hash"])
            if saved is None or saved[2] != record or file_hash(saved[1]) != record["audio_sha256"]:
                raise ValueError("Speech cache publication requires unchanged verified audio")
            if identity != self.identity(record["request"], engine_metadata=record.get("engine_metadata")):
                raise ValueError("Speech cache identity does not match the accepted performance")
            self._publish(digest(identity), self._entry(identity, saved))

    def matches(self, request, accepted):
        """Validate a chapter's reused performance against its current inputs."""
        identity = self.identity(request)
        with self.lock:
            self._load()
            saved = self._accepted(accepted.get("request_hash"))
            if saved is None:
                return False
            path, audio, record = saved
            if (record["audio_sha256"] != accepted.get("audio_sha256")
                    or file_hash(audio) != record["audio_sha256"]):
                raise ValueError("Cached chapter speech audio changed")
            entry = self._current_entry(digest(identity))
            if (entry is not None and entry["request_hash"] == record["request_hash"]
                    and entry["accepted_sha256"] == file_hash(path)):
                return True
            try:
                return identity == self.identity(record["request"], engine_metadata=record.get("engine_metadata"))
            except ValueError:
                return False
