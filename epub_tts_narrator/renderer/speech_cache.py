"""An index of successful speech, independent of its place in the book.

Take directories retain the original full request and attempt diagnostics. Only
accepted, byte-verified audio is published to this synthesis-content index.
"""

from pathlib import Path
import re
from threading import RLock

from .common import digest, file_hash, read_json, regular_file, write_json
from .speech_identity import generation_compatible, generation_identity, synthesis_engine_identity


class SpeechCache:
    def __init__(self, work, voices):
        self.work = Path(work)
        self.voices = {voice["asset_hash"]: voice for voice in voices.values()}
        self.entries = {}
        self.performances = {}
        self.lock = RLock()
        self.loaded = False

    def identity(self, request, *, engine_metadata=None):
        from .recovery_provenance import bind_legacy_recovery_metadata

        metadata = bind_legacy_recovery_metadata(self.work / "clips", request, engine_metadata)
        return generation_identity(request, self.voices, engine_metadata=metadata)

    @staticmethod
    def _performance_key(identity):
        return digest({key: value for key, value in identity.items() if key != "engine"})

    def _register(self, key, entry):
        self.entries[key] = entry
        self.performances.setdefault(self._performance_key(entry["identity"]), set()).add(key)

    @staticmethod
    def _concrete_entry(entry, record):
        # A logical alias may cover several real models. Retain the original
        # model as another lookup so switching back without the alias can find
        # every accepted take, including ones displaced from the shared key.
        if entry["identity"].get("engine", {}).get("engine") != "drama_speech_cache_alias":
            return None
        engine = dict(record["request"]["engine"])
        engine.pop("cache_model_alias", None)
        identity = dict(entry["identity"], engine=synthesis_engine_identity(engine))
        return dict(entry, identity=identity)

    def _register_concrete(self, entry, record):
        concrete = self._concrete_entry(entry, record)
        if concrete is not None:
            key = digest(concrete["identity"])
            # A persisted concrete lookup takes precedence over a derived one.
            if key not in self.entries:
                self._register(key, concrete)

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
            self._register(path.stem, entry)
            self._register_concrete(entry, saved[2])
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
            key = digest(identity)
            entry = self._entry(identity, saved)
            if key not in self.entries:
                self._register(key, entry)
            self._register_concrete(entry, record)
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
            self._register(key, self._read_index(path))
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
        self._register(key, entry)

    def _verified_entry(self, entry, *, verify_audio=True):
        saved = self._accepted(entry["request_hash"])
        if saved is None or file_hash(saved[0]) != entry["accepted_sha256"]:
            return None
        _, audio, record = saved
        if verify_audio and (file_hash(audio) != record["audio_sha256"]
                             or record["audio_sha256"] != entry["audio_sha256"]):
            raise ValueError("Cached speech audio changed on disk")
        return saved

    def find(self, identity, *, request=None):
        with self.lock:
            self._load()
            key = digest(identity)
            entry = self._current_entry(key)
            if entry is not None:
                saved = self._verified_entry(entry)
                if saved is not None:
                    self._publish(key, entry)
                    return saved[1:]
                self.entries.pop(key, None)
            if request is not None:
                # Search only identical text/reference/settings candidates. The
                # original engine retains revision evidence that canonical
                # local identities may replace with checkpoint file hashes.
                candidates = self.performances.get(self._performance_key(identity), ())
                for candidate_key in sorted(candidates):
                    candidate = self._current_entry(candidate_key)
                    if candidate is None:
                        continue
                    saved = self._verified_entry(candidate, verify_audio=False)
                    if saved is None:
                        continue
                    record = saved[2]
                    if generation_compatible(candidate["identity"], identity,
                            saved_engine=record["request"]["engine"], current_engine=request["engine"],
                            engine_metadata=record.get("engine_metadata")):
                        saved = self._verified_entry(candidate)
                        if saved is None:
                            continue
                        # Add a lookup alias, preserving the original request,
                        # successful verification, and waveform byte-for-byte.
                        self._publish(key, self._entry(identity, saved))
                        return saved[1:]
            return None

    def remember(self, identity, record):
        with self.lock:
            if record.get("report", {}).get("passed") is not True:
                raise ValueError("Only successful speech can enter the cache")
            saved = self._accepted(record["request_hash"])
            if saved is None or saved[2] != record or file_hash(saved[1]) != record["audio_sha256"]:
                raise ValueError("Speech cache publication requires unchanged verified audio")
            if identity != self.identity(record["request"], engine_metadata=record.get("engine_metadata")):
                raise ValueError("Speech cache identity does not match the accepted performance")
            entry = self._entry(identity, saved)
            self._publish(digest(identity), entry)
            concrete = self._concrete_entry(entry, record)
            if concrete is not None:
                self._publish(digest(concrete["identity"]), concrete)

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
                original = self.identity(record["request"], engine_metadata=record.get("engine_metadata"))
            except ValueError:
                # An indexed reference can remain usable after its profile or
                # selection metadata changes. Never guess an unindexed voice.
                original = next((entry["identity"] for entry in self.entries.values()
                    if entry["request_hash"] == record["request_hash"]
                    and entry["accepted_sha256"] == file_hash(path)), None)
            return original is not None and generation_compatible(original, identity,
                saved_engine=record["request"]["engine"], current_engine=request["engine"],
                engine_metadata=record.get("engine_metadata"))
