"""Durable chapter queue and immutable compressed voice registry.

Bounded synthesis coordinators share one warm engine pool. The next chapter can
use a free GPU while the previous chapter finishes ASR/JEV verification. A
separate bounded CPU pool assembles and publishes completed chapters.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
import time

from .chapter_protocol import ARTIFACT_NAMES, PROTOCOL, require_hash, validate_spec, validate_voice_manifest
from .common import digest, file_hash, read_json, regular_file, write_json


class ChapterJobs:
    def __init__(self, data_dir, executor, settings=None, *, publisher=None, maintenance=False):
        if type(maintenance) is not bool:
            raise ValueError("Maintenance mode must be a boolean")
        self._maintenance = maintenance
        self.root = Path(data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor, self.publisher = executor, publisher
        self.settings = dict(settings or {})
        self.identity = deepcopy(executor.identity)
        self.fingerprint = digest(self.identity)
        parallel = getattr(executor, "parallel_chapters", 1)
        if type(parallel) is not int or parallel < 1:
            raise ValueError("Executor chapter concurrency must be a positive integer")
        self._parallel_chapters = min(2, parallel)
        self._condition = threading.Condition(threading.RLock())
        self._stopping = self._draining = False
        self._started = self._closed = self._closing = False
        self._active = 0
        self._last_activity = time.monotonic()
        self._thread = self._assembly = self._file_lock = None
        self._threads = []
        self._db = self.root / "jobs.sqlite3"
        for name in ("voices", "render-work"):
            if (self.root / name).is_symlink():
                raise ValueError("Chapter server stores must not use symlinks")
            (self.root / name).mkdir(exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, idem TEXT NOT NULL, spec TEXT NOT NULL,
                    renderer TEXT NOT NULL, state TEXT NOT NULL, artifacts TEXT NOT NULL DEFAULT '[]',
                    error TEXT, created REAL NOT NULL, updated REAL NOT NULL,
                    UNIQUE(idem, renderer)
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_by_job ON events(job_id, sequence);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self._db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def capabilities(self):
        return {"protocol": PROTOCOL, "engine": deepcopy(self.identity),
                "renderer_fingerprint": self.fingerprint,
                "multi_speaker": getattr(self.executor, "multi_speaker", True),
                "single_speaker_paragraph": getattr(self.executor, "single_speaker_paragraph", False),
                "parallel_chapters": self._parallel_chapters,
                "maintenance": self._maintenance,
                "voice_format": "flac", "chapter_format": "m4a",
                "timing_precision": "paragraph_or_chunk"}

    def _voice(self, asset_id):
        require_hash(asset_id, "Voice asset")
        folder = self.root / "voices" / asset_id
        if folder.is_symlink():
            raise ValueError("Voice references must not use symlinks")
        manifest = validate_voice_manifest(read_json(regular_file(folder / "manifest.json")))
        audio = regular_file(folder / "reference.flac")
        if digest(manifest) != asset_id or file_hash(audio) != manifest["audio_sha256"]:
            raise ValueError("Registered voice failed its integrity check")
        return {"asset_hash": asset_id, "reference_audio": str(audio),
                "reference_text": manifest["reference_text"]}

    def ensure_voices(self, ids):
        if not isinstance(ids, list) or len(ids) > 2048:
            raise ValueError("Provide a list of at most 2048 voice hashes")
        missing = []
        for asset_id in dict.fromkeys(require_hash(value, "Voice asset") for value in ids):
            try:
                self._voice(asset_id)
            except FileNotFoundError:
                missing.append(asset_id)
            except ValueError:
                # Publication uses atomic rename, so readers see either a
                # complete immutable asset or an absent directory.
                if not (self.root / "voices" / asset_id).exists():
                    missing.append(asset_id)
                else:
                    raise
        return missing

    def _admit(self):
        if self._draining or self._stopping or self._closed:
            raise RuntimeError("Chapter server is draining; retry after restarting it")

    def _admit_job(self):
        self._admit()
        if self._maintenance:
            raise RuntimeError("Chapter server is in maintenance mode; start Encode to process books")

    def register_voice(self, asset_id, manifest, audio):
        manifest = validate_voice_manifest(manifest)
        require_hash(asset_id, "Voice asset")
        if digest(manifest) != asset_id:
            raise ValueError("Voice asset hash does not match its manifest")
        maximum = min(self.settings.get("max_request_bytes", 100 * 1024 * 1024), 64 * 1024 * 1024)
        if not isinstance(audio, bytes) or not audio.startswith(b"fLaC") or not 42 < len(audio) <= maximum:
            raise ValueError("Upload a compressed FLAC voice reference")
        if hashlib.sha256(audio).hexdigest() != manifest["audio_sha256"]:
            raise ValueError("Uploaded voice audio hash does not match its manifest")
        with self._condition:
            self._admit()
            self._active += 1
            self._last_activity = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix=".voice-", dir=self.root / "voices") as temporary:
                folder = Path(temporary)
                target = folder / "reference.flac"
                with target.open("wb") as stream:
                    stream.write(audio)
                    stream.flush()
                    os.fsync(stream.fileno())
                ffprobe = self.settings.get("ffprobe", "ffprobe")
                try:
                    result = subprocess.run(
                        [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_streams",
                         "-show_format", "-of", "json", str(target)],
                        stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True)
                    info = json.loads(result.stdout)
                    streams = info["streams"]
                    duration = float(info["format"]["duration"])
                    if (len(streams) != 1 or streams[0].get("codec_name") != "flac"
                            or not 0 < duration <= 120 or streams[0].get("channels") not in (1, 2)):
                        raise ValueError()
                except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError):
                    raise ValueError("Voice reference must be valid FLAC audio of at most 120 seconds") from None
                write_json(folder / "manifest.json", manifest)
                destination = self.root / "voices" / asset_id
                with self._condition:
                    if destination.exists():
                        self._voice(asset_id)
                    else:
                        os.rename(folder, destination)
                return {"id": asset_id, "audio_sha256": manifest["audio_sha256"]}
        finally:
            with self._condition:
                self._active -= 1
                self._last_activity = time.monotonic()
                self._condition.notify_all()

    def _row(self, db, job_id):
        require_hash(job_id, "Job ID")
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown chapter job")
        return row

    @staticmethod
    def _event(db, job_id, message, **fields):
        payload = {"message": str(message), "time": time.time(), **fields}
        db.execute("INSERT INTO events(job_id,payload) VALUES (?,?)",
                   (job_id, json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def _log(self, job_id, message):
        with self._condition, self._connect() as db:
            self._event(db, job_id, message)

    def submit(self, spec, idempotency_key):
        with self._condition:
            if self._maintenance:
                self._admit_job()
        spec = validate_spec(spec)
        require_hash(idempotency_key, "Idempotency key")
        identifier = digest({"key": idempotency_key, "renderer": spec["renderer_fingerprint"]})
        payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._condition, self._connect() as db:
            existing = db.execute("SELECT spec FROM jobs WHERE id=?", (identifier,)).fetchone()
        if existing is not None:
            if existing["spec"] != payload:
                raise ValueError("Idempotency key was already used for a different chapter")
            return self.status(identifier)
        if spec["renderer_fingerprint"] != self.fingerprint:
            raise ValueError("The server renderer changed; fetch capabilities before submitting")
        if self.ensure_voices(list(spec["voice_bindings"].values())):
            raise ValueError("Register every referenced voice before submitting the chapter")
        with self._condition, self._connect() as db:
            existing = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
            if existing is not None:
                if existing["spec"] != payload:
                    raise ValueError("Idempotency key was already used for a different chapter")
            else:
                self._admit_job()
                pending = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','generating','assembling')").fetchone()[0]
                if pending >= self.settings.get("max_pending_jobs", 256):
                    raise RuntimeError("Chapter job queue is full; retry later")
                now = time.time()
                db.execute("INSERT INTO jobs(id,idem,spec,renderer,state,created,updated) VALUES (?,?,?,?,?,?,?)",
                           (identifier, idempotency_key, payload, self.fingerprint, "queued", now, now))
                self._event(db, identifier, "Chapter queued", state="queued")
                self._last_activity = time.monotonic()
            db.commit()
            self._condition.notify_all()
        return self.status(identifier)

    def status(self, job_id):
        with self._condition, self._connect() as db:
            row = self._row(db, job_id)
            last = db.execute("SELECT COALESCE(MAX(sequence),0) FROM events WHERE job_id=?", (job_id,)).fetchone()[0]
            artifacts = json.loads(row["artifacts"])
            result = {"id": row["id"], "state": row["state"], "last_event": last,
                      "created": row["created"], "updated": row["updated"], "error": row["error"],
                      "spec_hash": digest(json.loads(row["spec"])), "artifacts": []}
        for item in artifacts:
            public = {key: item[key] for key in ("name", "sha256", "size")}
            public["url"] = (self.publisher.url(item["published"]) if self.publisher and item.get("published")
                             else f"/v1/jobs/{job_id}/artifacts/{item['name']}")
            result["artifacts"].append(public)
        return result

    def events(self, job_id, after=0):
        if type(after) is not int or after < 0:
            raise ValueError("Event cursor must be a nonnegative integer")
        with self._condition, self._connect() as db:
            self._row(db, job_id)
            rows = db.execute("SELECT sequence,payload FROM events WHERE job_id=? AND sequence>? ORDER BY sequence LIMIT 500",
                              (job_id, after)).fetchall()
            return [dict(json.loads(row["payload"]), sequence=row["sequence"]) for row in rows]

    def retry(self, job_id):
        with self._condition, self._connect() as db:
            self._admit_job()
            row = self._row(db, job_id)
            if row["renderer"] != self.fingerprint:
                raise ValueError("This job used another renderer; submit a new job specification")
            if row["state"] == "failed":
                pending = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','generating','assembling')").fetchone()[0]
                if pending >= self.settings.get("max_pending_jobs", 256):
                    raise RuntimeError("Chapter job queue is full; retry later")
                db.execute("UPDATE jobs SET state='queued',error=NULL,updated=? WHERE id=?", (time.time(), job_id))
                self._event(db, job_id, "Retry queued; accepted paragraphs will be reused", state="queued")
                self._last_activity = time.monotonic()
            db.commit()
            self._condition.notify_all()
        return self.status(job_id)

    def artifact(self, job_id, name):
        if name not in ARTIFACT_NAMES:
            raise KeyError("Unknown artifact")
        with self._condition, self._connect() as db:
            row = self._row(db, job_id)
            if row["state"] != "complete":
                raise ValueError("Chapter artifacts are not complete")
            item = next((item for item in json.loads(row["artifacts"]) if item["name"] == name), None)
            if item is None:
                raise KeyError("Unknown artifact")
        path = regular_file(self.root / item["path"])
        if not path.resolve().is_relative_to(self.root) or file_hash(path) != item["sha256"]:
            raise ValueError("Chapter artifact failed its integrity check")
        return path

    def start(self):
        with self._condition:
            if self._started:
                return
            self._admit()
            lock = (self.root / ".service.lock").open("a+")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.close()
                raise RuntimeError("Another chapter service is using this data directory") from None
            self._file_lock = lock
            if self._maintenance:
                # Keep the same exclusive store lease, but do not recover
                # interrupted states or start any queue/assembly consumers.
                self._started = True
                return
            with self._connect() as db:
                for row in db.execute("SELECT id FROM jobs WHERE state IN ('generating','assembling')").fetchall():
                    db.execute("UPDATE jobs SET state='queued',updated=? WHERE id=?", (time.time(), row["id"]))
                    self._event(db, row["id"], "Recovered interrupted chapter; accepted paragraphs retained", state="queued")
            workers = self.settings.get("assembly_workers", 2)
            self._assembly = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chapter-assembly")
            self._slots = threading.Semaphore(workers + self._parallel_chapters)
            self._started = True
            for index in range(self._parallel_chapters):
                thread = threading.Thread(target=self._run, name=f"chapter-synthesis-{index + 1}", daemon=False)
                thread.start()
                self._threads.append(thread)
            self._thread = self._threads[0]

    def _set_state(self, job_id, state, *, error=None, artifacts=None):
        with self._condition, self._connect() as db:
            db.execute("UPDATE jobs SET state=?,error=?,updated=? WHERE id=?", (state, error, time.time(), job_id))
            if artifacts is not None:
                db.execute("UPDATE jobs SET artifacts=? WHERE id=?", (json.dumps(artifacts), job_id))
            self._event(db, job_id, error or f"Chapter {state}", state=state)
            self._last_activity = time.monotonic()
            self._condition.notify_all()

    def _failure(self, job_id, exc):
        message = f"{type(exc).__name__}: {exc}"
        def redact(values):
            nonlocal message
            if isinstance(values, dict):
                for key, value in values.items():
                    if any(token in key.lower() for token in ("api_key", "secret", "access_key", "token")):
                        if isinstance(value, str) and value:
                            message = message.replace(value, "[redacted]")
                    else:
                        redact(value)
            elif isinstance(values, list):
                for value in values:
                    redact(value)
        redact(self.settings)
        redact(getattr(self.executor, "config", {}))
        # Redact first so truncation cannot expose a prefix of a credential.
        self._set_state(job_id, "failed", error=message[:2000])

    def _done(self):
        with self._condition:
            self._active -= 1
            self._last_activity = time.monotonic()
            self._slots.release()
            self._condition.notify_all()

    def _claim_next(self):
        """Claim state and activity under one lock before another worker reads."""
        with self._condition:
            while True:
                if self._stopping:
                    return None
                with self._connect() as db:
                    row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created,id LIMIT 1").fetchone()
                    if row is not None and self._slots.acquire(blocking=False):
                        try:
                            now = time.time()
                            db.execute("UPDATE jobs SET state='generating',error=NULL,updated=? WHERE id=?",
                                       (now, row["id"]))
                            self._event(db, row["id"], "Chapter generating", state="generating")
                            db.commit()
                        except BaseException:
                            self._slots.release()
                            raise
                        self._active += 1
                        self._last_activity = time.monotonic()
                        return row
                self._condition.wait(timeout=1)

    def _run(self):
        while True:
            row = self._claim_next()
            if row is None:
                return
            job_id = row["id"]
            try:
                if row["renderer"] != self.fingerprint:
                    raise ValueError("Server renderer changed since submission; resubmit with current capabilities")
                spec = validate_spec(json.loads(row["spec"]))
                voices = {voice: self._voice(asset) for voice, asset in spec["voice_bindings"].items()}
                prepared = self.executor.synthesize(spec, voices, self.root / "render-work",
                                                    lambda message: self._log(job_id, message))
                self._set_state(job_id, "assembling")
                self._assembly.submit(self._finish, job_id, spec, prepared)
            except BaseException as exc:
                # Model adapters execute on this worker thread. A raised
                # KeyboardInterrupt/SystemExit must not strand a durable job
                # and its active lease or silently kill the coordinator.
                try:
                    self._failure(job_id, exc)
                finally:
                    self._done()

    def _finish(self, job_id, spec, prepared):
        try:
            paths = self.executor.assemble(spec, prepared, self.root / "render-work",
                                           lambda message: self._log(job_id, message))
            if not isinstance(paths, dict) or not paths or set(paths) - ARTIFACT_NAMES:
                raise ValueError("Executor returned an unsupported artifact; WAV transfers are forbidden")
            artifacts = []
            for name, supplied in sorted(paths.items()):
                path = regular_file(supplied)
                if not path.resolve().is_relative_to(self.root) or path.suffix not in {".flac", ".m4a", ".json"}:
                    raise ValueError("Executor artifact must be compressed audio or JSON in the server store")
                item = {"name": name, "sha256": file_hash(path), "size": path.stat().st_size,
                        "path": path.relative_to(self.root).as_posix()}
                if self.publisher:
                    item["published"] = self.publisher.publish(job_id, name, path, item["sha256"])
                artifacts.append(item)
            self._set_state(job_id, "complete", artifacts=artifacts)
        except BaseException as exc:
            self._failure(job_id, exc)
        finally:
            self._done()

    def idle_seconds(self):
        with self._condition, self._connect() as db:
            pending = db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','generating','assembling') LIMIT 1").fetchone()
            return 0.0 if (pending and not self._maintenance) or self._active else max(0.0, time.monotonic() - self._last_activity)

    def begin_shutdown_if_idle(self, timeout):
        with self._condition:
            if self._draining or self._stopping or self.idle_seconds() < timeout:
                return False
            if self.settings.get("idle_action") == "runpod_stop":
                if not self.settings.get("persistent_data") or self.publisher is None:
                    return False
                with self._connect() as db:
                    for row in db.execute("SELECT artifacts FROM jobs WHERE state='complete'"):
                        if any(not item.get("published") for item in json.loads(row[0])):
                            return False
            self._draining = True
            return True

    def cancel_shutdown(self):
        with self._condition:
            self._draining = False
            self._last_activity = time.monotonic()
            self._condition.notify_all()

    def close(self):
        with self._condition:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._condition.wait()
                return
            self._closing = True
            self._stopping = self._draining = True
            self._condition.notify_all()
        try:
            for thread in self._threads:
                thread.join()
            if self._assembly:
                self._assembly.shutdown(wait=True)
            with self._condition:
                # Voice registration also holds an activity lease. Do not
                # release the process lock while an admitted upload publishes.
                while self._active:
                    self._condition.wait()
            close = getattr(self.executor, "close", None)
            if close:
                close()
        finally:
            with self._condition:
                if self._file_lock:
                    fcntl.flock(self._file_lock.fileno(), fcntl.LOCK_UN)
                    self._file_lock.close()
                self._closed = True
                self._closing = False
                self._condition.notify_all()
