"""Durable per-stage checkpoints so an interrupted analysis resumes from completed work.

The pending directory (``<output>/.analysis-pending/``) is never mistakable for a
completed bundle: it contains no ``manifest.json``, and ``_completed_directory``
requires the exact manifest inventory, so an interrupted output directory is
always invalid until publication replaces it.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

try:
    import fcntl
except ImportError:  # Non-POSIX platforms have no flock; the lease degrades to a marker file.
    fcntl = None

from .contracts import validate_document
from .pipeline import fingerprint, write_json
from .prompts import PROMPT_VERSION

PROGRESS_VERSION = 1
PENDING_DIRNAME = ".analysis-pending"
ANALYZER_VERSION = "0.1.0"
STAGES = ("analyzed", "speech-normalized-1", "samples-prepared-1",
          "speakers-decided", "speech-normalized-2", "samples-prepared-2")

_SCOPE = "analysis-progress"
_OWNER_FORMAT = "analysis-progress-owner"
_STATUSES = ("running", "interrupted", "failed")
_IDENTITY_FIELDS = ("source_fingerprint", "prompt_version", "analyzer_version",
                    "provider", "options_fingerprint")
_ERROR_LIMIT = 2000


def _read_json(path):
    def reject(_):
        raise ValueError("Non-finite numbers are not valid JSON")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _is_hex_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _pending_directory(output):
    return Path(output) / PENDING_DIRNAME


def _snapshot_relative(stage):
    return f"snapshots/{stage}.json"


def _next_stage(completed):
    return STAGES[len(completed)] if len(completed) < len(STAGES) else None


def _sign(record):
    return fingerprint({key: value for key, value in record.items() if key != "record_sha256"})


def _save_state(directory, record):
    signed = dict(record)
    signed["record_sha256"] = _sign(signed)
    write_json(Path(directory) / "state.json", signed, overwrite=True)


def pending_identity(document_or_source, client_identity, options_fingerprint):
    """Identity every input that must match exactly for a resume to be valid."""
    value = document_or_source
    if isinstance(value, dict) and isinstance(value.get("source"), dict):
        value = value["source"]
    looks_like_source = (isinstance(value, dict) and isinstance(value.get("book"), dict)
                         and isinstance(value.get("blocks"), list))
    if not looks_like_source:
        raise ValueError("analysis progress identity source: expected an analysis document or "
                         f"source object; received {type(value).__name__}")
    if not isinstance(client_identity, dict):
        raise ValueError("analysis progress identity provider: expected the client identity "
                         f"object; received {type(client_identity).__name__}")
    if not _is_hex_digest(options_fingerprint):
        raise ValueError("analysis progress identity options_fingerprint: expected a 64-character "
                         f"hex digest; received {options_fingerprint!r}")
    return {"source_fingerprint": fingerprint(value),
            "prompt_version": str(PROMPT_VERSION),
            "analyzer_version": ANALYZER_VERSION,
            "provider": deepcopy(client_identity),
            "options_fingerprint": options_fingerprint}


def _validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != set(_IDENTITY_FIELDS):
        raise ValueError("analysis progress identity: expected exactly "
                         f"{', '.join(_IDENTITY_FIELDS)}")
    for field in ("source_fingerprint", "options_fingerprint"):
        if not _is_hex_digest(identity[field]):
            raise ValueError(f"analysis progress identity {field}: expected a 64-character hex digest")
    for field in ("prompt_version", "analyzer_version"):
        if not isinstance(identity[field], str) or not identity[field].strip():
            raise ValueError(f"analysis progress identity {field}: expected a nonempty string")
    if not isinstance(identity["provider"], dict):
        raise ValueError("analysis progress identity provider: expected an object (client.identity)")
    return deepcopy(identity)


def is_pending_directory(output):
    """True when output is a real directory whose only entry is pending analysis state."""
    output = Path(output)
    try:
        if output.is_symlink() or not output.is_dir():
            return False
        entries = list(output.iterdir())
        if len(entries) != 1 or entries[0].name != PENDING_DIRNAME:
            return False
        pending = entries[0]
        return not pending.is_symlink() and pending.is_dir()
    except OSError:
        return False


class _Lease:
    """Advisory exclusive lease; the kernel releases the flock when the process dies."""

    def __init__(self, path):
        self._file = None
        if path.is_symlink():
            raise ValueError(f"Pending analysis lease must be a regular file: {path}")
        self._file = open(path, "a+b")
        if fcntl is not None:
            try:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._file.close()
                raise

    def close(self):
        if self._file is not None and not self._file.closed:
            if fcntl is not None:
                try:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            self._file.close()
        self._file = None


def _acquire_lease(path, output):
    try:
        return _Lease(path)
    except OSError as exc:
        raise FileExistsError(f"Another analysis is running on this directory: {output}; "
                              "wait for it to finish") from exc


def _write_owner(directory):
    write_json(Path(directory) / "owner.json",
               {"format": _OWNER_FORMAT, "version": 1,
                "pid": os.getpid(), "token": uuid.uuid4().hex}, overwrite=True)


def _initial_state(identity):
    return {"scope": _SCOPE, "version": PROGRESS_VERSION, "identity": deepcopy(identity),
            "status": "running", "current_stage": None, "completed_stages": [],
            "snapshots": {}, "error": None}


class PendingAnalysis:
    """Handle owning the pending directory's lease and progress record."""

    def __init__(self, output, identity, lease):
        self.output = Path(output)
        self.directory = _pending_directory(output)
        self.identity = deepcopy(identity)
        self._lease = lease

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()

    def close(self):
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    def _record(self):
        return _load_validated_state(self.directory)

    def begin_stage(self, stage):
        _require_stage(stage)
        record = self._record()
        completed = record["completed_stages"]
        expected = _next_stage(completed)
        if expected is None:
            raise ValueError(f"analysis progress stage {stage}: all stages are already completed")
        if stage in completed:
            raise ValueError(f"analysis progress stage {stage}: stage is already completed")
        if stage != expected:
            raise ValueError(f"analysis progress stage {stage}: expected the next stage "
                             f"{expected!r}; completed stages: {', '.join(completed) or 'none'}")
        record["status"] = "running"
        record["current_stage"] = stage
        record["error"] = None
        _save_state(self.directory, record)

    def complete_stage(self, stage, document):
        _require_stage(stage)
        record = self._record()
        completed = record["completed_stages"]
        expected = _next_stage(completed)
        if expected is None:
            raise ValueError(f"analysis progress stage {stage}: all stages are already completed")
        if stage in completed:
            raise ValueError(f"analysis progress stage {stage}: stage is already completed")
        if stage != expected:
            raise ValueError(f"analysis progress stage {stage}: expected the next stage "
                             f"{expected!r}; completed stages: {', '.join(completed) or 'none'}")
        snapshots = self.directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        path = snapshots / f"{stage}.json"
        if path.exists():
            # An unrecorded snapshot belongs to an attempt that crashed between the
            # snapshot write and the state update; identical content is reused, and
            # only such orphans may ever be replaced.
            try:
                orphan_matches = fingerprint(_read_json(path)) == fingerprint(document)
            except (OSError, ValueError):
                orphan_matches = False
            if not orphan_matches:
                path.unlink()
                write_json(path, document)
        else:
            write_json(path, document)  # Atomic no-clobber publication.
        record["snapshots"][stage] = {"path": _snapshot_relative(stage),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        record["completed_stages"] = completed + [stage]
        record["current_stage"] = None
        record["status"] = "running"
        _save_state(self.directory, record)

    def mark_failed(self, error, *, interrupted=False):
        message = error if isinstance(error, str) else str(error)
        if not message.strip():
            message = "Analysis was interrupted" if interrupted else "Analysis failed"
        if len(message) > _ERROR_LIMIT:
            message = message[:_ERROR_LIMIT] + "[truncated]"
        record = self._record()
        record["status"] = "interrupted" if interrupted else "failed"
        record["error"] = message
        _save_state(self.directory, record)


def _require_stage(stage):
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f"analysis progress stage: expected one of {', '.join(STAGES)}; "
                         f"received {stage!r}")


def create_pending(output, identity):
    """Create the output directory with fresh progress and hold its lease.

    An existing pending directory is attached to instead: the flock is released
    when the previous process died, so acquiring it proves no analysis is live.
    A live run keeps the lease, and the second create is refused.
    """
    output = Path(output)
    identity = _validate_identity(identity)
    pending = _pending_directory(output)
    if output.exists() or output.is_symlink():
        if not is_pending_directory(output):
            raise FileExistsError(f"Output directory already exists: {output}; choose another directory")
        handle = PendingAnalysis(output, identity, _acquire_lease(pending / "lease", output))
        _write_owner(pending)
        (pending / "snapshots").mkdir(exist_ok=True)
        if not (pending / "state.json").exists():
            # Crash between directory creation and the first state write; no
            # recorded progress exists, so this run starts the record fresh.
            _save_state(pending, _initial_state(identity))
        return handle
    output.mkdir(parents=True)
    try:
        pending.mkdir()
        (pending / "snapshots").mkdir()
        handle = PendingAnalysis(output, identity, _acquire_lease(pending / "lease", output))
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        try:
            output.rmdir()
        except OSError:
            pass
        raise
    _write_owner(pending)
    _save_state(pending, _initial_state(identity))
    return handle


def _load_validated_state(directory):
    path = Path(directory) / "state.json"

    def invalid(detail):
        return ValueError(f"Invalid pending analysis record: {path} {detail}; "
                          "remove the directory or choose another output")

    try:
        record = _read_json(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Pending analysis record is missing or unreadable: {path} ({exc}); "
                         "remove the directory or choose another output") from exc
    if not isinstance(record, dict):
        raise invalid("(expected an object)")
    if record.get("scope") != _SCOPE:
        raise invalid("(scope)")
    if record.get("version") != PROGRESS_VERSION:
        raise invalid("(version)")
    signature = record.get("record_sha256")
    if not _is_hex_digest(signature) or signature != _sign(record):
        raise invalid("(record_sha256 mismatch; the record was modified or damaged)")
    if record.get("status") not in _STATUSES:
        raise invalid(f"(status must be one of {', '.join(_STATUSES)})")
    completed = record.get("completed_stages")
    if (not isinstance(completed, list) or any(not isinstance(stage, str) for stage in completed)
            or len(set(completed)) != len(completed)
            or completed != list(STAGES[:len(completed)])):
        raise invalid("(completed_stages must be an ordered prefix of the analysis stages)")
    current = record.get("current_stage")
    if current is not None and (not isinstance(current, str) or current not in STAGES):
        raise invalid("(current_stage)")
    snapshots = record.get("snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(completed):
        raise invalid("(snapshots must record exactly the completed stages)")
    for stage, entry in snapshots.items():
        if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
                or entry["path"] != _snapshot_relative(stage)
                or not _is_hex_digest(entry["sha256"])):
            raise invalid(f"(snapshots.{stage})")
    try:
        _validate_identity(record.get("identity"))
    except ValueError as exc:
        raise invalid(f"(identity: {exc})") from exc
    if record.get("error") is not None and not isinstance(record["error"], str):
        raise invalid("(error must be a string or null)")
    return record


def inspect_pending(output):
    """Return the validated progress record, or None when nothing is pending."""
    output = Path(output)
    if not is_pending_directory(output):
        return None
    return _load_validated_state(_pending_directory(output))


def resume_document(output, identity):
    """Return (document, next_stage) to continue from, or None when nothing is pending.

    The last completed stage's snapshot is verified against the signed record and
    revalidated as an analysis document before it is trusted. A record whose
    identity differs from the current run is refused, never silently restarted.
    """
    record = inspect_pending(output)
    if record is None:
        return None
    wanted = _validate_identity(identity)
    differing = [field for field in _IDENTITY_FIELDS if record["identity"].get(field) != wanted[field]]
    if differing:
        raise ValueError("Pending analysis progress belongs to a different run configuration "
                         f"({', '.join('identity.' + field for field in differing)}): "
                         f"{Path(output) / PENDING_DIRNAME / 'state.json'}; "
                         "remove the directory or choose another output")
    completed = record["completed_stages"]
    if not completed:
        return None, STAGES[0]
    stage = completed[-1]
    snapshot = Path(output) / PENDING_DIRNAME / record["snapshots"][stage]["path"]
    try:
        raw = snapshot.read_bytes()
    except OSError as exc:
        raise ValueError(f"Pending analysis snapshot is missing or unreadable: {snapshot}; "
                         "remove the directory or choose another output") from exc
    if hashlib.sha256(raw).hexdigest() != record["snapshots"][stage]["sha256"]:
        raise ValueError(f"Pending analysis snapshot checksum mismatch: {snapshot}; "
                         "remove the directory or choose another output")
    try:
        document = _read_json(snapshot)
    except ValueError as exc:
        raise ValueError(f"Pending analysis snapshot is corrupt: {snapshot} ({exc}); "
                         "remove the directory or choose another output") from exc
    errors = validate_document(document)
    if errors:
        raise ValueError(f"Pending analysis snapshot is not a valid analysis document: {snapshot} "
                         f"({'; '.join(errors[:3])})")
    index = STAGES.index(stage) + 1
    return document, (STAGES[index] if index < len(STAGES) else None)


def discard_pending(output):
    """Remove the pending analysis state; refuses any directory that is not one."""
    output = Path(output)
    pending = _pending_directory(output)
    if not pending.exists():
        if output.is_dir() and any(output.iterdir()):
            raise ValueError(f"Refusing to remove {output}; it is not a pending analysis directory")
        return
    if not is_pending_directory(output):
        raise ValueError(f"Refusing to remove unexpected content at {pending}; "
                         "not a pending analysis directory")
    shutil.rmtree(pending)
