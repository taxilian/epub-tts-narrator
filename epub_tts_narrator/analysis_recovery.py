"""Recover coherent source-text drift from a verified publication snapshot."""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import uuid

from .analysis_directory import (_bundle_files, _completed_directory, _continuation_chain, _continuation_entry,
                                 _read_json, _regular_file, _validate_script_rows)
from .analysis_revision import _hash, _install, _render_active, related_directories
from .contracts import review_items, validate_document
from .pipeline import write_json

RECOVERY_DIRNAME = ".analysis-recovery"
JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class AnalysisRecovery:
    """The preserved drift and its relationship to the supplied source."""
    backup_directory: Path
    baseline_revision: Path
    preserved_files: tuple[str, ...]
    source_matches_preserved_edit: bool


@dataclass(frozen=True, slots=True)
class _Baseline:
    revision: Path
    prepared: Path
    document: dict[str, JsonValue]
    registry: dict[str, JsonValue]
    names: tuple[str, ...]
    hashes: dict[str, str]


def _inventory(directory):
    manifest = _read_json(_regular_file(directory, "manifest.json"))
    bundle_names = _bundle_files(manifest)
    _continuation_entry(manifest)
    names = ("analysis.json", "series-cast.json", *bundle_names, "manifest.json")
    expected = dict(manifest["sha256"])
    expected.update(manifest.get("analysis_directory", {}).get("sha256", {}))
    return manifest, names, expected


def _write_journal(transaction, journal):
    path = (Path(transaction) / "recovery.json")
    write_json(path, journal, overwrite=path.exists())
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _pending_recovery(directory):
    root = directory / RECOVERY_DIRNAME
    pending = []
    if root.is_symlink():
        raise ValueError(f"Analysis recovery directory must not be a symlink: {root}")
    if root.exists():
        for transaction in sorted(root.iterdir()):
            if transaction.is_symlink():
                raise ValueError(f"Interrupted analysis recovery must not be a symlink: {transaction}")
            if not (Path(transaction) / "recovery.json").exists():
                continue
            journal = _read_json(_regular_file(transaction, "recovery.json"))
            if (journal.get("version") == 1 and journal.get("scope") == "source-recovery"
                    and journal.get("status") in {"prepared", "applying"}):
                pending.append(transaction)
    if len(pending) > 1:
        raise ValueError(f"Multiple interrupted analysis recoveries need attention: {root}")
    return pending[0] if pending else None


def _find_baseline(directory, names, required_revision=None):
    root = directory / ".analysis-revisions"
    candidates = [required_revision] if required_revision is not None else sorted(root.glob("*/revision.json"))
    for candidate in candidates:
        journal_path = candidate / "revision.json" if candidate.is_dir() else candidate
        transaction = journal_path.parent
        try:
            if transaction.is_symlink() or not transaction.resolve().is_relative_to(root.resolve()):
                continue
            journal = _read_json(_regular_file(transaction, "revision.json"))
            if journal.get("version") != 1 or journal.get("status") != "complete":
                continue
            for entry in journal["books"]:
                prepared = Path(entry["prepared"]).resolve()
                if (Path(entry["directory"]).resolve() != directory
                        or not prepared.is_relative_to(transaction.resolve())
                        or set(entry["after_sha256"]) != set(names)):
                    continue
                if (prepared / "manifest.json").read_bytes() != (directory / "manifest.json").read_bytes():
                    continue
                hashes = {name: _hash(prepared / name) for name in names}
                if hashes != entry["after_sha256"]:
                    continue
                document, registry, _ = _completed_directory(prepared)
                return _Baseline(transaction, prepared, document, registry, tuple(names), hashes)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    raise ValueError(f"No verified publication snapshot matches the saved manifest: {directory}")


def _validate_drift(directory, baseline):
    document = _read_json(_regular_file(directory, "analysis.json"))
    if validate_document(document):
        raise ValueError("Recovery requires a structurally valid edited analysis")
    if (_read_json(_regular_file(directory, "source.json")) != document["source"]
            or _read_json(_regular_file(directory, "cast.json")) != document["cast"]
            or _read_json(_regular_file(directory, "review.json")) != review_items(document)
            or _read_json(_regular_file(directory, "series-cast.json")) != baseline.registry):
        raise ValueError("Recovery requires source-derived edits with unchanged cast and series identity")
    current_fixed = {key: value for key, value in document.items() if key not in {"source", "segments"}}
    baseline_fixed = {key: value for key, value in baseline.document.items() if key not in {"source", "segments"}}
    if current_fixed != baseline_fixed:
        raise ValueError("Recovery permits only source-block and segment text differences")
    current_source, baseline_source = deepcopy(document["source"]), deepcopy(baseline.document["source"])
    if len(current_source.get("blocks", [])) != len(baseline_source.get("blocks", [])):
        raise ValueError("Recovery does not permit source layout changes")
    changed_blocks = set()
    for current, saved in zip(current_source["blocks"], baseline_source["blocks"]):
        if current["text"] != saved["text"]:
            changed_blocks.add(current["id"])
        current["text"] = saved["text"] = ""
    if not changed_blocks or current_source != baseline_source:
        raise ValueError("Recovery permits only source-block text differences")
    current_segments, baseline_segments = deepcopy(document["segments"]), deepcopy(baseline.document["segments"])
    if len(current_segments) != len(baseline_segments):
        raise ValueError("Recovery does not permit segment layout changes")
    changed_segment_blocks = set()
    for current, saved in zip(current_segments, baseline_segments):
        if current["text"] != saved["text"]:
            changed_segment_blocks.add(current["block_id"])
        current["text"] = saved["text"] = ""
    if current_segments != baseline_segments or not changed_segment_blocks <= changed_blocks:
        raise ValueError("Recovery requires segment text to follow only the edited source blocks")
    _validate_script_rows(directory, _read_json(_regular_file(directory, "manifest.json")), document)
    return document


def _require_idle(directory):
    from .voice_profile_guard import _require_ready
    if reason := _render_active(directory):
        raise ValueError(f"Rendering is active; finish it before recovering analysis: {directory} ({reason})")
    for name in (".analysis-pending", ".analysis-update", ".analysis-publish"):
        if (directory / name).exists() or (directory / name).is_symlink():
            raise ValueError(f"Another analysis or update is pending; resume it before recovery: {directory / name}")
    voice_locks = [directory / "voices" / ".lock", *(directory / "voices").glob("*/.lock")]
    if any(path.exists() or path.is_symlink() for path in voice_locks):
        raise ValueError(f"Casting is active; finish it before recovering analysis: {directory}")
    _require_ready(directory)
    for journal in (directory / ".analysis-revisions").glob("*/revision.json"):
        try:
            status = _read_json(journal).get("status")
        except (OSError, ValueError):
            continue
        if status in {"preparing", "prepared", "applying", "rollback_incomplete"}:
            raise ValueError(f"An interrupted analysis revision needs attention before recovery: {journal.parent}")


def _check_authoritative_source(source, baseline):
    if not isinstance(source, dict) or not isinstance(source.get("book"), dict) or not isinstance(source.get("chapters"), list) or not isinstance(source.get("blocks"), list):
        raise ValueError("Recovery requires an extracted source object as the authoritative input")
    from .book_update import compare_sources
    comparison = compare_sources(baseline.document["source"], source, preserve_citations_from=baseline.document["cast"])
    if not isinstance(merged := comparison["source"], dict):
        raise ValueError("Recovery source alignment did not produce a source object")
    return merged


def _copy_inventory(source, destination, names, expected):
    for name in names:
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        _install(source / name, destination / name)
    if {name: _hash(destination / name) for name in names} != expected:
        raise ValueError(f"Recovery backup verification failed: {destination}")


def _apply_recovery(directory, transaction, journal, baseline, drift_document, authoritative_source):
    _require_idle(directory)
    before, after = Path(journal["backup"]), Path(journal["prepared"])
    names = baseline.names
    if {name: _hash(before / name) for name in names} != journal["before_sha256"]:
        raise ValueError(f"Recovery backup is missing or tampered: {before}")
    _copy_inventory(baseline.prepared, after, names, baseline.hashes)
    authoritative_merged = _check_authoritative_source(authoritative_source, baseline)
    ancestors = _continuation_chain(directory, baseline.document, baseline.registry)
    from .voice_profile_guard import _require_ready
    for path in ancestors:
        _require_ready(path)
    for name in names:
        current = _hash(directory / name)
        if current not in {journal["before_sha256"][name], baseline.hashes[name]}:
            raise ValueError(f"Analysis changed during recovery; preserved bytes were not overwritten: {directory / name}")
    journal["status"] = "applying"
    _write_journal(transaction, journal)
    for name in journal["mismatches"]:
        if _hash(directory / name) != baseline.hashes[name]:
            _install(after / name, directory / name)
    related_directories([directory])
    journal["status"] = "complete"
    _write_journal(transaction, journal)
    matches_preserved = (tuple(block["text"] for block in authoritative_merged["blocks"])
                         == tuple(block["text"] for block in drift_document["source"]["blocks"]))
    return AnalysisRecovery(Path(transaction), baseline.revision, tuple(journal["mismatches"]), matches_preserved)


def _resume_recovery(directory, transaction, authoritative_source):
    journal = _read_json(_regular_file(transaction, "recovery.json"))
    required = {"version", "scope", "status", "directory", "baseline_revision", "backup", "prepared",
                "before_sha256", "after_sha256", "mismatches"}
    if (set(journal) != required or journal["directory"] != str(directory)
            or Path(journal["backup"]).resolve() != (transaction / "before").resolve()
            or Path(journal["prepared"]).resolve() != (transaction / "after").resolve()):
        raise ValueError(f"Invalid analysis recovery journal: {(Path(transaction) / "recovery.json")}")
    manifest, names, _ = _inventory(directory)
    baseline = _find_baseline(directory, names, Path(journal["baseline_revision"]))
    if journal["after_sha256"] != baseline.hashes or set(journal["before_sha256"]) != set(names):
        raise ValueError(f"Recovery journal does not match its verified baseline: {(Path(transaction) / "recovery.json")}")
    drift = _validate_drift(Path(journal["backup"]), baseline)
    mismatches = [name for name in names if journal["before_sha256"][name] != baseline.hashes[name]]
    allowed = {"source.json", "analysis.json"} | {row["path"] for row in manifest["files"]["scripts"]}
    if mismatches != journal["mismatches"] or not set(mismatches) <= allowed:
        raise ValueError(f"Recovery journal has an unsafe replacement inventory: {(Path(transaction) / "recovery.json")}")
    return _apply_recovery(directory, transaction, journal, baseline, drift, authoritative_source)


def _start_recovery(directory, authoritative_source):
    manifest, names, expected = _inventory(directory)
    before_hashes = {name: _hash(directory / name) for name in names}
    mismatches = [name for name in names if name != "manifest.json" and before_hashes[name] != expected.get(name)]
    allowed = {"source.json", "analysis.json"} | {row["path"] for row in manifest["files"]["scripts"]}
    if not {"source.json", "analysis.json"} <= set(mismatches) or not set(mismatches) <= allowed:
        raise ValueError("Recovery supports only coherent source.json, analysis.json, and chapter-script checksum drift")
    baseline = _find_baseline(directory, names)
    drift = _validate_drift(directory, baseline)
    _check_authoritative_source(authoritative_source, baseline)
    transaction = directory / RECOVERY_DIRNAME / uuid.uuid4().hex
    before, after = transaction / "before", transaction / "after"
    _copy_inventory(directory, before, names, before_hashes)
    journal = {"version": 1, "scope": "source-recovery", "status": "prepared", "directory": str(directory),
               "baseline_revision": str(baseline.revision), "backup": str(before), "prepared": str(after),
               "before_sha256": before_hashes, "after_sha256": baseline.hashes, "mismatches": mismatches}
    _write_journal(transaction, journal)
    return _apply_recovery(directory, transaction, journal, baseline, drift, authoritative_source)


def recover_analysis_directory(directory: str | Path, authoritative_source: dict[str, JsonValue]) -> AnalysisRecovery | None:
    """Restore a verified baseline; never adopt unverified edited bytes."""
    from .renderer.run_locks import acquire_directory_lock
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError(f"Analysis directory must not be a symlink: {directory}")
    directory = directory.resolve()
    pending = _pending_recovery(directory)
    if pending is None:
        try:
            _completed_directory(directory)
            return None
        except ValueError:
            pass
    lock_path = directory / ".analysis-update.lock"
    try:
        lock = acquire_directory_lock(lock_path)
    except FileExistsError as exc:
        raise ValueError(f"Analysis revision is active or cannot be proven stale: {lock_path}") from exc
    try:
        _require_idle(directory)
        transaction = _pending_recovery(directory)
        if transaction is not None:
            return _resume_recovery(directory, transaction, authoritative_source)
        try:
            _completed_directory(directory)
            return None
        except ValueError as error:
            if "checksum mismatch" not in str(error):
                raise
        return _start_recovery(directory, authoritative_source)
    finally:
        lock.rmdir()
