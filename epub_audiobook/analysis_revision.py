"""Revise attribution while preserving voices and validated continuation history."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .analysis_directory import _completed_directory, _continuation_entry, continuation_directories
from .contracts import readiness
from .pipeline import fingerprint, write_json
from .series import _refresh_import_reviews, validate_registry


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render_active(path):
    # New renderers lock the book too, including when work_dir is customized.
    # Retain the original work lock check for older/default renderer sessions.
    return ((path / ".audio-render.lock").exists()
            or (path / "render-work" / ".render.lock").exists())


def related_directories(directories):
    """Include sibling descendants so a repaired parent cannot break their links.

    Explicit directory arguments may also identify descendants in other folders.
    There is no filesystem-wide search for unregistered forward references.
    """
    selected = {Path(path).resolve() for path in directories}
    if not selected:
        raise ValueError("At least one analysis directory is required")
    changed = True
    while changed:
        changed = False
        for parent in {path.parent for path in selected}:
            for manifest in parent.glob("*/manifest.json"):
                directory = manifest.parent.resolve()
                if directory in selected or manifest.parent.is_symlink():
                    continue
                try:
                    link = _continuation_entry(_read(manifest))
                except (ValueError, OSError, TypeError):
                    continue
                if link and (directory / link["path"]).resolve() in selected:
                    selected.add(directory)
                    changed = True
    loaded = {path: _completed_directory(path) for path in selected}
    for path in selected:
        continuation_directories(path)
    return sorted(selected, key=lambda path: (len(loaded[path][1]["books"]), str(path)))


def _snapshot(document):
    return {"source_fingerprint": fingerprint(document["source"]),
            "book": deepcopy(document["source"]["book"]), "cast": deepcopy(document["cast"]),
            "series_provenance": deepcopy(document["provenance"]["series"]),
            "issues": deepcopy(document["issues"]), "readiness": readiness(document)}


def _install(source, destination):
    """An individual file replacement is atomic; the journal covers the chain."""
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=".analysis-replace-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(Path(source).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def revise_directories(revisions, *, directories=None, expected_manifest_hashes=None, progress=print):
    """Stage and validate all changes, back up originals, and refresh descendants.

    This operation only revises attribution and its audit/review state. Source,
    cast, voice identities, and book locations stay fixed. Voice stores, aliases,
    render caches, and existing audiobooks are never part of the write inventory.
    """
    from .cli import export_bundle
    revisions = {Path(path).resolve(): deepcopy(document) for path, document in revisions.items()}
    paths = related_directories(directories or revisions)
    if not set(revisions) <= set(paths):
        raise ValueError("Revision directories must be included in the update chain")
    originals, manifests, names, original_hashes = {}, {}, {}, {}
    for path in paths:
        document, registry, bundle_names = _completed_directory(path)
        originals[path] = document, registry
        manifests[path] = _read(path / "manifest.json")
        names[path] = ["analysis.json", "series-cast.json", *bundle_names, "manifest.json"]
        original_hashes[path] = {name: _hash(path / name) for name in names[path]}
        if expected_manifest_hashes is not None:
            expected = expected_manifest_hashes.get(str(path), expected_manifest_hashes.get(path))
            if expected != original_hashes[path]["manifest.json"]:
                raise ValueError(f"Analysis changed while speaker decisions were being reviewed: {path}")
        candidate = revisions.get(path, document)
        if (candidate["source"] != document["source"] or candidate["cast"] != document["cast"]
                or candidate["provenance"].get("series") != document["provenance"].get("series")):
            raise ValueError("Speaker revisions must preserve source, cast, and series identities")
        if readiness(candidate)["errors"]:
            raise ValueError("Cannot publish an invalid speaker revision")
        if _render_active(path):
            raise ValueError(f"Rendering is active; finish it before revising analysis: {path}")
    locks = []
    transaction = None
    installed = []
    try:
        for path in paths:
            lock = path / ".analysis-update.lock"
            try:
                lock.mkdir()
            except FileExistsError as exc:
                raise ValueError(f"Analysis revision already active or interrupted; inspect {lock}") from exc
            locks.append(lock)
        transaction = paths[0] / ".analysis-revisions" / uuid.uuid4().hex
        transaction.mkdir(parents=True)
        snapshots, prepared, journal = {}, {}, {"version": 1, "status": "preparing", "books": []}
        for number, path in enumerate(paths, 1):
            original, registry = originals[path]
            document = deepcopy(revisions.get(path, original))
            registry = deepcopy(registry)
            registry["books"] = [deepcopy(snapshots.get((registry["series_id"], book["source_fingerprint"]), book))
                                 for book in registry["books"]]
            # A resolved parent attribution may have been copied into descendants
            # as a review issue. Refresh only such inherited review records.
            history = dict(registry, books=registry["books"][:-1])
            _refresh_import_reviews(document, history, document["provenance"]["series"])
            document["readiness"] = readiness(document)
            snapshot = _snapshot(document)
            registry["books"][-1] = snapshot
            snapshots[(registry["series_id"], snapshot["source_fingerprint"])] = snapshot
            validate_registry(registry)
            base = transaction / f"book-{number:04d}"
            before, after = base / "before", base / "after"
            before.mkdir(parents=True)
            for name in names[path]:
                (before / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path / name, before / name)
                if _hash(before / name) != original_hashes[path][name]:
                    raise ValueError(f"Analysis changed while staging the revision: {path}")
            export_bundle(document, after, allow_review=True)
            write_json(after / "analysis.json", document)
            write_json(after / "series-cast.json", registry)
            manifest = _read(after / "manifest.json")
            if {row["path"] for row in manifest["files"]["scripts"]} != {
                    row["path"] for row in manifests[path]["files"]["scripts"]}:
                raise ValueError("Speaker revisions must preserve the chapter file layout")
            manifest["analysis_directory"] = {"version": 1, "sha256": {
                name: _hash(after / name) for name in ("analysis.json", "series-cast.json")}}
            link = deepcopy(manifests[path].get("continuation"))
            if link:
                parent = (path / link["path"]).resolve()
                parent_folder = prepared.get(parent, parent)
                parent_manifest = _read(parent_folder / "manifest.json")
                parent_document, parent_registry, _ = _completed_directory(parent_folder)
                if registry["books"][:-1] != parent_registry["books"]:
                    raise ValueError("Revised continuation histories disagree")
                link.update(analysis_sha256=parent_manifest["analysis_directory"]["sha256"]["analysis.json"],
                            series_cast_sha256=parent_manifest["analysis_directory"]["sha256"]["series-cast.json"])
            manifest["continuation"] = link
            write_json(after / "manifest.json", manifest, overwrite=True)
            _completed_directory(after)
            prepared[path] = after
            journal["books"].append({"directory": str(path), "backup": str(before), "prepared": str(after),
                                     "before_sha256": original_hashes[path],
                                     "after_sha256": {name: _hash(after / name) for name in names[path]}})
        journal["status"] = "prepared"
        write_json(transaction / "revision.json", journal)
        # No source file is changed until the whole chain has been prepared.
        for path in paths:
            if any(_hash(path / name) != checksum for name, checksum in original_hashes[path].items()):
                raise ValueError(f"Analysis changed before publication: {path}")
            if _render_active(path):
                raise ValueError(f"Rendering started during revision: {path}")
        journal["status"] = "applying"
        write_json(transaction / "revision.json", journal, overwrite=True)
        for entry in journal["books"]:
            path = Path(entry["directory"])
            for name in names[path]:
                # Record the attempted replacement first: an interruption may
                # arrive after os.replace succeeds but before _install returns.
                installed.append((entry, name))
                _install(Path(entry["prepared"]) / name, path / name)
            progress(f"Updated speaker assignments and continuation metadata: {path}")
        for path in paths:
            _completed_directory(path)
            continuation_directories(path)
        journal["status"] = "complete"
        write_json(transaction / "revision.json", journal, overwrite=True)
        return {"directories": [str(path) for path in paths], "backup_directory": str(transaction)}
    except BaseException:
        rollback_errors = []
        for entry, name in reversed(installed):
            try:
                path = Path(entry["directory"]) / name
                if _hash(path) not in {entry["before_sha256"][name], entry["after_sha256"][name]}:
                    raise ValueError("A concurrent writer changed this file")
                _install(Path(entry["backup"]) / name, path)
            except BaseException as exc:
                rollback_errors.append(f"{entry['directory']}/{name}: {exc}")
        if transaction is not None and (transaction / "revision.json").exists():
            journal["status"] = "rollback_incomplete" if rollback_errors else "rolled_back"
            journal["rollback_errors"] = rollback_errors
            write_json(transaction / "revision.json", journal, overwrite=True)
        if rollback_errors:
            raise ValueError(f"Revision rollback needs attention; originals and journal retained in {transaction}")
        raise
    finally:
        for lock in reversed(locks):
            lock.rmdir()
