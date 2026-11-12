"""Revise attribution while preserving voices and validated continuation history."""

from contextlib import ExitStack, contextmanager
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


_RENDER_LOCKS = (".audio-render.lock", ("render-work", ".render.lock"))


def _render_active(path):
    """Return why a renderer still holds this book, or None when it is idle.

    New renderers lock the book too, including when work_dir is customized.
    Retain the original work lock check for older/default renderer sessions.
    A marked lock whose kernel lease no live process holds is proven idle by
    the read-only probe; anything legacy, malformed, or genuinely live reports
    an actionable reason instead.
    """
    from .renderer.run_locks import directory_lock_status
    reasons = []
    for name in _RENDER_LOCKS:
        lock = path.joinpath(*name) if isinstance(name, tuple) else path / name
        reason = directory_lock_status(lock)
        if reason:
            reasons.append(f"{lock}: {reason}")
    return "; ".join(reasons) or None


@contextmanager
def reserve_revision_scope(directories, *, scope="attribution"):
    """Hold the publication write scope before expensive revision work begins.

    Every book in the chain has its ``.analysis-update.lock`` acquired before
    any costly work. For scopes other than ``attribution``, render activity is
    probed before and after the locks so a renderer that started in between is
    refused here. Attribution revisions stay saveable while renderers are live,
    including the same book and its descendants: the running renderer finishes
    its loaded snapshot, a renderer starting during the publication window
    refuses itself through the ``.analysis-update.lock`` handshake in
    ``renderer.workflow._lock_book_audio``, and the next render sees the saved
    revision and re-renders the chapters whose attribution changed. A renderer
    starting after the reservation cannot hit a late lock conflict either way.
    Yield the mapping of directory to owned lock for
    ``revise_directories(reserved_update_locks=...)``; release on exit.
    """
    from .renderer.run_locks import acquire_directory_lock
    paths = related_directories(directories)
    locks = {}
    try:
        if scope != "attribution":
            _require_renderers_idle(paths)
        for path in paths:
            lock_path = path / ".analysis-update.lock"
            try:
                locks[path] = acquire_directory_lock(lock_path)
            except FileExistsError as exc:
                raise ValueError(
                    f"Analysis revision is already active, or the lock cannot be proven stale: {lock_path}. "
                    f"A marked lock left by a crashed process is reclaimed automatically when you rerun.") from exc
        if scope != "attribution":
            _require_renderers_idle(paths)
        yield dict(locks)
    finally:
        for lock in reversed(list(locks.values())):
            lock.rmdir()


def _require_renderers_idle(paths):
    for path in paths:
        reason = _render_active(path)
        if reason:
            raise ValueError(f"Rendering is active; finish it before revising analysis: {path} ({reason})")


def _require_owned_reservation(path, lock):
    from .renderer.run_locks import DirectoryLock
    if not isinstance(lock, DirectoryLock) or not lock.still_owns(path / ".analysis-update.lock"):
        raise ValueError(
            f"Reserved publication lock is not currently owned by this process: "
            f"{path / '.analysis-update.lock'}; reserve the revision scope again before publishing")


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


PROFILE_FIELDS = {"description", "traits", "speechTraits"}
_REFRESH_PROVENANCE = {"voice_samples", "voice_profile_refresh"}
_REVISION_SIDECARS = {
    "voice-profiles": frozenset({"audition-scripts.json"}),
    "source-edit": frozenset({"audition-scripts.json", "pronunciations.json", "scoped-pronunciations.json",
                              "performance-plan.json"}),
}


def _outside_profile_fields(voice):
    return {key: value for key, value in voice.items() if key not in PROFILE_FIELDS}


def profile_revision_differences(original, candidate):
    """Return every candidate change outside the voice-profile refresh allowlist.

    Only cast voice description/traits/speechTraits, their provenance
    ``cast_fingerprint`` rebind, and the refresh audit record may differ.
    Source, segments, issues, characters, roles, voice identity and order,
    sample scripts, and all other provenance stay exact.
    """
    errors = []
    for key in ("format", "version", "source", "segments", "issues"):
        if candidate[key] != original[key]:
            errors.append(key)
    if candidate["provenance"].get("series") != original["provenance"].get("series"):
        errors.append("provenance.series")

    def unbound(document):
        return {key: value for key, value in document["provenance"].items() if key not in _REFRESH_PROVENANCE}

    if unbound(candidate) != unbound(original):
        errors.append("provenance")
    refresh = candidate["provenance"].get("voice_profile_refresh")
    if refresh is not None and not isinstance(refresh, dict):
        errors.append("provenance.voice_profile_refresh")
    before, after = original["provenance"].get("voice_samples"), candidate["provenance"].get("voice_samples")
    if before is not None or after is not None:
        def frozen(samples):
            return {key: value for key, value in samples.items() if key != "cast_fingerprint"}
        if (not isinstance(before, dict) or not isinstance(after, dict)
                or frozen(before) != frozen(after)
                or after.get("cast_fingerprint") != fingerprint(candidate["cast"])):
            errors.append("provenance.voice_samples scripts or binding")
    if candidate["cast"]["characters"] != original["cast"]["characters"]:
        errors.append("cast.characters")
    if candidate["cast"]["roles"] != original["cast"]["roles"]:
        errors.append("cast.roles")
    if ({key: value for key, value in candidate["cast"].items() if key != "voices"}
            != {key: value for key, value in original["cast"].items() if key != "voices"}):
        errors.append("cast metadata")
    before_voices, after_voices = original["cast"]["voices"], candidate["cast"]["voices"]
    if [voice.get("id") for voice in after_voices] != [voice.get("id") for voice in before_voices]:
        errors.append("cast.voices identity or order")
    elif any(_outside_profile_fields(after) != _outside_profile_fields(before)
             for before, after in zip(before_voices, after_voices)):
        errors.append("cast.voice fields outside description/traits/speechTraits")
    fixed = {"format", "version", "source", "cast", "segments", "issues", "provenance", "readiness"}
    if ({key for key in candidate if key not in fixed} != {key for key in original if key not in fixed}
            or any(candidate[key] != original[key] for key in candidate if key not in fixed)):
        errors.append("document metadata")
    return errors


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


def revise_directories(revisions, *, directories=None, expected_manifest_hashes=None, progress=print,
                       scope="attribution", sidecars=None, speaker_decision_inputs=None,
                       reserved_update_locks=None):
    """Publish a revision, coordinating manual decisions with profile refreshes."""
    if scope not in {"attribution", "source-edit", "voice-profiles", "speaker-decisions"}:
        raise ValueError("Revision scope must be attribution, source-edit, voice-profiles, or speaker-decisions")
    if speaker_decision_inputs is not None and scope != "speaker-decisions":
        raise ValueError("speaker_decision_inputs are accepted only for speaker-decisions revisions")
    if reserved_update_locks is not None and scope == "speaker-decisions":
        raise ValueError("speaker-decisions revisions acquire their own locks through the profile-refresh lease")
    if scope != "speaker-decisions":
        return _revise_directories(revisions, directories=directories,
                                   expected_manifest_hashes=expected_manifest_hashes,
                                   progress=progress, scope=scope, sidecars=sidecars,
                                   reserved_update_locks=reserved_update_locks)
    if speaker_decision_inputs is None:
        raise ValueError("speaker-decisions revisions require trusted pre-UI input documents")
    from .voice_profile_guard import _require_ready, refresh_locks, require_valid_profiles
    for path in revisions:
        require_valid_profiles(path)
    paths = related_directories(directories or revisions)
    with ExitStack() as stack:
        covered = set()
        for path in paths:
            if path in covered:
                continue
            stack.enter_context(refresh_locks(path))
            covered.update(related_directories([path]))
            covered.update(continuation_directories(path))
        for path in covered:
            _require_ready(path, marker_only=True)
        return _revise_directories(revisions, directories=paths,
                                   expected_manifest_hashes=expected_manifest_hashes,
                                   progress=progress, scope=scope, sidecars=sidecars,
                                   speaker_decision_inputs=speaker_decision_inputs)


def _revise_directories(revisions, *, directories=None, expected_manifest_hashes=None, progress=print,
                        scope="attribution", sidecars=None, speaker_decision_inputs=None,
                        reserved_update_locks=None):
    """Stage and validate all changes, back up originals, and refresh descendants.

    This operation revises attribution and its audit/review state. Source and book
    locations stay fixed; cast and voice identities are fixed except for validated
    manual-decision appends. Voice stores, aliases, render caches, and existing
    audiobooks are never part of the write inventory.

    ``scope`` selects the change contract: ``"attribution"`` (default) permits
    speaker revisions only; ``"voice-profiles"`` instead permits exactly the
    allowlisted voice profile fields (description, traits, speechTraits), their
    provenance cast_fingerprint rebind, and the refresh audit record.
    ``"speaker-decisions"`` reconstructs manual choices from trusted pre-UI
    documents and their receipts, including exact append-only voice creation.
    ``sidecars`` optionally maps a directory to already validated JSON records
    keyed by plain existing filenames. Voice profiles accept only their audition
    binding; source edits accept that plus pronunciations.json,
    scoped-pronunciations.json, and performance-plan.json. Every file is backed
    up, replaced, and rolled back atomically with the revision.
    ``reserved_update_locks`` optionally maps exactly the update chain's
    directories to locks already owned through ``reserve_revision_scope``;
    publication reuses them instead of acquiring the locks again, and the
    render-activity re-checks trust that reservation's probe handshake.
    Attribution revisions never refuse live renderers (see
    ``reserve_revision_scope``); every other scope still probes render
    activity before staging and again immediately before publication.
    """
    from .cli import export_bundle
    sidecars = {} if sidecars is None else {
        Path(path).resolve(): {name: deepcopy(record) for name, record in files.items()}
        for path, files in sidecars.items()}
    if sidecars and scope not in _REVISION_SIDECARS:
        raise ValueError("Revision sidecars are supported only for voice-profile or source-edit bindings")
    revisions = {Path(path).resolve(): deepcopy(document) for path, document in revisions.items()}
    speaker_decision_inputs = {Path(path).resolve(): deepcopy(document)
                               for path, document in (speaker_decision_inputs or {}).items()}
    if scope == "speaker-decisions" and set(speaker_decision_inputs) != set(revisions):
        raise ValueError("Every speaker-decisions revision needs exactly one trusted pre-UI input document")
    paths = related_directories(directories or revisions)
    reserved = ({Path(path).resolve(): lock for path, lock in (reserved_update_locks or {}).items()}
                if reserved_update_locks is not None else {})
    if reserved and set(reserved) != set(paths):
        raise ValueError("Reserved revision locks must cover exactly the update chain")
    for path, lock in reserved.items():
        _require_owned_reservation(path, lock)
    if not set(revisions) <= set(paths):
        raise ValueError("Revision directories must be included in the update chain")
    if not set(sidecars) <= set(paths):
        raise ValueError("Sidecar directories must be included in the update chain")
    originals, manifests, names, original_hashes = {}, {}, {}, {}
    for path in paths:
        document, registry, bundle_names = _completed_directory(path)
        originals[path] = document, registry
        manifests[path] = _read(path / "manifest.json")
        names[path] = ["analysis.json", "series-cast.json", *bundle_names, "manifest.json"]
        for name in sidecars.get(path, {}):
            if name not in _REVISION_SIDECARS.get(scope, frozenset()) or name in names[path]:
                raise ValueError(f"Invalid revision sidecar name: {name}")
            target = path / name
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"Revision sidecars must already exist as regular files: {name}")
            names[path].append(name)
        original_hashes[path] = {name: _hash(path / name) for name in names[path]}
        if expected_manifest_hashes is not None:
            expected = expected_manifest_hashes.get(str(path), expected_manifest_hashes.get(path))
            if expected != original_hashes[path]["manifest.json"]:
                raise ValueError(f"Analysis changed while speaker decisions were being reviewed: {path}")
        candidate = revisions.get(path, document)
        if scope == "attribution":
            if (candidate["source"] != document["source"] or candidate["cast"] != document["cast"]
                    or candidate["provenance"].get("series") != document["provenance"].get("series")):
                raise ValueError("Speaker revisions must preserve source, cast, and series identities")
        elif scope == "source-edit":
            if path in revisions:
                if (candidate["source"].get("book") != document["source"].get("book")
                        or candidate["cast"] != document["cast"]
                        or candidate["provenance"].get("series") != document["provenance"].get("series")):
                    raise ValueError("Source-only updates must preserve book identity, cast, and series context")
            elif candidate != document:
                raise ValueError("Source-only updates may change only the selected book")
        elif scope == "voice-profiles":
            differences = profile_revision_differences(document, candidate)
            if differences:
                raise ValueError("Voice-profile revisions may change only cast voice description, traits, "
                                 "and speechTraits with their provenance bindings; rejected: "
                                 + "; ".join(differences[:6]))
        elif path in revisions:
            from .manual_speaker_revision import validate_manual_speaker_revision
            added_voices = validate_manual_speaker_revision(
                document, speaker_decision_inputs[path], candidate)
            sidecar = path / "audition-scripts.json"
            if added_voices and (sidecar.exists() or sidecar.is_symlink()):
                from .voice_profile_refresh import audition_sidecar_rebind
                sidecars.setdefault(path, {})[sidecar.name] = audition_sidecar_rebind(
                    sidecar, source_fingerprint=fingerprint(document["source"]),
                    cast_fingerprint_before=fingerprint(document["cast"]),
                    cast_fingerprint_after=fingerprint(candidate["cast"]),
                    voice_ids=[voice["id"] for voice in candidate["cast"]["voices"]])
                names[path].append(sidecar.name)
                original_hashes[path][sidecar.name] = _hash(sidecar)
        if readiness(candidate)["errors"]:
            raise ValueError("Cannot publish an invalid speaker revision")
        if scope != "attribution" and not reserved and (reason := _render_active(path)):
            raise ValueError(f"Rendering is active; finish it before revising analysis: {path} ({reason})")
    locks = []
    transaction = None
    installed = []
    try:
        from .renderer.run_locks import acquire_directory_lock
        for path in paths:
            if path in reserved:
                continue
            lock_path = path / ".analysis-update.lock"
            try:
                lock = acquire_directory_lock(lock_path)
            except FileExistsError as exc:
                raise ValueError(
                    f"Analysis revision is already active, or the lock cannot be proven stale: {lock_path}. "
                    f"A marked lock left by a crashed process is reclaimed automatically when you rerun.") from exc
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
            if scope in {"attribution", "speaker-decisions"}:
                _refresh_import_reviews(document, history, document["provenance"]["series"])
            document["readiness"] = readiness(document)
            snapshot = _snapshot(document)
            registry["books"][-1] = snapshot
            snapshots[(registry["series_id"], snapshot["source_fingerprint"])] = snapshot
            original_fingerprint = fingerprint(original["source"])
            if original_fingerprint != snapshot["source_fingerprint"]:
                snapshots[(registry["series_id"], original_fingerprint)] = snapshot
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
            for name, record in sidecars.get(path, {}).items():
                write_json(after / name, record)
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
                link.update(source_fingerprint=fingerprint(parent_document["source"]),
                            analysis_sha256=parent_manifest["analysis_directory"]["sha256"]["analysis.json"],
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
            if scope != "attribution" and not reserved and (reason := _render_active(path)):
                raise ValueError(f"Rendering started during revision: {path} ({reason})")
        # A reservation destroyed or replaced after entry cannot write: each
        # handle must still prove exclusive ownership immediately before the
        # first install, with no silent reacquisition of stale ownership.
        for path, lock in reserved.items():
            _require_owned_reservation(path, lock)
        journal["status"] = "applying"
        write_json(transaction / "revision.json", journal, overwrite=True)
        for entry in journal["books"]:
            path = Path(entry["directory"])
            for name in names[path]:
                # Record the attempted replacement first: an interruption may
                # arrive after os.replace succeeds but before _install returns.
                installed.append((entry, name))
                _install(Path(entry["prepared"]) / name, path / name)
            progress(f"Updated analysis and continuation metadata: {path}")
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
