"""Publish complete per-book analyses and carry series history forward read-only."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from .contracts import readiness, review_items, validate_document
from .pipeline import fingerprint, write_json
from .series import load_registry, save_registry, validate_registry


def default_output(input_path):
    return Path.cwd() / f"{Path(input_path).stem}.audiobook"


def validate_output(output):
    """Check a new destination without creating it or changing existing files."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output}; choose another directory")
    resolved = output.resolve()
    for parent in resolved.parents:
        if parent.exists():
            if not parent.is_dir():
                raise ValueError(f"Output parent is not a directory: {parent}")
            break
    return resolved


def fresh_registry(output):
    """Use the final path for stable IDs even when a run is interrupted."""
    return load_registry(validate_output(output) / "series-cast.json")


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _regular_file(directory, name):
    path = directory / name
    for component in (path, *path.parents):
        if component == directory:
            break
        if component.is_symlink():
            raise ValueError(f"Analysis directory must not contain symlinked files: {name}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Analysis directory needs a regular file: {name}")
    return path


def _read_json(path):
    def reject(value):
        raise ValueError("Non-finite numbers are not valid JSON")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _bundle_files(manifest):
    if not isinstance(manifest, dict) or manifest.get("format") != "epub-audiobook-bundle" or manifest.get("version") != 2:
        raise ValueError("Continuation requires a completed version-2 analysis directory")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"source", "cast", "review", "scripts"}:
        raise ValueError("Invalid analysis directory file inventory")
    if any(files[key] != f"{key}.json" for key in ("source", "cast", "review")):
        raise ValueError("Invalid analysis directory file inventory")
    scripts = files["scripts"]
    if not isinstance(scripts, list):
        raise ValueError("Invalid analysis directory chapter inventory")
    names = ["source.json", "cast.json", "review.json"]
    for index, chapter in enumerate(scripts, 1):
        expected = f"scripts/chapter-{index:04d}.jsonl"
        if not isinstance(chapter, dict) or chapter.get("path") != expected:
            raise ValueError("Invalid analysis directory chapter filename")
        names.append(expected)
    checksums = manifest.get("sha256")
    if not isinstance(checksums, dict) or set(checksums) != set(names):
        raise ValueError("Analysis directory checksum inventory is incomplete")
    return names


def _continuation_entry(manifest):
    """Missing links are supported for outputs created before directory ancestry."""
    link = manifest.get("continuation")
    if link is None:
        return None
    fields = {"version", "path", "series_id", "source_fingerprint", "analysis_sha256", "series_cast_sha256"}
    if (not isinstance(link, dict) or set(link) != fields or link.get("version") != 1
            or not isinstance(link.get("path"), str) or not link["path"].strip()
            or Path(link["path"]).is_absolute() or "\x00" in link["path"]
            or not isinstance(link.get("series_id"), str) or not link["series_id"]):
        raise ValueError("Invalid analysis directory continuation link")
    for field in ("source_fingerprint", "analysis_sha256", "series_cast_sha256"):
        value = link.get(field)
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("Invalid analysis directory continuation identity")
    return link


def _readiness_matches(saved, current):
    """Accept the former advisory-review status without rewriting prior books."""
    legacy = dict(current, ready=not current["errors"] and current["review_count"] == 0)
    return saved == current or saved == legacy


def _completed_directory(directory):
    """Validate the bundle and its matching cumulative series snapshot."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Previous analysis directory does not exist: {directory}")
    try:
        manifest = _read_json(_regular_file(directory, "manifest.json"))
        names = _bundle_files(manifest)
        _continuation_entry(manifest)
        for name in names:
            path = _regular_file(directory, name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"Analysis directory checksum mismatch: {name}")
        metadata = manifest.get("analysis_directory")
        if (not isinstance(metadata, dict) or metadata.get("version") != 1
                or not isinstance(metadata.get("sha256"), dict)
                or set(metadata["sha256"]) != {"analysis.json", "series-cast.json"}):
            raise ValueError("Analysis directory continuation metadata is missing or invalid")
        for name, checksum in metadata["sha256"].items():
            path = _regular_file(directory, name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Analysis directory checksum mismatch: {name}")
        document = _read_json(_regular_file(directory, "analysis.json"))
        errors = validate_document(document)
        if errors:
            raise ValueError("Invalid prior analysis: " + "; ".join(errors[:3]))
        registry = validate_registry(_read_json(_regular_file(directory, "series-cast.json")))
        source_hash = fingerprint(document["source"])
        state = readiness(document)
        if (manifest.get("source_fingerprint") != source_hash
                or manifest.get("book") != document["source"]["book"]
                or manifest.get("chapters") != document["source"]["chapters"]
                or manifest.get("provenance") != document["provenance"]
                or not _readiness_matches(manifest.get("readiness"), state)
                or _read_json(directory / "source.json") != document["source"]
                or _read_json(directory / "cast.json") != document["cast"]
                or _read_json(directory / "review.json") != review_items(document)):
            raise ValueError("Prior bundle and analysis disagree; use a complete analysis output")
        cursor = 0
        for chapter in manifest["files"]["scripts"]:
            start = cursor
            with (directory / chapter["path"]).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if (cursor >= len(document["segments"]) or not isinstance(row, dict)
                            or row.get("sequence") != cursor
                            or any(row.get(key) != value for key, value in document["segments"][cursor].items())):
                        raise ValueError("Prior chapter scripts and analysis disagree")
                    cursor += 1
            if (chapter.get("sequence_start") != start or chapter.get("sequence_end") != cursor
                    or chapter.get("segment_count") != cursor - start):
                raise ValueError("Prior chapter sequence inventory is inconsistent")
        if cursor != len(document["segments"]):
            raise ValueError("Prior chapter scripts do not cover the analysis")
        series = document["provenance"].get("series", {})
        if series.get("series_id") != registry["series_id"] or not registry["books"]:
            raise ValueError("Prior analysis and series registry have different identities")
        latest = registry["books"][-1]
        expected = {"source_fingerprint": source_hash, "book": document["source"]["book"],
                    "cast": document["cast"], "series_provenance": series,
                    "issues": document["issues"]}
        if (any(latest.get(key) != value for key, value in expected.items())
                or not _readiness_matches(latest.get("readiness"), state)):
            raise ValueError("Prior analysis does not match the latest series snapshot")
        return document, registry, names
    except FileNotFoundError as exc:
        raise ValueError("Previous analysis directory is incomplete; manifest.json, analysis.json, "
                         "series-cast.json, and all bundle files are required") from exc
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("Invalid analysis directory metadata") from exc


def _continuation_chain(directory, document, registry):
    """Validate each ancestor once, preserving nearest-first lookup order."""
    current = Path(directory).resolve()
    visited = {current}
    ancestors = []
    while True:
        manifest = _read_json(_regular_file(current, "manifest.json"))
        link = _continuation_entry(manifest)
        if link is None:
            return ancestors
        parent = (current / link["path"]).resolve()
        if parent in visited:
            raise ValueError(f"Cycle in analysis directory continuation chain: {parent}")
        visited.add(parent)
        try:
            parent_document, parent_registry, _ = _completed_directory(parent)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load continuation parent {parent}: {exc}") from exc
        parent_metadata = _read_json(parent / "manifest.json")["analysis_directory"]["sha256"]
        if (parent_registry["series_id"] != link["series_id"]
                or fingerprint(parent_document["source"]) != link["source_fingerprint"]
                or parent_metadata["analysis.json"] != link["analysis_sha256"]
                or parent_metadata["series-cast.json"] != link["series_cast_sha256"]):
            raise ValueError(f"Continuation parent identity does not match the recorded book: {parent}")
        if (registry["series_id"] != parent_registry["series_id"]
                or registry["books"][:-1] != parent_registry["books"]):
            raise ValueError(f"Continuation parent history does not match this book: {parent}")
        ancestors.append(parent)
        current, document, registry = parent, parent_document, parent_registry


def continuation_directories(directory):
    """Return validated parent book directories, nearest first, without writing.

    Paths are resolved relative to the book containing each link, so moving a
    series directory as a unit keeps its ancestry intact. Voice assets are
    deliberately excluded from validation: they can be created after analysis.
    Older standalone bundles without a continuation field have no ancestors.
    """
    directory = Path(directory).resolve()
    try:
        manifest = _read_json(_regular_file(directory, "manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid analysis directory manifest")
        if _continuation_entry(manifest) is None:
            return []
        document, registry, _ = _completed_directory(directory)
        return _continuation_chain(directory, document, registry)
    except OSError as exc:
        raise ValueError(f"Cannot read analysis directory continuation chain at {directory}: {exc}") from exc


def continuation_registry(previous_dir, source):
    """Read a complete previous output without updating any previous artifacts."""
    document, registry, _ = _completed_directory(previous_dir)
    _continuation_chain(previous_dir, document, registry)
    if any(book["source_fingerprint"] == fingerprint(source) for book in registry["books"]):
        raise ValueError("This book is already in the continuation history; select the preceding book's output "
                         "or start without --continue-from")
    return deepcopy(registry)


def _rollback(output, owned_directories, expected_files):
    """Remove only artifacts this publication still owns, preserving outsiders."""
    try:
        if _identity(output) != owned_directories.get(output):
            return
    except OSError:
        return
    for path, identity in reversed(list(expected_files.items())):
        try:
            if _identity(path.parent) != owned_directories.get(path.parent):
                continue
            if _identity(path) == identity:
                path.unlink()
        except OSError:
            pass
    for path, identity in reversed(list(owned_directories.items())):
        try:
            if _identity(path) == identity:
                path.rmdir()
        except OSError:
            # A concurrently added file or replaced directory belongs to its writer.
            pass


def publish_directory(document, registry, output, exporter, *, continue_from=None):
    """Commit an analysis, cumulative cast, and chapter bundle without clobbering."""
    output = validate_output(output)
    continuation = None
    if continue_from is not None:
        previous = Path(continue_from).resolve()
        prior_document, prior_registry, _ = _completed_directory(previous)
        _continuation_chain(previous, prior_document, prior_registry)
        if registry != prior_registry:
            raise ValueError("Continuation parent changed during analysis or has a different history")
        metadata = _read_json(previous / "manifest.json")["analysis_directory"]["sha256"]
        continuation = {"version": 1, "path": os.path.relpath(previous, output),
                        "series_id": prior_registry["series_id"],
                        "source_fingerprint": fingerprint(prior_document["source"]),
                        "analysis_sha256": metadata["analysis.json"],
                        "series_cast_sha256": metadata["series-cast.json"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".audiobook-analysis-", dir=output.parent))
    owned_directories, expected_files = {}, {}
    try:
        bundle = staging / "bundle"
        state = exporter(document, bundle, allow_review=True)
        write_json(bundle / "analysis.json", document)
        # save_registry checks its input snapshot before publication. Seeding the
        # temporary copy preserves that check and never changes the prior book.
        write_json(bundle / "series-cast.json", registry)
        save_registry(bundle / "series-cast.json", registry, document)
        manifest = _read_json(bundle / "manifest.json")
        manifest["analysis_directory"] = {"version": 1, "sha256": {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in ("analysis.json", "series-cast.json")}}
        manifest["continuation"] = continuation
        write_json(bundle / "manifest.json", manifest, overwrite=True)
        _, _, bundle_names = _completed_directory(bundle)
        names = ["analysis.json", "series-cast.json", *bundle_names, "manifest.json"]
        expected_files = {output / name: _identity(bundle / name) for name in names}
        output.mkdir()
        owned_directories[output] = _identity(output)
        scripts = output / "scripts"
        scripts.mkdir()
        owned_directories[scripts] = _identity(scripts)
        for name in names:
            os.link(bundle / name, output / name)
        return state
    except BaseException:
        _rollback(output, owned_directories, expected_files)
        raise
    finally:
        shutil.rmtree(staging)
