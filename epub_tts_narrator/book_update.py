"""Update an analyzed book with chapters appended to a serially published EPUB.

Serial fiction (Patreon, Royal Road and similar) grows by new chapters.
Rerunning ``analyze NEW.epub -o BOOK`` on an existing book directory whose
saved chapters are an exact prefix of the new EPUB analyzes only the appended
chapters. That delta analysis is the ordinary pipeline with the book's own
cumulative series registry as continuity context, so returning characters
keep their identities and voices (three-way identity consensus when enabled),
genuinely new characters get new voices, and inherited profiles stay
reference-only during ensemble design.

The result is merged into the same book directory as one atomic, backed-up
revision. The saved book identity, every existing chapter, block, segment,
chapter script file, cast record and voice ID stay byte-identical, and
``provenance.chunk_identity_source_fingerprint`` keeps the first publication's
source identity so the renderer reuses all existing verified audio. Each
update is recorded in ``provenance.book_updates`` with the new characters it
introduced. Edited, removed or reordered earlier chapters are refused with
their location; such a source is a different book, not an appended update.

The update is resumable: progress is checkpointed in ``BOOK/.analysis-update``
before the first model request and after each stage, and model responses come
from the normal request cache.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import uuid

from .analysis_directory import _completed_directory, _read_json
from .analysis_revision import _hash, _install, _render_active, _snapshot, related_directories
from .contracts import readiness, validate_document
from .pipeline import fingerprint, write_json
from .series import validate_registry

UPDATE_WORKSPACE = ".analysis-update"
UPDATE_VERSION = 1
_REFUSAL = ("Only chapters appended after the saved ones can update a saved book. Analyze the new "
            "EPUB into a separate output directory instead; the saved book was not changed")


def _chapter_difference(old, new, old_blocks, new_blocks):
    if old["title"] != new["title"]:
        return f"its title is now {new['title']!r}"
    if old["source_href"] != new["source_href"]:
        return "it comes from a different EPUB file"
    if old["id"] != new["id"]:
        return "an earlier EPUB file was added or removed"
    if len(old["block_ids"]) != len(new["block_ids"]):
        return f"it has {len(new['block_ids'])} paragraphs instead of {len(old['block_ids'])}"
    for number, (old_id, new_id) in enumerate(zip(old["block_ids"], new["block_ids"]), 1):
        before, after = old_blocks.get(old_id), new_blocks.get(new_id)
        if old_id != new_id or before is None or after is None:
            return "an earlier chapter's paragraphs changed"
        if before["text"] != after["text"]:
            return f"paragraph {number} text changed"
        if before != after:
            return f"paragraph {number} formatting or location changed"
    if old != new:
        return "its chapter metadata changed"
    return None


def compare_sources(saved, current):
    """Classify ``current`` against a saved source: appended chapters, or a refusal.

    Block and chapter IDs are positional, so an appended chapter leaves every
    earlier ID unchanged while any earlier edit that adds or removes a
    paragraph would shift them. Only an exact prefix is an update.
    """
    if saved.get("extraction_policy") != current.get("extraction_policy"):
        raise ValueError("The new EPUB was extracted with different extraction settings "
                         "(extraction.include_auxiliary or the extractor version). " + _REFUSAL)
    old_blocks = {block["id"]: block for block in saved["blocks"]}
    new_blocks = {block["id"]: block for block in current["blocks"]}
    old_chapters, new_chapters = saved["chapters"], current["chapters"]
    for index, old in enumerate(old_chapters):
        label = f"Chapter {index + 1} ({old['title']!r})"
        if index >= len(new_chapters):
            raise ValueError(f"{label} is missing from the new EPUB. {_REFUSAL}")
        difference = _chapter_difference(old, new_chapters[index], old_blocks, new_blocks)
        if difference is not None:
            raise ValueError(f"{label} changed in the new EPUB: {difference}. {_REFUSAL}")
    if current["blocks"][:len(saved["blocks"])] != saved["blocks"]:
        raise ValueError(f"The saved paragraphs are not the start of the new EPUB. {_REFUSAL}")
    return {"unchanged_chapters": len(old_chapters),
            "appended_chapters": deepcopy(new_chapters[len(old_chapters):])}


def _source_like(saved, current, chapters):
    """A source with the saved book identity and ``chapters`` from ``current``."""
    chapter_ids = {chapter["id"] for chapter in chapters}
    values = {"book": deepcopy(saved["book"]), "chapters": deepcopy(chapters),
              "blocks": [deepcopy(block) for block in current["blocks"] if block["chapter_id"] in chapter_ids]}
    result = {}
    for key in [*saved, *(key for key in current if key not in saved)]:
        result[key] = values[key] if key in values else deepcopy(current.get(key, saved.get(key)))
    return result


def delta_source(saved, current, comparison):
    """Only the appended chapters, with their real IDs and the saved book identity."""
    return _source_like(saved, current, comparison["appended_chapters"])


def merged_source(saved, current):
    """Every chapter of ``current`` under the saved book identity (cover/tags/IDs stay)."""
    return _source_like(saved, current, current["chapters"])


def _role_map(old_cast, new_cast):
    """Map delta roles onto existing ones; anything without a counterpart is new."""
    old_roles = old_cast["roles"]
    old_ids = {role["id"] for role in old_roles}
    mapping = {}
    for role in new_cast["roles"]:
        if role["id"] in old_ids:
            mapping[role["id"]] = role["id"]
            continue
        matches = [old for old in old_roles if old["kind"] == role["kind"]
                   and old["character_id"] == role["character_id"]
                   and (old["character_id"] is not None or old["voice_id"] == role["voice_id"])]
        same_voice = [old for old in matches if old["voice_id"] == role["voice_id"]]
        chosen = (same_voice or matches)[:1]
        mapping[role["id"]] = chosen[0]["id"] if chosen else role["id"]
    return mapping


def _remap_issue(issue, roles):
    issue = deepcopy(issue)
    if issue.get("target") in roles:
        issue["target"] = roles[issue["target"]]
    return issue


def merge_update(saved, delta, source, comparison, *, epub_sha256=None):
    """Append ``delta`` (the analyzed new chapters) to the ``saved`` analysis.

    Existing records are kept exactly; the delta contributes only records the
    book does not have. Returns the merged document; its update record is
    ``provenance.book_updates[-1]``.
    """
    old_cast, new_cast = saved["cast"], delta["cast"]
    old_characters = {character["id"] for character in old_cast["characters"]}
    old_voices = {voice["id"] for voice in old_cast["voices"]}
    roles = _role_map(old_cast, new_cast)
    appended_characters = [deepcopy(character) for character in new_cast["characters"]
                           if character["id"] not in old_characters]
    appended_roles = [deepcopy(role) for role in new_cast["roles"] if roles[role["id"]] == role["id"]
                      and role["id"] not in {old["id"] for old in old_cast["roles"]}]
    used_voices = {item["voice_id"] for item in [*appended_characters, *appended_roles]}
    appended_voices = [deepcopy(voice) for voice in new_cast["voices"]
                       if voice["id"] not in old_voices and voice["id"] in used_voices]
    cast = deepcopy(old_cast)
    cast["characters"].extend(appended_characters)
    cast["roles"].extend(appended_roles)
    cast["voices"].extend(appended_voices)
    role_voice = {role["id"]: role["voice_id"] for role in cast["roles"]}

    segments = deepcopy(saved["segments"])
    for row in delta["segments"]:
        row = deepcopy(row)
        if row.get("role_id") is not None:
            row["role_id"] = roles[row["role_id"]]
            row["voice_id"] = role_voice[row["role_id"]]
        if "simultaneous_speakers" in row:
            row["simultaneous_speakers"] = [
                {"role_id": roles[speaker["role_id"]], "voice_id": role_voice[roles[speaker["role_id"]]]}
                for speaker in row["simultaneous_speakers"]]
        attribution = row.get("attribution")
        if isinstance(attribution, dict) and isinstance(attribution.get("candidates"), list):
            attribution["candidates"] = list(dict.fromkeys(
                roles.get(candidate, candidate) for candidate in attribution["candidates"]))
        segments.append(row)

    issues = deepcopy(saved["issues"])
    by_id = {issue.get("id"): issue for issue in issues}
    for issue in delta["issues"]:
        mapped = _remap_issue(issue, roles)
        existing = by_id.get(mapped.get("id"))
        if existing == mapped:
            continue
        if existing is not None:
            # Issue IDs hash their content; a distinct new issue keeps its own.
            mapped["id"] = f"{mapped['id']}:update-{len(saved['provenance'].get('book_updates', [])) + 1}"
        by_id[mapped.get("id")] = mapped
        issues.append(mapped)

    def status(character):
        return character.get("series_origin", {}).get("status")

    provenance = deepcopy(saved["provenance"])
    previous = fingerprint(saved["source"])
    provenance.setdefault("chunk_identity_source_fingerprint", previous)
    provenance["source_fingerprint"] = fingerprint(source)
    if "bracket_delivery" in provenance:
        # Bracket groups are located by block and offset; the marker counts
        # and warns about the whole merged source.
        from .performance import bracket_ranges
        ranges, warnings = bracket_ranges(source)
        marker = provenance["bracket_delivery"]
        marker["groups"] = len({span["group_id"] for values in ranges.values() for span in values})
        marker["warnings"] = warnings
        drops = delta["provenance"].get("bracket_delivery", {}).get("inline_projection_drops", [])
        if drops:
            marker["inline_projection_drops"] = [*marker.get("inline_projection_drops", []), *deepcopy(drops)]
    update = {
        "version": UPDATE_VERSION,
        "previous_source_fingerprint": previous, "source_fingerprint": fingerprint(source),
        "delta_source_fingerprint": fingerprint(delta["source"]), "epub_sha256": epub_sha256,
        "added_chapters": [{"id": chapter["id"], "title": chapter["title"]}
                           for chapter in comparison["appended_chapters"]],
        "new_characters": [character["id"] for character in appended_characters if status(character) != "reused"],
        "returning_series_characters": [character["id"] for character in appended_characters
                                        if status(character) == "reused"],
        "existing_characters": [character["id"] for character in new_cast["characters"]
                                if character["id"] in old_characters],
        "added_roles": [role["id"] for role in appended_roles],
        "added_voices": [voice["id"] for voice in appended_voices],
        "role_map": {key: value for key, value in roles.items() if key != value},
        "analysis": deepcopy(delta["provenance"]),
    }
    provenance.setdefault("book_updates", []).append(update)
    document = deepcopy(saved)
    document.update(source=deepcopy(source), cast=cast, segments=segments, issues=issues,
                    provenance=provenance)
    document.pop("readiness", None)
    errors = validate_document(document)
    if errors:
        raise ValueError("The appended chapters could not be merged into the saved book: " + "; ".join(errors[:8]))
    document["readiness"] = readiness(document)
    return document


def keep_existing_auditions(document, saved):
    """Existing voices keep their saved audition scripts and samples exactly.

    New chapters can contain more speech for a returning voice; its prepared
    audition, and so any reference recording made from it, must not change.
    """
    before = saved["provenance"].get("voice_samples")
    after = document["provenance"].get("voice_samples")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return document
    result = deepcopy(document)
    samples = result["provenance"]["voice_samples"]
    old_voices = {voice["id"] for voice in saved["cast"]["voices"]}
    for key in ("samples", "scripts"):
        if isinstance(before.get(key), dict):
            merged = {voice: value for voice, value in samples.get(key, {}).items() if voice not in old_voices}
            merged.update(deepcopy(before[key]))
            samples[key] = merged
    pending = {voice: value for voice, value in samples.get("pending", {}).items() if voice not in old_voices}
    pending.update(deepcopy(before.get("pending", {})))
    if pending:
        samples["pending"] = pending
    else:
        samples.pop("pending", None)
    if "replaced_voice_ids" in before:
        samples["replaced_voice_ids"] = deepcopy(before["replaced_voice_ids"])
    return result


def _rebind_sidecars(directory, saved, document):
    """Rebind book-local sidecars that bind the source (and cast) fingerprints.

    Their entries reference existing blocks, which an append leaves unchanged;
    each entry is revalidated against the new source.
    """
    old_source, new_source = fingerprint(saved["source"]), fingerprint(document["source"])
    old_cast, new_cast = fingerprint(saved["cast"]), fingerprint(document["cast"])
    records = {}
    path = directory / "audition-scripts.json"
    if path.exists() or path.is_symlink():
        from .voice_profile_refresh import audition_sidecar_rebind
        record = audition_sidecar_rebind(path, source_fingerprint=old_source, cast_fingerprint_before=old_cast,
                                         cast_fingerprint_after=new_cast,
                                         voice_ids=[voice["id"] for voice in document["cast"]["voices"]])
        records[path.name] = dict(record, source_fingerprint=new_source)
    from .pronunciations import FILENAME as PRONUNCIATIONS, validate_entry
    path = directory / PRONUNCIATIONS
    if path.exists() or path.is_symlink():
        from .pronunciations import _load_local
        record = deepcopy(_load_local(directory, saved["source"]))
        record["source_fingerprint"] = new_source
        if "discovery" in record:
            record["discovery"]["source_fingerprint"] = new_source
        for entry in record["entries"]:
            validate_entry(entry, document["source"])
        records[path.name] = record
    from .scoped_pronunciations import FILENAME as SCOPED, local as scoped_local
    path = directory / SCOPED
    if path.exists() or path.is_symlink():
        record = deepcopy(scoped_local(directory, source=saved["source"]))
        record["source_fingerprint"] = new_source
        records[path.name] = record
    return records


def publish_update(directory, document, *, expected_manifest_sha256, progress=print):
    """Install an appended-chapter revision atomically, with backups and a journal.

    Existing chapter scripts must come out byte-identical; new scripts are
    added. Any failure restores every replaced file and removes added ones.
    """
    from .cli import export_bundle
    from .voice_profile_guard import refresh_locks, require_valid_profiles
    directory = Path(directory).resolve()
    require_valid_profiles(directory)
    with ExitStack() as stack:
        stack.enter_context(refresh_locks(directory))
        original, registry, bundle_names = _completed_directory(directory)
        manifest = _read_json(directory / "manifest.json")
        if _hash(directory / "manifest.json") != expected_manifest_sha256:
            raise ValueError("The saved book changed while its new chapters were analyzed; rerun the update")
        if _render_active(directory):
            raise ValueError(f"Rendering is active; finish it before updating the book: {directory}")
        lock = directory / ".analysis-update.lock"
        try:
            lock.mkdir()
        except FileExistsError as exc:
            raise ValueError(f"Analysis revision already active or interrupted; inspect {lock}") from exc
        stack.callback(lock.rmdir)
        sidecars = _rebind_sidecars(directory, original, document)
        transaction = directory / ".analysis-revisions" / uuid.uuid4().hex
        before, after = transaction / "book-0001" / "before", transaction / "book-0001" / "after"
        before.mkdir(parents=True)
        document = deepcopy(document)
        document["readiness"] = readiness(document)
        registry = deepcopy(registry)
        registry["books"][-1] = _snapshot(document)
        validate_registry(registry)
        export_bundle(document, after, allow_review=True)
        write_json(after / "analysis.json", document)
        write_json(after / "series-cast.json", registry)
        for name, record in sidecars.items():
            write_json(after / name, record)
        new_manifest = _read_json(after / "manifest.json")
        new_manifest["analysis_directory"] = {"version": 1, "sha256": {
            name: _hash(after / name) for name in ("analysis.json", "series-cast.json")}}
        new_manifest["continuation"] = deepcopy(manifest.get("continuation"))
        write_json(after / "manifest.json", new_manifest, overwrite=True)
        _, _, new_names = _completed_directory(after)
        old_scripts = [row["path"] for row in manifest["files"]["scripts"]]
        new_scripts = [row["path"] for row in new_manifest["files"]["scripts"]]
        if new_scripts[:len(old_scripts)] != old_scripts or any(
                new_manifest["sha256"][name] != manifest["sha256"][name] for name in old_scripts):
            raise ValueError("Appending chapters must keep every existing chapter script unchanged")
        replaced = [name for name in ["analysis.json", "series-cast.json", *bundle_names, *sidecars]
                    if name != "manifest.json" and (directory / name).exists()
                    and _hash(directory / name) != _hash(after / name)]
        added = [name for name in new_names if not (directory / name).exists()]
        original_hashes = {name: _hash(directory / name) for name in [*replaced, "manifest.json"]}
        for name in [*replaced, "manifest.json"]:
            (before / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, before / name)
        journal = {"version": 1, "kind": "appended-chapters", "status": "prepared",
                   "books": [{"directory": str(directory), "backup": str(before), "prepared": str(after),
                              "replaced": replaced, "added": added, "before_sha256": original_hashes}]}
        write_json(transaction / "revision.json", journal)
        installed, created = [], []
        try:
            if any(_hash(directory / name) != checksum for name, checksum in original_hashes.items()):
                raise ValueError(f"The saved book changed before publication: {directory}")
            if _render_active(directory):
                raise ValueError(f"Rendering started during the update: {directory}")
            journal["status"] = "applying"
            write_json(transaction / "revision.json", journal, overwrite=True)
            for name in added:
                created.append(name)
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                _install(after / name, directory / name)
            for name in [*replaced, "manifest.json"]:
                installed.append(name)
                _install(after / name, directory / name)
            _completed_directory(directory)
            journal["status"] = "complete"
            write_json(transaction / "revision.json", journal, overwrite=True)
        except BaseException:
            errors = []
            for name in reversed(installed):
                try:
                    _install(before / name, directory / name)
                except BaseException as exc:
                    errors.append(f"{name}: {exc}")
            for name in reversed(created):
                try:
                    if (directory / name).exists() and _hash(directory / name) == _hash(after / name):
                        (directory / name).unlink()
                except BaseException as exc:
                    errors.append(f"{name}: {exc}")
            journal["status"] = "rollback_incomplete" if errors else "rolled_back"
            journal["rollback_errors"] = errors
            write_json(transaction / "revision.json", journal, overwrite=True)
            if errors:
                raise ValueError(f"Book update rollback needs attention; originals and journal retained in {transaction}")
            raise
        progress(f"Original analysis files and revision journal: {transaction}")
        return transaction


def describe_update(document):
    """Human-readable lines for the latest update, naming new characters."""
    update = document["provenance"]["book_updates"][-1]
    names = {character["id"]: character["name"] for character in document["cast"]["characters"]}
    chapters = update["added_chapters"]
    lines = [f"Added {len(chapters)} chapter{'s' if len(chapters) != 1 else ''}: "
             + ", ".join(chapter["title"] for chapter in chapters)]
    for label, key in (("New character", "new_characters"),
                       ("Returning from an earlier book", "returning_series_characters"),
                       ("Returning", "existing_characters")):
        if update[key]:
            shown = [names.get(identity, identity) for identity in update[key]]
            more = f" and {len(shown) - 12} more" if key == "existing_characters" and len(shown) > 12 else ""
            shown = shown[:12] if more else shown
            lines.append(f"{label}: " + ", ".join(shown) + more)
    if not update["new_characters"] and not update["returning_series_characters"]:
        lines.append("No new characters.")
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    fresh = [identity for identity in update["added_voices"]
             if voices.get(identity, {}).get("series_origin", {}).get("kind") != "series_casting_reuse"]
    if fresh:
        lines.append(f"{len(fresh)} new voice(s) need casting before their chapters render.")
    return lines


def ensure_updatable(directory):
    """Refuse an update the book's dependents or a live render could not survive."""
    directory = Path(directory).resolve()
    if _render_active(directory):
        raise ValueError(f"Rendering is active; finish it before updating the book: {directory}")
    related = related_directories([directory])
    if related != [directory]:
        descendants = ", ".join(str(path) for path in related if path != directory)
        raise ValueError("A later book continues from this one, so its chapters cannot be extended in place "
                         f"(continued by {descendants}). Analyze the new EPUB into a separate output directory")


def update_book(directory, source, *, config, input_path=None, client_factory, speaker_judge=None,
                announce_cache=lambda path: Path(path).resolve(), progress=print, log=print):
    """Analyze and publish the chapters appended to ``directory``'s book.

    Returns the published document, or ``None`` when the source has no new
    chapters. ``client_factory`` builds the analysis client from
    ``config["provider"]`` only after every local check has passed.
    """
    from .analysis_progress import (STAGES, analysis_options_fingerprint, create_pending,
                                    pending_identity, resume_document)
    from .audition_scripts import prepare_document_samples
    from .pipeline import CachedRequests, analyze
    from .speaker_decisions import prompt_speaker_decisions
    directory = Path(directory).resolve()
    original, registry, _ = _completed_directory(directory)
    comparison = compare_sources(original["source"], source)
    if not comparison["appended_chapters"]:
        log("No new chapters: the EPUB's chapters match the saved book. The saved analysis is unchanged.")
        return None
    ensure_updatable(directory)
    manifest_hash = _hash(directory / "manifest.json")
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
    cache = Path(settings["cache_dir"] or directory.parent / ".epub-audiobook-cache").resolve()
    if cache == directory or directory in cache.parents:
        raise ValueError("The cache directory must be outside the book directory")
    if (directory / "name-script.json").exists():
        log("Note: this book has a legacy name-script.json. It covers only the earlier chapters; "
            "name-scoped pronunciation choices that depend on it are blocked until it is prepared again.")
    epub = Path(input_path) if input_path is not None and Path(input_path).suffix.lower() == ".epub" else None
    digest = epub_digest(epub) if epub is not None else None
    merged = merged_source(original["source"], source)
    delta = delta_source(original["source"], source, comparison)
    client = client_factory(**config["provider"])
    log(f"Updating {directory} with {len(comparison['appended_chapters'])} new chapter(s) using "
        f"{client.model} at {client.base_url}; only the new chapters' text will be sent to this endpoint.")
    model_cache = announce_cache(cache)
    identity = pending_identity(merged, client.identity, analysis_options_fingerprint(
        settings, config["performance_policy"],
        fingerprint({"book_update": UPDATE_VERSION, "base_manifest_sha256": manifest_hash})))
    workspace = update_workspace(directory)
    resumed = resume_document(workspace, identity)
    document, resume_index = None, 0
    if resumed is not None:
        document, next_stage = resumed
        resume_index = len(STAGES) if next_stage is None else STAGES.index(next_stage)
        if resume_index:
            log("Resuming the interrupted book update from its last completed stage.")
    pending = create_pending(workspace, identity)
    requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    workers = min(5, settings["attribution_workers"])
    try:
        if resume_index <= 0:
            pending.begin_stage("analyzed")
            analyzed = analyze(
                delta, client, model_cache, chunk_chars=settings["chunk_chars"],
                repair_attempts=settings["repair_attempts"], max_request_chars=settings["max_request_chars"],
                whole_book_discovery=settings["whole_book_discovery"],
                attribution_workers=settings["attribution_workers"],
                identity_consensus=settings["identity_consensus"],
                attribution_review=settings["attribution_review"],
                attribution_review_chunk_chars=settings["attribution_review_chunk_chars"],
                attribution_review_notes=settings["attribution_review_notes"],
                cast_review_notes=settings["cast_review_notes"],
                cast_reconciliation_rounds=settings["cast_reconciliation_rounds"],
                cast_reconciliation_notes=settings["cast_reconciliation_notes"],
                # The book's own history: returning characters keep their voices.
                series_registry=deepcopy(registry),
                policy=config["performance_policy"], progress=progress)
            document = merge_update(original, analyzed, merged, comparison, epub_sha256=digest)
            pending.complete_stage("analyzed", document)
        if resume_index <= 1:
            pending.begin_stage("speech-normalized-1")
            pending.complete_stage("speech-normalized-1", document)
        if resume_index <= 2:
            pending.begin_stage("samples-prepared-1")
            document = keep_existing_auditions(prepare_document_samples(
                document, requests, workers=workers, allow_incomplete=True, progress=progress), original)
            pending.complete_stage("samples-prepared-1", document)
        if resume_index <= 3:
            pending.begin_stage("speakers-decided")
            document = prompt_speaker_decisions(document, book_path=directory, jev_client=speaker_judge,
                                                decisions_cache=model_cache)
            pending.complete_stage("speakers-decided", document)
        if resume_index <= 4:
            pending.begin_stage("speech-normalized-2")
            pending.complete_stage("speech-normalized-2", document)
        if resume_index <= 5:
            pending.begin_stage("samples-prepared-2")
            document = keep_existing_auditions(prepare_document_samples(
                document, None, workers=workers, allow_incomplete=True, progress=progress), original)
            pending.complete_stage("samples-prepared-2", document)
        publish_update(directory, document, expected_manifest_sha256=manifest_hash, progress=log)
        if epub is not None:
            from .source_epub import remember_source_epub
            remember_source_epub(directory, epub, document["source"]["book"]["id"], epub_sha256=digest)
    except BaseException as error:
        pending.mark_failed(error, interrupted=isinstance(error, KeyboardInterrupt))
        pending.close()
        raise
    pending.close()
    clear_workspace(directory)
    for line in describe_update(document):
        log(line)
    log("Rendering reuses every existing chapter's audio; run audiobook-render render on this book "
        "to cast any new voices and render the new chapters.")
    return document


def plan_update(directory, source, config):
    """The ``plan`` estimate for appending ``source``'s new chapters to a saved book."""
    from .analysis_plan import plan_analysis
    directory = Path(directory).resolve()
    original, registry, _ = _completed_directory(directory)
    comparison = compare_sources(original["source"], source)
    update = {"unchanged_chapters": comparison["unchanged_chapters"],
              "added_chapters": [{"id": chapter["id"], "title": chapter["title"]}
                                 for chapter in comparison["appended_chapters"]]}
    if not comparison["appended_chapters"]:
        return {"output_directory": str(directory), "book_update": update,
                "note": "No new chapters; analyze would make no model requests."}
    planned = plan_analysis(delta_source(original["source"], source, comparison), config["analysis"],
                            config["performance_policy"], registry)
    return {**planned, "output_directory": str(directory), "continue_from": None, "book_update": update}


def latest_epub_sha256(document_or_manifest):
    """The SHA-256 of the EPUB that supplied the latest appended chapters, if any."""
    updates = document_or_manifest.get("provenance", {}).get("book_updates") or []
    value = updates[-1].get("epub_sha256") if updates and isinstance(updates[-1], dict) else None
    return value if isinstance(value, str) else None


def accepted_epub_digests(document_or_manifest):
    """EPUB SHA-256 values that belong to this book: the original and the latest update."""
    book = document_or_manifest.get("source", document_or_manifest).get("book", {})
    accepted = {book.get("id")} if isinstance(book.get("id"), str) else set()
    latest = latest_epub_sha256(document_or_manifest)
    if latest is not None:
        accepted.add(latest)
    return accepted


def epub_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def update_workspace(directory):
    return Path(directory) / UPDATE_WORKSPACE


def clear_workspace(directory):
    from .analysis_progress import discard_pending
    workspace = update_workspace(directory)
    if workspace.is_dir() and not workspace.is_symlink():
        discard_pending(workspace)
        os.rmdir(workspace)
