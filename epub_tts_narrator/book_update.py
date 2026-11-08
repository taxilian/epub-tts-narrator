"""Update an analyzed book from a newer download of a serially published EPUB.

Serial fiction (Patreon, Royal Road and similar) grows by new chapters, and
authors revise earlier ones. Rerunning ``analyze NEW.epub -o BOOK`` on an
existing book directory aligns the new EPUB with the saved source paragraph by
paragraph (by content, not position), then analyzes only appended chapters and
edited or inserted paragraphs, each edit with a little unchanged context. That
delta analysis is the ordinary pipeline with the book's own cumulative series
registry as continuity context, so returning characters keep their identities
and voices (three-way identity consensus when enabled), genuinely new
characters get new voices, and inherited profiles stay reference-only during
ensemble design.

The result is merged into the same book directory as one atomic, backed-up
revision. The saved book identity, every unchanged paragraph's block and
segments (speaker choices included), every unchanged chapter's script rows and
every existing cast record and voice ID are kept, and
``provenance.chunk_identity_source_fingerprint`` keeps the first publication's
source identity so the renderer reuses all unchanged verified audio. Each
update is recorded in ``provenance.book_updates`` with the chapters it added
and edited and the new characters it introduced. Removed, inserted or
reordered chapters are refused with their location.

The update is resumable: progress is checkpointed in ``BOOK/.analysis-update``
before the first model request and after each stage, and model responses come
from the normal request cache.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .analysis_directory import _completed_directory, _read_json
from .analysis_revision import _hash, _install, _render_active, _snapshot, related_directories
from .contracts import readiness, validate_document
from .pipeline import fingerprint, write_json
from .series import validate_registry

UPDATE_WORKSPACE = ".analysis-update"
UPDATE_VERSION = 2
# Unchanged paragraphs on each side of an edit that the model reads again as
# context. Their saved analysis is kept; disagreement becomes a review issue.
CONTEXT_BLOCKS = 2
_REFUSAL = ("A saved book can be updated with appended chapters and with edits inside its existing "
            "chapters, but not with chapters removed, inserted or reordered. Analyze the new EPUB into a "
            "separate output directory instead; the saved book was not changed")


def _block_key(block):
    """What a paragraph says and how it is marked up; its EPUB location is not content."""
    return json.dumps({"text": block["text"], "tag": block.get("source", {}).get("tag"),
                       "inline_spans": block.get("inline_spans", []),
                       "break_before": block.get("break_before")}, sort_keys=True, ensure_ascii=False)


def _number(identifier, prefix):
    match = re.fullmatch(re.escape(prefix) + r"(\d+)", identifier)
    return int(match.group(1)) if match else 0


def _matched(old_keys, new_keys):
    matcher = SequenceMatcher(None, old_keys, new_keys, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def compare_sources(saved, current):
    """Align ``current`` with a saved source by paragraph content.

    Every paragraph is keyed by its text and markup (not its position or EPUB
    location), and each chapter's saved and new paragraphs are aligned with
    ``difflib``. Unchanged paragraphs keep their saved block IDs, so their
    segments, speaker choices and rendered audio stay attached; edited,
    inserted and appended paragraphs get new IDs after the highest saved one,
    and saved paragraphs no longer present are removed. Chapters keep their
    saved IDs by position. A saved chapter with no paragraph in common with its
    position, or that matches a neighbouring position better, means chapters
    were removed, inserted or reordered, which is refused.

    Returns the merged ``source`` (saved book identity, new content, stable
    IDs) with a description of what changed.
    """
    if saved.get("extraction_policy") != current.get("extraction_policy"):
        raise ValueError("The new EPUB was extracted with different extraction settings "
                         "(extraction.include_auxiliary or the extractor version). " + _REFUSAL)
    old_blocks = {block["id"]: block for block in saved["blocks"]}
    new_blocks = {block["id"]: block for block in current["blocks"]}
    old_chapters, new_chapters = saved["chapters"], current["chapters"]
    if len(new_chapters) < len(old_chapters):
        missing = old_chapters[len(new_chapters)]
        raise ValueError(f"Chapter {len(new_chapters) + 1} ({missing['title']!r}) is missing from the new EPUB. "
                         + _REFUSAL)

    def keys(chapter, blocks):
        return [_block_key(blocks[identifier]) for identifier in chapter["block_ids"]]

    old_keys = [keys(chapter, old_blocks) for chapter in old_chapters]
    new_keys = [keys(chapter, new_blocks) for chapter in new_chapters]
    next_block = max((_number(identifier, "b") for identifier in old_blocks), default=0)
    next_chapter = max((_number(chapter["id"], "ch") for chapter in old_chapters), default=0)
    chapters, blocks, edited, block_map, changed = [], [], [], {}, set()
    for index, chapter in enumerate(new_chapters):
        record = deepcopy(chapter)
        old = old_chapters[index] if index < len(old_chapters) else None
        if old is not None:
            label = f"Chapter {index + 1} ({old['title']!r})"
            own = _matched(old_keys[index], new_keys[index])
            if old_keys[index] and new_keys[index] and own == 0:
                raise ValueError(f"{label} has no paragraph in common with chapter {index + 1} of the new EPUB "
                                 f"({chapter['title']!r}). " + _REFUSAL)
            for other in (index - 1, index + 1):
                if 0 <= other < len(new_chapters) and _matched(old_keys[index], new_keys[other]) > own:
                    raise ValueError(f"{label} now appears at another position in the new EPUB. " + _REFUSAL)
            record["id"] = old["id"]
            opcodes = SequenceMatcher(None, old_keys[index], new_keys[index], autojunk=False).get_opcodes()
        else:
            next_chapter += 1
            record["id"] = f"ch{next_chapter:04d}"
            opcodes = [("insert", 0, 0, 0, len(chapter["block_ids"]))]
        identifiers = [None] * len(chapter["block_ids"])
        summary = {"edited_paragraphs": 0, "inserted_paragraphs": 0, "removed_paragraphs": 0}
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                for offset in range(i2 - i1):
                    identifiers[j1 + offset] = old["block_ids"][i1 + offset]
                continue
            fresh = []
            for position in range(j1, j2):
                next_block += 1
                identifiers[position] = f"b{next_block:06d}"
                fresh.append(identifiers[position])
                changed.add(identifiers[position])
            if tag == "replace":
                summary["edited_paragraphs"] += min(i2 - i1, j2 - j1)
                summary["inserted_paragraphs"] += max(0, (j2 - j1) - (i2 - i1))
                summary["removed_paragraphs"] += max(0, (i2 - i1) - (j2 - j1))
                for old_id in old["block_ids"][i1:i2]:
                    block_map[old_id] = list(fresh)
            elif tag == "insert":
                summary["inserted_paragraphs"] += j2 - j1
            else:
                summary["removed_paragraphs"] += i2 - i1
        for position, identifier in enumerate(chapter["block_ids"]):
            block = deepcopy(new_blocks[identifier])
            block.update(id=identifiers[position], chapter_id=record["id"])
            blocks.append(block)
        record["block_ids"] = identifiers
        chapters.append(record)
        if old is not None and any(summary.values()):
            edited.append({"id": record["id"], "title": record["title"], "number": index + 1, **summary})
    kept = {block["id"] for block in blocks}
    shared = sum(1 for identifier in old_blocks if identifier in kept)
    if old_blocks and shared * 2 < len(old_blocks):
        # Shared chapter headings alone must not make another book an edit.
        raise ValueError(f"The new EPUB keeps only {shared} of the saved book's {len(old_blocks)} paragraphs; "
                         "it looks like a different book or a wholesale rewrite. " + _REFUSAL)
    return {"unchanged_chapters": len(old_chapters) - len(edited),
            "appended_chapters": deepcopy(chapters[len(old_chapters):]), "edited_chapters": edited,
            "changed_blocks": sorted(changed),
            "removed_blocks": [identifier for identifier in old_blocks if identifier not in kept],
            "block_map": block_map, "source": _source_like(saved, current, chapters, blocks)}


def has_changes(comparison):
    return bool(comparison["appended_chapters"] or comparison["edited_chapters"])


def _source_like(saved, current, chapters, blocks):
    """A source with the saved book identity and the given chapters and blocks."""
    values = {"book": deepcopy(saved["book"]), "chapters": deepcopy(chapters), "blocks": deepcopy(blocks)}
    result = {}
    for key in [*saved, *(key for key in current if key not in saved)]:
        result[key] = values[key] if key in values else deepcopy(current.get(key, saved.get(key)))
    return result


def delta_source(comparison):
    """What the model analyzes: appended chapters, and each edit with its context.

    A run of changed paragraphs is analyzed with up to ``CONTEXT_BLOCKS``
    unchanged paragraphs on each side as its own section, so attribution sees
    the surrounding dialogue. Sections carry ``<chapter>~edit-N`` IDs;
    ``merge_update`` returns their segments to the real chapter and keeps only
    those of changed paragraphs. Returns ``None`` when no paragraph needs the
    model (only removed paragraphs).
    """
    source = comparison["source"]
    changed = set(comparison["changed_blocks"])
    by_id = {block["id"]: block for block in source["blocks"]}
    appended = {chapter["id"] for chapter in comparison["appended_chapters"]}
    chapters, blocks = [], []
    for chapter in source["chapters"]:
        identifiers = chapter["block_ids"]
        if chapter["id"] in appended:
            windows = [(0, len(identifiers))] if identifiers else []
        else:
            windows = []
            for position, identifier in enumerate(identifiers):
                if identifier not in changed:
                    continue
                low = max(0, position - CONTEXT_BLOCKS)
                high = min(len(identifiers), position + 1 + CONTEXT_BLOCKS)
                if windows and low <= windows[-1][1]:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], high))
                else:
                    windows.append((low, high))
        for number, (low, high) in enumerate(windows, 1):
            section = deepcopy(chapter)
            if chapter["id"] not in appended:
                section["id"] = f"{chapter['id']}~edit-{number}"
            section["block_ids"] = identifiers[low:high]
            chapters.append(section)
            blocks.extend(dict(deepcopy(by_id[identifier]), chapter_id=section["id"])
                          for identifier in section["block_ids"])
    if not chapters:
        return None
    return _source_like(source, source, chapters, blocks)


def merged_source(saved, current):
    """Every chapter of ``current`` under the saved book identity, with stable IDs."""
    return compare_sources(saved, current)["source"]


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


def _is_citation(value):
    return (isinstance(value, dict) and isinstance(value.get("block_id"), str)
            and isinstance(value.get("quote"), str))


def rebind_citations(value, comparison, *, dropped=None):
    """Retarget ``{block_id, quote}`` citations of edited paragraphs.

    A citation of a removed paragraph moves to the paragraph that replaced it
    when that still contains the quotation, and is dropped otherwise (its block
    ID is appended to ``dropped``). Other citations are untouched.
    """
    removed = set(comparison["removed_blocks"])
    if not removed:
        return deepcopy(value)
    texts = {block["id"]: block["text"] for block in comparison["source"]["blocks"]}

    def visit(item):
        if isinstance(item, list):
            result = []
            for entry in item:
                if _is_citation(entry) and entry["block_id"] in removed:
                    target = next((identifier for identifier in comparison["block_map"].get(entry["block_id"], [])
                                   if entry["quote"] in texts[identifier]), None)
                    if target is not None:
                        result.append(dict(deepcopy(entry), block_id=target))
                    elif dropped is not None:
                        dropped.append(entry["block_id"])
                    continue
                result.append(visit(entry))
            return result
        if isinstance(item, dict):
            return {key: visit(entry) for key, entry in item.items()}
        return deepcopy(item)
    return visit(value)


def _remap_row(row, roles, role_voice, chapter_ids):
    row = deepcopy(row)
    if "chapter_id" in row:
        # Edit sections are analyzed under their own IDs; return to the chapter.
        row["chapter_id"] = chapter_ids[row["block_id"]]
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
    return row


def _speakers(rows):
    return [(row["start"], row["end"], row.get("role_id"),
             tuple(speaker["role_id"] for speaker in row.get("simultaneous_speakers", [])))
            for row in rows]


def merge_update(saved, delta, comparison, *, epub_sha256=None):
    """Merge ``delta`` (the analyzed new chapters and edits) into the ``saved`` analysis.

    Unchanged paragraphs keep their saved segments, including speaker choices;
    changed paragraphs take the delta's segments; removed paragraphs leave,
    with citations of them moved to their replacement or dropped and issues
    about them retired. Existing cast records are kept and the delta adds only
    those the book lacks. Where the model now reads an unchanged context
    paragraph's speakers differently, a review issue is added instead of
    changing it. ``delta`` is ``None`` when nothing needed the model. Returns
    the merged document; its update record is ``provenance.book_updates[-1]``.
    """
    source = comparison["source"]
    number = len(saved["provenance"].get("book_updates", [])) + 1
    old_cast = saved["cast"]
    new_cast = delta["cast"] if delta is not None else {"characters": [], "roles": [], "voices": []}
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
    dropped = []
    cast = rebind_citations(old_cast, comparison, dropped=dropped)
    rebased_traits = []
    for voice in cast["voices"]:
        for trait in voice.get("traits", []):
            if trait.get("basis") == "text" and not trait.get("evidence"):
                # Every quotation supporting it was edited out of the book.
                trait["basis"] = "casting"
                rebased_traits.append({"voice_id": voice["id"], "trait": trait.get("name")})
    cast["characters"].extend(appended_characters)
    cast["roles"].extend(appended_roles)
    cast["voices"].extend(appended_voices)
    role_voice = {role["id"]: role["voice_id"] for role in cast["roles"]}

    chapter_ids = {block["id"]: block["chapter_id"] for block in source["blocks"]}
    changed = set(comparison["changed_blocks"])
    saved_rows, delta_rows = {}, {}
    for row in saved["segments"]:
        saved_rows.setdefault(row["block_id"], []).append(row)
    for row in delta["segments"] if delta is not None else []:
        delta_rows.setdefault(row["block_id"], []).append(_remap_row(row, roles, role_voice, chapter_ids))
    segments, disagreements = [], []
    for block in source["blocks"]:
        if block["id"] in changed:
            segments.extend(delta_rows[block["id"]])
            continue
        kept = [rebind_citations(row, comparison, dropped=dropped) for row in saved_rows[block["id"]]]
        segments.extend(kept)
        if block["id"] in delta_rows and _speakers(delta_rows[block["id"]]) != _speakers(kept):
            disagreements.append(block["id"])

    removed = set(comparison["removed_blocks"])
    removed_segments = {row["id"] for row in saved["segments"] if row["block_id"] in removed}

    def head(issue):
        target = issue.get("target")
        return target.split(":", 1)[0] if isinstance(target, str) else None

    issues = []
    for issue in saved["issues"]:
        if issue.get("target") in removed_segments or head(issue) in removed:
            continue
        # Issues inherited from an earlier book cite that book's paragraphs.
        issues.append(deepcopy(issue) if "series_origin" in issue
                      else rebind_citations(issue, comparison, dropped=dropped))
    by_id = {issue.get("id"): issue for issue in issues}
    for issue in delta["issues"] if delta is not None else []:
        mapped = _remap_issue(issue, roles)
        if head(mapped) in chapter_ids and head(mapped) not in changed:
            # A concern about an unchanged context paragraph: its saved
            # analysis stands, and any disagreement is reviewed below.
            continue
        existing = by_id.get(mapped.get("id"))
        if existing == mapped:
            continue
        if existing is not None:
            # Issue IDs hash their content; a distinct new issue keeps its own.
            mapped["id"] = f"{mapped['id']}:update-{number}"
        by_id[mapped.get("id")] = mapped
        issues.append(mapped)
    titles = {chapter["id"]: chapter["title"] for chapter in source["chapters"]}
    for block_id in disagreements:
        issues.append({
            "id": f"book-update-{number}:context:{block_id}", "kind": "update_context_attribution",
            "target": next(row["id"] for row in segments if row["block_id"] == block_id),
            "reason": (f"After an edit nearby in {titles[chapter_ids[block_id]]!r}, the model reads this "
                       "unchanged paragraph's speakers differently. Its saved speakers were kept; check them."),
            "resolved": False})

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
        drops = (delta or {}).get("provenance", {}).get("bracket_delivery", {}).get("inline_projection_drops", [])
        if drops:
            marker["inline_projection_drops"] = [*marker.get("inline_projection_drops", []), *deepcopy(drops)]
    update = {
        "version": UPDATE_VERSION,
        "previous_source_fingerprint": previous, "source_fingerprint": fingerprint(source),
        "delta_source_fingerprint": fingerprint(delta["source"]) if delta is not None else None,
        "epub_sha256": epub_sha256,
        "added_chapters": [{"id": chapter["id"], "title": chapter["title"]}
                           for chapter in comparison["appended_chapters"]],
        "edited_chapters": deepcopy(comparison["edited_chapters"]),
        "new_characters": [character["id"] for character in appended_characters if status(character) != "reused"],
        "returning_series_characters": [character["id"] for character in appended_characters
                                        if status(character) == "reused"],
        "existing_characters": [character["id"] for character in new_cast["characters"]
                                if character["id"] in old_characters],
        "added_roles": [role["id"] for role in appended_roles],
        "added_voices": [voice["id"] for voice in appended_voices],
        "role_map": {key: value for key, value in roles.items() if key != value},
        "context_reviews": disagreements,
        "dropped_citations": len(dropped),
        "rebased_traits": rebased_traits,
        "strategy": (saved_strategy(delta) if delta is not None else None),
        "analysis": deepcopy(delta["provenance"]) if delta is not None else None,
    }
    provenance.setdefault("book_updates", []).append(update)
    document = deepcopy(saved)
    document.update(source=deepcopy(source), cast=cast, segments=segments, issues=issues,
                    provenance=provenance)
    document.pop("readiness", None)
    errors = validate_document(document)
    if errors:
        raise ValueError("The new and edited chapters could not be merged into the saved book: "
                         + "; ".join(errors[:8]))
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


def _rebind_pronunciation(entry, comparison, source):
    """Move a choice's citations off edited paragraphs, or cite another occurrence.

    Returns ``None`` when the word no longer occurs anywhere in the book.
    """
    from .pronunciations import _evidence_contains_word
    entry = rebind_citations(entry, comparison)
    if entry["evidence"]:
        return entry
    block = next((block for block in source["blocks"]
                  if _evidence_contains_word(entry["word"], entry["word"], block["text"])), None)
    return None if block is None else dict(entry, evidence=[{"block_id": block["id"], "quote": entry["word"]}])


def _rebind_sidecars(directory, saved, document):
    """Rebind book-local sidecars that bind the source (and cast) fingerprints.

    Pronunciation citations of edited paragraphs move to their replacement or
    another occurrence of the word; a choice for a word no longer in the book
    is retired. Every kept entry is revalidated against the new source.
    """
    comparison = compare_sources(saved["source"], document["source"])
    if comparison["source"] != document["source"]:
        raise ValueError("The updated source does not align with the saved book")
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
        entries = [_rebind_pronunciation(entry, comparison, document["source"]) for entry in record["entries"]]
        record["entries"] = [entry for entry in entries if entry is not None]
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


def _script_rows(directory, manifest):
    """Script rows by source chapter, without their book-wide ``sequence`` numbers.

    An edit that adds or removes segments renumbers every later row's
    sequence; that is bookkeeping, not a change to those chapters.
    """
    rows = {}
    for entry in manifest["files"]["scripts"]:
        for line in (Path(directory) / entry["path"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.pop("sequence", None)
                rows.setdefault(row["chapter_id"], []).append(row)
    return rows


def publish_update(directory, document, *, expected_manifest_sha256, progress=print):
    """Install a book update atomically, with backups and a journal.

    Unchanged chapters' script rows must come out identical apart from their
    book-wide sequence numbers; edited chapters' scripts are replaced and new
    ones added. Any failure restores every replaced file and removes added ones.
    """
    from .cli import export_bundle
    from .renderer.run_locks import acquire_directory_lock
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
        lock_path = directory / ".analysis-update.lock"
        try:
            lock = acquire_directory_lock(lock_path)
        except FileExistsError as exc:
            raise ValueError(
                f"Analysis revision is already active, or the lock cannot be proven stale: {lock_path}. "
                f"A marked lock left by a crashed process is reclaimed automatically when you rerun.") from exc
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
        if new_scripts[:len(old_scripts)] != old_scripts:
            raise ValueError("The edited chapters would divide the book into different chapter files; analyze "
                             "the new EPUB into a separate output directory instead")
        edited = {chapter["id"] for chapter in document["provenance"]["book_updates"][-1].get("edited_chapters", [])}
        before_rows, after_rows = _script_rows(directory, manifest), _script_rows(after, new_manifest)
        if any(after_rows.get(chapter) != rows for chapter, rows in before_rows.items() if chapter not in edited):
            raise ValueError("An update must keep every unchanged chapter's script rows unchanged")
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
    lines = []
    if chapters:
        lines.append(f"Added {len(chapters)} chapter{'s' if len(chapters) != 1 else ''}: "
                     + ", ".join(chapter["title"] for chapter in chapters))
    for chapter in update.get("edited_chapters", []):
        counts = [f"{chapter[key]} {label}" for key, label in (
            ("edited_paragraphs", "edited"), ("inserted_paragraphs", "inserted"),
            ("removed_paragraphs", "removed")) if chapter[key]]
        lines.append(f"Edited chapter {chapter['number']} ({chapter['title']}): "
                     + ", ".join(counts) + " paragraph(s)")
    if update.get("context_reviews"):
        lines.append(f"{len(update['context_reviews'])} unchanged paragraph(s) next to an edit now read "
                     "differently; their saved speakers were kept and added to the review list.")
    if update.get("rebased_traits"):
        lines.append(f"{len(update['rebased_traits'])} voice trait(s) lost their quoted evidence to an edit "
                     "and are now marked as casting choices.")
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


def saved_strategy(document):
    """The analysis strategy a saved book was made with: "narrator" or "consensus"."""
    return "narrator" if document["provenance"].get("strategy") == "narrator_read" else "consensus"


def update_book(directory, source, *, config, input_path=None, client_factory, speaker_judge=None,
                announce_cache=lambda path: Path(path).resolve(), progress=print, log=print, strategy=None):
    """Analyze and publish the new and edited chapters of ``directory``'s book.

    ``strategy`` ("consensus" or "narrator") selects the analysis for the new
    paragraphs; by default the one the book was analyzed with.

    Returns the published document, or ``None`` when the source has no new or
    edited chapters. ``client_factory`` builds the analysis client from
    ``config["provider"]`` only after every local check has passed, and only
    when some paragraph needs the model (removing paragraphs needs none).
    """
    from .analysis_progress import (STAGES, analysis_cache_directory, analysis_options_fingerprint,
                                    create_pending, pending_identity, resume_document)
    from .audition_scripts import prepare_document_samples
    from .pipeline import CachedRequests, analyze
    from .speaker_decisions import prompt_speaker_decisions
    directory = Path(directory).resolve()
    original, registry, _ = _completed_directory(directory)
    if strategy not in (None, "consensus", "narrator"):
        raise ValueError("strategy must be consensus or narrator")
    strategy = strategy or saved_strategy(original)
    comparison = compare_sources(original["source"], source)
    if not has_changes(comparison):
        log("No new or edited chapters: the EPUB's chapters match the saved book. "
            "The saved analysis is unchanged.")
        return None
    ensure_updatable(directory)
    manifest_hash = _hash(directory / "manifest.json")
    settings = config["analysis"]
    merged = comparison["source"]
    delta = delta_source(comparison)
    if delta is not None and not config["provider"]["model"]:
        raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
    cache = analysis_cache_directory(directory, settings["cache_dir"])
    if (directory / "name-script.json").exists():
        log("Note: this book has a legacy name-script.json. It covers only the earlier chapters; "
            "name-scoped pronunciation choices that depend on it are blocked until it is prepared again.")
    epub = Path(input_path) if input_path is not None and Path(input_path).suffix.lower() == ".epub" else None
    digest = epub_digest(epub) if epub is not None else None
    changes = (f"{len(comparison['appended_chapters'])} new and "
               f"{len(comparison['edited_chapters'])} edited chapter(s)")
    client = client_factory(**config["provider"]) if delta is not None else None
    if client is not None:
        log(f"Updating {directory} with {changes} using {client.model} at {client.base_url}"
            f"{' (narrator read)' if strategy == 'narrator' else ''}; only the new "
            "chapters and the edited paragraphs with their neighbours will be sent to this endpoint.")
    else:
        log(f"Updating {directory} with {changes}; only paragraphs were removed, so no model is needed.")
    model_cache = announce_cache(cache)
    identity = pending_identity(merged, client.identity if client is not None else {"model": None},
                                analysis_options_fingerprint(
        settings, config["performance_policy"],
        # A narrator-read update never resumes a consensus update's checkpoints.
        fingerprint({"book_update": UPDATE_VERSION, "base_manifest_sha256": manifest_hash,
                     **({"strategy": strategy} if strategy != "consensus" else {})})))
    workspace = update_workspace(directory)
    resumed = resume_document(workspace, identity)
    document, resume_index = None, 0
    if resumed is not None:
        document, next_stage = resumed
        resume_index = len(STAGES) if next_stage is None else STAGES.index(next_stage)
        if resume_index:
            log("Resuming the interrupted book update from its last completed stage.")
    pending = create_pending(workspace, identity)
    requests = (CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                               max_request_chars=settings["max_request_chars"], progress=progress)
                if client is not None else None)
    workers = min(5, settings["attribution_workers"])
    try:
        if resume_index <= 0:
            pending.begin_stage("analyzed")
            if delta is None:
                analyzed = None
            elif strategy == "narrator":
                from .narrator_read import narrated_analyze, settings_notes
                analyzed = narrated_analyze(
                    delta, client, model_cache, series_registry=deepcopy(registry),
                    policy=config["performance_policy"], workers=settings["attribution_workers"],
                    repair_attempts=settings["repair_attempts"], max_request_chars=settings["max_request_chars"],
                    progress=progress, **settings_notes(settings))
            else:
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
            document = merge_update(original, analyzed, comparison, epub_sha256=digest)
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
    log("Rendering reuses the audio of every unchanged paragraph; run audiobook-render render on this "
        "book to cast any new voices and render the new and edited passages.")
    return document


def plan_update(directory, source, config):
    """The ``plan`` estimate for updating a saved book with ``source``'s new and edited chapters."""
    from .analysis_plan import plan_analysis
    directory = Path(directory).resolve()
    original, registry, _ = _completed_directory(directory)
    comparison = compare_sources(original["source"], source)
    update = {"unchanged_chapters": comparison["unchanged_chapters"],
              "added_chapters": [{"id": chapter["id"], "title": chapter["title"]}
                                 for chapter in comparison["appended_chapters"]],
              "edited_chapters": deepcopy(comparison["edited_chapters"])}
    delta = delta_source(comparison)
    if delta is None:
        note = ("No new or edited chapters" if not has_changes(comparison)
                else "Only paragraphs were removed") + "; analyze would make no model requests."
        return {"output_directory": str(directory), "book_update": update, "note": note}
    planned = plan_analysis(delta, config["analysis"], config["performance_policy"], registry)
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
