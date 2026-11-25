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
and edited and the new characters it introduced. New chapters may come
between saved ones; a saved chapter or paragraph now divided into several (a
newer extractor reading the EPUB's table of contents or line-break paragraphs)
keeps its speakers without the model. Removed or reordered chapters are refused
with their location.

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
_REFUSAL = ("A saved book can be updated with new chapters (appended or between its chapters), with its "
            "chapters or paragraphs split apart, and with edits inside its existing chapters, but not with "
            "chapters removed or reordered. Analyze the new EPUB into a separate output directory instead; "
            "the saved book was not changed")
# Extraction rules that only change how a book divides into chapters and paragraphs: an
# update may add (or drop) them, because the alignment below keeps the saved IDs anyway.
LAYOUT_POLICY_PARTS = frozenset({"toc-anchor-chapters=split", "line-break-paragraphs=indent"})


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


def _policy_parts(source):
    policy = source.get("extraction_policy")
    return {part.strip() for part in policy.split(";")} if isinstance(policy, str) else set()


def _check_policy(saved, current):
    """Refuse a different extraction unless it differs only in layout rules."""
    if saved.get("extraction_policy") == current.get("extraction_policy"):
        return
    old, new = _policy_parts(saved), _policy_parts(current)
    if (isinstance(saved.get("extraction_policy"), str) and isinstance(current.get("extraction_policy"), str)
            and (old ^ new) <= LAYOUT_POLICY_PARTS):
        return
    raise ValueError("The new EPUB was extracted with different extraction settings "
                     "(extraction.include_auxiliary or the extractor version). " + _REFUSAL)


def _splits(old_blocks, new_blocks):
    """For a replaced run, old paragraphs that are now several consecutive new ones.

    Returns ``{old position: [new positions]}`` within the run: an old
    paragraph whose text is exactly its new paragraphs' texts joined by a
    space (how the extractor joins a line break) was split, not edited.
    """
    found, j = {}, 0
    for i, old in enumerate(old_blocks):
        text, pieces, joined = old["text"], [], ""
        for k in range(j, len(new_blocks)):
            joined = new_blocks[k]["text"] if not pieces else f"{joined} {new_blocks[k]['text']}"
            pieces.append(k)
            if joined == text:
                break
            if not text.startswith(joined + " "):
                pieces = []
                break
        if len(pieces) > 1 and joined == text:
            found[i] = pieces
            j = pieces[-1] + 1
    return found


def compare_sources(saved, current, *, preserve_citations_from=None):
    """Align ``current`` with a saved source by paragraph content.

    Every paragraph is keyed by its text and markup (not its position or EPUB
    location), and the whole book's saved and new paragraphs are aligned with
    ``difflib``. Unchanged paragraphs keep their saved block IDs, so their
    segments, speaker choices and rendered audio stay attached; edited,
    inserted and appended paragraphs get new IDs after the highest saved one,
    and saved paragraphs no longer present are removed. A saved paragraph that
    is now several paragraphs whose text joins back to it exactly was split
    (``split_blocks``): its pieces get new IDs and ``merge_update`` divides
    its saved segments among them, with no model request. A one-for-one
    replacement cited by ``preserve_citations_from`` instead keeps its old
    block ID, leaving that cast record intact while its changed text still
    gets a new chunk ID.

    Each saved chapter keeps its ID on the first new chapter holding its
    paragraphs; a saved chapter divided into several (``split_chapters``)
    gives its later parts new IDs, and a chapter of only new paragraphs, at
    the end or between saved chapters, is a new chapter (``appended_chapters``)
    analyzed whole. A saved chapter with no paragraph left, joined into another
    saved chapter, or now in a different order is refused with its location.

    Returns the merged ``source`` (saved book identity, new content, stable
    IDs) with a description of what changed.
    """
    _check_policy(saved, current)
    old_blocks = {block["id"]: block for block in saved["blocks"]}
    new_blocks = {block["id"]: block for block in current["blocks"]}
    old_chapters, new_chapters = saved["chapters"], current["chapters"]
    old_order = [(index, identifier) for index, chapter in enumerate(old_chapters) for identifier in chapter["block_ids"]]
    new_order = [(index, identifier) for index, chapter in enumerate(new_chapters) for identifier in chapter["block_ids"]]

    def label(index):
        return f"Chapter {index + 1} ({old_chapters[index]['title']!r})"

    # A chapter moved elsewhere: compare each saved chapter with the new chapter sharing most of its paragraphs.
    from collections import Counter
    new_counts = [Counter(_block_key(new_blocks[identifier]) for identifier in chapter["block_ids"])
                  for chapter in new_chapters]
    best = {}
    for index, chapter in enumerate(old_chapters):
        counts = Counter(_block_key(old_blocks[identifier]) for identifier in chapter["block_ids"])
        shares = [sum((counts & other).values()) for other in new_counts]
        if counts and shares and max(shares) * 2 > sum(counts.values()):
            best[index] = shares.index(max(shares))
    placed = sorted(best)
    for position, index in enumerate(placed):
        if any(best[later] < best[index] for later in placed[position + 1:]):
            raise ValueError(f"{label(index)} now appears at another position in the new EPUB. " + _REFUSAL)

    old_keys = [_block_key(old_blocks[identifier]) for _, identifier in old_order]
    new_keys = [_block_key(new_blocks[identifier]) for _, identifier in new_order]
    next_block = max((_number(identifier, "b") for identifier in old_blocks), default=0)
    next_chapter = max((_number(chapter["id"], "ch") for chapter in old_chapters), default=0)
    cited_blocks, pending, replaced = set(), [preserve_citations_from], set()
    while pending:
        item = pending.pop()
        if _is_citation(item):
            cited_blocks.add(item["block_id"])
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)

    identifiers = [None] * len(new_order)
    origin = [None] * len(new_order)        # saved chapter a kept or split paragraph comes from
    kind = [None] * len(new_order)          # "kept", "split" or "changed"
    block_map, split_blocks, changed = {}, {}, set()
    # Per new chapter: edited/inserted/removed/split paragraph counts.
    counts = [{"edited_paragraphs": 0, "inserted_paragraphs": 0, "removed_paragraphs": 0, "split_paragraphs": 0}
              for _ in new_chapters]

    def fresh_block():
        nonlocal next_block
        next_block += 1
        return f"b{next_block:06d}"

    def chapter_at(position):
        """The new chapter a removal at new ``position`` belongs to."""
        if not new_order:
            return None
        return new_order[min(max(position, 0), len(new_order) - 1)][0]

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old_keys, new_keys, autojunk=False).get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                identifiers[j1 + offset] = old_order[i1 + offset][1]
                origin[j1 + offset] = old_order[i1 + offset][0]
                kind[j1 + offset] = "kept"
            continue
        splits = _splits([old_blocks[old_order[i][1]] for i in range(i1, i2)],
                         [new_blocks[new_order[j][1]] for j in range(j1, j2)]) if tag == "replace" else {}
        for old_offset, pieces in splits.items():
            old_index, old_id = old_order[i1 + old_offset]
            fresh = []
            for piece in pieces:
                position = j1 + piece
                identifiers[position], origin[position], kind[position] = fresh_block(), old_index, "split"
                fresh.append(identifiers[position])
            split_blocks[old_id] = fresh
            block_map[old_id] = list(fresh)
            counts[new_order[j1 + pieces[0]][0]]["split_paragraphs"] += 1
        old_rest = [i for i in range(i1, i2) if i - i1 not in splits]
        split_new = {j1 + piece for pieces in splits.values() for piece in pieces}
        new_rest = [j for j in range(j1, j2) if j not in split_new]
        if not old_rest and not new_rest:
            continue
        previous = [old_order[i][1] for i in old_rest] if tag == "replace" else []
        fresh = []
        for offset, position in enumerate(new_rest):
            old_id = previous[offset] if len(previous) == len(new_rest) else None
            if old_id in cited_blocks:
                identifiers[position] = old_id
                replaced.add(old_id)
            else:
                identifiers[position] = fresh_block()
            kind[position] = "changed"
            fresh.append(identifiers[position])
            changed.add(identifiers[position])
        for old_id in previous:
            block_map[old_id] = list(fresh)
        edited = min(len(old_rest), len(new_rest)) if tag == "replace" else 0
        where = new_order[new_rest[0]][0] if new_rest else chapter_at(j1 - 1 if j1 else j1)
        if where is not None:
            counts[where]["edited_paragraphs"] += edited
            counts[where]["inserted_paragraphs"] += len(new_rest) - edited
            counts[where]["removed_paragraphs"] += len(old_rest) - edited

    # Chapter IDs: the first new chapter holding a saved chapter's paragraphs keeps its ID.
    sources = [set() for _ in new_chapters]
    for position, (chapter_index, _) in enumerate(new_order):
        if origin[position] is not None:
            sources[chapter_index].add(origin[position])
    chapter_ids, taken, owner = [None] * len(new_chapters), set(), {}
    for index, found in enumerate(sources):
        if found and min(found) not in taken:
            chapter_ids[index] = old_chapters[min(found)]["id"]
            taken.add(min(found))
            owner[min(found)] = index
    # A saved chapter with no paragraphs (a cover) keeps its ID on an empty new chapter in the same place.
    for old_index, chapter in enumerate(old_chapters):
        if chapter["block_ids"]:
            continue
        low = max((owner[i] for i in owner if i < old_index), default=-1)
        high = min((owner[i] for i in owner if i > old_index), default=len(new_chapters))
        empty = [index for index in range(low + 1, high) if not new_chapters[index]["block_ids"]
                 and chapter_ids[index] is None]
        same = [index for index in empty if new_chapters[index]["title"] == chapter["title"]]
        if same or len(empty) == 1:
            chapter_ids[(same or empty)[0]] = chapter["id"]
            taken.add(old_index)
            owner[old_index] = (same or empty)[0]
    for old_index, chapter in enumerate(old_chapters):
        if old_index in taken:
            continue
        if any(old_index in found for found in sources):
            raise ValueError(f"{label(old_index)} is now part of another chapter in the new EPUB. " + _REFUSAL)
        if chapter["block_ids"] and len(new_chapters) >= len(old_chapters):
            raise ValueError(f"{label(old_index)} has no paragraph in common with the new EPUB. " + _REFUSAL)
        raise ValueError(f"{label(old_index)} is missing from the new EPUB. " + _REFUSAL)
    order = [owner[index] for index in sorted(owner)]
    if order != sorted(order):
        first = next(index for index in sorted(owner) if any(owner[later] < owner[index] for later in owner if later > index))
        raise ValueError(f"{label(first)} now appears at another position in the new EPUB. " + _REFUSAL)

    chapters, blocks, edited, retitled, split_chapters, appended, derived = [], [], [], [], [], [], {}
    old_by_id = {chapter["id"]: chapter for chapter in old_chapters}
    for index, chapter in enumerate(new_chapters):
        record = deepcopy(chapter)
        if chapter_ids[index] is None:
            next_chapter += 1
            chapter_ids[index] = f"ch{next_chapter:04d}"
        record["id"] = chapter_ids[index]
        positions = [position for position, (owner_index, _) in enumerate(new_order) if owner_index == index]
        record["block_ids"] = [identifiers[position] for position in positions]
        for position in positions:
            block = deepcopy(new_blocks[new_order[position][1]])
            block.update(id=identifiers[position], chapter_id=record["id"])
            blocks.append(block)
        chapters.append(record)
        old = old_by_id.get(record["id"])
        number = index + 1
        summary = counts[index]
        if old is None and not sources[index]:
            appended.append(deepcopy(record))
            continue
        if old is None:
            derived[index] = old_chapters[min(sources[index])]["id"]
        elif old["title"] != record["title"]:
            retitled.append({"id": record["id"], "number": number,
                             "previous_title": old["title"], "title": record["title"]})
        edits = {key: summary[key] for key in ("edited_paragraphs", "inserted_paragraphs", "removed_paragraphs")}
        if any(edits.values()):
            edited.append({"id": record["id"], "title": record["title"], "number": number, **edits})
    # Layout changed by splitting: chapters split off a saved one, the saved chapter they left,
    # and chapters whose paragraphs were split.
    donors = set(derived.values())
    for index, record in enumerate(chapters):
        if index in derived or record["id"] in donors or (counts[index]["split_paragraphs"]
                                                          and record["id"] in old_by_id):
            split_chapters.append({"id": record["id"], "title": record["title"], "number": index + 1,
                                   "from": derived.get(index, record["id"]),
                                   "split_paragraphs": counts[index]["split_paragraphs"]})
    kept_ids = {block["id"] for block in blocks}
    shared = sum(1 for identifier in old_blocks if identifier in kept_ids or identifier in split_blocks)
    if old_blocks and shared * 2 < len(old_blocks):
        # Shared chapter headings alone must not make another book an edit.
        raise ValueError(f"The new EPUB keeps only {shared} of the saved book's {len(old_blocks)} paragraphs; "
                         "it looks like a different book or a wholesale rewrite. " + _REFUSAL)
    changed_chapters = {chapter["id"] for chapter in [*edited, *retitled, *split_chapters]} & {c["id"] for c in old_chapters}
    return {"unchanged_chapters": len(old_chapters) - len(changed_chapters),
            "appended_chapters": appended, "edited_chapters": edited,
            "retitled_chapters": retitled, "split_chapters": split_chapters,
            "split_blocks": split_blocks, "changed_blocks": sorted(changed),
            "removed_blocks": [identifier for identifier in old_blocks
                               if identifier not in kept_ids or identifier in replaced],
            "block_map": block_map, "source": _source_like(saved, current, chapters, blocks)}


def reread_comparison(saved, reasons):
    """A comparison that reads the saved paragraphs ``reasons`` names again, text unchanged.

    The paragraphs keep their block IDs; ``delta_source`` gives each its
    neighbours as context and ``merge_update`` replaces only their segments.
    ``reasons`` maps block IDs to why the saved split is wrong
    (``split_check.misread_blocks``).
    """
    chapters = {block["id"]: block["chapter_id"] for block in saved["blocks"]}
    counts = {}
    for block_id in reasons:
        counts[chapters[block_id]] = counts.get(chapters[block_id], 0) + 1
    reread = [{"id": chapter["id"], "title": chapter["title"], "number": number, "reread_paragraphs": counts[chapter["id"]]}
              for number, chapter in enumerate(saved["chapters"], 1) if chapter["id"] in counts]
    return {"unchanged_chapters": len(saved["chapters"]) - len(reread), "appended_chapters": [],
            "edited_chapters": [], "retitled_chapters": [], "split_chapters": [], "split_blocks": {},
            "reread_chapters": reread,
            "reread_reasons": dict(sorted(reasons.items())), "changed_blocks": sorted(reasons),
            "removed_blocks": [], "block_map": {}, "source": deepcopy(saved)}


def has_changes(comparison):
    return bool(comparison["appended_chapters"] or comparison["edited_chapters"]
                or comparison["retitled_chapters"] or comparison.get("split_chapters"))


def _split_segments(rows, old_text, pieces):
    """Divide a split paragraph's saved segments among its pieces ``[(block_id, text)]``.

    The pieces joined by single spaces are ``old_text``. A segment inside one
    piece moves there unchanged apart from its offsets and ID; a segment that
    spans pieces (narration running across the new paragraph break) becomes
    one segment per piece with the same speaker, keeping only the scoped
    performance controls inside each part (``slice_performance``). Returns the new
    segments, an ``{old segment ID: first new segment ID}`` map, and how many
    controls or spoken-form edits a division dropped.
    """
    from .performance_plan import slice_performance
    result, renamed, dropped = [], {}, 0
    offset = 0
    spans = []
    for block_id, text in pieces:
        spans.append((block_id, offset, offset + len(text)))
        offset += len(text) + 1
    for row in rows:
        parts = [(block_id, max(row["start"], low), min(row["end"], high))
                 for block_id, low, high in spans if row["start"] < high and row["end"] > low]
        for number, (block_id, start, end) in enumerate(parts):
            local_start = start - next(low for piece, low, _ in spans if piece == block_id)
            local_end = local_start + (end - start)
            moved = deepcopy(row)
            moved.update(id=f"{block_id}:{local_start}:{local_end}", block_id=block_id,
                         start=local_start, end=local_end, text=old_text[start:end])
            if (start, end) != (row["start"], row["end"]):
                # Divided, or trimmed of the space between two new paragraphs.
                cut = start - row["start"]
                respelled = False
                for key in ("speech_normalization", "spoken_text", "performance"):
                    if key in moved:
                        dropped += key != "spoken_text"
                        respelled = True
                        moved.pop(key)
                controls = moved.get("performance_controls")
                if controls is not None and respelled:
                    # Its control offsets are in the spoken form just dropped.
                    dropped += len(controls["controls"])
                    moved.pop("performance_controls")
                elif controls is not None:
                    try:
                        sliced = slice_performance(controls, row["text"], cut, cut + (end - start))
                    except ValueError:
                        # A pronunciation the new break cuts through.
                        dropped += len(controls["controls"])
                        sliced = None
                    if sliced is not None and sliced["controls"]:
                        moved["performance_controls"] = sliced
                    else:
                        moved.pop("performance_controls")
            if number == 0:
                renamed[row["id"]] = moved["id"]
            result.append(moved)
    return result, renamed, dropped


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


def merge_update(saved, delta, comparison, *, epub_sha256=None, keep_cast=False):
    """Merge ``delta`` (the analyzed new chapters and edits) into the ``saved`` analysis.

    Unchanged paragraphs keep their saved segments, including speaker choices;
    changed paragraphs take the delta's segments; removed paragraphs leave,
    with citations of them moved to their replacement or dropped and issues
    about them retired. Existing cast records are kept and the delta adds only
    those the book lacks. Where the model now reads an unchanged context
    paragraph's speakers differently, a review issue is added instead of
    changing it. With ``keep_cast`` the cast stays exactly as saved: a changed
    paragraph whose new speaker the book lacks keeps its saved segments and
    gets a review issue. ``update_book`` no longer asks for it: a re-read adds
    the speaker, also in a book later books continue from.
    ``delta`` is ``None`` when nothing needed the model. Returns
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
    if keep_cast:
        appended_characters, appended_roles = [], []
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
    # A delta role the cast lacks (only with ``keep_cast``) keeps its own voice here, so its
    # paragraph is found below and keeps its saved segments.
    role_voice = {**{role["id"]: role["voice_id"] for role in new_cast["roles"]},
                  **{role["id"]: role["voice_id"] for role in cast["roles"]}}

    chapter_ids = {block["id"]: block["chapter_id"] for block in source["blocks"]}
    changed = set(comparison["changed_blocks"])
    saved_rows, delta_rows = {}, {}
    for row in saved["segments"]:
        saved_rows.setdefault(row["block_id"], []).append(row)
    for row in delta["segments"] if delta is not None else []:
        delta_rows.setdefault(row["block_id"], []).append(_remap_row(row, roles, role_voice, chapter_ids))
    segments, disagreements, kept_for_cast = [], [], []
    known_roles = {role["id"] for role in cast["roles"]}
    old_texts = {block["id"]: block["text"] for block in saved["source"]["blocks"]}
    new_texts = {block["id"]: block["text"] for block in source["blocks"]}
    split_rows, renamed_segments, dropped_controls = {}, {}, 0
    for old_id, pieces in comparison.get("split_blocks", {}).items():
        rows = [rebind_citations(row, comparison, dropped=dropped) for row in saved_rows.get(old_id, [])]
        divided, renamed, lost = _split_segments(rows, old_texts[old_id],
                                                 [(piece, new_texts[piece]) for piece in pieces])
        renamed_segments.update(renamed)
        dropped_controls += lost
        for row in divided:
            split_rows.setdefault(row["block_id"], []).append(row)
    for block in source["blocks"]:
        if block["id"] in split_rows:
            segments.extend(split_rows[block["id"]])
            continue
        if block["id"] in changed and keep_cast and any(
                role_id not in known_roles for row in delta_rows[block["id"]]
                for role_id in [row.get("role_id"), *(item["role_id"] for item in row.get("simultaneous_speakers", []))]
                if role_id is not None):
            kept_for_cast.append(block["id"])
            segments.extend(rebind_citations(row, comparison, dropped=dropped) for row in saved_rows[block["id"]])
            continue
        if block["id"] in changed:
            segments.extend(delta_rows[block["id"]])
            continue
        kept = [rebind_citations(row, comparison, dropped=dropped) for row in saved_rows[block["id"]]]
        segments.extend(kept)
        if block["id"] in delta_rows and _speakers(delta_rows[block["id"]]) != _speakers(kept):
            disagreements.append(block["id"])

    removed = set(comparison["removed_blocks"])
    kept_segments = {row["id"] for row in segments}
    # Segments of removed paragraphs, and saved segments of re-read paragraphs that the new split replaced.
    removed_segments = {row["id"] for row in saved["segments"]
                        if row["block_id"] in removed or row["block_id"] in changed and row["id"] not in kept_segments}

    def head(issue):
        target = issue.get("target")
        return target.split(":", 1)[0] if isinstance(target, str) else None

    issues = []
    for issue in saved["issues"]:
        if issue.get("target") in renamed_segments:
            issue = dict(issue, target=renamed_segments[issue["target"]])
        elif issue.get("target") in removed_segments or head(issue) in removed:
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
    for block_id in kept_for_cast:
        issues.append({
            "id": f"book-update-{number}:new-speaker:{block_id}", "kind": "update_context_attribution",
            "target": next(row["id"] for row in segments if row["block_id"] == block_id),
            "reason": (f"Read again in {titles[chapter_ids[block_id]]!r}, this paragraph's split names a speaker "
                       "the book has no voice for, and a later book continues from this one, so its cast cannot "
                       "change here. Its saved split was kept; check its speakers."),
            "resolved": False})
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
        marker = provenance["bracket_delivery"]
        ranges, warnings = bracket_ranges(source, version=marker.get("version", 1))
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
        "retitled_chapters": deepcopy(comparison["retitled_chapters"]),
        **({"split_chapters": deepcopy(comparison["split_chapters"]),
            "split_paragraphs": len(comparison["split_blocks"]),
            "split_dropped_controls": dropped_controls} if comparison.get("split_chapters") else {}),
        **({"reread_chapters": deepcopy(comparison["reread_chapters"]),
            "reread_paragraphs": deepcopy(comparison["reread_reasons"])} if comparison.get("reread_chapters") else {}),
        "new_characters": [character["id"] for character in appended_characters if status(character) != "reused"],
        "returning_series_characters": [character["id"] for character in appended_characters
                                        if status(character) == "reused"],
        "existing_characters": [character["id"] for character in new_cast["characters"]
                                if character["id"] in old_characters],
        "added_roles": [role["id"] for role in appended_roles],
        "added_voices": [voice["id"] for voice in appended_voices],
        "role_map": {key: value for key, value in roles.items() if key != value},
        "context_reviews": disagreements,
        **({"reread_kept_for_cast": kept_for_cast} if kept_for_cast else {}),
        "dropped_citations": len(dropped),
        "rebased_traits": rebased_traits,
        "strategy": (saved_strategy(delta) if delta is not None else None),
        "analysis": deepcopy(delta["provenance"]) if delta is not None else None,
    }
    provenance.setdefault("book_updates", []).append(update)
    if comparison.get("reread_reasons"):
        from .split_check import record_rereads
        record_rereads(provenance, {"source": source}, comparison["reread_reasons"])
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
    comparison = compare_sources(saved["source"], document["source"], preserve_citations_from=saved["cast"])
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
    from .chapter_ignores import FILENAME as IGNORES, rebind_validated
    path = directory / IGNORES
    if path.exists() or path.is_symlink():
        try:
            record = rebind_validated(directory, saved, document)
        except ValueError as exc:
            raise ValueError(f"The book update changes the layout of an ignored chapter; review the "
                             f"chapter ignores before updating this book: {exc}") from exc
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
        if reason := _render_active(directory):
            raise ValueError(f"Rendering is active; finish it before updating the book: {directory} ({reason})")
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
        latest = document["provenance"]["book_updates"][-1]
        edited = {chapter["id"] for chapter in [*latest.get("edited_chapters", []), *latest.get("reread_chapters", []),
                                                *latest.get("split_chapters", [])]}
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
            if reason := _render_active(directory):
                raise ValueError(f"Rendering started during the update: {directory} ({reason})")
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
    for chapter in update.get("retitled_chapters", []):
        lines.append(f"Retitled chapter {chapter['number']} ({chapter['previous_title']}): {chapter['title']}")
    if chapters:
        numbers = {chapter["id"]: number for number, chapter in enumerate(document["source"]["chapters"], 1)}
        lines.append(f"Added {len(chapters)} chapter{'s' if len(chapters) != 1 else ''}: "
                     + ", ".join(f"{chapter['title']} (chapter {numbers[chapter['id']]})" if chapter["id"] in numbers
                                 else chapter["title"] for chapter in chapters))
    for chapter in update.get("edited_chapters", []):
        counts = [f"{chapter[key]} {label}" for key, label in (
            ("edited_paragraphs", "edited"), ("inserted_paragraphs", "inserted"),
            ("removed_paragraphs", "removed")) if chapter[key]]
        lines.append(f"Edited chapter {chapter['number']} ({chapter['title']}): "
                     + ", ".join(counts) + " paragraph(s)")
    split = update.get("split_chapters", [])
    if split:
        parts = sum(1 for chapter in split if chapter["from"] != chapter["id"])
        lines.append(f"Divided the saved chapters differently, as the EPUB's table of contents and paragraphs do: "
                     f"{parts} chapter(s) split off, {update.get('split_paragraphs', 0)} paragraph(s) split into "
                     "several; their saved speakers and directions were kept"
                     + (f" ({update['split_dropped_controls']} line control(s) that spanned a new break were dropped)"
                        if update.get("split_dropped_controls") else "") + ".")
    for chapter in update.get("reread_chapters", []):
        lines.append(f"Read again chapter {chapter['number']} ({chapter['title']}): {chapter['reread_paragraphs']} "
                     "paragraph(s) whose saved split did not match their quotation marks")
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


def ensure_updatable(directory, comparison=None):
    """Refuse only updates the book's dependents could not survive."""
    directory = Path(directory).resolve()
    if reason := _render_active(directory):
        raise ValueError(f"Rendering is active; finish it before updating the book: {directory} ({reason})")
    related = related_directories([directory])
    if (related != [directory] and comparison is not None
            and comparison.get("appended_chapters")):
        descendants = ", ".join(str(path) for path in related if path != directory)
        raise ValueError("A later book continues from this one, so its chapters cannot be extended in place "
                         f"(continued by {descendants}). Analyze the new EPUB into a separate output directory")
    return related


def saved_strategy(document):
    """The analysis strategy a saved book was made with: "narrator" or "consensus"."""
    return "narrator" if document["provenance"].get("strategy") == "narrator_read" else "consensus"


def update_book(directory, source, *, config, input_path=None, client_factory, speaker_judge=None,
                announce_cache=lambda path: Path(path).resolve(), progress=print, log=print, strategy=None,
                reread=None, base_document=None):
    """Analyze and publish the new and edited chapters of ``directory``'s book.

    ``strategy`` ("consensus" or "narrator") selects the analysis for the new
    paragraphs; by default the one the book was analyzed with.

    ``reread`` (``{block_id: reason}``, from ``split_check.misread_blocks``)
    instead reads those saved paragraphs again with their neighbours, keeping
    their text and block IDs; ``source`` is then ignored. ``base_document``
    is the saved analysis to build on (default: the published one), such as
    the published one after a model-free tag repair.

    Returns the published document, or ``None`` when the source has no new or
    edited chapters. ``client_factory`` builds the analysis client from
    ``config["provider"]`` only after every local check has passed, and only
    when some paragraph needs the model (removing paragraphs needs none).
    """
    directory = Path(directory).resolve()
    original, registry, _ = _completed_directory(directory)
    if base_document is not None:
        if base_document["source"] != original["source"]:
            raise ValueError("The analysis to build on does not match the saved book's source")
        original = base_document
    if strategy not in (None, "consensus", "narrator"):
        raise ValueError("strategy must be consensus or narrator")
    strategy = strategy or saved_strategy(original)
    if reread:
        comparison = reread_comparison(original["source"], reread)
    else:
        comparison = compare_sources(original["source"], source, preserve_citations_from=original["cast"])
    epub = Path(input_path) if input_path is not None and Path(input_path).suffix.lower() == ".epub" else None
    digest = epub_digest(epub) if epub is not None and not reread else None
    if (comparison["retitled_chapters"] and not comparison["appended_chapters"] and not comparison["edited_chapters"]
            and not comparison["split_chapters"]):
        from .chapter_title_update import retitled_document, title_update_sidecars
        related = ensure_updatable(directory, comparison)
        expected = {str(path): _hash(path / "manifest.json") for path in related}
        document = retitled_document(original, comparison, epub_sha256=digest)
        if document is None:
            raise ValueError("A title-only update requires identical source text, markup, chapter and block "
                             "layout, and extraction metadata; the saved book was not changed")
        sidecars = title_update_sidecars(directory, original, document)
        from .analysis_revision import revise_directories
        result = revise_directories(
            {directory: document}, directories=related, expected_manifest_hashes=expected,
            scope="source-edit", sidecars={directory: sidecars}, progress=log)
        log(f"Original analysis files and revision journal: {result['backup_directory']}")
        if epub is not None:
            from .source_epub import remember_source_epub
            remember_source_epub(directory, epub, document["source"]["book"]["id"], epub_sha256=digest)
        for line in describe_update(document):
            log(line)
        return document
    if not has_changes(comparison) and not reread:
        log("No new or edited chapters: the EPUB's chapters match the saved book. "
            "The saved analysis is unchanged.")
        return None
    from .analysis_progress import (STAGES, analysis_cache_directory, analysis_options_fingerprint,
                                    create_pending, equivalent_options_fingerprints, pending_identity,
                                    resume_document)
    from .audition_scripts import prepare_document_samples
    from .pipeline import CachedRequests, analyze
    from .speaker_decisions import prompt_speaker_decisions
    related = ensure_updatable(directory, comparison)
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
    changes = (f"{len(reread)} paragraph(s) read again" if reread else
               f"{len(comparison['appended_chapters'])} new and "
               f"{len(comparison['edited_chapters'])} edited chapter(s)"
               + (f", {len(comparison['split_chapters'])} chapter(s) divided differently"
                  if comparison["split_chapters"] else ""))
    client = client_factory(**config["provider"]) if delta is not None else None
    if client is not None:
        log(f"Updating {directory} with {changes} using {client.model} at {client.base_url}"
            f"{' (narrator read)' if strategy == 'narrator' else ''}; only "
            + ("those paragraphs" if reread else "the new chapters and the edited paragraphs")
            + " with their neighbours will be sent to this endpoint.")
    else:
        log(f"Updating {directory} with {changes}; no paragraph is new or edited, so no model is needed.")
    model_cache = announce_cache(cache)
    # A narrator-read update never resumes a consensus update's checkpoints.
    update_fingerprint = fingerprint({"book_update": UPDATE_VERSION, "base_manifest_sha256": manifest_hash,
                                      **({"strategy": strategy} if strategy != "consensus" else {}),
                                      **({"reread": comparison["reread_reasons"],
                                          "base": fingerprint(original)} if reread else {})})
    identity = pending_identity(merged, client.identity if client is not None else {"model": None},
                                analysis_options_fingerprint(settings, config["performance_policy"],
                                                             update_fingerprint))
    equivalent = equivalent_options_fingerprints(settings, config["performance_policy"], update_fingerprint)
    workspace = update_workspace(directory)
    resumed = resume_document(workspace, identity, equivalent)
    document, resume_index = None, 0
    if resumed is not None:
        document, next_stage = resumed
        resume_index = len(STAGES) if next_stage is None else STAGES.index(next_stage)
        if resume_index:
            log("Resuming the interrupted book update from its last completed stage.")
    pending = create_pending(workspace, identity, equivalent)
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
            # A re-read may add a speaker the book lacks, also in a book later books continue
            # from: publication appends it and refreshes their series history.
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
        if related == [directory]:
            publish_update(directory, document, expected_manifest_sha256=manifest_hash, progress=log)
        else:
            from .analysis_revision import revise_directories
            result = revise_directories(
                {directory: document}, directories=related,
                expected_manifest_hashes={str(path): _hash(path / "manifest.json") for path in related},
                scope="source-edit", progress=log)
            log(f"Original analysis files and revision journal: {result['backup_directory']}")
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
    comparison = compare_sources(original["source"], source, preserve_citations_from=original["cast"])
    update = {"unchanged_chapters": comparison["unchanged_chapters"],
              "added_chapters": [{"id": chapter["id"], "title": chapter["title"]}
                                 for chapter in comparison["appended_chapters"]],
              "edited_chapters": deepcopy(comparison["edited_chapters"]),
              "retitled_chapters": deepcopy(comparison["retitled_chapters"]),
              "split_chapters": deepcopy(comparison["split_chapters"])}
    delta = delta_source(comparison)
    if delta is None:
        note = ("No new, edited, or retitled chapters" if not has_changes(comparison)
                else "Only chapter titles changed" if comparison["retitled_chapters"] and not comparison["edited_chapters"]
                else "Only chapters or paragraphs were divided differently or removed") + "; analyze would make no model requests."
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


def same_book_epub(saved_source, epub):
    """True when ``epub`` extracts to the same book as the saved source.

    Uses the updater's own comparison: the same extraction settings, no
    removed, inserted or reordered chapters, and not a wholesale rewrite.
    A repacked or typo-corrected revision of the analyzed book qualifies;
    a genuinely different book does not, and neither does an unreadable one.
    The candidate is re-extracted with the saved source's supported
    auxiliary-spine setting; any other policy difference still refuses.
    """
    from .epub import extract_epub
    policy = saved_source.get("extraction_policy")
    include_auxiliary = isinstance(policy, str) and any(
        part.strip() == "auxiliary-spine=include" for part in policy.split(";"))
    try:
        compare_sources(saved_source, extract_epub(epub, include_auxiliary=include_auxiliary))
    except (ValueError, OSError, KeyError):
        return False
    return True


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


def misread_splits(directory, *, chapter_ids=None):
    """``(document, {block_id: reason})``: the saved book after the model-free tag repair, and its misread paragraphs.

    ``chapter_ids`` limits the paragraphs to those chapters (the repair is book-wide and model-free).
    """
    from .narration_boundaries import repair_narration_boundaries
    from .split_check import misread_blocks
    original, _, _ = _completed_directory(Path(directory).resolve())
    repaired = repair_narration_boundaries(original)
    misread = misread_blocks(repaired)
    if chapter_ids is not None:
        chapters = {block["id"]: block["chapter_id"] for block in repaired["source"]["blocks"]}
        wanted = set(chapter_ids)
        misread = {block_id: reason for block_id, reason in misread.items() if chapters[block_id] in wanted}
    return repaired, misread


def reread_misread_splits(directory, config, *, client_factory, speaker_judge=None,
                          announce_cache=lambda path: Path(path).resolve(), progress=print,
                          strategy=None, document=None, misread=None):
    """Read again, with their neighbours, the paragraphs whose saved split contradicts their quotation marks.

    ``document`` and ``misread`` come from ``misread_splits`` (found here when
    omitted). Publishes through ``update_book(reread=...)`` with backups and
    returns how many paragraphs were read again (0 when none needed it).
    """
    if misread is None:
        document, misread = misread_splits(directory)
    if not misread:
        return 0
    print(f"{len(misread)} paragraph(s) are split differently from their quotation marks or italics (speech "
          "read by the narrator, or narration read in a character's voice); reading them again.")
    update_book(directory, document["source"], config=config, strategy=strategy,
                client_factory=client_factory, speaker_judge=speaker_judge,
                announce_cache=announce_cache, progress=progress, reread=misread,
                base_document=document)
    return len(misread)
