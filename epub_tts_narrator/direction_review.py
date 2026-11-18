"""Rerunning ``analyze BOOK`` reviews every spoken line's direction and corrects the wrong ones.

Unlike the direction refresh (``direction_refresh``), which directs every
line again, this asks a reviewer to look at each line's current direction in
its scene (whole chapters per request, the narration's cues beside their
lines) and to give a corrected direction only for lines whose direction is
wrong (``rerun_review_prompts.DIRECTION_CHECK``). Only those lines change, so
only their audio is rendered again. Directions a person typed are never sent
or changed.

Each reviewed paragraph is recorded in ``provenance.direction_checks`` by its
text, split and directions, so later reruns review only paragraphs that
changed since (or everything again when ``VERSION`` or the prompt changes).
Answers are cached in the book's request cache. Publication goes through
``revise_directories`` scope ``attribution`` with backups.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from .rerun_review_prompts import DIRECTION_REVIEW_VERSION

VERSION = f"1:{DIRECTION_REVIEW_VERSION}"
MODE = "direction_review"


def _state(block, rows):
    return hashlib.sha256(json.dumps([block["text"], [
        [row["start"], row["end"], row["kind"], row.get("role_id"),
         sorted(member["role_id"] for member in row.get("simultaneous_speakers", [])),
         row["delivery"]["direction"], row.get("performance_controls")] for row in rows]],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def block_states(document):
    rows = {}
    for segment in document["segments"]:
        rows.setdefault(segment["block_id"], []).append(segment)
    return {block["id"]: _state(block, rows.get(block["id"], [])) for block in document["source"]["blocks"]}


def unreviewed_blocks(document):
    """Paragraphs with a spoken line not reviewed at this version since they last changed."""
    from .direction_refresh import SPOKEN_KINDS
    record = document["provenance"].get("direction_checks")
    checked = record.get("blocks", {}) if isinstance(record, dict) and record.get("version") == VERSION else {}
    spoken = {segment["block_id"] for segment in document["segments"] if segment["kind"] in SPOKEN_KINDS}
    return [block_id for block_id, state in block_states(document).items()
            if block_id in spoken and checked.get(block_id) != state]


def record_reviews(document, block_ids):
    record = document["provenance"].get("direction_checks")
    blocks = dict(record.get("blocks", {})) if isinstance(record, dict) and record.get("version") == VERSION else {}
    states = block_states(document)
    blocks.update({block_id: states[block_id] for block_id in block_ids})
    document["provenance"]["direction_checks"] = {"version": VERSION, "blocks": dict(sorted(blocks.items()))}


def review_directions(document, requests, block_ids, *, workers=1, progress=lambda message: None):
    """``(revised document, corrected segment IDs)``: wrong directions in ``block_ids`` replaced."""
    from .direction_policy import typed
    from .direction_refresh import lines
    from .narrator_read import (CONTEXT_AFTER, CONTEXT_BEFORE, MARKUP_CHARS, _context, _direction, _item_row,
                                _numbers, _parallel, _render_blocks, _unwrap, chapter_groups)
    from .rerun_review_prompts import DIRECTION_CHECK
    from .speech_cues import line_cues, speakers
    revised = deepcopy(document)
    segments = revised["segments"]
    selected = set(block_ids)
    targets = [line for line in lines(revised, selected) if not typed(segments[line["index"]])]
    if not targets:
        return revised, []
    source = revised["source"]
    numbers = _numbers(source)
    positions = {block["id"]: position for position, block in enumerate(source["blocks"])}
    texts = {block["id"]: block["text"] for block in source["blocks"]}
    rows_by_block = {}
    for segment in segments:
        if segment["block_id"] in selected:
            rows_by_block.setdefault(segment["block_id"], []).append(segment)
    people = speakers(revised["cast"])
    cues = {}
    for block_id, rows in rows_by_block.items():
        cues.update(line_cues(rows, texts[block_id], people))
    by_block = {}
    for line in targets:
        by_block.setdefault(line["block_id"], []).append({"id": f"{line['qid']} {line['label']}", "start": line["start"]})
    chapters = {block["chapter_id"] for block in source["blocks"] if block["id"] in by_block}
    groups = [group for group in chapter_groups({"blocks": [block for block in source["blocks"]
                                                            if block["chapter_id"] in chapters]}, MARKUP_CHARS)
              if any(block["id"] in by_block for block in group)]
    by_qid = {line["qid"]: segments[line["index"]] for line in targets}

    def run(group):
        first, last = positions[group[0]["id"]], positions[group[-1]["id"]]
        expected = [marker["id"].split(" ")[0] for block in group for marker in by_block.get(block["id"], [])]
        payload = {"mode": MODE, "v": DIRECTION_REVIEW_VERSION,
                   "before": _context(source["blocks"][max(0, first - 20):first], CONTEXT_BEFORE, tail=True),
                   "text": _render_blocks(group, numbers, by_block),
                   "after": _context(source["blocks"][last + 1:last + 21], CONTEXT_AFTER, tail=False),
                   "d": {qid: by_qid[qid]["delivery"]["direction"] for qid in expected}}
        group_cues = {qid: cues[id(by_qid[qid])]["clause"] for qid in expected if id(by_qid[qid]) in cues}
        if group_cues:
            payload["cues"] = group_cues
        allowed = set(expected)

        def validate(response):
            rows = _unwrap(response.get("q")) if isinstance(response, dict) else None
            if not isinstance(rows, list):
                raise ValueError('response must be {"q":[[qid,delivery]]}')
            result = {}
            for row in rows:
                row = _item_row(row, "q")
                if not (isinstance(row, list) and len(row) >= 2 and isinstance(row[0], str)):
                    raise ValueError("each q entry must be [qid, delivery]")
                qid = row[0].strip().strip("‹›").split(" ")[0]
                if qid not in allowed:
                    raise ValueError(f"{qid[:20]!r} is not a line id in text")
                result[qid] = _direction(row[1])
            return result
        return requests.run(DIRECTION_CHECK, payload, validate)

    progress(f"Direction review: reviewing {len(targets)} line(s) in {len(groups)} chapter group(s)")
    corrections = {}
    for result in _parallel(run, groups, workers):
        corrections.update(result)
    changed = []
    for qid, direction in corrections.items():
        segment = by_qid[qid]
        if direction and direction != segment["delivery"]["direction"]:
            segment["delivery"] = {"direction": direction, "evidence": []}
            changed.append(segment["id"])
    return revised, changed


def review_book_directions(directory, config, client_factory, progress):
    """``analyze BOOK``: review the directions of every paragraph not reviewed since it changed; True when published."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, reserve_revision_scope, revise_directories
    from .pipeline import CachedRequests
    directory = Path(directory).resolve()
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    targets = unreviewed_blocks(document)
    if not targets:
        print("Line directions: every spoken line was already reviewed and has not changed since.")
        return False
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("Rerunning analyze reviews line directions; set provider.model in epub-tts-narrator.toml, "
                         "or select a TOML file with --config")
    print(f"Reviewing line directions in {len(targets)} paragraph(s) not reviewed since they last changed; only "
          "wrong directions are replaced, and directions you typed are kept.")
    with ExitStack() as stack:
        reserved = stack.enter_context(reserve_revision_scope([directory]))
        client = client_factory(config["provider"])
        print(f"Reviewing directions with {client.model} at {client.base_url}; book passages will be sent to this "
              "endpoint.", file=sys.stderr)
        requests = CachedRequests(client, analysis_cache_directory(directory, settings["cache_dir"]),
                                  repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        revised, changed = review_directions(document, requests, targets,
                                             workers=min(5, settings["attribution_workers"]), progress=progress)
        record_reviews(revised, targets)
        try:
            directories = related_directories([directory])
            hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                      for path in directories}
            if hashes[str(directory)] != original_manifest_hash:
                raise ValueError(f"Analysis changed while its directions were being reviewed: {directory}")
            result = revise_directories({directory: revised}, directories=directories,
                                        expected_manifest_hashes=hashes, scope="attribution", progress=print,
                                        reserved_update_locks=reserved or None)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Direction review publication did not succeed: {exc} Its answers are cached; rerun "
                             "analyze to publish them without new requests.") from exc
    print(f"Line directions reviewed: {len(changed)} corrected; their audio renders again on the next render. "
          f"Original analysis files and revision journal: {result['backup_directory']}")
    return True
