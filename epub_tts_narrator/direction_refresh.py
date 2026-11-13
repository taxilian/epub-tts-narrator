"""Direction-only refresh of a published book's performance directions.

Speakers, cast, voices, source text and spelling edits are fixed input: the
model sees each spoken line already labelled with its speaker and writes only
how it is performed. One pass, no jury. The refresh replaces, for the selected
chapters:

- every spoken line's ``delivery`` (dialogue, thought and written text),
- every narration row's ``delivery`` (set only by a rare ``read`` note),
- the delivery, pause and sound controls in ``performance_controls``.

Mid-line changes become style spans and italics in dialogue become emphasis,
both without review (``narrator_read.prekept_direction``). Other notes (paired
tags, pauses, sounds) go through the shared remove-only review. Pronunciation
controls and spelling edits are kept; a row that has pronunciation controls
keeps its new line direction but takes no new in-line notes.

A direction a person typed (``delivery.basis == "user"``) is kept with its
row's controls, and the row takes no new notes.

Every response is cached in the book's own analysis cache, so a rerun replays
completed requests. Publication goes through ``revise_directories`` scope
``attribution`` (backups, descendant rebinding), in the command module.
"""

from copy import deepcopy
import threading

from .direction_policy import VERSION as POLICY, typed
from .narrator_prompts import DIRECTIONS, DIRECTIONS_VERSION, HINTS
from .narrator_read import (MARKER_CLOSE, MARKER_OPEN, MARKUP_CHARS, _attach_notes, _changes, _context, _direction,
                            _item_row, _numbers, _parallel, _parse_note, _render_blocks, _unwrap, chapter_groups,
                            prekept_direction, CONTEXT_AFTER, CONTEXT_BEFORE)
from .performance_plan import text_sha256, validate_performance
from .provider import OutputTruncatedError
from .speech_cues import line_cues, speakers
from .spoken_text import build_spoken


MODE = "direction_refresh"
SPOKEN_KINDS = {"dialogue": "", "thought": ", thought", "letter": ", written"}
REFRESHED_CONTROLS = {"delivery", "pause", "vocalization"}
# Notes the refresh accepts: directions only, never spelling edits or voice changes.
NOTE_KINDS = {"read", "shift", "control"}


def _speaker(segment, names):
    """A line's speaker as the prompt shows it; "?" when it is not settled."""
    group = segment.get("simultaneous_speakers")
    if group:
        label = " & ".join(names.get(member["role_id"], "?") for member in group)
    else:
        label = names.get(segment.get("role_id"), "?")
    label = label.replace(MARKER_OPEN, "").replace(MARKER_CLOSE, "").strip() or "?"
    return label + SPOKEN_KINDS[segment["kind"]]


def _role_names(cast):
    characters = {character["id"]: character.get("name", "") for character in cast.get("characters", [])}
    return {role["id"]: characters.get(role.get("character_id"), "") for role in cast.get("roles", [])
            if role.get("kind") != "narration"}


def lines(document, block_ids=None):
    """Book-order spoken lines as {qid, segment index, block_id, start, label}.

    IDs number every spoken line in the whole book, so a chapter subset asks
    with the same IDs a whole-book run would use.
    """
    wanted = None if block_ids is None else set(block_ids)
    names = _role_names(document["cast"])
    found = []
    number = 0
    for index, segment in enumerate(document["segments"]):
        if segment["kind"] not in SPOKEN_KINDS:
            continue
        number += 1
        if wanted is not None and segment["block_id"] not in wanted:
            continue
        found.append({"qid": f"q{number}", "index": index, "block_id": segment["block_id"],
                      "start": segment["start"], "label": _speaker(segment, names)})
    return found


def _validator(expected, blocks_by_number):
    expected_set = set(expected)

    def validate(response):
        if not isinstance(response, dict):
            raise ValueError("response must be an object")
        rows = _unwrap(response.get("q", []))
        if not isinstance(rows, list):
            raise ValueError('"q" must be an array of [qid, delivery, changes?] arrays')
        deliveries, notes, errors, dropped = {}, [], [], 0
        for row in rows:
            row = _item_row(row, "q")
            if not isinstance(row, list) or not row or not isinstance(row[0], str):
                errors.append("each q entry must be [qid, delivery, ...]")
                continue
            # "q17 Mara" or "‹q17›" both name q17.
            qid = row[0].strip().strip(MARKER_OPEN + MARKER_CLOSE).split(" ")[0]
            if qid not in expected_set:
                errors.append(f"{qid[:20]!r} is not a line id in text")
                continue
            if qid in deliveries:
                errors.append(f"{qid} given twice")
                continue
            deliveries[qid] = _direction(row[1]) if len(row) > 1 else ""
            # Changes follow the delivery; a model that adds a quote slot first
            # (the markup row's shape) still has them read.
            changes = next((found for found in (_changes(item) for item in row[2:4]) if found), [])
            for change in changes:
                note = (_parse_note([qid, "then", change[0], change[1]], expected_set, blocks_by_number)
                        if isinstance(change, list) and len(change) == 2 and isinstance(change[0], str) else None)
                if note is None:
                    dropped += 1
                else:
                    notes.append(note)
        missing = [qid for qid in expected if qid not in deliveries]
        if missing:
            errors.append(f"missing deliveries for {len(missing)} line id(s): {', '.join(missing[:30])}")
        if errors:
            raise ValueError("; ".join(errors[:12]))
        raw_notes = _unwrap(response.get("p", []))
        for row in raw_notes if isinstance(raw_notes, list) else []:
            note = _parse_note(row, expected_set, blocks_by_number)
            if note is None or note["kind"] not in NOTE_KINDS:
                dropped += 1
            else:
                notes.append(note)
        return {"deliveries": deliveries, "notes": notes, "dropped_notes": dropped}

    return validate


def request_directions(document, requests, targets, *, notes="", cues=None, workers=1, group_chars=MARKUP_CHARS,
                       progress=lambda message: None):
    """Ask for every target line's delivery and notes; returns (deliveries, notes, stats).

    ``cues`` maps a qid to the narration that says how its speaker says it; it is
    sent with that line's request so the delivery agrees with the book.
    """
    cues = cues or {}
    source = document["source"]
    numbers = _numbers(source)
    positions = {block["id"]: position for position, block in enumerate(source["blocks"])}
    by_block = {}
    for line in targets:
        by_block.setdefault(line["block_id"], []).append({"id": f"{line['qid']} {line['label']}",
                                                          "start": line["start"]})
    selected = [block for block in source["blocks"] if block["id"] in by_block]
    # Whole chapters per request, as the narrator read's markup groups them;
    # paragraphs without spoken lines stay in for the scene.
    chapters = {block["chapter_id"] for block in selected}
    groups = [group for group in chapter_groups({"blocks": [block for block in source["blocks"]
                                                            if block["chapter_id"] in chapters]}, group_chars)
              if any(block["id"] in by_block for block in group)]
    system, hint = (DIRECTIONS + HINTS, notes.strip()) if notes.strip() else (DIRECTIONS, "")
    lock = threading.Lock()
    done, subdivided = [0], [0]

    def request(group):
        first, last = positions[group[0]["id"]], positions[group[-1]["id"]]
        payload = {"mode": MODE, "v": DIRECTIONS_VERSION,
                   "before": _context(source["blocks"][max(0, first - 20):first], CONTEXT_BEFORE, tail=True),
                   "text": _render_blocks(group, numbers, by_block),
                   "after": _context(source["blocks"][last + 1:last + 21], CONTEXT_AFTER, tail=False)}
        expected = [marker["id"].split(" ")[0] for block in group for marker in by_block.get(block["id"], [])]
        group_cues = {qid: cues[qid] for qid in expected if qid in cues}
        if group_cues:
            payload["cues"] = group_cues
        if hint:
            payload["hints"] = hint
        validate = _validator(expected, {numbers[block["id"]]: block for block in group})
        try:
            return requests.run(system, payload, validate)
        except OutputTruncatedError:
            if len(group) < 2:
                raise
            with lock:
                subdivided[0] += 1
            middle = len(group) // 2
            halves = [request(group[:middle]), request(group[middle:])]
            return {"deliveries": {**halves[0]["deliveries"], **halves[1]["deliveries"]},
                    "notes": halves[0]["notes"] + halves[1]["notes"],
                    "dropped_notes": halves[0]["dropped_notes"] + halves[1]["dropped_notes"]}

    def run(group):
        result = request(group)
        with lock:
            done[0] += 1
            progress(f"Direction refresh: {done[0]}/{len(groups)} chapter group(s) complete")
        return result

    progress(f"Direction refresh: directing {len(targets)} line(s) in {len(groups)} chapter group(s)")
    deliveries, found, dropped = {}, [], 0
    for result in _parallel(run, groups, workers):
        deliveries.update(result["deliveries"])
        found.extend(result["notes"])
        dropped += result["dropped_notes"]
    return deliveries, found, {"groups": len(groups), "subdivided_groups": subdivided[0], "dropped_notes": dropped}


def _strip_controls(segment):
    """Remove refreshed controls; True when the row keeps others (no new notes then)."""
    record = segment.get("performance_controls")
    if record is None:
        return False
    kept = [control for control in record["controls"] if control["kind"] not in REFRESHED_CONTROLS]
    if not kept and not segment.get("speech_normalization", {}).get("edits"):
        segment.pop("performance_controls")
        if segment.get("spoken_text") == segment["text"] and "performance" not in segment:
            # Written only for the controls just removed.
            segment.pop("spoken_text")
        return False
    spoken = build_spoken(segment)[0]
    segment["performance_controls"] = validate_performance(
        {"version": record["version"], "text_sha256": text_sha256(spoken), "controls": kept}, spoken)
    return bool(kept)


def refresh_directions(document, requests, *, block_ids=None, notes="", workers=1, group_chars=MARKUP_CHARS,
                       progress=lambda message: None):
    """Return (revised document, report); the input document is not changed.

    ``block_ids`` limits the refresh to those paragraphs (whole chapters from the
    command); ``notes`` are optional operator notes sent as fallible hints.
    """
    from .attribution_performance import review_inline_performance

    revised = deepcopy(document)
    segments = revised["segments"]
    selected = set(block_ids) if block_ids is not None else {block["id"] for block in revised["source"]["blocks"]}
    # A person's typed directions stay; their lines are not asked about.
    kept = {id(segment) for segment in segments if segment["block_id"] in selected and typed(segment)}
    targets = [line for line in lines(revised, selected) if id(segments[line["index"]]) not in kept]
    before = {segment["id"]: (segment["delivery"]["direction"], deepcopy(segment.get("performance_controls")))
              for segment in segments if segment["block_id"] in selected}
    people = speakers(revised["cast"])
    block_rows = {}
    for segment in segments:
        if segment["block_id"] in selected:
            block_rows.setdefault(segment["block_id"], []).append(segment)
    texts = {block["id"]: block["text"] for block in revised["source"]["blocks"]}
    cues = {}
    for block_id, rows in block_rows.items():
        cues.update(line_cues(rows, texts[block_id], people))
    line_cue_text = {line["qid"]: cues[id(segments[line["index"]])]["clause"] for line in targets
                     if id(segments[line["index"]]) in cues}
    deliveries, found, request_stats = request_directions(revised, requests, targets, notes=notes, cues=line_cue_text,
                                                          workers=workers, group_chars=group_chars, progress=progress)
    by_qid = {line["qid"]: segments[line["index"]] for line in targets}
    rows_by_block, locked = {}, set()
    for segment in segments:
        if segment["block_id"] not in selected:
            continue
        if id(segment) in kept:
            locked.add(id(segment))
        elif segment["kind"] in SPOKEN_KINDS or segment["kind"] == "narration":
            segment["delivery"] = {"direction": "", "evidence": []}
            if _strip_controls(segment):
                locked.add(id(segment))
        rows_by_block.setdefault(segment["block_id"], []).append(segment)
    for line in targets:
        segments[line["index"]]["delivery"] = {"direction": deliveries.get(line["qid"], ""), "evidence": []}
    stats = {"attached": 0, "dropped": 0, "delivery": 0, "shifts": 0, "narration_reads": 0, "italic_emphasis": 0,
             "cue_sounds": 0, "cue_scoped": 0, "cue_tones": 0}
    notes_by_block = {}
    qid_block = {line["qid"]: line["block_id"] for line in targets}
    for note in found:
        kind, value = note["target"]
        notes_by_block.setdefault(qid_block[value] if kind == "q" else value, []).append(note)
    blocks = {block["id"]: block for block in revised["source"]["blocks"]}
    for block_id, rows in rows_by_block.items():
        block = blocks[block_id]
        if block_id not in notes_by_block and not block.get("inline_spans") and not any(id(row) in cues for row in rows):
            continue
        # Angle brackets in a row's text can be kept speech in older books; leave those rows' notes out.
        open_rows = [row for row in rows if id(row) not in locked and not set("<>") & set(row["text"])]
        open_ids = {id(row) for row in open_rows}
        _attach_notes(open_rows, notes_by_block.get(block_id, []), block,
                      {qid: row for qid, row in by_qid.items() if id(row) in open_ids}, stats,
                      {key: value for key, value in cues.items() if key in open_ids})
    # The shared review writes its own audit record; keep the analysis audit and
    # file this refresh's beside it.
    provenance = revised.setdefault("provenance", {})
    saved_audit = provenance.pop("inline_performance", None)
    if any("inline_proposals" in segment for segment in segments):
        revised = review_inline_performance(revised, requests, workers=min(5, workers), progress=progress,
                                            prekept=prekept_direction)
    provenance = revised["provenance"]
    review = provenance.pop("inline_performance", None)
    if saved_audit is not None:
        provenance["inline_performance"] = saved_audit
    changed = [segment["id"] for segment in revised["segments"] if segment["id"] in before
               and before[segment["id"]] != (segment["delivery"]["direction"],
                                             segment.get("performance_controls"))]
    directed = sum(1 for line in targets if revised["segments"][line["index"]]["delivery"]["direction"])
    report = {"version": DIRECTIONS_VERSION, "policy": POLICY, "lines": len(targets), "directed": directed,
              "narration_reads": stats["narration_reads"], "changes": stats["shifts"],
              "italic_emphasis": stats["italic_emphasis"], "notes_attached": stats["attached"],
              "cued_lines": len(line_cue_text), "cue_sounds": stats["cue_sounds"], "cue_scoped": stats["cue_scoped"],
              "cue_tones": stats["cue_tones"],
              "notes_dropped": stats["dropped"] + request_stats["dropped_notes"],
              "kept_pronunciation_rows": len(locked - kept), "kept_typed_directions": len(kept),
              "changed_segments": len(changed),
              "groups": request_stats["groups"], "subdivided_groups": request_stats["subdivided_groups"],
              "review": review["counts"] if review else {}, "hints": bool(notes.strip())}
    return revised, report
