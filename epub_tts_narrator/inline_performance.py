"""Source-bound projection of approved inline speech metadata onto new segments.

Approved ``speech_normalization`` edits and ``performance_controls`` records bind
to exact source spans and to the approved spoken form :func:`build_spoken` derives
from them. Later attribution passes re-slice blocks (bracket delivery,
narration-boundary repairs, semantic attribution corrections, speaker completion)
and model responses may never carry inline fields. :func:`project_segments`
re-attaches the originals' approved metadata to the new rows of the same
immutable source span: an edit survives only when one new row cleanly contains
it, a delivery style splits at a new boundary, and a position event lands exactly
once (an event on a boundary belongs to the following piece; the final end
belongs to the final piece). Metadata a boundary cuts through, or whose position
would have to be guessed inside an approved replacement, is dropped and recorded
in a bounded audit instead of approximated. Reserved bracket ``performance``
metadata and ``spoken_text`` remain owned by ``apply_bracket_delivery``; stored
``performance_controls`` records take ``validate_performance`` canonical form.
"""

from copy import deepcopy

from .performance_plan import PERFORMANCE_VERSION, text_sha256, validate_performance
from .spoken_text import build_spoken


MAX_DROPS_RECORDED = 64
_ROW_KEYS = ("block_id", "start", "end", "text")


class _DropAudit:
    """Collect bounded, text-free drop records so a loss is never silent."""

    def __init__(self, drops):
        if drops is not None and not isinstance(drops, list):
            raise ValueError("drops must be a list or None")
        self.drops = drops
        self.omitted = 0

    def add(self, block_id, kind, start, end, reason):
        if self.drops is None:
            return
        if len(self.drops) >= MAX_DROPS_RECORDED:
            self.omitted += 1
            return
        self.drops.append({"block_id": block_id, "kind": kind,
                           "start": start, "end": end, "reason": reason})

    def finish(self):
        if self.drops is not None and self.omitted:
            self.drops.append({"block_id": None, "kind": "audit", "start": None, "end": None,
                               "reason": f"{self.omitted} additional drop(s) not recorded"})


def _point_source(offset, spans):
    """Map one approved row's spoken offset onto an absolute source offset."""
    for span in spans:
        if span["spoken_start"] <= offset < span["spoken_end"]:
            break
    else:
        if not spans or offset != spans[-1]["spoken_end"]:
            return None
        span = spans[-1]
    if span["kind"] == "source":
        return span["source_start"] + offset - span["spoken_start"]
    if offset == span["spoken_start"]:
        return span["source_start"]
    if offset == span["spoken_end"]:
        return span["source_end"]
    return None


def _range_source(start, end, spans):
    bounds = _point_source(start, spans)
    final = _point_source(end, spans)
    return None if bounds is None or final is None or final < bounds else (bounds, final)


def _interval_spoken(start, end, spans):
    """Map an absolute source range onto one row's spoken bounds; None when ambiguous."""
    pieces = []
    for span in spans:
        if span["kind"] == "source":
            low, high = max(start, span["source_start"]), min(end, span["source_end"])
            if low < high:
                pieces.append((span["spoken_start"] + low - span["source_start"],
                               span["spoken_start"] + high - span["source_start"]))
        elif span["source_start"] >= start and span["source_end"] <= end:
            pieces.append((span["spoken_start"], span["spoken_end"]))
        elif start < span["source_end"] and span["source_start"] < end:
            return None
    if not pieces:
        return None
    if any(left_end != right_start for (_, left_end), (right_start, _) in zip(pieces, pieces[1:])):
        return None
    return pieces[0][0], pieces[-1][1]


def _point_spoken(point, spans):
    """Map an absolute source point onto one row's spoken offset; None when ambiguous."""
    for span in spans:
        if span["kind"] == "source":
            if span["source_start"] <= point < span["source_end"]:
                return span["spoken_start"] + point - span["source_start"]
        elif point == span["source_start"]:
            return span["spoken_start"]
        elif point == span["source_end"]:
            return span["spoken_end"]
    if spans and point == spans[-1]["source_end"] and spans[-1]["kind"] == "source":
        return spans[-1]["spoken_end"]
    return None


def _annotations(row, block_id, audit):
    """Convert one approved row into absolute source-bound edit/control annotations."""
    spoken, spans = build_spoken(row)
    result = []
    record = row.get("speech_normalization")
    if record:
        for edit in record["edits"]:
            result.append({"kind": "edit", "start": row["start"] + edit["start"],
                           "end": row["start"] + edit["end"], "record_version": record["version"],
                           "edit": {key: deepcopy(edit[key])
                                    for key in ("original", "replacement", "direction", "reason")}})
    if "performance_controls" in row:
        for control in validate_performance(row["performance_controls"], spoken)["controls"]:
            payload = {key: deepcopy(value) for key, value in control.items()}
            if "offset" in control:
                point = _point_source(control["offset"], spans)
                if point is None:
                    audit.add(block_id, control["kind"], row["start"], row["end"],
                              "position falls inside an approved replacement; no source-bound anchor exists")
                    continue
                result.append({"kind": control["kind"], "start": point, "end": point, "control": payload})
                continue
            bounds = _range_source(control["start"], control["end"], spans)
            if bounds is None:
                audit.add(block_id, control["kind"], row["start"], row["end"],
                          "boundaries fall inside an approved replacement; no source-bound anchor exists")
                continue
            result.append({"kind": control["kind"], "start": bounds[0], "end": bounds[1], "control": payload})
    return result


def _row_controls(index, new_rows, spoken, spans, annotations, block_id, audit):
    """Build this row's canonical controls; split styles, place events exactly once."""
    row_start, row_end = new_rows[index]["start"], new_rows[index]["end"]
    last = index == len(new_rows) - 1
    controls = []
    for annotation in annotations:
        control = annotation["control"]
        if "offset" in control:
            point = annotation["start"]
            if not (row_start <= point < row_end or last and point == row_end):
                continue
            offset = _point_spoken(point, spans)
            if offset is None:
                audit.add(block_id, control["kind"], point, point,
                          "position falls inside an approved replacement of the new segmentation")
                continue
            controls.append({**{key: deepcopy(value) for key, value in control.items() if key != "offset"},
                             "offset": offset})
            continue
        start, end = max(annotation["start"], row_start), min(annotation["end"], row_end)
        if start >= end:
            continue
        interval = _interval_spoken(start, end, spans)
        if interval is None:
            audit.add(block_id, control["kind"], start, end,
                      "this fragment has no exact position in the approved spoken text")
            continue
        left, right = interval
        if control["kind"] == "pronunciation" and (
                spoken[left:right] != control["text"]
                or left and spoken[left - 1].isalnum() or right < len(spoken) and spoken[right].isalnum()):
            audit.add(block_id, "pronunciation", start, end,
                      "the approved spoken word is not preserved at this position")
            continue
        item = {key: deepcopy(value) for key, value in control.items()
                if key not in {"start", "end", "text"}}
        item.update(start=left, end=right, text=spoken[left:right])
        controls.append(item)
    return controls


def _require_tiling(rows, label, block_id):
    cursor = rows[0]["start"]
    for row in rows:
        if (type(row["start"]) is not int or type(row["end"]) is not int
                or row["start"] != cursor or row["end"] <= row["start"]
                or not isinstance(row["text"], str) or len(row["text"]) != row["end"] - row["start"]):
            raise ValueError(f"{label} segments of block {block_id!r} must tile their source span in order")
        cursor = row["end"]


def _project_block(old_rows, new_rows, block_id, audit):
    _require_tiling(old_rows, "original", block_id)
    _require_tiling(new_rows, "new", block_id)
    if (old_rows[0]["start"], old_rows[-1]["end"]) != (new_rows[0]["start"], new_rows[-1]["end"]):
        raise ValueError(f"Projection requires identical source bounds in block {block_id!r}")
    if "".join(row["text"] for row in old_rows) != "".join(row["text"] for row in new_rows):
        raise ValueError(f"Projection requires identical canonical source text in block {block_id!r}")
    inline_binding = any("performance_controls" in row for row in old_rows)
    annotations = sorted((item for row in old_rows for item in _annotations(row, block_id, audit)),
                         key=lambda item: (item["start"], item["end"]))
    contained = lambda item: any(row["start"] <= item["start"] and item["end"] <= row["end"]
                                 for row in new_rows)
    edits, dropped_ranges, spans_and_events = [], [], []
    for annotation in annotations:
        if annotation["kind"] == "edit":
            if contained(annotation):
                edits.append(annotation)
            else:
                dropped_ranges.append((annotation["start"], annotation["end"]))
                audit.add(block_id, "edit", annotation["start"], annotation["end"],
                          "a new segment boundary cuts this edit; the canonical words remain")
        elif annotation["kind"] == "pronunciation" and not contained(annotation):
            audit.add(block_id, "pronunciation", annotation["start"], annotation["end"],
                      "a new segment boundary cuts this pronunciation")
        else:
            spans_and_events.append(annotation)
    for item in [row for row in spans_and_events
                 if any((row["start"] < end and start < row["end"])
                        or (row["start"] == row["end"] and start <= row["start"] <= end)
                        for start, end in dropped_ranges)]:
        spans_and_events.remove(item)
        audit.add(block_id, item["kind"], item["start"], item["end"],
                   "its approved spelling was dropped with a cut edit")
    output = []
    for index, row in enumerate(new_rows):
        row = deepcopy(row)
        row.pop("speech_normalization", None)
        row.pop("performance_controls", None)
        row_edits = [item for item in edits
                     if row["start"] <= item["start"] and item["end"] <= row["end"]]
        if row_edits:
            row["speech_normalization"] = {
                "version": row_edits[0]["record_version"],
                "edits": [{"start": item["start"] - row["start"], "end": item["end"] - row["start"],
                           **item["edit"]} for item in row_edits]}
            # Bracket-only spoken text predates the attached replacements.
            row.pop("spoken_text", None)
        spoken, spans = build_spoken(row)
        if row_edits:
            # The contract requires the derived spoken form beside approved edits.
            row["spoken_text"] = spoken
        controls = _row_controls(index, new_rows, spoken, spans, spans_and_events, block_id, audit)
        if controls or row_edits and inline_binding:
            row["performance_controls"] = validate_performance(
                {"version": PERFORMANCE_VERSION, "text_sha256": text_sha256(spoken), "controls": controls},
                spoken)
        output.append(row)
    return output


def _rows(rows, label):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} rows must be a list of segment objects")
    if any(key not in row for row in rows for key in _ROW_KEYS):
        raise ValueError(f"{label} rows require block_id, start, end and text")
    return rows


def project_segments(original_rows, new_rows, *, drops=None):
    """Return detached ``new_rows`` carrying the originals' approved inline metadata.

    ``original_rows`` and ``new_rows`` are segment lists that each tile the same
    source blocks in document order. Rows of blocks without any approved
    ``speech_normalization`` or ``performance_controls`` pass through unchanged.
    For an annotated block, surviving edits are re-attached to the single new row
    containing them, controls are rebased onto that row's approved spoken text,
    and rows without surviving metadata lose the fields entirely. The function is
    pure; supply ``drops`` as a list to receive a bounded audit (at most
    ``MAX_DROPS_RECORDED`` text-free records plus an overflow marker) of every
    dropped edit or control. Raises ``ValueError`` on malformed rows, non-tiling
    input, changed source bounds, or stale approved records.
    """
    audit = _DropAudit(drops)
    _rows(original_rows, "original")
    _rows(new_rows, "new")
    annotated_blocks = {row["block_id"] for row in original_rows
                        if "speech_normalization" in row or "performance_controls" in row}
    if not annotated_blocks:
        audit.finish()
        return [deepcopy(row) for row in new_rows]
    grouped = {}
    for row in new_rows:
        grouped.setdefault(row["block_id"], []).append(row)
    missing = annotated_blocks - grouped.keys()
    if missing:
        raise ValueError("Projection requires new segments for annotated block(s): "
                         + ", ".join(repr(value) for value in sorted(missing)))
    result, placed = [], set()
    for row in new_rows:
        block_id = row["block_id"]
        if block_id not in annotated_blocks:
            result.append(deepcopy(row))
            continue
        if block_id in placed:
            continue
        placed.add(block_id)
        result.extend(_project_block([item for item in original_rows if item["block_id"] == block_id],
                                     grouped[block_id], block_id, audit))
    audit.finish()
    return result
