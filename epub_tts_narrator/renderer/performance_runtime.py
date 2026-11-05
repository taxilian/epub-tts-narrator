"""Bind reviewed performance plans to the same source chunks used by rendering."""

from copy import deepcopy
from pathlib import Path
import json

from ..performance_plan import slice_performance, text_sha256, validate_performance
from ..spoken_text import build_spoken, effective_delivery


def load_render_performance(bundle, settings, chapters):
    mode = settings.get("performance_mode", "legacy")
    if mode == "legacy":
        return None
    if mode != "expressive":
        raise ValueError("render.performance_mode must be legacy or expressive")
    path = settings.get("performance_plan") or Path(bundle["directory"]) / "performance-plan.json"
    if not Path(path).exists():
        raise ValueError("Expressive rendering requires a prepared performance plan; run prepare-performance first")
    from ..performance_planning import load_prepared_performance
    return load_prepared_performance(bundle, path, chapters=sorted(chapters))["segments"]


def _with_normalization_directions(row, record, *, suppressed_normalizations=()):
    """Scope approved spelling-repair directions to their spoken replacements.

    Earlier renderers applied every spelling hint to the entire passage. Split
    only delivery ranges at exact replacement boundaries, preserving all words,
    pronunciation controls and the original source-to-spoken mapping.
    """
    text, mapping = build_spoken(row)
    record = validate_performance(record, text)
    edits = row.get("speech_normalization", {}).get("edits", [])
    hints = []
    for edit in edits:
        if (edit["start"], edit["end"]) in suppressed_normalizations:
            continue
        direction = edit.get("direction", "").strip()
        if not direction:
            continue
        for span in mapping:
            if (span["kind"] == "replacement"
                    and span["source_start"] == row["start"] + edit["start"]
                    and span["source_end"] == row["start"] + edit["end"]):
                hints.append({"kind": "delivery", "tag": "style", "direction": direction,
                              "start": span["spoken_start"], "end": span["spoken_end"]})
    if not hints:
        return record
    delivery = [c for c in record["controls"] if c["kind"] == "delivery"]
    boundaries = sorted({point for c in delivery + hints for point in (c["start"], c["end"])})
    merged = []
    for a, b in zip(boundaries, boundaries[1:]):
        active = [c for c in delivery + hints if c["start"] <= a and b <= c["end"]]
        if not active:
            continue
        if len(active) == 1:
            control = dict(active[0], start=a, end=b, text=text[a:b])
        else:
            descriptions = [c["direction"] if c["tag"] == "style" else c["tag"] for c in active]
            control = {"kind": "delivery", "tag": "style", "start": a, "end": b,
                       "text": text[a:b], "direction": "; ".join(dict.fromkeys(descriptions))}
        merged.append(control)
    record["controls"] = [c for c in record["controls"] if c["kind"] != "delivery"] + merged
    return validate_performance(record, text)


def _inline_performance(row):
    """Validate the optional analyzer inline controls bound to this row's spoken text."""
    record = row.get("performance_controls")
    if record is None:
        return None
    try:
        return validate_performance(record, build_spoken(row)[0])
    except ValueError as exc:
        raise ValueError(f"Segment {row['id']} has invalid inline performance controls: {exc}") from exc


def _merge_performance_records(row, inline, external):
    """Merge inline attribution controls with a plan record for the same row.

Exact duplicate controls deduplicate. Same-kind overlapping spans and
differing events at one position fail with the row ID; neither source is
silently preferred or dropped.
"""
    spoken = build_spoken(row)[0]
    controls, seen = [], set()
    for record, label in ((inline, "inline"), (external, "plan")):
        try:
            canonical = validate_performance(record, spoken)
        except ValueError as exc:
            raise ValueError(f"Segment {row['id']} has invalid {label} performance controls: {exc}") from exc
        for control in canonical["controls"]:
            key = json.dumps(control, sort_keys=True)
            if key not in seen:
                seen.add(key)
                controls.append(control)
    for kind in ("delivery", "pronunciation"):
        spans = sorted((c for c in controls if c["kind"] == kind), key=lambda c: (c["start"], c["end"]))
        for left, right in zip(spans, spans[1:]):
            if left["end"] > right["start"]:
                raise ValueError(f"Segment {row['id']} has conflicting {kind} controls between "
                                 "inline attribution and the performance plan")
    positions = {}
    for control in controls:
        if "offset" not in control:
            continue
        prior = positions.setdefault((control["kind"], control["offset"]), control)
        if prior != control:
            raise ValueError(f"Segment {row['id']} has conflicting {control['kind']} events at one "
                             "position between inline attribution and the performance plan")
    try:
        merged = {"version": 1, "text_sha256": text_sha256(spoken), "controls": controls}
        return validate_performance(merged, spoken)
    except ValueError as exc:
        raise ValueError(f"Segment {row['id']} inline and plan performance controls cannot combine: {exc}") from exc


def row_performance(row, records, *, normalization=True):
    inline = _inline_performance(row)
    if records is None:
        record = inline
    else:
        if row["id"] not in records:
            raise ValueError("Performance plan is missing a selected source segment")
        external = records[row["id"]]
        record = _merge_performance_records(row, inline, external) if inline is not None else external
    if record is None:
        return None
    if normalization:
        return _with_normalization_directions(row, record)
    return validate_performance(record, build_spoken(row)[0])


def protected_spans(text, spans, record, maximum):
    """Do not split a phoneme-controlled word across independent requests."""
    if record is None:
        return spans
    protected = [c for c in record["controls"] if c["kind"] == "pronunciation"]
    boundaries = {0, len(text)}
    for _, end in spans:
        for control in protected:
            if control["end"] - control["start"] > maximum:
                raise ValueError("A pronunciation span exceeds render.max_chunk_chars")
            if control["start"] < end < control["end"]:
                boundaries.add(control["start"])
                end = control["end"]
        boundaries.add(end)
    ordered = sorted(boundaries)
    return list(zip(ordered, ordered[1:])) or [(0, 0)]


def bind_chunk_performance(chunk, record, *, sparse=False):
    if record is None:
        return
    row = chunk["segment"]
    spoken = build_spoken(row)[0]
    scoped = (slice_performance(
        record, spoken, chunk["spoken_start"], chunk["spoken_end"])
        if spoken else validate_performance(record, spoken))
    # Sparse legacy binding keeps control-free chunks byte-identical to plans
    # without a sidecar. Any scoped control binds, not only pronunciations, so
    # inline attribution cues reach synthesis in the default mode too.
    if sparse and not scoped["controls"]:
        return
    chunk["expressive_performance"] = scoped
    scene = deepcopy(row)
    scene.pop("speech_normalization", None)
    chunk["expressive_direction"] = effective_delivery(scene)


def bind_traits(plan):
    for chapter in plan["chapters"]:
        for chunk in chapter["chunks"]:
            for turn in chunk.get("paragraph_turns", [chunk]):
                if "expressive_performance" in turn:
                    turn["expressive_traits"] = deepcopy(turn.get("speech_traits", {}))
    return plan


def is_expressive(chunk):
    return "expressive_performance" in chunk or any(
        "expressive_performance" in turn for turn in chunk.get("paragraph_turns", []))


def synthesis_direction(chunk, voice_id):
    if "expressive_performance" in chunk:
        return chunk["expressive_direction"]
    from .speech_traits import performance_direction
    return performance_direction(chunk, voice_id)


def require_engine_support(plan, engine):
    if (any(is_expressive(chunk) for chapter in plan["chapters"] for chunk in chapter["chunks"])
            and not getattr(engine, "capabilities", {}).get("structured_performance", False)):
        raise ValueError("This TTS backend does not support structured performance or pronunciation controls; select a supported Drama 3 adapter")


def verification_options(request):
    chunk = request.get("chunk", {})
    if not is_expressive(chunk):
        return {}
    if "paragraph_turns" not in chunk:
        return {"performance": validate_performance(chunk["expressive_performance"], chunk["spoken_text"])}
    from ..performance_plan import combine_performances
    parts = [(turn["spoken_text"], turn.get("expressive_performance") or {
        "version": 1, "text_sha256": text_sha256(turn["spoken_text"]), "controls": []})
        for turn in chunk["paragraph_turns"]]
    return {"performance": combine_performances(parts, separator="")}
