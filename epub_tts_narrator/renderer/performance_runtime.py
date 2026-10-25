"""Bind reviewed performance plans to the same source chunks used by rendering."""

from copy import deepcopy
from pathlib import Path

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


def _with_normalization_directions(row, record):
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


def row_performance(row, records):
    if records is None:
        return None
    if row["id"] not in records:
        raise ValueError("Performance plan is missing a selected source segment")
    return _with_normalization_directions(row, records[row["id"]])


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


def bind_chunk_performance(chunk, record):
    if record is None:
        return
    row = chunk["segment"]
    spoken = build_spoken(row)[0]
    chunk["expressive_performance"] = (slice_performance(
        record, spoken, chunk["spoken_start"], chunk["spoken_end"])
        if spoken else validate_performance(record, spoken))
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
        raise ValueError("This TTS backend does not support structured performance plans; use legacy mode or a supported Drama 3 adapter")


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
