"""Project immutable pronunciation controls onto approved clip repairs.

The repair model authors plain text, delivery and clip-local traits. Confirmed
pronunciation controls stay runtime-owned: they are rebound only to the exact
controlled word occurrences, in source order, and any ambiguity or unsupported
control rejects the plan instead of approximating it.
"""

from copy import deepcopy
import re

from ..performance_plan import PERFORMANCE_VERSION, text_sha256, validate_performance


_WORD_START = r"(?<![^\W_])"
_WORD_END = r"(?![^\W_])"
_WORD_TOKEN = _WORD_START + r"[^\W_]+(?:['’\-‐‑][^\W_]+)*" + _WORD_END


def source_performance_records(request):
    """Return the exact structured records carried by a speech request."""
    chunk = request.get("chunk", {}) if isinstance(request, dict) else {}
    if request.get("kind") == "speech":
        return [chunk["expressive_performance"]] if "expressive_performance" in chunk else []
    if request.get("kind") == "paragraph_speech":
        return [turn["expressive_performance"] for turn in chunk.get("paragraph_turns", [])
                if isinstance(turn, dict) and "expressive_performance" in turn]
    return []


def repair_traits(request):
    """Return the original habitual-voice traits for a single-speech repair."""
    if request.get("kind") != "speech":
        return ""
    from .performance_compiler import performance_kwargs

    chunk = request["chunk"]
    performer = request.get("performer")
    voice_id = performer["voice_id"] if performer else chunk["segment"]["voice_id"]
    return performance_kwargs(chunk, voice_id).get("traits", "")


def _word_tokens(text):
    return [(match.group(), match.start(), match.end())
            for match in re.finditer(_WORD_TOKEN, text)]


def controlled_repair_guidance(request):
    """Model-visible repair constraints for a controlled request, else None.

    Only a request carrying confirmed pronunciation controls gains this
    machine-consumed contract; plain requests keep their legacy payload bytes.
    """
    records = source_performance_records(request)
    if not records or request.get("kind") != "speech":
        return None
    canonical = validate_performance(records[0], request["chunk"]["spoken_text"])
    protected = [control["text"] for control in canonical["controls"]
                 if control["kind"] == "pronunciation"]
    if not protected:
        return None
    return {
        "version": 1,
        "word_exact": True,
        "protected_spans": protected,
        "allowed_changes": ["punctuation outside protected spans", "delivery", "speech_traits",
                            "part boundaries"],
        "forbidden_changes": ["word spelling", "word order", "word case", "adding words",
                              "dropping words", "text inside protected spans"],
        "instruction": ("This clip carries confirmed pronunciations. Every original word must appear "
                        "verbatim, in the same order and case, exactly once across the ordered parts; "
                        "only punctuation outside the protected spans, delivery, speech_traits and part "
                        "boundaries may change. The runtime reapplies each confirmed pronunciation to "
                        "its exact words."),
    }


def _control_token_ranges(pronunciations, source_tokens):
    by_start = {start: position for position, (_, start, _) in enumerate(source_tokens)}
    by_end = {end: position for position, (_, _, end) in enumerate(source_tokens)}
    ranges = []
    for control in pronunciations:
        first = by_start.get(control["start"])
        last = by_end.get(control["end"])
        if first is None or last is None:
            raise ValueError("Confirmed pronunciation control does not align with whole source words")
        ranges.append((first, last, control))
    return ranges


def project_plan_performance(request, plan):
    """Return runtime-owned performance records for every approved plain part.

    Only pronunciation controls are currently recoverable across a model-authored
    repair. The plan may change punctuation, delivery and part boundaries, but its
    exact word sequence (order and case included) must match the confirmed source
    record. A control covering several words must survive wholly inside one part
    with its exact interior text; a part boundary through it rejects the plan.
    Model-authored control fields are never accepted.
    """
    records = source_performance_records(request)
    parts = plan["parts"]
    if not records:
        return [None] * len(parts)
    if request.get("kind") != "speech":
        raise ValueError("Controlled paragraph clip repair is unsupported; original turns remain authoritative")
    chunk = request["chunk"]
    text = chunk["spoken_text"]
    canonical = validate_performance(records[0], text)
    unsupported = sorted({control["kind"] for control in canonical["controls"]
                          if control["kind"] != "pronunciation"})
    if unsupported:
        raise ValueError("Clip repair cannot preserve unsupported performance control(s): "
                         + ", ".join(unsupported))
    pronunciations = [control for control in canonical["controls"] if control["kind"] == "pronunciation"]
    if not pronunciations:
        return [None] * len(parts)

    source_tokens = _word_tokens(text)
    plan_tokens = []
    for index, part in enumerate(parts):
        plan_tokens.extend((word, index, start, end) for word, start, end in _word_tokens(part["text"]))
    if (len(source_tokens) != len(plan_tokens)
            or any(source[0] != plan[0] for source, plan in zip(source_tokens, plan_tokens))):
        raise ValueError("Clip repair would drop, add, reorder, or respell a confirmed pronunciation word")
    projected = [[] for _ in parts]
    for first, last, control in _control_token_ranges(pronunciations, source_tokens):
        targets = plan_tokens[first:last + 1]
        part_indexes = {index for _, index, _, _ in targets}
        if len(part_indexes) != 1:
            raise ValueError("Clip repair splits a confirmed pronunciation span across parts")
        part_index = part_indexes.pop()
        part_text = parts[part_index]["text"]
        start, end = targets[0][2], targets[-1][3]
        if part_text[start:end] != control["text"]:
            raise ValueError("Clip repair changes text inside a confirmed pronunciation span")
        projected[part_index].append({**deepcopy(control), "start": start, "end": end,
                                      "text": part_text[start:end]})

    traits = repair_traits(request)
    result = []
    for part, controls in zip(parts, projected):
        if not controls and not traits:
            result.append(None)
            continue
        result.append(validate_performance(
            {"version": PERFORMANCE_VERSION, "text_sha256": text_sha256(part["text"]),
             "controls": controls}, part["text"]))
    return result
