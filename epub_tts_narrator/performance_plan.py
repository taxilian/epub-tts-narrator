"""Portable, exact-text-bound performance controls, independent of any TTS engine.

Offsets count Unicode code points in the approved spoken text, not source bytes
or model markup. Provenance remains in the planner's record; this module returns
only the effective controls that belong in generation/cache identities.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
import unicodedata

from .spoken_text import MAX_DELIVERY_CHARACTERS


PERFORMANCE_VERSION = 1
DELIVERY_TAGS = frozenset({"style", "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"})
VOCAL_SOUNDS = frozenset({"sigh", "laugh", "chuckle", "breath", "gasp", "sob", "cough", "throat_clear"})
MAX_CONTROLS = 256
_VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CONSONANTS = frozenset("B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split())
_PROVENANCE = {"evidence", "reason"}


def text_sha256(text):
    if not isinstance(text, str) or any(unicodedata.category(c) == "Cs" for c in text):
        raise ValueError("Performance text must be Unicode text without surrogate characters")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plain_direction(value, label="Performance direction", *, required=False):
    if (not isinstance(value, str) or len(value) > MAX_DELIVERY_CHARACTERS
            or any(c in value for c in "[]<>")
            or any(unicodedata.category(c) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for c in value)
            or required and not value.strip()):
        raise ValueError(f"{label} must be bounded plain text without control syntax")
    return value.strip()


def validate_phonemes(value):
    """Validate CMU inventory/stress syntax, not pronunciation equivalence."""
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ValueError("Performance phonemes must be bounded CMU Arpabet text")
    tokens = value.split(" ")
    if not 1 <= len(tokens) <= 128:
        raise ValueError("Performance pronunciation has too many phonemes")
    for token in tokens:
        if token in _CONSONANTS or token in _VOWELS:
            continue
        if len(token) >= 2 and token[-1] in "012" and token[:-1] in _VOWELS:
            continue
        raise ValueError("Performance phonemes contain an invalid CMU Arpabet symbol or stress")
    return value


def _provenance(record, label):
    if "reason" in record and (not isinstance(record["reason"], str) or len(record["reason"]) > 1000):
        raise ValueError(f"{label}.reason must be bounded text")
    if "evidence" not in record:
        return
    evidence = record["evidence"]
    if not isinstance(evidence, list) or len(evidence) > 32:
        raise ValueError(f"{label}.evidence must be a bounded list")
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {"block_id", "quote"}
                or not isinstance(item["block_id"], str) or not 1 <= len(item["block_id"]) <= 200
                or not isinstance(item["quote"], str) or not 1 <= len(item["quote"]) <= 2000):
            raise ValueError(f"{label}.evidence requires bounded block_id and quote strings")


def _span(item, text, label):
    a, b = item.get("start"), item.get("end")
    if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text)
            or not isinstance(item.get("text"), str) or item["text"] != text[a:b]):
        raise ValueError(f"{label} must identify an exact nonempty spoken-text span")


def _sort_key(item):
    return (item.get("start", item.get("offset")),
            {"pause": 0, "vocalization": 1, "delivery": 2, "pronunciation": 3}[item["kind"]],
            item.get("end", 0))


def validate_performance(record, text):
    """Return canonical effective controls; reject stale text and ambiguous scope.

Delivery ranges cannot overlap each other. Pronunciation ranges cannot overlap
each other, but may be wholly contained by a delivery range. This allows a
whispered name without inventing a general conflicting-style nesting policy.
"""
    expected_hash = text_sha256(text)
    if (not isinstance(record, dict)
            or set(record) - {"version", "text_sha256", "controls"} - _PROVENANCE
            or type(record.get("version")) is not int or record["version"] != PERFORMANCE_VERSION
            or record.get("text_sha256") != expected_hash
            or not isinstance(record.get("controls"), list) or len(record["controls"]) > MAX_CONTROLS):
        raise ValueError("Performance record requires version 1, its exact spoken-text SHA-256 and bounded controls")
    _provenance(record, "performance")
    controls = []
    for index, original in enumerate(record["controls"]):
        label = f"performance.controls[{index}]"
        if not isinstance(original, dict):
            raise ValueError(f"{label} must be an object")
        _provenance(original, label)
        item = {k: deepcopy(v) for k, v in original.items() if k not in _PROVENANCE}
        kind = item.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{label}.kind must be text")
        if kind == "delivery":
            if set(item) - {"kind", "start", "end", "text", "tag", "direction"}:
                raise ValueError(f"{label} has unknown delivery fields")
            _span(item, text, label)
            if not isinstance(item.get("tag"), str) or item["tag"] not in DELIVERY_TAGS:
                raise ValueError(f"{label}.tag is unsupported")
            item["direction"] = plain_direction(item.get("direction", ""), label + ".direction", required=item["tag"] == "style")
        elif kind == "pronunciation":
            if set(item) != {"kind", "start", "end", "text", "alphabet", "phonemes"}:
                raise ValueError(f"{label} requires pronunciation bounds, alphabet and phonemes")
            _span(item, text, label)
            if item["alphabet"] != "cmu_arpabet":
                raise ValueError(f"{label}.alphabet is unsupported")
            # A phoneme control replaces one lexical word; adjacent punctuation
            # and whitespace remain canonical text outside the model token.
            if not re.fullmatch(r"[^\W_]+(?:['’\-][^\W_]+)*", item["text"], re.UNICODE):
                raise ValueError(f"{label} pronunciation must replace one word without surrounding punctuation")
            if (item["start"] and text[item["start"] - 1].isalnum()
                    or item["end"] < len(text) and text[item["end"]].isalnum()):
                raise ValueError(f"{label} pronunciation must not replace only part of a word")
            validate_phonemes(item["phonemes"])
        elif kind in {"pause", "vocalization"}:
            fields = {"kind", "offset", "duration_ms" if kind == "pause" else "sound"}
            if set(item) != fields or type(item.get("offset")) is not int or not 0 <= item["offset"] <= len(text):
                raise ValueError(f"{label} requires a valid event offset and event value")
            if kind == "pause":
                if type(item["duration_ms"]) is not int or not 100 <= item["duration_ms"] <= 5000:
                    raise ValueError(f"{label}.duration_ms must be an integer from 100 through 5000")
            elif not isinstance(item["sound"], str) or item["sound"] not in VOCAL_SOUNDS:
                raise ValueError(f"{label}.sound is unsupported")
        else:
            raise ValueError(f"{label}.kind is unsupported")
        controls.append(item)
    controls.sort(key=_sort_key)
    deliveries = [c for c in controls if c["kind"] == "delivery"]
    phones = [c for c in controls if c["kind"] == "pronunciation"]
    for spans in (deliveries, phones):
        if any(left["end"] > right["start"] for left, right in zip(spans, spans[1:])):
            raise ValueError("Performance spans of the same kind must not overlap")
    for phone in phones:
        for delivery in deliveries:
            if (phone["start"] < delivery["end"] and delivery["start"] < phone["end"]
                    and not delivery["start"] <= phone["start"] < phone["end"] <= delivery["end"]):
                raise ValueError("Performance pronunciation may be inside delivery, but must not cross its boundary")
        if any(phone["start"] < c["offset"] < phone["end"] for c in controls if "offset" in c):
            raise ValueError("Performance events must not fall inside a pronunciation span")
    # Two identical events at a position usually indicate a projection bug;
    # differently valued simultaneous events remain explicit, ordered controls.
    events = [tuple(sorted(c.items())) for c in controls if "offset" in c]
    if len(events) != len(set(events)):
        raise ValueError("Performance contains duplicate events")
    return {"version": PERFORMANCE_VERSION, "text_sha256": expected_hash, "controls": controls}


def slice_performance(record, text, start, end):
    """Project controls onto a nonempty exact slice, retaining each event once.

Interior boundary events belong to the following slice; an event at the final
text boundary belongs to the last slice. Pronunciations are atomic and a caller
must move its split point rather than cutting a phoneme replacement in half.
"""
    canonical = validate_performance(record, text)
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("Performance slice requires valid nonempty spoken-text bounds")
    projected = []
    for control in canonical["controls"]:
        item = dict(control)
        if "offset" in item:
            if start <= item["offset"] < end or item["offset"] == end == len(text):
                item["offset"] -= start
                projected.append(item)
            continue
        a, b = max(start, item["start"]), min(end, item["end"])
        if a >= b:
            continue
        if item["kind"] == "pronunciation" and (a != item["start"] or b != item["end"]):
            raise ValueError("A rendering split must not cut through a pronunciation span")
        item.update(start=a - start, end=b - start, text=text[a:b])
        projected.append(item)
    sliced = text[start:end]
    return validate_performance({"version": PERFORMANCE_VERSION, "text_sha256": text_sha256(sliced), "controls": projected}, sliced)


def combine_performances(parts, *, separator=" "):
    """Join ``(spoken_text, record_or_None)`` pairs and rebase portable offsets.

This is only an offset operation: voice, source and paragraph compatibility
remain the caller's responsibility. The returned record binds the joined text.
"""
    if not isinstance(separator, str):
        raise ValueError("Performance separator must be text")
    texts, controls, cursor = [], [], 0
    for text, record in parts:
        text_sha256(text)
        if texts:
            cursor += len(separator)
        if record is not None:
            for original in validate_performance(record, text)["controls"]:
                item = dict(original)
                for field in ("start", "end", "offset"):
                    if field in item:
                        item[field] += cursor
                controls.append(item)
        texts.append(text)
        cursor += len(text)
    text = separator.join(texts)
    return validate_performance({"version": PERFORMANCE_VERSION, "text_sha256": text_sha256(text), "controls": controls}, text)
