"""Compact attribution wire: source-anchored segments and sparse inline proposals.

The model proposes speaker boundaries with unique exact ``start_anchor``
substrings instead of echoing source text, plus a sparse, per-segment set of
optional inline proposals (stylized-spelling normalizations and local
delivery/pause/vocalization controls). Code reconstructs the exact source text
locally and validates it with the existing ``align_units`` contract, so the
canonical source is never model-authored. Proposals travel on a private row
field through subdivision only and are removed before publication; the
separate remove-only review stage owns approval.
"""

from copy import deepcopy
import unicodedata

from .performance_plan import DELIVERY_TAGS, VOCAL_SOUNDS, delivery_overlap_error
from .pipeline import align_attribution_response, align_units
from .spoken_text import MAX_DELIVERY_CHARACTERS

_MAX_PROPOSAL_EDITS = 16
_MAX_PROPOSAL_CONTROLS = 32
_MAX_LOCATOR_CHARACTERS = 2000
_REQUIRED_EDIT_KEYS = {"original", "replacement", "reason"}
_EDIT_KEYS = {"original", "replacement", "reason", "direction", "left", "right"}
_DELIVERY_KEYS = {"kind", "tag", "original", "reason", "direction", "emotion", "left", "right"}
_PAUSE_KEYS = {"kind", "duration_ms", "position", "original", "reason", "left", "right"}
_VOCALIZATION_KEYS = {"kind", "sound", "position", "original", "reason", "left", "right"}


def compact_cast(cast):
    """Attribution wire cast: identities and scopes without creation payload.

    All character/role/voice IDs, names, aliases, and identity/scope
    descriptions stay on the wire; voice creation profiles (speechTraits,
    traits) and cast evidence are stripped only from the request. The stored
    cast is never mutated.
    """
    fields = {"characters": ("id", "name", "aliases", "description", "voice_id"),
              "roles": ("id", "kind", "description", "character_id", "voice_id", "source_scope"),
              "voices": ("id",)}
    return {collection: [{key: deepcopy(record[key]) for key in fields[collection] if key in record}
                         for record in cast[collection]] for collection in fields}


def _plain_bounded(value, label, *, required, maximum=None, strip=True):
    if (not isinstance(value, str) or any(char in value for char in "[]<>")
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for char in value)):
        raise ValueError(f"{label} must be plain text without bracket control syntax")
    if strip:
        value = value.strip()
    if required and not value.strip():
        raise ValueError(f"{label} must be a nonempty plain string")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters (received {len(value)})")
    return value


def _locator(raw, where):
    parts = {}
    for key in ("original", "left", "right"):
        if key == "original" or raw.get(key) is not None:
            value = raw.get(key)
            if (not isinstance(value, str) or key == "original" and not value
                    or len(value) > _MAX_LOCATOR_CHARACTERS):
                raise ValueError(f"{where}.{key} requires bounded exact source text")
            parts[key] = value
    parts.setdefault("left", "")
    parts.setdefault("right", "")
    return parts


def _resolved_span(locator, piece_text, where):
    left, original, right = locator["left"], locator["original"], locator["right"]
    composed = left + original + right
    first = piece_text.find(composed)
    if first == -1:
        raise ValueError(f"{where}: locator does not occur in the segment; copy original, left, "
                         "and right exactly from the source")
    if piece_text.find(composed, first + 1) != -1:
        raise ValueError(f"{where}: locator occurs more than once in the segment; extend left/right "
                         "context until the match is unique")
    start = first + len(left)
    return start, start + len(original)


def _position(raw, where):
    if raw.get("position") not in ("before", "after"):
        raise ValueError(f"{where}.position must be 'before' or 'after'")
    return raw["position"]


def _parse_edit(raw, piece_text, where):
    if not isinstance(raw, dict) or set(raw) - _EDIT_KEYS or not _REQUIRED_EDIT_KEYS <= set(raw):
        raise ValueError(f"{where}: each edit requires original, replacement, reason, and direction; "
                         "optional left, and right")
    locator = _locator(raw, where)
    start, end = _resolved_span(locator, piece_text, where)
    replacement = _plain_bounded(raw["replacement"], f"{where}.replacement", required=True,
                                 maximum=_MAX_LOCATOR_CHARACTERS, strip=False)
    reason = _plain_bounded(raw["reason"], f"{where}.reason", required=True, maximum=1000)
    direction = _plain_bounded(raw.get("direction", ""), f"{where}.direction", required=True, maximum=240)
    if replacement == locator["original"]:
        raise ValueError(f"{where}: an edit must change its original spelling; "
                         "omit proposals that need no change")
    return {"original": locator["original"], "replacement": replacement, "reason": reason,
            "direction": direction, "left": locator["left"], "right": locator["right"],
            "span": (start, end)}


def _parse_control(raw, piece_text, where):
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
        raise ValueError(f"{where}: each control requires a kind")
    kind = raw["kind"]
    if kind == "delivery":
        if set(raw) - _DELIVERY_KEYS or not {"kind", "tag", "original", "reason"} <= set(raw):
            raise ValueError(f"{where}: delivery controls require kind, tag, original, and reason")
        tag = raw["tag"]
        if not isinstance(tag, str) or tag not in DELIVERY_TAGS:
            raise ValueError(f"{where}.tag is unsupported; use one of {', '.join(sorted(DELIVERY_TAGS))}")
        direction = _plain_bounded(raw.get("direction", ""), f"{where}.direction",
                                   required=tag == "style" and raw.get("emotion") is None, maximum=240)
        emotion = (_plain_bounded(raw["emotion"], f"{where}.emotion", required=True, maximum=80)
                   if raw.get("emotion") is not None else "")
        if emotion or tag != "style" and direction:
            direction = "; ".join(value for value in (tag if tag != "style" else "", direction, emotion) if value)
            direction = _plain_bounded(direction, f"{where}.direction", required=True, maximum=240)
            tag = "style"
        record = {"kind": "delivery", "tag": tag, "direction": direction,
                  "reason": _plain_bounded(raw["reason"], f"{where}.reason", required=True, maximum=1000)}
    elif kind == "pause":
        if set(raw) - _PAUSE_KEYS or not {"kind", "duration_ms", "position", "original", "reason"} <= set(raw):
            raise ValueError(f"{where}: pause controls require kind, duration_ms, position, original, and reason")
        duration = raw["duration_ms"]
        if type(duration) is not int or not 100 <= duration <= 5000:
            raise ValueError(f"{where}.duration_ms must be an integer from 100 through 5000")
        record = {"kind": "pause", "duration_ms": duration, "position": _position(raw, where),
                  "reason": _plain_bounded(raw["reason"], f"{where}.reason", required=True, maximum=1000)}
    elif kind == "vocalization":
        if set(raw) - _VOCALIZATION_KEYS or not {"kind", "sound", "position", "original", "reason"} <= set(raw):
            raise ValueError(f"{where}: vocalization controls require kind, sound, position, original, and reason")
        if raw["sound"] not in VOCAL_SOUNDS:
            raise ValueError(f"{where}.sound is unsupported; use one of {', '.join(sorted(VOCAL_SOUNDS))}")
        record = {"kind": "vocalization", "sound": raw["sound"], "position": _position(raw, where),
                  "reason": _plain_bounded(raw["reason"], f"{where}.reason", required=True, maximum=1000)}
    else:
        raise ValueError(f"{where}.kind is unsupported; use delivery, pause, or vocalization")
    locator = _locator(raw, where)
    return dict(record, locator=locator, span=_resolved_span(locator, piece_text, where))


def parse_proposals(payload, piece_text, where, base_direction=""):
    """Validate a segment's sparse inline proposals against its exact text.

    Returns cleaned proposals with locally resolved canonical spans, or None
    when the segment proposes nothing. Unknown shapes raise ValueError so the
    ordinary cached-request repair loop corrects them.
    """
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) - {"edits", "controls"}:
        raise ValueError(f"{where}: proposals must contain only edits and controls arrays")
    edits_raw = payload.get("edits", [])
    controls_raw = payload.get("controls", [])
    if not isinstance(edits_raw, list) or not isinstance(controls_raw, list):
        raise ValueError(f"{where}: proposals edits and controls must be arrays")
    if len(edits_raw) > _MAX_PROPOSAL_EDITS:
        raise ValueError(f"{where}: at most {_MAX_PROPOSAL_EDITS} edits per segment; keep proposals sparse")
    if len(controls_raw) > _MAX_PROPOSAL_CONTROLS:
        raise ValueError(f"{where}: at most {_MAX_PROPOSAL_CONTROLS} controls per segment; keep proposals sparse")
    edits = [_parse_edit(raw, piece_text, f"{where}.edits[{index}]") for index, raw in enumerate(edits_raw)]
    composed = len(base_direction.strip()) + sum(len(edit["direction"]) + 2 for edit in edits)
    if composed > MAX_DELIVERY_CHARACTERS:
        raise ValueError(f"{where}: composed delivery directions would exceed {MAX_DELIVERY_CHARACTERS} "
                         "characters; keep directions brief and proposals sparse")
    for left, right in zip(edits, edits[1:]):
        if left["span"][1] > right["span"][0]:
            raise ValueError(f"{where}: edit spans must not overlap")
    controls = [_parse_control(raw, piece_text, f"{where}.controls[{index}]")
                for index, raw in enumerate(controls_raw)]
    deliveries = sorted((control for control in controls if control["kind"] == "delivery"),
                        key=lambda control: control["span"][0])
    if (problem := delivery_overlap_error([(*control["span"], control["tag"]) for control in deliveries])):
        raise ValueError(f"{where}: {problem}")
    for edit in edits:
        covered = [control for control in deliveries
                   if control["span"][0] < edit["span"][1] and edit["span"][0] < control["span"][1]]
        if len([c for c in covered if c["tag"] == "style"]) > 1 or len([c for c in covered if c["tag"] != "style"]) > 1:
            raise ValueError(f"{where}: at most one delivery control may target a single edit span")
    events = [(control["span"][0 if control["position"] == "before" else 1],
               control["kind"], control.get("duration_ms"), control.get("sound"))
              for control in controls if control["kind"] in ("pause", "vocalization")]
    if len(events) != len(set(events)):
        raise ValueError(f"{where}: duplicate identical events at one position are not allowed")
    cleaned = {"edits": [{key: edit[key] for key in
                          ("original", "replacement", "reason", "direction", "left", "right")}
                         for edit in edits],
               "controls": [dict({key: value for key, value in control.items()
                                  if key not in ("locator", "span")}, **control["locator"])
                            for control in controls]}
    if not cleaned["edits"] and not cleaned["controls"]:
        return None
    return cleaned


def _reconstruct_unit(unit, annotated):
    """Return (legacy-format unit response, per-piece raw proposal payloads)."""
    pieces = annotated.get("segments")
    if not isinstance(pieces, list) or not pieces:
        raise ValueError(f"Each target unit requires nonempty segments at unit {unit['unit_id']!r}; "
                         "return an array covering this complete source unit.")
    if any(not isinstance(piece, dict) for piece in pieces):
        raise ValueError(f"Every segment in unit {unit['unit_id']!r} must be an object")
    pieces = [dict(piece) for piece in pieces]
    for index, piece in enumerate(pieces):
        emotion = piece.pop("emotion", None)
        if emotion is not None:
            emotion = _plain_bounded(emotion, "Segment emotion", required=True, maximum=80)
            delivery = piece.get("delivery")
            if not isinstance(delivery, dict) or not isinstance(delivery.get("direction", ""), str):
                raise ValueError(f"segments[{index}].emotion requires a delivery object")
            delivery = deepcopy(delivery)
            delivery["direction"] = _plain_bounded(
                "; ".join(value for value in (delivery.get("direction", ""), emotion) if value),
                f"segments[{index}].delivery.direction", required=True, maximum=240)
            piece["delivery"] = delivery
    annotated = dict(annotated, segments=pieces)
    text = unit["text"]
    echo = any("text" in piece for piece in pieces)
    if echo and any("start_anchor" in piece for piece in pieces):
        raise ValueError(f"Unit {unit['unit_id']!r} mixes text echoes with start_anchor segments; "
                         "return one format: exact text segments or source-anchored segments")
    if echo:
        return annotated, [piece.get("proposals") for piece in pieces]
    if pieces[0].get("start_anchor"):
        raise ValueError(f"unit {unit['unit_id']!r} segments[0]: the first segment starts at the unit "
                         "start; omit start_anchor there")
    boundaries = [0]
    for index, piece in enumerate(pieces[1:], start=1):
        where = f"unit {unit['unit_id']!r} segments[{index}]"
        anchor = piece.get("start_anchor")
        if isinstance(anchor, dict):
            if set(anchor) != {"left", "right"} or any(not isinstance(anchor[key], str) or not anchor[key]
                                                        for key in ("left", "right")):
                raise ValueError(f"{where}: context start_anchor requires nonempty exact left and right text")
            left, right = anchor["left"], anchor["right"]
            anchor = left + right
        else:
            left = ""
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(f"{where}: a nonempty exact start_anchor is required for every segment "
                              "after the first")
        if len(anchor) > _MAX_LOCATOR_CHARACTERS:
            raise ValueError(f"{where}: start_anchor exceeds {_MAX_LOCATOR_CHARACTERS} characters")
        first = text.find(anchor)
        if first == -1:
            raise ValueError(f"{where}: start_anchor does not occur in the unit; copy it exactly, "
                             "preserving spelling, case, punctuation, and whitespace")
        if text.find(anchor, first + 1) != -1:
            raise ValueError(f"{where}: start_anchor occurs more than once in the unit; extend it with "
                             "surrounding characters until the match is unique")
        first += len(left)
        if first <= boundaries[-1]:
            raise ValueError(f"{where}: start_anchor must begin strictly after the previous segment")
        boundaries.append(first)
    boundaries.append(len(text))
    rebuilt, payloads = [], []
    for index, piece in enumerate(pieces):
        start, end = boundaries[index], boundaries[index + 1]
        segment_text = text[start:end]
        if not segment_text:
            raise ValueError(f"unit {unit['unit_id']!r} segments[{index}]: start_anchor boundaries produce "
                             "an empty segment; anchors must mark distinct segment starts")
        cleaned = {key: piece[key] for key in piece if key not in ("start_anchor", "proposals")}
        cleaned["text"] = segment_text
        rebuilt.append(cleaned)
        payloads.append(piece.get("proposals"))
    return {"unit_id": unit["unit_id"], "segments": rebuilt}, payloads


def align_compact_units(source, cast, units, response):
    """Validate compact or legacy responses and return aligned segment rows.

    Compact units carry start_anchor boundaries instead of echoed text; exact
    text is reconstructed locally and validated by the existing align_units
    contract. Legacy text-echo responses stay valid for shipped caches and
    older models. Proposals are never applied here: they travel on the private
    row field ``inline_proposals`` for the separate remove-only review and are
    stripped before publication.
    """
    if not isinstance(response, dict) or not isinstance(response.get("units"), list):
        raise ValueError("Attribution must return a units array")
    returned = response["units"]
    expected_ids = [unit["unit_id"] for unit in units]
    received_ids = [item.get("unit_id") if isinstance(item, dict) and isinstance(item.get("unit_id"), str)
                    else None for item in returned]
    if received_ids != expected_ids:
        return align_units(source, cast, units, response)
    rebuilt, piece_payloads, piece_locations = [], [], []
    for unit, annotated in zip(units, returned):
        reconstructed, payloads = _reconstruct_unit(unit, annotated)
        rebuilt.append(reconstructed)
        for index, payload in enumerate(payloads):
            piece_payloads.append(payload)
            piece_locations.append(f"units[].unit_id {unit['unit_id']!r} segments[{index}]")
    aligned = align_attribution_response(source, cast, units, {"units": rebuilt})
    if len(aligned) != len(piece_payloads):
        raise AssertionError("Compact reconstruction lost aligned segment rows")
    pieces = (piece for unit in rebuilt for piece in unit["segments"])
    for row, payload, piece, location in zip(aligned, piece_payloads, pieces, piece_locations):
        delivery = piece.get("delivery")
        base_direction = delivery.get("direction", "") if isinstance(delivery, dict) else ""
        payload = parse_proposals(payload, piece["text"], location, base_direction=base_direction)
        if payload is not None:
            row["inline_proposals"] = payload
    return aligned
