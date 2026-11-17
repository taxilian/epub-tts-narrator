"""Send a chosen name as written after its phoneme tags keep being misread.

A person-chosen CMU reading reaches Drama 3 as phoneme tags. For some names,
lines and voices the tags themselves lead it astray: on one narrator's line
"Ce'Nedra replied." five tagged takes all came out "Sendera" while five takes
of the plain spelling all said the chosen reading, and another voice said a
tagged "Hettar" as "HAY-tar" in four of five takes and as written in three of
five. Plain spelling is misread too, on other lines, so neither way is kept
for good: the passage's own takes decide (``plain_next``). Once tagged takes
misread a chosen name ``MISREAD_LIMIT`` times, the next take sends the same
request with only its CMU pronunciation tags removed; when plain takes have
misread as often, tags return.

Only what Drama 3 is sent changes. The original request stays the take's
identity and its verification: every take, tagged or not, must say the chosen
reading (``name_shape``, ``name_phonemes``). The take records which tags it
left out (``marker``), and ``speech_identity`` accepts its audio for the
original request only through ``validate_marker``.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from .common import digest

STRATEGY = "plain_spelling"
_POLICY = {"version": 1, "strategy": STRATEGY, "removed": "cmu_arpabet pronunciation controls",
           "words": "unchanged", "voice": "unchanged", "verification": "unchanged"}
MISREAD_LIMIT = 2
METADATA_KEY = "spelling_recovery"
_MISREAD = "pronunciation_mismatch"


def _tagged(control):
    return (isinstance(control, Mapping) and control.get("kind") == "pronunciation"
            and control.get("alphabet") == "cmu_arpabet")


def plain_spelling_request(request):
    """``request`` with its CMU pronunciation tags removed, or ``None`` when it has none (or is not one line)."""
    if not isinstance(request, Mapping) or request.get("kind") != "speech" or "recovery" in request:
        return None
    chunk = request.get("chunk")
    performance = chunk.get("expressive_performance") if isinstance(chunk, Mapping) else None
    controls = performance.get("controls") if isinstance(performance, Mapping) else None
    if not isinstance(controls, list) or not any(_tagged(control) for control in controls):
        return None
    result = deepcopy(dict(request))
    result["chunk"]["expressive_performance"]["controls"] = [
        control for control in controls if not _tagged(control)]
    return result


def marker(request):
    """The record a plain-spelling take keeps: which tags of which request it left out."""
    if plain_spelling_request(request) is None:
        raise ValueError("Plain spelling recovery needs one line with chosen phoneme tags")
    removed = [{key: control.get(key) for key in ("start", "end", "text", "phonemes")}
               for control in request["chunk"]["expressive_performance"]["controls"] if _tagged(control)]
    return {"version": 1, "strategy": STRATEGY, "policy_sha256": digest(_POLICY),
            "request_sha256": digest(request), "removed": removed}


def validate_marker(request, recorded):
    """The request a plain-spelling take was made from, only when its record matches ``request``."""
    if not isinstance(recorded, Mapping) or dict(recorded) != marker(request):
        raise ValueError("Plain spelling recovery record differs from its original request")
    return plain_spelling_request(request)


def plain_next(records):
    """Whether the next take should send the chosen names as written, from a passage's saved takes."""
    tagged = plain = 0
    for record in records:
        report = record.get("report") or {}
        if report.get("reasons") != [_MISREAD] or not record.get("audio_file"):
            continue
        if METADATA_KEY in (record.get("engine_metadata") or {}):
            plain += 1
        else:
            tagged += 1
    return tagged >= MISREAD_LIMIT and plain < tagged
