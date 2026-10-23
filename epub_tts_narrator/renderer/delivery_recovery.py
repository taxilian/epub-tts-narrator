"""Audited plain delivery after a directed performance repeatedly fails.

This changes a synthesis instruction, never its words, voice, or verification
requirements. The original request remains the cache identity, and the actual
neutral performance must still be verified before it can satisfy that request.
"""

from collections.abc import Mapping
from copy import deepcopy
import re

from .common import digest
from .engines import EngineError, _directed_text, _text
from .speech_text import trim_speech_text
from ..spoken_text import effective_delivery


_ENGINES = frozenset({"drama3_mlx", "drama3_mlx_audio", "fish_http", "fish_tts_pool"})
_POLICY = {"version": 1, "strategy": "plain_delivery", "decision": "validated_analysis_model",
           "words": "unchanged", "voice": "unchanged", "verification": "unchanged"}
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _speech_request(request):
    """Return a supported single performance, including a bound recovery child."""
    if not isinstance(request, Mapping):
        return None
    if request.get("kind") == "speech_recovery_part":
        part, actual = request.get("part"), request.get("synthesis_request")
        if (type(request.get("version")) is not int or request["version"] != 1
                or not isinstance(part, Mapping) or set(part) != {"start", "end", "text"}
                or type(part["start"]) is not int or type(part["end"]) is not int
                or part["start"] < 0 or not isinstance(part["text"], str)
                or part["end"] != part["start"] + len(part["text"])
                or any(not isinstance(request.get(field), str)
                       or _HASH.fullmatch(request[field]) is None
                       for field in ("parent_request_hash", "recovery_request_hash"))
                or not isinstance(actual, Mapping) or actual.get("kind") != "speech"
                or not isinstance(actual.get("chunk"), Mapping)
                or actual["chunk"].get("spoken_text") != part["text"]):
            return None
        request = actual
    if request.get("kind") != "speech" or "recovery" in request:
        return None
    engine, chunk = request.get("engine"), request.get("chunk")
    if (not isinstance(engine, Mapping) or engine.get("engine") not in _ENGINES
            or not isinstance(chunk, Mapping) or "paragraph_turns" in chunk
            or chunk.get("silent_control") is True
            or not isinstance(chunk.get("segment"), Mapping)
            or not isinstance(request.get("voice_asset_hash"), str)
            or _HASH.fullmatch(request["voice_asset_hash"]) is None):
        return None
    segment = chunk["segment"]
    # A normalization edit's delivery is coupled to its approved pronunciation.
    # Do not silently drop that style or alter its source-bound edit record.
    normalization = segment.get("speech_normalization")
    if normalization is not None:
        if (not isinstance(normalization, Mapping)
                or normalization.get("version") != 1 or normalization.get("edits") != []):
            return None
    try:
        text = _text(trim_speech_text(_text(chunk.get("spoken_text"), "text")), "text")
        direction = effective_delivery(segment)
        _directed_text(text, direction)
    except (TypeError, ValueError, EngineError):
        return None
    if not direction:
        return None
    return request


def plain_delivery_request(request):
    """Copy a supported request, changing only its effective delivery to plain.

    Unsupported requests and already plain requests return ``None``. The saved
    source segment, wording, reference voice, and original request are untouched.
    """
    speech = _speech_request(request)
    if speech is None:
        return None
    result = deepcopy(dict(request))
    actual = result["synthesis_request"] if request["kind"] == "speech_recovery_part" else result
    actual["chunk"]["segment"]["delivery"]["direction"] = ""
    return result


def delivery_marker(request, resolution):
    """Bind the bounded fallback to the original text, voice, and direction.

    Recovery children bind their full synthesis request, because the enclosing
    sentence audit independently binds the child to its original source slice.
    """
    speech = _speech_request(request)
    if speech is None:
        raise ValueError("Plain delivery recovery requires a supported directed speech request")
    from .speech_resolver import validate_resolution

    if (not isinstance(resolution, Mapping) or resolution.get("action") != "plain_delivery"
            or not isinstance(resolution.get("audit"), dict)
            or not isinstance(resolution.get("failures"), list)):
        raise ValueError("Plain delivery recovery requires a validated model decision")
    audit, failures = resolution["audit"], resolution["failures"]
    payload = audit.get("payload")
    permitted = payload.get("permitted_actions") if isinstance(payload, dict) else None
    if not isinstance(permitted, list):
        raise ValueError("Plain delivery decision is missing its permitted actions")
    proposal = validate_resolution(
        audit, speech["chunk"]["spoken_text"], failures, audit.get("identity"),
        can_rewrite="rewrite" in permitted, can_split="split" in permitted, can_plain_delivery=True)
    if not isinstance(proposal, Mapping) or proposal.get("action") != "plain_delivery":
        raise ValueError("Plain delivery recovery does not match its validated model decision")
    return {"version": 1, "strategy": "plain_delivery", "policy_sha256": digest(_POLICY),
            "request_sha256": digest(speech), "original_text": speech["chunk"]["spoken_text"],
            "original_direction": effective_delivery(speech["chunk"]["segment"]),
            "replacement_direction": "", "audit": deepcopy(audit), "failures": deepcopy(failures)}


def validate_delivery_marker(request, marker):
    """Return the exact neutral request only when its saved audit still matches."""
    if not isinstance(marker, Mapping):
        raise ValueError("Invalid plain delivery recovery audit")
    expected = delivery_marker(request, {"action": "plain_delivery", "audit": marker.get("audit"),
                                         "failures": marker.get("failures")})
    if dict(marker) != expected:
        raise ValueError("Plain delivery recovery audit differs from its original request")
    result = plain_delivery_request(request)
    if result is None:
        raise ValueError("Plain delivery recovery no longer has a supported request")
    return result
