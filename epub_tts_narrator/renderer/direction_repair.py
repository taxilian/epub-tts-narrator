"""Auditable authority for synthesis-only scene-control replacement."""

from copy import deepcopy
import json
import re
import unicodedata

from ..spoken_text import effective_delivery
from .common import digest
from .delivery_recovery import _speech_request
from .direction_repair_prompts import PROPOSE, REVIEW
from .engines import _directed_text
from .speech_text import trim_speech_text
from .speech_traits import performance_direction


def supported_request(request):
    return _speech_request(request, allow_normalization=True)


def policy(provider):
    return {"name": "scene_direction_repair", "version": 1, "provider": deepcopy(provider),
            "proposal_prompt_sha256": digest(PROPOSE), "review_prompt_sha256": digest(REVIEW),
            "reviewers": 3, "approvals": 3, "max_direction_chars": 160, "candidates": 1}


def _plain(value, maximum):
    return (isinstance(value, str) and 0 < len(value) <= maximum and value == value.strip()
            and not any(char in value for char in '[]<>`"“”')
            and not any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"} for char in value))


def proposal_valid(value):
    return (isinstance(value, dict) and set(value) == {"control_related", "replacement_direction", "reason"}
            and type(value["control_related"]) is bool and _plain(value["reason"], 800)
            and (_plain(value["replacement_direction"], 160) if value["control_related"]
                 else value["replacement_direction"] == ""))


def review_valid(value):
    flags = {"equivalent", "nonspoken", "preserves_words", "preserves_traits"}
    return (isinstance(value, dict) and set(value) == flags | {"reason"}
            and all(type(value[key]) is bool for key in flags) and _plain(value["reason"], 800))


def make_payload(request, failures):
    speech = supported_request(request)
    if speech is None:
        raise ValueError("Direction repair requires supported directed speech")
    if (not isinstance(failures, list) or not 2 <= len(failures) <= 30
            or any(not isinstance(row, dict) or type(row.get("attempt")) is not int
                   or row["attempt"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("audio_sha256")))
                   or not row.get("reasons") for row in failures)
            or len({row["attempt"] for row in failures}) != len(failures)):
        raise ValueError("Direction repair needs distinct failed takes")
    chunk = speech["chunk"]
    performer = speech.get("performer")
    voice_id = performer["voice_id"] if performer else chunk["segment"]["voice_id"]
    payload = {"mode": "direction_repair", "request_sha256": digest(speech),
               "requested_text": chunk["spoken_text"], "source": deepcopy(chunk["segment"]),
               "original_direction": effective_delivery(chunk["segment"]),
               "permanent_traits": deepcopy(chunk.get("speech_traits", {})),
               "performance_direction": performance_direction(chunk, voice_id),
               "failures": deepcopy(failures)}
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False)) > 64000:
        raise ValueError("Direction repair context exceeds its bounded payload")
    return payload


def review_payload(payload, proposal, reviewer):
    return {**deepcopy(payload), "mode": "direction_review", "reviewer": reviewer,
            "replacement_direction": proposal["replacement_direction"]}


def validate_marker(request, marker):
    if (not isinstance(marker, dict)
            or set(marker) != {"identity", "payload", "proposal", "reviews"}
            or not isinstance(marker["identity"], dict)
            or marker["identity"] != policy(marker["identity"].get("provider"))
            or not isinstance(marker["payload"], dict)):
        raise ValueError("Invalid direction repair authority")
    payload = make_payload(request, marker["payload"].get("failures"))
    proposal = marker["proposal"]
    if (marker["payload"] != payload or not proposal_valid(proposal) or not proposal["control_related"]
            or proposal["replacement_direction"] == payload["original_direction"]
            or not isinstance(marker["reviews"], list) or len(marker["reviews"]) != 3):
        raise ValueError("Direction repair differs from the original intention")
    for number, review in enumerate(marker["reviews"], 1):
        if (not isinstance(review, dict) or set(review) != {"payload", "response"}
                or review["payload"] != review_payload(payload, proposal, number)
                or not review_valid(review["response"])
                or not all(review["response"][flag] for flag in
                           ("equivalent", "nonspoken", "preserves_words", "preserves_traits"))):
            raise ValueError("Direction repair lacks three independent approvals")
    return proposal["replacement_direction"]


def repaired_direction(request, replacement):
    speech = supported_request(request)
    if speech is None or not _plain(replacement, 160):
        raise ValueError("Invalid synthesis-only direction")
    chunk = speech["chunk"]
    performer = speech.get("performer")
    voice_id = performer["voice_id"] if performer else chunk["segment"]["voice_id"]
    return performance_direction(chunk, voice_id, scene_direction=replacement)


def confirm_metadata(request, metadata):
    replacement = validate_marker(request, metadata.get("direction_repair"))
    if any(key in metadata for key in ("speech_resolution", "delivery_recovery")):
        raise ValueError("Direction repair must retain original words and permanent traits")
    speech = supported_request(request)
    text = trim_speech_text(speech["chunk"]["spoken_text"])
    direction = repaired_direction(request, replacement)
    directed, expected = _directed_text(text, direction)
    if (metadata.get("synthesis_text") not in (text, directed)
            or any(metadata.get(key) != expected[key] for key in
                   ("direction", "direction_applied", "applied_control"))):
        raise ValueError("Recorded synthesis does not prove the reviewed direction repair")
