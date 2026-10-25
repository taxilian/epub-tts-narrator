"""Bind bounded fragment synthesis to its original source and reviewed delivery."""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re

from .common import digest, file_hash, read_json, regular_file
from .delivery_recovery import delivery_marker, plain_delivery_request, validate_delivery_marker
from .recurring_resolution import RecurringResolution
from .speech_identity import _DRAMA_ENGINES, generation_identity
from .speech_text import trim_speech_text


def inherited_delivery(directory, child, verifier, config, synthesize_plain):
    session = RecurringResolution(directory, child, child["part"]["text"], verifier.speech_resolver,
                                  config.get("verification", {}), can_rewrite=True, can_split=False,
                                  can_plain_delivery=(callable(synthesize_plain)
                                                      and plain_delivery_request(child) is not None))
    choice = session.active_delivery()
    marker = delivery_marker(child, choice) if choice is not None else None
    actual = validate_delivery_marker(child, marker) if marker is not None else child
    return marker, actual["synthesis_request"]


def reject_exhausted_join(directory, child):
    from .retry_budget import generation_attempts
    from .workflow import RenderBlocked, SpeechVerificationFailed

    if (directory / "accepted.json").exists() or (directory / "accepted.json").is_symlink():
        return
    for number in reversed(generation_attempts(directory)):
        record = read_json(regular_file(directory / f"attempt-{number:03d}.json"))
        if "fragment_recovery" not in record.get("engine_metadata", {}):
            continue
        if record.get("request") != child or record.get("request_hash") != digest(child):
            raise RenderBlocked("Fragment join does not match its original source request")
        report = record.get("report", {})
        reasons = report.get("reasons", [])
        pending = report.get("retry_on_resume") or (reasons and all(
            reason == "generation_or_verification_error" or reason.startswith("verification_failed:")
            for reason in reasons))
        if report.get("passed") is False and not pending:
            raise SpeechVerificationFailed("Cutoff-tail recovery is exhausted; the rejected join is retained.",
                                           request=child, report=report)
        return


def fragment_generation_identity(request, voices_by_asset, metadata):
    from .fragment_recovery import _BINDING, _STRATEGY, _VERSION, _piece_request, fragment_plan

    if (not isinstance(request, Mapping) or request.get("kind") != "speech"
            or "recovery" in request or not isinstance(request.get("chunk"), Mapping)):
        raise ValueError("Fragment recovery requires its bound part speech request")
    body = metadata.get("fragment_recovery") if isinstance(metadata, Mapping) else None
    text = request["chunk"].get("spoken_text")
    plan = fragment_plan(text)
    if (not isinstance(body, Mapping) or body.get("version") != 1 or body.get("strategy") != _STRATEGY
            or body.get("child_acceptance_version") != 1 or plan is None or body.get("parts") != plan
            or metadata.get("synthesis_text") != trim_speech_text(text)):
        raise ValueError("Invalid cutoff-tail fragment generation audit")
    binding = body.get("binding")
    if (not isinstance(binding, Mapping) or set(binding) != set(_BINDING)
            or binding["kind"] != "speech_recovery_part" or binding["version"] != 1
            or not isinstance(binding.get("part"), Mapping) or binding["part"].get("text") != text
            or any(not isinstance(binding.get(field), str)
                   or re.fullmatch(r"[0-9a-f]{64}", binding[field]) is None
                   for field in ("parent_request_hash", "recovery_request_hash"))):
        raise ValueError("Fragment recovery is not bound to its recovery part request")
    child = {**deepcopy(binding), "synthesis_request": request}
    if body.get("request_hash") != digest(child):
        raise ValueError("Fragment recovery request binding changed")
    actual_child = validate_delivery_marker(child, body["delivery_recovery"]) if "delivery_recovery" in body else child
    synthesis = actual_child["synthesis_request"]
    evidence = body.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(plan):
        raise ValueError("Fragment recovery is missing piece evidence")
    fragment_request = {**child, "recovery": {"version": _VERSION, "strategy": _STRATEGY, "parts": plan}}
    for piece, row in zip(plan, evidence):
        actual, piece_request = _piece_request(child, fragment_request, piece, synthesis=synthesis)
        accepted = row.get("acceptance") if isinstance(row, Mapping) else None
        if (not isinstance(row, Mapping) or any(row.get(key) != value for key, value in piece.items())
                or row.get("request_hash") != digest(piece_request) or not isinstance(accepted, Mapping)
                or accepted.get("request") != piece_request or accepted.get("request_hash") != digest(piece_request)):
            raise ValueError("Fragment recovery piece binding changed")
        encoded = (json.dumps(accepted, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        report, piece_metadata = accepted.get("report", {}), accepted.get("engine_metadata")
        if (row.get("accepted_sha256") != hashlib.sha256(encoded).hexdigest()
                or row.get("audio_sha256") != accepted.get("audio_sha256")
                or row.get("validation_key") != accepted.get("validation_key")
                or row.get("audio_file") != f"fragments/{digest(piece_request)}/{accepted.get('audio_file')}"
                or not isinstance(report, Mapping) or report.get("passed") is not True
                or report.get("reasons") or report.get("expected_text") != piece["text"]
                or not isinstance(piece_metadata, Mapping) or "fragment_recovery" in piece_metadata
                or (request["engine"].get("engine") in _DRAMA_ENGINES
                    and not isinstance(piece_metadata.get("synthesis_text"), str))):
            raise ValueError("Fragment recovery piece acceptance changed")
        from .exhaustion_selection import validate_record
        validate_record(accepted)
        generation_identity(actual, voices_by_asset, engine_metadata=piece_metadata)
    return generation_identity(request, voices_by_asset)
