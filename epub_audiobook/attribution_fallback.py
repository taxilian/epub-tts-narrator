"""Bounded recovery of explicitly truncated attribution requests.

Only whole existing target units are partitioned. Source, prompts, identities,
and ordinary successful request keys remain unchanged.
"""

from concurrent.futures import CancelledError
from copy import deepcopy
import json
from pathlib import Path

from .pipeline import align_attribution_response, align_units, fingerprint, write_json
from .prompts import PROMPT_VERSION
from .provider import ProviderError


MAX_ATTRIBUTION_SPLIT_DEPTH = 2
TRUNCATED_OUTPUT = "The model output was truncated. Increase the output token limit or reduce the input."
_PAYLOAD_KEYS = {"final_cast", "performance_policy", "target_units", "context_units",
                 "prior_segments", "scene_summary"}


def _request_metadata(requests, system, payload):
    return {
        "request_fingerprint": fingerprint(payload),
        "request_cache_fingerprint": fingerprint({
            "provider": requests.client.identity, "prompt_version": PROMPT_VERSION,
            "system": system, "payload": payload,
        }),
    }


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _split_marker(requests, system, payload):
    """Resume hints are separate from successful, validated model responses."""
    marker = {"version": 1, "reason": "model_output_truncated",
              **_request_metadata(requests, system, payload),
              "observation_origin": "provider_response", "prior_observation_fingerprint": None}
    key = marker["request_cache_fingerprint"]
    path = Path(requests.cache_dir) / "attribution-subdivisions" / f"{key}.json"
    # A later successful parent response takes precedence over this hint.
    if not (Path(requests.cache_dir) / f"{key}.json").exists():
        try:
            if path.is_symlink() or not path.is_file():
                return path, marker, False
            with path.open(encoding="utf-8") as stream:
                raw = stream.read(2049)
            saved = json.loads(raw) if len(raw) <= 2048 else None
            if isinstance(saved, dict) and set(saved) == set(marker):
                observation = saved["prior_observation_fingerprint"]
                valid_origin = ((saved["observation_origin"] == "provider_response" and observation is None)
                                or (saved["observation_origin"] == "prior_run" and _sha256(observation)))
                if valid_origin and all(saved[key] == value for key, value in marker.items()
                                        if key not in {"observation_origin", "prior_observation_fingerprint"}):
                    return path, saved, True
        except (OSError, ValueError, RecursionError):
            pass
    return path, marker, False


def record_attribution_truncation(requests, system, payload, *, prior_observation_fingerprint=None):
    """Import an observed failure for this exact request without a model call.

    An external importer must verify the exact provider/system/payload failed
    with explicit output truncation, and fingerprint that prior observation.
    This hint only selects smaller requests; it never supplies accepted output.
    """
    if (not isinstance(system, str) or not system
            or not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS
            or not isinstance(payload["target_units"], list) or len(payload["target_units"]) < 2
            or any(not isinstance(unit, dict) or not isinstance(unit.get("unit_id"), str)
                   for unit in payload["target_units"])
            or len({unit["unit_id"] for unit in payload["target_units"]}) != len(payload["target_units"])):
        raise ValueError("A truncation hint requires the exact attribution system/payload with at least two distinct target units")
    if prior_observation_fingerprint is not None and not _sha256(prior_observation_fingerprint):
        raise ValueError("prior_observation_fingerprint must be a lowercase SHA256 fingerprint")
    path, marker, _ = _split_marker(requests, system, payload)
    marker.update(observation_origin="prior_run" if prior_observation_fingerprint is not None else "provider_response",
                  prior_observation_fingerprint=prior_observation_fingerprint)
    write_json(path, marker, overwrite=True)
    return deepcopy(marker)


def _restrict_payload(payload, selected, positions, prior_segments):
    """Promote sibling units to ordered read-only context, without losing slices."""
    ids = {unit["unit_id"] for unit in selected}
    context = payload["context_units"] + [unit for unit in payload["target_units"] if unit["unit_id"] not in ids]
    context = {(unit["unit_id"], unit["start"], len(unit["text"])): unit for unit in context}
    return dict(payload, target_units=selected,
                context_units=sorted(context.values(), key=lambda unit: (positions[unit["block_id"]], unit["start"])),
                prior_segments=prior_segments)


def _validate_combined(source, cast, units, segments):
    """Revalidate original target coverage/order while retaining restoration audits."""
    response, index = {"units": []}, 0
    for unit in units:
        pieces, cursor = [], unit["start"]
        end = cursor + len(unit["text"])
        while index < len(segments):
            item = segments[index]
            if item["block_id"] != unit["block_id"] or item["start"] >= end:
                break
            if item["start"] != cursor or item["end"] > end:
                raise ValueError("Subdivided attribution did not preserve exact target-unit boundaries")
            pieces.append({
                "text": item["text"], "kind": item["kind"], "role_id": item["role_id"],
                "delivery": item["delivery"], **item["attribution"],
            })
            cursor = item["end"]
            index += 1
        response["units"].append({"unit_id": unit["unit_id"], "segments": pieces})
    if index != len(segments):
        raise ValueError("Subdivided attribution included segments outside its original targets")
    validated = align_units(source, cast, units, response)
    stripped = [{key: value for key, value in item.items() if key != "source_restoration"} for item in segments]
    if validated != stripped:
        raise ValueError("Subdivided attribution changed previously validated source spans")


def attribute_with_subdivision(source, cast, requests, system, payload, *, sequential_prior=False,
                               cancelled=None, progress=lambda message: None):
    """Return (aligned segments, subdivision provenance or None).

    A normal request uses the original system/payload byte-for-byte. Children
    execute sequentially inside the calling worker. Serial attribution carries
    preceding accepted child segments forward; independent workers never share
    those annotations. At most two split levels (four leaves) are allowed.
    """
    positions = {block["id"]: index for index, block in enumerate(source["blocks"])}

    def run_part(current, depth):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Attribution was cancelled")
        units = current["target_units"]
        record = {
            **_request_metadata(requests, system, current),
            "target_unit_ids": [unit["unit_id"] for unit in units],
            "context_units_fingerprint": fingerprint(current["context_units"]),
            "subdivision_depth": depth,
        }
        can_split = depth < MAX_ATTRIBUTION_SPLIT_DEPTH and len(units) > 1
        path, marker, split = _split_marker(requests, system, current) if can_split else (None, None, False)
        replayed_hint = split
        if not split:
            try:
                aligned = requests.run(
                    system, current, lambda response: align_attribution_response(source, cast, units, response),
                    cancelled=cancelled,
                )
            except ProviderError as error:
                if (not can_split or error.status_code is not None or error.code is not None
                        or str(error) != TRUNCATED_OUTPUT):
                    raise
                marker = record_attribution_truncation(requests, system, current)
            else:
                record.update(status="validated", segment_count=len(aligned))
                return aligned, record

        progress(f"Subdividing truncated attribution request at depth {depth + 1}/{MAX_ATTRIBUTION_SPLIT_DEPTH} "
                 f"({len(units)} target units)")
        midpoint = len(units) // 2
        combined, children = [], []
        for selected in (units[:midpoint], units[midpoint:]):
            prior = ((current["prior_segments"] + combined)[-6:] if sequential_prior
                     else current["prior_segments"])
            child_payload = _restrict_payload(current, selected, positions, prior)
            aligned, child_record = run_part(child_payload, depth + 1)
            combined.extend(aligned)
            children.append(child_record)
        _validate_combined(source, cast, units, combined)
        record.update(
            status="subdivided", original_attempt_status="truncated",
            truncation_observation={
                "origin": marker["observation_origin"],
                "prior_observation_fingerprint": marker["prior_observation_fingerprint"],
                "replayed_hint": replayed_hint,
            },
            children=children, segment_count=len(combined),
        )
        return combined, record

    segments, record = run_part(payload, 0)
    if record["status"] == "validated":
        return segments, None
    return segments, {"version": 1, "max_split_depth": MAX_ATTRIBUTION_SPLIT_DEPTH,
                      "prior_segment_mode": "serial" if sequential_prior else "independent",
                      "source_fingerprint": fingerprint(source), "cast_fingerprint": fingerprint(cast),
                      **record}
