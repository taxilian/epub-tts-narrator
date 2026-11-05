"""Adaptive whole-cast coordination with lossless target-voice partitions."""

from copy import deepcopy
import json

from .adaptive_requests import request_size, run_partitioned
from .pipeline import check_cast, fingerprint
from .voice_ensemble import (
    CONTRAST_VERSION, VERSION, _build_cast, _contrast_context,
    _inherited, _payload, _validate_response,
)
from .voice_ensemble_prompts import COMPACT_CONTRAST_GUIDANCE, ENSEMBLE_SYSTEM


GROUP_GUIDANCE = """
ADAPTIVE TARGET VOICE SCOPE
This request is one lossless partition of the canonical cast, not a claim that
unlisted records are absent. scope.full_cast_fingerprint identifies that cast;
scope.included_ids and scope.omitted_counts describe this request's coverage.
Only scope.target_voice_ids are design targets and may be upserted. Every other
voice, whether included or omitted, is read-only context. Included records are
complete; never drop or trim their contents or text-trait evidence to fit.
cross_group_contrast supplies concise read-only advisory findings involving
omitted voices so coordination can continue without copying their profiles.
Omitted profiles still exist and must not be renamed, removed, or upserted.
Return only target upserts and their reasons; an empty voices array is valid.
When contrast_findings or cross_group_contrast.findings is grouped, the compact
contrast encoding below applies independently to each grouped value.
"""


def design_adaptive_ensemble(source, cast, requests, *, policy, progress=lambda message: None):
    """Coordinate the root cast when possible, otherwise partition target IDs."""
    if len(cast["voices"]) < 2 or not any(not _inherited(voice) for voice in cast["voices"]):
        from .voice_ensemble import design_ensemble
        return design_ensemble(source, cast, requests, policy=policy, progress=progress)

    from .voice_contrast import contrast_findings
    findings_before = contrast_findings(cast["voices"])
    designable = [voice["id"] for voice in cast["voices"] if not _inherited(voice)]
    full_fingerprint = fingerprint(cast)
    limit = getattr(requests, "max_request_chars", 200000)
    validate_full = _validate_response(source, cast)
    root = {}

    def build(subset):
        targets = list(subset)
        selected = set(targets)
        if len(targets) == len(designable):
            payload = _payload(source, cast, policy, findings_before)
            system = ENSEMBLE_SYSTEM
            if request_size(system, payload) > limit:
                payload["contrast_findings"] = _contrast_context(findings_before)
                if isinstance(payload["contrast_findings"], dict):
                    system += COMPACT_CONTRAST_GUIDANCE
            root.update(system=system, size=request_size(system, payload))
        else:
            roles = [role for role in cast["roles"] if role["voice_id"] in selected]
            character_ids = {character["id"] for character in cast["characters"] if character["voice_id"] in selected}
            character_ids.update(role["character_id"] for role in roles if role["character_id"] is not None)
            scoped = {"characters": [character for character in cast["characters"] if character["id"] in character_ids],
                      "roles": roles, "voices": [voice for voice in cast["voices"] if voice["id"] in selected]}
            internal = [finding for finding in findings_before
                        if all(voice_id in selected for voice_id in finding["voice_ids"])]
            cross = [finding for finding in findings_before
                     if any(voice_id in selected for voice_id in finding["voice_ids"])
                     and any(voice_id not in selected for voice_id in finding["voice_ids"])]
            payload = _payload(source, scoped, policy, internal)
            # Root retains its legacy bounded character context. Partitions must
            # instead carry every selected record completely and losslessly.
            payload["cast"]["characters"] = deepcopy(scoped["characters"])
            payload["contrast_findings"] = _contrast_context(internal)
            payload["cross_group_contrast"] = {"read_only": True, "findings": _contrast_context(cross)}
            scope = {
                "kind": "target_voice_partition",
                "full_cast_fingerprint": full_fingerprint,
                "target_voice_ids": targets,
                "included_ids": {key: [row["id"] for row in scoped[key]] for key in scoped},
                "omitted_counts": {key: len(cast[key]) - len(scoped[key]) for key in scoped},
            }
            payload["cast_fingerprint"] = full_fingerprint
            payload["scope"] = scope
            system = ENSEMBLE_SYSTEM + GROUP_GUIDANCE
            if any(isinstance(value, dict) for value in
                   (payload["contrast_findings"], payload["cross_group_contrast"]["findings"])):
                system += COMPACT_CONTRAST_GUIDANCE
        if len(targets) == 1 and request_size(system, payload) > limit:
            raise ValueError(f"Voice ensemble target {targets[0]} needs {request_size(system, payload)} characters, "
                             f"exceeding the configured request budget ({limit}); its complete profile and mandatory "
                             "text-trait evidence cannot be partitioned or trimmed.")
        return system, payload

    def execute(system, payload):
        scope = payload.get("scope")
        if scope is None:
            validate = validate_full
        else:
            targets = set(scope["target_voice_ids"])
            def validate(response):
                if isinstance(response, dict) and isinstance(response.get("voices"), list):
                    for upsert in response["voices"]:
                        if (isinstance(upsert, dict) and isinstance(upsert.get("id"), str)
                                and upsert["id"] not in targets):
                            raise ValueError("Voice ensemble partition may upsert only its target voice IDs: "
                                             f"{upsert['id']}")
                return validate_full(response)
        accepted = requests.run(system, payload, validate)
        return {"accepted": accepted, "request_characters": request_size(system, payload),
                "request_fingerprint": fingerprint(payload), "guidance_fingerprint": fingerprint(system),
                "scope": deepcopy(scope)}

    progress("Designing the voice cast as one ensemble")
    results = run_partitioned(designable, build, execute, progress=progress,
                              label="voice ensemble", requests=requests)
    covered = [voice_id for subset, _ in results for voice_id in subset]
    if covered != designable:
        raise ValueError("Adaptive voice ensemble partitions must cover every designable voice exactly once")

    if len(results) == 1 and list(results[0][0]) == designable:
        updated = results[0][1]["accepted"]["cast"]
        reasons = deepcopy(results[0][1]["accepted"]["reasons"])
    else:
        updated = deepcopy(cast)
        reasons = {}
        for subset, result in results:
            accepted = result["accepted"]
            group_voices = {voice["id"]: voice for voice in accepted["cast"]["voices"]}
            upserts = []
            for voice_id in subset:
                voice = group_voices[voice_id]
                upserts.append({"id": voice_id, **{key: deepcopy(voice[key])
                                                   for key in ("description", "traits", "speechTraits") if key in voice}})
            updated = _build_cast(updated, upserts)
            reasons.update(deepcopy(accepted["reasons"]))
        check_cast(source, updated)

    groups = []
    for subset, result in results:
        groups.append({"target_voice_ids": list(subset), "scope": result["scope"],
                       "request_characters": result["request_characters"],
                       "request_fingerprint": result["request_fingerprint"],
                       "guidance_fingerprint": result["guidance_fingerprint"],
                       "reasons": deepcopy(result["accepted"]["reasons"]),
                       "cast_fingerprint_after": fingerprint(result["accepted"]["cast"])})
    original = {voice["id"]: voice for voice in cast["voices"]}
    changed = {}
    for voice in updated["voices"]:
        if voice != original[voice["id"]]:
            changed[voice["id"]] = {"reason": reasons[voice["id"]],
                                    "description_fingerprint": fingerprint(voice["description"]),
                                    "speech_traits_fingerprint": fingerprint(voice.get("speechTraits"))}
    return updated, {"version": VERSION, "contrast_version": CONTRAST_VERSION,
                     "status": "applied" if changed else "unchanged",
                     "source_fingerprint": fingerprint(source),
                     "cast_fingerprint_before": full_fingerprint,
                     "cast_fingerprint_after": fingerprint(updated),
                     "guidance_fingerprint": fingerprint(root["system"]),
                     "request_characters": root["size"], "updated": changed,
                     "contrast_findings_before": findings_before,
                     "contrast_findings_after": contrast_findings(updated["voices"]),
                     "adaptive": {"version": 1, "strategy": "binary_target_voice_partitions",
                                  "target_voice_ids": designable, "groups": groups}}
