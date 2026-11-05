"""Source-bounded attribution recovery without a book-independent depth ceiling."""

from concurrent.futures import CancelledError
from copy import deepcopy

from .pipeline import align_attribution_response, fingerprint


SCOPE_GUIDANCE = """
final_cast is an explicitly scoped view when cast_scope is present. Omitted
records still exist in the canonical cast; this request does not establish their
absence. Use only supplied role IDs and source evidence. Keep genuinely unknown
speakers unresolved rather than inventing a role or using a fallback narrator.
"""


def attribute_adaptively(source, cast, requests, system, payload, *, sequential_prior=False,
                         cancelled=None, progress=lambda message: None, validator=None, legacy=None):
    from .adaptive_requests import capacity_error, request_size, run_partitioned
    from .attribution_compact import compact_cast
    from .attribution_fallback import _validate_combined

    validate = validator or align_attribution_response
    units = payload["target_units"]
    accepted_prior = []
    scoped = []
    observations = []

    def check_cancelled():
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Attribution was cancelled")

    check_cancelled()
    if legacy is not None:
        legacy_system, legacy_payload = legacy
        found, accepted = requests.cached(
            legacy_system, legacy_payload,
            lambda response: align_attribution_response(source, cast, units, response), cancelled=cancelled)
        if found:
            return accepted, None

    def build(selected):
        check_cancelled()
        prior = payload["prior_segments"]
        if sequential_prior:
            prior = (prior + accepted_prior)[-6:]
        current = dict(payload, target_units=selected, prior_segments=deepcopy(prior))
        current_system = system
        if request_size(system, current) > requests.max_request_chars:
            from .cast_scope import scoped_cast
            view, scope = scoped_cast(source, cast, selected + current["context_units"])
            current["final_cast"] = compact_cast(view)
            current["cast_scope"] = scope
            current_system += SCOPE_GUIDANCE
        return current_system, current

    def execute(current_system, current):
        check_cancelled()
        selected = current["target_units"]
        visible = {collection: {record["id"] for record in current["final_cast"][collection]}
                   for collection in ("characters", "roles", "voices")}
        request_cast = {collection: [record for record in cast[collection] if record["id"] in visible[collection]]
                        for collection in visible}
        result = requests.run(current_system, current,
                              lambda response: validate(source, request_cast, selected, response), cancelled=cancelled)
        accepted_prior.extend(result)
        scoped.append(current.get("cast_scope"))
        return result

    try:
        leaves = run_partitioned(units, build, execute, requests=requests, progress=progress,
                                 label="Attributing source regions", observe=observations.append)
    except ValueError as error:
        if error.__cause__ is not None and capacity_error(error.__cause__):
            raise error.__cause__ from None
        raise
    combined = [segment for _, segments in leaves for segment in segments]
    _validate_combined(source, cast, units, combined)
    if len(leaves) == 1 and not scoped[0]:
        return combined, None
    return combined, {"version": 2, "strategy": "source_bounded",
                      "source_fingerprint": fingerprint(source), "cast_fingerprint": fingerprint(cast),
                      "automatic_tree_node_bound": 2 * len(units) - 1,
                      "prior_segment_mode": "serial" if sequential_prior else "independent",
                      "subdivision_observations": observations,
                      "truncation_observation": {"replayed_hint": bool(observations)
                                                  and all(row["replayed_hint"] for row in observations)},
                      "regions": [{"target_unit_ids": [unit["unit_id"] for unit in selected],
                                   "segment_count": len(segments), "cast_scope": scope}
                                  for (selected, segments), scope in zip(leaves, scoped)]}
