"""Adaptive, source-region-independent cast candidate discovery.

The shared adaptive runner partitions the flattened source atoms, and every
request is independent: it sees only the shared seed cast plus its own region,
never accumulated cast or sibling context. Model responses use unbounded
response-local IDs (c1/r1/v1 with arbitrary suffixes) that this module
deterministically namespaces per source region before the shared DISCOVER
merge contract applies them with defer_merges=True. Region results combine
under strict no-last-write-win rules for shared seed records: differing
observations are preserved as explicit unresolved evidence, equal names never
merge, merges stay deferred for the identity jury, and every new candidate
carries exact visible source evidence validated before cache acceptance. Voice
entries stay brief placeholders; the whole-cast ensemble pass owns profiles.
"""

from copy import deepcopy
import json
import re

from .casting_prompts import DISCOVERY_PROMPT_VERSION, DISCOVERY_SYSTEM, ID_ALLOCATION_GUIDANCE, sample_available_ids
from .pipeline import _bounded_validation_errors, check_cast, check_evidence, fingerprint, merge_patch, seed_cast
from .prompts import SERIES_GUIDANCE, TYPOGRAPHY_GUIDANCE
from .series import series_context, validate_series_proposals


VERSION = 1
DISCOVERY_PASS_LABEL = "d"
COLLECTIONS = ("characters", "roles", "voices")
_ID_PREFIX = {"characters": "char", "roles": "role", "voices": "voice"}
_LOCAL_PREFIX = {"characters": "c", "roles": "r", "voices": "v"}
# IDs the application itself allocated: a pass label plus a region token (the
# supplied unit interval plus the input-cast fingerprint segment). A response
# referencing one that is not already canonical is reaching across regions and
# is rejected rather than silently merged.
_ALLOCATED_PATTERN = re.compile(r"^(?:char|role|voice)_[dc][0-9A-Za-z_-]*_.+$")
_LOCAL_PATTERN = {collection: re.compile(rf"^{prefix}[0-9A-Za-z_-]{{1,48}}$")
                  for collection, prefix in _LOCAL_PREFIX.items()}


def flatten_atoms(batches):
    """Flatten make_units batches into source atoms with stable original IDs."""
    return [unit for batch in batches for unit in batch]


def cast_ids(cast):
    return {collection: {record["id"] for record in cast[collection]} for collection in COLLECTIONS}


def _region_token(atoms, cast_fingerprint):
    """Collision-free region token: the supplied unit interval plus the input
    cast fingerprint segment that binds sequential newcomer namespaces."""
    return f"{atoms[0]['unit_id']}-{atoms[-1]['unit_id']}-{cast_fingerprint[:16]}"


def _region_size(system, payload):
    return len(system) + len(json.dumps(payload, ensure_ascii=False))


def _unit_intervals(atoms):
    """Per-unit [start, end) intervals per block; containment is judged against
    single supplied units, never merged block spans."""
    intervals = {}
    for unit in atoms:
        intervals.setdefault(unit["block_id"], []).append((unit["start"], unit["start"] + len(unit["text"])))
    return intervals


def _quote_visible_in_region(text, quote, intervals):
    """True when the quote occurs at some position fully inside one interval."""
    if not isinstance(quote, str) or not quote:
        return False
    return any(text.find(quote, low, high) != -1 for low, high in intervals)


def _block_intervals(atoms):
    """Merged source-relative [start, end) intervals per block for these units."""
    spans = {}
    for unit in atoms:
        spans.setdefault(unit["block_id"], []).append((unit["start"], unit["start"] + len(unit["text"])))
    intervals = []
    for block_id, runs in spans.items():
        cursor = None
        for start, end in sorted(runs):
            if cursor is None:
                cursor = [start, end]
            elif start <= cursor[1]:
                cursor[1] = max(cursor[1], end)
            else:
                intervals.append({"block_id": block_id, "start": cursor[0], "end": cursor[1]})
                cursor = [start, end]
        if cursor is not None:
            intervals.append({"block_id": block_id, "start": cursor[0], "end": cursor[1]})
    return intervals


def _has_typography(atoms):
    return any("inline_spans" in unit or "break_before" in unit for unit in atoms)


def _region_request(source, base_cast, policy, atoms, series_value=None, *,
                    source_fingerprint=None, cast_fingerprint=None):
    """Build the independent discovery request for one source region."""
    source_fingerprint = fingerprint(source) if source_fingerprint is None else source_fingerprint
    cast_fingerprint = fingerprint(base_cast) if cast_fingerprint is None else cast_fingerprint
    token = _region_token(atoms, cast_fingerprint)
    payload = {"mode": "casting_discovery", "version": VERSION,
               "prompt_version": DISCOVERY_PROMPT_VERSION,
               "book": deepcopy(source["book"]), "performance_policy": deepcopy(policy),
               "known_cast": deepcopy(base_cast), "available_ids": sample_available_ids(),
               "id_allocation": deepcopy(ID_ALLOCATION_GUIDANCE), "region": token,
               "target_units": deepcopy(atoms), "source_fingerprint": source_fingerprint}
    if series_value is not None:
        payload["series_cast"] = deepcopy(series_value)
    return payload, token


def _scope_ids(response, base_cast, label, token, *, supplied_ids=None):
    """Validate and namespace response-local IDs; reject cross-region references.

    Returns the scoped deep copy, the set of defined IDs per collection, and the
    newly allocated namespaced IDs per collection. Canonical IDs pass through as
    upserts; every other ID must be an unallocated response-local ID for its
    collection, which is deterministically namespaced for this region.
    """
    if not isinstance(response, dict):
        raise ValueError("Casting discovery must return a JSON object")
    for collection in COLLECTIONS:
        if not isinstance(response.get(collection), list):
            raise ValueError(f"Discovery needs an array named {collection}")
    if not isinstance(response.get("merges"), list) or not isinstance(response.get("issues"), list):
        raise ValueError("Discovery needs issues and merges arrays")
    base_ids = cast_ids(base_cast)
    scoped = deepcopy(response)
    defined = {collection: set() for collection in COLLECTIONS}
    allocated = {collection: [] for collection in COLLECTIONS}
    mapping = {}

    def allocate(collection, value, field):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a nonempty string ID")
        if value in base_ids[collection]:
            if supplied_ids is not None and value not in supplied_ids[collection]:
                raise ValueError(f"{field} references an existing ID not supplied in cast_view")
            return value
        if _ALLOCATED_PATTERN.match(value):
            raise ValueError(f"{field} references an allocated ID from another source region's "
                             "namespace; define your own response-local ID (for example c1, r2, v3) instead")
        if _LOCAL_PATTERN[collection].match(value) is None:
            expected = f"{_LOCAL_PREFIX[collection]}<suffix> (for example {_LOCAL_PREFIX[collection]}1)"
            raise ValueError(f"{field} must be an existing cast ID or an unallocated response-local ID matching "
                              f"{expected}")
        if (collection, value) not in mapping:
            namespaced = f"{_ID_PREFIX[collection]}_{label}{token}_{value}"
            mapping[(collection, value)] = namespaced
            allocated[collection].append(namespaced)
        return mapping[(collection, value)]

    def reference(target, value, field):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a nonempty string ID")
        if value in base_ids[target]:
            if supplied_ids is not None and value not in supplied_ids[target]:
                raise ValueError(f"{field} references an existing ID not supplied in cast_view")
            return value
        mapped = mapping.get((target, value))
        if mapped is not None:
            return mapped
        if _ALLOCATED_PATTERN.match(value):
            raise ValueError(f"{field} references an allocated ID from another source region; reference only "
                             "existing cast IDs or response-local IDs defined in this response")
        raise ValueError(f"{field} references neither an existing cast ID nor a "
                         f"{target[:-1]} defined in this response; define it in the {target} array first")

    for collection in COLLECTIONS:
        for index, record in enumerate(scoped[collection]):
            if not isinstance(record, dict):
                raise ValueError(f"Invalid {collection} record at index {index}")
            record["id"] = allocate(collection, record.get("id"), f"{collection}[{index}].id")
            defined[collection].add(record["id"])
    for index, record in enumerate(scoped["characters"]):
        record["voice_id"] = reference("voices", record.get("voice_id"), f"characters[{index}].voice_id")
    for index, record in enumerate(scoped["roles"]):
        character_id = record.get("character_id")
        if character_id is not None:
            record["character_id"] = reference("characters", character_id, f"roles[{index}].character_id")
        record["voice_id"] = reference("voices", record.get("voice_id"), f"roles[{index}].voice_id")
    for index, merge in enumerate(scoped["merges"]):
        for field in ("from_character_id", "into_character_id"):
            merge[field] = reference("characters", merge.get(field), f"merges[{index}].{field}")
    raw_findings = scoped.get("findings", [])
    if "findings" in scoped and not isinstance(raw_findings, list):
        raise ValueError("findings must be an array of anchored missing-speaker objects")
    for index, finding in enumerate(raw_findings):
        if not isinstance(finding, dict):
            raise ValueError(f"findings[{index}] must be an object with block_id, start, end, quote, and character_id")
        finding["character_id"] = reference("characters", finding.get("character_id"),
                                            f"findings[{index}].character_id")
    for issue in scoped["issues"]:
        target = issue.get("target") if isinstance(issue, dict) else None
        if not isinstance(target, str):
            continue
        for collection in COLLECTIONS:
            canonical = target in base_ids[collection]
            local = _LOCAL_PATTERN[collection].match(target) is not None
            if canonical or target in defined[collection] or local:
                issue["target"] = reference(collection, target, "issues[].target")
                break
    return scoped, defined, allocated


def _guard_canonical_records(scoped, base_cast):
    """Coverage protections: preserve canonical identities, profiles, and links."""
    for record in scoped["characters"]:
        base = next((row for row in base_cast["characters"] if row["id"] == record["id"]), None)
        if base is None:
            continue
        for field in ("name", "voice_id"):
            if record.get(field) != base.get(field):
                raise ValueError(f"characters[{record['id']}].{field} must stay unchanged in coverage; renaming or "
                                 "recasting an existing candidate is unsupported and requires identity review")
    for record in scoped["roles"]:
        base = next((row for row in base_cast["roles"] if row["id"] == record["id"]), None)
        if base is None:
            continue
        for field in ("kind", "character_id", "voice_id"):
            if record.get(field) != base.get(field):
                raise ValueError(f"roles[{record['id']}].{field} must stay unchanged in coverage; re-linking an "
                                 "existing role is unsupported and requires identity review")
    for record in scoped["voices"]:
        base = next((row for row in base_cast["voices"] if row["id"] == record["id"]), None)
        if base is not None and record != base:
            raise ValueError(f"voices[{record['id']}] must not be modified by coverage; the whole-cast ensemble "
                             "pass owns existing voice profiles")


def region_validator(source, base_cast, payload, *, label, strict_canonical, findings_normalizer=None):
    """Acceptance gate for one region response, shared by discovery and coverage.

    The returned closure scopes and namespaces IDs, enforces canonical-record
    protections when strict, requires exact nonempty evidence on every new
    character that is visible inside this region's supplied units, applies the
    shared merge contract with defer_merges=True, and runs full cast
    validation — so CachedRequests caches a response only after schema,
    evidence, and visibility all pass.
    """
    token = payload["region"]
    region_atoms = payload["target_units"]
    base_ids = cast_ids(base_cast)
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    region_intervals = _unit_intervals(region_atoms)
    supplied_ids = cast_ids(payload["cast_view"]["records"]) if "cast_view" in payload else None

    def require_region_visibility(items, field, failures, *, nonempty=False):
        if not isinstance(items, list) or (nonempty and not items):
            failures.append(field + ": nonempty evidence array required")
            return
        for index, item in enumerate(items):
            block_id = item.get("block_id") if isinstance(item, dict) else None
            location = f"{field}[{index}]"
            if isinstance(block_id, str) and block_id in blocks:
                location += "@" + block_id
            try:
                check_evidence([item], source)
            except ValueError:
                failures.append(location)
                continue
            if not _quote_visible_in_region(blocks[block_id], item["quote"], region_intervals.get(block_id, [])):
                failures.append(location)

    def validate(response):
        scoped, defined, allocated = _scope_ids(response, base_cast, label, token, supplied_ids=supplied_ids)
        if strict_canonical:
            _guard_canonical_records(scoped, base_cast)
        evidence_failures = []
        for record_index, record in enumerate(scoped["characters"]):
            if record["id"] not in base_ids["characters"]:
                require_region_visibility(record.get("evidence"), f"characters[{record_index}].evidence",
                                           evidence_failures, nonempty=True)
                for trait_index, trait in enumerate(record.get("traits", []) if isinstance(record.get("traits"), list) else []):
                    items = trait.get("evidence") if isinstance(trait, dict) else None
                    if isinstance(items, list) and items:
                        require_region_visibility(items, f"characters[{record_index}].traits[{trait_index}].evidence",
                                                   evidence_failures)
        previous_voices = {record["id"]: record for record in base_cast["voices"]}
        for voice in scoped["voices"]:
            previous_traits = previous_voices.get(voice["id"], {}).get("traits", [])
            traits = voice.get("traits", [])
            for index, trait in enumerate(traits if isinstance(traits, list) else []):
                if isinstance(trait, dict) and trait not in previous_traits and isinstance(trait.get("evidence"), list):
                    require_region_visibility(trait["evidence"], f"voices[{voice['id']}].traits[{index}].evidence",
                                               evidence_failures)
        for merge_index, merge in enumerate(scoped["merges"]):
            if isinstance(merge, dict) and isinstance(merge.get("evidence"), list) and merge["evidence"]:
                require_region_visibility(merge["evidence"], f"merges[{merge_index}].evidence", evidence_failures)
        if evidence_failures:
            raise ValueError(_bounded_validation_errors(evidence_failures,
                " Evidence quotations must be exact and visible in the supplied units; correct every listed field."))
        available = {collection: sorted(base_ids[collection] | set(allocated[collection]))
                     for collection in COLLECTIONS}
        updated, issues, summary = merge_patch(source, base_cast, scoped, available, defer_merges=True)
        accepted = {"response": scoped, "cast": updated, "issues": issues, "summary": summary,
                    "merges": deepcopy(scoped["merges"]), "allocation": {"label": label, "token": token}}
        if findings_normalizer is not None:
            visible_characters = supplied_ids["characters"] if supplied_ids is not None else base_ids["characters"]
            accepted["findings"] = findings_normalizer(scoped, visible_characters, set(allocated["characters"]), region_atoms)
            certified = scoped.get("certified", False)
            if type(certified) is not bool:
                raise ValueError("certified must be a boolean when present; set it true only after every "
                                 "target unit was checked")
            accepted["certified"] = certified
        return accepted

    return validate


def new_combine_state():
    return {"variants": {}, "added": {collection: [] for collection in COLLECTIONS},
            "added_ids": set(), "observations": [], "unapplied": [],
            "narration_candidates": [], "narration_scopes": []}


def _carried_record(collection, base_record, record):
    """Apply an upsert the way merge_patch does, keeping stored profile fields."""
    merged = deepcopy(record)
    if collection == "voices":
        for field in ("speechTraits", "traits"):
            if field not in merged and base_record is not None and field in base_record:
                merged[field] = deepcopy(base_record[field])
        merged.setdefault("traits", [])
    return merged


def combine_region_results(source, base_cast, results, state, *, kept_characters=None,
                           apply_shared_updates=True):
    """Combine independent region responses without any last-write-win.

    Shared canonical records follow unanimity: identical upserts apply once,
    differing observations are preserved in state and surfaced as explicit
    unresolved issues while the shared record stays as-is. With
    apply_shared_updates=False (coverage), no differing shared upsert ever
    applies: a single observation becomes an identity_proposal issue instead.
    New candidates are appended in region order; when kept_characters is
    supplied (coverage), only candidates reachable from processed findings
    enter the ledger. Conflicting narration-role bindings on a shared role
    keep the shared role unchanged and preserve each region's binding as a new
    scoped narration-role candidate bound to that region's original units.
    Returns the rebuilt ledger and this round's combiner issues.
    """
    base_records = {collection: {record["id"]: record for record in base_cast[collection]}
                    for collection in COLLECTIONS}
    issues = []
    for _subset, accepted in results:
        response = accepted["response"]
        for collection in COLLECTIONS:
            for record in response[collection]:
                base_record = base_records[collection].get(record["id"])
                if base_record is None or record == base_record:
                    continue
                bucket = state["variants"].setdefault((collection, record["id"]), [])
                blob = json.dumps(record, sort_keys=True, ensure_ascii=False)
                if any(blob == seen for seen, _record in bucket):
                    continue
                bucket.append((blob, deepcopy(record)))
                if len(bucket) > 1:
                    state["observations"].append({"collection": collection, "id": record["id"],
                                                  "records": [deepcopy(item) for _seen, item in bucket]})
                    issues.append({"kind": "shared_record_conflict", "target": record["id"],
                                   "reason": f"{len(bucket)} independent source regions proposed differing updates "
                                   f"for the shared {collection[:-1]} record {record['id']}; the shared record is "
                                   "preserved unchanged and all observations are retained in provenance for review."})
                elif not apply_shared_updates:
                    state["observations"].append({"collection": collection, "id": record["id"],
                                                  "records": [deepcopy(record)]})
                    issues.append({"kind": "identity_proposal", "target": record["id"],
                                   "reason": f"A coverage region proposed a metadata change for the existing "
                                   f"{collection[:-1]} record {record['id']}; the shared record is preserved "
                                   "byte-identical and the proposal is retained in provenance for review."})
        for role in response["roles"]:
            base_record = base_records["roles"].get(role.get("id"))
            if (base_record is None or base_record.get("kind") != "narration" or role == base_record
                    or not isinstance(role, dict) or not role.get("character_id")):
                continue
            state["narration_candidates"].append((list(_subset), accepted["allocation"]["token"], deepcopy(role)))
    for _subset, accepted in results:
        response = accepted["response"]
        keep = kept_characters(accepted) if kept_characters is not None else None
        new_ids = {collection: {record["id"] for record in response[collection]
                                if record["id"] not in base_records[collection]}
                   for collection in COLLECTIONS}
        if keep is None:
            selected = new_ids
        else:
            characters = new_ids["characters"] & set(keep)
            roles = {record["id"] for record in response["roles"]
                     if record["id"] in new_ids["roles"] and record.get("character_id") in characters}
            referenced_voices = {record.get("voice_id") for record in response["characters"]
                                 if record["id"] in characters}
            referenced_voices.update(record.get("voice_id") for record in response["roles"] if record["id"] in roles)
            voices = new_ids["voices"] & referenced_voices
            selected = {"characters": characters, "roles": roles, "voices": voices}
            for collection in COLLECTIONS:
                for dropped in sorted(new_ids[collection] - selected[collection]):
                    state["unapplied"].append({"collection": collection, "id": dropped,
                                               "reason": "no processed finding references this candidate"})
        for collection in COLLECTIONS:
            by_id = {record["id"]: record for record in response[collection]}
            for record_id in sorted(selected[collection]):
                if (collection, record_id) in state["added_ids"]:
                    continue
                state["added"][collection].append(_carried_record(collection, None, by_id[record_id]))
                state["added_ids"].add((collection, record_id))
    for candidate_subset, _token, role in state["narration_candidates"]:
        bucket = state["variants"].get(("roles", role["id"]), [])
        if len(bucket) <= 1 or ( "roles", f"role_d{_token}_{role['id']}" ) in state["added_ids"]:
            continue
        scoped_role = {"id": f"role_d{_token}_{role['id']}", "kind": "narration",
                       "character_id": role["character_id"], "voice_id": role["voice_id"],
                       "description": role.get("description", ""),
                       "source_scope": {"unit_ids": [unit["unit_id"] for unit in candidate_subset],
                                        "intervals": _block_intervals(candidate_subset)}}
        state["added"]["roles"].append(scoped_role)
        state["added_ids"].add(("roles", scoped_role["id"]))
        state["narration_scopes"].append({"role_id": scoped_role["id"], "character_id": role["character_id"],
                                          "unit_ids": scoped_role["source_scope"]["unit_ids"],
                                          "intervals": scoped_role["source_scope"]["intervals"]})
    ledger = deepcopy(base_cast)
    for (collection, record_id), bucket in state["variants"].items():
        if len(bucket) == 1 and apply_shared_updates:
            index = next(i for i, record in enumerate(ledger[collection]) if record["id"] == record_id)
            ledger[collection][index] = _carried_record(collection, base_records[collection][record_id], bucket[0][1])
    for collection in COLLECTIONS:
        ledger[collection].extend(state["added"][collection])
    check_cast(source, ledger)
    return ledger, issues


def _add_issue(issues, issue):
    key = fingerprint({"kind": issue["kind"], "target": issue["target"], "reason": issue["reason"]})
    if key not in {fingerprint({"kind": item["kind"], "target": item["target"], "reason": item["reason"]})
                   for item in issues}:
        issues.append(deepcopy(issue))


def _preflight_budget(build, atoms, limit, label):
    smallest = min(atoms, key=lambda unit: len(unit["text"]))
    system, payload = build([smallest])
    size = _region_size(system, payload)
    if size > limit:
        raise ValueError(f"The smallest {label} request is {size} characters, exceeding the configured request "
                         f"budget ({limit}). The supplied cast is never trimmed or dropped to fit; "
                         "raise analysis.max_request_chars.")


def discover_candidates(source, requests, batches, *, policy, series_registry=None,
                        progress=lambda message: None):
    """Discover cast candidates with independent, source-bounded region requests.

    Returns {"cast", "pending_merges", "scene_summaries", "regions", "provenance"}.
    regions lists the actual source-unit lists the runner produced; each scene
    summary binds its region's unit IDs and source intervals. Region and
    combiner review issues live under provenance["issues"], with conflicting
    shared-record observations under provenance["shared_record_observations"]
    and region-scoped narration-role candidates under
    provenance["narration_scopes"]. Never edits the source and never performs
    voice design.
    """
    from .adaptive_requests import run_partitioned

    atoms = flatten_atoms(batches)
    if not atoms:
        raise ValueError("Cast discovery needs at least one source unit")
    seed = seed_cast()
    series_value = series_context(series_registry, source) if series_registry is not None else None
    system = DISCOVERY_SYSTEM + (TYPOGRAPHY_GUIDANCE if _has_typography(atoms) else "")
    if series_value is not None:
        system += SERIES_GUIDANCE
    limit = getattr(requests, "max_request_chars", 200000)
    source_fingerprint = fingerprint(source)
    seed_fingerprint = fingerprint(seed)

    def build(subset):
        payload, _token = _region_request(source, seed, policy, subset, series_value,
                                          source_fingerprint=source_fingerprint,
                                          cast_fingerprint=seed_fingerprint)
        return system, payload

    def execute(system_value, payload):
        base_validator = region_validator(source, seed, payload, label=DISCOVERY_PASS_LABEL,
                                          strict_canonical=False, findings_normalizer=None)

        def validate(response):
            accepted = base_validator(response)
            if series_registry is not None:
                validate_series_proposals(source, accepted["cast"], series_registry)
            return accepted

        return requests.run(system_value, payload, validate)

    fits = lambda system_value, payload: _region_size(system_value, payload) <= limit
    _preflight_budget(build, atoms, limit, "cast discovery")
    progress(f"Discovering cast candidates across {len(atoms)} source unit(s)")
    results = run_partitioned(atoms, build, execute, fits=fits, progress=progress,
                              label="Casting discovery", requests=requests)
    state = new_combine_state()
    ledger, combine_issues = combine_region_results(source, seed, results, state)
    if series_registry is not None:
        validate_series_proposals(source, ledger, series_registry)
    issues = []
    for _subset, accepted in results:
        for issue in accepted["issues"]:
            _add_issue(issues, issue)
    for issue in combine_issues:
        _add_issue(issues, issue)
    pending_merges, seen_merges = [], set()
    regions, scene_summaries, region_provenance = [], [], []
    for index, (subset, accepted) in enumerate(results):
        token = accepted["allocation"]["token"]
        regions.append(subset)
        scene_summaries.append({"region": index + 1, "token": token,
                                "unit_ids": [unit["unit_id"] for unit in subset],
                                "intervals": _block_intervals(subset), "summary": accepted["summary"]})
        region_provenance.append({"region": index + 1, "token": token,
                                  "unit_ids": [unit["unit_id"] for unit in subset],
                                  "unit_count": len(subset)})
        for merge in accepted["merges"]:
            blob = json.dumps(merge, sort_keys=True, ensure_ascii=False)
            if blob not in seen_merges:
                seen_merges.add(blob)
                pending_merges.append(deepcopy(merge))
    provenance = {"version": VERSION, "prompt_version": DISCOVERY_PROMPT_VERSION,
                  "guidance_fingerprint": fingerprint(system), "source_fingerprint": source_fingerprint,
                  "cast_fingerprint": fingerprint(ledger), "regions": region_provenance,
                  "shared_record_observations": state["observations"],
                  "narration_scopes": state["narration_scopes"], "issues": issues}
    return {"cast": ledger, "pending_merges": pending_merges, "scene_summaries": scene_summaries,
            "regions": regions, "provenance": provenance}
