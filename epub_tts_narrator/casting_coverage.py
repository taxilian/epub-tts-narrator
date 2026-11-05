"""Finite, source-anchored cast coverage verification.

An independent scan checks the entire source once through the shared adaptive
runner. Requests disclose the full candidate view when it fits the request
budget and otherwise a complete-record subset selected by cast_scope.scoped_cast
— findings always apply to the full canonical cast either way. Existing
bindings are validated against the records the supplied view actually contains,
never against unseen canonical IDs. Missing-speaker findings are exact
source-anchored spans fully contained in a single supplied unit; deterministic
finding IDs derive from the source fingerprint and anchor offsets, never from
model reasons, and error messages never echo quotes or source excerpts. Each
finding is processed at most once; only dirty source atoms are re-verified
after applied changes; repeated findings, ledger cycles, and non-binding
anchors become explicit unresolved review issues instead of an unbounded loop.
Termination needs no user-facing round knob: every round either adds a
never-processed anchor (a finite supply) or stops, and an identical
(scan, findings, ledger) round state is a detected cycle. The coverage
allocation namespace binds the input cast fingerprint so sequential newcomers
reported as the same response-local ID in one atom never overwrite each other.
Missing or empty findings never count as certification: a region is clean only
when its scan explicitly certifies that every target unit was checked.
"""

from copy import deepcopy
import json

from .cast_scope import scoped_cast
from .casting_discovery import (_add_issue, _has_typography, _preflight_budget, _region_size, _unit_intervals,
                                cast_ids, combine_region_results, flatten_atoms, new_combine_state,
                                region_validator)
from .casting_prompts import COVERAGE_PROMPT_VERSION, COVERAGE_SYSTEM, ID_ALLOCATION_GUIDANCE, sample_available_ids
from .pipeline import check_cast, fingerprint
from .prompts import TYPOGRAPHY_GUIDANCE


VERSION = 2
COVERAGE_PASS_LABEL = "c"
_MAX_FINDING_REASON = 1000


def finding_id(source_fingerprint, block_id, start, end):
    """Deterministic anchor identity, independent of any model-supplied reason."""
    return "cov_" + fingerprint([source_fingerprint, block_id, start, end])[:16]


def _region_request(source, cast, policy, atoms, *, source_fingerprint=None, cast_fingerprint=None,
                    scoped=False, review_notes=""):
    """Build the coverage request for one source region.

    The candidate view discloses exactly which records it contains. With
    scoped=False the view is the complete cast; with scoped=True it is the
    complete-record subset cast_scope selects for these units. Records are
    never trimmed either way.
    """
    source_fingerprint = fingerprint(source) if source_fingerprint is None else source_fingerprint
    cast_fingerprint = fingerprint(cast) if cast_fingerprint is None else cast_fingerprint
    token = f"{atoms[0]['unit_id']}-{atoms[-1]['unit_id']}-{cast_fingerprint[:16]}"
    if scoped:
        view, scope = scoped_cast(source, cast, atoms)
        omitted = sorted({record["id"] for record in cast["characters"]} - set(scope["included_ids"]))
        cast_view = {"complete": scope["complete"], "full_cast_fingerprint": scope["full_cast_fingerprint"],
                     "included_character_ids": scope["included_ids"], "omitted_character_ids": omitted,
                     "disclosure": "Complete-record subset selected from the full cast for these target units; "
                                   "every supplied character, role, and voice record is entire and untrimmed, "
                                   "and other source regions are screened separately.",
                     "records": view}
    else:
        cast_view = {"complete": True, "full_cast_fingerprint": cast_fingerprint,
                     "included_character_ids": sorted(record["id"] for record in cast["characters"]),
                     "omitted_character_ids": [],
                     "disclosure": "Complete candidate records; nothing is omitted from this view.",
                     "records": deepcopy(cast)}
    payload = {"mode": "casting_coverage", "version": VERSION,
               "prompt_version": COVERAGE_PROMPT_VERSION,
               "book": deepcopy(source["book"]), "performance_policy": deepcopy(policy),
               "cast_view": cast_view, "available_ids": sample_available_ids(),
               "id_allocation": deepcopy(ID_ALLOCATION_GUIDANCE), "region": token,
               "target_units": deepcopy(atoms), "source_fingerprint": source_fingerprint}
    if review_notes:
        payload["review_notes"] = review_notes
    return payload, token


def _anchor_atom_indices(atoms, block_id, start, end):
    return [index for index, unit in enumerate(atoms)
            if unit["block_id"] == block_id and unit["start"] < end and start < unit["start"] + len(unit["text"])]


def _contiguous_unit_groups(atoms, indices):
    groups, current = [], []
    for index in sorted(indices):
        if current and index != current[-1] + 1:
            groups.append([atoms[i] for i in current])
            current = []
        current.append(index)
    if current:
        groups.append([atoms[i] for i in current])
    return groups


def verify_coverage(source, cast, requests, batches, *, policy, progress=lambda message: None,
                    review_notes=""):
    """Verify source coverage of the candidate cast with finite anchor-bound repair.

    Returns {"cast", "pending_merges", "dirty_regions", "provenance", "issues"}.
    dirty_regions lists the actual source-unit groups whose anchors received
    applied changes, for the follow-up identity jury pass. Never edits the
    source, never redesigns voices, and never runs the identity jury itself.
    """
    from .adaptive_requests import run_partitioned

    if not isinstance(review_notes, str):
        raise ValueError("review_notes must be a string")
    check_cast(source, cast)
    atoms = flatten_atoms(batches)
    if not atoms:
        raise ValueError("Coverage verification needs at least one source unit")
    source_fingerprint = fingerprint(source)
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    system = COVERAGE_SYSTEM + (TYPOGRAPHY_GUIDANCE if _has_typography(atoms) else "")
    limit = getattr(requests, "max_request_chars", 200000)

    def normalize_findings(scoped, supplied_ids, defined_characters, region_atoms):
        raw = scoped.get("findings", [])
        if "findings" in scoped and not isinstance(raw, list):
            raise ValueError("findings must be an array of anchored missing-speaker objects")
        unit_intervals = _unit_intervals(region_atoms)
        findings = []
        seen = set()
        for index, item in enumerate(raw):
            location = f"findings[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{location} must be an object with block_id, start, end, quote, and character_id")
            block_id = item.get("block_id")
            if not isinstance(block_id, str) or block_id not in blocks:
                raise ValueError(f"{location}.block_id: unknown source block ID; anchor findings to blocks covering "
                                 "this region's target_units")
            text = blocks[block_id]
            start, end = item.get("start"), item.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f"{location}.start/end must be exact zero-based end-exclusive offsets into block "
                                 f"{block_id} (length {len(text)})")
            quote = item.get("quote")
            if not isinstance(quote, str) or quote != text[start:end]:
                received = "a string" if isinstance(quote, str) else type(quote).__name__
                raise ValueError(f"{location}.quote must equal the exact source text of block {block_id} at offsets "
                                 f"{start}:{end}; received {received} of the wrong span. Copy the anchored span "
                                 "verbatim.")
            if not any(low <= start and end <= high for low, high in unit_intervals.get(block_id, [])):
                raise ValueError(f"{location}: anchor {block_id}:{start}:{end} lies outside this region's "
                                 "target_units or is not fully contained in a single supplied unit; report only "
                                 "anchors fully inside the supplied units")
            character_id = item.get("character_id")
            if (not isinstance(character_id, str) or not character_id
                    or (character_id not in supplied_ids and character_id not in defined_characters)):
                raise ValueError(f"{location}.character_id must reference a character supplied in cast_view or a "
                                 "character defined in this response's characters array")
            reason = item.get("reason", "")
            if not isinstance(reason, str) or len(reason) > _MAX_FINDING_REASON:
                raise ValueError(f"{location}.reason must be a brief string of at most {_MAX_FINDING_REASON} characters")
            identifier = finding_id(source_fingerprint, block_id, start, end)
            if identifier in seen:
                raise ValueError(f"{location}: duplicate source anchor in findings")
            seen.add(identifier)
            findings.append({"finding_id": identifier,
                             "block_id": block_id, "start": start, "end": end,
                             "character_id": character_id, "reason": reason})
        return findings

    state = new_combine_state()
    processed, issues, pending_merges = {}, [], []
    seen_merges, finding_records, region_provenance = set(), [], []
    dirty_indices, rounds, budget_exhausted = set(), 0, False
    seen_round_states = set()
    scan = atoms
    ledger = deepcopy(cast)
    while True:
        rounds += 1
        progress(f"Casting coverage scan round {rounds} over {len(scan)} source unit(s)")
        region_cast = ledger
        region_cast_fingerprint = fingerprint(region_cast)
        region_ids = cast_ids(region_cast)["characters"]

        def build(subset, _cast=region_cast, _cast_fingerprint=region_cast_fingerprint):
            payload, _token = _region_request(source, _cast, policy, subset,
                                              source_fingerprint=source_fingerprint,
                                              cast_fingerprint=_cast_fingerprint, scoped=False,
                                              review_notes=review_notes)
            if _region_size(system, payload) <= limit:
                return system, payload
            payload, _token = _region_request(source, _cast, policy, subset,
                                              source_fingerprint=source_fingerprint,
                                              cast_fingerprint=_cast_fingerprint, scoped=True,
                                              review_notes=review_notes)
            return system, payload

        def execute(system_value, payload, _cast=region_cast):
            validator = region_validator(source, _cast, payload, label=COVERAGE_PASS_LABEL,
                                         strict_canonical=True, findings_normalizer=normalize_findings)
            return requests.run(system_value, payload, validator)

        fits = lambda system_value, payload: _region_size(system_value, payload) <= limit
        if rounds == 1:
            _preflight_budget(build, atoms, limit, "coverage verification")
        results = run_partitioned(scan, build, execute, fits=fits, progress=progress,
                                  label="Casting coverage", requests=requests)
        new_findings = []
        for subset, accepted in results:
            for merge in accepted["merges"]:
                blob = json.dumps(merge, sort_keys=True, ensure_ascii=False)
                if blob not in seen_merges:
                    seen_merges.add(blob)
                    pending_merges.append(deepcopy(merge))
            token = accepted["allocation"]["token"]
            region_provenance.append({"round": rounds, "token": token,
                                      "unit_ids": [unit["unit_id"] for unit in subset],
                                      "unit_count": len(subset), "certified": accepted["certified"],
                                      "findings": len(accepted["findings"])})
            if not accepted["certified"]:
                _add_issue(issues, {"kind": "coverage_scan_uncertified", "target": token,
                                    "reason": "The coverage scan for this region did not explicitly certify that "
                                    "every target unit was checked; treat coverage as unverified for these units."})
            for issue in accepted["issues"]:
                _add_issue(issues, issue)
            for finding in accepted["findings"]:
                identifier = finding["finding_id"]
                if identifier in processed:
                    processed[identifier]["repeats"] += 1
                    _add_issue(issues, {"kind": "coverage_finding_unresolved", "target": identifier,
                                        "reason": f"The source anchor {finding['block_id']}:{finding['start']}:"
                                        f"{finding['end']} was reported again after it was processed once; the "
                                        "applied change did not clear it and requires manual review."})
                else:
                    new_findings.append(finding)
        kept = {finding["character_id"] for finding in new_findings}
        ledger, combine_issues = combine_region_results(source, cast, results, state,
                                                        kept_characters=lambda accepted: kept,
                                                        apply_shared_updates=False)
        for issue in combine_issues:
            _add_issue(issues, issue)
        if not new_findings:
            break
        round_state = fingerprint({"scan": [unit["unit_id"] for unit in scan],
                                   "findings": sorted(finding["finding_id"] for finding in new_findings),
                                   "cast": fingerprint(ledger)})
        if round_state in seen_round_states:
            budget_exhausted = True
            _add_issue(issues, {"kind": "coverage_repair_cycle", "target": "verify_coverage",
                                "reason": "Coverage repair reproduced an identical scan, finding, and ledger state; "
                                "stopping the cycle. Remaining anchors require manual review."})
            break
        seen_round_states.add(round_state)
        newly_dirty = set()
        for finding in new_findings:
            disposition = "binding" if finding["character_id"] in region_ids else "addition"
            record = {"finding_id": finding["finding_id"], "block_id": finding["block_id"],
                      "start": finding["start"], "end": finding["end"], "character_id": finding["character_id"],
                      "disposition": disposition, "round": rounds, "repeats": 0}
            processed[finding["finding_id"]] = record
            finding_records.append(record)
            newly_dirty.update(_anchor_atom_indices(atoms, finding["block_id"], finding["start"], finding["end"]))
        dirty_indices.update(newly_dirty)
        scan = [atoms[index] for index in sorted(newly_dirty)]
    ledger_ids = {record["id"] for record in ledger["characters"]}
    pending_merges = [merge for merge in pending_merges
                      if {merge["from_character_id"], merge["into_character_id"]} <= ledger_ids]
    provenance = {"version": VERSION, "prompt_version": COVERAGE_PROMPT_VERSION,
                  "guidance_fingerprint": fingerprint(system), "source_fingerprint": source_fingerprint,
                  "cast_fingerprint_before": fingerprint(cast), "cast_fingerprint_after": fingerprint(ledger),
                  "rounds": rounds, "round_budget_exhausted": budget_exhausted,
                  "regions": region_provenance, "findings": finding_records,
                  "shared_record_observations": state["observations"],
                  "unapplied_candidates": state["unapplied"]}
    return {"cast": ledger, "pending_merges": pending_merges,
            "dirty_regions": _contiguous_unit_groups(atoms, dirty_indices),
            "provenance": provenance, "issues": issues}
