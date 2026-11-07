"""Resolve independent discovery candidates before source-backed coverage review."""

from copy import deepcopy
import re

from .pipeline import check_cast, fingerprint, merge_patch


VERSION = 1


def _add_issues(issues, additions):
    known = {row["id"] for row in issues}
    for item in additions:
        identifier = item.get("id", "issue_" + fingerprint(item)[:16])
        if identifier not in known:
            issues.append(dict(item, id=identifier, resolved=item.get("resolved", False)))
            known.add(identifier)


def _metered_identity_progress(progress, meter, base, span):
    """Advance the overall meter from identity vote and collision progress."""
    def report(message):
        progress(message)
        vote = re.search(r"(?:Series )?Identity jury[^:]*: (\d+)/(\d+) reviewer responses complete$", message)
        set_complete = re.search(r"Collision jury: (\d+)/(\d+) reviewer response set\(s\) complete$", message)
        wave = re.match(r"Collision wave \d+: (\d+) merge hypothesis case", message)
        if vote is not None:
            done, total = int(vote.group(1)), max(1, int(vote.group(2)))
            meter(base + span * 0.75 * done / total,
                  f"Identity jury {done}/{total}")
        elif set_complete is not None:
            done, total = int(set_complete.group(1)), max(1, int(set_complete.group(2)))
            meter(base + span * (0.75 + 0.25 * done / total),
                  f"Collision jury {done}/{total} reviewer sets")
        elif wave is not None:
            meter(base + span * 0.75, f"Collision review {int(wave.group(1))} cases")
    return report


def _unreviewed_merges(source, cast, pending, issues):
    """Retain the explicitly disabled jury's existing, advisory merge behavior."""
    if not pending:
        return cast
    patch = {"characters": [], "roles": [], "voices": [], "merges": pending,
             "issues": [], "summary": "Explicitly disabled independent identity review."}
    updated, additions, _ = merge_patch(source, cast, patch,
                                        {key: [] for key in ("characters", "roles", "voices")})
    _add_issues(issues, additions)
    return updated


def gather_cast(source, requests, batches, *, policy, series_registry=None,
                identity_consensus=True, workers=1, review_notes="", progress=lambda message: None,
                meter=lambda fraction, stage: None):
    """Build a canonical cast without a configured number of discovery passes."""
    from .casting_discovery import discover_candidates
    from .casting_coverage import verify_coverage
    from .casting_voices import consolidate_merged_voices
    from .identity import audit_identities

    discovered = discover_candidates(source, requests, batches, policy=policy,
                                     series_registry=series_registry,
                                     workers=workers, progress=progress, meter=meter)
    meter(0.25, "Auditing identities")
    cast = discovered["cast"]
    issues = []
    _add_issues(issues, discovered["provenance"]["issues"])
    pending = discovered["pending_merges"]
    identity_passes = []
    collision_state = None
    if identity_consensus:
        identity_progress = _metered_identity_progress(progress, meter, 0.25, 0.07)
        cast, additions, identity = audit_identities(
            source, cast, requests, batches, adaptive=True, pending_merges=pending,
            workers=workers, review_notes=review_notes, progress=identity_progress)
        identity_passes.append(identity)
        collision_state = identity.get("collision_resolution", {}).get("state")
        _add_issues(issues, additions)
    else:
        cast = _unreviewed_merges(source, cast, pending, issues)
    check_cast(source, cast)
    meter(0.32, "Casting coverage")

    before_coverage = deepcopy(cast)
    coverage = verify_coverage(source, cast, requests, batches, policy=policy,
                               review_notes=review_notes, progress=progress)
    cast = coverage["cast"]
    meter(0.40, "Rechecking identities")
    _add_issues(issues, coverage.get("issues", []))
    pending = coverage["pending_merges"]
    dirty = coverage["dirty_regions"]
    if identity_consensus and (dirty or pending):
        identity_progress = _metered_identity_progress(progress, meter, 0.40, 0.03)
        cast, additions, identity = audit_identities(
            source, cast, requests, batches, adaptive=True, scan_regions=dirty,
            pending_merges=pending, workers=workers,
            collision_state=collision_state,
            review_notes=review_notes, progress=identity_progress)
        identity_passes.append(identity)
        collision_state = identity.get("collision_resolution", {}).get("state")
        _add_issues(issues, additions)
    elif not identity_consensus:
        cast = _unreviewed_merges(source, cast, pending, issues)
    approved_merges = []
    for item in identity_passes:
        for case in item.get("cases", []):
            if case.get("status") != "applied":
                continue
            if case.get("origin") == "identity_collision_screen":
                approved_merges.append(case["merge"])
            else:
                approved_merges.extend(case["patch"]["merges"])
    cast, archived_voices, voice_issues = consolidate_merged_voices(
        cast, [discovered["cast"], before_coverage, coverage["cast"]], approved_merges)
    _add_issues(issues, voice_issues)
    voice_targets = {row["voice"]["id"]: row["canonical_voice_id"] for row in archived_voices}
    for issue in issues:
        if issue["target"] in voice_targets:
            original_target = issue["target"]
            issue.setdefault("identity_original_target", original_target)
            issue["target"] = voice_targets[original_target]
    if identity_consensus and series_registry is not None:
        meter(0.43, "Series identity")
        identity_progress = _metered_identity_progress(progress, meter, 0.43, 0.02)
        cast, additions, continuity = audit_identities(
            source, cast, requests, batches, adaptive=True, scan_regions=[],
            collision_state=collision_state, resolve_candidate_collisions=False,
            series_registry=series_registry, workers=workers, review_notes=review_notes,
            progress=identity_progress)
        identity_passes.append(continuity)
        _add_issues(issues, additions)
    check_cast(source, cast)
    meter(0.45, "Casting complete")

    identity = deepcopy(identity_passes[-1]) if identity_passes else None
    if identity is not None and len(identity_passes) > 1:
        identity["cases"] = [deepcopy(case) for item in identity_passes for case in item.get("cases", [])]
        identity["scans"] = [dict(deepcopy(scan), casting_pass=index + 1)
                             for index, item in enumerate(identity_passes) for scan in item.get("scans", [])]
        identity["casting_passes"] = identity_passes
    if identity is not None:
        for decision in identity.get("cases", []):
            if decision.get("status") != "applied":
                continue
            for merge in decision.get("patch", {}).get("merges", []):
                for issue in issues:
                    if issue["target"] == merge["from_character_id"]:
                        issue.setdefault("identity_original_target", issue["target"])
                        issue["target"] = merge["into_character_id"]
    return {"cast": cast, "issues": issues, "scene_summaries": discovered["scene_summaries"],
            "regions": discovered["regions"], "identity": identity,
            "provenance": {"version": VERSION, "strategy": "adaptive_evidence_first",
                           "discovery": discovered["provenance"], "coverage": coverage["provenance"],
                           "candidate_ledger": {"discovery": deepcopy(discovered["cast"]),
                                                "coverage": deepcopy(coverage["cast"])},
                           "superseded_candidate_voices": archived_voices,
                           "cast_fingerprint": fingerprint(cast)}}


def summary_for_units(summaries, regions, units):
    """Return only summaries for discovery regions intersecting these targets."""
    targets = {unit["unit_id"] for unit in units}
    selected = [deepcopy(summary) for summary, region in zip(summaries, regions)
                if targets.intersection(unit["unit_id"] for unit in region)]
    if not selected:
        raise ValueError("Casting summaries do not cover the attribution target units")
    return selected[0] if len(selected) == 1 else {"scope": "source_regions", "summaries": selected}
