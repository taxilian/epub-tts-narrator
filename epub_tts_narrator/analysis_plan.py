"""Read-only work bounds for automatic, source-sized book analysis."""

from .attribution_review import count_review_batches
from .casting_discovery import discovery_regions
from .pipeline import make_units
from .series import series_context


def plan_analysis(source, settings, policy, registry=None):
    batches = make_units(source, settings["chunk_chars"])
    atoms = [unit for batch in batches for unit in batch]
    context = series_context(registry, source) if registry is not None else None
    consensus = settings["identity_consensus"]
    series_scans = int(bool(consensus and context and context["prior_books"]))
    review = settings["attribution_review"]
    review_scans = count_review_batches(source, settings["attribution_review_chunk_chars"]) if review else 0
    identity_scans = int(consensus)
    discovery = discovery_regions(atoms)
    baseline = (len(batches) + len(discovery) + 2 + identity_scans
                + series_scans + review_scans)
    planned = {
        "book": source["book"], "chapters": len(source["chapters"]), "blocks": len(source["blocks"]),
        "text_characters": sum(len(block["text"]) for block in source["blocks"]),
        "source_units": len(atoms), "analysis_batches": len(batches),
        "casting_strategy": "adaptive_evidence_first", "work_limits": "derived_from_source",
        "discovery_regions_minimum": len(discovery),
        "discovery_requests_minimum": len(discovery), "coverage_scan_requests_minimum": 1,
        "adaptive_source_tree_node_bound": max(0, 2 * len(atoms) - 1),
        "identity_consensus": consensus, "identity_scan_requests_minimum": identity_scans,
        "identity_verdicts_per_case": 3 if consensus else 0,
        "series_identity_scan_requests_minimum": series_scans,
        "series_continuity_rounds_maximum": 2 if series_scans else 0,
        "attribution_review": review, "attribution_review_scan_requests": review_scans,
        "attribution_review_verdicts_per_proposal": 3 if review else 0,
        "voice_ensemble_requests_estimate": 1, "baseline_model_requests": baseline,
        "baseline_assumptions": {"nonempty_cast": True, "ensemble_design_needed": True},
        "request_count_is_estimate": True,
        "note": (
            "The baseline is a planning estimate, not a guaranteed lower or upper bound, before cache reuse, repairs and retries. "
            "Discovery, identity scans and source-backed coverage checks use automatic source-sized partitions, "
            "not a configured number of reconciliation rounds. Each independent 60,000-character discovery "
            "region has its own at most 2N-1 adaptive tree for N original source units; "
            "local preflight splits make no model call. Provider capacity and output truncation can add splits. "
            "Coverage repairs follow distinct source-anchored findings and stop on repeated findings or no progress. "
            "Identity cases need three blind verdicts and split cases may add one preparation request. "
            "Candidate collision reviews use bounded matching waves rather than all name pairs, and may batch disjoint cases. "
            "Series continuity, voice-target grouping, speaker completion and auditions add data-dependent work. "
            "The baseline assumes voices need design; that stage can instead be skipped. "
            "Characters are not tokens. No book-level request, partition or repair quota needs configuring."
        ),
        "exclusions": source["exclusions"],
    }
    if context is not None:
        planned["series"] = {"series_id": registry["series_id"], "books": len(context["prior_books"]),
                             "characters": len(context["characters"])}
    return planned
