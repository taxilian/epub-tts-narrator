"""Shared identity vote validation factories.

``make_vote_accept`` factors the exact per-vote acceptance closure used by
``identity._votes`` — verdict vocabulary, bounded source-bound reason, exact
visible evidence citations, and approval context beyond names or aliases —
into one reusable factory. The adaptive collision batch mode validates every
batched vote entry through this identical contract; the legacy single-case
closure is routed through the same factory separately. Do not weaken one
path without the other, and never substitute a weaker judge validator.
"""

from __future__ import annotations

from copy import deepcopy

from .identity import _evidence, _has_context


def make_vote_accept(*, reviewer, lens, source, source_units, cast_view, identity_case,
                     proposed_cast, request_fingerprint, fingerprints,
                     prior_series=None, series_registry_fingerprint=None, review_round=1):
    """Build the exact identity vote acceptance closure for one reviewer.

    ``fingerprints`` carries the vote's audit fields (``source_fingerprint``,
    ``original_cast_fingerprint``, ``proposed_cast_fingerprint``,
    ``patch_fingerprint``); ``request_fingerprint`` identifies the exact request
    payload this response answers. Validation behavior and the produced vote
    shape match ``identity._votes`` exactly.
    """
    def accept(response):
        if not isinstance(response, dict) or response.get("verdict") not in {"approve", "reject", "uncertain"}:
            raise ValueError("Identity reviewer must return approve, reject, or uncertain")
        if not isinstance(response.get("reason"), str) or not response["reason"].strip() or len(response["reason"]) > 2400:
            raise ValueError("Identity review needs a concise source-bound reason")
        _evidence(response.get("evidence"), source, source_units)
        if response["verdict"] == "approve" and not _has_context(response["evidence"], cast_view, identity_case,
                prior_series, proposed_cast):
            raise ValueError("Identity approval needs narrative evidence beyond names or aliases alone")
        vote = {"reviewer_id": reviewer, "lens": lens, "verdict": response["verdict"],
                "reason": response["reason"], "evidence": deepcopy(response["evidence"]),
                "request_fingerprint": request_fingerprint,
                "source_fingerprint": fingerprints["source_fingerprint"],
                "original_cast_fingerprint": fingerprints["original_cast_fingerprint"],
                "proposed_cast_fingerprint": fingerprints["proposed_cast_fingerprint"],
                "patch_fingerprint": fingerprints["patch_fingerprint"]}
        if series_registry_fingerprint is not None:
            vote["series_registry_fingerprint"] = series_registry_fingerprint
        if review_round > 1:
            vote["series_review_round"] = review_round
        return vote
    return accept


def make_batch_vote_accept(*, reviewer, lens, payload, source, request_fingerprint):
    """Build the acceptance closure for one identity_collision_votes batch request.

    The response must return exactly one vote for every supplied case_id:
    omissions, duplicate case IDs, unknown case IDs, and malformed vote entries
    reject the entire batch response. Each individual vote is then validated
    with the exact identity vote contract against that case's own visible
    source units and pair views, so an invalid vote never approves anything.
    """
    entries = {entry["case_id"]: entry for entry in payload["cases"]}

    def accept(response):
        if not isinstance(response, dict) or set(response) != {"votes"} or not isinstance(response["votes"], list):
            raise ValueError("Collision vote batch must return exactly a votes array")
        votes = response["votes"]
        if len(votes) != len(entries):
            raise ValueError(f"Collision vote batch must return exactly {len(entries)} vote(s), "
                             f"one per supplied case; received {len(votes)}")
        accepted, seen = [], set()
        for vote in votes:
            if not isinstance(vote, dict) or set(vote) != {"case_id", "verdict", "reason", "evidence"}:
                raise ValueError("Each batch vote needs exactly case_id, verdict, reason, and evidence")
            case_id = vote["case_id"]
            if not isinstance(case_id, str) or case_id not in entries:
                raise ValueError("Batch vote references an unknown case_id; return one vote per supplied case_id")
            if case_id in seen:
                raise ValueError(f"Batch vote duplicates case_id {case_id}")
            seen.add(case_id)
            entry = entries[case_id]
            single = make_vote_accept(
                reviewer=reviewer, lens=lens, source=source,
                source_units=entry["source_units"], cast_view=entry["original_cast"],
                identity_case={"kind": entry["kind"], "character_ids": entry["character_ids"]},
                proposed_cast=entry["proposed_cast"], request_fingerprint=request_fingerprint,
                fingerprints={key: entry[key] for key in ("source_fingerprint", "original_cast_fingerprint",
                                                           "proposed_cast_fingerprint", "patch_fingerprint")})
            accepted.append(dict(single({"verdict": vote["verdict"], "reason": vote["reason"],
                                         "evidence": vote["evidence"]}), case_id=case_id))
        return accepted
    return accept
