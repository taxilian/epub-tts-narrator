"""Bounded adaptive resolution of same-name cast collisions.

The lexical name screen only nominates merge hypotheses; it never merges by
name. Each wave nominates source-ordered, disjoint similar-name pairs between
live identity lineages, three independent blind reviewers judge every nominated
case through capacity-batched ``identity_collision_votes`` requests, and only
unanimously approved disjoint merges are applied against the immutable wave
snapshot. Both participants of a rejected, uncertain, or failed case retire for
this book and never try new partners; unapproved pairs stay unresolved and are
never treated as proven separate. Waves stop at the first wave that produces no
approved merge or has no eligible pair, so at most K-1 cases are nominated for
K live lineages — without quadratic pair enumeration, fixed wave caps, or
fixed request caps.

Negative constraints persist in the returned ``state``: every non-applied
source pair case forbids cross-origin merges, unresolved splits and
single-character uncertainty freeze the affected lineages, and retired
lineages are never renominated. The caller may pass that state back as
``prior`` for later post-coverage audits. This provenance never claims
exhaustive pairwise identity certification.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from difflib import SequenceMatcher
from threading import Event, Lock

from .cast_scope import CAST_SCOPE_GUIDANCE, scoped_cast
from .identity import (IDENTITY_VERSION, _fingerprint, _issue, _names, _unique_evidence, _unanimous)
from .identity_collision_prompts import VOTE_BATCH
from .identity_prompts import LENSES
from .identity_vote_validation import make_batch_vote_accept


COLLISION_VERSION = 1

COLLISION_VOTE_SYSTEM = VOTE_BATCH + CAST_SCOPE_GUIDANCE

_NOMINATION_REASON = ("Names or aliases resemble each other. Resemblance only nominates this merge "
                      "hypothesis; three independent reviewers must verify identity from source evidence.")

_EMPTY_AVAILABLE = {"characters": [], "roles": [], "voices": []}

_PAIR_SOURCE_SCOPE = "pair_evidence_and_chapter_neighbors"

_STATE_FIELDS = {"version", "source_fingerprint", "lineage_members", "forbidden_pairs", "unresolved_pairs", "frozen_ids", "retired_ids"}


class _Lineages:
    """Union-find over candidate IDs with forbidden-pair adjacency between lineages."""

    def __init__(self):
        self.parent = {}
        self.members = {}
        self.adjacent = {}

    def add(self, character_id):
        if character_id not in self.parent:
            self.parent[character_id] = character_id
            self.members[character_id] = {character_id}
            self.adjacent[character_id] = set()
        return self.find(character_id)

    def find(self, character_id):
        root = character_id
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[character_id] != root:
            self.parent[character_id], character_id = root, self.parent[character_id]
        return root

    def union(self, left, right):
        left_root, right_root = self.add(left), self.add(right)
        if left_root == right_root:
            return left_root
        if self.forbids(left_root, right_root):
            raise ValueError("Identity source decision conflicts with a retained negative lineage constraint")
        for member in self.members[right_root]:
            self.parent[member] = left_root
        self.members[left_root] |= self.members[right_root]
        del self.members[right_root]
        neighbors = (self.adjacent[left_root] | self.adjacent[right_root]) - {left_root, right_root}
        self.adjacent[left_root] = neighbors
        del self.adjacent[right_root]
        for neighbor in neighbors:
            self.adjacent[neighbor].difference_update({left_root, right_root})
            self.adjacent[neighbor].add(left_root)
        return left_root

    def forbid(self, left, right):
        left_root, right_root = self.add(left), self.add(right)
        if left_root == right_root:
            raise ValueError("Identity lineage state combines a forbidden original-character pair")
        self.adjacent[left_root].add(right_root)
        self.adjacent[right_root].add(left_root)
        return True

    def forbids(self, left_root, right_root):
        return right_root in self.adjacent.get(left_root, ())


def _gram_keys(name):
    return {name[index:index + 3] for index in range(len(name) - 2)} if len(name) >= 3 else set()


def _resembles(left, right):
    """The exact identity._collision_cases resemblance predicate for one pair."""
    left_names, right_names = _names(left), _names(right)
    return bool(left_names & right_names) or any(min(len(a), len(b)) >= 4 and
                 SequenceMatcher(None, a, b).ratio() >= .88 for a in left_names for b in right_names)


def _similar_pairs(rows):
    """Source-ordered similar-name candidate pairs without quadratic enumeration.

    Exact shared names and shared character 3-grams nominate candidates through
    inverted indexes; the exact ``identity._collision_cases`` resemblance
    predicate then confirms each candidate. Names only nominate, so a candidate
    the indexes miss can delay a hypothesis; it can never merge anything.
    """
    if len(rows) < 2:
        return []
    position = {row["id"]: index for index, row in enumerate(rows)}
    exact, grams = {}, {}
    for row in rows:
        for name in _names(row):
            exact.setdefault(name, []).append(row)
            for gram in _gram_keys(name):
                grams.setdefault(gram, set()).add(row["id"])
    candidates = set()
    for group in exact.values():
        candidates.update((first["id"], second["id"]) for first, second in zip(group, group[1:]))
    for group in grams.values():
        ordered = sorted(group, key=position.__getitem__)
        candidates.update(zip(ordered, ordered[1:]))
    by_id = {row["id"]: row for row in rows}
    confirmed = [(position[left], position[right], left, right) for left, right in candidates
                 if _resembles(by_id[left], by_id[right])]
    confirmed.sort()
    return [(by_id[left], by_id[right]) for _, _, left, right in confirmed]


def _pair_context(source, cast_view):
    """The pair's cited evidence blocks plus their same-chapter neighbors — nothing more."""
    blocks = source["blocks"]
    index_by_id = {block["id"]: index for index, block in enumerate(blocks)}
    indexes = set()
    def citations(value):
        if isinstance(value, dict):
            if isinstance(value.get("block_id"), str) and isinstance(value.get("quote"), str):
                if value["block_id"] in index_by_id:
                    indexes.add(index_by_id[value["block_id"]])
            for child in value.values():
                citations(child)
        elif isinstance(value, list):
            for child in value:
                citations(child)

    citations(cast_view)
    for index in list(indexes):
        chapter = blocks[index]["chapter_id"]
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(blocks) and blocks[neighbor]["chapter_id"] == chapter:
                indexes.add(neighbor)
    return [{"unit_id": "identity_context_" + blocks[index]["id"], "block_id": blocks[index]["id"],
             "chapter_id": blocks[index]["chapter_id"], "start": 0, "text": blocks[index]["text"]}
            for index in sorted(indexes)]


def _prepare_case(source, wave_snapshot, wave_hash, source_hash, case_id, left_row, right_row):
    """Build one merge-hypothesis request entry with pair-scoped views.

    The full proposed cast exists only transiently here: its canonical
    fingerprint is disclosed truthfully, and only the pair-scoped proposed view
    is retained, so memory never scales with a full-cast copy per pair.
    """
    from .pipeline import merge_patch
    merge = {"from_character_id": right_row["id"], "into_character_id": left_row["id"],
             "evidence": _unique_evidence(left_row["evidence"] + right_row["evidence"])}
    patch = {"characters": [], "roles": [], "voices": [], "merges": [merge], "issues": [],
             "summary": "Collision resolution merge hypothesis pending independent unanimous approval."}
    proposed, _, _ = merge_patch(source, wave_snapshot, patch, _EMPTY_AVAILABLE)
    view, scope = scoped_cast(source, wave_snapshot, (), character_ids=[left_row["id"], right_row["id"]], identity_case=True)
    proposed_view, proposed_scope = scoped_cast(source, proposed, (), character_ids=[left_row["id"]], identity_case=True)
    return {"case_id": case_id, "kind": "merge", "character_ids": [left_row["id"], right_row["id"]],
            "reason": _NOMINATION_REASON, "evidence": deepcopy(merge["evidence"]),
            "source_units": _pair_context(source, view), "source_scope": _PAIR_SOURCE_SCOPE,
            "original_cast": view, "cast_scope": scope, "patch": patch,
            "patch_fingerprint": _fingerprint(patch),
            "proposed_cast": proposed_view, "proposed_cast_scope": proposed_scope,
            "proposed_cast_fingerprint": proposed_scope["full_cast_fingerprint"],
            "source_fingerprint": source_hash, "original_cast_fingerprint": wave_hash}


def _lineage_locked(lineages, root, locked_ids):
    return any(member in locked_ids for member in lineages.members[root])


def _parse_prior(prior):
    if prior is None:
        return None
    if not isinstance(prior, dict):
        raise ValueError("Identity collision prior must be a state object or None")
    if isinstance(prior.get("state"), dict):
        prior = prior["state"]
    missing = _STATE_FIELDS - set(prior)
    if missing:
        raise ValueError("Identity collision prior state is missing field(s): " + ", ".join(sorted(missing)))
    if type(prior["version"]) is not int or prior["version"] != COLLISION_VERSION:
        raise ValueError("Identity collision prior state has an unsupported version")
    if (not isinstance(prior["source_fingerprint"], str) or len(prior["source_fingerprint"]) != 64
            or any(value not in "0123456789abcdef" for value in prior["source_fingerprint"])):
        raise ValueError("Identity collision prior source_fingerprint must be a SHA256 fingerprint")
    for field in ("forbidden_pairs", "unresolved_pairs"):
        pairs = prior[field]
        if not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2
                                              or any(not isinstance(item, str) for item in pair) for pair in pairs):
            raise ValueError(f"Identity collision prior state field {field} must be pairs of character IDs")
    for field in ("frozen_ids", "retired_ids"):
        ids = prior[field]
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError(f"Identity collision prior state field {field} must be character IDs")
    members = prior["lineage_members"]
    if not isinstance(members, dict) or any(not isinstance(survivor, str)
                                            or not isinstance(row, list) or any(not isinstance(item, str) for item in row)
                                            for survivor, row in members.items()):
        raise ValueError("Identity collision prior state lineage_members must map character IDs to ID lists")
    return prior


def collision_state_for_source(prior, source_fingerprint):
    state = _parse_prior(prior)
    if state is not None and state["source_fingerprint"] != source_fingerprint:
        raise ValueError("Identity collision prior source_fingerprint does not match the current source")
    return state


def prior_merge_conflict(prior, character_ids, source_fingerprint):
    state = collision_state_for_source(prior, source_fingerprint)
    if state is None:
        return False
    left, right = character_ids
    members = state["lineage_members"]
    left_members, right_members = set(members.get(left, [left])), set(members.get(right, [right]))
    return any((a in left_members and b in right_members) or (b in left_members and a in right_members)
               for a, b in state["forbidden_pairs"])


def _source_constraints(source_cases):
    """Split source audit cases into applied merges, forbidden pairs, and frozen IDs.

    Every non-applied source pair case (confirmed_separate, unresolved,
    invalid proposal, overlapping cases) forbids cross-origin merges; a
    non-applied single-ID case (unresolved split, non-pair uncertain) freezes
    the affected ID. An entry without a status is treated as non-applied.
    """
    applied, negatives, frozen = [], [], []
    for row in source_cases:
        if not isinstance(row, dict):
            raise ValueError("Identity source_cases entries must be case records")
        ids = row.get("character_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids) \
                or len(ids) != len(set(ids)):
            raise ValueError("Identity source_cases entries need distinct character IDs")
        if row.get("status") == "applied":
            if row.get("kind") == "merge" and len(ids) == 2:
                applied.append((ids[0], ids[1]))
            elif row.get("kind") == "split":
                parts = row.get("patch", {}).get("characters", [])
                part_ids = [part.get("id") for part in parts if isinstance(part, dict)]
                if len(part_ids) < 2 or any(not isinstance(value, str) for value in part_ids):
                    raise ValueError("Applied split source_cases need the distinct patch.characters IDs")
                frozen.extend(ids + part_ids)
                negatives.extend((left, right) for index, left in enumerate(part_ids) for right in part_ids[index + 1:])
            continue
        if len(ids) == 2:
            negatives.append(tuple(sorted(ids)))
        else:
            frozen.extend(ids)
    return applied, negatives, frozen


def _wave_votes(requests, source, entries, source_hash, wave_hash, workers, progress, cancelled, counter):
    """Three blind reviewers judge every prepared case in capacity-batched requests.

    Reviewers see identical case regions and scope; only reviewer_id/lens
    differ. Provider capacity subdivides a reviewer's batch through
    ``run_partitioned`` (at most 2N-1 nodes); schema exhaustion or an
    indivisible case leaves the affected cases unresolved with error votes,
    never a false approval. Non-capacity provider failures stay fatal and
    dispatch no further requests.
    """
    from .adaptive_requests import request_size, run_partitioned
    limit = getattr(requests, "max_request_chars", 200000)
    worst_reviewer, worst_lens = max(LENSES, key=lambda row: len(row[0]) + len(row[1]))
    items = [entry["case_id"] for entry in entries.values()]
    failure_lock, failures = Lock(), []

    def review(task):
        reviewer, lens = task
        collected = {}

        def execute(system_prompt, payload):
            accept = make_batch_vote_accept(reviewer=reviewer, lens=lens, payload=payload, source=source,
                                            request_fingerprint=_fingerprint(payload))
            votes = requests.run(system_prompt, payload, accept, cancelled=cancelled)
            counter[0] += 1
            for vote in votes:
                collected[vote["case_id"]] = vote
            return votes

        def build(subset):
            payload = {"mode": "identity_collision_votes", "collision_version": COLLISION_VERSION,
                       "identity_version": IDENTITY_VERSION, "source_fingerprint": source_hash,
                       "original_cast_fingerprint": wave_hash,
                       "cases": [entries[case_id] for case_id in subset],
                       "reviewer_id": reviewer, "lens": lens}
            return COLLISION_VOTE_SYSTEM, payload

        def fits(system_prompt, payload):
            # Size every reviewer's request with the longest lens so all three
            # reviewers receive identical case partitions.
            worst = dict(payload, reviewer_id=worst_reviewer, lens=worst_lens)
            return request_size(system_prompt, worst) <= limit

        try:
            run_partitioned(items, build, execute, fits=fits, progress=progress,
                            label=f"Collision jury ({reviewer})", requests=requests)
        except ValueError as exc:
            # Exhausted schema repairs or an indivisible case cannot vote.
            for case_id in items:
                if case_id not in collected:
                    collected[case_id] = {"case_id": case_id, "reviewer_id": reviewer, "lens": lens,
                                          "verdict": "error", "reason": str(exc)}
        progress(f"Collision jury ({reviewer}): {len(collected)}/{len(items)} case verdict(s) complete")
        return reviewer, collected

    def run(task):
        try:
            if cancelled.is_set():
                raise CancelledError("Collision reviews cancelled after a fatal request failure")
            return review(task)
        except BaseException as exc:
            # Signal from the failing worker before its future is collected; a
            # sibling's cancellation must never hide the actual provider error.
            with failure_lock:
                if not failures:
                    failures.append(exc)
                cancelled.set()
            raise

    tasks = list(LENSES)
    results = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    pending, next_index, finished = set(), 0, 0
    try:
        while next_index < min(workers, len(tasks)) and not cancelled.is_set():
            pending.add(pool.submit(run, tasks[next_index]))
            next_index += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.discard(future)
                reviewer, collected = future.result()
                results[reviewer] = collected
                finished += 1
                progress(f"Collision jury: {finished}/{len(tasks)} reviewer response set(s) complete")
            while next_index < len(tasks) and len(pending) < workers and not cancelled.is_set():
                pending.add(pool.submit(run, tasks[next_index]))
                next_index += 1
    except BaseException:
        cancelled.set()
        for future in pending:
            future.cancel()
        if failures:
            raise failures[0]
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return {case_id: [results[reviewer][case_id] for reviewer, _ in LENSES] for case_id in items}


def resolve_collisions(source, cast, requests, *, source_cases=(), original_cast=None,
                       prior=None, workers=1, progress=lambda message: None):
    """Resolve same-name cast collisions with bounded adaptive merge hypotheses.

    Returns ``(updated_cast, issues, provenance)``. ``provenance["state"]``
    persists negative constraints, lineage memberships, and retired lineages so
    a later post-coverage audit can pass it — or the whole provenance object —
    back as ``prior``. The canonical full cast is kept locally with all records
    and evidence; only approved ``merge_patch`` operations alter identities.
    """
    from .pipeline import check_cast, merge_patch
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Collision workers must be an integer from 1 to 8")
    if not callable(progress):
        raise ValueError("Collision progress must be callable")
    check_cast(source, cast)
    if original_cast is not None and (not isinstance(original_cast, dict)
                                      or not isinstance(original_cast.get("characters"), list)):
        raise ValueError("Identity collision original_cast must be a cast object with a characters array")
    source_hash = _fingerprint(source)  # Computed once per run.
    state_prior = collision_state_for_source(prior, source_hash)
    lineages = _Lineages()
    for row in original_cast["characters"] if original_cast is not None else ():
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Identity collision original_cast characters need string IDs")
        lineages.add(row["id"])
    for row in cast["characters"]:
        lineages.add(row["id"])
    frozen_ids, retired_ids = set(), set()
    forbidden_pairs, unresolved_pairs = set(), set()
    if state_prior is not None:
        for survivor, members in state_prior["lineage_members"].items():
            for member in members:
                lineages.union(survivor, member)
        for field, target in (("forbidden_pairs", forbidden_pairs), ("unresolved_pairs", unresolved_pairs)):
            for pair in state_prior[field]:
                left, right = sorted(pair)
                target.add((left, right))
                lineages.forbid(left, right)
        frozen_ids.update(state_prior["frozen_ids"])
        retired_ids.update(state_prior["retired_ids"])
    applied_merges, negative_pairs, source_frozen = _source_constraints(source_cases)
    for origin, target in applied_merges:
        lineages.union(origin, target)
    for pair in negative_pairs:
        if lineages.forbid(*pair):
            forbidden_pairs.add(pair)
    frozen_ids.update(source_frozen)

    progress_lock = Lock()

    def safe_progress(message):
        with progress_lock:
            progress(message)

    updated = deepcopy(cast)
    issues, case_records, waves = [], [], []
    counter = [0]
    wave_index = 0
    while True:
        wave_index += 1
        representatives = {}
        for row in updated["characters"]:
            representatives[lineages.find(row["id"])] = row
        locked_ids = frozen_ids | retired_ids
        live_roots = [root for root in representatives if not _lineage_locked(lineages, root, locked_ids)]
        candidates = _similar_pairs([representatives[root] for root in live_roots])
        selected, used = [], set()
        for left_row, right_row in candidates:
            left_root, right_root = lineages.find(left_row["id"]), lineages.find(right_row["id"])
            if left_root in used or right_root in used or lineages.forbids(left_root, right_root):
                continue
            selected.append((left_root, right_root, left_row, right_row))
            used.update((left_root, right_root))
        wave = {"index": wave_index, "live_lineages": len(live_roots), "case_ids": [],
                "applied_case_ids": [], "request_count": 0}
        waves.append(wave)
        if not selected:
            safe_progress(f"Collision wave {wave_index}: no eligible merge hypothesis over "
                          f"{len(live_roots)} live lineage(s); stopping")
            break
        safe_progress(f"Collision wave {wave_index}: {len(selected)} merge hypothesis case(s) over "
                      f"{len(live_roots)} live lineages")
        wave_snapshot = deepcopy(updated)
        wave_hash = _fingerprint(wave_snapshot)
        entries, pairs, nominated = {}, {}, []
        for left_root, right_root, left_row, right_row in selected:
            case_id = "identity_collision_" + _fingerprint(sorted((left_row["id"], right_row["id"])))[:20]
            record = {"id": case_id, "kind": "merge", "character_ids": [left_row["id"], right_row["id"]],
                      "reason": _NOMINATION_REASON, "origin": "identity_collision_screen",
                      "wave": wave_index, "votes": [], "status": None}
            wave["case_ids"].append(case_id)
            nominated.append(record)
            case_records.append(record)
            pairs[case_id] = (left_root, right_root)
            try:
                entry = _prepare_case(source, wave_snapshot, wave_hash, source_hash, case_id, left_row, right_row)
            except ValueError as exc:
                record.update(status="invalid_proposal", error=str(exc))
                retired_ids.update(lineages.members[left_root] | lineages.members[right_root])
                pair = tuple(sorted((left_row["id"], right_row["id"])))
                forbidden_pairs.add(pair)
                unresolved_pairs.add(pair)
                lineages.forbid(left_root, right_root)
                issues.append(_issue(record, "Merge hypothesis failed deterministic validation: " + str(exc)))
                continue
            entries[case_id] = entry
            record.update(patch_fingerprint=entry["patch_fingerprint"],
                           proposed_cast_fingerprint=entry["proposed_cast_fingerprint"],
                           merge={key: entry["patch"]["merges"][0][key]
                                  for key in ("from_character_id", "into_character_id")})
        votes_by_case = {}
        if entries:
            requests_before = counter[0]
            votes_by_case = _wave_votes(requests, source, entries, source_hash, wave_hash,
                                        workers, safe_progress, Event(), counter)
            wave["request_count"] = counter[0] - requests_before
        applied = []
        for record in nominated:
            if record["status"] == "invalid_proposal":
                continue
            votes = votes_by_case.get(record["id"], [])
            record["votes"] = votes
            if _unanimous(votes):
                record["status"] = "applied"
                applied.append(record)
                wave["applied_case_ids"].append(record["id"])
                continue
            record["status"] = "unresolved"
            left_root, right_root = pairs[record["id"]]
            retired_ids.update(lineages.members[left_root] | lineages.members[right_root])
            pair = tuple(sorted(record["character_ids"]))
            forbidden_pairs.add(pair)
            unresolved_pairs.add(pair)
            lineages.forbid(left_root, right_root)
            issues.append(_issue(record, "Three reviewers did not unanimously approve: "
                                 + ", ".join(vote["verdict"] for vote in votes)))
        # Apply only unanimous disjoint merges, against the immutable wave snapshot.
        wave_result = deepcopy(wave_snapshot)
        for record in applied:
            entry = entries[record["id"]]
            wave_result, new_issues, _ = merge_patch(source, wave_result, entry["patch"], _EMPTY_AVAILABLE)
            issues.extend(issue for issue in new_issues if issue["kind"] != "identity_merge")
            lineages.union(*pairs[record["id"]])
        updated = wave_result
        if not applied:
            break
    check_cast(source, updated)
    lineage_members = {}
    for row in updated["characters"]:
        root = lineages.find(row["id"])
        if len(lineages.members[root]) > 1:
            lineage_members[row["id"]] = sorted(lineages.members[root])
    provenance = {"version": COLLISION_VERSION, "policy": "adaptive_collision_unanimous_3",
                  "source_fingerprint": source_hash, "original_cast_fingerprint": _fingerprint(cast),
                  "final_cast_fingerprint": _fingerprint(updated), "request_count": counter[0],
                  "waves": waves, "cases": case_records,
                  "disclosure": {"pair_certification": "non_exhaustive",
                                 "note": "Name resemblance only nominated these merge hypotheses. Unapproved "
                                         "pairs remain unresolved and retired lineages are not proven separate; "
                                         "pairs never nominated were never judged. This provenance does not claim "
                                         "exhaustive pairwise identity certification."},
                   "state": {"version": COLLISION_VERSION, "source_fingerprint": source_hash,
                             "lineage_members": lineage_members,
                            "forbidden_pairs": sorted(list(pair) for pair in forbidden_pairs),
                            "unresolved_pairs": sorted(list(pair) for pair in unresolved_pairs),
                            "frozen_ids": sorted(frozen_ids), "retired_ids": sorted(retired_ids)}}
    safe_progress(f"Collision resolution applied {len([record for record in case_records if record['status'] == 'applied'])} "
                  f"merge(s) across {len(waves)} wave(s); {len(retired_ids)} retired and {len(frozen_ids)} frozen lineage ID(s)")
    return updated, issues, provenance
