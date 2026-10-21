"""Mandatory source-bound jury for every unassigned spoken passage.

Unlike the optional semantic scanner, deterministic coverage selects the cases.
Three blind assignments must agree on each source span before it is replaced.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from threading import Lock

from .attribution_review import _bounded_map, _evidence, _object, _reason, _unit, _visible
from .contracts import has_assigned_speakers, readiness, validate_document
from .performance import apply_bracket_delivery
from .pipeline import DEFAULT_POLICY, align_units, complete_empty_delivery, fingerprint, write_json
from .prompts import COMMON, PERFORMANCE_POLICY_GUIDANCE
from .provider import ProviderError

COMPLETION_VERSION = "2"
CONTEXT_CHARS = (12000, 100000, 160000)
OUTPUT_TRUNCATED_MESSAGE = "The model output was truncated. Increase the output token limit or reduce the input."
REVIEWERS = (
    ("speaker_completion_1", "Trace explicit speech tags and the speaker of continuing quotations."),
    ("speaker_completion_2", "Distinguish the actual speaker from mentioned people, listeners, and narrative actions."),
    ("speaker_completion_3", "Check alternative identities, narrative boundaries, and solo versus genuinely simultaneous delivery."),
)
ATTRIBUTION_ISSUE_KINDS = frozenset({
    "attribution", "speaker_attribution", "unassigned_speaker", "attribution_consensus", "speaker_completion",
})

SYSTEM = COMMON + PERFORMANCE_POLICY_GUIDANCE + """
Mandatory independent speaker completion, version 2.
Every target block contains text without a selected performer. Independently
assign the COMPLETE target block using the source and existing final_cast. No peer
answers are supplied. Existing assignments, character descriptions, and candidates
are fallible context, subject to the protected assignments described below.
An unassigned passage is not necessarily dialogue or thought: reconsider its kind
from the source before selecting a performer. Ordinary first-person narration,
reported actions, and narrative responses belong to the narrative frame even
when near dialogue. Examine exact wording, surrounding actions, quotation marks,
and available inline_spans (such as emphasis) together; typography is evidence,
not a conclusive rule on its own. Trace who actually communicates words rather
than choosing the nearest name or alternating speakers. A listener, addressee,
or person mentioned is not thereby the speaker.
An existing unassigned segment may incorrectly combine several speakers and
narration. Split it wherever the source changes performer or kind; old unassigned
boundaries are not constraints. Separate speech tags/actions from communicated
words and distinguish successive turns from genuinely simultaneous speech.
Use only existing role IDs. Never invent a character, role, voice, or source fact.
Preserve all original local delivery directions, including empty directions.
Preserve every original segment whose basis is user, including its performer,
kind, text boundaries, and delivery. You cannot override a user's assignment.
When retry_unresolved is true, ALL original segments with assigned performers
are protected, not only user assignments. Preserve their performer, kind, source
boundaries, and delivery; reconsider only the remaining unassigned source spans.
Only select multiple performers when exact source evidence establishes that these
identified characters say the SAME words together. Represent that as
simultaneous_role_ids with at least two distinct existing dialogue roles and
role_id=null, basis=explicit or context, candidates=[], review_required=false.
Alternative possible speakers are candidates, NEVER simultaneous performers.
A plural attribution covering several separate quotations does not establish
that each quotation was spoken in unison. Adjacent calls may be sequential with
unknown order; leave their performers unresolved unless evidence establishes
which person speaks which call, or actually establishes simultaneous same words.
Do not invent a representative narrator or group performer for an unknown crowd.
If the cast lacks the actual speaker, or the visible source is still insufficient,
leave role_id=null, basis=unresolved, review_required=true with plausible known
roles in candidates. Do not guess merely to complete the audiobook.
Return exactly {"block_id":"target block ID", "reason":"brief source-grounded
explanation", "evidence":[{"block_id":"visible ID","quote":"exact source quote"}],
"segments":[{"text":"exact contiguous source substring", "kind":"narration|dialogue|thought|letter|heading|other",
"role_id":"existing role ID or null", "basis":"explicit|context|unresolved",
"reason":"brief attribution explanation", "evidence":[], "candidates":[],
"review_required":false, "delivery":{"direction":"","evidence":[]}}]}.
The optional simultaneous_role_ids field is allowed ONLY on a simultaneous piece.
Return exactly one complete block, not just its unassigned span. Segment texts
must concatenate VERBATIM to the entire target text, including whitespace,
punctuation, brackets, and spelling. Each resolved spoken segment needs exact
source evidence for its performer; the top-level evidence must include a quote
from the target block. Evidence may use only visible target/context source units.
Nonempty delivery directions require exact supporting evidence. Return no other
properties, no user basis, no review of other blocks, and no private reasoning.
"""

FINAL_CONTEXT_GUIDANCE = """
Final independent source-context review. Additional context may include earlier
or later chapters containing exact source evidence for several candidate
identities. These are retrieval leads, not an answer or a change to the target.
Evaluate every plausible identity against source evidence, including identities
of internal, telepathic, or otherwise indirectly named speakers. Trace plural
speech tags across interrupted quotations and subsequent actions. The same
first-person words can be spoken in unison. A plural speech attribution can
continue across a narrative interruption unless source evidence establishes an
actual speaker change; singular grammar alone does not establish that change.
Do not infer unison from uncertainty, and leave genuine ambiguity unresolved.
"""


def _spoken(item):
    return any(char.isalnum() for char in item.get("spoken_text", item["text"]))


def unassigned_segments(document):
    """Return all unassigned spans containing speakable letters or numbers."""
    return [item for item in document["segments"] if _spoken(item) and not has_assigned_speakers(item)]


def _finished_case(case):
    """Recognize an exhausted jury, never an interrupted or incomplete one."""
    status = case.get("status")
    rounds = case.get("rounds")
    if status not in ("partial", "unresolved", "needs_user") or not isinstance(rounds, list):
        return False
    if not 1 <= len(rounds) <= len(CONTEXT_CHARS):
        return False
    if status != "needs_user" and len(rounds) != len(CONTEXT_CHARS):
        return False
    reviewer_ids = {row[0] for row in REVIEWERS}
    for number, turn in enumerate(rounds, 1):
        if not isinstance(turn, dict) or type(turn.get("round")) is not int or turn["round"] != number:
            return False
        votes = turn.get("votes")
        if (not isinstance(votes, list) or len(votes) != len(REVIEWERS)
                or any(not isinstance(vote, dict) or not isinstance(vote.get("reviewer_id"), str)
                       or vote.get("status") not in ("valid", "error") for vote in votes)
                or {vote["reviewer_id"] for vote in votes} != reviewer_ids):
            return False
    return status != "needs_user" or any(
        vote.get("status") == "error" and vote.get("output_truncated") is True
        for vote in rounds[-1]["votes"])


def _manual_script_ancestors(document, source_hash, cast_hash):
    """Trace only recorded, source-bound manual decisions back from this script."""
    provenance = document["provenance"]
    history = provenance.get("speaker_decision_history", [])
    audits = [provenance.get("speaker_decisions"), *(history if isinstance(history, list) else [])]
    edges = []
    for audit in audits:
        if (isinstance(audit, dict) and type(audit.get("version")) is int and audit["version"] == 1
                and audit.get("source_fingerprint") == source_hash
                and audit.get("cast_fingerprint") == cast_hash
                and isinstance(audit.get("decisions"), list) and audit["decisions"]
                and isinstance(audit.get("input_script_fingerprint"), str)
                and isinstance(audit.get("output_script_fingerprint"), str)):
            edges.append((audit["input_script_fingerprint"], audit["output_script_fingerprint"]))
    ancestors = {fingerprint(document["segments"])}
    while True:
        parents = {before for before, after in edges if after in ancestors}
        if parents <= ancestors:
            return ancestors
        ancestors.update(parents)


def _only_manual_assignments(before, after):
    """Allow assigned user spans without overlooking edits to unresolved text."""
    if not isinstance(before, list) or len(before) != len(after):
        return False
    changed = {"role_id", "voice_id", "kind", "attribution", "simultaneous_speakers"}
    for old, current in zip(before, after):
        if old == current:
            continue
        if (not isinstance(old, dict) or has_assigned_speakers(old)
                or current.get("attribution", {}).get("basis") != "user"
                or not has_assigned_speakers(current)
                or {key: value for key, value in old.items() if key not in changed}
                != {key: value for key, value in current.items() if key not in changed}):
            return False
    return True


def _completed_unresolved_cases(document):
    """Find each block's latest matching completed case, including v1 audits.

    Reuse requires the same source, cast, and complete block records, allowing
    recorded user assignments to replace formerly unassigned spans. Decisions
    elsewhere therefore do not change the matching case.
    Saved audits never supply new speaker assignments here. The case is only
    an exhaustion marker or a stable namespace for an explicit fresh attempt.
    """
    missing = {item["block_id"] for item in unassigned_segments(document)}
    if not missing:
        return {}
    current = {block_id: [] for block_id in missing}
    for item in document["segments"]:
        if item["block_id"] in current:
            current[item["block_id"]].append(item)
    provenance = document["provenance"]
    history = provenance.get("speaker_completion_history", [])
    audits = [provenance.get("speaker_completion"), *(reversed(history) if isinstance(history, list) else [])]
    source_hash, cast_hash = fingerprint(document["source"]), fingerprint(document["cast"])
    manual_ancestors = _manual_script_ancestors(document, source_hash, cast_hash)
    completed = {}
    for audit in audits:
        if (not isinstance(audit, dict) or audit.get("version") not in ("1", COMPLETION_VERSION)
                or audit.get("status") != "blocked"
                or audit.get("source_fingerprint") != source_hash
                or audit.get("cast_fingerprint") != cast_hash
                or audit.get("context_chars") != list(CONTEXT_CHARS)
                or not isinstance(audit.get("cases"), list)):
            continue
        for case in audit["cases"]:
            if not isinstance(case, dict) or not isinstance(case.get("block_id"), str):
                continue
            block_id = case["block_id"]
            if block_id not in missing or block_id in completed or not _finished_case(case):
                continue
            saved = case.get("applied_segments", case.get("original_segments"))
            audited_manual_change = (isinstance(audit.get("output_script_fingerprint"), str)
                                     and audit["output_script_fingerprint"] in manual_ancestors
                                     and _only_manual_assignments(saved, current[block_id]))
            if saved == current[block_id] or audited_manual_change:
                completed[block_id] = {"version": audit["version"], "case": case}
    return completed


def automatic_completion_targets(document, *, retry_exhausted=False):
    """Select unfinished blocks; explicit retries include previously exhausted juries."""
    if type(retry_exhausted) is not bool:
        raise ValueError("retry_exhausted must be a boolean")
    missing = {item["block_id"] for item in unassigned_segments(document)}
    exhausted = set() if retry_exhausted else set(_completed_unresolved_cases(document))
    return [block["id"] for block in document["source"]["blocks"]
            if block["id"] in missing - exhausted]


def _performers(item):
    if item.get("simultaneous_speakers"):
        return tuple(sorted((row["role_id"], row["voice_id"]) for row in item["simultaneous_speakers"]))
    return ((item["role_id"], item["voice_id"]),)


def _decision(item):
    return (_performers(item), item["kind"], item["delivery"]["direction"],
            item["attribution"]["review_required"], item["attribution"]["basis"] == "unresolved")


def _signature(items):
    """Compare performer/kind/delivery per source span, ignoring cosmetic splits."""
    runs = []
    for item in items:
        key = _decision(item)
        if runs and runs[-1][0] == key and runs[-1][2] == item["start"]:
            runs[-1] = (key, runs[-1][1], item["end"])
        else:
            runs.append((key, item["start"], item["end"]))
    return runs


def _compact(items):
    result = []
    for item in items:
        row = {key: deepcopy(item[key]) for key in ("id", "block_id", "start", "end", "text", "kind", "role_id", "delivery")}
        row.update(deepcopy(item["attribution"]))
        if item.get("simultaneous_speakers"):
            row["simultaneous_role_ids"] = [performer["role_id"] for performer in item["simultaneous_speakers"]]
        result.append(row)
    return result


def _context(source, index, budget):
    """Nearest chapter neighbors first; full target block never gets truncated."""
    blocks, selected, remaining = source["blocks"], {}, budget
    chapter = blocks[index]["chapter_id"]
    distance = 1
    while remaining:
        eligible = False
        for position in (index - distance, index + distance):
            if not 0 <= position < len(blocks) or blocks[position]["chapter_id"] != chapter:
                continue
            eligible = True
            block = blocks[position]
            length = min(remaining, len(block["text"]))
            start = len(block["text"]) - length if position < index else 0
            if length:
                selected[position] = _unit(block, start, start + length)
                remaining -= length
            if not remaining:
                break
        if not eligible:
            break
        distance += 1
    return [selected[position] for position in sorted(selected)]


def _final_context(document, index, budget, originals, rounds):
    """Balanced identity-evidence retrieval; never expose votes or their reasons."""
    source, cast = document["source"], document["cast"]
    blocks = source["blocks"]
    positions = {block["id"]: offset for offset, block in enumerate(blocks)}
    role_ids = set()
    for item in originals:
        role_ids.update(item["attribution"]["candidates"])
        role_ids.update(role for role, _ in _performers(item) if role is not None)
    for turn in rounds:
        for vote in turn["votes"]:
            if vote.get("status") == "valid":
                for item in vote["proposed_segments"]:
                    role_ids.update(role for role, _ in _performers(item) if role is not None)
                    role_ids.update(item["attribution"]["candidates"])
    characters = {row["id"]: row for row in cast["characters"]}
    roles = {row["id"]: row for row in cast["roles"]}
    evidence_lists = [characters.get(roles.get(role, {}).get("character_id"), {}).get("evidence", [])
                      for role in sorted(role_ids)]
    # Round-robin candidate evidence avoids exhausting the budget on one identity.
    anchors = []
    for offset in range(max((len(rows) for rows in evidence_lists), default=0)):
        for rows in evidence_lists:
            if offset < len(rows) and rows[offset]["block_id"] in positions:
                anchor = positions[rows[offset]["block_id"]]
                if anchor not in anchors and anchor != index:
                    anchors.append(anchor)
    selected = {positions[item["block_id"]]: item for item in _context(source, index, budget // 2)}
    remaining = budget - sum(len(item["text"]) for item in selected.values())
    def include(position):
        nonlocal remaining
        if position == index or position in selected or not remaining:
            return
        block = blocks[position]
        length = min(remaining, len(block["text"]))
        if length:
            selected[position] = _unit(block, 0, length)
            remaining -= length
    for anchor in anchors:
        include(anchor)
    for distance in range(1, 5):
        for anchor in anchors:
            for position in (anchor - distance, anchor + distance):
                if 0 <= position < len(blocks) and blocks[position]["chapter_id"] == blocks[anchor]["chapter_id"]:
                    include(position)
    # Spend any unused allowance on additional nearby chapter context.
    for item in _context(source, index, budget):
        include(positions[item["block_id"]])
    return [selected[position] for position in sorted(selected)]


def _preserved_protected(proposed, originals, *, preserve_assigned=False):
    for old in originals:
        overlaps = [item for item in proposed if item["start"] < old["end"] and item["end"] > old["start"]]
        for item in overlaps:
            protected = old["attribution"]["basis"] == "user" or (preserve_assigned and has_assigned_speakers(old))
            if protected and (
                    _performers(item) != _performers(old) or item["kind"] != old["kind"]):
                raise ValueError("Speaker completion cannot change an explicit user assignment" if
                                 old["attribution"]["basis"] == "user" else
                                 "Speaker completion retry cannot change an already assigned source span")


def _slice(item, start, end):
    result = deepcopy(item)
    if (start, end) != (item["start"], item["end"]):
        result.update(start=start, end=end, id=f"{item['block_id']}:{start}:{end}",
                      text=item["text"][start - item["start"]:end - item["start"]])
        # Bracket metadata is derived again after all source spans are assembled.
        result.pop("spoken_text", None)
        result.pop("performance", None)
    return result


def _original_delivery(proposed, originals):
    """Delivery belongs to the saved source span, not this speaker-only jury."""
    output = []
    for piece in proposed:
        for old in originals:
            start, end = max(piece["start"], old["start"]), min(piece["end"], old["end"])
            if start < end:
                item = _slice(piece, start, end)
                item["delivery"] = deepcopy(old["delivery"])
                output.append(item)
    return output


def _accept(response, document, batch, originals, *, preserve_assigned=False):
    _object(response, ("block_id", "reason", "evidence", "segments"), "Speaker completion")
    unit = batch["target_units"][0]
    if response["block_id"] != unit["block_id"]:
        raise ValueError("Speaker completion must return its one target block, never a context block")
    _reason(response["reason"], "Speaker completion")
    visible = _visible(batch)
    _evidence(response["evidence"], visible, required=True, target=unit["block_id"])
    pieces = response["segments"]
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("Speaker completion needs a complete nonempty segments array")
    keys = {"text", "kind", "role_id", "basis", "reason", "evidence", "candidates", "review_required", "delivery"}
    for index, piece in enumerate(pieces):
        path = f"segments[{index}]"
        expected = keys | ({"simultaneous_role_ids"} if isinstance(piece, dict) and "simultaneous_role_ids" in piece else set())
        _object(piece, sorted(expected), path)
        _evidence(piece["evidence"], visible, required=piece["basis"] != "unresolved" and
                  isinstance(piece["text"], str) and any(char.isalnum() for char in piece["text"]), label=path + ".evidence")
        delivery = complete_empty_delivery(piece["delivery"])
        _object(delivery, ("direction", "evidence"), path + ".delivery")
        _evidence(delivery["evidence"], visible, label=path + ".delivery.evidence")
    # align_units safely restores boundary whitespace for the general analyzer;
    # this jury contract is stricter, so first require the actual exact full text.
    if any(not isinstance(piece["text"], str) for piece in pieces) or "".join(piece["text"] for piece in pieces) != unit["text"]:
        raise ValueError("Speaker completion segment text must concatenate to the exact complete source block")
    proposed = align_units(document["source"], document["cast"], [unit],
                           {"units": [{"unit_id": unit["unit_id"], "segments": pieces}]})
    _preserved_protected(proposed, originals, preserve_assigned=preserve_assigned)
    proposed = _original_delivery(proposed, originals)
    resolved = all(not _spoken(item) or (has_assigned_speakers(item) and
                   item["attribution"]["basis"] != "unresolved" and not item["attribution"]["review_required"])
                   for item in proposed)
    return {"status": "valid", "resolved": resolved, "raw_response": deepcopy(response),
            "proposed_segments": proposed, "assignment_fingerprint": fingerprint(_signature(proposed))}


def _independent_valid_votes(votes):
    return (len(votes) == 3 and {vote["reviewer_id"] for vote in votes} == {row[0] for row in REVIEWERS}
            and len({vote["request_fingerprint"] for vote in votes}) == 3
            and all(vote.get("status") == "valid" for vote in votes))


def _unanimous(votes):
    return (_independent_valid_votes(votes)
            and all(vote.get("resolved") is True for vote in votes)
            and len({vote["assignment_fingerprint"] for vote in votes}) == 1)


def _agreed_spans(votes, originals, *, preserve_assigned=False):
    """Unanimity is per interval, with every reviewer's boundaries respected."""
    if not _independent_valid_votes(votes):
        return []
    proposals = [vote["proposed_segments"] for vote in votes]
    boundaries = sorted({item[key] for rows in [originals, *proposals]
                         for item in rows for key in ("start", "end")})
    agreed = []
    for start, end in zip(boundaries, boundaries[1:]):
        if any((old["attribution"]["basis"] == "user" or (preserve_assigned and has_assigned_speakers(old)))
               and old["start"] <= start < old["end"]
               for old in originals):
            continue
        choices = [[item for item in rows if item["start"] <= start and item["end"] >= end]
                   for rows in proposals]
        if any(len(items) != 1 for items in choices):
            continue
        items = [rows[0] for rows in choices]
        if (all(has_assigned_speakers(item) and item["attribution"]["basis"] != "unresolved"
                and not item["attribution"]["review_required"] for item in items)
                and len({_decision(item) for item in items}) == 1):
            if agreed and agreed[-1][0] is items[0] and agreed[-1][2] == start:
                agreed[-1] = (items[0], agreed[-1][1], end)
            else:
                agreed.append((items[0], start, end))
    output = []
    original_boundaries = {item[key] for item in originals for key in ("start", "end")}
    for item, start, end in agreed:
        piece = _slice(item, start, end)
        if (output and output[-1]["end"] == start and start not in original_boundaries
                and _decision(output[-1]) == _decision(piece)
                and output[-1]["delivery"] == piece["delivery"]):
            previous = output[-1]
            previous.update(end=end, id=f"{piece['block_id']}:{previous['start']}:{end}",
                            text=previous["text"] + piece["text"])
            for evidence in piece["attribution"]["evidence"]:
                if evidence not in previous["attribution"]["evidence"]:
                    previous["attribution"]["evidence"].append(deepcopy(evidence))
            if piece["attribution"]["basis"] == "context":
                previous["attribution"]["basis"] = "context"
        else:
            output.append(piece)
    return output


def _apply(proposed, originals, retained_reviews, *, preserve_assigned=False):
    """Keep user spans and existing delivery evidence, and unrelated review flags."""
    output = []
    for piece in proposed:
        for old in originals:
            start, end = max(piece["start"], old["start"]), min(piece["end"], old["end"])
            if start >= end:
                continue
            if old["attribution"]["basis"] == "user" or (preserve_assigned and has_assigned_speakers(old)):
                continue
            item = _slice(piece, start, end)
            item["delivery"] = deepcopy(old["delivery"])
            if has_assigned_speakers(old) and old["attribution"]["review_required"]:
                if item.get("simultaneous_speakers"):
                    issue = {"kind": "preserved_segment_review", "target": old["block_id"],
                             "reason": old["attribution"]["reason"], "resolved": False,
                             "source_segment_id": old["id"], "source_start": old["start"], "source_end": old["end"]}
                    issue["id"] = "speaker-completion-preserved-review:" + fingerprint(issue)
                    retained_reviews[issue["id"]] = issue
                else:
                    item["attribution"]["review_required"] = True
                    item["attribution"]["reason"] += " Previous review retained: " + old["attribution"]["reason"]
            output.append(item)
    return output


def _overlay(originals, patches):
    """Keep untouched original spans, including protected user IDs and metadata."""
    if not patches:
        return deepcopy(originals)
    boundaries = sorted({item[key] for item in [*originals, *patches] for key in ("start", "end")})
    runs = []
    for start, end in zip(boundaries, boundaries[1:]):
        item = next((row for row in patches if row["start"] <= start and row["end"] >= end), None)
        if item is None:
            item = next(row for row in originals if row["start"] <= start and row["end"] >= end)
        # Coalesce only fragments of the same source record. This retains exact
        # user records and avoids multiplying unchanged spans at jury boundaries.
        if runs and runs[-1][0] is item and runs[-1][2] == start:
            runs[-1] = (item, runs[-1][1], end)
        else:
            runs.append((item, start, end))
    return [_slice(item, start, end) for item, start, end in runs]


def complete_speakers(document, requests, *, workers=5, progress=lambda message: None, target_block_ids=None,
                      retry_unresolved=False):
    """Resolve every unassigned spoken block with three independently complete votes.

    Three bounded rounds widen chapter and identity-evidence context. Transport failures
    propagate while successful votes remain in CachedRequests; a separate audit
    checkpoint records every case, including those not yet dispatched. Exhausted
    output truncation goes to manual selection without discarding other results.
    A null or invalid response is never an approval. Disagreement remains visibly
    blocked.

    An explicit retry gives finished unresolved cases fresh request identities,
    while an interrupted retry of the same saved document keeps identical keys.
    Already assigned source intervals are protected throughout that retry.
    """
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("workers must be an integer from 1 to 5")
    if type(retry_unresolved) is not bool:
        raise ValueError("retry_unresolved must be a boolean")
    errors = validate_document(document)
    if errors:
        raise ValueError("Invalid input analysis: " + "; ".join(errors[:12]))
    original = deepcopy(document)
    source, cast = original["source"], original["cast"]
    positions = {block["id"]: index for index, block in enumerate(source["blocks"])}
    selected = None
    if target_block_ids is not None:
        if (not isinstance(target_block_ids, (list, tuple, set, frozenset)) or not target_block_ids
                or any(not isinstance(value, str) for value in target_block_ids)
                or len(set(target_block_ids)) != len(target_block_ids)):
            raise ValueError("target_block_ids must contain unique source block IDs")
        selected = set(target_block_ids)
        if selected - positions.keys():
            raise ValueError("target_block_ids contains unknown source block IDs")
    missing = unassigned_segments(original)
    targets = {item["block_id"] for item in missing if selected is None or item["block_id"] in selected}
    by_block = {block["id"]: [] for block in source["blocks"]}
    for item in original["segments"]:
        by_block[item["block_id"]].append(item)
    originals = {block_id: by_block[block_id] for block_id in targets}
    completed = _completed_unresolved_cases(original) if retry_unresolved else {}
    cases = [{"block_id": block["id"], "original_segments": deepcopy(originals[block["id"]]),
              "unassigned_segment_ids": [item["id"] for item in missing if item["block_id"] == block["id"]],
              "completion_retry_token": fingerprint(completed[block["id"]]) if block["id"] in completed else None,
              "status": "pending", "rounds": []}
             for block in source["blocks"] if block["id"] in targets]
    audit = {"version": COMPLETION_VERSION, "source_fingerprint": fingerprint(source),
             "cast_fingerprint": fingerprint(cast), "input_script_fingerprint": fingerprint(original["segments"]),
             "provider": deepcopy(getattr(getattr(requests, "client", None), "identity", None)),
             "acceptance": "three independent matching resolved, source-validated assignments per span; no peer answers",
             "context_chars": list(CONTEXT_CHARS), "workers": min(workers, len(cases) * 3), "status": "running",
             "retry_unresolved": retry_unresolved,
             "scope": {"mode": "target_blocks" if selected is not None else "all_unassigned_blocks",
                       "target_block_ids": [case["block_id"] for case in cases]},
             "initial_unassigned_segments": len(missing), "targeted_unassigned_segments": sum(len(case["unassigned_segment_ids"]) for case in cases),
             "cases": cases}
    compact_cast = {"characters": [{key: row[key] for key in ("id", "name", "aliases", "description", "voice_id")}
                                   for row in cast["characters"]],
                    "roles": deepcopy(cast["roles"]),
                    "voices": [{"id": row["id"], "description": row["description"]} for row in cast["voices"]]}
    shared = {"final_cast": compact_cast, "performance_policy": original["provenance"].get("performance_policy", DEFAULT_POLICY),
              "completion_version": COMPLETION_VERSION, "source_fingerprint": audit["source_fingerprint"],
              "cast_fingerprint": audit["cast_fingerprint"], "retry_unresolved": retry_unresolved}
    cache_dir = getattr(requests, "cache_dir", None)
    checkpoint = Path(cache_dir) / "speaker-completion-audits" / (fingerprint(audit) + ".json") if cache_dir is not None else None
    lock = Lock()
    def save():
        if checkpoint is not None:
            write_json(checkpoint, audit, overwrite=True)
    def report(message):
        with lock:
            progress(message)
    save()
    replacements, retained_reviews = {}, {}
    try:
        for round_index, context_limit in enumerate(CONTEXT_CHARS, 1):
            pending = [case for case in cases if case["status"] not in {"resolved", "needs_user"}]
            if not pending:
                break
            report(f"Completing unassigned speakers: {len(pending)} block(s), three independent reviewers each, round {round_index}/{len(CONTEXT_CHARS)}")
            system = SYSTEM if round_index <= 2 else SYSTEM + FINAL_CONTEXT_GUIDANCE
            tasks = []
            for case in pending:
                block_id = case["block_id"]
                unit = _unit(source["blocks"][positions[block_id]])
                base = dict(shared, mode="speaker_completion", target_units=[unit], context_units=[],
                            original_segments=_compact(originals[block_id]), round=round_index,
                            input_block_fingerprint=fingerprint(originals[block_id]),
                            completion_retry_token=case["completion_retry_token"])
                # Leave room for serialization, reviewer metadata, and bounded repair hints.
                available = max(0, getattr(requests, "max_request_chars", 200000) - len(system)
                                - len(json.dumps(base, ensure_ascii=False)) - 8000)
                budget = min(context_limit, available)
                context = (_context(source, positions[block_id], budget) if round_index <= 2 else
                           _final_context(original, positions[block_id], budget, originals[block_id], case["rounds"]))
                batch = {"target_units": [unit], "context_units": context}
                round_record = {"round": round_index, "context_block_ids": [item["block_id"] for item in context],
                                "context_units_fingerprint": fingerprint(context),
                                "context_chars": sum(len(item["text"]) for item in context), "votes": []}
                case["rounds"].append(round_record)
                for reviewer, lens in REVIEWERS:
                    payload = dict(base, context_units=context, reviewer_id=reviewer, lens=lens)
                    tasks.append((case, round_record, batch, payload))
            save()
            def vote(task, cancelled):
                case, round_record, batch, payload = task
                reviewer = payload["reviewer_id"]
                record = {"reviewer_id": reviewer, "lens": payload["lens"], "request_fingerprint": fingerprint(payload)}
                last_response = None
                def accept(response):
                    nonlocal last_response
                    last_response = deepcopy(response)
                    return _accept(response, original, batch, originals[case["block_id"]],
                                   preserve_assigned=retry_unresolved)
                try:
                    record.update(requests.run(system, payload, accept, cancelled=cancelled))
                except ValueError as error:
                    record.update(status="error", resolved=False, reason=str(error), raw_response=last_response)
                except BaseException as error:
                    if isinstance(error, ProviderError) and str(error) == OUTPUT_TRUNCATED_MESSAGE:
                        record.update(status="error", resolved=False, reason=str(error),
                                      output_truncated=True, raw_response=last_response)
                    else:
                        record.update(status="interrupted", resolved=False, error_type=type(error).__name__, raw_response=last_response)
                        with lock:
                            round_record["votes"].append(record)
                            save()
                        raise
                with lock:
                    round_record["votes"].append(record)
                    save()
                    progress(f"Speaker completion {case['block_id']} round {round_index}/{len(CONTEXT_CHARS)}: {reviewer} completed")
                return record
            _bounded_map(tasks, vote, workers)
            for case in pending:
                record = case["rounds"][-1]
                record["votes"].sort(key=lambda row: row["reviewer_id"])
                current = replacements.get(case["block_id"], originals[case["block_id"]])
                agreed = _agreed_spans(record["votes"], current if retry_unresolved else originals[case["block_id"]],
                                       preserve_assigned=retry_unresolved)
                record["agreed_spans"] = [{"start": item["start"], "end": item["end"],
                                          "performers": [list(row) for row in _performers(item)], "kind": item["kind"]}
                                         for item in agreed]
                record["status"] = "unanimous" if _unanimous(record["votes"]) else "partial" if agreed else "unresolved"
                if agreed:
                    patches = _apply(agreed, originals[case["block_id"]], retained_reviews,
                                     preserve_assigned=retry_unresolved)
                    replacements[case["block_id"]] = _overlay(
                        replacements.get(case["block_id"], originals[case["block_id"]]), patches)
                current = replacements.get(case["block_id"], originals[case["block_id"]])
                if not any(_spoken(item) and not has_assigned_speakers(item) for item in current):
                    case.update(status="resolved", resolution_round=round_index)
                elif any(vote.get("output_truncated") is True for vote in record["votes"]):
                    case["status"] = "needs_user"
                    report(f"Speaker completion {case['block_id']}: the model could not return a complete answer. "
                           "Keeping validated assignments; remaining passages will be offered for speaker selection.")
                else:
                    case["status"] = "partial" if case["block_id"] in replacements else "unresolved"
            save()
    except BaseException as error:
        audit.update(status="interrupted", error_type=type(error).__name__)
        for case in cases:
            if case["status"] != "resolved":
                case["status"] = "interrupted"
        save()
        raise
    result = deepcopy(original)
    result["segments"] = [deepcopy(item) for block in source["blocks"]
                          for item in replacements.get(block["id"], by_block[block["id"]])]
    if "bracket_delivery" in original["provenance"] and replacements:
        decorated = apply_bracket_delivery(result, original["provenance"]["bracket_delivery"]["direction"])
        decorated_by_block = {block_id: [] for block_id in replacements}
        for item in decorated["segments"]:
            if item["block_id"] in decorated_by_block:
                decorated_by_block[item["block_id"]].append(item)
        # Deriving bracket cues also canonicalizes IDs. Restore protected source
        # records exactly, including custom user IDs and all saved metadata.
        for block_id, items in decorated_by_block.items():
            protected = [item for item in by_block[block_id]
                         if item["attribution"]["basis"] == "user"
                         or (retry_unresolved and has_assigned_speakers(item))]
            decorated_by_block[block_id] = _overlay(items, protected)
        result["segments"] = [deepcopy(item) for block in source["blocks"]
                              for item in decorated_by_block.get(block["id"], by_block[block["id"]])]
    remaining = unassigned_segments(result)
    resolved_blocks = {case["block_id"] for case in cases if case["status"] == "resolved"}
    resolved_targets = resolved_blocks | {item["id"] for item in missing
        if item["block_id"] in replacements and not any(other["block_id"] == item["block_id"]
            and other["start"] < item["end"] and other["end"] > item["start"] for other in remaining)}
    resolved_issues = []
    for issue in result["issues"]:
        if not issue["resolved"] and issue["kind"] in ATTRIBUTION_ISSUE_KINDS and issue["target"] in resolved_targets:
            issue.update(resolved=True, resolution={"method": "speaker_completion", "version": COMPLETION_VERSION})
            resolved_issues.append(issue["id"])
    existing_issues = {issue["id"] for issue in result["issues"]}
    result["issues"].extend(issue for key, issue in retained_reviews.items() if key not in existing_issues)
    for case in cases:
        if case["block_id"] in replacements:
            case["applied_segments"] = [deepcopy(item) for item in result["segments"] if item["block_id"] == case["block_id"]]
    # Unassigned spans already carry mandatory review flags. A second block
    # issue would double-count the same decision and remain stale after the
    # user explicitly assigns that span. All failed votes remain in this audit.
    remaining = unassigned_segments(result)
    audit.update(status="blocked" if remaining else "complete", output_script_fingerprint=fingerprint(result["segments"]),
                 resolved_blocks=len(resolved_blocks), unresolved_blocks=len(cases) - len(resolved_blocks),
                 updated_blocks=len(replacements), preserved_review_issue_ids=list(retained_reviews),
                 remaining_unassigned_segments=len(remaining),
                 remaining_unassigned_block_ids=list(dict.fromkeys(item["block_id"] for item in remaining)),
                 resolved_issue_ids=resolved_issues)
    provenance = result["provenance"]
    if "speaker_completion" in provenance:
        provenance.setdefault("speaker_completion_history", []).append(deepcopy(provenance["speaker_completion"]))
    provenance["speaker_completion"] = audit
    state = readiness(result)
    if state["errors"]:
        raise ValueError("Invalid speaker completion: " + "; ".join(state["errors"][:12]))
    result["readiness"] = state
    save()
    report(f"Speaker completion: {len(resolved_blocks)} block(s) resolved; {len(remaining)} unassigned spoken segment(s) remain"
           + (" — audiobook rendering remains blocked." if remaining else "."))
    return result
