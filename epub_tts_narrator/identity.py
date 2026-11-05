"""Evidence-bound identity audits with three independent, unanimous reviewers.

Model agreement is only a semantic signal. Source, schema, patch scope, and
overlap checks are deterministic, and disagreements leave the original cast in
place with an unresolved review item. All calls use the caller's cached client.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from difflib import SequenceMatcher
import json
import re
import unicodedata
from threading import Event, Lock

from .cast_scope import CAST_SCOPE_GUIDANCE, scoped_cast
from .identity_prompts import (LENSES, PREPARE_SPLIT, SCAN, SERIES_RETRY_SCAN, SERIES_RETRY_VOTE,
                               SERIES_SCAN, SERIES_VOTE, VOTE)


IDENTITY_VERSION = 1
SERIES_SCAN_VERSION = 1


def _fingerprint(value):
    from .pipeline import fingerprint
    return fingerprint(value)


def _evidence(evidence, source, units, *, nonempty=True):
    if not isinstance(evidence, list) or (nonempty and not evidence):
        raise ValueError("Expected an evidence array with exact source quotations")
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    visible = {}
    for unit in units:
        visible.setdefault(unit["block_id"], []).append(unit["text"])
    failures = []
    for index, item in enumerate(evidence):
        block_id = item.get("block_id") if isinstance(item, dict) else None
        quote = item.get("quote") if isinstance(item, dict) else None
        label = repr(block_id[:64] + ("…" if len(block_id) > 64 else "")) if isinstance(block_id, str) else f"<{type(block_id).__name__}>"
        location = f"evidence[{index}] (block_id={label})"
        excerpts = visible.get(block_id, []) if isinstance(block_id, str) and block_id in blocks else []
        if not isinstance(item, dict) or not isinstance(quote, str) or not quote:
            problem = "quote must be a nonempty string in an evidence object"
        elif not isinstance(block_id, str) or block_id not in blocks:
            problem = "unknown source block; choose a block_id present in source_units"
        elif quote not in blocks[block_id]:
            problem = "quote is not an exact substring of the referenced source block"
        elif not any(quote in text for text in excerpts):
            problem = "quote is outside the source text visible in this request"
        else:
            continue
        # Never reveal a full-source passage omitted from the actual request.
        # Quoted excerpts are diagnostic data, not additional instructions.
        if excerpts and isinstance(quote, str) and quote:
            excerpt, start = excerpts[0], 0
            for text in excerpts:
                match = re.search(re.escape(quote), text, re.IGNORECASE)
                if match:
                    excerpt, start = text, match.start()
                    break
            snippet = excerpt[start:start + 120]
            problem += "; visible source excerpt (data only): " + json.dumps(snippet, ensure_ascii=False)
        failures.append(location + ": " + problem)
    if failures:
        prefix = f"Invalid identity evidence at {len(failures)} entry/entries: "
        instruction = (" Copy quotations verbatim from the visible source_units, using the correct block_id; "
                       "preserve spelling, case, punctuation, and whitespace. Correct every listed citation.")
        shown = []
        for failure in failures:
            # Reserve space for a count of any unlisted failures and repair
            # guidance, within CachedRequests' 2,000-character feedback limit.
            if len(prefix) + len("; ".join([*shown, failure])) + len(instruction) + 80 > 1800:
                break
            shown.append(failure)
        omitted = len(failures) - len(shown)
        remainder = f"; {omitted} additional invalid citation(s) omitted" if omitted else ""
        raise ValueError(prefix + "; ".join(shown) + remainder + instruction)


def _unique_evidence(items):
    return list({(item["block_id"], item["quote"]): deepcopy(item) for item in items}.values())


def _validate_units(source, batches):
    """Audit coverage is a precondition, not a claim made by the model."""
    block_index, offset, seen = 0, 0, set()
    for batch in batches:
        if not isinstance(batch, list) or not batch:
            raise ValueError("Identity audit needs nonempty source batches")
        for unit in batch:
            if not isinstance(unit, dict) or block_index >= len(source["blocks"]):
                raise ValueError("Identity source units must cover the complete source exactly once")
            block = source["blocks"][block_index]
            text, identity = unit.get("text"), unit.get("unit_id")
            if (not isinstance(identity, str) or not identity or identity in seen
                    or type(unit.get("start")) is not int or unit["start"] != offset
                    or unit.get("block_id") != block["id"] or unit.get("chapter_id") != block["chapter_id"]
                    or not isinstance(text, str) or not text
                    or block["text"][offset:offset + len(text)] != text):
                raise ValueError("Identity source units must cover original text exactly, contiguously, in order")
            seen.add(identity)
            offset += len(text)
            if offset == len(block["text"]):
                block_index += 1
                offset = 0
    if block_index != len(source["blocks"]) or offset:
        raise ValueError("Identity source units omit source text")


def _has_context(evidence, cast, case, prior=None, proposed=None):
    characters = [row for row in cast["characters"] if row["id"] in case["character_ids"]]
    if proposed:
        original_ids = {row["id"] for row in cast["characters"]}
        characters.extend(row for row in proposed["characters"] if row["id"] in case["character_ids"] or row["id"] not in original_ids)
    if prior:
        characters.extend(row for row in prior["characters"] if row["id"] == case.get("prior_character_id"))
    name_words = {word for row in characters for name in _names(row) for word in name.split()}
    # This only rejects obvious label-only citations. The jury must still decide
    # whether a longer quotation actually establishes identity or separation.
    connective = {"a", "an", "the", "and", "or", "aka"}
    return any(set(re.findall(r"\w+", item["quote"].casefold())) - name_words - connective for item in evidence)


def _visible_citations(value, source, units):
    if isinstance(value, dict):
        if "block_id" in value and "quote" in value:
            _evidence([value], source, units)
        for child in value.values():
            _visible_citations(child, source, units)
    elif isinstance(value, list):
        for child in value:
            _visible_citations(child, source, units)


def _case_key(case):
    return (case["kind"], tuple(sorted(case["character_ids"])))


def _case_id(case):
    return "identity_" + _fingerprint(_case_key(case))[:20]


def _accept_scan(response, source, cast, units):
    if not isinstance(response, dict) or not isinstance(response.get("cases"), list):
        raise ValueError("Identity scan must return a cases array")
    known = {item["id"] for item in cast["characters"]}
    cases = []
    for row in response["cases"]:
        if not isinstance(row, dict) or row.get("kind") not in {"merge", "split", "keep_separate", "uncertain"}:
            raise ValueError("Invalid identity case kind")
        ids = row.get("character_ids")
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in known for i in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("Identity cases need distinct existing character IDs")
        if row["kind"] == "split" and len(ids) != 1:
            raise ValueError("Split case must target one conflated character")
        if row["kind"] in {"merge", "keep_separate"} and len(ids) != 2:
            raise ValueError("Pair identity case must target exactly two characters")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip() or len(row["reason"]) > 2400:
            raise ValueError("Identity case needs a concise nonempty reason")
        _evidence(row.get("evidence"), source, units)
        cases.append({"kind": row["kind"], "character_ids": sorted(ids),
                      "reason": row["reason"], "evidence": deepcopy(row["evidence"])})
    return cases


def _source_context(source, cast, case, batches, all_units, whole_book):
    if whole_book:
        return all_units
    # Include complete discovering batches and the actual original blocks behind
    # all cast quotations. No summarization or silent context-budget truncation.
    block_ids = {unit["block_id"] for i in case.get("scan_indexes", []) for unit in batches[i]}
    block_ids.update(item["block_id"] for item in case["evidence"])

    def citations(value):
        if isinstance(value, dict):
            if isinstance(value.get("block_id"), str) and isinstance(value.get("quote"), str):
                block_ids.add(value["block_id"])
            for child in value.values():
                citations(child)
        elif isinstance(value, list):
            for child in value:
                citations(child)

    citations(cast)
    indexes = {i for i, block in enumerate(source["blocks"]) if block["id"] in block_ids}
    for i in list(indexes):
        chapter = source["blocks"][i]["chapter_id"]
        for neighbor in (i - 1, i + 1):
            if 0 <= neighbor < len(source["blocks"]) and source["blocks"][neighbor]["chapter_id"] == chapter:
                indexes.add(neighbor)
    return [{"unit_id": "identity_context_" + source["blocks"][i]["id"],
             "block_id": source["blocks"][i]["id"], "chapter_id": source["blocks"][i]["chapter_id"],
             "start": 0, "text": source["blocks"][i]["text"]} for i in sorted(indexes)]


def _accept_series_scan(response, source, characters, prior, units):
    if not isinstance(response, dict) or set(response) != {"decisions"} or not isinstance(response["decisions"], list):
        raise ValueError("Series identity scan must return exactly a decisions array")
    rows = response["decisions"]
    if len(rows) != len(characters):
        raise ValueError("Series identity scan must decide EVERY current character exactly once")
    prior_ids = {row["id"] for row in prior["characters"]}
    accepted = []
    for row, character in zip(rows, characters):
        if not isinstance(row, dict) or set(row) != {"character_id", "decision", "prior_character_ids", "reason", "evidence"}:
            raise ValueError("Each series decision needs only character_id, decision, prior_character_ids, reason, evidence")
        if row["character_id"] != character["id"]:
            raise ValueError("Series identity scan must decide each supplied current character once, in input order")
        if not isinstance(row["decision"], str) or row["decision"] not in {"new", "match", "uncertain"}:
            raise ValueError("Series identity decision must be new, match, or uncertain")
        candidates = row["prior_character_ids"]
        if (not isinstance(candidates, list) or any(not isinstance(item, str) or item not in prior_ids for item in candidates)
                or len(candidates) != len(set(candidates))):
            raise ValueError("Series identity candidates must be distinct existing prior-series character IDs")
        if row["decision"] == "match" and len(candidates) != 1:
            raise ValueError("A series match must propose exactly one prior identity")
        if row["decision"] == "new" and candidates:
            raise ValueError("A new character decision must have no proposed prior identities")
        if not isinstance(row["reason"], str) or not row["reason"].strip() or len(row["reason"]) > 2400:
            raise ValueError("Series identity decisions need concise source-grounded reasons")
        _evidence(row["evidence"], source, units)
        if row["decision"] == "match" and not _has_context(
                row["evidence"], {"characters": [character]},
                {"character_ids": [character["id"]], "prior_character_id": candidates[0]}, prior):
            raise ValueError("A series match needs current narrative evidence beyond names or aliases alone")
        accepted.append(deepcopy(row))
    return accepted


def _series_scan_payload(source, cast, prior, registry_hash, units, characters, scope):
    # Long shared references precede decision-specific data. All content is
    # reference material; current_characters defines the exact output scope.
    return {"source_units": units, "prior_series": prior, "current_characters": characters,
            "mode": "series_identity_scan", "series_scan_version": SERIES_SCAN_VERSION,
            "source_fingerprint": _fingerprint(source), "current_cast_fingerprint": _fingerprint(cast),
            "series_registry_fingerprint": registry_hash, "source_scope": scope}


def _series_scan_requests(source, cast, prior, registry_hash, units, requests, whole_book, *, retry_fields=None,
                          adaptive=False):
    """Prefer complete source; otherwise group characters via bisection, visibly.

    Legacy bounded mode gives each character its own request. Adaptive mode
    bisects the character list instead, so each request carries a group under
    complete source until a single character still cannot fit; only that leaf
    falls back to its selected blocks. Cited blocks are mandatory; complete
    neighboring/mention blocks are added within budget. Omitted context is
    explicitly disclosed, never treated as evidence that the person is new. An
    oversized mandatory block fails.
    """
    characters = cast["characters"]
    if not characters:
        return []
    full = _series_scan_payload(source, cast, prior, registry_hash, units, characters, "complete_source")
    if retry_fields:
        full.update(deepcopy(retry_fields))
    limit = getattr(requests, "max_request_chars", 200000)
    def size(payload):
        return len(SERIES_RETRY_SCAN if retry_fields else SERIES_SCAN) + len(json.dumps(payload, ensure_ascii=False))
    if whole_book or size(full) <= limit:
        return [full]
    # Reserve modest room for CachedRequests' validation-repair feedback.
    budget = max(1, limit - min(4000, limit // 10))
    blocks = source["blocks"]
    by_id = {block["id"]: index for index, block in enumerate(blocks)}
    result = []
    def source_units(indexes):
        return [{"unit_id": "series_context_" + blocks[index]["id"], "block_id": blocks[index]["id"],
                 "chapter_id": blocks[index]["chapter_id"], "start": 0, "text": blocks[index]["text"]}
                for index in sorted(indexes)]
    def character_request(character):
        selected = {by_id[item["block_id"]] for item in character["evidence"]}
        names = [name for name in [character["name"], *character["aliases"]] if name.strip()]
        patterns = [re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.IGNORECASE) for name in names]
        mentions = [index for index, block in enumerate(blocks) if any(pattern.search(block["text"]) for pattern in patterns)]
        if not selected:
            selected.add(mentions[0] if mentions else 0)
        payload = _series_scan_payload(source, cast, prior, registry_hash, source_units(selected), [character], "selected_blocks")
        if retry_fields:
            payload.update(deepcopy(retry_fields))
        payload["context_selection"] = {"total_source_blocks": len(blocks), "selected_source_blocks": len(selected),
                                         "omitted_source_blocks": len(blocks) - len(selected),
                                         "method": "all character-cited blocks, then complete neighboring and mention blocks within budget"}
        if size(payload) > budget:
            raise ValueError("Series identity scan's mandatory current-character evidence and prior cast exceed analysis.max_request_chars; increase the request budget")
        neighbors = []
        for index in sorted(selected):
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(blocks) and blocks[neighbor]["chapter_id"] == blocks[index]["chapter_id"]:
                    neighbors.append(neighbor)
        # Interleave early and late appearances instead of silently reading
        # only the earliest part of a frequently mentioned character's story.
        balanced = []
        for offset in range((len(mentions) + 1) // 2):
            balanced.append(mentions[offset])
            balanced.append(mentions[-offset - 1])
        for index in dict.fromkeys(neighbors + balanced):
            if index in selected:
                continue
            candidate = selected | {index}
            payload["source_units"] = source_units(candidate)
            payload["context_selection"].update(selected_source_blocks=len(candidate), omitted_source_blocks=len(blocks) - len(candidate))
            if size(payload) <= budget:
                selected = candidate
        payload["source_units"] = source_units(selected)
        payload["context_selection"].update(selected_source_blocks=len(selected), omitted_source_blocks=len(blocks) - len(selected))
        if len(selected) == len(blocks):
            payload["source_scope"] = "complete_source"
        return payload
    if adaptive:
        def grouped(group):
            payload = _series_scan_payload(source, cast, prior, registry_hash, units, group, "complete_source")
            if retry_fields:
                payload.update(deepcopy(retry_fields))
            if size(payload) <= budget:
                return [payload]
            if len(group) == 1:
                return [character_request(group[0])]
            middle = len(group) // 2
            return grouped(group[:middle]) + grouped(group[middle:])
        return grouped(characters)
    for character in characters:
        result.append(character_request(character))
    return result


def _scan_series_identities(source, cast, registry, prior, requests, all_units, whole_book, progress,
                            *, review_round=1, feedback=None, review_notes="", adaptive=False):
    registry_hash = _fingerprint(registry)
    retry_fields = ({"series_review_round": review_round, "series_review_feedback": deepcopy(feedback or []),
                     "review_notes": review_notes} if review_round > 1 else None)
    payloads = _series_scan_requests(source, cast, prior, registry_hash, all_units, requests,
                                     whole_book and not adaptive, retry_fields=retry_fields, adaptive=adaptive)
    provenance = {"version": SERIES_SCAN_VERSION, "source_fingerprint": _fingerprint(source),
                  "input_cast_fingerprint": _fingerprint(cast), "series_registry_fingerprint": registry_hash,
                  "requests": [], "decisions": [], "request_count": len(payloads)}
    for index, payload in enumerate(payloads):
        progress(f"Checking prior-series continuity {index + 1}/{len(payloads)}: {len(payload['current_characters'])} current character(s)")
        decisions = requests.run(SERIES_RETRY_SCAN if review_round > 1 else SERIES_SCAN, payload, lambda response, payload=payload:
            _accept_series_scan(response, source, payload["current_characters"], prior, payload["source_units"]))
        request_hash = _fingerprint(payload)
        provenance["requests"].append({"index": index, "request_fingerprint": request_hash,
            "character_ids": [row["id"] for row in payload["current_characters"]],
            "source_scope": payload["source_scope"],
            "source_block_ids": list(dict.fromkeys(unit["block_id"] for unit in payload["source_units"])),
            "context_selection": deepcopy(payload.get("context_selection")), "decision_count": len(decisions)})
        provenance["decisions"].extend(dict(row, request_fingerprint=request_hash) for row in decisions)
    if [row["character_id"] for row in provenance["decisions"]] != [row["id"] for row in cast["characters"]]:
        raise ValueError("Series identity scan did not decide the entire current cast exactly once")
    updated, issues = deepcopy(cast), []
    by_id = {row["id"]: row for row in updated["characters"]}
    for decision in provenance["decisions"]:
        character = by_id[decision["character_id"]]
        character.pop("series_identity", None)
        if decision["decision"] == "match":
            character["series_identity"] = {"character_id": decision["prior_character_ids"][0], "basis": "context",
                "reason": decision["reason"], "evidence": deepcopy(decision["evidence"]), "review_required": False}
        elif decision["decision"] == "uncertain":
            candidates = ", ".join(decision["prior_character_ids"]) or "no established candidate"
            issues.append({"kind": "series_identity", "target": character["id"],
                           "reason": "Prior-series continuity remains unresolved; retain a separate voice pending review. "
                                     + decision["reason"] + " Candidates: " + candidates})
    return updated, issues, provenance


def _series_retry_feedback(cases, decisions=()):
    """Brief fallible leads for the scanner; never supplied to blind jurors."""
    result = []
    for case in cases:
        if case["status"] != "unresolved":
            continue
        result.append({"character_id": case["character_ids"][0], "proposed_prior_character_id": case["prior_character_id"],
            "dissenting_reviews": [{"reviewer_id": vote["reviewer_id"], "verdict": vote["verdict"],
                                     "reason": vote.get("reason", "")[:1200],
                                     "evidence": [{"block_id": item["block_id"], "quote": item["quote"][:800]}
                                                  for item in vote.get("evidence", [])[:2]]}
                                    for vote in case.get("votes", []) if vote.get("verdict") != "approve"]})
    for decision in decisions:
        if decision["decision"] != "uncertain" or not decision["prior_character_ids"]:
            continue
        result.append({"kind": "scanner_uncertainty", "character_id": decision["character_id"],
            "candidate_prior_character_ids": deepcopy(decision["prior_character_ids"]),
            "reason": decision["reason"][:1200],
            "evidence": [{"block_id": item["block_id"], "quote": item["quote"][:800]}
                         for item in decision["evidence"][:2]]})
    return result


def _names(character):
    return {value for name in [character["name"], *character["aliases"]]
            if (value := " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", name).casefold())))}


def _collision_cases(cast, existing):
    """Name resemblance only triggers review; it never establishes identity."""
    paired = {tuple(sorted(case["character_ids"])) for case in existing if len(case["character_ids"]) == 2}
    for i, first in enumerate(cast["characters"]):
        for second in cast["characters"][i + 1:]:
            ids = tuple(sorted((first["id"], second["id"])))
            if ids in paired:
                continue
            left, right = _names(first), _names(second)
            similar = bool(left & right) or any(min(len(a), len(b)) >= 4 and
                         SequenceMatcher(None, a, b).ratio() >= .88 for a in left for b in right)
            if similar:
                yield {"kind": "keep_separate", "character_ids": list(ids),
                       "reason": "Names or aliases resemble each other. Verify the existing separation using narrative evidence; resemblance is not identity proof.",
                       "evidence": _unique_evidence(first["evidence"] + second["evidence"]),
                       "origin": "name_collision_screen", "scan_indexes": []}


def _empty_patch(**changes):
    return dict({"characters": [], "roles": [], "voices": [], "merges": [], "issues": [],
                 "summary": "Identity correction independently reviewed against source evidence."}, **changes)


def _split_ids(case, cast=None):
    prefix = case["id"]
    available = {}
    for collection, name in (("characters", "char"), ("roles", "role"), ("voices", "voice")):
        existing = {row["id"] for row in cast[collection]} if cast else set()
        allocated, index = [], 1
        while len(allocated) < 8:
            candidate = f"{prefix}_{name}_{index}"
            if candidate not in existing:
                allocated.append(candidate)
            index += 1
        available[collection] = allocated
    return available


def _accept_split(response, source, cast, case, units, available):
    from .pipeline import merge_patch
    if not isinstance(response, dict) or response.get("merges") != []:
        raise ValueError("A split proposal cannot contain merges")
    target = case["character_ids"][0]
    original = next(character for character in cast["characters"] if character["id"] == target)
    owned = {role["id"]: role for role in cast["roles"] if role["character_id"] == target}
    allowed = {"characters": {target, *available["characters"]},
               "roles": set(owned) | set(available["roles"]), "voices": set(available["voices"])}
    response = deepcopy(response)
    for collection in allowed:
        rows = response.get(collection)
        if not isinstance(rows, list):
            raise ValueError(f"Split patch {collection} must be an array")
        originals = {row["id"]: row for row in cast[collection]}
        retained = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Split patch {collection} must contain record objects")
            if row.get("id") not in allowed[collection]:
                if row == originals.get(row.get("id")):
                    continue  # A repeated unchanged record is an exact no-op.
                raise ValueError(f"Split patch must not edit unrelated or unallocated {collection}; "
                                 "omit unchanged records, which the application retains automatically")
            retained.append(row)
        response[collection] = retained
    parts = {row["id"]: row for row in response["characters"]}
    if target not in parts or len(parts) < 2:
        raise ValueError("Split patch must preserve and refine the original and add a distinct character")
    if parts[target].get("voice_id") != original["voice_id"]:
        raise ValueError("The retained split person must preserve the original voice ID")
    if any(row.get("voice_id") not in available["voices"] for identity, row in parts.items() if identity != target):
        raise ValueError("New split characters need allocated distinct voice IDs")
    incoming_roles = {role["id"]: role for role in response["roles"]}
    if not set(owned).issubset(incoming_roles):
        raise ValueError("Split patch must explicitly redistribute every role owned by the conflated character")
    for role in response["roles"]:
        if role.get("character_id") not in parts:
            raise ValueError("Split roles must belong to one of the split people")
        old = owned.get(role["id"])
        if old and (role.get("kind") != old["kind"] or
                    (old["kind"] == "narration" and old["voice_id"] != original["voice_id"] and role.get("voice_id") != old["voice_id"])):
            raise ValueError("Split patch must preserve role kinds and independently cast narration voices")
    patch = deepcopy(response)
    for character in patch["characters"]:
        _evidence(character.get("evidence"), source, units)
        character.pop("series_identity", None)
        character.pop("series_origin", None)
        character.pop("series_identity_candidates", None)
    _visible_citations(patch, source, units)
    proposed, issues, _ = merge_patch(source, cast, patch, available)
    return patch, proposed, issues


def _issue(case, reason):
    return {"kind": "identity_consensus", "target": case["character_ids"][0],
            "reason": f"{case['id']}: {reason}"}


def _vote_payload(cast_view, case, reviewer, lens, *, source_fingerprint, cast_fingerprint):
    """One reviewer's request; identical key order preserves cache identities.

    ``cast_view`` is the original cast supplied to this reviewer — the complete
    cast, or a complete-record subset disclosed by an attached cast_scope
    object — while ``cast_fingerprint`` always identifies the audit's complete
    original cast so votes stay auditable against it.
    """
    payload = {"source_units": case["source_units"], "original_cast": cast_view}
    if "vote_scope" in case:
        payload["cast_scope"] = deepcopy(case["vote_scope"])
    if "prior_series" in case:
        payload["prior_series"] = case["prior_series"]
    payload.update({"mode": "identity_vote", "identity_version": IDENTITY_VERSION,
               "source_fingerprint": source_fingerprint,
               "original_cast_fingerprint": cast_fingerprint})
    if "prior_series" in case:
        payload["series_registry_fingerprint"] = case["series_registry_fingerprint"]
    if "proposed_scope" in case:
        payload["proposed_cast_scope"] = deepcopy(case["proposed_scope"])
    payload.update({"case": case["case"], "proposed_cast": case["proposed_cast"],
                "patch": case["patch"],
                "proposed_cast_fingerprint": case.get("canonical_proposed_fingerprint", _fingerprint(case["proposed_cast"])),
               "patch_fingerprint": _fingerprint(case["patch"]),
                "reviewer_id": reviewer, "lens": lens})
    review_round = case.get("series_review_round", 1)
    if review_round > 1:
        # Fresh blind samples even if the second scan proposes an identical
        # cast. The round discriminator leaves first-round payloads intact.
        payload["series_review_round"] = review_round
    return payload


def _vote_cast_view(requests, full_cast, prepared, system, *, source_fingerprint, cast_fingerprint, source):
    """Choose one case's complete affected records, never unrelated cast context.

    Returns ``(cast_view, scope, viable)``. A False ``viable`` result means even
    the explicit complete-record scope exceeds the request budget; the case
    must stay unresolved without dispatching a partial jury. The size probe
    uses the longest reviewer identity, so a viable full-cast decision always
    fits every reviewer's actual payload.
    """
    limit = getattr(requests, "max_request_chars", 200000)
    proposed = prepared["proposed_cast"]
    prepared["canonical_proposed_fingerprint"] = _fingerprint(proposed)
    affected_ids = set(prepared["case"]["character_ids"])
    affected_ids.update(row["id"] for row in prepared["patch"].get("characters", []))
    proposed_ids = {row["id"] for row in proposed["characters"]}
    proposed_view, proposed_scope = scoped_cast(
        source, proposed, (), character_ids=sorted(affected_ids & proposed_ids), identity_case=True)
    prepared["proposed_cast"], prepared["proposed_scope"] = proposed_view, proposed_scope
    reviewer, lens = max(LENSES, key=lambda row: len(row[0]) + len(row[1]))
    def size(cast_view, scope, prompt):
        probe = dict(prepared, vote_cast=cast_view)
        if scope is not None:
            probe["vote_scope"] = scope
        payload = _vote_payload(cast_view, probe, reviewer, lens,
                                source_fingerprint=source_fingerprint, cast_fingerprint=cast_fingerprint)
        return len(prompt) + len(json.dumps(payload, ensure_ascii=False))
    view, scope = scoped_cast(source, full_cast, (),
                              character_ids=sorted(set(prepared["case"]["character_ids"])), identity_case=True)
    if size(view, scope, system + CAST_SCOPE_GUIDANCE) <= limit:
        return view, scope, True
    return view, scope, False


def _votes(requests, source, cast, cases, workers, *, progress=lambda message: None, label="Identity jury"):
    """The immutable request snapshot prevents peer votes or earlier edits leaking in."""
    from .identity_vote_validation import make_vote_accept
    tasks = [(case, reviewer, lens) for case in cases for reviewer, lens in LENSES]
    cancelled, failure_lock, failures = Event(), Lock(), []
    source_hash, cast_hash = _fingerprint(source), _fingerprint(cast)

    def review(task):
        case, reviewer, lens = task
        # Providers can reuse an identical serialized prompt prefix. Put the
        # long shared source/reference data first, then the candidate, and the
        # private reviewer identity last. Fingerprints sort object keys, so this
        # ordering preserves existing validated response-cache identities.
        cast_view = case.get("vote_cast", cast)
        payload = _vote_payload(cast_view, case, reviewer, lens,
                                source_fingerprint=source_hash, cast_fingerprint=cast_hash)
        review_round = case.get("series_review_round", 1)

        accept = make_vote_accept(
            reviewer=reviewer, lens=lens, source=source, source_units=case["source_units"],
            cast_view=cast_view, identity_case=case["case"], proposed_cast=case["proposed_cast"],
            prior_series=case.get("prior_series"), request_fingerprint=_fingerprint(payload),
            fingerprints={key: payload[key] for key in ("source_fingerprint", "original_cast_fingerprint",
                                                       "proposed_cast_fingerprint", "patch_fingerprint")},
            series_registry_fingerprint=payload.get("series_registry_fingerprint"), review_round=review_round)

        try:
            system = (SERIES_RETRY_VOTE if review_round > 1 else SERIES_VOTE) if "prior_series" in case else VOTE
            if "vote_scope" in case:
                system += CAST_SCOPE_GUIDANCE
            vote = requests.run(system, payload, accept, cancelled=cancelled)
        except ValueError as exc:
            # Exhausted schema repairs or a context-budget failure cannot vote.
            vote = {"reviewer_id": reviewer, "lens": lens, "verdict": "error", "reason": str(exc),
                    "request_fingerprint": _fingerprint(payload),
                    **{key: payload[key] for key in ("source_fingerprint", "original_cast_fingerprint",
                         "proposed_cast_fingerprint", "patch_fingerprint", "series_registry_fingerprint", "series_review_round") if key in payload}}
        return case["case"]["id"], vote

    def run(task):
        try:
            if cancelled.is_set():
                raise CancelledError("Identity reviews cancelled after a fatal request failure")
            return review(task)
        except BaseException as exc:
            # Signal from the failing worker, before its future is collected.
            # Other workers may finish/cache an already successful response,
            # but CachedRequests will not start another repair/API attempt.
            with failure_lock:
                if not failures:
                    failures.append(exc)
                cancelled.set()
            raise

    results = {}
    if tasks:
        # Submit only a bounded window. A fatal transport/authentication failure
        # must not dispatch the remaining book's reviews while waiting for other
        # requests already in flight to finish and preserve their cache entries.
        pool = ThreadPoolExecutor(max_workers=workers)
        pending, completed = {}, [None] * len(tasks)
        next_index = finished = 0
        try:
            while next_index < min(workers, len(tasks)) and not cancelled.is_set():
                pending[pool.submit(run, tasks[next_index])] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Collect all finished results before scheduling successors so a
                # simultaneous successful completion cannot mask a fatal error.
                for future in done:
                    index = pending.pop(future)
                    completed[index] = future.result()
                    finished += 1
                    if finished == 1 or finished % 5 == 0 or finished == len(tasks):
                        progress(f"{label}: {finished}/{len(tasks)} reviewer responses complete")
                while next_index < len(tasks) and len(pending) < workers and not cancelled.is_set():
                    pending[pool.submit(run, tasks[next_index])] = next_index
                    next_index += 1
        except BaseException:
            cancelled.set()
            for future in pending:
                future.cancel()
            if failures:
                # A sibling's cancellation must never hide the actual provider
                # error that caused cancellation, regardless of completion order.
                raise failures[0]
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        for identity, vote in completed:
            results.setdefault(identity, []).append(vote)
    return results


def _unanimous(votes):
    return len(votes) == 3 and {vote["reviewer_id"] for vote in votes} == {row[0] for row in LENSES} and all(vote["verdict"] == "approve" for vote in votes)


_BUDGET_UNDISPATCHED = ("The complete case records exceed the request budget even with an explicit "
                        "complete-record cast scope; no reviewer was dispatched and the original cast is retained.")


def _adaptive_case_context(source, cast, case, all_units):
    view, _ = scoped_cast(source, cast, (), character_ids=case["character_ids"], identity_case=True)
    return _source_context(source, view, dict(case, scan_indexes=[]), [], all_units, False)


def _adaptive_identity_scans(source, initial, all_units, requests, progress, *, regions=None):
    """Run adaptive book scans; return one ``(payload, scope, found, group)`` leaf per actual scan, in order.

    Each leaf carries the full cast when the request fits and a complete-record
    cast scope otherwise. Every character is visible in at least one executed
    scan: additional candidate groups are scheduled — and their coverage
    verified — for anyone the regions did not include, instead of assuming the
    lexical scope was complete. No record or evidence is ever trimmed.
    """
    from .adaptive_requests import run_partitioned, request_size
    limit = getattr(requests, "max_request_chars", 200000)
    system = SCAN + CAST_SCOPE_GUIDANCE
    source_hash, cast_hash = _fingerprint(source), _fingerprint(initial)
    all_ids = sorted(row["id"] for row in initial["characters"])
    records = []

    def build(units, group=()):
        def construct(cast_view, scope):
            payload = {"mode": "identity_scan", "identity_version": IDENTITY_VERSION,
                       "source_fingerprint": source_hash, "original_cast_fingerprint": cast_hash,
                       "original_cast": cast_view, "source_units": units,
                       "source_scope": "complete_source" if units == all_units else "source_batch"}
            if scope is not None:
                payload["cast_scope"] = scope
            return payload
        full = construct(initial, {"complete": True, "full_cast_fingerprint": cast_hash,
                                   "included_ids": list(all_ids),
                                   "omitted_counts": {"characters": 0, "roles": 0, "voices": 0},
                                   "selection": "complete cast; every character, role, and voice record supplied"})
        if request_size(system, full) <= limit:
            return system, full
        view, scope = scoped_cast(source, initial, units, character_ids=sorted(set(group)))
        return system, construct(view, scope)

    def execute(system_prompt, payload):
        found = requests.run(system_prompt, payload, lambda response: _accept_scan(
            response, source, payload["original_cast"], payload["source_units"]))
        records.append((payload, payload["cast_scope"], found))
        return found

    def scan(items, label, group=()):
        start = len(records)
        run_partitioned(items, lambda units: build(units, group), execute,
                        progress=progress, label=label, requests=requests)
        return [(payload, scope, found, tuple(group)) for payload, scope, found in records[start:]]

    leaves = []
    for region in (regions if regions is not None else [all_units]):
        leaves.extend(scan(region, "Adaptive identity scan"))
    covered = set()
    for payload, scope, found, group in leaves:
        covered.update(scope["included_ids"])
    # Explicit regions (dirty re-audit) are the caller's targeted selection; the
    # candidate-coverage guarantee applies to computed adaptive book scans.
    if regions is not None:
        return leaves
    missing = [row["id"] for row in initial["characters"] if row["id"] not in covered]
    if missing:
        progress("Scheduling additional identity candidate group(s) for " + str(len(missing))
                 + " character(s) not visible in any region")
        added = scan(all_units, "Adaptive identity candidate group", missing)
        leaves.extend(added)
        for payload, scope, found, group in added:
            covered.update(scope["included_ids"])
        still = [row["id"] for row in initial["characters"] if row["id"] not in covered]
        if still:
            raise ValueError("Identity candidate(s) " + ", ".join(still) + " did not occur in any executed "
                             "adaptive scan; candidate coverage is a precondition, not an assumption")
    return leaves


def audit_identities(source, cast, requests, batches, *, whole_book=False, pending_merges=(),
                      series_registry=None, workers=1, review_notes="", progress=lambda message: None,
                      adaptive=False, scan_regions=None, collision_state=None, resolve_candidate_collisions=True):
    """Return an audited copy, new unresolved issue specs, and reproducible evidence.

Scans and candidate preparation finish before blind judging. Mutations are then
applied only for disjoint, source-valid candidates with unanimous approval.
"""
    from .pipeline import check_cast, check_evidence, merge_patch
    from .series import series_context, validate_series_proposals
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Identity workers must be an integer from 1 to 8")
    if not isinstance(review_notes, str):
        raise ValueError("Identity review_notes must be a string")
    if not isinstance(adaptive, bool):
        raise ValueError("Identity adaptive must be a boolean")
    if type(resolve_candidate_collisions) is not bool:
        raise ValueError("Identity resolve_candidate_collisions must be a boolean")
    if collision_state is not None and not adaptive:
        raise ValueError("Identity collision_state requires adaptive identity review")
    check_cast(source, cast)
    _validate_units(source, batches)
    initial = deepcopy(cast)
    all_units = [unit for batch in batches for unit in batch]
    if scan_regions is not None:
        if not adaptive:
            raise ValueError("Identity scan_regions requires adaptive identity scans")
        if not isinstance(scan_regions, list) or any(
                not isinstance(region, list) or any(not isinstance(unit, dict) for unit in region)
                for region in scan_regions):
            raise ValueError("Identity scan_regions must be a list of source-unit region lists")
        by_id = {unit["unit_id"]: unit for unit in all_units}
        for region in scan_regions:
            for unit in region:
                if by_id.get(unit.get("unit_id")) != unit:
                    raise ValueError("Identity scan_regions may only contain source units from the audited batches")
    scans = [all_units] if whole_book else batches
    if not scans or not all_units:
        raise ValueError("Identity audit needs source units")
    source_hash, cast_hash = _fingerprint(source), _fingerprint(initial)
    if adaptive:
        from .identity_collisions import collision_state_for_source
        collision_state = collision_state_for_source(collision_state, source_hash)
    provenance = {"version": IDENTITY_VERSION, "policy": "unanimous_3", "source_fingerprint": source_hash,
                  "original_cast_fingerprint": cast_hash, "original_cast": deepcopy(initial),
                  "scans": [], "cases": [], "series_cases": []}
    cases = {}

    def add(case, indexes=()):
        key = _case_key(case)
        if key in cases:
            current = cases[key]
            current["evidence"] = _unique_evidence(current["evidence"] + case["evidence"])
            current["scan_indexes"] = sorted(set(current["scan_indexes"]) | set(indexes))
            if case.get("merge"):
                current["merge"] = deepcopy(case["merge"])
            current.setdefault("observations", []).append(case["reason"])
        else:
            cases[key] = dict(deepcopy(case), id=_case_id(case), scan_indexes=list(indexes))

    context_batches, context_whole = batches, whole_book
    if adaptive:
        leaves = _adaptive_identity_scans(source, initial, all_units, requests, progress, regions=scan_regions)
        for leaf_index, (payload, scope, found, group) in enumerate(leaves):
            progress(f"Auditing character identities {leaf_index + 1}/{len(leaves)}: "
                     f"{len(payload['source_units'])} source unit(s), {len(scope['included_ids'])} visible cast record(s)")
            provenance["scans"].append({"index": leaf_index, "request_fingerprint": _fingerprint(payload),
                                        "case_count": len(found), "source_scope": payload["source_scope"],
                                        "cast_scope": deepcopy(scope), "candidate_character_ids": list(group)})
            for found_case in found:
                add(found_case, [leaf_index])
        context_batches, context_whole = [payload["source_units"] for payload, _, _, _ in leaves], False
    else:
        for index, units in enumerate(scans):
            progress(f"Auditing character identities {index + 1}/{len(scans)}")
            payload = {"mode": "identity_scan", "identity_version": IDENTITY_VERSION,
                       "source_fingerprint": source_hash, "original_cast_fingerprint": cast_hash,
                       "original_cast": initial, "source_units": units,
                       "source_scope": "complete_source" if len(scans) == 1 else "source_batch"}
            found = requests.run(SCAN, payload, lambda response: _accept_scan(response, source, initial, units))
            provenance["scans"].append({"index": index, "request_fingerprint": _fingerprint(payload), "case_count": len(found)})
            for case in found:
                add(case, range(len(batches)) if whole_book else [index])
    known = {row["id"] for row in initial["characters"]}
    for merge in pending_merges:
        origin, target = merge.get("from_character_id"), merge.get("into_character_id")
        if origin not in known or target not in known or origin == target:
            raise ValueError("Pending identity merge must reference two distinct retained characters")
        check_evidence(merge.get("evidence"), source, nonempty=True)
        add({"kind": "merge", "character_ids": sorted((origin, target)),
             "reason": "Discovery proposed that these records represent the same person; independent verification is required.",
             "evidence": deepcopy(merge["evidence"]), "merge": deepcopy(merge), "origin": "pending_discovery_merge"})
    if not adaptive:
        for case in _collision_cases(initial, list(cases.values())):
            add(case)
    cases = list(cases.values())
    prepared, issues = [], []
    for case in cases:
        record = deepcopy(case)
        record["votes"] = []
        provenance["cases"].append(record)
        if case["kind"] == "uncertain":
            record["status"] = "unresolved"
            issues.append(_issue(case, case["reason"]))
            continue
        if adaptive and case["kind"] == "merge":
            from .identity_collisions import prior_merge_conflict
            if prior_merge_conflict(collision_state, case["character_ids"], source_hash):
                record["status"] = "prior_identity_conflict"
                issues.append(_issue(case, "A retained negative identity constraint conflicts with this merge; "
                                     "source records remain separate for review."))
                continue
        units = (_adaptive_case_context(source, initial, case, all_units) if adaptive else
                 _source_context(source, initial, case, context_batches, all_units, context_whole))
        available = {"characters": [], "roles": [], "voices": []}
        split_payload = split_system = None
        if case["kind"] == "split":
            available = _split_ids(case, initial)
            split_payload = {"mode": "identity_prepare_split", "identity_version": IDENTITY_VERSION,
                             "case": case, "original_cast": initial, "available_ids": available,
                             "source_units": units, "source_fingerprint": source_hash,
                             "original_cast_fingerprint": cast_hash}
            split_system = PREPARE_SPLIT
            if adaptive:
                limit = getattr(requests, "max_request_chars", 200000)
                view, scope = scoped_cast(source, initial, (), character_ids=case["character_ids"], identity_case=True)
                split_payload["original_cast"] = view
                split_payload["cast_scope"] = scope
                split_system = PREPARE_SPLIT + CAST_SCOPE_GUIDANCE
                if len(split_system) + len(json.dumps(split_payload, ensure_ascii=False)) > limit:
                    record["status"] = "unresolved"
                    issues.append(_issue(case, _BUDGET_UNDISPATCHED))
                    continue
        try:
            if case["kind"] == "split":
                patch, proposed, extra = requests.run(split_system, split_payload,
                    lambda response: _accept_split(response, source, initial, case, units, available))
            else:
                merges = []
                if case["kind"] == "merge":
                    merges = [case.get("merge") or {"from_character_id": case["character_ids"][1],
                              "into_character_id": case["character_ids"][0], "evidence": case["evidence"]}]
                patch = _empty_patch(merges=merges)
                proposed, extra, _ = merge_patch(source, initial, patch, available)
            record.update(patch=deepcopy(patch), proposed_cast_fingerprint=_fingerprint(proposed))
            candidate = {"case": case, "patch": patch, "proposed_cast": proposed,
                         "source_units": units, "available": available, "issues": extra, "record": record}
            if adaptive:
                vote_cast, vote_scope, viable = _vote_cast_view(requests, initial, candidate, VOTE,
                                                                source_fingerprint=source_hash,
                                                                cast_fingerprint=cast_hash, source=source)
                if not viable:
                    record["status"] = "unresolved"
                    issues.append(_issue(case, _BUDGET_UNDISPATCHED))
                    continue
                candidate["vote_cast"] = vote_cast
                if vote_scope is not None:
                    candidate["vote_scope"] = vote_scope
            prepared.append(candidate)
        except ValueError as exc:
            record.update(status="invalid_proposal", error=str(exc))
            issues.append(_issue(case, "Proposal failed deterministic validation: " + str(exc)))
    progress(f"Independently reviewing {len(prepared)} identity decisions with three reviewers each")
    votes = _votes(requests, source, initial, prepared, workers, progress=progress)
    # No ordering of pairwise approvals can authorize an unreviewed transitive
    # merge. Overlapping mutations, or contradictory separation findings, wait.
    mutations = [case for case in cases if case["kind"] in {"merge", "split"}]
    conflicts = set()
    for i, left in enumerate(mutations):
        for right in mutations[i + 1:]:
            if set(left["character_ids"]) & set(right["character_ids"]):
                conflicts.update((left["id"], right["id"]))
        for right in cases:
            if right["kind"] in {"keep_separate", "uncertain"} and set(left["character_ids"]) & set(right["character_ids"]):
                conflicts.add(left["id"])
    updated = deepcopy(initial)
    for candidate in prepared:
        case, record = candidate["case"], candidate["record"]
        record["votes"] = votes.get(case["id"], [])
        if case["id"] in conflicts:
            record["status"] = "overlapping_cases"
            issues.append(_issue(case, "Overlapping or conflicting identity cases require review; original records retained."))
        elif not _unanimous(record["votes"]):
            record["status"] = "unresolved"
            issues.append(_issue(case, "Three reviewers did not unanimously approve: " + ", ".join(v["verdict"] for v in record["votes"])))
        else:
            updated, new_issues, _ = merge_patch(source, updated, candidate["patch"], candidate["available"])
            # Remove only the automatic identity_merge issue created by THIS
            # approved patch. Existing caller issues are never touched here.
            issues.extend(issue for issue in new_issues if case["kind"] != "merge" or issue["kind"] != "identity_merge")
            record["status"] = "applied" if case["kind"] in {"merge", "split"} else "confirmed_separate"
    if adaptive and resolve_candidate_collisions:
        from .identity_collisions import resolve_collisions
        updated, collision_issues, collision_provenance = resolve_collisions(
            source, updated, requests, original_cast=initial, source_cases=provenance["cases"],
            prior=collision_state, workers=workers, progress=progress)
        issues.extend(collision_issues)
        provenance["collision_resolution"] = collision_provenance
        provenance["cases"].extend(deepcopy(collision_provenance["cases"]))
    elif adaptive and collision_state is not None:
        provenance["collision_resolution"] = {"state": deepcopy(collision_state)}
    if series_registry is not None:
        prior = series_context(series_registry, source)
        provenance["series_registry_fingerprint"] = _fingerprint(series_registry)
        feedback = None
        for review_round in (1, 2):
            round_issues = []
            if prior["prior_books"]:
                updated, continuity_issues, continuity_scan = _scan_series_identities(
                    source, updated, series_registry, prior, requests, all_units, whole_book and not adaptive, progress,
                    review_round=review_round, feedback=feedback, review_notes=review_notes, adaptive=adaptive)
                round_issues.extend(continuity_issues)
                provenance["series_scan"] = continuity_scan
                provenance["series_review_rounds"] = review_round
            validate_series_proposals(source, updated, series_registry)
            series_initial = deepcopy(updated)
            series_prepared = []
            series_hash = _fingerprint(series_initial)
            provenance["series_cases"] = []
            for character in series_initial["characters"]:
                proposal = character.get("series_identity")
                if not proposal or proposal["basis"] != "context" or proposal["review_required"]:
                    continue
                case = {"id": "identity_series_" + _fingerprint([character["id"], proposal["character_id"]])[:20],
                        "kind": "series_match", "character_ids": [character["id"]],
                        "prior_character_id": proposal["character_id"], "reason": proposal["reason"],
                        "evidence": deepcopy(proposal["evidence"]), "scan_indexes": []}
                record = deepcopy(case)
                provenance["series_cases"].append(record)
                candidate = {"case": case, "patch": {"series_identity": deepcopy(proposal)},
                    "proposed_cast": series_initial, "prior_series": prior, "record": record,
                    "series_registry_fingerprint": provenance["series_registry_fingerprint"],
                    "source_units": (_adaptive_case_context(source, series_initial, case, all_units) if adaptive else
                                     _source_context(source, series_initial, case, context_batches, all_units, context_whole))}
                if review_round > 1:
                    candidate["series_review_round"] = review_round
                if adaptive:
                    system = SERIES_RETRY_VOTE if review_round > 1 else SERIES_VOTE
                    vote_cast, vote_scope, viable = _vote_cast_view(requests, series_initial, candidate, system,
                                                                    source_fingerprint=source_hash,
                                                                    cast_fingerprint=series_hash, source=source)
                    if not viable:
                        record.update(votes=[], status="unresolved")
                        next(row for row in updated["characters"] if row["id"] == character["id"])["series_identity"].update(
                            basis="uncertain", review_required=True)
                        round_issues.append(_issue(case, _BUDGET_UNDISPATCHED))
                        continue
                    candidate["vote_cast"] = vote_cast
                    if vote_scope is not None:
                        candidate["vote_scope"] = vote_scope
                series_prepared.append(candidate)
            if series_prepared:
                progress(f"Independently reviewing {len(series_prepared)} series identity matches (round {review_round}/2)")
            series_votes = _votes(requests, source, series_initial, series_prepared, min(workers, 5),
                                  progress=progress, label=f"Series identity jury (round {review_round}/2)")
            by_id = {row["id"]: row for row in updated["characters"]}
            for candidate in series_prepared:
                case, record = candidate["case"], candidate["record"]
                record["votes"] = series_votes.get(case["id"], [])
                record["status"] = "confirmed" if _unanimous(record["votes"]) else "unresolved"
                if record["status"] == "unresolved":
                    proposal = by_id[case["character_ids"][0]]["series_identity"]
                    proposal.update(basis="uncertain", review_required=True)
                    round_issues.append(_issue(case, "Prior-series identity did not receive unanimous independent approval; voice reuse is withheld."))
            unresolved = any(case["status"] == "unresolved" for case in provenance["series_cases"])
            scan_decisions = provenance.get("series_scan", {}).get("decisions", [])
            uncertain_candidates = any(row["decision"] == "uncertain" and row["prior_character_ids"]
                                       for row in scan_decisions)
            if review_round == 1 and (unresolved or uncertain_candidates) and prior["prior_books"]:
                provenance.setdefault("series_review_history", []).append({
                    "round": review_round, "series_scan": deepcopy(provenance["series_scan"]),
                    "series_cases": deepcopy(provenance["series_cases"]),
                    "final_cast": deepcopy(updated), "final_cast_fingerprint": _fingerprint(updated),
                    "retry_triggers": (["jury_dissent"] if unresolved else [])
                                      + (["scanner_uncertainty_with_candidates"] if uncertain_candidates else []),
                    "issues": deepcopy(round_issues)})
                feedback = _series_retry_feedback(provenance["series_cases"], scan_decisions)
                progress("Unresolved series continuity: rechecking the complete current cast once, followed by fresh blind reviews")
                continue
            # Only this stage's superseded first-round issues are retired.
            # Within-book and caller-owned issues remain untouched.
            issues.extend(round_issues)
            break
    check_cast(source, updated)
    provenance["final_cast_fingerprint"] = _fingerprint(updated)
    return updated, issues, provenance
