"""Explicit series continuity with source-qualified history and stable identities."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import uuid

from .contracts import validate_document
from .narration_identity import author_name_key, author_narrator_name
from .speech_traits import validate_speech_traits


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _names(character):
    return {" ".join(unicodedata.normalize("NFKC", name).casefold().split())
            for name in [character["name"], *character["aliases"]] if name.strip()}


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get("format") != "epub-audiobook-series" or registry.get("version") != 1:
        raise ValueError("Invalid series registry format")
    try:
        uuid.UUID(registry["series_id"])
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("Invalid series registry identity") from None
    if not isinstance(registry.get("books"), list):
        raise ValueError("Series registry books must be an array")
    seen_books, owners, character_voices = set(), {}, {}
    for book in registry["books"]:
        if not isinstance(book, dict) or not isinstance(book.get("source_fingerprint"), str) or not book["source_fingerprint"]:
            raise ValueError("Series history needs source fingerprints")
        if book["source_fingerprint"] in seen_books or not isinstance(book.get("book"), dict):
            raise ValueError("Invalid or duplicate series book")
        seen_books.add(book["source_fingerprint"])
        if "series_provenance" in book and not isinstance(book["series_provenance"], dict):
            raise ValueError("Invalid series provenance")
        if not isinstance(book.get("issues", []), list) or any(
                not isinstance(issue, dict) or type(issue.get("resolved")) is not bool
                or any(not isinstance(issue.get(key), str) or not issue[key] for key in ("id", "kind", "target", "reason"))
                for issue in book.get("issues", [])):
            raise ValueError("Invalid series review issues")
        for issue in book.get("issues", []):
            origin = issue.get("series_origin")
            if "series_origin" in issue and not isinstance(origin, dict):
                raise ValueError("Invalid series review origin")
            if isinstance(origin, dict) and origin.get("kind") == "inherited_series_review":
                if (origin.get("series_id") != registry["series_id"] or not isinstance(origin.get("book"), dict)
                        or any(not isinstance(origin.get(key), str) or not origin[key]
                               for key in ("source_fingerprint", "issue_id", "issue_kind", "issue_target", "reason"))):
                    raise ValueError("Invalid inherited series review origin")
        cast = book.get("cast")
        if not isinstance(cast, dict):
            raise ValueError("Series history needs cast snapshots")
        indexes = {}
        for key in ("characters", "roles", "voices"):
            rows = cast.get(key)
            if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] for row in rows):
                raise ValueError("Invalid series cast records")
            indexes[key] = {row["id"]: row for row in rows}
            if len(indexes[key]) != len(rows):
                raise ValueError("Duplicate series cast IDs")
        for character in cast["characters"]:
            if (not isinstance(character.get("name"), str) or not isinstance(character.get("description"), str)
                    or not isinstance(character.get("aliases"), list)
                    or any(not isinstance(alias, str) for alias in character["aliases"])
                    or character.get("voice_id") not in indexes["voices"]):
                raise ValueError("Invalid series character")
            owner = owners.setdefault(character["voice_id"], character["id"])
            if owner != character["id"]:
                raise ValueError("Distinct series characters cannot share a voice ID")
            if character_voices.setdefault(character["id"], character["voice_id"]) != character["voice_id"]:
                raise ValueError("A series character must retain its established voice ID")
        for role in cast["roles"]:
            if role.get("kind") not in {"narration", "dialogue"} or role.get("voice_id") not in indexes["voices"]:
                raise ValueError("Invalid series role")
            if role.get("character_id") is not None and role["character_id"] not in indexes["characters"]:
                raise ValueError("Unknown character in series role")
        for voice in cast["voices"]:
            if not isinstance(voice.get("description"), str) or not isinstance(voice.get("traits"), list):
                raise ValueError("Invalid series voice")
            if "speechTraits" in voice:
                validate_speech_traits(voice["speechTraits"])
            for trait in voice["traits"]:
                if (not isinstance(trait, dict) or not isinstance(trait.get("name"), str)
                        or not isinstance(trait.get("value"), str) or trait.get("basis") not in {"text", "casting"}
                        or not isinstance(trait.get("evidence"), list)):
                    raise ValueError("Invalid series voice trait")
    return registry


def load_registry(path):
    path = Path(path)
    if not path.exists():
        # Stable before the first successful save, so interrupted runs can reuse
        # their validated cache. The path itself is never sent to the provider.
        identity = uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())).hex
        return {"format": "epub-audiobook-series", "version": 1, "series_id": identity, "books": []}
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ValueError("The series registry is not valid JSON") from None
    return validate_registry(registry)


def _history(registry, source):
    validate_registry(registry)
    books = registry["books"]
    characters, voices, narrators = {}, {}, {}
    for book in books:
        for voice in book["cast"]["voices"]:
            voices.setdefault(voice["id"], {"voice": voice, "source_fingerprint": book["source_fingerprint"], "book": book["book"]})
        for character in book["cast"]["characters"]:
            if character["id"] not in characters:
                characters[character["id"]] = deepcopy(character)
            else:
                old = characters[character["id"]]
                old["aliases"] = list(dict.fromkeys([*old["aliases"], old["name"], *character["aliases"]]))
                old.update(name=character["name"], description=character["description"])
        for role in book["cast"]["roles"]:
            if role["kind"] == "narration" and role.get("character_id"):
                narrators.setdefault(role["character_id"], set()).add(role["voice_id"])
    return books, characters, voices, narrators


def _author_narration_history(books, *, exclude_source=None):
    """Nearest prior author scope wins; ambiguous recent casting stays ambiguous."""
    result = {}
    for book in books:
        if book["source_fingerprint"] == exclude_source:
            continue
        authors = {}
        protected_voices = {character["voice_id"] for character in book["cast"]["characters"]}
        protected_voices.update(role["voice_id"] for role in book["cast"]["roles"] if role["id"] == "narrator_main")
        for role in book["cast"]["roles"]:
            name = author_narrator_name(role, book["book"])
            if name is not None:
                record = authors.setdefault(author_name_key(name), {"author_name": name, "voice_ids": set(), "role_ids": []})
                record["role_ids"].append(role["id"])
                if role["voice_id"] not in protected_voices:
                    record["voice_ids"].add(role["voice_id"])
        for key, record in authors.items():
            result[key] = {**record, "voice_ids": sorted(record["voice_ids"]),
                           "source_fingerprint": book["source_fingerprint"], "book": book["book"],
                           "cast": book["cast"]}
    return result


def series_context(registry, source):
    books, characters, voices, narrators = _history(registry, source)
    resolved = _resolved_review_origins(registry)
    voice_context = []
    for identity, record in voices.items():
        stored = record["voice"]
        voice = {"id": identity, "description": stored["description"],
                 "traits": [{"name": trait["name"], "value": trait["value"],
                             "basis_in_prior_book": trait["basis"]}
                            for trait in stored["traits"]],
                 "source_fingerprint": record["source_fingerprint"]}
        if "speechTraits" in stored:
            voice["speechTraits"] = stored["speechTraits"]
        voice_context.append(voice)
    return {"series_id": registry["series_id"],
            "prior_books": [{"source_fingerprint": book["source_fingerprint"], "book": book["book"],
                             "readiness": book.get("readiness"),
                             "unresolved_issues": [issue for issue in book.get("issues", [])
                                                   if not issue["resolved"] and _review_key(_review_origin(registry, book, issue)) not in resolved]}
                            for book in books],
            "characters": [{key: character[key] for key in ("id", "name", "aliases", "description", "voice_id")}
                           for character in characters.values()],
            "voices": voice_context,
            "narrator_voices": [{"character_id": identity, "voice_ids": sorted(values)} for identity, values in narrators.items()],
            "author_narrator_voices": [{key: deepcopy(record[key]) for key in
                                        ("author_name", "voice_ids", "role_ids", "source_fingerprint", "book")}
                                       for record in _author_narration_history(books, exclude_source=_hash(source)).values()]}


def validate_series_proposals(source, cast, registry):
    from .pipeline import _bounded_validation_errors, check_evidence
    _, prior, _, _ = _history(registry, source)
    evidence_errors = []
    for character_index, character in enumerate(cast["characters"]):
        proposal = character.get("series_identity")
        if proposal is None:
            continue
        if not isinstance(proposal, dict) or proposal.get("character_id") not in prior:
            raise ValueError("series_identity must reference an existing prior-series character")
        if (proposal.get("basis") not in {"context", "uncertain"} or type(proposal.get("review_required")) is not bool
                or not isinstance(proposal.get("reason"), str) or not proposal["reason"].strip()):
            raise ValueError("series_identity needs basis, reason and boolean review_required")
        if proposal["basis"] == "uncertain" and not proposal["review_required"]:
            raise ValueError("An uncertain series identity must require review")
        try:
            check_evidence(proposal.get("evidence"), source, nonempty=not proposal["review_required"],
                           label=f"cast.characters[{character_index}].series_identity.evidence",
                           record_id=character.get("id"))
        except ValueError as error:
            evidence_errors.append(str(error))
    if evidence_errors:
        raise ValueError(_bounded_validation_errors(evidence_errors))


def _confirmed_series_matches(source, cast, registry, consensus, prior):
    """Check the audit's bindings before allowing a match without name overlap.

    This consumes application-produced audit provenance, never a model-supplied
    flag on a cast record. Direct imports deliberately do not provide it.
    """
    if consensus is None:
        return None
    from .identity import _has_context
    from .identity_prompts import LENSES
    from .pipeline import check_evidence
    source_hash, registry_hash = _hash(source), _hash(registry)
    if (not isinstance(consensus, dict) or consensus.get("policy") != "unanimous_3"
            or consensus.get("source_fingerprint") != source_hash
            or consensus.get("series_registry_fingerprint") != registry_hash
            or consensus.get("final_cast_fingerprint") != _hash(cast)
            or not isinstance(consensus.get("series_cases"), list)):
        raise ValueError("Series identity consensus does not match the current source, registry, and cast")
    current = {character["id"]: character for character in cast["characters"]}
    initial = deepcopy(cast)
    initial_by_id = {character["id"]: character for character in initial["characters"]}
    cases = {}
    for case in consensus["series_cases"]:
        ids = case.get("character_ids") if isinstance(case, dict) else None
        if (not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str)
                or ids[0] not in current or ids[0] in cases or case.get("kind") != "series_match"
                or case.get("prior_character_id") not in prior
                or case.get("status") not in {"confirmed", "unresolved"}):
            raise ValueError("Series identity consensus has an invalid or duplicate character case")
        local_id, identity = ids[0], case["prior_character_id"]
        proposal = current[local_id].get("series_identity")
        if (not isinstance(proposal, dict) or proposal.get("character_id") != identity
                or proposal.get("reason") != case.get("reason")
                or proposal.get("evidence") != case.get("evidence")
                or case.get("id") != "identity_series_" + _hash([local_id, identity])[:20]):
            raise ValueError("Series identity consensus case does not match its current context proposal")
        check_evidence(case.get("evidence"), source, nonempty=True)
        # The jury saw the same cast before rejected proposals were downgraded
        # to uncertain. Reconstruct that snapshot to verify every vote's hashes.
        initial_by_id[local_id]["series_identity"].update(basis="context", review_required=False)
        cases[local_id] = case
    initial_hash, expected_reviewers = _hash(initial), dict(LENSES)
    confirmed = {}
    for local_id, case in cases.items():
        character, identity = current[local_id], case["prior_character_id"]
        proposal, votes = character["series_identity"], case.get("votes")
        if (case["status"] != "confirmed" or proposal["basis"] != "context" or proposal["review_required"]
                or not isinstance(votes, list) or len(votes) != 3):
            continue
        expected_patch = _hash({"series_identity": proposal})
        reviewers, request_hashes = set(), set()
        valid = True
        for vote in votes:
            if not isinstance(vote, dict):
                valid = False
                break
            reviewer, request_hash = vote.get("reviewer_id"), vote.get("request_fingerprint")
            if (not isinstance(reviewer, str) or reviewer not in expected_reviewers or reviewer in reviewers
                    or vote.get("lens") != expected_reviewers[reviewer] or vote.get("verdict") != "approve"
                    or not isinstance(request_hash, str) or re.fullmatch(r"[0-9a-f]{64}", request_hash) is None
                    or request_hash in request_hashes or vote.get("source_fingerprint") != source_hash
                    or vote.get("series_registry_fingerprint") != registry_hash
                    or vote.get("original_cast_fingerprint") != initial_hash
                    or vote.get("proposed_cast_fingerprint") != initial_hash
                    or vote.get("patch_fingerprint") != expected_patch):
                valid = False
                break
            try:
                check_evidence(vote.get("evidence"), source, nonempty=True)
            except ValueError:
                valid = False
                break
            if not _has_context(vote["evidence"], cast, case, {"characters": [prior[identity]]}):
                valid = False
                break
            reviewers.add(reviewer)
            request_hashes.add(request_hash)
        if valid and reviewers == set(expected_reviewers):
            confirmed[local_id] = identity
    return confirmed


def _scanned_uncertain_candidates(cast, consensus, prior):
    """Keep final scanner uncertainty visible; these candidates never authorize reuse."""
    if consensus is None or "series_scan" not in consensus:
        return {}
    scan = consensus["series_scan"]
    if not isinstance(scan, dict) or not isinstance(scan.get("decisions"), list):
        raise ValueError("Series identity scan needs a decisions array")
    current = {character["id"] for character in cast["characters"]}
    result = {}
    for row in scan["decisions"]:
        if not isinstance(row, dict):
            raise ValueError("Series identity scan decisions must be objects")
        if row.get("decision") != "uncertain":
            continue
        local_id, candidates = row.get("character_id"), row.get("prior_character_ids")
        if (not isinstance(local_id, str) or local_id not in current or local_id in result
                or not isinstance(candidates, list)
                or any(not isinstance(identity, str) or identity not in prior for identity in candidates)
                or len(candidates) != len(set(candidates))):
            raise ValueError("Uncertain series scan candidates must reference one current character and distinct prior IDs")
        result[local_id] = list(candidates)
    return result


def reconcile_series_cast(source, cast, registry, *, confirmed_identity_matches=None):
    """Canonicalize identities; only a bound audit may override name candidates.

    ``confirmed_identity_matches`` is the full provenance returned by
    ``audit_identities``, supplied by the pipeline after its independent jury.
    Ordinary imports omit it and retain the conservative exact-alias guard.
    """
    from .pipeline import base_document, segment
    validate_series_proposals(source, cast, registry)
    books, prior, prior_voices, prior_narrators = _history(registry, source)
    confirmed = _confirmed_series_matches(source, cast, registry, confirmed_identity_matches, prior)
    scanned_uncertain = _scanned_uncertain_candidates(cast, confirmed_identity_matches, prior)
    updated = deepcopy(cast)
    context = series_context(registry, source)
    namespace = uuid.UUID(registry["series_id"])
    source_hash = _hash(source)

    def stable(kind, local_id):
        return f"series_{kind}_" + uuid.uuid5(namespace, f"{source_hash}:{kind}:{local_id}").hex

    exact_records = {}
    for book in books:
        if book["source_fingerprint"] == source_hash:
            for old in book["cast"]["characters"]:
                record_hash = old.get("series_origin", {}).get("original_character_fingerprint")
                if record_hash:
                    exact_records.setdefault(record_hash, set()).add(old["id"])
    matches, uncertain, proposals, match_basis = {}, {}, {}, {}
    for character in cast["characters"]:
        names = _names(character)
        candidates = [identity for identity, old in prior.items() if names & _names(old)]
        if character["id"] in scanned_uncertain:
            uncertain[character["id"]] = list(dict.fromkeys([*candidates, *scanned_uncertain[character["id"]]]))
            continue
        exact = exact_records.get(_hash(character), set())
        if len(exact) == 1:
            proposals.setdefault(next(iter(exact)), []).append(character["id"])
            match_basis[character["id"]] = "identical_character_record_in_identical_source"
            continue
        proposal = character.get("series_identity")
        if proposal:
            identity = proposal["character_id"]
            name_words = {tuple(re.findall(r"\w+", name)) for name in names | _names(prior[identity])}
            evidence_words = [tuple(re.findall(r"\w+", item["quote"].casefold())) for item in proposal["evidence"]]
            evidence_has_context = any(words and words not in name_words for words in evidence_words)
            verified = confirmed is not None and confirmed.get(character["id"]) == identity
            if (proposal["basis"] == "context" and not proposal["review_required"] and evidence_has_context
                    and (verified or (confirmed is None and candidates == [identity]))):
                proposals.setdefault(identity, []).append(character["id"])
                match_basis[character["id"]] = ("unanimous_identity_consensus_with_current_source_evidence" if verified
                                                 else "model_context_with_current_source_evidence")
                continue
            candidates = list(dict.fromkeys([*candidates, identity]))
        if candidates:
            uncertain[character["id"]] = candidates
    for identity, local_ids in proposals.items():
        if len(local_ids) == 1:
            matches[local_ids[0]] = identity
        else:
            for local_id in local_ids:
                uncertain[local_id] = [identity]
    character_ids = {character["id"]: matches.get(character["id"], stable("char", character["id"])) for character in cast["characters"]}
    voice_ids = {voice["id"]: stable("voice", voice["id"]) for voice in cast["voices"]}
    for character in cast["characters"]:
        if character["id"] not in matches and character_ids[character["id"]] in prior:
            character_ids[character["id"]] = stable("char", character["id"] + ":" + _hash(character))
    for voice in cast["voices"]:
        if voice_ids[voice["id"]] in prior_voices:
            voice_ids[voice["id"]] = stable("voice", voice["id"] + ":" + _hash(cast))
    for character in cast["characters"]:
        if character["id"] in matches:
            voice_ids[character["voice_id"]] = prior[matches[character["id"]]]["voice_id"]

    author_origins, author_voice_records = {}, {}
    author_history = _author_narration_history(books, exclude_source=source_hash)
    protected_voices = {character["voice_id"] for character in cast["characters"]}
    protected_voices.update(role["voice_id"] for role in cast["roles"] if role["id"] == "narrator_main")
    for role in cast["roles"]:
        name = author_narrator_name(role, source["book"])
        record = author_history.get(author_name_key(name)) if name is not None else None
        if record is None or len(record["voice_ids"]) != 1 or role["voice_id"] in protected_voices:
            continue
        identity = record["voice_ids"][0]
        # Remap the local voice once, preserving any explicit author/editor
        # sharing instead of separating the roles during continuity linking.
        voice_ids[role["voice_id"]] = identity
        author_voice_records[identity] = {
            "voice": next(voice for voice in record["cast"]["voices"] if voice["id"] == identity),
            "source_fingerprint": record["source_fingerprint"], "book": record["book"],
        }
        origin = {"series_id": registry["series_id"], "basis": "author_narration_casting_continuity",
                  "author_name": name, "source_fingerprint": record["source_fingerprint"],
                  "book": deepcopy(record["book"])}
        for shared in cast["roles"]:
            if shared["voice_id"] == role["voice_id"]:
                author_origins[shared["id"]] = origin

    def inherited_voice(identity):
        record = author_voice_records.get(identity, prior_voices[identity])
        voice = deepcopy(record["voice"])
        origin = {"kind": "series_casting_reuse", "series_id": registry["series_id"],
                  "source_fingerprint": record["source_fingerprint"], "voice_id": identity}
        voice["series_origin"] = origin
        for index, trait in enumerate(voice["traits"]):
            trait["series_origin"] = dict(origin, trait_index=index, original_basis=trait["basis"])
            trait["basis"], trait["evidence"] = "casting", []
        return voice

    voices = {}
    for voice in updated["voices"]:
        new_id = voice_ids[voice["id"]]
        voice["id"] = new_id
        voices[new_id] = (inherited_voice(new_id) if new_id in prior_voices
                          and prior_voices[new_id]["source_fingerprint"] != source_hash else voice)
    for character in updated["characters"]:
        local_id = character["id"]
        character["id"] = character_ids[local_id]
        character["voice_id"] = voice_ids[character["voice_id"]]
        character["series_origin"] = {"series_id": registry["series_id"], "status": "reused" if local_id in matches else "new",
                                      "source_fingerprint": source_hash, "local_character_id": local_id,
                                      "original_character_fingerprint": _hash(next(item for item in cast["characters"] if item["id"] == local_id))}
    issues = []
    for local_id, candidates in uncertain.items():
        issues.append({"kind": "series_identity", "target": character_ids[local_id],
                       "reason": "Possible prior-series identity remains separate pending review: "
                                 + (", ".join(candidates) or "no established candidate")})
    default_narrators = {role["voice_id"] for book in books for role in book["cast"]["roles"]
                        if role["id"] == "narrator_main" and role["kind"] == "narration"
                        and role.get("character_id") is None}
    superseded = set()
    voice_replacements = {}

    def reuse_narration(role, identity):
        current_voice = role["voice_id"]
        superseded.add(current_voice)
        voice_replacements[current_voice] = identity
        role["voice_id"] = identity
        if prior_voices[identity]["source_fingerprint"] == source_hash:
            voices[identity] = dict(deepcopy(voices[current_voice]), id=identity)
        else:
            voices[identity] = inherited_voice(identity)

    for role in updated["roles"]:
        local_character = role["character_id"]
        shares_character_voice = any(character["voice_id"] == role["voice_id"] for character in cast["characters"])
        role["character_id"] = character_ids.get(local_character)
        role["voice_id"] = voice_ids[role["voice_id"]]
        if role["id"] in author_origins:
            role["series_origin"] = deepcopy(author_origins[role["id"]])
        if role["kind"] == "narration" and local_character in matches:
            choices = prior_narrators.get(matches[local_character], set())
            if len(choices) == 1:
                identity = next(iter(choices))
                reuse_narration(role, identity)
            elif choices:
                issues.append({"kind": "series_narrator", "target": role["id"],
                               "reason": "Multiple prior narration voices exist for this character; retain this book's casting pending review."})
        elif role["id"] == "narrator_main" and role["kind"] == "narration" and local_character is None:
            if shares_character_voice and default_narrators:
                issues.append({"kind": "series_narrator", "target": role["id"],
                               "reason": "Current narration explicitly shares a character voice; retain that relationship and review the prior default narration casting."})
            elif len(default_narrators) == 1:
                identity = next(iter(default_narrators))
                reuse_narration(role, identity)
                role["series_origin"] = {"series_id": registry["series_id"], "basis": "default_narration_casting_continuity"}
            elif default_narrators:
                issues.append({"kind": "series_narrator", "target": role["id"],
                               "reason": "Multiple prior default narration voices exist; choose a series casting voice during review."})
    used_voices = {item["voice_id"] for item in [*updated["characters"], *updated["roles"]]}
    # Discard only newly superseded voice records; preserve other unused model
    # profiles, which can still be useful when reviewing the discovered cast.
    for identity in superseded - used_voices:
        voices.pop(identity, None)
    voice_ids = {local_id: voice_replacements.get(identity, identity) if identity not in used_voices else identity
                 for local_id, identity in voice_ids.items()}
    updated["voices"] = list(voices.values())
    inherited_issues = _inherited_review_issues(registry, updated, set(matches.values()))
    issues.extend(inherited_issues)
    narrator = next(role for role in updated["roles"] if role["kind"] == "narration")
    errors = validate_document(base_document(source, updated, [segment(source, block["id"], 0, block["text"], narrator)
                                                              for block in source["blocks"]]))
    if errors:
        raise ValueError("; ".join(errors[:12]))
    provenance = {"series_id": registry["series_id"], "context_fingerprint": _hash(context),
                  "source_fingerprint": source_hash, "input_cast_fingerprint": _hash(cast),
                  "identity_matches": [{"local_character_id": local_id, "character_id": identity,
                                        "basis": match_basis[local_id],
                                        "prior_review_required": any(issue["target"] in {identity, prior[identity]["voice_id"]}
                                                                     for issue in inherited_issues)} for local_id, identity in matches.items()],
                  "unresolved_identity_candidates": [{"character_id": character_ids[local_id], "candidates": values}
                                                      for local_id, values in uncertain.items()],
                  "character_id_map": character_ids, "voice_id_map": voice_ids,
                  "inherited_review_issues": [{"target": issue["target"], **issue["series_origin"]}
                                               for issue in inherited_issues],
                  "human_approved": False}
    return updated, issues, provenance


def _review_origin(registry, book, issue):
    origin = issue.get("series_origin", {})
    if origin.get("kind") == "inherited_series_review":
        return deepcopy(origin)
    return {"kind": "inherited_series_review", "series_id": registry["series_id"],
            "source_fingerprint": book["source_fingerprint"], "book": deepcopy(book["book"]),
            "issue_id": issue["id"], "issue_kind": issue["kind"],
            "issue_target": issue["target"], "reason": issue["reason"]}


def _review_key(origin):
    return origin["series_id"], origin["source_fingerprint"], origin["issue_id"]


def _resolved_review_origins(registry):
    # A saved, explicitly resolved copy closes this exact source-qualified issue
    # for future inheritance. A new concern needs its own issue ID.
    return {_review_key(_review_origin(registry, book, issue)) for book in registry["books"]
            for issue in book.get("issues", []) if issue["resolved"]}


def _inherited_review_issues(registry, cast, reused_characters):
    """Keep source-qualified review state separate from identity/voice selection."""
    voices = {item["voice_id"] for item in [*cast["characters"], *cast["roles"]]}
    issues = {}
    resolved = _resolved_review_origins(registry)
    original_records = {_review_key(_review_origin(registry, book, issue)): _review_origin(registry, book, issue)
                        for book in registry["books"] for issue in book.get("issues", [])
                        if issue.get("series_origin", {}).get("kind") != "inherited_series_review"}
    for book in registry["books"]:
        targets = {identity: {identity} for identity in reused_characters | voices}
        for old_role in book["cast"]["roles"]:
            if old_role["kind"] == "dialogue" and old_role["character_id"] in reused_characters:
                targets.setdefault(old_role["id"], set()).add(old_role["character_id"])
            elif old_role["kind"] == "narration":
                for role in cast["roles"]:
                    if (role["kind"] == "narration" and role["voice_id"] == old_role["voice_id"]
                            and role["character_id"] == old_role["character_id"]
                            and role["id"] == old_role["id"]
                            and (role["id"] == "narrator_main" or role["description"] == old_role["description"])):
                        targets.setdefault(old_role["id"], set()).add(role["id"])
        for issue in book.get("issues", []):
            origin = _review_origin(registry, book, issue)
            if issue["resolved"] or _review_key(origin) in resolved:
                continue
            origin = original_records.get(_review_key(origin), origin)
            for target in sorted(targets.get(issue["target"], set())):
                inherited = {"kind": origin["issue_kind"], "target": target,
                             "reason": "Unresolved prior-series review: " + origin["reason"],
                             "series_origin": deepcopy(origin)}
                # A later snapshot may already carry this exact issue. Retain
                # its original origin instead of growing a chain of wrappers.
                issues[(*_review_key(origin), target)] = inherited
    return list(issues.values())


def add_series_issues(issues, additions, provenance):
    """Retain review flags and update targets when local cast IDs become global."""
    identities = dict(provenance["character_id_map"], **provenance["voice_id_map"])
    for issue in issues:
        issue["target"] = identities.get(issue["target"], issue["target"])
    for issue in additions:
        origin = issue.get("series_origin", {})
        if (origin.get("kind") == "inherited_series_review"
                and origin["source_fingerprint"] == provenance["source_fingerprint"]
                and any(existing["id"] == origin["issue_id"] for existing in issues)):
            continue
        identifier = "issue_" + _hash(issue)[:16]
        if not any(item["id"] == identifier for item in issues):
            issues.append(dict(issue, id=identifier, resolved=False))


def _refresh_import_reviews(document, registry, provenance):
    """Refresh history on every import while retaining the selected cast exactly."""
    cast = document["cast"]
    additions = _inherited_review_issues(registry, cast, {character["id"] for character in cast["characters"]})
    by_origin = {(*_review_key(issue["series_origin"]), issue["target"]): issue for issue in additions}
    resolved = _resolved_review_origins(registry)
    existing_origins = set()
    for issue in document["issues"]:
        origin = issue.get("series_origin", {})
        if origin.get("kind") != "inherited_series_review":
            continue
        key = (*_review_key(origin), issue["target"])
        existing_origins.add(key)
        if issue["resolved"]:
            resolved.add(_review_key(origin))
        elif _review_key(origin) in resolved:
            issue["resolved"] = True
        elif key in by_origin:
            issue.update(deepcopy(by_origin[key]))
    additions = [issue for key, issue in by_origin.items()
                 if key not in existing_origins and _review_key(issue["series_origin"]) not in resolved]
    add_series_issues(document["issues"], additions, provenance)
    inherited = [issue for issue in document["issues"]
                 if not issue["resolved"] and issue.get("series_origin", {}).get("kind") == "inherited_series_review"]
    review_origins = [{"target": issue["target"], **issue["series_origin"]} for issue in inherited]
    if provenance.get("inherited_review_issues", []) != review_origins:
        provenance["inherited_review_issues"] = review_origins
        provenance["review_context_fingerprint"] = _hash(series_context(registry, document["source"]))
    voices = {character["id"]: character["voice_id"] for character in cast["characters"]}
    for match in provenance.get("identity_matches", []):
        match["prior_review_required"] = any(issue["target"] in {match["character_id"], voices.get(match["character_id"])}
                                              for issue in inherited)


def link_analysis(document, registry):
    """Seed a series from a completed analysis without inference or model calls."""
    from .contracts import readiness
    errors = validate_document(document)
    if errors:
        raise ValueError("Cannot register an invalid analysis")
    validate_registry(registry)
    existing = document["provenance"].get("series")
    if existing and existing.get("series_id") != registry["series_id"]:
        raise ValueError("This analysis is already linked to a different series")
    result = deepcopy(document)
    previous = next((book for book in registry["books"]
                     if book["source_fingerprint"] == _hash(document["source"])
                     and book.get("series_provenance", {}).get("input_cast_fingerprint") == _hash(document["cast"])), None)
    if existing:
        cast, issues, provenance = result["cast"], [], deepcopy(existing)
    elif previous is not None:
        cast, issues, provenance = deepcopy(previous["cast"]), [], deepcopy(previous["series_provenance"])
        retained_issues = {item["id"]: deepcopy(item) for item in previous.get("issues", [])}
        retained_issues.update({item["id"]: item for item in result["issues"]})
        result["issues"] = list(retained_issues.values())
    else:
        cast, issues, provenance = reconcile_series_cast(result["source"], result["cast"], registry)
    result["cast"] = cast
    roles = {role["id"]: role for role in cast["roles"]}
    for item in result["segments"]:
        item["voice_id"] = roles[item["role_id"]]["voice_id"] if item["role_id"] else None
        if "simultaneous_speakers" in item:
            item["simultaneous_speakers"] = [
                {"role_id": speaker["role_id"], "voice_id": roles[speaker["role_id"]]["voice_id"]}
                for speaker in item["simultaneous_speakers"]]
    add_series_issues(result["issues"], issues, provenance)
    _refresh_import_reviews(result, registry, provenance)
    provenance["operation"] = "register_existing_analysis_without_model_calls"
    result["provenance"]["series"] = provenance
    result["readiness"] = readiness(result)
    if result["readiness"]["errors"]:
        raise ValueError("; ".join(result["readiness"]["errors"][:12]))
    return result


def save_registry(path, registry, document):
    """Publish a source-qualified cast snapshot, rejecting concurrent registry changes."""
    from .pipeline import write_json
    from .contracts import readiness
    errors = validate_document(document)
    if errors:
        raise ValueError("Cannot save an invalid analysis to a series registry")
    series = document.get("provenance", {}).get("series", {})
    if series.get("series_id") != registry["series_id"]:
        raise ValueError("Analysis does not belong to this explicitly selected series")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("Series registry is busy; the validated analysis remains available") from None
    try:
        os.close(descriptor)
        if _hash(load_registry(path)) != _hash(registry):
            raise ValueError("Series registry changed during analysis; refusing to overwrite another update")
        updated = deepcopy(registry)
        source_hash = _hash(document["source"])
        snapshot = {"source_fingerprint": source_hash, "book": deepcopy(document["source"]["book"]),
                    "cast": deepcopy(document["cast"]), "series_provenance": deepcopy(series),
                    "issues": deepcopy(document["issues"]), "readiness": readiness(document)}
        position = next((index for index, book in enumerate(updated["books"]) if book["source_fingerprint"] == source_hash), None)
        if position is None:
            updated["books"].append(snapshot)
        else:
            updated["books"][position] = snapshot
        validate_registry(updated)
        write_json(path, updated, overwrite=True)
    finally:
        lock.unlink()
