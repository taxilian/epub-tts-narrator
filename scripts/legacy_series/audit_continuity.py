"""Read-only continuity audit for this ordered seven-book trial.

Run from any directory with --root pointing to the legacy run. With no --book,
inspect completed book-XX/analysis.json files under that root. This checks
mechanical guarantees and surfaces identity review candidates;
it does not establish literary identity from name resemblance or jury agreement.
Only -o writes a file, exclusively unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
import unicodedata

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from epub_tts_narrator.contracts import readiness, validate_document
from epub_tts_narrator.cli import source_from
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import _inherited_review_issues, validate_registry
from bundle_validation import load_bundle, validate_native_bundle

REVIEWERS = {f"identity_reviewer_{number}" for number in range(1, 4)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def names(character):
    return {normalized for value in [character["name"], *character["aliases"]]
            if (normalized := " ".join(unicodedata.normalize("NFKC", value).casefold().split()))}


def profile(voice):
    return (voice["description"], [(trait["name"], trait["value"]) for trait in voice["traits"]])


def issue_key(issue):
    origin = issue.get("series_origin", {})
    return (origin.get("series_id"), origin.get("source_fingerprint"),
            origin.get("issue_id"), issue.get("target"))


def compact_issue(issue):
    result = {key: issue[key] for key in ("id", "kind", "target", "resolved") if key in issue}
    origin = issue.get("series_origin", {})
    if origin.get("kind") == "inherited_series_review":
        result["origin"] = {key: origin.get(key) for key in
                            ("series_id", "source_fingerprint", "issue_id", "issue_kind", "issue_target")}
    return result


def finding(report, code, **details):
    report["errors"].append(dict(code=code, **details))


def valid_evidence(evidence, blocks):
    return (isinstance(evidence, list) and bool(evidence) and all(
        isinstance(item, dict) and isinstance(item.get("quote"), str) and bool(item["quote"])
        and item.get("block_id") in blocks and item["quote"] in blocks[item["block_id"]]
        for item in evidence))


def audit_votes(case, blocks, source_hash, registry_hash=None):
    """Return machine failures; never include source quotations in the report."""
    votes = case.get("votes", [])
    failures = []
    if len(votes) != 3 or {vote.get("reviewer_id") for vote in votes} != REVIEWERS:
        failures.append("three_distinct_expected_reviewers_missing")
    if any(vote.get("verdict") != "approve" for vote in votes):
        failures.append("not_unanimously_approved")
    request_ids = [vote.get("request_fingerprint") for vote in votes]
    if len(set(request_ids)) != 3 or any(not value for value in request_ids):
        failures.append("independent_request_fingerprints_missing")
    if any(not valid_evidence(vote.get("evidence"), blocks) for vote in votes):
        failures.append("current_source_vote_evidence_invalid")
    if any(vote.get("source_fingerprint") != source_hash for vote in votes):
        failures.append("vote_source_fingerprint_mismatch")
    if registry_hash is not None and any(vote.get("series_registry_fingerprint") != registry_hash for vote in votes):
        failures.append("vote_registry_fingerprint_mismatch")
    return failures


def audit_bundle(folder, document, report):
    bundle = folder / "bundle"
    if not bundle.exists():
        return {"present": False}
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        finding(report, "bundle_manifest_missing")
        return {"present": True, "files_checked": 0}
    try:
        result = validate_native_bundle(load_bundle(bundle), "", document)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # Fixed diagnostics only: source text and metadata values stay private.
        finding(report, "bundle_native_validation_failed", exception=type(exc).__name__)
        return {"present": True, "files_checked": 0}
    return {"present": True, **result}


def audit_script_review(document, report):
    review = document["provenance"].get("attribution_review")
    if review is None:
        return {"present": False}
    blocks = {row["id"]: row["text"] for row in document["source"]["blocks"]}
    if (review.get("source_fingerprint") != fingerprint(document["source"])
            or review.get("cast_fingerprint") != fingerprint(document["cast"])
            or review.get("output_script_fingerprint") != fingerprint(document["segments"])):
        finding(report, "attribution_review_fingerprint_mismatch")
    applied = 0
    reviewers = {f"attribution_reviewer_{n}" for n in range(1,4)}
    for change in review.get("changes", []):
        if change.get("status") != "applied":
            continue
        applied += 1
        votes = change.get("votes", [])
        if (len(votes) != 3 or {row.get("reviewer_id") for row in votes} != reviewers
                or any(row.get("verdict") != "approve" for row in votes)
                or len({row.get("request_fingerprint") for row in votes}) != 3
                or any(not valid_evidence(row.get("evidence"), blocks) for row in votes)
                or any(row.get("proposed_segments_fingerprint") != fingerprint(change["proposed_segments"]) for row in votes)):
            finding(report, "attribution_correction_consensus_failed", block_id=change.get("block_id"))
        actual = [row for row in document["segments"] if row["block_id"] == change.get("block_id")]
        if actual != change.get("applied_segments"):
            finding(report, "attribution_correction_differs_from_script", block_id=change.get("block_id"))
    if applied != review.get("applied_blocks"):
        finding(report, "attribution_review_count_mismatch")
    return {"present": True, "proposed_blocks":review.get("proposed_blocks"),
            "applied_blocks":applied,"unresolved_blocks":review.get("unresolved_blocks")}


def audit_book(root, number):
    folder = root / f"book-{number:02}"
    report = {"book": number, "analysis": str(folder / "analysis.json"), "errors": [],
              "review_candidates": [], "reused_characters": [], "narrators": [],
              "passed_mechanical_checks": False}
    document = read_json(folder / "analysis.json")
    registry = read_json(folder / "registry-before.json")
    extracted = source_from(folder / "source.json")
    contract_errors = validate_document(document)
    if contract_errors:
        # Validation diagnostics can include exact source spellings. Keep only
        # field locations here so this report never becomes a story-text dump.
        finding(report, "analysis_contract_invalid", count=len(contract_errors),
                locations=[message.split(":", 1)[0] for message in contract_errors])
        return report
    try:
        validate_registry(registry)
    except (ValueError, KeyError, TypeError):
        finding(report, "prior_registry_invalid")
        return report
    if len(registry["books"]) != number - 1:
        finding(report, "prior_registry_book_count_mismatch", expected=number - 1, actual=len(registry["books"]))
    source, cast, provenance = document["source"], document["cast"], document["provenance"]
    series, jury = provenance.get("series", {}), provenance.get("identity_consensus", {})
    source_hash, registry_hash = fingerprint(source), fingerprint(registry)
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    report.update(title=source["book"]["title"], source_fingerprint=source_hash,
                  series_id=series.get("series_id"), characters=len(cast["characters"]),
                  voices=len(cast["voices"]), segments=len(document["segments"]),
                  source_characters=sum(map(len, blocks.values())), readiness=readiness(document))
    if source != extracted:
        finding(report, "analysis_source_differs_from_saved_extraction")
    if provenance.get("source_fingerprint") != source_hash or series.get("source_fingerprint") != source_hash:
        finding(report, "analysis_source_fingerprint_mismatch")
    if series.get("series_id") != registry["series_id"]:
        finding(report, "series_id_mismatch")
    reconstructed = defaultdict(str)
    for segment in document["segments"]:
        reconstructed[segment["block_id"]] += segment["text"]
    if dict(reconstructed) != blocks:
        finding(report, "source_text_not_exactly_preserved")
    report["script_characters"] = sum(len(segment["text"]) for segment in document["segments"])
    if (jury.get("policy") != "unanimous_3" or jury.get("source_fingerprint") != source_hash
            or jury.get("series_registry_fingerprint") != registry_hash):
        finding(report, "identity_consensus_provenance_missing_or_mismatched")

    prior, prior_voices, owners, character_voices = {}, {}, {}, {}
    for book in registry["books"]:
        for voice in book["cast"]["voices"]:
            prior_voices.setdefault(voice["id"], voice)
        for character in book["cast"]["characters"]:
            if character["id"] not in prior:
                prior[character["id"]] = {**character, "aliases": list(character["aliases"])}
            else:
                earlier = prior[character["id"]]
                earlier["aliases"] = list(set(earlier["aliases"] + character["aliases"] + [character["name"]]))
            owners[character["voice_id"]] = character["id"]
            character_voices[character["id"]] = character["voice_id"]
    voices = {voice["id"]: voice for voice in cast["voices"]}
    current = {character["id"]: character for character in cast["characters"]}
    matches = series.get("identity_matches", [])
    match_groups = defaultdict(list)
    for match in matches:
        match_groups[match.get("character_id")].append(match)
    for identity, entries in match_groups.items():
        if len(entries) != 1 or identity not in current or identity not in prior:
            finding(report, "identity_match_mapping_ambiguous", character_id=identity, entries=len(entries))
    for character in cast["characters"]:
        identity, voice_id = character["id"], character["voice_id"]
        if voice_id in owners and owners[voice_id] != identity:
            finding(report, "voice_id_has_distinct_character_owners", character_id=identity, voice_id=voice_id)
        owners[voice_id] = identity
        if identity in character_voices and character_voices[identity] != voice_id:
            finding(report, "recurring_character_changed_voice_id", character_id=identity)
        origin = character.get("series_origin", {})
        if origin.get("series_id") != registry["series_id"] or origin.get("source_fingerprint") != source_hash:
            finding(report, "character_series_origin_mismatch", character_id=identity)
        if identity in prior:
            row = {"character_id": identity, "name": character["name"], "voice_id": voice_id}
            report["reused_characters"].append(row)
            if origin.get("status") != "reused" or len(match_groups.get(identity, [])) != 1:
                finding(report, "recurring_character_missing_unique_match", character_id=identity)
                continue
            match = match_groups[identity][0]
            local_id = match.get("local_character_id")
            row["local_character_id"] = local_id
            if series.get("character_id_map", {}).get(local_id) != identity or origin.get("local_character_id") != local_id:
                finding(report, "local_global_character_mapping_mismatch", character_id=identity)
            if match.get("basis") == "identical_character_record_in_identical_source":
                row["consensus_trace"] = "same_source_idempotent_reuse"
                if not any(book["source_fingerprint"] == source_hash for book in registry["books"]):
                    finding(report, "same_source_reuse_without_prior_source", character_id=identity)
                continue
            cases = [case for case in jury.get("series_cases", [])
                     if case.get("character_ids") == [local_id] and case.get("prior_character_id") == identity]
            if len(cases) != 1:
                finding(report, "series_reviewer_mapping_ambiguous", character_id=identity, matching_cases=len(cases))
            else:
                case = cases[0]
                failures = audit_votes(case, blocks, source_hash, registry_hash)
                if case.get("status") != "confirmed":
                    failures.append("series_case_not_confirmed")
                if failures:
                    finding(report, "reused_identity_consensus_failed", character_id=identity,
                            case_id=case.get("id"), failures=failures)
                row["consensus_trace"] = {"case_id": case.get("id"), "valid": not failures,
                                          "reviewers": [vote.get("reviewer_id") for vote in case.get("votes", [])]}
        else:
            if origin.get("status") != "new":
                finding(report, "new_character_has_reused_status", character_id=identity)
            left = names(character)
            for old_id, old in prior.items():
                right = names(old)
                exact = sorted(left & right)
                similar = [(a, b) for a in sorted(left) for b in sorted(right)
                           if a != b and min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= .88]
                if exact or similar:
                    report["review_candidates"].append({"kind": "new_id_name_overlap", "character_id": identity,
                        "name": character["name"], "prior_character_id": old_id, "prior_name": old["name"],
                        "exact_names": exact, "similar_names": similar,
                        "interpretation": "Investigate context; resemblance is not identity proof."})
    for voice_id, voice in voices.items():
        if voice_id in prior_voices and profile(voice) != profile(prior_voices[voice_id]):
            finding(report, "established_voice_profile_changed", voice_id=voice_id)

    # Narration roles are scoped within a book, not recurring role identifiers.
    for role in cast["roles"]:
        if role["kind"] != "narration":
            continue
        identity, expected = role["character_id"], set()
        scope = "independent_narration"
        if identity in prior:
            scope = "identified_recurring_character"
            expected = {old["voice_id"] for book in registry["books"] for old in book["cast"]["roles"]
                        if old["kind"] == "narration" and old["character_id"] == identity}
        elif role["id"] == "narrator_main" and identity is None:
            scope = "unidentified_main_frame"
            expected = {old["voice_id"] for book in registry["books"] for old in book["cast"]["roles"]
                        if old["kind"] == "narration" and old["id"] == "narrator_main" and old["character_id"] is None}
        concerns = [compact_issue(issue) for issue in document["issues"]
                    if issue["kind"] == "series_narrator" and issue["target"] == role["id"] and not issue["resolved"]]
        report["narrators"].append({"local_role_id": role["id"], "character_id": identity,
                                  "voice_id": role["voice_id"], "scope": scope,
                                  "prior_voice_candidates": sorted(expected), "review_issues": concerns})
        if len(expected) == 1 and role["voice_id"] not in expected and not concerns:
            finding(report, "narration_voice_changed_without_review", role_id=role["id"], voice_id=role["voice_id"])

    report["unresolved_identity_candidates"] = series.get("unresolved_identity_candidates", [])
    report["identity_decisions"] = []
    for collection in ("cases", "series_cases"):
        for case in jury.get(collection, []):
            row = {key: case.get(key) for key in ("id", "kind", "character_ids", "prior_character_id", "status")}
            row["verdicts"] = [vote.get("verdict") for vote in case.get("votes", [])]
            report["identity_decisions"].append(row)
            if case.get("status") not in {"applied", "confirmed", "confirmed_separate"}:
                report["review_candidates"].append(dict(row, review_kind="unresolved_identity_decision"))
            elif collection == "cases":
                failures = audit_votes(case, blocks, source_hash)
                if failures:
                    finding(report, "accepted_identity_case_consensus_failed", case_id=case.get("id"), failures=failures)
    reused = set(current) & set(prior)
    expected_issues = _inherited_review_issues(registry, cast, reused)
    inherited = [issue for issue in document["issues"]
                 if issue.get("series_origin", {}).get("kind") == "inherited_series_review"]
    actual_keys = {issue_key(issue) for issue in inherited}
    for issue in expected_issues:
        if issue_key(issue) not in actual_keys:
            finding(report, "prior_review_issue_not_carried", origin=list(issue_key(issue)))
    report["inherited_issues"] = [compact_issue(issue) for issue in inherited]
    report["unresolved_issues"] = [compact_issue(issue) for issue in document["issues"] if not issue["resolved"]]
    report["issue_counts"] = dict(Counter(issue["kind"] for issue in document["issues"] if not issue["resolved"]))
    report["bundle"] = audit_bundle(folder, document, report)
    report["attribution_review"] = audit_script_review(document, report)
    report["passed_mechanical_checks"] = not report["errors"]
    identity_issue_kinds = {"identity", "identity_merge", "identity_consensus", "series_identity", "series_narrator"}
    report["needs_identity_review"] = bool(report["review_candidates"] or report["unresolved_identity_candidates"]
                                          or any(issue["kind"] in identity_issue_kinds for issue in report["unresolved_issues"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing legacy seven-book run directory")
    parser.add_argument("--book", type=int, choices=range(1, 8))
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    protected = {root / name for name in ("series-cast.json", "inventory.json", "run_book.py", "audit_continuity.py")}
    for number in range(1, 8):
        folder = root / f"book-{number:02}"
        protected.update(folder / name for name in
                         ("analysis.json", "source.json", "registry-before.json", "settings.local.toml", "review.json", "summary.json"))
        if args.output and (folder / "bundle") in args.output.resolve().parents:
            parser.error("A report must not replace exported bundle files")
    if args.output and args.output.resolve() in protected:
        parser.error("A report must not replace run inputs or outputs")
    if args.output and args.output.exists() and not args.overwrite:
        parser.error("Report exists; choose a new output or explicitly use --overwrite")
    if args.output and args.output.exists() and args.overwrite:
        try:
            previous = read_json(args.output)
        except (OSError, ValueError):
            parser.error("--overwrite only replaces an existing continuity audit report")
        if not isinstance(previous, dict) or previous.get("format") != "soulship-continuity-audit":
            parser.error("--overwrite only replaces an existing continuity audit report")
    numbers = [args.book] if args.book else [number for number in range(1, 8)
                                           if (root / f"book-{number:02}" / "analysis.json").is_file()]
    books = []
    for number in numbers:
        try:
            result = audit_book(root, number)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            # No raw exception messages: they may contain story text or data.
            result = {"book": number, "errors": [{"code": "audit_input_or_schema_failure", "exception": type(exc).__name__}],
                      "review_candidates": [], "passed_mechanical_checks": False}
        books.append(result)
        print(f"Book {number:02}: {len(result['errors'])} errors; "
              f"{len(result.get('reused_characters', []))} recurring characters; "
              f"{len(result['review_candidates'])} identity review candidates; "
              f"{len(result.get('unresolved_identity_candidates', []))} unresolved series matches")
    report = {"format": "soulship-continuity-audit", "version": 1,
              "generated_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
              "books": books, "completed_books_audited": len(books),
              "errors": sum(len(book["errors"]) for book in books),
              "limitation": "Mechanical checks and candidate screening do not establish semantic identity accuracy. No registry or analysis files were changed."}
    if args.output:
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Report: {args.output.resolve()}")
    elif not books:
        print("No completed analyses found.")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
