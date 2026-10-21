#!/usr/bin/env python3
"""Run small, original-fiction identity regression cases against a configured model.

This is a live model check, not a measure of accuracy across real novels. It does
not discover a cast or attribute speech: it deliberately supplies known cast
mistakes to the identity audit, then checks the resulting identity structure.
All credentials and provider settings come from the required TOML file.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epub_tts_narrator.config import load_config
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.pipeline import (
    CachedRequests, base_document, check_cast, fingerprint, make_units, segment,
    write_json,
)
from epub_tts_narrator.provider import OpenAICompatibleClient, ProviderError


FIXTURE_ROOT = PROJECT_ROOT / "examples" / "identity-consensus"


def normalized(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def covered_by_distinct_characters(groups, characters):
    """Match expected identifying terms to separate records, without fixed IDs."""
    descriptions = [normalized(" ".join([
        item["name"], *item["aliases"], item["description"],
    ])) for item in characters]
    choices = [[index for index, text in enumerate(descriptions)
                if all(normalized(term) in text for term in group)]
               for group in groups]

    def assign(index, used):
        return index == len(choices) or any(
            candidate not in used and assign(index + 1, used | {candidate})
            for candidate in choices[index]
        )

    return assign(0, set())


def audit_document(source, cast, issues, provenance):
    """Use narrative-only fixtures to compute contract/readiness checks honestly."""
    narrator = next(role for role in cast["roles"] if role["id"] == "narrator_main")
    segments = [segment(source, block["id"], 0, block["text"], narrator,
                        reason="Synthetic fixture contains narrative prose only.")
                for block in source["blocks"]]
    normalized_issues = []
    for issue in issues:
        normalized_issue = deepcopy(issue)
        normalized_issue.setdefault("id", "identity_case_" + fingerprint(issue)[:16])
        normalized_issue.setdefault("resolved", False)
        normalized_issues.append(normalized_issue)
    return base_document(source, cast, segments, normalized_issues,
                         {"identity_audit": provenance})


def check_expectations(fixture, cast, state, provenance):
    expected = fixture["expected"]
    characters = cast["characters"]
    names = {normalized(name) for character in characters
             for name in [character["name"], *character["aliases"]]}
    checks = [{"check": "character_count", "passed": len(characters) == expected["character_count"],
               "expected": expected["character_count"], "actual": len(characters)}]
    if expected.get("identity_groups"):
        checks.append({"check": "distinct_identity_descriptions",
                       "passed": covered_by_distinct_characters(expected["identity_groups"], characters),
                       "expected": expected["identity_groups"]})
    if expected.get("names_or_aliases"):
        missing = [name for name in expected["names_or_aliases"] if normalized(name) not in names]
        checks.append({"check": "known_names_and_aliases", "passed": not missing, "missing": missing})
    if expected.get("preserve_character_ids"):
        actual_ids = {character["id"] for character in characters}
        checks.append({"check": "stable_character_ids",
                       "passed": actual_ids == set(expected["preserve_character_ids"]),
                       "expected": expected["preserve_character_ids"], "actual": sorted(actual_ids)})
    if expected.get("preserve_voice_ids"):
        actual_voices = {character["id"]: character["voice_id"] for character in characters}
        checks.append({"check": "stable_character_voices",
                       "passed": all(actual_voices.get(key) == value
                                     for key, value in expected["preserve_voice_ids"].items()),
                       "expected": expected["preserve_voice_ids"], "actual": actual_voices})
    cases = provenance.get("cases", [])
    if expected.get("decision"):
        kind, status = expected["decision"]["kind"], expected["decision"]["status"]
        checks.append({"check": "expected_audit_decision",
                       "passed": any(case.get("kind") == kind and case.get("status") == status for case in cases),
                       "expected": expected["decision"]})
    accepted = [case for case in cases if case.get("status") in {"applied", "confirmed_separate"}]
    checks.append({"check": "accepted_decisions_have_three_approvals",
                   "passed": all(len(case.get("votes", [])) == 3
                                 and len({vote.get("reviewer_id") for vote in case["votes"]}) == 3
                                 and all(vote.get("verdict") == "approve" for vote in case["votes"])
                                 for case in accepted),
                   "accepted_decision_count": len(accepted)})
    checks.extend([
        {"check": "valid_contract", "passed": not state["errors"], "errors": state["errors"]},
        {"check": "ready_without_unresolved_identity_issues", "passed": state["ready"] is expected["ready"],
         "expected": expected["ready"], "actual": state["ready"], "review_count": state["review_count"]},
    ])
    return checks


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="TOML provider configuration; no environment variables")
    parser.add_argument("-o", "--output", required=True, type=Path, help="New JSON report path (never overwritten)")
    parser.add_argument("--case", action="append", dest="cases", metavar="CASE_ID", help="Run only this case; repeat to select several")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory beneath the report's parent; defaults to <report-stem>-cache")
    parser.add_argument("--check-fixtures", action="store_true", help="Validate fixture contracts without any API requests")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("The report output already exists; choose a new path.")
    cache = (args.cache_dir or output.parent / (output.stem + "-cache")).resolve()
    if not cache.is_relative_to(output.parent) or cache == output.parent:
        raise ValueError("The cache directory must be beneath the report output's parent directory.")
    fixtures = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(FIXTURE_ROOT.glob("*.json"))]
    selected = set(args.cases or [fixture["id"] for fixture in fixtures])
    unknown = selected - {fixture["id"] for fixture in fixtures}
    if unknown:
        raise ValueError("Unknown case ID: " + ", ".join(sorted(unknown)))
    fixtures = [fixture for fixture in fixtures if fixture["id"] in selected]
    if not fixtures:
        raise ValueError("No identity fixtures were selected.")
    for fixture in fixtures:
        check_cast(fixture["source"], fixture["cast"])
    config = load_config(args.config)
    workers = min(5, config["analysis"]["attribution_workers"])
    report = {
        "format": "epub-audiobook-identity-regression", "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Original synthetic fiction; identity audit only, not book discovery or speaker attribution.",
        "limitation": "Small regression suite; passing does not establish an error rate or guarantee correct identities in other books.",
        "fixture_check_only": args.check_fixtures, "max_concurrent_requests": workers, "cases": [],
    }
    client = None
    if not args.check_fixtures:
        from epub_tts_narrator.identity import audit_identities
        client = OpenAICompatibleClient(**config["provider"])
        report["provider"] = client.identity
    for fixture in fixtures:
        case_id = fixture["id"]
        print(f"Checking identity case: {case_id}", file=sys.stderr, flush=True)
        result = {"id": case_id, "description": fixture["description"],
                  "source_fingerprint": fingerprint(fixture["source"]),
                  "initial_character_count": len(fixture["cast"]["characters"]),
                  "expected": fixture["expected"]}
        if args.check_fixtures:
            result.update(passed=True, checks=[{"check": "fixture_contract", "passed": True}])
        else:
            requests = CachedRequests(client, cache / case_id,
                                      repair_attempts=config["analysis"]["repair_attempts"],
                                      max_request_chars=config["analysis"]["max_request_chars"],
                                      progress=lambda text: print(text, file=sys.stderr, flush=True))
            try:
                cast, issues, provenance = audit_identities(
                    deepcopy(fixture["source"]), deepcopy(fixture["cast"]), requests,
                    make_units(fixture["source"], config["analysis"]["chunk_chars"]),
                    whole_book=True, workers=workers,
                    progress=lambda text: print(text, file=sys.stderr, flush=True),
                )
                document = audit_document(fixture["source"], cast, issues, provenance)
                state = readiness(document)
                checks = check_expectations(fixture, cast, state, provenance)
                result.update(passed=all(check["passed"] for check in checks), checks=checks,
                              cast=cast, issues=document["issues"], readiness=state, provenance=provenance)
            except Exception as exc:
                # Exception text can contain provider responses or config values.
                # Keep it out of both logs and the shareable report.
                result.update(passed=False, error_type=type(exc).__name__,
                              checks=[{"check": "audit_completed", "passed": False}])
                if isinstance(exc, ProviderError):
                    # Our provider exception messages are fixed, sanitized text.
                    # Stop instead of repeating failed authentication/outage calls.
                    result["provider_error"] = str(exc)
                    report["aborted"] = True
                    print(str(exc), file=sys.stderr, flush=True)
        report["cases"].append(result)
        print(f"{case_id}: {'PASS' if result['passed'] else 'FAIL'}", file=sys.stderr, flush=True)
        if report.get("aborted"):
            break
    report["passed"] = all(result["passed"] for result in report["cases"])
    report["passed_count"] = sum(result["passed"] for result in report["cases"])
    report["case_count"] = len(report["cases"])
    write_json(output, report)
    print(f"{report['passed_count']}/{report['case_count']} cases passed. Report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Identity validation setup failed ({type(exc).__name__}); check paths, case IDs, and configuration.", file=sys.stderr)
        raise SystemExit(2)
