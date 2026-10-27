#!/usr/bin/env python3
"""Optional live accent-casting comparison on small original-fiction fixtures.

Uses the configured analyzer model, never TTS or a real book directory. Reports
validated proposals and human-review criteria, not an invented acoustic score.
Rerun the same command to resume cached requests after an interruption. Use
--check-fixtures for an offline contract check without loading configuration.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epub_tts_narrator.config import load_config
from epub_tts_narrator.pipeline import (CachedRequests, DEFAULT_POLICY, check_cast,
    fingerprint, make_units, merge_patch, seed_cast, write_json)
from epub_tts_narrator.prompts import DISCOVER
from epub_tts_narrator.provider import OpenAICompatibleClient
from epub_tts_narrator.voice_ensemble import design_ensemble
from epub_tts_narrator.voice_ensemble_prompts import ENSEMBLE_SYSTEM
from epub_tts_narrator.voice_profile_refresh import refresh_voice_profiles
from epub_tts_narrator.voice_profile_refresh_prompts import SYSTEM as PROFILE_SYSTEM
from epub_tts_narrator.voice_prompt_refresh import refresh_voice_descriptions
from epub_tts_narrator.voice_refresh_prompts import SYSTEM as DESCRIPTION_SYSTEM


CASES = {
    "japanese_immigrant": {
        "name": "Himari",
        "text": "Himari grew up in Japan, moved to Ohio at twelve, and speaks fluent English. She still speaks Japanese with her family.",
        "review": "Consider a restrained Japanese influence as casting; do not infer General American from fluency or require a literal accent label. No nationality-based text trait.",
    },
    "mexican_immigrant": {
        "name": "Andre",
        "text": "Andre grew up in Mexico and moved to Ohio at twelve. He speaks fluent English at school and Spanish with his family.",
        "review": "Consider a restrained Mexican Spanish influence as casting; preserve fluent intelligible speech without a caricature. No nationality-based text trait.",
    },
    "same_name_local": {
        "name": "Himari",
        "text": "Himari was born and raised in Ohio. Her family has spoken only English at home for generations, and she has never lived abroad.",
        "review": "Use individual upbringing, not a Japanese accent assigned solely from the name. A suitable local American accent is valid casting.",
    },
    "explicit_american": {
        "name": "Himari",
        "text": "Himari moved from Japan as a teenager. She speaks with a General American accent, without a trace of her former accent.",
        "source_accent": "General American accent",
        "review": "Preserve the explicit General American accent despite the migration background; preserve its exact text trait in ensemble output.",
    },
    "inherited_american": {
        "name": "Andre",
        "text": "Andre moved from Mexico as a teenager and speaks both English and Spanish fluently.",
        "inherited": True,
        "review": "Preserve the established General American series voice. A new book's background is not authorization to recast it.",
    },
    "user_scottish": {
        "name": "Himari",
        "text": "Himari moved from Japan as a teenager and speaks fluent English.",
        "current_prompt": "An adult woman with a light mid-range tone, precise consonants, and a deliberately chosen soft Edinburgh accent.",
        "review": "For description refresh, preserve the explicit current-prompt Edinburgh choice rather than the automatic analyzed General American profile or a background-based substitute.",
    },
    "name_only": {
        "name": "Himari",
        "text": "Himari placed the lantern on the table and greeted the travelers.",
        "review": "A name may inform tentative creative casting, but never establishes origin, upbringing, or an audible accent as a source fact.",
    },
}
STAGES = ("discovery", "ensemble", "profile", "prompt")


def fixture(case_id):
    case = CASES[case_id]
    source = {
        "book": {"id": "accent-eval-" + case_id, "title": "Synthetic Accent Evaluation", "authors": ["Fixture"], "language": "en"},
        "chapters": [{"id": "ch1", "title": "One", "source_href": "chapter.xhtml", "block_ids": ["b1"]}],
        "blocks": [{"id": "b1", "chapter_id": "ch1", "text": case["text"],
                    "source": {"href": "chapter.xhtml", "element_id": "p1", "tag": "p"}}],
        "exclusions": [], "warnings": [], "extraction_policy": "Original synthetic fixture; no EPUB or book files read.",
    }
    cast = seed_cast()
    accent = {"name": "accent", "value": "General American accent", "basis": "casting", "evidence": []}
    if "source_accent" in case:
        accent.update(basis="text", evidence=[{"block_id": "b1", "quote": case["source_accent"]}])
    profile = {
        "id": "v1", "description": "An adult performer with a light mid-range tone, precise consonants, and a General American accent.",
        "speechTraits": "General American accent; precise consonants", "traits": [accent],
    }
    cast["voices"].append(profile)
    cast["characters"].append({"id": "c1", "name": case["name"], "aliases": [], "description": case["text"],
                               "evidence": [{"block_id": "b1", "quote": case["text"]}], "voice_id": "v1"})
    cast["roles"].append({"id": "r1", "kind": "dialogue", "character_id": "c1", "voice_id": "v1", "description": "Character dialogue."})
    result = {"source": source, "cast": cast, "manifest": {"source_fingerprint": fingerprint(source)}}
    if case.get("inherited"):
        profile["series_origin"] = {"kind": "series_casting_reuse", "series_id": "prior-series", "voice_id": "v1"}
        result["series_context"] = {"voices": [deepcopy(profile)]}
    return result


def run_case(case_id, stage, requests):
    data = fixture(case_id)
    if stage == "profile":
        return refresh_voice_profiles(data, requests, voice_ids=["v1"], workers=1)
    if stage == "prompt":
        current = {"v1": CASES[case_id]["current_prompt"]} if "current_prompt" in CASES[case_id] else None
        return refresh_voice_descriptions(data, requests, voice_ids=["v1"], current_prompts=current, workers=1)
    if stage == "ensemble":
        cast, audit = design_ensemble(data["source"], data["cast"], requests, policy=DEFAULT_POLICY)
        return {"cast": cast, "audit": audit}
    if stage != "discovery":
        raise ValueError("Unknown accent evaluation stage")
    cast = seed_cast()
    available = {"characters": ["c1", "c2"], "roles": ["r1", "r2"], "voices": ["v1", "v2"]}
    payload = {"mode": "discover", "book": data["source"]["book"], "known_cast": cast,
               "available_ids": available, "performance_policy": DEFAULT_POLICY,
               "target_units": [unit for batch in make_units(data["source"]) for unit in batch],
               "context_units": [], "scene_summaries": []}
    return requests.run(DISCOVER, payload, lambda response: merge_patch(data["source"], cast, response, available))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Required for live evaluation; global TOML configuration")
    parser.add_argument("-o", "--output", type=Path, help="JSON report; requests are checkpointed in a sibling cache directory")
    parser.add_argument("--case", choices=tuple(CASES), action="append", dest="cases")
    parser.add_argument("--stage", choices=STAGES, action="append", dest="stages", help="Repeat to compare stages; defaults to profile")
    parser.add_argument("--check-fixtures", action="store_true", help="Offline validation only; no provider or output files")
    args = parser.parse_args(argv)
    cases, stages = list(dict.fromkeys(args.cases or CASES)), list(dict.fromkeys(args.stages or ["profile"]))
    for case_id in cases:
        data = fixture(case_id)
        check_cast(data["source"], data["cast"])
    if args.check_fixtures:
        print(f"Validated {len(cases)} synthetic fixtures; no model requests.")
        return 0
    if args.config is None or args.output is None:
        parser.error("Live evaluation requires --config and --output")
    # User-correction and inheritance fixtures test a refresh, not rediscovery.
    jobs = [(case_id, stage) for case_id in cases for stage in stages
            if not (case_id == "user_scottish" and stage != "prompt")
            and not (case_id == "inherited_american" and stage == "discovery")]
    if not jobs:
        parser.error("No selected case applies to the selected stage")
    output = args.output.absolute()
    cache = output.parent / (output.stem + "-cache")
    if output.is_symlink() or cache.is_symlink():
        raise ValueError("Evaluation output and cache must not be symlinks")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    client = OpenAICompatibleClient(**config["provider"])
    settings = config["analysis"]
    requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=print)
    systems = {"discovery": DISCOVER, "ensemble": ENSEMBLE_SYSTEM,
               "profile": PROFILE_SYSTEM, "prompt": DESCRIPTION_SYSTEM}
    run_identity = fingerprint({"provider": client.identity, "jobs": jobs,
                                "fixtures": {case_id: CASES[case_id] for case_id in cases},
                                "systems": {stage: systems[stage] for stage in stages}})
    report = {"format": "voice-accent-evaluation", "version": 1, "provider": client.identity,
              "run_identity": run_identity,
              "status": "running", "assessment": "manual_review_required", "results": []}
    if output.exists():
        import json
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("format") != report["format"] or previous.get("run_identity") != run_identity:
            raise ValueError("Output belongs to a different evaluation; choose another path")
    write_json(output, report, overwrite=True)
    for case_id, stage in jobs:
        print(f"Evaluating {case_id}: {stage}")
        result = run_case(case_id, stage, requests)
        report["results"].append({"case": case_id, "stage": stage,
                                  "input": fixture(case_id), "review": CASES[case_id]["review"], "result": result})
        write_json(output, report, overwrite=True)
    report["status"] = "complete"
    write_json(output, report, overwrite=True)
    print(f"Saved {len(jobs)} proposals for manual review to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
