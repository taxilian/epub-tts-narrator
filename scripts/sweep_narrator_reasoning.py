#!/usr/bin/env python3
"""Sweep model x reasoning effort for the experimental narrator read on a chapter subset.

Live model calls: needs --config (credentials are read from that TOML only and
never printed). Each cell gets its own request cache, because cache keys
deliberately exclude model and reasoning effort. Results (documents, metrics,
pairwise agreement) go under --output, which contains book text: keep it in a
gitignored run directory.

  python3.11 scripts/sweep_narrator_reasoning.py --config epub-tts-narrator.toml \
      --epub BOOK.epub --chapters 3-7 --cell deepseek-v4.1-flash:low --cell glm-5.3-flash:high \
      [--continue-from runs/book-01] [--repeats 2] [--reference runs/book/analysis.json] -o runs/sweep
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_analyses import compare  # noqa: E402
from epub_tts_narrator.config import load_config  # noqa: E402
from epub_tts_narrator.epub import extract_epub  # noqa: E402
from epub_tts_narrator.narrator_read import narrated_analyze  # noqa: E402
from epub_tts_narrator.provider import OpenAICompatibleClient, ProviderError  # noqa: E402


def commented_provider(config_path, base_url):
    """Read an alternative provider block that is commented out in the TOML file.

    Looks for a commented `base_url = "BASE_URL"` line and the commented
    `api_key` within the next few lines. The key is used in memory only and
    never printed.
    """
    import tomllib
    lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        text = line.lstrip("# ").strip()
        if not text.startswith("base_url"):
            continue
        try:
            if tomllib.loads(text).get("base_url") != base_url:
                continue
        except tomllib.TOMLDecodeError:
            continue
        found = {"base_url": base_url}
        for follow in lines[index + 1:index + 6]:
            candidate = follow.lstrip("# ").strip()
            if candidate.startswith(("api_key", "model")):
                try:
                    found.update(tomllib.loads(candidate))
                except tomllib.TOMLDecodeError:
                    pass
        if "api_key" in found:
            return found
    raise ValueError("No commented provider block with that base_url and an api_key was found")


def subset(source, first, last):
    chapters = source["chapters"][first - 1:last]
    keep = {block_id for chapter in chapters for block_id in chapter["block_ids"]}
    result = deepcopy(source)
    result["chapters"] = deepcopy(chapters)
    result["blocks"] = [block for block in source["blocks"] if block["id"] in keep]
    return result


def restrict(document, source):
    """Reduce a full-book reference analysis to the subset's blocks."""
    keep = {block["id"] for block in source["blocks"]}
    result = deepcopy(document)
    result["source"] = source
    result["segments"] = [row for row in document["segments"] if row["block_id"] in keep]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epub", type=Path, required=True)
    parser.add_argument("--chapters", default="1-5", help="1-based inclusive chapter range, e.g. 3-7")
    parser.add_argument("--cell", action="append", required=True, help="MODEL:EFFORT (EFFORT may be 'default')")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--reference", type=Path, help="Full-book analysis.json to score against")
    parser.add_argument("--commented-provider", metavar="BASE_URL",
                        help="Use a provider block that is commented out in --config (base_url and api_key)")
    parser.add_argument("--cache-dir", type=Path,
                        help="Reuse one existing request cache (e.g. from an analyze --strategy narrator run); single cell only")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cache_dir is not None and (len(args.cell) != 1 or args.repeats != 1):
        parser.error("--cache-dir needs exactly one cell and one repeat: cache keys exclude model and effort")

    config = load_config(args.config)
    first, _, last = args.chapters.partition("-")
    source = extract_epub(args.epub, include_auxiliary=config["extraction"]["include_auxiliary"])
    source = subset(source, int(first), int(last or first))
    registry = None
    if args.continue_from is not None:
        from epub_tts_narrator.analysis_directory import continuation_registry
        registry = continuation_registry(args.continue_from, source)
    reference = None
    if args.reference is not None:
        path = args.reference / "analysis.json" if args.reference.is_dir() else args.reference
        reference = restrict(json.loads(path.read_text(encoding="utf-8")), source)
    args.output.mkdir(parents=True, exist_ok=True)
    results, documents = [], {}
    for cell in args.cell:
        model, _, effort = cell.partition(":")
        for repeat in range(1, args.repeats + 1):
            name = f"{model}-{effort or 'default'}-r{repeat}"
            settings = dict(config["provider"], model=model)
            if args.commented_provider:
                alternative = commented_provider(args.config, args.commented_provider)
                settings.update(base_url=alternative["base_url"], api_key=alternative["api_key"])
            settings["reasoning_effort"] = None if effort in {"", "default"} else effort
            client = OpenAICompatibleClient(**settings)
            print(f"== {name}", file=sys.stderr, flush=True)
            started = time.monotonic()
            row = {"cell": name, "model": model, "reasoning_effort": settings["reasoning_effort"], "repeat": repeat}
            try:
                document = narrated_analyze(source, client, args.cache_dir or args.output / "cache" / name,
                                            series_registry=registry,
                                            policy=config["performance_policy"], workers=args.workers,
                                            repair_attempts=config["analysis"]["repair_attempts"],
                                            max_request_chars=config["analysis"]["max_request_chars"],
                                            progress=lambda message: print(message, file=sys.stderr, flush=True))
            except (ProviderError, ValueError) as error:
                row.update(error=str(error)[:500], wall_seconds=round(time.monotonic() - started, 1))
                results.append(row)
                continue
            documents[name] = document
            (args.output / f"{name}.analysis.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            metrics = document["provenance"]["narrator_read"]["metrics"]
            row.update(wall_seconds=round(time.monotonic() - started, 1), metrics=metrics,
                       verification=document["provenance"]["narrator_read"]["verification"],
                       characters=len(document["cast"]["characters"]),
                       unassigned=sum(1 for item in document["segments"]
                                      if item["role_id"] is None and not item.get("simultaneous_speakers")))
            if reference is not None:
                row["vs_reference"] = compare(reference, document, count=0)[0]
            results.append(row)
    names = sorted(documents)
    pairwise = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            summary = compare(documents[left], documents[right], count=0)[0]
            pairwise[f"{left} vs {right}"] = {key: summary[key] for key in ("char_agreement_rate", "strict_name_agreement_rate", "unresolved")}
    report = {"chapters": args.chapters, "blocks": len(source["blocks"]),
              "characters": sum(len(block["text"]) for block in source["blocks"]),
              "cells": results, "pairwise": pairwise}
    (args.output / "sweep-results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in results:
        total = row.get("metrics", {}).get("total", {})
        print(f"{row['cell']:38s} {row.get('wall_seconds', 0):7.1f}s  calls={total.get('provider_calls')} "
              f"prompt={total.get('prompt_tokens')} completion={total.get('completion_tokens')} "
              f"reasoning={total.get('reasoning_tokens')} unassigned={row.get('unassigned')} "
              f"ref_agree={row.get('vs_reference', {}).get('char_agreement_rate')} {row.get('error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
