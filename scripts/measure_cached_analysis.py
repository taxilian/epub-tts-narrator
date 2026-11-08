#!/usr/bin/env python3
"""Measure a completed consensus analysis by replaying it from its request cache.

No network: the client refuses every live call. The real `analyze` pipeline is
replayed with the same configuration, and every request it would make is
counted by stage (payload mode) with exact request/response character sizes.
Use this to measure the existing pipeline, whose provider client did not record
token usage. A cache miss stops the replay and is reported (the run then
covered only the stages before it). Prints numbers only, never book text.

  python3.11 scripts/measure_cached_analysis.py --config epub-tts-narrator.toml \
      --epub BOOK.epub --cache runs/.epub-audiobook-cache [--continue-from runs/book-01] -o measure.json
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epub_tts_narrator import pipeline  # noqa: E402
from epub_tts_narrator.config import load_config  # noqa: E402
from epub_tts_narrator.epub import extract_epub  # noqa: E402
from epub_tts_narrator.provider import OpenAICompatibleClient  # noqa: E402


class Uncached(Exception):
    pass


class ReplayOnlyClient:
    def __init__(self, identity):
        self.identity = identity
        self.model, self.base_url = identity["model"], identity["base_url"]

    def complete(self, system, payload):
        raise Uncached(payload.get("mode") or "attribution")


def stage_of(payload):
    mode = payload.get("mode")
    if mode:
        return mode
    if "final_cast" in payload and "target_units" in payload:
        return "attribution"
    return "other:" + ",".join(sorted(payload)[:4])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epub", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = config["analysis"]
    source = extract_epub(args.epub, include_auxiliary=config["extraction"]["include_auxiliary"])
    registry = None
    if args.continue_from is not None:
        from epub_tts_narrator.analysis_directory import continuation_registry
        registry = continuation_registry(args.continue_from, source)
    else:
        from epub_tts_narrator.analysis_directory import fresh_registry
        registry = fresh_registry(args.output.parent / "measured-book")
    identity = OpenAICompatibleClient(**config["provider"]).identity
    stats = defaultdict(lambda: {"requests": 0, "system_chars": 0, "payload_chars": 0, "response_chars": 0})
    original_run = pipeline.CachedRequests.run

    lock = threading.Lock()

    def counted_run(self, system, payload, validate, *, cancelled=None, repair_protocol=None):
        size = 0
        for key in self._cache_keys(system, payload):
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                size = path.stat().st_size
                break
        with lock:
            row = stats[stage_of(payload)]
            row["requests"] += 1
            row["system_chars"] += len(system)
            row["payload_chars"] += len(json.dumps(payload, ensure_ascii=False))
            row["response_chars"] += size
        return original_run(self, system, payload, validate, cancelled=cancelled, repair_protocol=repair_protocol)

    pipeline.CachedRequests.run = counted_run
    stopped = None
    try:
        pipeline.analyze(source, ReplayOnlyClient(identity), args.cache, chunk_chars=settings["chunk_chars"],
                         repair_attempts=settings["repair_attempts"], max_request_chars=settings["max_request_chars"],
                         whole_book_discovery=settings["whole_book_discovery"],
                         attribution_workers=settings["attribution_workers"], identity_consensus=settings["identity_consensus"],
                         attribution_review=settings["attribution_review"],
                         attribution_review_chunk_chars=settings["attribution_review_chunk_chars"],
                         attribution_review_notes=settings["attribution_review_notes"],
                         cast_review_notes=settings["cast_review_notes"],
                         cast_reconciliation_rounds=settings["cast_reconciliation_rounds"],
                         cast_reconciliation_notes=settings["cast_reconciliation_notes"],
                         series_registry=registry, policy=config["performance_policy"])
    except Uncached as miss:
        stopped = f"cache miss at stage {miss.args[0]}"
    finally:
        pipeline.CachedRequests.run = original_run
    total = {key: sum(row[key] for row in stats.values()) for key in ("requests", "system_chars", "payload_chars", "response_chars")}
    report = {"source_chars": sum(len(block["text"]) for block in source["blocks"]),
              "stopped": stopped, "stages": dict(stats), "total": total,
              "note": "Counts validated requests only; failed validation attempts and retries are not included."}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
