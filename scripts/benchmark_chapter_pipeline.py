"""Benchmark the actual chapter pipeline: freeze equal work, then run it.

``prepare BUNDLE --config CFG --chapters LIST --cases 160 --output DIR`` uses
the real render plan to freeze 160 unique source clips across four specified
source chapters into four balanced benchmark-only chapter groups with portable
FLAC references, a synthetic benchmark identity, and a checksum-bound input
hash. The book is never modified and the corpus stays private (0600).

``run FROZEN --config CFG --scheduler legacy|pipeline --output DIR`` renders
the frozen groups through the production chapter service and client (real
HTTP, sessions, ACK, verified downloads, final M4B assembly) in a fresh
isolated cache/output root. Resumed or cached runs are labeled and excluded
from speed gates; no durable execution grants exist. This tool never performs
paid compute on its own: live runs are the parent plan's explicitly approved
decision. --fake-engine swaps only the two model factories for offline QA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.benchmark_rendering import GpuSampler
from scripts.chapter_benchmark_corpus import DEFAULT_CASES, GROUP_COUNT, load_corpus
from scripts.chapter_benchmark_report import (
    finalize, init_output, new_report, save_manifest, save_report, set_group_state,
)
from scripts.chapter_benchmark_service import (
    ChapterService, Delivery, build_selection, cached_record, connect, group_folder,
)


def _parse_chapters(value):
    try:
        chapters = [int(item) for item in value.split(",")]
    except ValueError:
        raise ValueError("Chapters must be a comma-separated list of source chapter numbers") from None
    if len(chapters) != GROUP_COUNT or len(set(chapters)) != GROUP_COUNT:
        raise ValueError(f"Specify exactly {GROUP_COUNT} distinct source chapters, e.g. 1,2,3,4")
    return chapters


def _require_case_count(cases, sample):
    if cases % GROUP_COUNT:
        raise ValueError(f"Cases must be a multiple of {GROUP_COUNT} so the four groups stay balanced")
    if cases != DEFAULT_CASES and sample is None:
        raise ValueError(f"Only the default {DEFAULT_CASES}-case corpus may prepare without a name; "
                         "pass --sample NAME to record an explicit exploratory sample")


def cmd_prepare(args):
    from scripts.chapter_benchmark_corpus import freeze_bundle
    _require_case_count(args.cases, args.sample)
    chapters = _parse_chapters(args.chapters)
    path, body = freeze_bundle(args.book, args.config, chapters, args.cases, args.output,
                               seed=args.seed, sample=args.sample)
    print(f"Prepared {len(body['cases'])} unique cases in {GROUP_COUNT} balanced groups "
          f"({body['total_source_words']} canonical source words, "
          f"{len(body['warmup'])} warmup clips, input hash {body['input_hash'][:16]}...): {path}")


def cmd_run(args):
    from epub_tts_narrator.renderer.config import load_config
    from epub_tts_narrator.renderer.diagnostics import configured_secrets, safe_error_text
    from scripts.chapter_benchmark_assembly import assemble_book

    corpus, voices = load_corpus(args.frozen)
    config = load_config(args.config, require_verifier=not args.fake_engine,
                         require_engine=not args.fake_engine)
    if config["remote_chapters"]["url"]:
        raise ValueError("The benchmark run serves its own loopback chapter service; "
                         "clear remote_chapters.url in the configuration")
    config["render_pool"]["scheduler"] = args.scheduler
    output = Path(args.output).resolve()
    manifest, resumed = init_output(output, corpus, args.scheduler, fake_engine=args.fake_engine)
    config["render"].update(work_dir=str(output / "work"), output_dir=str(output / "book"),
                            chapter_format="m4a")
    config["chapter_server"]["data_dir"] = str(output / "server-data")
    secrets = configured_secrets(config)
    report = new_report(corpus=corpus, corpus_path=args.frozen, config=config, capabilities={},
                        scheduler=args.scheduler, output=output, resumed=resumed,
                        warm_words=sum(entry["words"] for entry in corpus["warmup"]),
                        fake_engine=args.fake_engine)
    save_manifest(output, manifest)
    save_report(output, report)
    service = ChapterService(config, fake_engine=args.fake_engine)
    delivery = None
    sampler = GpuSampler()

    def save():
        save_manifest(output, manifest)
        save_report(output, report)

    def pending_downloads():
        """Submit pending groups through production routes; reuse validated caches."""
        submitted = {}
        began = time.perf_counter()
        for group in range(GROUP_COUNT):
            if manifest["groups"][group]["state"] == "complete":
                continue
            selection = build_selection(corpus, group, bindings)
            folder = group_folder(output / "work", selection)
            if cached_record(folder, selection) is not None:
                set_group_state(manifest, group, "complete")
                save()
                continue
            _selection, group_folder_path, record = delivery.submit(corpus, group, bindings)
            submitted[record["id"]] = (group, group_folder_path, record)
        try:
            if submitted:
                delivery.wait({job_id: (folder, record)
                               for job_id, (_group, folder, record) in submitted.items()})
            for job_id, (group, _folder, _record) in submitted.items():
                set_group_state(manifest, group, "complete")
            save()
        except BaseException:
            # Durable validated checkpoints decide completion, never the error path.
            for job_id, (group, folder, record) in submitted.items():
                if cached_record(folder, record["selection"]) is not None:
                    set_group_state(manifest, group, "complete")
            save()
            raise
        return time.perf_counter() - began

    try:
        began = time.perf_counter()
        client, capabilities = connect(config, service.start())
        report.update(renderer_identity=capabilities.get("engine"),
                      renderer_fingerprint=capabilities.get("renderer_fingerprint"),
                      parallel_chapters=capabilities.get("parallel_chapters"))
        delivery = Delivery(client, capabilities, output / "work", notify=print)
        bindings = delivery.register_voices(voices)
        report["timing"]["cold_setup_seconds"] = time.perf_counter() - began
        save_report(output, report)
        with sampler:
            began = time.perf_counter()
            warmup_selection = build_selection(corpus, None, bindings)
            warmup_folder = group_folder(output / "work", warmup_selection)
            warmup_hit = cached_record(warmup_folder, warmup_selection) is not None
            if not warmup_hit:
                _selection, folder, record = delivery.submit(corpus, None, bindings)
                delivery.wait({record["id"]: (folder, record)})
            report["timing"]["warmup_cache_hit"] = warmup_hit
            report["timing"]["warmup_seconds"] = time.perf_counter() - began
            save_report(output, report)
            report["timing"]["active_pipeline_seconds"] = pending_downloads()
            report["gpu_samples"] = list(sampler.samples)
            save_report(output, report)
            began = time.perf_counter()
            downloaded = {}
            for group in range(GROUP_COUNT):
                selection = build_selection(corpus, group, bindings)
                folder = group_folder(output / "work", selection)
                descriptor = cached_record(folder, selection)
                if descriptor is None:
                    raise ValueError(f"Benchmark group {group + 1} lacks a validated download")
                downloaded[group] = (folder, descriptor)
            assembled = assemble_book(corpus, downloaded, config, output / "book", notify=print)
            report["timing"]["assembly_seconds"] = time.perf_counter() - began
            report["book"] = {"path": assembled["path"],
                              "sha256": assembled["result"]["files_sha256"]["book.m4b"]}
            manifest["status"] = "complete"
            save()
            finalize(report, manifest, status="complete")
            save_report(output, report)
            print(f"Benchmark complete: {assembled['path']}")
            print(f"Timing seconds: {report['timing']}")
            print(f"Accepted {report['accepted_source_words']} of {report['total_source_words']} "
                  f"frozen source words; speed gate eligible: {report['speed_gate_eligible']}")
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["status"] = status
        report["gpu_samples"] = list(sampler.samples)
        finalize(report, manifest, status=status,
                 error=safe_error_text(f"{type(exc).__name__}: {exc}", secrets))
        save_manifest(output, manifest)
        save_report(output, report)
        raise
    finally:
        if delivery is not None:
            delivery.close()
        service.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="freeze an equal-work corpus from a book bundle")
    prepare.add_argument("book", type=Path)
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--chapters", required=True, help="four comma-separated source chapter numbers")
    prepare.add_argument("--cases", type=int, default=DEFAULT_CASES)
    prepare.add_argument("--sample", help="explicit exploratory sample name for non-default case counts")
    prepare.add_argument("--seed", type=int, default=1729)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="render a frozen corpus through the production chapter pipeline")
    run.add_argument("frozen", type=Path)
    run.add_argument("--config", type=Path)
    run.add_argument("--scheduler", choices=("legacy", "pipeline"), default="legacy")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--fake-engine", action="store_true",
                     help="offline QA only: fake the external engine/verifier model leaves while the "
                          "real RenderPool schedules; the report is permanently marked fake_engine "
                          "and ineligible for speed gates")
    args = parser.parse_args(argv)
    try:
        {"prepare": cmd_prepare, "run": cmd_run}[args.command](args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
