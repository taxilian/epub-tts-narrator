"""Fixed-book first-pass TTS benchmark; never edits book voices or render caches."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
from threading import Event, Thread
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, regular_file, write_json
from epub_tts_narrator.renderer.diagnostics import configured_secrets, safe_error_text


def length_class(text):
    count = len(text.split())
    return "short" if count <= 12 else "medium" if count <= 40 else "long"


def select_cases(chapters, voices, *, seed=1729):
    """Freeze 24 heterogeneous requests, approximately matching Book 5 lengths."""
    from epub_tts_narrator.renderer.bundle import chunk_speakers
    from epub_tts_narrator.spoken_text import effective_delivery

    groups = {name: [] for name in ("short", "medium", "long")}
    for chapter in chapters:
        for chunk in chapter["chunks"]:
            if chunk["silent_control"]:
                continue
            turns = chunk.get("paragraph_turns", [chunk])
            speakers = [chunk_speakers(turn) for turn in turns]
            if any(len(s) != 1 for s in speakers):
                continue
            ids = {s[0]["voice_id"] for s in speakers}
            if len(ids) != 1:
                continue
            voice_id = next(iter(ids))
            voice = voices[voice_id]
            case = {"id": digest({"chapter": chapter["id"], "chunk": chunk["chunk_id"]})[:24],
                    "chapter_number": chapter["number"], "chunk_id": chunk["chunk_id"],
                    "paragraph": "paragraph_turns" in chunk, "expected_text": chunk["spoken_text"],
                    "voice_asset": voice["asset_hash"],
                    "turns": [{"text": t["spoken_text"], "direction": effective_delivery(t["segment"])} for t in turns]}
            case["length_class"] = length_class(case["expected_text"])
            case["seed_requested"] = int(digest({"seed": seed, "case": case["id"]})[:8], 16)
            groups[case["length_class"]].append(case)
    chosen, usage = [], Counter()
    # Long cases first ensure real paragraph work is represented, then all
    # selected cases are interleaved using a fixed, independent ordering key.
    for name, count in (("long", 4), ("medium", 7), ("short", 13)):
        remaining = groups[name]
        for position in range(count):
            eligible = [c for c in remaining if c["voice_asset"] in usage or len(usage) < 8]
            # Most long Book 5 requests are grouped paragraphs. Retain that
            # workload even when the stable hash order favors plain passages.
            paragraphs = [c for c in eligible if c["paragraph"]]
            if name == "long" and position < 3 and paragraphs:
                eligible = paragraphs
            if not eligible:
                raise ValueError(f"Need {count} distinct {name} cases within eight saved voices")
            case = min(eligible, key=lambda c: (usage[c["voice_asset"]],
                                               digest({"seed": seed, "case": c["id"]})))
            chosen.append(case)
            usage[case["voice_asset"]] += 1
            remaining.remove(case)
    return sorted(chosen, key=lambda c: digest({"order": seed, "case": c["id"]}))


def prepare(args):
    from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
    from epub_tts_narrator.renderer.bundle import chunk_speakers, load_bundle
    from epub_tts_narrator.renderer.chapter_client import _voice_assets
    from epub_tts_narrator.renderer.config import load_config
    from epub_tts_narrator.renderer.workflow import make_render_plan

    output = args.output.resolve()
    if output.exists():
        raise ValueError("Benchmark corpus output must be a new directory")
    bundle = load_bundle(args.book)
    config = resolve_book_config(bundle, load_config(args.config), work_dir=output / "unused-plan-work")
    plan = make_render_plan(bundle, config)
    if plan["errors"]:
        raise ValueError("Book render plan has unresolved errors")
    store = voice_store(bundle, config)
    used = {s["voice_id"] for ch in plan["chapters"] for c in ch["chunks"] if not c["silent_control"]
            for s in chunk_speakers(c)}
    voices = {key: store.load(key) for key in sorted(used)}
    if any(value is None for value in voices.values()):
        raise ValueError("All planned book voices must exist before preparing this comparison")
    cases = select_cases(plan["chapters"], voices, seed=args.seed)
    chosen = {case["voice_asset"] for case in cases}
    selected = {v["asset_hash"]: v for v in voices.values() if v["asset_hash"] in chosen}
    output.mkdir(parents=True)
    bindings, assets = _voice_assets(selected, output, config["render"]["ffmpeg"])
    packed = {}
    for asset, voice in selected.items():
        transfer = assets[bindings[asset]]
        packed[asset] = {"kind": "reference", "asset_hash": asset,
                         "reference_audio": str(transfer["path"].relative_to(output)),
                         "reference_sha256": transfer["manifest"]["audio_sha256"],
                         "reference_text": voice["reference_text"],
                         "source_reference_sha256": voice["reference_sha256"]}
    body = {"format": "epub-tts-render-benchmark", "version": 1,
            "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "selection_seed": args.seed, "cases": cases, "voices": packed}
    write_json(output / "corpus.json", dict(body, corpus_sha256=digest(body)))
    print(f"Prepared {len(cases)} cases and {len(packed)} lossless voice copies: {output / 'corpus.json'}")


def load_corpus(path):
    path = Path(path).resolve()
    corpus = read_json(regular_file(path))
    body = {k: v for k, v in corpus.items() if k != "corpus_sha256"}
    if (corpus.get("format") != "epub-tts-render-benchmark" or corpus.get("version") != 1
            or corpus.get("corpus_sha256") != digest(body)):
        raise ValueError("Benchmark corpus checksum mismatch")
    voices = deepcopy(corpus["voices"])
    for voice in voices.values():
        relative = Path(voice["reference_audio"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Corpus references must be portable relative paths")
        audio = regular_file(path.parent / relative)
        if file_hash(audio) != voice["reference_sha256"]:
            raise ValueError("Benchmark reference changed")
        voice["reference_audio"] = str(audio.resolve())
    ids = set()
    for case in corpus["cases"]:
        if (not isinstance(case.get("id"), str) or not re.fullmatch(r"[0-9a-f]{24}", case["id"])
                or case["id"] in ids or case.get("voice_asset") not in voices
                or type(case.get("seed_requested")) is not int or not 0 <= case["seed_requested"] <= 0xFFFFFFFF
                or type(case.get("paragraph")) is not bool or not case.get("turns")
                or case.get("expected_text") != "".join(turn["text"] for turn in case["turns"])
                or case.get("length_class") != length_class(case["expected_text"])):
            raise ValueError("Invalid fixed benchmark case")
        ids.add(case["id"])
    return corpus, voices


def summarize(rows, generation_wall_seconds, verification_wall_seconds=None):
    measured = [row for row in rows if not row["warmup"]]
    generated = [row for row in measured if "audio_seconds" in row and "error" not in row]
    complete = bool(measured) and all("passed" in row for row in measured)
    accepted = [row for row in measured if row.get("passed") is True]
    audio = sum(row["audio_seconds"] for row in generated)
    verified = sum(row["audio_seconds"] for row in accepted)
    latencies = sorted(row["latency_seconds"] for row in measured)
    return {"cases": len(measured), "generated": len(generated),
            "generation_failures": len(measured) - len(generated),
            "verification_complete": complete,
            "accepted": len(accepted) if complete else None,
            "rejected_or_failed": len(measured) - len(accepted) if complete else None,
            "generation_wall_seconds": generation_wall_seconds,
            "verification_wall_seconds": verification_wall_seconds,
            "generated_audio_seconds": audio,
            "verified_audio_seconds": verified if complete else None,
            "generated_audio_per_generation_second": audio / generation_wall_seconds if generation_wall_seconds else None,
            "verified_audio_per_generation_second": verified / generation_wall_seconds if complete and generation_wall_seconds else None,
            "verified_audio_per_total_second": verified / (generation_wall_seconds + verification_wall_seconds)
                if complete and verification_wall_seconds is not None and generation_wall_seconds + verification_wall_seconds else None,
            "latency_median_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * .95))] if latencies else None}


class GpuSampler:
    """Optional telemetry only; an unavailable sampler cannot fail synthesis."""
    def __init__(self):
        self.samples, self.stop, self.thread = [], Event(), None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            def sample():
                while not self.stop.is_set():
                    try:
                        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw",
                                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
                        if p.returncode == 0:
                            rows = [[float(x.strip()) for x in line.split(",")] for line in p.stdout.splitlines()]
                            self.samples.append({"time": time.time(), "gpus": rows})
                    except (OSError, ValueError, subprocess.TimeoutExpired):
                        pass
                    self.stop.wait(1)
            self.thread = Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)


def generate(args):
    from epub_tts_narrator.renderer.config import load_config
    from epub_tts_narrator.renderer.fish_http import inspect_wav
    from epub_tts_narrator.renderer.pool import create_render_engine

    corpus, voices = load_corpus(args.corpus)
    config = load_config(args.config)
    workers = config.get("render_pool", {}).get("workers", [])
    if args.worker == "local" or not workers:
        if args.concurrency != 1:
            raise ValueError("Local MLX comparison requires concurrency 1")
        # Keep all lazy model operations on the existing persistent MLX thread.
        workers = [{"name": "benchmark-local", "type": "local", "concurrency": 1}]
    else:
        workers = [w for w in workers if args.worker is None or w["name"] == args.worker]
        if len(workers) != 1 or workers[0]["type"] == "local":
            raise ValueError("Select one remote worker, or use --worker local")
        workers = [dict(workers[0], concurrency=args.concurrency)]
    config.setdefault("render_pool", {})["workers"] = workers
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"format": "epub-tts-render-benchmark-results", "version": 1,
              "corpus_sha256": corpus["corpus_sha256"], "corpus": str(args.corpus.resolve()),
              "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "concurrency": args.concurrency, "results": [], "status": "warming",
              "seed_note": "Requested seeds are fixed inputs, not proof that a backend applies them. Stock SGLang has no per-request seed control; its trials remain stochastic.",
              "timing_note": "First-pass calls only, no retries or accepted-cache reuse. Generation makespan includes all successful and failed calls. Warmup is separate. Verification runs afterward; total denominator is the sum of both measured phases, not a pipelined-production estimate."}
    path = args.output / "manifest.json"
    engine = create_render_engine(config)
    secrets = configured_secrets(config)
    report["engine"] = engine.identity
    report["seed_control"] = ("unsupported_stochastic" if "sglang" in str(workers[0].get("protocol", "")).lower()
                              else "requested; application depends on recorded backend metadata")
    report["transport_protocol"] = workers[0].get("protocol", "local")
    write_json(path, report)

    def run(case, warmup=False):
        target = args.output / (case["id"] + ("-warmup" if warmup else "") + ".wav")
        row = {"case_id": case["id"], "warmup": warmup, "length_class": case["length_class"],
               "expected_text": case["expected_text"], "seed_requested": 0 if warmup else case["seed_requested"]}
        started = time.perf_counter()
        try:
            voice = voices[case["voice_asset"]]
            turns = [dict(t, voice=voice) for t in case["turns"]]
            if case["paragraph"]:
                metadata = engine.synthesize_dialogue(turns, target, seed=row["seed_requested"])
            else:
                metadata = engine.synthesize(turns[0]["text"], voice, target,
                                             seed=row["seed_requested"], direction=turns[0]["direction"])
            row["engine_metadata"] = metadata
            row["audio_seconds"] = inspect_wav(target)["duration_seconds"]
            row.update(audio=target.name, audio_sha256=file_hash(target))
        except Exception as exc:
            row.update(error=safe_error_text(f"{type(exc).__name__}: {exc}", secrets), passed=False)
        row["latency_seconds"] = time.perf_counter() - started
        return row

    try:
        warm = {}
        for case in sorted(corpus["cases"], key=lambda c: len(c["expected_text"])):
            warm.setdefault(case["voice_asset"], case)
        started = time.perf_counter()
        for case in warm.values():
            report["results"].append(run(case, True))
            write_json(path, report)
        report["warmup_wall_seconds"] = time.perf_counter() - started
        if any("error" in row for row in report["results"]):
            raise RuntimeError("Reference warmup failed; inspect manifest before measuring")
        report["status"] = "generating"
        with GpuSampler() as sampler:
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(run, case) for case in corpus["cases"]]
                for future in as_completed(futures):
                    report["results"].append(future.result())
                    write_json(path, report)
            elapsed = time.perf_counter() - started
        report.update(gpu_samples=sampler.samples, generation_wall_seconds=elapsed, status="generated")
        report["summary"] = summarize(report["results"], elapsed)
        write_json(path, report)
        print(json.dumps(report["summary"], indent=2))
    finally:
        engine.close()


def verify(args):
    from epub_tts_narrator.renderer.assembly import normalize_clip
    from epub_tts_narrator.renderer.chapter_verification import create_chapter_verifier
    from epub_tts_narrator.renderer.config import load_config
    from epub_tts_narrator.renderer.quality import validate_audio

    config = load_config(args.config)
    report = read_json(regular_file(args.manifest))
    if report.get("status") != "generated":
        raise ValueError("Verification requires a complete, not previously verified generation run")
    verifier = create_chapter_verifier(config["verification"], config.get("jev"), config.get("provider"))
    report["verification_identity"] = verifier.identity
    report["quality"] = deepcopy(config["quality"])
    secrets = configured_secrets(config)
    warmups = [row for row in report["results"] if row["warmup"] and "error" not in row]
    try:
        # Initialize ASR separately; this never changes the measured TTS audio.
        started = time.perf_counter()
        if warmups:
            row = warmups[0]
            raw = regular_file(args.manifest.parent / row["audio"])
            if file_hash(raw) != row["audio_sha256"]:
                raise ValueError("Warmup audio changed")
            normalized = args.manifest.parent / "verification-warmup.wav"
            normalize_clip(raw, normalized, sample_rate=config["render"]["sample_rate"], ffmpeg=config["render"]["ffmpeg"])
            validate_audio(normalized, row["expected_text"], verifier, config["quality"], purpose="speech")
            normalized.unlink()
        report["verification_warmup_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        for row in report["results"]:
            if row["warmup"] or "error" in row:
                continue
            raw = regular_file(args.manifest.parent / row["audio"])
            if file_hash(raw) != row["audio_sha256"]:
                raise ValueError("Measured audio changed before verification")
            began = time.perf_counter()
            normalized = args.manifest.parent / (row["case_id"] + "-verified.wav")
            try:
                normalize_clip(raw, normalized, sample_rate=config["render"]["sample_rate"], ffmpeg=config["render"]["ffmpeg"])
                result = validate_audio(normalized, row["expected_text"], verifier, config["quality"], purpose="speech")
                row.update(verification=result, passed=result["passed"])
            except Exception as exc:
                row.update(verification_error=safe_error_text(f"{type(exc).__name__}: {exc}", secrets), passed=False)
            finally:
                normalized.unlink(missing_ok=True)
            row["verification_seconds"] = time.perf_counter() - began
            write_json(args.manifest, report)
        report["verification_wall_seconds"] = time.perf_counter() - started
        report["summary"] = summarize(report["results"], report["generation_wall_seconds"], report["verification_wall_seconds"])
        report["status"] = "verified"
        write_json(args.manifest, report)
        print(json.dumps(report["summary"], indent=2))
    finally:
        close = getattr(verifier, "close", None)
        if close:
            close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("book", type=Path)
    prepare_parser.add_argument("--seed", type=int, default=1729)
    prepare_parser.add_argument("--output", type=Path, required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("corpus", type=Path)
    generate_parser.add_argument("--worker")
    generate_parser.add_argument("--concurrency", type=int, choices=(1, 4, 8), default=1)
    generate_parser.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("manifest", type=Path)
    for p in (prepare_parser, generate_parser, verify_parser):
        p.add_argument("--config", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "generate": generate, "verify": verify}[args.command](args)


if __name__ == "__main__":
    main()
