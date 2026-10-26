"""Run manifest and honest reporting for the chapter pipeline benchmark.

The manifest records input-bound pending units before any expensive work and
durably marks completed units; it never carries sessions, job identifiers, or
execution grants. The report keeps warmup, active pipeline, assembly, and
cold setup times separate, counts frozen canonical source words once, marks
resumed or cached runs ineligible for speed gates, and refuses to claim
throughput for partial, failed, or interrupted runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.renderer.common import read_json, regular_file, write_json

MANIFEST_FORMAT = "epub-tts-chapter-pipeline-benchmark-run"
REPORT_FORMAT = "epub-tts-chapter-pipeline-benchmark-results"
SEED_NOTE = ("Requested seeds are frozen per case as seed_requested. The pinned SGLang backend does "
             "not apply per-request seeds, so seed_applied stays false; repeated matched runs and "
             "unchanged acceptance rules, not seed equality, control stochastic variation.")


def group_word_counts(corpus):
    counts = [0] * 4
    for case in corpus["cases"]:
        counts[case["group"]] += case["words"]
    return counts


def metrics_window(start, end):
    """Bounded counter deltas over one labeled measurement window.

    Counters are monotone, so subtraction isolates the window without
    resetting live gauges; instantaneous queued/active gauges come from the
    window's end and must read zero once its work has fully drained.
    """
    window = {}
    for stage, after in end.items():
        if not isinstance(after, dict):
            continue
        before = start.get(stage, {})
        entry = {"queued_now": after["queued_now"], "active_now": after["active_now"]}
        for key in ("wait_seconds", "wait_count", "active_seconds", "active_count"):
            entry[key] = round(after[key] - before.get(key, 0.0), 6)
        for kind in ("wait_buckets", "active_buckets"):
            entry[kind] = {name: after[kind][name] - before.get(kind, {}).get(name, 0)
                           for name in after[kind]}
        entry["outcomes"] = {name: after["outcomes"][name] - before.get("outcomes", {}).get(name, 0)
                             for name in after["outcomes"]}
        window[stage] = entry
    return window


def build_telemetry(window_start, warmup_end, active_end, gauges):
    """Stage histograms/outcomes per window plus lane/backlog/logical gauges.

    ``gauges`` comes from the live pool before shutdown; None means legacy
    scheduling without a lane runtime, reported as explicitly unavailable.
    """
    telemetry = {"warmup_window": metrics_window(window_start, warmup_end),
                 "active_window": metrics_window(warmup_end, active_end)}
    if gauges is None:
        telemetry["resource_lanes"] = "unavailable"
        telemetry["backlog"] = "unavailable"
        telemetry["logical"] = "unavailable"
        telemetry["runtime_limits"] = "unavailable"
        telemetry["backlog_peak_integration"] = "unavailable"
        return telemetry
    telemetry["resource_lanes"] = gauges["lanes"]
    telemetry["backlog"] = gauges["backlog"]
    telemetry["logical"] = gauges["logical"]
    telemetry["runtime_limits"] = gauges["runtime_limits"]
    captured = isinstance(gauges["backlog"], dict) and any(
        key.startswith("peak_") for key in gauges["backlog"])
    telemetry["backlog_peak_integration"] = "captured" if captured else "pending"
    return telemetry


def init_output(output, corpus, scheduler, *, fake_engine):
    """Create a fresh run root or validate an interrupted one for resume."""
    output = Path(output).resolve()
    manifest_path = output / "run-manifest.json"
    if not output.exists():
        output.mkdir(parents=True)
        return {"format": MANIFEST_FORMAT, "version": 1,
                "corpus_sha256": corpus["corpus_sha256"], "input_hash": corpus["input_hash"],
                "scheduler": scheduler, "fake_engine": bool(fake_engine), "status": "running",
                "groups": [{"group": group, "state": "pending"}
                           for group in range(4)]}, False
    if not manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("Benchmark output exists without this run's manifest; choose a new --output")
    manifest = read_json(regular_file(manifest_path))
    if (manifest.get("format") != MANIFEST_FORMAT or manifest.get("version") != 1
            or manifest.get("corpus_sha256") != corpus["corpus_sha256"]
            or manifest.get("input_hash") != corpus["input_hash"]
            or manifest.get("scheduler") != scheduler):
        raise ValueError("Saved benchmark run manifest does not match this corpus and scheduler")
    if manifest.get("fake_engine") != bool(fake_engine):
        # Fake and real model runs must never mix inside one resume state.
        raise ValueError("Saved benchmark run used a different engine policy (--fake-engine); "
                         "choose a new --output")
    if manifest.get("status") == "complete":
        raise ValueError("This benchmark output already completed; choose a new --output")
    manifest["status"] = "running"
    save_manifest(output, manifest)
    return manifest, True


def save_manifest(output, manifest):
    write_json(Path(output) / "run-manifest.json", manifest)


def set_group_state(manifest, group, state):
    manifest["groups"][group]["state"] = state


def new_report(*, corpus, corpus_path, config, capabilities, scheduler, output, resumed,
               warm_words, fake_engine=False):
    from epub_tts_narrator.renderer.render_pipeline_config import effective_scheduling
    words = group_word_counts(corpus)
    if fake_engine:
        eligible, reason = False, "fake_engine"
    elif resumed:
        eligible, reason = False, "resumed_or_cached"
    else:
        eligible, reason = True, "fresh"
    return {"format": REPORT_FORMAT, "version": 1, "mode": "chapter_pipeline",
            "corpus": str(Path(corpus_path).resolve()), "corpus_sha256": corpus["corpus_sha256"],
            "input_hash": corpus["input_hash"], "sample": corpus.get("sample"),
            "scheduler": scheduler, "effective_scheduling": effective_scheduling(config),
            "renderer_identity": capabilities.get("engine"),
            "renderer_fingerprint": capabilities.get("renderer_fingerprint"),
            "parallel_chapters": capabilities.get("parallel_chapters"),
            "fake_engine": bool(fake_engine),
            "engine_policy": "fake_engine" if fake_engine else "configured_models",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "status": "running",
            "speed_gate_eligible": eligible,
            "eligibility_reason": reason,
            "groups": [{"group": group, "chapter_number": group + 2, "state": "pending",
                        "source_words": words[group]} for group in range(4)],
            "warmup_source_words": warm_words,
            "total_source_words": corpus["total_source_words"], "accepted_source_words": 0,
            "timing": {"cold_setup_seconds": None, "warmup_seconds": None, "warmup_cache_hit": False,
                       "active_pipeline_seconds": None, "assembly_seconds": None, "elapsed_seconds": None},
            "seed_applied": False, "seed_note": SEED_NOTE,
            "gpu_samples": [], "gpu_telemetry": "unavailable",
            "accepted_words_per_active_second": None}


def save_report(output, report):
    write_json(Path(output) / "report.json", report)


def finalize(report, manifest, *, status, error=None):
    """Close out the report; only complete eligible runs earn a throughput claim."""
    report["status"] = status
    report["groups"] = [{**group, "state": manifest["groups"][group["group"]]["state"]}
                        for group in report["groups"]]
    complete_words = sum(group["source_words"] for group in report["groups"] if group["state"] == "complete")
    report["accepted_source_words"] = complete_words if status in {"complete", "failed"} else 0
    timing = report["timing"]
    if timing.get("elapsed_seconds") is None:
        timing["elapsed_seconds"] = sum(value for value in (
            timing.get("cold_setup_seconds"), timing.get("warmup_seconds"),
            timing.get("active_pipeline_seconds"), timing.get("assembly_seconds")) if value)
    eligible = status == "complete" and report["speed_gate_eligible"] and \
        bool(timing.get("active_pipeline_seconds"))
    if eligible:
        report["accepted_words_per_active_second"] = \
            report["total_source_words"] / timing["active_pipeline_seconds"]
    elif status != "complete":
        report["partial_note"] = ("A partial, failed, or interrupted run reports elapsed cost only; "
                                  "it never claims accepted-work throughput.")
    if not report["gpu_samples"]:
        report["gpu_telemetry"] = "unavailable"
    else:
        report["gpu_telemetry"] = "sampled"
    if error is not None:
        report["error"] = error
    return report
