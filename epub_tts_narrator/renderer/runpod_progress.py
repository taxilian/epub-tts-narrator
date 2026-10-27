"""Passive Runpod queue inspection; observations never grant execution."""

import json
import math
from pathlib import Path
import re
import time

from .chapter_client import ChapterClient, ChapterClientError
from .render_metrics import STAGES

_SCHEDULE_INT_KEYS = {
    "jobs": ("queued", "generating", "assembling", "complete", "failed", "cancelled"),
    "capacity": ("tts_slots", "max_inflight", "pipeline_projection_max_inflight",
                 "chapter_concurrency", "review_concurrency", "synthesis_backlog",
                 "pending_audio_mb"),
    "logical": ("pending", "inflight", "owners", "global_limit", "chapter_share"),
    "backlog": ("pending_count", "peak_pending_count", "materialized_bytes",
                "peak_materialized_bytes", "max_materialized_bytes",
                "low_water_bytes", "waiters"),
}
_SCHEDULE_MODES = {"pipeline", "legacy"}
_SCHEDULE_BASES = {"pipeline_global_transactions", "legacy_per_chapter_window"}


def _public_scheduling(value):
    """Project the server's scheduling block onto fixed numeric keys.

    The chapter-service wrapper nests the engine snapshot under
    ``engine``; a direct engine block is accepted too. Unknown keys,
    stages, modes, and non-numeric values are dropped, so a broken or
    future server can never push arbitrary labels into observer output.
    Anything unusable becomes absent rather than invented.
    """
    if not isinstance(value, dict):
        return None
    # Queue counts live on the wrapper; engine gauges live on the nested
    # engine block (or directly on the block when served without one).
    jobs_source = value
    engine = value.get("engine")
    if isinstance(engine, dict):
        jobs_source, value = value, engine
    public = {}
    mode = value.get("mode")
    if mode in _SCHEDULE_MODES:
        public["mode"] = mode
    jobs = jobs_source.get("jobs")
    if isinstance(jobs, dict):
        projected = {key: jobs[key] for key in _SCHEDULE_INT_KEYS["jobs"]
                     if type(jobs.get(key)) is int}
        if projected:
            public["jobs"] = projected
    capacity = value.get("capacity")
    if isinstance(capacity, dict):
        projected = {key: capacity[key] for key in _SCHEDULE_INT_KEYS["capacity"]
                     if type(capacity.get(key)) is int}
        basis = capacity.get("inflight_basis")
        if basis in _SCHEDULE_BASES:
            projected["inflight_basis"] = basis
        if projected:
            public["capacity"] = projected
    logical = value.get("logical")
    if isinstance(logical, dict):
        projected = {key: logical[key] for key in _SCHEDULE_INT_KEYS["logical"]
                     if type(logical.get(key)) is int}
        closed = logical.get("closed")
        if isinstance(closed, bool):
            projected["closed"] = closed
        if projected:
            public["logical"] = projected
    backlog = value.get("backlog")
    if isinstance(backlog, dict):
        projected = {key: backlog[key] for key in _SCHEDULE_INT_KEYS["backlog"]
                     if type(backlog.get(key)) is int}
        blocked = backlog.get("new_generation_blocked")
        if isinstance(blocked, bool):
            projected["new_generation_blocked"] = blocked
        if projected:
            public["backlog"] = projected
    stages = value.get("stages")
    if isinstance(stages, dict):
        projected = {}
        for stage, gauges in stages.items():
            if stage not in STAGES or not isinstance(gauges, dict):
                continue
            queued, active = gauges.get("queued_now"), gauges.get("active_now")
            if type(queued) is int or type(active) is int:
                projected[stage] = {"queued_now": queued if type(queued) is int else 0,
                                    "active_now": active if type(active) is int else 0}
        if projected:
            public["stages"] = projected
    return public or None


def _mib(count):
    return count / (1024 * 1024)


def _scheduling_line(scheduling):
    """One fixed-label text line; numbers only, no free-form server text."""
    parts = [scheduling.get("mode") or "unknown-mode"]
    jobs = scheduling.get("jobs")
    if jobs:
        parts.append("jobs " + " ".join(f"{key} {jobs[key]}" for key in
                                        _SCHEDULE_INT_KEYS["jobs"] if key in jobs))
    capacity = scheduling.get("capacity")
    if capacity:
        detail = []
        if "tts_slots" in capacity:
            detail.append(f"slots {capacity['tts_slots']}")
        if "max_inflight" in capacity:
            basis = capacity.get("inflight_basis", "")
            detail.append(f"inflight {capacity['max_inflight']}"
                          + (f" ({basis})" if basis else ""))
        if detail:
            parts.append(", ".join(detail))
    logical = scheduling.get("logical")
    if logical and ("inflight" in logical or "pending" in logical):
        parts.append(f"logical {logical.get('pending', 0)}/{logical.get('global_limit', 0)}"
                     f" active {logical.get('inflight', 0)} owners {logical.get('owners', 0)}")
    backlog = scheduling.get("backlog")
    if backlog:
        detail = f"backlog {backlog.get('pending_count', 0)} pend (peak {backlog.get('peak_pending_count', 0)})"
        if "materialized_bytes" in backlog:
            detail += (f", {_mib(backlog['materialized_bytes']):.1f} MiB"
                       f" (peak {_mib(backlog.get('peak_materialized_bytes', 0)):.1f} MiB)")
        if backlog.get("waiters"):
            detail += f", wait {backlog['waiters']}"
        if backlog.get("new_generation_blocked"):
            detail += ", BLOCKED above soft high-water"
        parts.append(detail)
    stages = scheduling.get("stages")
    if stages:
        parts.append(" ".join(f"{stage} {gauges['queued_now']}q/{gauges['active_now']}a"
                              for stage, gauges in sorted(stages.items())))
    return "scheduling: " + "; ".join(parts)


def run_progress(settings, args):
    if type(args.limit) is not int or not 1 <= args.limit <= 256:
        raise ValueError("Progress limit must be between 1 and 256")
    if not math.isfinite(args.poll_seconds) or not 0.5 <= args.poll_seconds <= 60:
        raise ValueError("Progress poll interval must be between 0.5 and 60 seconds")
    if args.job is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.job):
        raise ValueError("Invalid progress job ID")
    cfg = settings["deployment"]
    path = Path(cfg["state_dir"]) / "state.json"
    if not path.is_file():
        print("No deployment state. Run runpod encode BOOK to bootstrap when ready.")
        return 0
    state = json.loads(path.read_text())
    pod_id = state.get("pod_id")
    if not pod_id:
        print("No worker allocated. Saved work is paused; progress never starts compute.")
        return 0
    client = ChapterClient({"url": f"https://{pod_id}-{cfg['service_port']}.proxy.runpod.net",
                            "api_key": state["api_key"], "timeout_seconds": 10})
    cursors = {}
    try:
        while True:
            try:
                response = client.request("GET", f"v1/jobs?limit={args.limit}")
            except ChapterClientError as exc:
                raise ChapterClientError("Cannot read worker progress. The service may be stopped or require an update; "
                                         "no compute was started and no jobs were resumed.", status=exc.status) from None
            jobs = response.get("jobs")
            if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
                raise ChapterClientError("Worker returned an invalid progress overview")
            if args.job is not None:
                jobs = [job for job in jobs if job.get("id") == args.job]
                if not jobs:
                    try:
                        job = client.request("GET", f"v1/jobs/{args.job}")
                    except ChapterClientError as exc:
                        if exc.status != 404:
                            raise
                        print(json.dumps({"jobs": []}) if args.json else "No current run for this chapter; completed run history is not retained.")
                        return 0
                    jobs = [job]
            public = []
            for job in jobs:
                record = {key: job.get(key) for key in (
                    "id", "state", "paused", "owner_connected", "chapter_number", "total_chapters",
                    "source_fingerprint", "updated", "last_event")}
                if job.get("state") in {"generating", "assembling"} or args.job is not None:
                    identifier = job.get("id")
                    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier):
                        raise ChapterClientError("Worker returned an invalid progress job ID")
                    last = job.get("last_event", 0)
                    after = max(0, last - 500) if type(last) is int else 0
                    try:
                        batch = client.request("GET", f"v1/jobs/{identifier}/events?after={after}")
                    except ChapterClientError as exc:
                        if exc.status == 404:
                            continue
                        raise
                    for event in reversed(batch.get("events", [])):
                        message = event.get("message", "") if isinstance(event, dict) else ""
                        if not isinstance(message, str):
                            continue
                        match = re.search(r"Chapter (\d+) / (\d+) \| Blocks (\d+) / (\d+) complete \| elapsed ([0-9:]+)", message)
                        if match:
                            record["progress"] = {"completed_blocks": int(match[3]), "total_blocks": int(match[4]),
                                                  "elapsed": match[5]}
                            break
                public.append(record)
            scheduling = _public_scheduling(response.get("scheduling"))
            if args.json:
                print(json.dumps({"jobs": public,
                                  **({"scheduling": scheduling} if scheduling else {})}), flush=True)
            elif not jobs:
                print("No saved chapter jobs.", flush=True)
                if scheduling:
                    print(_scheduling_line(scheduling), flush=True)
            else:
                if scheduling:
                    line = _scheduling_line(scheduling)
                    if cursors.get("scheduling") != line:
                        cursors["scheduling"] = line
                        print(line, flush=True)
                for job in public:
                    identifier = job.get("id")
                    signature = (job.get("state"), job.get("paused"), job.get("last_event"))
                    if args.watch and cursors.get(identifier) == signature:
                        continue
                    cursors[identifier] = signature
                    status = "paused: no connected rendering client" if job.get("paused") else str(job.get("state"))
                    chapter = (f"Chapter {job['chapter_number']} / {job['total_chapters']}"
                               if job.get("chapter_number") is not None else "Chapter")
                    progress = job.get("progress")
                    detail = (f"; blocks {progress['completed_blocks']} / {progress['total_blocks']}; "
                              f"elapsed {progress['elapsed']}" if progress else "")
                    print(f"{chapter}: {status}{detail}; job {identifier}; event {job.get('last_event')}", flush=True)
            if not args.watch:
                return 0
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 0
