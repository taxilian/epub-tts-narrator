"""Passive Runpod queue inspection; observations never grant execution."""

import json
import math
from pathlib import Path
import re
import time

from .chapter_client import ChapterClient, ChapterClientError


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
                    job = client.request("GET", f"v1/jobs/{args.job}")
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
                    batch = client.request("GET", f"v1/jobs/{identifier}/events?after={after}")
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
            if args.json:
                print(json.dumps({"jobs": public}), flush=True)
            elif not jobs:
                print("No saved chapter jobs.", flush=True)
            else:
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
