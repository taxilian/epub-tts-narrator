"""Bounded whole-segment recovery for output-truncated speech spelling scans."""

from copy import deepcopy
import json
from pathlib import Path

from .pipeline import fingerprint, write_json
from .prompts import PROMPT_VERSION
from .provider import OutputTruncatedError
from .speech_prompts import SCAN as compact_scan, pack_payload


MAX_SPLIT_DEPTH = 2


def _split_hint(requests, system, payload):
    cache = getattr(requests, "cache_dir", None)
    client = getattr(requests, "client", None)
    if cache is None or client is None:
        return None, None, False
    entries = []
    for prompt, data in ((compact_scan, pack_payload(payload)), (system, payload)):
        key = fingerprint({"provider": client.identity, "prompt_version": PROMPT_VERSION,
                           "system": prompt, "payload": data})
        marker = {"version": 1, "reason": "model_output_truncated", "request_cache_fingerprint": key}
        entries.append((Path(cache) / "speech-spelling-subdivisions" / f"{key}.json", marker, key))
    path, marker, _ = entries[0]
    if any((Path(cache) / f"{key}.json").exists() for _, _, key in entries):
        return path, marker, False
    for candidate, expected, _ in entries:
        if candidate.is_file() and not candidate.is_symlink():
            try:
                with candidate.open(encoding="utf-8") as stream:
                    raw = stream.read(2049)
                if len(raw) <= 2048 and json.loads(raw) == expected:
                    return path, marker, True
            except (OSError, ValueError, RecursionError):
                continue
    return path, marker, False


def scan_with_recovery(requests, system, payload, rows, run, progress):
    """Split only confirmed output exhaustion, keeping all original source visible.

    Children run sequentially inside the current worker. Hints only bypass a
    known failed parent; every child's response still passes the real validator.
    """
    targets = payload["target_segments"]

    def scan(part_rows, part_payload, depth):
        ids = [row["id"] for row in part_rows]
        record = {"target_segment_ids": ids, "request_fingerprint": fingerprint(pack_payload(part_payload)),
                  "depth": depth}
        can_split = len(part_rows) > 1 and depth < MAX_SPLIT_DEPTH
        path, marker, replay = _split_hint(requests, system, part_payload) if can_split else (None, None, False)
        if not replay:
            try:
                return run(part_payload, part_rows), record
            except OutputTruncatedError as exc:
                if not can_split:
                    raise OutputTruncatedError(
                        f"{exc}\nSpeech spelling subdivision stopped at depth {depth}/{MAX_SPLIT_DEPTH} "
                        f"with {len(ids)} target segment(s). Check provider.max_output_tokens and "
                        "the provider/model output limit; no partial response was accepted.") from exc
                if path is not None:
                    write_json(path, marker, overwrite=True)
        progress(f"Speech spelling scan batch {payload['batch']}: subdividing truncated scan "
                 f"at depth {depth + 1}/{MAX_SPLIT_DEPTH}" + (" (cached truncation hint)" if replay else ""))
        accepted, children = [], []
        midpoint = len(part_rows) // 2
        for selected in (part_rows[:midpoint], part_rows[midpoint:]):
            selected_ids = {row["id"] for row in selected}
            child = dict(payload,
                         target_segments=[deepcopy(row) for row in targets if row["segment_id"] in selected_ids],
                         context_segments=[deepcopy(row) for row in targets if row["segment_id"] not in selected_ids])
            found, child_record = scan(selected, child, depth + 1)
            accepted.extend(found)
            children.append(child_record)
        record.update(status="subdivided", replayed_hint=replay, subscans=children)
        return accepted, record

    return scan(rows, payload, 0)
