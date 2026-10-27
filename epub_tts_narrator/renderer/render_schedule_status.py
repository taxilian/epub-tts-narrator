"""Read-only scheduling status for progress and diagnostics.

One numeric snapshot of the renderer's current scheduling state: the active
scheduler mode, effective capacity, the logical clip scheduler's gauges, the
attempt-backlog gauges (including its monotonic peaks), and the per-stage
queued/active counts from :mod:`render_metrics`. Everything is observed from
already-initialized resources only: observing never creates an engine,
model, pod, session, lease renewal, or any other resource, never mutates a
counter, and never blocks a worker beyond taking the small existing locks
the gauges already use.

Conservatism rules: a runtime that exposes none of the known scheduling
attributes reports nothing (``None``) rather than invented values; a section
whose read raises is reported as absent while the rest of the snapshot
still renders; unknown plugins and plain engines stay silent. Snapshots
carry only numbers, booleans, and the fixed labels defined here — never
book text, transcripts, paths, URLs, keys, or user labels. There is no
per-run history: the module holds one bounded runtime slot (the current
process's pooled render, registered by the chapter executor) and nothing
else; queues stay ephemeral in memory.
"""

from __future__ import annotations

from threading import Lock

from .render_metrics import METRICS

# Fixed basis labels so a legacy per-chapter window can never be mistaken
# for the pipeline's global transaction ceiling.
LEGACY_WINDOW_BASIS = "legacy_per_chapter_window"
PIPELINE_BASIS = "pipeline_global_transactions"


_runtime = None
_runtime_lock = Lock()


def register_runtime(engine, effective=None):
    """Point the snapshot at this process's pooled render engine.

    One slot, replaced wholesale; ``effective`` is the already-computed
    ``effective_scheduling`` mapping (or None when unknown). Registration
    is passive: no engine attribute is created and nothing is started.
    """
    global _runtime
    with _runtime_lock:
        _runtime = (engine, effective)


def clear_runtime(engine):
    """Drop the slot only when it still holds this exact engine."""
    global _runtime
    with _runtime_lock:
        if _runtime is not None and (_runtime[0] is engine or engine is None):
            _runtime = None


def _int(value):
    return value if type(value) is int else None


def _safe(read):
    try:
        return read()
    except Exception:
        # An observation must never fail a render or an observer; the
        # section is reported absent instead.
        return None


def _logical_status(engine):
    scheduler = getattr(engine, "logical", None)
    if scheduler is None:
        return None
    snapshot = _safe(lambda: scheduler.snapshot())
    return snapshot if isinstance(snapshot, dict) else None


def _backlog_status(engine):
    resources = getattr(engine, "resources", None)
    holder = resources if resources is not None else engine
    backlog = getattr(holder, "backlog", None)
    if backlog is None:
        return None
    snapshot = _safe(lambda: backlog.snapshot())
    return snapshot if isinstance(snapshot, dict) else None


def _capacity_status(engine, effective):
    slots = _int(_safe(lambda: getattr(engine, "max_parallel", None)))
    inflight = _int(_safe(lambda: getattr(engine, "max_inflight", None)))
    logical = getattr(engine, "logical", None)
    if logical is not None and inflight is not None:
        mode, basis = "pipeline", PIPELINE_BASIS
    elif slots is not None and inflight is not None:
        mode, basis = "legacy", LEGACY_WINDOW_BASIS
    else:
        mode, basis = None, None
    capacity = {"tts_slots": slots, "max_inflight": inflight, "inflight_basis": basis}
    if mode == "legacy" and isinstance(effective, dict):
        projection = _int(effective.get("max_inflight"))
        if projection is not None:
            capacity["pipeline_projection_max_inflight"] = projection
    if isinstance(effective, dict):
        for key in ("chapter_concurrency", "review_concurrency",
                    "synthesis_backlog", "pending_audio_mb"):
            value = _int(effective.get(key))
            if value is not None:
                capacity[key] = value
    return mode, capacity


def _stage_status():
    stages = {}
    snapshot = _safe(METRICS.snapshot)
    if not isinstance(snapshot, dict):
        return stages
    for stage, gauges in snapshot.items():
        if not isinstance(gauges, dict):
            continue
        queued, active = _int(gauges.get("queued_now")), _int(gauges.get("active_now"))
        if queued or active:
            stages[stage] = {"queued_now": queued or 0, "active_now": active or 0}
    return stages


def scheduling_status():
    """The current numeric scheduling snapshot, or None when nothing runs.

    Resolution order: the registered process runtime, else the pipeline
    resources scoped on this thread (a render runner's own transaction),
    else only the process-wide stage gauges. ``None`` means no usable
    scheduling state exists; callers must skip the section entirely.
    """
    with _runtime_lock:
        registered = _runtime
    engine = registered[0] if registered else None
    effective = registered[1] if registered else None
    if engine is None:
        from .render_resource_context import current_resources
        engine = current_resources()
    logical = _logical_status(engine) if engine is not None else None
    backlog = _backlog_status(engine) if engine is not None else None
    mode, capacity = _capacity_status(engine, effective) if engine is not None else (None, None)
    stages = _stage_status()
    usable = any((logical, backlog, stages, mode is not None,
                  isinstance(capacity, dict)
                  and any(type(value) is int for value in capacity.values())))
    if not usable:
        return None
    return {"available": True, "mode": mode, "capacity": capacity,
            "logical": logical, "backlog": backlog, "stages": stages}
