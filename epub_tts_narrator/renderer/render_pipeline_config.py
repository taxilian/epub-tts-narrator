"""Validated render_pool scheduler settings and their effective values.

These controls change how clip work is scheduled, never what is synthesized
or accepted. They live in ``render_pool`` only: they never enter
``RENDER_CONTRACT``, ``SAFE_RENDER_KEYS``, chapter settings, or any
generation/verification/cache identity. ``0`` means automatic where a
setting supports it, mirroring ``max_inflight``.

Defaults keep today's behavior: ``scheduler = "legacy"``. The pipeline
scheduler is an explicit opt-in until measured throughput gates pass.
"""

from __future__ import annotations

from copy import deepcopy

SCHEDULERS = ("legacy", "pipeline")

PIPELINE_RENDER_POOL_DEFAULTS = {
    "scheduler": "legacy",
    "chapter_concurrency": 0,  # automatic: 4 for pipeline, 2 effective in legacy mode
    "synthesis_backlog": 0,    # automatic: min(256, 4 * total TTS slots)
    "pending_audio_mb": 256,   # soft high-water for materialized untranscribed audio
    "review_concurrency": 2,   # total active leaf network review calls
}


def validate_pipeline_settings(pool, tts_slots):
    """Validate the scheduler controls without any external or model access."""
    scheduler = pool.get("scheduler", "legacy")
    if not isinstance(scheduler, str) or scheduler not in SCHEDULERS:
        raise ValueError("render_pool.scheduler must be legacy or pipeline")
    chapter_concurrency = pool.get("chapter_concurrency", 0)
    if type(chapter_concurrency) is not int or not 0 <= chapter_concurrency <= 16:
        raise ValueError("render_pool.chapter_concurrency must be 0 (automatic) "
                         "or an integer from 1 through 16")
    backlog = pool.get("synthesis_backlog", 0)
    if type(backlog) is not int or not 0 <= backlog <= 256:
        raise ValueError("render_pool.synthesis_backlog must be 0 (automatic) "
                         "or an integer from the total TTS slots through 256")
    if backlog and backlog < tts_slots:
        raise ValueError("render_pool.synthesis_backlog must be at least the total "
                         "TTS worker concurrency")
    pending = pool.get("pending_audio_mb", PIPELINE_RENDER_POOL_DEFAULTS["pending_audio_mb"])
    if type(pending) is not int or not 64 <= pending <= 4096:
        raise ValueError("render_pool.pending_audio_mb must be an integer from 64 through 4096")
    review = pool.get("review_concurrency", PIPELINE_RENDER_POOL_DEFAULTS["review_concurrency"])
    if type(review) is not int or not 1 <= review <= 8:
        raise ValueError("render_pool.review_concurrency must be an integer from 1 through 8")


def chapter_share_for(pool_or_effective):
    """Logical runners one chapter may occupy: non-borrowable per chapter."""
    from .render_pipeline import chapter_share
    return chapter_share(pool_or_effective["max_inflight"], pool_or_effective["chapter_concurrency"])


def effective_scheduling(config):
    """Effective scheduling values for a loaded renderer configuration.

    Explicit values win; automatic values derive from the total TTS slot
    capacity. Automatic ``max_inflight`` stays at or above that capacity in
    every mode, and chapter counts never multiply the shared physical caps.
    """
    pool = config["render_pool"]
    workers = pool.get("workers") or []
    slots = sum(worker.get("concurrency", 1) for worker in workers)
    remote = any(worker.get("type") != "local" for worker in workers)
    scheduler = pool.get("scheduler", "legacy")
    chapter_concurrency = pool.get("chapter_concurrency") or (4 if scheduler == "pipeline" else 2)
    max_inflight = pool.get("max_inflight", 0) or (min(256, max(64, 16 * slots)) if remote else 16)
    backlog = pool.get("synthesis_backlog") or min(256, 4 * slots)
    return {
        "scheduler": scheduler,
        "chapter_concurrency": chapter_concurrency,
        "max_inflight": max_inflight,
        "synthesis_backlog": backlog,
        "pending_audio_mb": pool.get("pending_audio_mb",
                                     PIPELINE_RENDER_POOL_DEFAULTS["pending_audio_mb"]),
        "review_concurrency": pool.get("review_concurrency",
                                       PIPELINE_RENDER_POOL_DEFAULTS["review_concurrency"]),
    }


def project_pipeline_controls(local_render_pool):
    """Deep-copied scheduler controls for a remote worker projection."""
    pool = local_render_pool or {}
    projected = {key: deepcopy(pool.get(key, default))
                 for key, default in PIPELINE_RENDER_POOL_DEFAULTS.items()}
    projected["max_inflight"] = deepcopy(pool.get("max_inflight", 0))
    return projected
