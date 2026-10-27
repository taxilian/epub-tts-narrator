"""Pipeline resource runtime: one bounded lane set owned by its RenderPool.

``RenderResources`` instantiates the declared physical lanes for one pooled
render: a TTS lane with one worker per declared endpoint slot, the
two-worker audio-preparation lane, the single recognition dispatch lane, the
review HTTP lane sized to ``review_concurrency``, and the
``2 * review_concurrency`` blind-review coordination lane. Worker counts are
declared by configuration, never inferred; model-affine callables keep
delegating to ``LocalRuntime``, so lanes never duplicate a model.

Submission contracts:

* Synthesis (:meth:`RenderPool`) submits from the caller's context, so the
  lane's dequeue stop check prevents any new GPU start after owner loss.
* Verification leaves — preparation, recognition, review parents, and review
  HTTP leaves — drain instead: they are submitted from a fresh context so a
  scoped owner predicate that expired mid-take cannot discard an already
  started take's durable verification. The global stop controller still
  gates queued work at dequeue, and the job re-establishes the resource
  scope inside its own copied context for nested leaf routing.

``close`` drains the decision and verification lanes (started-take
verification publishes its durable result) and cancels queued synthesis, so
shutdown never starts new GPU work and never leaks lane threads.
"""

from __future__ import annotations

from contextvars import Context

from .render_backlog import BacklogBudget
from .render_lanes import ResourceLane
from .render_resource_context import resource_scope

# Pending bounds that no admissible submitter pattern can exceed: at most
# ``max_inflight`` transaction runners block on one leaf each, and the
# coordination workers add at most one queued HTTP leaf each.
_REVIEW_HEADROOM = 2


class RenderResources:
    """Declared resource lanes for one pooled render; lifetime: its RenderPool."""

    def __init__(self, *, tts_slots, synthesis_backlog, review_concurrency,
                 max_inflight, pending_audio_mb=256):
        if type(tts_slots) is not int or tts_slots < 1:
            raise ValueError("tts_slots must be a positive integer")
        if type(synthesis_backlog) is not int or synthesis_backlog < 1:
            raise ValueError("synthesis_backlog must be a positive integer")
        if type(review_concurrency) is not int or review_concurrency < 1:
            raise ValueError("review_concurrency must be a positive integer")
        if type(max_inflight) is not int or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        if type(pending_audio_mb) is not int or pending_audio_mb < 1:
            raise ValueError("pending_audio_mb must be a positive integer")
        pending = max_inflight
        review_pending = max_inflight + _REVIEW_HEADROOM * review_concurrency
        self.tts = ResourceLane(name="render-tts", workers=tts_slots,
                                stage="synthesis", max_pending=synthesis_backlog)
        self.preparation = ResourceLane(name="render-preparation", workers=2,
                                        stage="preparation", max_pending=pending)
        self.recognition = ResourceLane(name="render-recognition", workers=1,
                                        stage="recognition", max_pending=pending)
        self.review = ResourceLane(name="render-review", workers=review_concurrency,
                                   stage="review", max_pending=review_pending)
        self.review_coordination = ResourceLane(
            name="render-review-coordination", workers=2 * review_concurrency,
            stage="review_coordination", max_pending=review_pending)
        # Attempt-scoped untranscribed-audio backpressure: the exact pending
        # count follows the declared synthesis backlog and the soft byte
        # high-water follows ``render_pool.pending_audio_mb``.
        self.backlog = BacklogBudget(
            max_pending=synthesis_backlog,
            max_materialized_bytes=pending_audio_mb * 1024 * 1024)
        self._closed = False

    def close(self):
        """Wake backlog waiters, drain verification, cancel synthesis, join."""
        self.backlog.close()
        for lane in (self.review_coordination, self.review, self.recognition,
                     self.preparation):
            lane.close(drain=True)
        self.tts.close(drain=False)
        self._closed = True

    def run_preparation(self, function, *args, owner=(None, None), **kwargs):
        """Route one preparation leaf; drains past a lost owner."""
        return self._drained(self.preparation, function, owner, args, kwargs)

    def run_recognition(self, function, *args, owner=(None, None), **kwargs):
        """Route one recognition leaf; drains past a lost owner."""
        return self._drained(self.recognition, function, owner, args, kwargs)

    def run_review_parent(self, function, *args, owner=(None, None), **kwargs):
        """Run a parent review routine on coordination capacity, never on HTTP."""
        return self._drained(self.review_coordination, function, owner, args, kwargs)

    def run_review_leaf(self, function, *args, owner=(None, None), **kwargs):
        """Route one actual network review request onto the review lane."""
        return self._drained(self.review, function, owner, args, kwargs)

    def _drained(self, lane, function, owner, args, kwargs):
        def job(*job_args, **job_kwargs):
            with resource_scope(self, owner):
                return function(*job_args, **job_kwargs)

        def submission():
            return lane.submit(job, *args, owner=owner, **kwargs).result()

        # A fresh Context strips inherited scoped stop conditions from the
        # copied job context: verification of a started take drains past a
        # lost owner. ResourceLane still applies the global stop controller
        # at dequeue, and each job receives its own private context copy.
        return Context().run(submission)
