"""Scoped context that carries pipeline render resources to leaf operations.

A pipeline render establishes one :class:`RenderResources` runtime (owned by
its ``RenderPool``) and wraps each clip transaction in :func:`resource_scope`.
Code far from the pool — normalization, acoustic metrics, recognition
adapters, and network review leaves — then routes its single leaf operation
to the matching bounded lane. Without a scope, every helper below calls the
function directly on the caller thread, which is exactly the legacy path.

Task 5 (attempt tickets) reads the same context to bind tickets to the
owning transaction; task 6 (shared chapter scheduler) establishes one scope
per logical clip transaction with the chapter's ``(run, chapter)`` owner key.

The leaf helpers never place their *calling* routine on a lane: callers are
transaction runners or coordination workers, and only the one leaf operation
they name is routed.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_resources: ContextVar = ContextVar("renderer_pipeline_resources", default=None)
_owner: ContextVar = ContextVar("renderer_resource_owner", default=(None, None))


def current_resources():
    """The pipeline ``RenderResources`` in scope, or None for direct execution."""
    return _resources.get()


def current_owner():
    """The ``(run/submission, chapter)`` key used for lane fairness."""
    return _owner.get()


@contextmanager
def resource_scope(resources, owner=(None, None)):
    """Route leaf operations of this transaction subtree through ``resources``.

    Nested scopes replace both values for their extent; leaving the scope
    restores the previous ones, so sibling transactions never inherit another
    transaction's runtime or owner key.
    """
    resources_token = _resources.set(resources)
    owner_token = _owner.set(owner)
    try:
        yield
    finally:
        _resources.reset(resources_token)
        _owner.reset(owner_token)


def run_preparation(function, *args, **kwargs):
    """Run one audio-preparation leaf (normalize_clip, _audio_metrics)."""
    resources = _resources.get()
    if resources is None:
        return function(*args, **kwargs)
    return resources.run_preparation(function, *args, owner=_owner.get(), **kwargs)


def run_recognition(function, *args, **kwargs):
    """Run one recognition leaf on the single serialized ASR dispatch lane."""
    resources = _resources.get()
    if resources is None:
        return function(*args, **kwargs)
    return resources.run_recognition(function, *args, owner=_owner.get(), **kwargs)


def run_review_leaf(function, *args, **kwargs):
    """Run one actual network review request on the shared review lane."""
    resources = _resources.get()
    if resources is None:
        return function(*args, **kwargs)
    return resources.run_review_leaf(function, *args, owner=_owner.get(), **kwargs)
