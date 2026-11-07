"""Per-request sampling for engines that decode concurrent requests.

``serve`` normally applies a request's sampling by setting attributes on its
single-request engine around the call. An engine that decodes several
requests at once cannot share those attributes, so ``serve`` passes each
request's values through this context variable instead.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_SAMPLING: ContextVar[dict | None] = ContextVar("audiobook_request_sampling", default=None)


@contextmanager
def request_sampling(values):
    token = _SAMPLING.set(dict(values))
    try:
        yield
    finally:
        _SAMPLING.reset(token)


def current_sampling():
    """This request's sampling values, or None outside a concurrent request."""
    return _SAMPLING.get()
