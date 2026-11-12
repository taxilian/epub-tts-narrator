"""Which job holds each slot on each render node, for the run shell (read-only, in-process).

A render pool registers its nodes when it is built: every TTS worker
(``render_pool.workers``, ``role = "speech"``, one slot per ``concurrency``)
and every recognizer (the controller's own and any remote ``serve`` worker
with the ``recognition`` role, ``role = "verify"``). While a take generates
or is transcribed, the pool records the job on that node: its book, chapter
and block, which ``ChapterProgress.task`` binds in a context variable for
everything that runs for the block, including work handed to lane threads.

Only names, counts, IDs and timings are kept: no text, URLs or keys.
Observing never creates or changes anything, and with no pool (or without
the run shell) nothing reads it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import itertools
import threading
import time


_JOB = ContextVar("render_node_job", default=None)


@contextmanager
def job(book=None, chapter=None, block=None):
    """Bind the block whose work runs inside this context (and in work submitted from it)."""
    token = _JOB.set({"book": book, "chapter": chapter, "block": block})
    try:
        yield
    finally:
        _JOB.reset(token)


def current_job():
    return dict(_JOB.get() or {})


class NodeActivity:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._nodes = {}        # (name, role) -> {"name", "role", "slots", "local", "owner"}
        self._jobs = {}         # token -> {"name", "role", "book", "chapter", "block", "started"}
        self._tokens = itertools.count(1)
        self.chapter_capacity = None

    def register(self, owner, name, role, slots, *, local, probe=None):
        """``probe()`` (optional) returns the node's own report of its work, polled by ``probe_nodes``."""
        with self._lock:
            self._nodes[(name, role)] = {"name": name, "role": role, "slots": int(slots), "local": bool(local),
                                         "owner": owner, "probe": probe, "work": None, "probed": None}

    def unregister(self, owner):
        """Forget the nodes one pool registered (its close)."""
        with self._lock:
            for key in [key for key, node in self._nodes.items() if node["owner"] is owner]:
                del self._nodes[key]
            # Jobs still recorded for a closed pool's nodes go with them.
            live = {key[0] + "\0" + key[1] for key in self._nodes}
            self._jobs = {token: item for token, item in self._jobs.items()
                          if item["name"] + "\0" + item["role"] in live}

    def probe_nodes(self, max_age=2.0):
        """Ask each node that can say what it is doing (rows decoding, prefilling, waiting).

        Called by an observer on its own thread, at most every ``max_age``
        seconds per node; a remote node is a short read-only health request. A
        node that cannot be reached reports nothing until it can.
        """
        now = self._clock()
        with self._lock:
            due = [(key, node["probe"]) for key, node in self._nodes.items()
                   if callable(node["probe"]) and (node["probed"] is None or now - node["probed"] >= max_age)]
            for key, _ in due:
                self._nodes[key]["probed"] = now
        for key, probe in due:
            try:
                work = probe()
            except Exception:
                work = None
            with self._lock:
                if key in self._nodes:
                    self._nodes[key]["work"] = dict(work) if isinstance(work, dict) else None

    def set_chapter_capacity(self, count):
        with self._lock:
            self.chapter_capacity = count

    @contextmanager
    def busy(self, name, role, job=None):
        """Record a block's job (default: the current one) on ``name`` while the body runs."""
        token = next(self._tokens)
        with self._lock:
            self._jobs[token] = {"name": name, "role": role, **(current_job() if job is None else job),
                                 "started": self._clock()}
        try:
            yield
        finally:
            with self._lock:
                self._jobs.pop(token, None)

    def snapshot(self):
        """``[{"name", "role", "slots", "local", "jobs": [{"book", "chapter", "block", "elapsed"}]}]``."""
        now = self._clock()
        with self._lock:
            nodes = [dict(node, jobs=[]) for node in self._nodes.values()]
            index = {(node["name"], node["role"]): node for node in nodes}
            for item in sorted(self._jobs.values(), key=lambda value: value["started"]):
                node = index.get((item["name"], item["role"]))
                if node is not None:
                    node["jobs"].append({"book": item.get("book"), "chapter": item.get("chapter"),
                                         "block": item.get("block"), "elapsed": now - item["started"]})
        for node in nodes:
            for private in ("owner", "probe", "probed"):
                node.pop(private, None)
        order = {"speech": 0, "verify": 1}
        return sorted(nodes, key=lambda node: (not node["local"], node["name"], order.get(node["role"], 2)))


ACTIVITY = NodeActivity()
