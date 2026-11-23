"""Which job holds each slot on each render node, for the run shell (read-only, in-process).

A render pool registers its nodes when it is built: every TTS worker
(``render_pool.workers``, ``role = "speech"``, one slot per ``concurrency``)
and every recognizer (the controller's own and any remote ``serve`` worker
with the ``recognition`` role, ``role = "verify"``). While a take generates
or is transcribed, the pool records the job on that node: its book, chapter
and block, which ``ChapterProgress.task`` binds in a context variable for
everything that runs for the block, including work handed to lane threads.

Each finished generation also records how many seconds of audio a node made
(``made``), so the run shell can show each node's speed over the last
``SPEED_WINDOW`` seconds: audio frames per second (Drama 3's codec makes
``AUDIO_FRAMES_PER_SECOND`` frames for each second of audio) and how many
times faster than real time.

A pool may also let a person turn a speech node off and on (``switch``,
the run shell's ``o``): each node then reports its ``state`` (``on``,
``away``, ``turning off``, ``shutting down``, ``off``, ``turning on``).

Only names, counts, IDs and timings are kept: no text, URLs or keys.
Observing never creates or changes anything, and with no pool (or without
the run shell) nothing reads it.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import itertools
import threading
import time


_JOB = ContextVar("render_node_job", default=None)
# Drama 3's codec: 44.1 kHz audio from 2048-sample frames.
AUDIO_FRAMES_PER_SECOND = 44100 / 2048
SPEED_WINDOW = 300.0  # seconds of history in the rolling speed


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
        self._made = deque()    # (time, name, audio seconds) of finished generations, newest last
        self.chapter_capacity = None

    def register(self, owner, name, role, slots, *, local, probe=None, switch=None, state=None):
        """``probe()`` (optional) returns the node's own report of its work, polled by ``probe_nodes``.

        ``switch(on, notify)`` (optional) turns the node off or on and returns
        what happens in plain words; ``state()`` says where it is now.
        """
        with self._lock:
            self._nodes[(name, role)] = {"name": name, "role": role, "slots": int(slots), "local": bool(local),
                                         "owner": owner, "probe": probe, "work": None, "probed": None,
                                         "registered": self._clock(), "switch": switch, "state": state}

    def switch(self, name, on, notify=None):
        """Turn speech node ``name`` off (``on=False``) or on; returns what happens. ValueError if it cannot be."""
        with self._lock:
            node = self._nodes.get((name, "speech"))
            switch = node and node["switch"]
        if not callable(switch):
            raise ValueError(f"{name} is not a speech node this render can turn on or off")
        return switch(on, notify)

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

    def made(self, name, audio_seconds):
        """Record ``audio_seconds`` of audio a generation on ``name`` just finished (ignored if not a number)."""
        if not isinstance(audio_seconds, (int, float)) or isinstance(audio_seconds, bool) or audio_seconds <= 0:
            return
        now = self._clock()
        with self._lock:
            self._made.append((now, name, float(audio_seconds)))
            while self._made and now - self._made[0][0] > SPEED_WINDOW:
                self._made.popleft()

    def _speed(self, made, since, now):
        """``{"audio_seconds", "span", "realtime", "fps"}`` for audio made since ``since``, or ``None``."""
        span = min(SPEED_WINDOW, now - since)
        if span <= 0:
            return None
        seconds = sum(amount for when, amount in made if now - when <= span)
        realtime = seconds / span
        return {"audio_seconds": round(seconds, 1), "span": round(span, 1), "realtime": round(realtime, 2),
                "fps": round(realtime * AUDIO_FRAMES_PER_SECOND, 1)}

    def speeds(self):
        """``{"nodes": {name: speed}, "overall": speed}`` over the last ``SPEED_WINDOW`` (each ``None`` until known)."""
        now = self._clock()
        with self._lock:
            made = list(self._made)
            started = {node["name"]: node["registered"] for node in self._nodes.values() if node["role"] == "speech"}
        nodes = {name: self._speed([(when, amount) for when, who, amount in made if who == name], since, now)
                 for name, since in started.items()}
        overall = self._speed([(when, amount) for when, _, amount in made], min(started.values()), now) if started else None
        return {"nodes": nodes, "overall": overall}

    def snapshot(self):
        """``[{"name", "role", "slots", "local", "state", "switchable", "jobs": [{"book", "chapter", "block", "elapsed"}]}]``."""
        now = self._clock()
        with self._lock:
            nodes = [dict(node, jobs=[]) for node in self._nodes.values()]
            index = {(node["name"], node["role"]): node for node in nodes}
            for item in sorted(self._jobs.values(), key=lambda value: value["started"]):
                node = index.get((item["name"], item["role"]))
                if node is not None:
                    node["jobs"].append({"book": item.get("book"), "chapter": item.get("chapter"),
                                         "block": item.get("block"), "elapsed": now - item["started"]})
        speeds = self.speeds()["nodes"]
        for node in nodes:
            state = node.pop("state", None)
            try:
                node["state"] = state() if callable(state) else "on"
            except Exception:
                node["state"] = "on"
            node["switchable"] = callable(node.pop("switch", None))
            for private in ("owner", "probe", "probed", "registered"):
                node.pop(private, None)
            if node["role"] == "speech":
                node["speed"] = speeds.get(node["name"])
        order = {"speech": 0, "verify": 1}
        return sorted(nodes, key=lambda node: (not node["local"], node["name"], order.get(node["role"], 2)))


ACTIVITY = NodeActivity()
