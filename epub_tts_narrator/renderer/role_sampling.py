"""Separate Drama 3 sampling for character speech (``[dialogue_sampling]``).

Narration keeps the ``[engine]`` sampling. A request whose every passage is a
character's own speech (segment kind ``dialogue``, ``thought`` or ``letter``)
may use its own ``temperature`` and ``top_p``. Only values that differ from the
``[engine]`` values change anything: such a request's engine identity carries
them, so its cache identity differs from the same request at engine sampling,
and its synthesis runs inside ``request_sampling`` so every engine path (the
local MLX loop, the batched decoder, ``serve`` workers) applies them.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math

from .config import SAMPLING_DEFAULTS

CHARACTER_KINDS = frozenset({"dialogue", "thought", "letter"})
KEYS = ("temperature", "top_p")
LIMITS = {"temperature": 2, "top_p": 1}


def validate(section, engine_type):
    """Check a loaded ``[dialogue_sampling]`` table; unset values stay None."""
    for key in KEYS:
        value = section.get(key)
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= LIMITS[key]:
            raise ValueError(f"dialogue_sampling.{key} must be positive and at most {LIMITS[key]}")
        if engine_type != "drama3_mlx":
            raise ValueError("dialogue_sampling requires engine.type = drama3_mlx")


def dialogue_overrides(config):
    """The character-speech sampling values that differ from ``[engine]``."""
    section = config.get("dialogue_sampling") or {}
    engine = config.get("engine") or {}
    overrides = {}
    for key in KEYS:
        value = section.get(key)
        if value is not None and float(value) != float(engine.get(key, SAMPLING_DEFAULTS[key])):
            overrides[key] = float(value)
    return overrides


def is_character_speech(chunk):
    turns = chunk.get("paragraph_turns") or [chunk]
    return all(isinstance(turn.get("segment"), dict)
               and turn["segment"].get("kind") in CHARACTER_KINDS for turn in turns)


def chunk_overrides(chunk, config):
    """This chunk's sampling overrides; empty for narration and mixed requests."""
    overrides = dialogue_overrides(config)
    return overrides if overrides and is_character_speech(chunk) else {}


def with_overrides(engine_identity, overrides):
    """An engine identity that records the sampling this request is generated with."""
    if not overrides:
        return engine_identity
    identity = deepcopy(engine_identity)
    if isinstance(identity.get("sampling"), dict):
        identity["sampling"].update(overrides)  # render pool / serve client identities
    else:
        identity.update(overrides)  # a local Drama 3 engine's own identity
    return identity


@contextmanager
def chunk_sampling(chunk, config):
    """Generate this chunk's takes with its sampling (a no-op for narration)."""
    overrides = chunk_overrides(chunk, config)
    if not overrides:
        yield
        return
    from .request_sampling import request_sampling
    with request_sampling(overrides):
        yield


def effective(engine):
    """Sampling for one synthesis call: this request's values over the engine's own."""
    from .request_sampling import current_sampling
    requested = current_sampling() or {}
    return {key: requested.get(key, getattr(engine, key))
            for key in ("temperature", "top_p", "top_k")}
