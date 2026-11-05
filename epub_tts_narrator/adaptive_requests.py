"""Adaptive request partitioning driven by exact size and capacity signals."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Protocol, TypeAlias, TypeVar

from .prompts import PROMPT_VERSION
from .provider import OutputTruncatedError, ProviderError, ResponseCapacityError


JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")
_TRUNCATED_OUTPUT = "The model output was truncated. Increase the output token limit or reduce the input."
_MARKER_BYTES = 2048


class _ProviderIdentity(Protocol):
    @property
    def identity(self) -> Mapping[str, JsonValue]: ...


class _RequestCache(Protocol[P, R]):
    max_request_chars: int
    cache_dir: str | Path
    client: _ProviderIdentity

    def cached(self, system: str, payload: P,
               validate: Callable[[JsonObject], R]) -> tuple[bool, R | None]: ...


def request_size(system: str, payload: P) -> int:
    """Return the character count used by CachedRequests' request budget."""
    return len(system) + len(json.dumps(payload, ensure_ascii=False))


def capacity_error(error: BaseException) -> bool:
    """Recognize only explicit provider capacity failures, never message guesses."""
    if isinstance(error, (OutputTruncatedError, ResponseCapacityError)):
        return True
    if isinstance(error, ProviderError) and error.code == "context_length_exceeded":
        return True
    return (type(error) is ProviderError and error.status_code is None and error.code is None
            and str(error) == _TRUNCATED_OUTPUT)


def _fingerprint(value: P) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _split_marker(requests: _RequestCache[P, R], system: str,
                  payload: P) -> tuple[Path, JsonObject, bool]:
    key = _fingerprint({"provider": requests.client.identity, "prompt_version": PROMPT_VERSION,
                        "system": system, "payload": payload})
    marker: JsonObject = {"version": 1, "reason": "provider_capacity",
                          "request_fingerprint": _fingerprint(payload),
                          "request_cache_fingerprint": key}
    directory = Path(requests.cache_dir) / "adaptive-subdivisions"
    path = directory / f"{key}.json"
    try:
        if directory.is_symlink() or path.is_symlink() or not path.is_file():
            return path, marker, False
        with path.open("rb") as stream:
            raw = stream.read(_MARKER_BYTES + 1)
        saved = json.loads(raw.decode("utf-8")) if len(raw) <= _MARKER_BYTES else None
        valid = (isinstance(saved, dict) and set(saved) == set(marker)
                 and type(saved["version"]) is int and saved == marker)
        return path, marker, valid
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return path, marker, False


def _write_marker(path: Path, marker: JsonObject) -> None:
    """Atomically persist hashed failure metadata without following marker symlinks."""
    if path.parent.is_symlink() or path.is_symlink():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        return
    descriptor, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(marker, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run_partitioned(
    items: Sequence[T],
    build: Callable[[list[T]], tuple[str, P]],
    execute: Callable[[str, P], R],
    *,
    fits: Callable[[str, P], bool] | None = None,
    progress: Callable[[str], None] = lambda message: None,
    label: str = "Adaptive request",
    requests: _RequestCache[P, R] | None = None,
    cache_validator: Callable[[str, P, JsonObject], R] | None = None,
    observe: Callable[[JsonObject], None] = lambda event: None,
) -> list[tuple[list[T], R]]:
    """Run complete ordered leaves, subdividing only on exact capacity signals.

    Children execute sequentially from left to right. A full binary tree has at
    most ``2 * len(items) - 1`` nodes when every leaf is indivisible. Cached
    failure markers skip known failed parents. An optional cache validator may
    accept a newer parent success without allowing a replacement provider call.
    """
    if requests is None and fits is None:
        raise ValueError("fits(system, payload) is required when requests is not supplied")

    def run_part(current: list[T]) -> list[tuple[list[T], R]]:
        if not current:
            return []
        system, payload = build(current)
        path, marker, replay = ((None, None, False) if requests is None or len(current) == 1
                                else _split_marker(requests, system, payload))
        if replay and requests is not None and cache_validator is not None:
            found, accepted = requests.cached(
                system, payload, lambda response: cache_validator(system, payload, response))
            if found:
                if accepted is None:
                    raise ValueError("requests.cached returned found=True without a validated result")
                return [(current, accepted)]
        within_budget = ((requests is None or request_size(system, payload) <= requests.max_request_chars)
                         and (fits is None or fits(system, payload)))

        def subdivide(reason: str, detail: str) -> list[tuple[list[T], R]]:
            observe({"reason": reason, "target_count": len(current),
                     "request_fingerprint": _fingerprint(payload), "replayed_hint": replay})
            progress(f"{label}: subdividing {len(current)} items {detail}")
            midpoint = len(current) // 2
            return run_part(current[:midpoint]) + run_part(current[midpoint:])

        if not within_budget:
            if len(current) == 1:
                if requests is not None:
                    raise ValueError(
                        f"{label}: items contains an indivisible request with request_size(system, payload)="
                        f"{request_size(system, payload)}, exceeding requests.max_request_chars="
                        f"{requests.max_request_chars}; no item content was dropped"
                    )
                raise ValueError(
                    f"{label}: items contains an indivisible request rejected by fits(system, payload); "
                    "reduce its system/payload request_size or increase the provider capacity"
                )
            return subdivide("input_budget", "to fit the request budget")
        if replay:
            return subdivide("provider_capacity", "using a cached capacity failure")
        try:
            return [(current, execute(system, payload))]
        except ProviderError as error:
            if not capacity_error(error):
                raise
            if len(current) == 1:
                raise ValueError(
                    f"{label}: items contains an indivisible one-item partition after an explicit provider "
                    "capacity failure; reduce its system/payload request_size or increase provider capacity"
                ) from error
            if requests is not None and path is not None and marker is not None:
                _write_marker(path, marker)
            return subdivide("provider_capacity", "after a provider capacity failure")

    return run_part(list(items))
