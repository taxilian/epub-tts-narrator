"""Curses-free rendered-clip indexing and take presentation for review-ui."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from .speaker_decisions import _display


class ClipRef(TypedDict):
    chunk_id: str
    request_hash: str


class ChapterIndex(TypedDict):
    status: str
    work_directory: Path
    segments: dict[str, list[ClipRef]]
    error: NotRequired[str]


class ChapterFailure(TypedDict):
    """One persisted chapter failure from render-work/status.json.

    ``segment_ids`` are the renderer's own stable segment IDs; empty means the
    failure was only recorded at chapter level and no exact passage linkage
    exists. ``chunk_id``/``request_hash`` are optional renderer linkage.
    """
    number: int
    error_type: str
    error: str
    segment_ids: list[str]
    chunk_id: NotRequired[str]
    request_hash: NotRequired[str]


class TakeRow(TypedDict):
    attempt: int
    accepted: bool
    audio_path: str | Path | None
    passed: bool
    reasons: list[str]
    text_confirmation_method: str | None
    duration_seconds: float
    transcript: str
    echo: bool
    eligible: bool
    ineligible_reason: str | None


def _flat(value: str | BaseException) -> str:
    return " ".join(_display(value).split())


def playback_notice(outcome: bool | BaseException) -> str:
    """Map play_sample's completion result or supported failure to one safe notice."""
    if outcome is True:
        return "Playback finished"
    if outcome is False:
        return "Playback interrupted"
    return "Playback failed: " + _flat(outcome)


def format_take(row: TakeRow) -> str:
    """Describe one attempt without displaying its transcript or book text."""
    attempt = row.get("attempt", "?")
    accepted = "*" if row.get("accepted") else " "
    status = "passed" if row.get("passed") is True else "failed"
    reasons = row.get("reasons") or []
    reason_text = ",".join(str(reason) for reason in reasons)
    duration = row.get("duration_seconds")
    duration_text = f"{duration:.2f}s" if type(duration) in {int, float} else "unknown"
    confirmation = row.get("text_confirmation_method") or "none"
    echo = "yes" if row.get("echo") else "no"
    eligible = "yes" if row.get("eligible") else "no"
    result = (f"{accepted} Take {attempt}  {status}  duration={duration_text}  "
              f"reasons={reason_text or 'none'}  confirmation={confirmation}  "
              f"echo={echo}  eligible={eligible}")
    if not row.get("eligible") and row.get("ineligible_reason"):
        result += "  ineligible=" + str(row["ineligible_reason"])
    return result


def archive_checkpoint(work_directory: str | Path, chapter_number: int, *, token: str | int = "") -> Path | None:
    """Supersede one completed-chapter checkpoint after a take selection changes."""
    path = Path(work_directory) / "completed-chapters" / f"chapter-{chapter_number:04d}.json"
    if not path.exists():
        return None
    suffix = hashlib.sha256(path.read_bytes() + str(token).encode("utf-8")).hexdigest()[:12]
    destination = path.with_name(path.name + ".superseded-" + suffix)
    path.rename(destination)
    return destination


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Rendered record must be a JSON object")
    return value


def load_failed_chapters(work_directory: str | Path) -> dict[int, ChapterFailure]:
    """Parse failed_chapters persisted by ChaptersRenderFailed in status.json.

    Missing, malformed, or non-aggregate status records simply carry no
    failures; the review UI then shows the ordinary not-rendered state.
    """
    try:
        record = _read_json(Path(work_directory) / "status.json")
    except (OSError, ValueError):
        return {}
    if record.get("error_type") != "ChaptersRenderFailed":
        return {}
    entries = record.get("failed_chapters")
    if not isinstance(entries, list):
        return {}
    failures: dict[int, ChapterFailure] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        number, error_type, error = entry.get("number"), entry.get("error_type"), entry.get("error")
        if type(number) is not int or not isinstance(error_type, str) or not isinstance(error, str):
            continue
        failure: ChapterFailure = {"number": number, "error_type": error_type,
                                   "error": error, "segment_ids": []}
        segment_ids = entry.get("segment_ids")
        if isinstance(segment_ids, list):
            failure["segment_ids"] = [segment_id for segment_id in segment_ids
                                      if isinstance(segment_id, str)]
        if isinstance(entry.get("chunk_id"), str):
            failure["chunk_id"] = entry["chunk_id"]
        if isinstance(entry.get("request_hash"), str):
            failure["request_hash"] = entry["request_hash"]
        failures[number] = failure
    return failures


def _acceptance_clips(acceptance: list[Any] | None):
    """Yield (chunk_id, request_hash) for ordinary and simultaneous clip records."""
    for row in acceptance if isinstance(acceptance, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("chunk_id"), str):
            continue
        lanes = row.get("speaker_lanes")
        records = lanes if isinstance(lanes, list) else [row]
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("request_hash"), str):
                yield row["chunk_id"], record["request_hash"]


def _chunk_segments(chunk: Any) -> list[str]:
    turns = chunk.get("paragraph_turns") if isinstance(chunk, dict) else None
    rows = turns if isinstance(turns, list) else [chunk]
    result = []
    for row in rows:
        segment = row.get("segment") if isinstance(row, dict) else None
        segment_id = segment.get("id") if isinstance(segment, dict) else None
        if isinstance(segment_id, str) and segment_id not in result:
            result.append(segment_id)
    return result


def _add_clip(index: defaultdict[str, list[ClipRef]], segment_ids: list[str], ref: ClipRef) -> None:
    for segment_id in segment_ids:
        if ref not in index[segment_id]:
            index[segment_id].append(ref)


def _fallback_segments(work_directory: str | Path, acceptance: list[Any] | None) -> defaultdict[str, list[ClipRef]]:
    """Recover segment membership only from clips named by this chapter's checkpoint."""
    index: defaultdict[str, list[ClipRef]] = defaultdict(list)
    for chunk_id, request_hash in _acceptance_clips(acceptance):
        try:
            record = _read_json(Path(work_directory) / "clips" / request_hash / "accepted.json")
        except (OSError, ValueError):
            continue
        request = record.get("request")
        chunk = request.get("chunk") if isinstance(request, dict) else None
        _add_clip(index, _chunk_segments(chunk), {"chunk_id": chunk_id, "request_hash": request_hash})
    return index


class RenderIndex:
    """Lazily cache chapter-local clip indexes; never scan the book's clip directory."""

    def __init__(self, directory: str | Path, *, config: dict[str, Any] | None = None):
        self.directory = Path(directory).resolve()
        self.config = config
        self._bundle: dict[str, Any] | None = None
        self._chapters: dict[int, ChapterIndex] = {}
        self._failures: dict[int, ChapterFailure] | None = None

    def _settings(self) -> dict[str, Any]:
        from .renderer.config import DEFAULTS
        settings: dict[str, Any] = deepcopy(DEFAULTS)
        if isinstance(self.config, dict):
            for section, values in self.config.items():
                if section in settings and isinstance(values, dict):
                    settings[section].update(deepcopy(values))
        return settings

    def _load(self, chapter_number: int) -> ChapterIndex:
        from .renderer.book_context import resolve_book_config
        from .renderer.bundle import load_bundle, make_plan
        if self._bundle is None:
            self._bundle = load_bundle(self.directory)
        bundle = self._bundle
        settings = self._settings()
        settings["render"]["chapter_numbers"] = [chapter_number]
        resolved: dict[str, Any] = resolve_book_config(bundle, settings)
        work = Path(resolved["render"]["work_dir"])
        if not work.is_dir():
            return {"status": "render_work_missing", "work_directory": work, "segments": {}}
        checkpoint = work / "completed-chapters" / f"chapter-{chapter_number:04d}.json"
        if not checkpoint.is_file():
            return {"status": "checkpoint_missing", "work_directory": work, "segments": {}}
        acceptance = _read_json(checkpoint).get("acceptance")
        try:
            plan: dict[str, Any] = make_plan(bundle, resolved)
            planned: list[dict[str, Any]] = plan["chapters"][0]["chunks"]
        except (OSError, ValueError):
            # The recomputed plan is an optimization; tagging/settings can move
            # on after a render. The checkpoint's clip records stay authoritative.
            planned = []
        planned_ids = {chunk["chunk_id"] for chunk in planned if not chunk.get("silent_control")}
        saved_ids = {chunk_id for chunk_id, _ in _acceptance_clips(acceptance)}
        index: defaultdict[str, list[ClipRef]]
        if planned_ids == saved_ids and saved_ids:
            hashes: defaultdict[str, list[str]] = defaultdict(list)
            for chunk_id, request_hash in _acceptance_clips(acceptance):
                hashes[chunk_id].append(request_hash)
            index = defaultdict(list)
            for chunk in planned:
                for request_hash in hashes.get(chunk["chunk_id"], []):
                    ref: ClipRef = {"chunk_id": chunk["chunk_id"], "request_hash": request_hash}
                    _add_clip(index, _chunk_segments(chunk), ref)
        else:
            index = _fallback_segments(work, acceptance)
        return {"status": "indexed", "work_directory": work, "segments": dict(index)}

    def chapter(self, chapter_number: int) -> ChapterIndex:
        """Return one cached chapter index, computing it only on first use."""
        if chapter_number not in self._chapters:
            self._chapters[chapter_number] = self._load(chapter_number)
        return self._chapters[chapter_number]

    def failures(self) -> dict[int, ChapterFailure]:
        """Return the cached book-level failure map from render-work/status.json."""
        if self._failures is None:
            self._failures = self._load_failures()
        return self._failures

    def _load_failures(self) -> dict[int, ChapterFailure]:
        from .renderer.book_context import resolve_book_config
        from .renderer.bundle import load_bundle
        try:
            if self._bundle is None:
                self._bundle = load_bundle(self.directory)
            resolved: dict[str, Any] = resolve_book_config(self._bundle, self._settings())
        except (ValueError, OSError):
            return {}
        work = Path(resolved["render"]["work_dir"])
        if not work.is_dir():
            return {}
        return load_failed_chapters(work)

    def reset(self) -> None:
        """Drop saved-document, chapter, and failure assumptions after a publication."""
        self._bundle = None
        self._chapters.clear()
        self._failures = None
