"""Takes from other chapters that say a chosen pronunciation right, for chapter history.

Drama 3 says a name the way its history says it (``continuity``): with a
correct example in history a narrator says a chosen name right far more
often than with none. Within a chapter the latest history
slot already prefers an earlier take saying the line's chosen readings. A
chapter's first line using a reading has no such take, and its retries keep
the same history, so it misread again and again. This index supplies one:
the same voice's accepted take from another chapter of the book whose saved
transcript says the reading in its own shape (``name_shape.heard_right``).

The index reads accepted records once per render work (only those carrying a
CMU reading) and is checked again before an example is used, so a take
retired since (marked bad, misread) is never heard. Only takes that may be
chapter history at all are indexed: ordinary verified generations of their
exact text, not repairs, joins, choices, stalls or takes that went silent.
The nearest earlier take in book order is chosen, else the nearest later one.
"""

from __future__ import annotations

from pathlib import Path
import threading

_INDEXES = {}
_LOCK = threading.Lock()


def for_work(work):
    """The shared example index of one render work (built on first use)."""
    key = str(Path(work).resolve())
    with _LOCK:
        if key not in _INDEXES:
            _INDEXES[key] = ReadingExamples(work)
        return _INDEXES[key]


def _sequence(request):
    from .continuity import _segments
    chunk = request.get("chunk")
    if not isinstance(chunk, dict):
        return None
    numbers = [segment.get("sequence") for segment in _segments(chunk)]
    return min(numbers) if numbers and all(type(number) is int for number in numbers) else None


def _chapter(request):
    from .continuity import _segments
    chunk = request.get("chunk")
    if not isinstance(chunk, dict):
        return None
    return next(iter(_segments(chunk)), {}).get("chapter_id")


class ReadingExamples:
    def __init__(self, work):
        self.work = Path(work)
        self._candidates = None
        self._lock = threading.Lock()

    @staticmethod
    def _candidate(path, record):
        """An index row for one accepted record that may be an example, or None."""
        from .continuity import _NOT_HISTORY, _readings_heard_right, request_speakers
        from .stall_recovery import recovered
        if not isinstance(record, dict) or not isinstance(record.get("request"), dict):
            return None
        metadata, report = record.get("engine_metadata") or {}, record.get("report") or {}
        marker = metadata.get("continuity_codes")
        if (report.get("passed") is not True or not isinstance(marker, dict) or _NOT_HISTORY.intersection(metadata)
                or recovered(report) or metadata.get("finish_reason") == "silence"
                or "performer" in record["request"] or not isinstance(marker.get("file"), str)):
            return None
        readings = _readings_heard_right(record)
        sequence = _sequence(record["request"])
        if not readings or sequence is None:
            return None
        try:
            voices = sorted(request_speakers(record["request"]))
        except (KeyError, TypeError, ValueError):
            return None
        return {"accepted": Path(path), "record_audio": record.get("audio_sha256"),
                "request_hash": record.get("request_hash"), "marker": marker,
                "readings": readings, "voices": voices, "sequence": sequence,
                "chapter": _chapter(record["request"])}

    def _build(self):
        from .common import read_json
        candidates = []
        clips = self.work / "clips"
        if not clips.is_dir() or clips.is_symlink():
            return candidates
        for path in sorted([*clips.glob("*/accepted.json"), *clips.glob("*/parts/*/accepted.json")]):
            try:
                if path.is_symlink() or b'"cmu_arpabet"' not in path.read_bytes():
                    continue
                candidate = self._candidate(path, read_json(path))
            except (ValueError, OSError):
                continue
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def candidates(self):
        with self._lock:
            if self._candidates is None:
                self._candidates = self._build()
            return list(self._candidates)

    def add(self, audio_path, accepted):
        """Index a take this render just accepted into chapter history."""
        candidate = self._candidate(Path(audio_path).parent / "accepted.json", accepted)
        if candidate is None:
            return
        with self._lock:
            if self._candidates is None:
                return  # built from disk on first use, which includes it
            if not any(item["accepted"] == candidate["accepted"] and item["record_audio"] == candidate["record_audio"]
                       for item in self._candidates):
                self._candidates.append(candidate)

    def example(self, speakers, readings, *, sequence, chapter, voices_by_asset, wanted=()):
        """A history entry from another chapter saying the most of ``readings`` right, or None.

        Among those, one also saying more of ``wanted`` (all the line's chosen
        readings) right comes first: a line using two chosen names needs a
        turn saying both, since a turn saying only the missing one would
        trade away the turn saying the other.
        """
        from .common import read_json
        from .continuity_codes import load_codes
        from .common import digest
        readings = set(readings)
        wanted = set(wanted) | readings
        if not readings or sequence is None:
            return None
        scored = []
        for item in self.candidates():
            covered = len(readings & item["readings"])
            if (not covered or item["chapter"] == chapter or not set(item["voices"]) <= set(speakers)
                    or not all(asset in voices_by_asset for asset in item["marker"].get("speakers", []))):
                continue
            # Nearest earlier take in book order first, then the nearest later one.
            before = item["sequence"] < sequence
            scored.append((covered, len(wanted & item["readings"]), before,
                           item["sequence"] if before else -item["sequence"], item))
        for *_, item in sorted(scored, key=lambda row: row[:4], reverse=True):
            try:
                current = read_json(item["accepted"])
                codes = item["accepted"].parent / item["marker"]["file"]
                if (not isinstance(current, dict) or current.get("audio_sha256") != item["record_audio"]
                        or (current.get("engine_metadata") or {}).get("continuity_codes") != item["marker"]):
                    continue  # retired or replaced since the index was read
                load_codes(codes, item["marker"])
                # An example must say its names as chosen, by its phonemes
                # (older takes were judged only by the recognizer's spelling).
                from .name_phonemes import judge
                verdicts = judge(current, item["accepted"].parent)
                if verdicts and not all(verdict["ok"] for verdict in verdicts):
                    continue
            except (ValueError, OSError):
                continue
            marker = item["marker"]
            return {"path": str(codes), "marker": dict(marker), "pinnable": False, "voices": list(item["voices"]),
                    "position": None, "readings": frozenset(item["readings"]), "example": True,
                    "identity": {"codes_sha256": marker["sha256"], "text_sha256": digest(marker["text"]),
                                 "speakers": list(marker["speakers"]), "request_hash": item["request_hash"],
                                 "audio_sha256": item["record_audio"]}}
        return None
