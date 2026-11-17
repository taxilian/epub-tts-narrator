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

The choice depends only on what is saved, never on when the index was read:
an example line usually has several passing takes on disk (each rendering of
it under another history keeps its own), and listing order once decided
between them, so a fresh render chose differently from the one that made a
take, and strict history reuse rendered every later line of the voice again
with no source change. A line whose saved takes passed keeps hearing the
example they heard (``prefer``: those takes' heard history, the chapter
checkpoint's take first) while its sound and codes are still on disk, even
when the example has since been retired or judged a misreading, unless a
person marked it bad: the line's take was made after exactly that audio and
already said the name right, which is all an example is for, and a new
example would make strict history reuse render it and every later line of
its voice again. A line needing a new take hears only valid examples, equal
ones ordered by request hash.

A character keeps their voice through a series, so when the book has no such
take, the same voice's take from an earlier book of the series is heard
(nearest book first, ``analysis_directory.continuation_directories``; each
book's render work at its default ``BOOK/render-work``). It must be the very
same voice recording (asset hash) and say the name right by its phonemes,
like every example.
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


def note_part(folder):
    """Register a split-passage part folder (``clips/HASH/parts/PART``) the render is making with every index that read its book."""
    folder = Path(folder)
    if folder.parent.name != "parts" or folder.parent.parent.parent.name != "clips":
        return
    work = folder.parent.parent.parent.parent
    with _LOCK:
        indexes = list(_INDEXES.values())
    for index in indexes:
        index._note_part(work, folder)


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
        self._series = None
        self._part_folders = {}
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

    def _earlier_works(self):
        """Render works of the series' earlier books, nearest first (``[]`` for a first book)."""
        book = self.work.parent
        if not (book / "manifest.json").is_file():
            return []  # a render work outside its book has no series to look in
        try:
            from ..analysis_directory import continuation_directories
            earlier = continuation_directories(book)
        except (ValueError, OSError, KeyError, TypeError):
            return []
        return [work for work in (Path(directory) / "render-work" for directory in earlier)
                if (work / "clips").is_dir() and not (work / "clips").is_symlink()]

    def _build(self):
        from .common import read_json
        candidates = []
        for distance, work in enumerate([self.work, *self._earlier_works()]):
            clips = work / "clips"
            if not clips.is_dir() or clips.is_symlink():
                continue
            for path in sorted([*clips.glob("*/accepted.json"), *clips.glob("*/parts/*/accepted.json")]):
                try:
                    if path.is_symlink() or b'"cmu_arpabet"' not in path.read_bytes():
                        continue
                    candidate = self._candidate(path, read_json(path))
                except (ValueError, OSError):
                    continue
                if candidate is not None:
                    # How many books back it was spoken (0: this book).
                    candidates.append(dict(candidate, book=distance))
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

    def kept(self, identity, speakers, readings, *, chapter, voices_by_asset):
        """The history entry for an example a line's saved take heard (``identity``), while it is on disk; else None.

        Found by its request hash and exact sound and codes in this book's or
        an earlier book's takes, retired or not (a take's attempt records and
        codes are never renamed); never one a person marked bad, never one of
        the line's own chapter in this book, never another voice.
        """
        kind, entry = self._heard(identity, speakers, readings, chapter=chapter, voices_by_asset=voices_by_asset)
        return entry if kind == "example" else None

    def _heard(self, identity, speakers, readings, *, chapter, voices_by_asset):
        """``("example", entry)``, ``("chapter", None)`` for a take of the line's own chapter, or ``("gone", None)``."""
        from .common import digest, read_json
        from .continuity import _chosen_readings, request_speakers
        from .continuity_codes import load_codes
        from .take_feedback import judged_bad_audio
        if not isinstance(identity, dict) or not isinstance(identity.get("request_hash"), str):
            return "gone", None
        request_hash, audio = identity["request_hash"], identity.get("audio_sha256")
        if len(request_hash) != 64 or not all(c in "0123456789abcdef" for c in request_hash):
            return "gone", None
        for distance, work in enumerate(self._works()):
            direct = work / "clips" / request_hash
            folders = [direct] if direct.is_dir() else self._parts(work).get(request_hash, [])
            for folder in folders:
                if not folder.is_dir() or folder.is_symlink():
                    continue
                for path in sorted(folder.glob("*.json")):
                    if path.name.endswith(".codes.json") or path.is_symlink():
                        continue
                    try:
                        record = read_json(path)
                    except (ValueError, OSError):
                        continue
                    marker = (record.get("engine_metadata") or {}).get("continuity_codes") if isinstance(record, dict) else None
                    if (not isinstance(marker, dict) or record.get("audio_sha256") != audio
                            or record.get("request_hash") != request_hash
                            or marker.get("sha256") != identity.get("codes_sha256")
                            or not isinstance(record.get("request"), dict)):
                        continue
                    if not distance and _chapter(record["request"]) == chapter:
                        return "chapter", None
                    try:
                        voices = sorted(request_speakers(record["request"]))
                        if (not set(voices) <= set(speakers)
                                or not all(asset in voices_by_asset for asset in marker.get("speakers", []))
                                or audio in judged_bad_audio(folder)):
                            return "gone", None
                        named = frozenset(_chosen_readings(record["request"]))
                        if not set(readings) & named:
                            return "gone", None
                        codes = folder / marker["file"]
                        load_codes(codes, marker)
                    except (ValueError, OSError, KeyError, TypeError):
                        return "gone", None
                    return "example", {
                        "path": str(codes), "marker": dict(marker), "pinnable": False, "voices": voices,
                        "position": None, "readings": named, "example": True,
                        "identity": {"codes_sha256": marker["sha256"], "text_sha256": digest(marker["text"]),
                                     "speakers": list(marker["speakers"]), "request_hash": request_hash,
                                     "audio_sha256": audio}}
        return "gone", None

    def _parts(self, work):
        """``{request hash: [folder]}`` of a render work's split-passage parts.

        Read from disk once per process: only the render makes take folders,
        and it registers every part it makes (``note_part``). Globbing them
        for every line searched every take folder of the book per line.
        """
        key = Path(work).resolve()
        with self._lock:
            found = self._part_folders.get(key)
        if found is None:
            found = {}
            clips = Path(work) / "clips"
            if clips.is_dir() and not clips.is_symlink():
                for folder in clips.glob("*/parts/*"):
                    found.setdefault(folder.name, []).append(folder)
            with self._lock:
                found = self._part_folders.setdefault(key, found)
        return found

    def _note_part(self, work, folder):
        with self._lock:
            found = self._part_folders.get(Path(work).resolve())
            if found is not None and folder not in found.get(folder.name, []):
                found.setdefault(folder.name, []).append(folder)

    def _works(self):
        with self._lock:
            if self._series is None:
                self._series = [self.work, *self._earlier_works()]
            return list(self._series)

    def example(self, speakers, readings, *, sequence, chapter, voices_by_asset, wanted=(), prefer=(), avoid=()):
        """A history entry from another chapter saying the most of ``readings`` right, or None.

        Among those, one also saying more of ``wanted`` (all the line's chosen
        readings) right comes first: a line using two chosen names needs a
        turn saying both, since a turn saying only the missing one would
        trade away the turn saying the other. Equal ones are ordered by
        request hash, never by listing order. An example the line's passing
        saved takes heard (``prefer``: continuity take identities, most
        relevant first) is heard again while it is on disk (``kept``), and a line
        whose saved take passed hearing no example from elsewhere hears none.
        ``prefer`` holds one history per saved take, the most relevant (the
        chapter checkpoint's take) first; only that one decides. Takes in
        ``avoid`` (request hashes a person told this line not to hear) never are.
        """
        avoid = set(avoid)
        if prefer:
            kinds = []
            for identity in prefer[0]:
                if isinstance(identity, dict) and identity.get("request_hash") in avoid:
                    kinds.append("gone")
                    continue
                kind, entry = self._heard(identity, speakers, readings, chapter=chapter,
                                          voices_by_asset=voices_by_asset)
                if kind == "example":
                    return entry
                kinds.append(kind)
            if all(kind == "chapter" for kind in kinds):
                # It said the name right without an example (or any history): keep that.
                return None
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
            earlier_book = item.get("book", 0)
            if (not covered or item["request_hash"] in avoid or (not earlier_book and item["chapter"] == chapter)
                    or not set(item["voices"]) <= set(speakers)
                    or not all(asset in voices_by_asset for asset in item["marker"].get("speakers", []))):
                continue
            # This book first, then the nearest earlier book; within a book the
            # nearest earlier take in book order, then the nearest later one.
            before = earlier_book > 0 or item["sequence"] < sequence
            scored.append((covered, len(wanted & item["readings"]), -earlier_book, before,
                           item["sequence"] if before else -item["sequence"], str(item["request_hash"]), item))
        for *_, item in sorted(scored, key=lambda row: row[:6], reverse=True):
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
