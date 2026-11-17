"""Troubleshoot a passage blocked on a name: what it heard, the voice's other takes of the word, its stream.

Drama 3 says a name the way its chapter history says it, so a passage that
keeps misreading a chosen name usually hears something saying it wrong: a
reading example from another chapter or book, or its own voice's earlier
lines. Three views of a failed passage (``HistoryReview``, Textual-free; the
run shell draws them):

* ``heard()``: the takes its attempts heard as history (Fish's first turn,
  the latest turn, a reading example from elsewhere), each with what the
  name check heard for the passage's problem word.
* ``word_takes()``: every take of the passage's voice, in this book and the
  series' earlier books, that says the problem word, with an excerpt around
  the word to play.
* ``stream()``: the voice's chain of takes in the chapter that led to the
  passage, each heard by the next (``continuity``: first + latest turns).

And three ways out:

* ``avoid(rows)``: the passage alone never hears these takes again
  (``history-avoid.json``, read by ``continuity.ChapterContinuity``); nothing
  else changes, so nothing else renders again. The passage has no passing
  take, so no later line depends on what it heard.
* ``set_aside(rows)``: a person's ``redo`` verdict (``take_feedback``) on each
  take: never used or heard again, its acceptance retired (renamed, never
  deleted), but not counted as a verification disagreement. Under strict
  history reuse every later line that heard one renders again.
* ``stream_from(row)``: the takes to set aside to render the voice's stream
  again from ``row`` (by default its first line in the chapter): that take,
  every later take of the voice in the chain, and every later take of the
  voice in the chapter's saved checkpoint.

Marking a take good or bad goes through ``take_review.TakeReview`` as in the
takes dialog.
"""

from __future__ import annotations

from pathlib import Path

AVOID_FILENAME = "history-avoid.json"
AVOID_FORMAT = "audiobook-history-avoid"
EXCERPT_DIRECTORY = "name-excerpts"
EXCERPT_PAD_SECONDS = 1.2


def load_avoid(work):
    """``{chunk_id: frozenset(request_hash)}``: takes a passage must not hear (empty when none or unreadable)."""
    from .common import read_json, regular_file
    path = Path(work) / AVOID_FILENAME
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        record = read_json(regular_file(path))
    except (ValueError, OSError):
        return {}
    chunks = record.get("chunks") if isinstance(record, dict) and record.get("format") == AVOID_FORMAT else None
    if not isinstance(chunks, dict):
        return {}
    return {chunk: frozenset(item.get("request_hash") for item in items
                             if isinstance(item, dict) and isinstance(item.get("request_hash"), str))
            for chunk, items in chunks.items() if isinstance(chunk, str) and isinstance(items, list)}


def _save_avoid(work, chunk_id, rows):
    from datetime import datetime, timezone
    from .common import read_json, regular_file, write_json
    path = Path(work) / AVOID_FILENAME
    record = {"format": AVOID_FORMAT, "version": 1, "chunks": {}}
    if path.is_file() and not path.is_symlink():
        try:
            saved = read_json(regular_file(path))
            if isinstance(saved, dict) and saved.get("format") == AVOID_FORMAT and isinstance(saved.get("chunks"), dict):
                record = saved
        except (ValueError, OSError):
            pass
    items = record["chunks"].setdefault(chunk_id, [])
    known = {item.get("request_hash") for item in items if isinstance(item, dict)}
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in rows:
        if row["request_hash"] not in known:
            items.append({"request_hash": row["request_hash"], "audio_sha256": row.get("audio_sha256"),
                          "recorded_at": when})
            known.add(row["request_hash"])
    write_json(path, record)


def _segments(request):
    chunk = request.get("chunk") if isinstance(request, dict) else None
    if not isinstance(chunk, dict):
        return []
    turns = chunk.get("paragraph_turns")
    return [row.get("segment") or {} for row in (turns if isinstance(turns, list) else [chunk]) if isinstance(row, dict)]


def _text(request):
    chunk = request.get("chunk") if isinstance(request, dict) else None
    if not isinstance(chunk, dict):
        return ""
    if isinstance(chunk.get("spoken_text"), str):
        return chunk["spoken_text"]
    return " ".join(str(row.get("spoken_text") or "") for row in chunk.get("paragraph_turns") or [] if isinstance(row, dict))


def _chapter_names(work):
    """``{chapter_id: (number, title)}`` from the render work's current plan."""
    from .common import read_json
    try:
        status = read_json(Path(work) / "status.json")
        plan = read_json(Path(work) / "runs" / status["plan_hash"] / "plan.json")
    except (ValueError, OSError, KeyError, TypeError):
        return {}
    return {chapter.get("id"): (chapter.get("number"), chapter.get("title"))
            for chapter in plan.get("chapters") or [] if isinstance(chapter, dict)}


class HistoryReview:
    """The history views of one failed passage (``review``: its ``take_review.TakeReview``)."""

    def __init__(self, review):
        self.review = review
        self.work = Path(review.work)
        self._names = {}

    # Where things are ------------------------------------------------------

    def works(self):
        """This book's render work, then the series' earlier books' (nearest first)."""
        from .reading_examples import for_work
        return for_work(self.work)._works()

    def _records(self):
        return self.review._records()

    def request(self):
        """The passage's latest saved request, or None."""
        records = self._records()
        return records[-1].get("request") if records else None

    def chunk_id(self):
        request = self.request()
        chunk = request.get("chunk") if isinstance(request, dict) else None
        return chunk.get("chunk_id") if isinstance(chunk, dict) else None

    def voices(self):
        from .continuity import request_speakers
        request = self.request()
        try:
            return frozenset(request_speakers(request)) if isinstance(request, dict) else frozenset()
        except (KeyError, TypeError, ValueError):
            return frozenset()

    def chapter(self):
        segments = _segments(self.request())
        return segments[0].get("chapter_id") if segments else None

    def problem_words(self):
        """``{reading key: word as written}`` the passage's attempts misread, most recent first."""
        found = {}
        for record in reversed(self._records()):
            words = ((record.get("report") or {}).get("name_phonemes") or {}).get("words") or []
            for word in words:
                if isinstance(word, dict) and not word.get("ok") and isinstance(word.get("key"), str):
                    found.setdefault(word["key"], word.get("text") or word["key"])
        return found

    def _name(self, work, chapter_id):
        if work not in self._names:
            self._names[work] = _chapter_names(work)
        return self._names[work].get(chapter_id)

    def _row(self, work, folder, record, relation, *, key=None, identity=None):
        """One take as the screens show it."""
        from .common import read_json
        request = record.get("request") or {}
        segments = _segments(request)
        chapter_id = segments[0].get("chapter_id") if segments else None
        sequences = [segment.get("sequence") for segment in segments if type(segment.get("sequence")) is int]
        named = self._name(work, chapter_id)
        where = ("" if work == self.work else f"{work.parent.name}, ") + (
            f"chapter {named[0]} ({named[1]})" if named and named[0] is not None else f"{chapter_id or 'unknown chapter'}")
        accepted = folder / "accepted.json"
        in_use = False
        if accepted.is_file() and not accepted.is_symlink():
            try:
                in_use = read_json(accepted).get("audio_sha256") == record.get("audio_sha256")
            except (ValueError, OSError):
                in_use = False
        row = {"work": work, "folder": folder, "request_hash": record.get("request_hash"),
               "attempt": record.get("attempt"), "audio_sha256": record.get("audio_sha256"),
               "audio_path": str(folder / record["audio_file"]) if isinstance(record.get("audio_file"), str) else None,
               "book": work.parent.name, "chapter_id": chapter_id, "chapter": named[0] if named else None,
               "sequence": min(sequences) if sequences else None, "where": where, "text": _text(request),
               "relation": relation, "in_use": in_use, "identity": identity, "word": None, "span": None,
               "record": record}
        if key is not None:
            self._hear_word(row, key)
        return row

    def _hear_word(self, row, key):
        """What the name check heard for ``key`` in this take, and the word's span for an excerpt."""
        from . import name_phonemes
        from .name_shape import _readings, reading_key
        from .performance_runtime import verification_options
        record = row["record"]
        try:
            performance = verification_options(record.get("request") or {}).get("performance")
            controls = [control for control, phonemes in _readings(performance) if reading_key(phonemes) == key]
        except (ValueError, KeyError, TypeError):
            controls = []
        if not controls:
            return
        report = record.get("report") or {}
        spans = name_phonemes._spans(report, report.get("expected_text") or "", controls[0])
        row["span"] = spans[0] if spans else None
        try:
            verdicts = name_phonemes.judge(record, row["folder"]) or []
        except (ValueError, OSError):
            verdicts = []
        word = next((item for item in verdicts if item.get("key") == key), None)
        if word is not None:
            row["word"] = {"text": word.get("text"), "heard": word.get("heard"), "ok": bool(word.get("ok")),
                           "uncertain": bool(word.get("uncertain"))}

    def _locate(self, identity):
        """``(work, folder, record)`` of a take a request heard (``continuity`` identity), or None."""
        from .common import read_json
        request_hash, audio = identity.get("request_hash"), identity.get("audio_sha256")
        if not isinstance(request_hash, str) or len(request_hash) != 64:
            return None
        from .reading_examples import for_work
        index = for_work(self.work)
        for work in self.works():
            clips = Path(work) / "clips"
            for folder in [clips / request_hash, *index._parts(Path(work)).get(request_hash, ())]:
                if not folder.is_dir() or folder.is_symlink():
                    continue
                for path in [folder / "accepted.json", *sorted(folder.glob("*.json"))]:
                    if path.name.endswith(".codes.json") or not path.is_file() or path.is_symlink():
                        continue
                    try:
                        record = read_json(path)
                    except (ValueError, OSError):
                        continue
                    if (isinstance(record, dict) and record.get("request_hash") == request_hash
                            and record.get("audio_sha256") == audio and isinstance(record.get("request"), dict)):
                        return Path(work), folder, record
        return None

    # The three views ---------------------------------------------------------

    def heard(self, key=None):
        """The takes the passage's attempts heard as history, newest attempt's first; ``key`` judges that word."""
        rows, seen = [], set()
        attempts = [record for record in reversed(self._records()) if isinstance(record.get("request"), dict)]
        counts = {}
        for record in attempts:
            for take in (record["request"].get("continuity") or {}).get("takes") or []:
                ident = (take.get("request_hash"), take.get("audio_sha256"))
                counts[ident] = counts.get(ident, 0) + 1
        chapter = self.chapter()
        for record in attempts:
            takes = (record["request"].get("continuity") or {}).get("takes") or []
            for index, take in enumerate(takes):
                ident = (take.get("request_hash"), take.get("audio_sha256"))
                if ident in seen:
                    continue
                seen.add(ident)
                found = self._locate(take)
                if found is None:
                    continue
                work, folder, heard_record = found
                segments = _segments(heard_record["request"])
                elsewhere = work != self.work or (segments and segments[0].get("chapter_id") != chapter)
                if elsewhere:
                    relation = "reading example from " + ("an earlier book" if work != self.work else "another chapter")
                elif index == 0 and len(takes) > 1:
                    relation = "first turn (the voice's first line)"
                else:
                    relation = "latest turn (the voice's previous line)"
                row = self._row(work, folder, heard_record, relation, key=key, identity=dict(take))
                row["heard_by"] = counts[ident]
                rows.append(row)
        return rows

    def immediate(self, key=None):
        """The takes the passage's newest attempt heard (its first and latest turns): its immediate history."""
        attempts = [record for record in reversed(self._records()) if isinstance(record.get("request"), dict)]
        if not attempts:
            return []
        wanted = {(take.get("request_hash"), take.get("audio_sha256"))
                  for take in (attempts[0]["request"].get("continuity") or {}).get("takes") or []}
        return [row for row in self.heard(key) if (row["request_hash"], row["audio_sha256"]) in wanted]

    def word_takes(self, key):
        """Every take of the passage's voice saying ``key`` (a chosen reading), this book then earlier books, in book order."""
        from .common import read_json
        from .continuity import request_speakers
        voices = self.voices()
        rows = []
        for work in self.works():
            clips = Path(work) / "clips"
            if not clips.is_dir() or clips.is_symlink():
                continue
            for path in sorted([*clips.glob("*/accepted.json"), *clips.glob("*/parts/*/accepted.json")]):
                try:
                    raw = path.read_bytes()
                    if path.is_symlink() or b'"cmu_arpabet"' not in raw:
                        continue
                    record = read_json(path)
                    if not voices or not set(request_speakers(record["request"])) & voices:
                        continue
                except (ValueError, OSError, KeyError, TypeError):
                    continue
                row = self._row(Path(work), path.parent, record, "a take of this voice saying the word", key=key)
                if row["span"] is not None or row["word"] is not None:
                    rows.append(row)
        rows.sort(key=lambda row: (row["work"] != self.work, str(row["book"]),
                                   row["sequence"] if row["sequence"] is not None else 0))
        return rows

    def stream(self, key=None):
        """The voice's chain of takes in this chapter that led to the passage, first line first.

        Walks back from the passage's newest attempt: its latest turn, that
        take's own latest turn, and so on to the chapter's first line (Fish's
        pinned first turn). A reading example from elsewhere ends the walk.
        """
        attempts = [record for record in reversed(self._records()) if isinstance(record.get("request"), dict)]
        if not attempts:
            return []
        chapter = self.chapter()
        chain, seen = [], set()
        takes = (attempts[0]["request"].get("continuity") or {}).get("takes") or []
        first = takes[0] if len(takes) > 1 else None
        current = takes[-1] if takes else None
        while current is not None:
            ident = (current.get("request_hash"), current.get("audio_sha256"))
            if ident in seen:
                break
            seen.add(ident)
            found = self._locate(current)
            if found is None:
                break
            work, folder, record = found
            segments = _segments(record["request"])
            if work != self.work or not segments or segments[0].get("chapter_id") != chapter:
                break
            chain.append(self._row(work, folder, record, "in the stream", key=key, identity=dict(current)))
            earlier = (record["request"].get("continuity") or {}).get("takes") or []
            if first is None and len(earlier) > 1:
                first = earlier[0]
            current = earlier[-1] if earlier else None
        if first is not None and (first.get("request_hash"), first.get("audio_sha256")) not in seen:
            found = self._locate(first)
            if found is not None and found[0] == self.work:
                chain.append(self._row(*found, "the voice's first line in the chapter", key=key, identity=dict(first)))
        chain.sort(key=lambda row: row["sequence"] if row["sequence"] is not None else 0)
        if chain:
            chain[0]["relation"] = "the voice's first line in the chapter"
        return chain

    def stream_from(self, start, key=None):
        """The takes to set aside to render the voice's stream again from ``start`` (a ``stream()`` row).

        ``start``, every later take of the chain, and every later take of the
        voice in the chapter's saved checkpoint (lines after the passage),
        each once, in book order.
        """
        from .common import read_json
        from .continuity import request_speakers
        floor = start["sequence"] if start["sequence"] is not None else 0
        rows = [row for row in self.stream(key) if (row["sequence"] or 0) >= floor]
        known = {(row["request_hash"], row["audio_sha256"]) for row in rows}
        voices = self.voices()
        for path in sorted((self.work / "completed-chapters").glob("*.json")):
            try:
                checkpoint = read_json(path)
            except (ValueError, OSError):
                continue
            for entry in checkpoint.get("acceptance") or []:
                for lane in entry.get("speaker_lanes", [entry]) if isinstance(entry, dict) else []:
                    found = self._locate({"request_hash": lane.get("request_hash"), "audio_sha256": lane.get("audio_sha256")})
                    if found is None or found[0] != self.work:
                        continue
                    record = found[2]
                    segments = _segments(record["request"])
                    try:
                        same_voice = bool(set(request_speakers(record["request"])) & voices)
                    except (KeyError, TypeError, ValueError):
                        same_voice = False
                    sequence = min([s.get("sequence") for s in segments if type(s.get("sequence")) is int] or [-1])
                    ident = (record.get("request_hash"), record.get("audio_sha256"))
                    if (same_voice and segments and segments[0].get("chapter_id") == self.chapter()
                            and sequence >= floor and ident not in known):
                        known.add(ident)
                        rows.append(self._row(*found, "later in the stream", key=key))
        rows.sort(key=lambda row: row["sequence"] if row["sequence"] is not None else 0)
        return rows

    # Ways out ------------------------------------------------------------------

    def avoid(self, rows):
        """The passage never hears these takes again; nothing else changes. Returns how many were added."""
        chunk = self.chunk_id()
        if not isinstance(chunk, str) or not rows:
            return 0
        before = load_avoid(self.work).get(chunk, frozenset())
        _save_avoid(self.work, chunk, rows)
        return len(load_avoid(self.work).get(chunk, frozenset()) - before)

    def review_of(self, row):
        """A ``TakeReview`` for the passage a row's take belongs to (to mark it)."""
        from .take_review import TakeReview
        return TakeReview(row["work"], self.review.config, self.review.verifier, row["request_hash"],
                          directory=row["folder"], book=row["work"].parent)

    def set_aside(self, rows):
        """A person's ``redo`` on each take: never used or heard again, its acceptance retired. Rows set aside."""
        done = []
        for row in rows:
            if type(row.get("attempt")) is not int:
                continue
            try:
                self.review_of(row).mark(row["attempt"], "redo")
            except (ValueError, OSError, KeyError):
                continue
            done.append(row)
        return done

    def excerpt(self, row):
        """A short file of the take around the word (``row["span"]``), for listening; the whole take without one."""
        import subprocess
        from .stored_audio import locate
        source = locate(row["audio_path"]) if row.get("audio_path") else None
        if source is None or row.get("span") is None:
            return row.get("audio_path")
        start, end = row["span"]
        folder = self.work / EXCERPT_DIRECTORY
        folder.mkdir(exist_ok=True)
        word = "".join(ch for ch in str((row.get("word") or {}).get("text") or "name") if ch.isalnum()) or "name"
        target = folder / f"{row['book']}-{word}-{str(row['request_hash'])[:12]}-take-{row['attempt']}-excerpt.wav"
        if not target.exists():
            ffmpeg = (self.review.config.get("render") or {}).get("ffmpeg") or "ffmpeg"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{max(0.0, start - EXCERPT_PAD_SECONDS):.3f}",
                            "-to", f"{end + EXCERPT_PAD_SECONDS:.3f}", "-i", str(source), str(target)],
                           check=True, capture_output=True)
        return str(target)
