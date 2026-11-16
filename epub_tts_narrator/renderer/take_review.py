"""A person's review of every take of one passage: listen, choose the one used, judge verification.

The run shell opens this for a passage that failed (no take passed) and for
any rejected take, so a person can compare every retained take of the same
passage and

* **use** one: ``take_selection.select_take`` (an audited
  ``manual_take_selection`` for a failed take, never a claim that it passed;
  a passed take is accepted as ordinary), so the chapter's retry or rebuild
  and every later render use it;
* **mark** takes as acceptable (verification rejected a good take) or bad
  (it should not be used), saved in ``take_feedback`` for later study of the
  detector. A take marked bad is never used again: when it is the take used,
  its acceptance is retired (renamed, never deleted, see
  ``retire_judged_bad``), so the passage renders again on the chapter's
  rebuild or the next render, and no recheck, resume or best-effort choice
  picks that take again;
* optionally **teach the recognizer**: when a take judged good failed only
  because words were heard differently (``text_mismatch`` with nothing else
  wrong) and every difference is an eligible word block, those blocks become
  person-approved (``listened``) heard variants, exactly like a confirmed
  ``pronounce-name`` test sentence. Anything structural (a missing or extra
  word, a number, a negation) records nothing.
"""

from __future__ import annotations

from pathlib import Path


NOTE = "Accepted by a person listening in the run shell"



def word_differences(expected, heard, *, limit=6):
    """Where ``heard`` differs from ``expected``, word by word, in plain words (at most ``limit``).

    Case, punctuation and hyphens are ignored (``sun-browned`` and
    ``sunbrowned`` are one word), and so are words the recognizer only
    spelled another way (``Valida`` for ``Valeda``: the same Double
    Metaphone sounds), so only real differences remain:
    ``missing at the end: "with their hands close to their weapons."``.
    A chosen name heard rearranged is reported separately (``name_shape``).
    """
    import difflib
    import re
    said = [token for token in re.findall(r"\S+", expected or "")]
    got = [token for token in re.findall(r"\S+", heard or "")]
    key = lambda token: re.sub(r"[^\w]", "", token.replace("\u2019", "'")).casefold()
    said_keys, got_keys = [key(token) for token in said], [key(token) for token in got]
    lines = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, said_keys, got_keys, autojunk=False).get_opcodes():
        if tag == "equal" or "".join(said_keys[a:b]) == "".join(got_keys[c:d]):
            continue  # also "sun browned" for "sun-browned": the same letters
        if tag == "replace" and _sounds_alike("".join(said_keys[a:b]), "".join(got_keys[c:d])):
            continue
        wanted, instead = " ".join(said[a:b]), " ".join(got[c:d])
        if tag == "delete":
            where = "at the end" if b == len(said) and d == len(got) else "after " + _quoted(" ".join(said[max(0, a - 3):a]))
            lines.append(f"missing {where}: {_quoted(wanted)}")
        elif tag == "insert":
            lines.append(f"extra after {_quoted(' '.join(said[max(0, a - 3):a]))}: {_quoted(instead)}")
        else:
            lines.append(f"{_quoted(wanted)} heard as {_quoted(instead)}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"and {len(lines) - limit} more"]
    return lines


def _sounds_alike(left, right):
    """Whether two spellings share a Double Metaphone key (a recognizer's spelling of the same sound)."""
    try:
        from metaphone import doublemetaphone
    except ImportError:
        return False
    keys = lambda word: {key for key in doublemetaphone(word) if key}  # noqa: E731
    return bool(left and right and not left.isdigit() and not right.isdigit() and keys(left) & keys(right))


def _quoted(text):
    return "\u201c" + (text if len(text) <= 120 else text[:117] + "\u2026") + "\u201d"


def _tone(report):
    """``a steady 7.1 kHz tone (…)`` when a take was rejected for one, else ``""``."""
    from .steady_tone import describe
    return describe(report)


def _misread(report):
    """``Va'Leda heard as valdea`` when a chosen pronunciation was heard rearranged, else ``""``."""
    from .name_shape import describe
    return describe(report)


def _finished_from(record):
    """The take a completion finished (``take_completion``), else ``None``."""
    completion = (record.get("engine_metadata") or {}).get("completion")
    source = completion.get("source_attempt") if isinstance(completion, dict) else None
    return source if type(source) is int else None


class TakeReview:
    """Takes of one passage, and a person's decisions about them. Safe to call from a UI thread."""

    def __init__(self, work, config, verifier, request_hash, *, directory=None, book=None, list_takes=None,
                 select_take=None, note=NOTE):
        """``list_takes``/``select_take`` default to ``take_selection``'s; review-ui passes its own."""
        from .take_feedback import clip_folder
        self.work = Path(work)
        self._list_takes, self._select_take, self.note = list_takes, select_take, note
        # The book (bundle) directory, for edits a person makes to the passage.
        self.book = Path(book) if book is not None else None
        # The take folder: ``clips/HASH``, or ``clips/HASH/parts/PART`` for one
        # part of a passage split into shorter parts.
        self.directory = (Path(directory) if directory is not None
                          else clip_folder(self.work, request_hash) or self.work / "clips" / request_hash)
        self.clips = self.directory.parent
        self.config = config
        self.verifier = verifier
        self.request_hash = request_hash
        self.player_command = list((config.get("casting") or {}).get("player") or ["/usr/bin/afplay"])

    def takes(self):
        """Every retained take, oldest first: what the screen needs and nothing more."""
        from .take_feedback import verdicts
        marks = verdicts(self.work, self.request_hash)
        used = self.used()
        rows = []
        for row in self._lister()(self.clips, self.request_hash):
            record = row.get("record") or {}
            report = record.get("report") or {}
            accepted = row["accepted"] or (used is not None and used.get("attempt") == row["attempt"]
                                           and used.get("audio_sha256") == record.get("audio_sha256"))
            rows.append({
                "attempt": row["attempt"], "passed": row["passed"], "accepted": accepted,
                "eligible": row["eligible"], "ineligible_reason": row["ineligible_reason"],
                "reasons": list(row["reasons"]), "duration_seconds": row["duration_seconds"],
                "transcript": row["transcript"], "expected_text": report.get("expected_text") or "",
                "audio_path": str(row["audio_path"]), "learnable": self._learnable(report) is not None,
                "verdict": marks.get(row["attempt"]), "finished_from": _finished_from(record),
                "stopped": report.get("generation_stopped"), "misread": _misread(report),
                "tone": _tone(report),
            })
        if used is not None and not any(row["accepted"] for row in rows):
            # The used take is not one of the listed attempts' audio (a repaired or
            # copied take); it is still shown, so every take of the passage is there.
            same = next((row for row in rows if row["attempt"] == used.get("attempt")), None)
            if same is not None:
                same["accepted"] = True
            elif isinstance(used.get("audio_file"), str):
                report = used.get("report") or {}
                rows.append({
                    "attempt": used.get("attempt"), "passed": report.get("passed") is True, "accepted": True,
                    "eligible": False, "ineligible_reason": "it is the take already used",
                    "reasons": list(report.get("reasons") or []), "duration_seconds": None,
                    "transcript": report.get("transcript") or "", "expected_text": report.get("expected_text") or "",
                    "audio_path": str(self.directory / used["audio_file"]), "learnable": False,
                    "verdict": marks.get(used.get("attempt")), "finished_from": _finished_from(used),
                })
                rows.sort(key=lambda row: row["attempt"] if type(row["attempt"]) is int else 0)
        return rows

    def _records(self):
        """Every saved attempt record of this passage, oldest first, including attempts that made no audio."""
        from .common import read_json, regular_file
        records = []
        if self.directory.is_symlink() or not self.directory.is_dir():
            return records
        for path in sorted(self.directory.glob("attempt-[0-9][0-9][0-9]*.json")):
            if not path.name[len("attempt-"):-len(".json")].isdigit():
                continue
            try:
                record = read_json(regular_file(path))
            except (ValueError, OSError):
                continue
            if isinstance(record, dict) and record.get("request_hash") == self.request_hash:
                records.append(record)
        return sorted(records, key=lambda record: record.get("attempt") if type(record.get("attempt")) is int else 0)

    def passage(self):
        """``{"text", "kind", "segment_ids", "directions"}`` of the passage, from its saved request, or None."""
        records = self._records()
        request = records[-1].get("request") if records else None
        if not isinstance(request, dict) and self.used() is not None:
            request = self.used().get("request")
        chunk = request.get("chunk") if isinstance(request, dict) else None
        if not isinstance(chunk, dict):
            return None
        turns = chunk.get("paragraph_turns")
        segments = [row.get("segment") for row in (turns if isinstance(turns, list) else [chunk])
                    if isinstance(row, dict) and isinstance(row.get("segment"), dict)]
        text = chunk.get("spoken_text")
        if not isinstance(text, str):
            text = " ".join(str(row.get("spoken_text") or "") for row in turns or [] if isinstance(row, dict))
        return {"text": text, "kind": request.get("kind"),
                "segment_ids": [segment.get("id") for segment in segments if isinstance(segment.get("id"), str)],
                "directions": [str((segment.get("delivery") or {}).get("direction") or "")
                               for segment in segments]}

    def attempts(self):
        """What every attempt did, oldest first: ``{"attempt", "audio", "passed", "reasons", "error", "pending"}``.

        Attempts that made no audio (an engine error, an interrupted take)
        are listed too, with their error; their takes cannot be heard.
        """
        rows = []
        for record in self._records():
            report = record.get("report") if isinstance(record.get("report"), dict) else {}
            error = report.get("provider_error") or report.get("error") or ""
            if report.get("error_type") and error and not str(error).startswith(str(report["error_type"])):
                error = f"{report['error_type']}: {error}"
            rows.append({"attempt": record.get("attempt"), "audio": isinstance(record.get("audio_file"), str),
                         "passed": report.get("passed") is True, "reasons": list(report.get("reasons") or []),
                         "error": str(error), "echo": "history_echo" in report,
                         "pending": report.get("reasons") == ["generation_incomplete"]})
        return rows

    def reset_for_fresh_takes(self):
        """Before a person's fresh retry: set aside this passage's saved best-effort choice and clip repair.

        A pending best-effort choice would otherwise be resumed instead of
        making new takes, and a finished clip repair would not be tried
        again. They are renamed under a ``fresh-retry`` suffix beside a
        ``fresh-retry-*.json`` record, never deleted; an accepted take is
        never touched. Returns the names set aside.
        """
        if (self.directory / "accepted.json").exists() or (self.directory / "accepted.json").is_symlink():
            return []
        names = [name for name in RETIRED_FILES[1:]
                 if (self.directory / name).exists() or (self.directory / name).is_symlink()]
        if not names:
            return []
        return retire_files(self.directory, names, "fresh-retry", {"reason": "person_requested_fresh_takes"})

    def rejected_attempts(self):
        """Attempt numbers of this passage's rejected takes with audio (cheap: no audio is read)."""
        from .take_selection import retained_attempts
        return [number for number, record in retained_attempts(self.clips, self.request_hash)
                if (record.get("report") or {}).get("passed") is not True]

    def feedback(self):
        """``{(request_hash, attempt): verdict}`` for every take judged in this render work."""
        from .take_feedback import load
        return {(entry.get("request_hash"), entry.get("attempt")): entry.get("verdict")
                for entry in load(self.work)["takes"].values()}

    def _lister(self):
        if self._list_takes is None:
            from .take_selection import list_takes
            self._list_takes = list_takes
        return self._list_takes

    def _selector(self):
        if self._select_take is None:
            from .take_selection import select_take
            self._select_take = select_take
        return self._select_take

    def used(self):
        """The accepted record for this passage (a manual selection keeps its own audio copy), or None."""
        from .common import read_json, regular_file
        path = self.directory / "accepted.json"
        if not path.exists() and not path.is_symlink():
            return None
        record = read_json(regular_file(path))
        return record if isinstance(record, dict) else None

    def _learnable(self, report):
        """Eligible heard-variant blocks for a take that failed only on how words were heard."""
        from .heard_variants import _transcript_ok, eligible_blocks, recognizer_key
        if self.config.get("_heard_variants_directory") is None or recognizer_key(self.verifier) is None:
            return None
        if not _transcript_ok(report):
            return None
        blocks = eligible_blocks(report["expected_text"], report["transcript"], require_plausible=False)
        return blocks or None

    def _row(self, attempt):
        row = next((item for item in self._lister()(self.clips, self.request_hash) if item["attempt"] == attempt), None)
        if row is None:
            raise ValueError(f"Take {attempt} does not exist")
        return row

    def _learn(self, row):
        from .stored_audio import audio_hash
        report = (row.get("record") or {}).get("report") or {}
        blocks = self._learnable(report)
        if not blocks:
            return []
        from .heard_variants import record_sightings, recognizer_key
        record_sightings(self.config["_heard_variants_directory"], recognizer_key(self.verifier), blocks,
                         source="listened", audio_sha256=audio_hash(row["audio_path"]),
                         request_hash=self.request_hash, transcript=report["transcript"])
        return [(" ".join(block["expected"]), " ".join(block["heard"])) for block in blocks]

    def accept(self, attempt, *, learn=False):
        """Use one take for this passage; with ``learn``, record its eligible heard words as person-approved.

        A person choosing a take also judges it good, which the feedback keeps.
        """
        from .take_feedback import record_verdict
        row = self._row(attempt)
        result = self._selector()(self.clips, self.request_hash, attempt, note=self.note)
        record_verdict(self.work, self.request_hash, row, "good")
        learned = self._learn(row) if learn else []
        return {"attempt": attempt, "basis": result.get("basis"), "learned": learned}

    def mark(self, attempt, verdict, *, learn=False):
        """Judge one take ``good``, ``bad`` or ``truncated`` (``None`` clears).

        Marking the used take bad or truncated retires its acceptance
        (``retired`` is then true): the passage has no take until its chapter
        renders it again, which for a truncated take keeps what it said and
        generates only the rest (``take_completion``). Any other mark leaves
        the used take alone.
        """
        from .take_feedback import record_verdict
        row = self._row(attempt)
        record_verdict(self.work, self.request_hash, row, verdict)
        learned = self._learn(row) if learn and verdict == "good" else []
        retired = bool(retire_judged_bad(self.directory)) if verdict in ("bad", "truncated") else False
        return {"attempt": attempt, "verdict": verdict, "learned": learned, "retired": retired}


GOOD_NOTE = "A take a person marked good while listening in the run shell"


def use_judged_good(directory, request_hash):
    """Accept a retained take of this passage that a person marked good; its record, or None.

    The most recently judged good take whose audio is unchanged and usable is
    accepted through ``take_selection.select_take`` (a failed take becomes an
    audited manual selection, never a pass). Only take folders in a render
    work's ``clips`` (a passage, or one part of a split passage) have person
    verdicts.
    """
    from .take_feedback import load, render_work
    from .take_selection import current_acceptance, list_takes, select_take
    directory = Path(directory)
    work = render_work(directory)
    if work is None:
        return None
    try:
        good = [entry for entry in load(work)["takes"].values()
                if entry.get("request_hash") == request_hash and entry.get("verdict") == "good"]
        if not good:
            return None
        rows = {row["attempt"]: row for row in list_takes(directory.parent, request_hash)}
    except (ValueError, OSError):
        return None
    for entry in sorted(good, key=lambda item: (str(item.get("recorded_at") or ""), item.get("attempt") or 0),
                        reverse=True):
        row = rows.get(entry.get("attempt"))
        if row is None or not row["eligible"] or row["record"].get("audio_sha256") != entry.get("audio_sha256"):
            continue
        try:
            select_take(directory.parent, request_hash, row["attempt"], note=GOOD_NOTE)
        except (ValueError, OSError):
            continue
        record = current_acceptance(directory, request_hash)
        if record is not None:
            return record
    return None


# A retired acceptance's files: the record, and any best-effort choice or
# repair that would otherwise publish the same take again.
RETIRED_FILES = ("accepted.json", "exhaustion-selection.json", "clip-repair.json", "clip-repair-parts")


def retire_judged_bad(directory):
    """Retire this passage's acceptance when a person marked its take bad; the retired names, or ``[]``.

    Files are renamed under an evidence-bound ``judged-bad`` suffix beside a
    ``judged-bad-*.json`` record, never deleted. With no acceptance the
    chapter checkpoint, the speech cache and every later render find no take,
    so the passage renders again.
    """
    from .common import read_json
    from .take_feedback import judged_bad_audio
    directory = Path(directory)
    path = directory / "accepted.json"
    if path.is_symlink() or not path.is_file():
        return []
    bad = judged_bad_audio(directory)
    record = read_json(path) if bad else None
    if not isinstance(record, dict) or record.get("audio_sha256") not in bad:
        return []
    return retire_acceptance(directory, record, "judged-bad", {"reason": "judged_bad"})


def retire_acceptance(directory, record, label, evidence):
    """Rename a passage's acceptance (and any choice or repair that would publish it again) under ``label``.

    ``record`` is the acceptance being retired; ``evidence`` says why and is
    saved in ``LABEL-SUFFIX.json`` with the names it retired. Nothing is
    deleted. Returns the retired names.
    """
    return retire_files(directory, RETIRED_FILES, label,
                        {**evidence, "attempt": record.get("attempt"), "audio_sha256": record.get("audio_sha256")})


def retire_files(directory, names, label, evidence):
    """Rename the named files of a passage's take folder under ``label``; never deletes. Returns the new names."""
    from datetime import datetime, timezone
    import os
    from .common import digest, write_json
    directory = Path(directory)
    evidence = {**evidence, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # The same take can be retired again (restored by hand, say): each
    # retirement keeps its own names.
    suffix = digest({**evidence, "nonce": os.urandom(8).hex()})[:12]
    moved = []
    for name in names:
        source = directory / name
        if not source.exists() and not source.is_symlink():
            continue
        stem, dot, extension = name.partition(".")
        target = directory / (f"{stem}.{label}-{suffix}{dot}{extension}" if dot else f"{stem}.{label}-{suffix}")
        if target.exists() or target.is_symlink():
            raise ValueError(f"A retired {label} record already exists: {target.name}")
        source.rename(target)
        moved.append(target.name)
    write_json(directory / f"{label}-{suffix}.json", {"version": 1, "evidence": evidence, "retired": moved})
    return moved


def retire_all_judged_bad(work):
    """Retire every acceptance in this render work whose take a person marked bad; their request hashes."""
    from .take_feedback import clip_folder, load
    retired = []
    import re
    hashes = sorted({entry.get("request_hash") for entry in load(work)["takes"].values()
                     if entry.get("verdict") in ("bad", "truncated") and isinstance(entry.get("request_hash"), str)
                     and re.fullmatch(r"[0-9a-f]{64}", entry["request_hash"])})
    for request_hash in hashes:
        directory = clip_folder(work, request_hash)
        if directory is None:
            continue
        if retire_judged_bad(directory):
            retired.append(request_hash)
    return retired


# The name the failed-passage repair first had.
FailedTakeRepair = TakeReview
