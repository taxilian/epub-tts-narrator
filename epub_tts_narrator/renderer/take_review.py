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


class TakeReview:
    """Takes of one passage, and a person's decisions about them. Safe to call from a UI thread."""

    def __init__(self, work, config, verifier, request_hash):
        self.work = Path(work)
        self.clips = self.work / "clips"
        self.config = config
        self.verifier = verifier
        self.request_hash = request_hash
        self.player_command = list((config.get("casting") or {}).get("player") or ["/usr/bin/afplay"])

    def takes(self):
        """Every retained take, oldest first: what the screen needs and nothing more."""
        from .take_feedback import verdicts
        from .take_selection import list_takes
        marks = verdicts(self.work, self.request_hash)
        used = self.used()
        rows = []
        for row in list_takes(self.clips, self.request_hash):
            report = row["record"].get("report") or {}
            accepted = row["accepted"] or (used is not None and used.get("attempt") == row["attempt"]
                                           and used.get("audio_sha256") == row["record"].get("audio_sha256"))
            rows.append({
                "attempt": row["attempt"], "passed": row["passed"], "accepted": accepted,
                "eligible": row["eligible"], "ineligible_reason": row["ineligible_reason"],
                "reasons": list(row["reasons"]), "duration_seconds": row["duration_seconds"],
                "transcript": row["transcript"], "expected_text": report.get("expected_text") or "",
                "audio_path": str(row["audio_path"]), "learnable": self._learnable(report) is not None,
                "verdict": marks.get(row["attempt"]),
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
                    "audio_path": str(self.clips / self.request_hash / used["audio_file"]), "learnable": False,
                    "verdict": marks.get(used.get("attempt")),
                })
                rows.sort(key=lambda row: row["attempt"] if type(row["attempt"]) is int else 0)
        return rows

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

    def used(self):
        """The accepted record for this passage (a manual selection keeps its own audio copy), or None."""
        from .common import read_json, regular_file
        path = self.clips / self.request_hash / "accepted.json"
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
        from .take_selection import list_takes
        row = next((item for item in list_takes(self.clips, self.request_hash) if item["attempt"] == attempt), None)
        if row is None:
            raise ValueError(f"Take {attempt} does not exist")
        return row

    def _learn(self, row):
        from .stored_audio import audio_hash
        report = row["record"].get("report") or {}
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
        from .take_selection import select_take
        row = self._row(attempt)
        result = select_take(self.clips, self.request_hash, attempt, note=NOTE)
        record_verdict(self.work, self.request_hash, row, "good")
        learned = self._learn(row) if learn else []
        return {"attempt": attempt, "basis": result.get("basis"), "learned": learned}

    def mark(self, attempt, verdict, *, learn=False):
        """Judge one take ``good`` or ``bad`` (``None`` clears).

        Marking the used take bad retires its acceptance (``retired`` is then
        true): the passage has no take until its chapter renders it again.
        Any other mark leaves the used take alone.
        """
        from .take_feedback import record_verdict
        row = self._row(attempt)
        record_verdict(self.work, self.request_hash, row, verdict)
        learned = self._learn(row) if learn and verdict == "good" else []
        retired = bool(retire_judged_bad(self.clips / self.request_hash)) if verdict == "bad" else False
        return {"attempt": attempt, "verdict": verdict, "learned": learned, "retired": retired}


GOOD_NOTE = "A take a person marked good while listening in the run shell"


def use_judged_good(directory, request_hash):
    """Accept a retained take of this passage that a person marked good; its record, or None.

    The most recently judged good take whose audio is unchanged and usable is
    accepted through ``take_selection.select_take`` (a failed take becomes an
    audited manual selection, never a pass). Only passage folders directly in
    ``render-work/clips`` have person verdicts.
    """
    from .take_feedback import load
    from .take_selection import current_acceptance, list_takes, select_take
    directory = Path(directory)
    if directory.parent.name != "clips":
        return None
    try:
        good = [entry for entry in load(directory.parent.parent)["takes"].values()
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
    from datetime import datetime, timezone
    from .common import digest, read_json, write_json
    from .take_feedback import judged_bad_audio
    directory = Path(directory)
    path = directory / "accepted.json"
    if path.is_symlink() or not path.is_file():
        return []
    bad = judged_bad_audio(directory)
    record = read_json(path) if bad else None
    if not isinstance(record, dict) or record.get("audio_sha256") not in bad:
        return []
    evidence = {"reason": "judged_bad", "attempt": record.get("attempt"),
                "audio_sha256": record["audio_sha256"],
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    suffix = digest(evidence)[:12]
    moved = []
    for name in RETIRED_FILES:
        source = directory / name
        if not source.exists() and not source.is_symlink():
            continue
        stem, dot, extension = name.partition(".")
        target = directory / (f"{stem}.judged-bad-{suffix}{dot}{extension}" if dot else f"{stem}.judged-bad-{suffix}")
        if target.exists() or target.is_symlink():
            raise ValueError(f"A retired judged-bad record already exists: {target.name}")
        source.rename(target)
        moved.append(target.name)
    write_json(directory / f"judged-bad-{suffix}.json", {"version": 1, "evidence": evidence, "retired": moved})
    return moved


def retire_all_judged_bad(work):
    """Retire every acceptance in this render work whose take a person marked bad; their request hashes."""
    from .take_feedback import load
    clips = Path(work) / "clips"
    retired = []
    import re
    hashes = sorted({entry.get("request_hash") for entry in load(work)["takes"].values()
                     if entry.get("verdict") == "bad" and isinstance(entry.get("request_hash"), str)
                     and re.fullmatch(r"[0-9a-f]{64}", entry["request_hash"])})
    for request_hash in hashes:
        directory = clips / request_hash
        if directory.parent.is_symlink() or directory.is_symlink() or not directory.is_dir():
            continue
        if retire_judged_bad(directory):
            retired.append(request_hash)
    return retired


# The name the failed-passage repair first had.
FailedTakeRepair = TakeReview
