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
  detector, never changing which take is used;
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
        return rows

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
        from .common import file_hash, regular_file
        report = row["record"].get("report") or {}
        blocks = self._learnable(report)
        if not blocks:
            return []
        from .heard_variants import record_sightings, recognizer_key
        record_sightings(self.config["_heard_variants_directory"], recognizer_key(self.verifier), blocks,
                         source="listened", audio_sha256=file_hash(regular_file(row["audio_path"])),
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
        """Judge one take ``good`` or ``bad`` (``None`` clears); never changes which take is used."""
        from .take_feedback import record_verdict
        row = self._row(attempt)
        record_verdict(self.work, self.request_hash, row, verdict)
        learned = self._learn(row) if learn and verdict == "good" else []
        return {"attempt": attempt, "verdict": verdict, "learned": learned}


# The name the failed-passage repair first had.
FailedTakeRepair = TakeReview
