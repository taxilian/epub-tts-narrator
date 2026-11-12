"""Repairs a person can make to a chapter that failed on one passage.

When a chapter fails because no take of one passage passed verification, the
run shell can list every retained take of that passage, play them, and let the
person accept one as actually good. Acceptance goes through
``take_selection.select_take`` (an audited ``manual_take_selection``, never a
claim that the take passed verification; an ordinary passed take is accepted
as ordinary), so the chapter's retry and any later render reuse it.

Optionally the person's judgement also teaches the recognizer: when the take
failed only because words were heard differently (``text_mismatch`` with
nothing else wrong) and every difference is an eligible word block, those
blocks are recorded as person-approved (``listened``) heard variants, exactly
like a confirmed ``pronounce-name`` test sentence. Anything structural (a
missing or extra word, a number, a negation) records nothing.
"""

from __future__ import annotations

from pathlib import Path


NOTE = "Accepted by a person listening in the run shell"


class FailedTakeRepair:
    """Takes of one failed passage, and accepting one of them. Safe to call from a UI thread."""

    def __init__(self, work, config, verifier, request_hash):
        self.clips = Path(work) / "clips"
        self.config = config
        self.verifier = verifier
        self.request_hash = request_hash
        self.player_command = list((config.get("casting") or {}).get("player") or ["/usr/bin/afplay"])

    def takes(self):
        """Every retained take, newest last: what the screen needs and nothing more."""
        from .take_selection import list_takes
        rows = []
        for row in list_takes(self.clips, self.request_hash):
            report = row["record"].get("report") or {}
            rows.append({
                "attempt": row["attempt"], "passed": row["passed"], "accepted": row["accepted"],
                "eligible": row["eligible"], "ineligible_reason": row["ineligible_reason"],
                "reasons": list(row["reasons"]), "duration_seconds": row["duration_seconds"],
                "transcript": row["transcript"], "expected_text": report.get("expected_text") or "",
                "audio_path": str(row["audio_path"]), "learnable": self._learnable(report) is not None,
            })
        return rows

    def _learnable(self, report):
        """Eligible heard-variant blocks for a take that failed only on how words were heard."""
        from .heard_variants import _transcript_ok, eligible_blocks, recognizer_key
        if self.config.get("_heard_variants_directory") is None or recognizer_key(self.verifier) is None:
            return None
        if not _transcript_ok(report):
            return None
        blocks = eligible_blocks(report["expected_text"], report["transcript"], require_plausible=False)
        return blocks or None

    def accept(self, attempt, *, learn=False):
        """Accept one take; with ``learn``, record its eligible heard words as person-approved."""
        from .common import file_hash, regular_file
        from .take_selection import list_takes, select_take
        row = next((item for item in list_takes(self.clips, self.request_hash) if item["attempt"] == attempt), None)
        if row is None:
            raise ValueError(f"Take {attempt} does not exist")
        result = select_take(self.clips, self.request_hash, attempt, note=NOTE)
        learned = []
        if learn:
            report = row["record"].get("report") or {}
            blocks = self._learnable(report)
            if blocks:
                from .heard_variants import record_sightings, recognizer_key
                record_sightings(self.config["_heard_variants_directory"], recognizer_key(self.verifier), blocks,
                                 source="listened", audio_sha256=file_hash(regular_file(row["audio_path"])),
                                 request_hash=self.request_hash, transcript=report["transcript"])
                learned = [(" ".join(block["expected"]), " ".join(block["heard"])) for block in blocks]
        return {"attempt": attempt, "basis": result.get("basis"), "learned": learned}
