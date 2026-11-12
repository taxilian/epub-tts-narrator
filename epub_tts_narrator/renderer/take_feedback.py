"""A person's judgement of verification decisions, kept to improve the detector later.

``render-work/take-feedback.json`` records, per take (request hash and attempt),
whether a person listening judged it good or bad, together with what
verification decided (passed, reasons, method) and the take's audio hash, so a
later judgement stays bound to the same audio. Two kinds of disagreement are
what detection work needs: a **false rejection** (verification rejected a take
a person judged good) and a **false acceptance** (it passed a take a person
judged bad). ``export`` writes every disagreement with the detail needed to
study it: the passage, what the recognizer heard, the reasons and metrics, and
where the audio is.

The file lives in the render work directory, which already holds the book's
text and audio; the export is written there unless a path is given. Nothing
here changes which take is used or any verification policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import threading
import time


FILENAME = "take-feedback.json"
EXPORT_FILENAME = "take-feedback-review.json"
FORMAT = "audiobook-take-feedback"
VERDICTS = ("good", "bad")
_LOCK = threading.Lock()


def _key(request_hash, attempt):
    return f"{request_hash}:{int(attempt)}"


def _empty():
    return {"format": FORMAT, "version": 1, "takes": {}}


def load(work):
    from .common import read_json, regular_file
    path = Path(work) / FILENAME
    if not path.exists() and not path.is_symlink():
        return _empty()
    record = read_json(regular_file(path))
    if (not isinstance(record, dict) or record.get("format") != FORMAT or record.get("version") != 1
            or not isinstance(record.get("takes"), dict)):
        raise ValueError(f"Invalid take feedback file: {path}")
    return record


@contextmanager
def _locked(work):
    from .run_locks import acquire_directory_lock
    work = Path(work)
    with _LOCK:
        lock = None
        for delay in (0, 0.05, 0.1, 0.2, 0.4, 0.8):
            time.sleep(delay)
            try:
                lock = acquire_directory_lock(work / ".take-feedback.lock")
                break
            except FileExistsError as exc:  # Held by another process reviewing the same book.
                failure = exc
        if lock is None:
            raise failure
        try:
            yield
        finally:
            lock.rmdir()


def record_verdict(work, request_hash, row, verdict):
    """Save (or with ``None`` clear) a person's verdict on one take listed by ``take_selection``."""
    from .common import write_json
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError("A take verdict is good, bad or none")
    report = row["record"].get("report") or {}
    with _locked(work):
        record = load(work)
        key = _key(request_hash, row["attempt"])
        if verdict is None:
            record["takes"].pop(key, None)
        else:
            record["takes"][key] = {
                "request_hash": request_hash, "attempt": row["attempt"], "verdict": verdict,
                "audio_sha256": row["record"].get("audio_sha256"),
                "passed": report.get("passed") is True, "reasons": list(report.get("reasons") or []),
                "text_confirmation_method": report.get("text_confirmation_method"),
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        write_json(Path(work) / FILENAME, record)


def verdicts(work, request_hash):
    """``{attempt: verdict}`` for one passage's takes."""
    takes = load(work)["takes"]
    return {entry["attempt"]: entry["verdict"] for entry in takes.values() if entry.get("request_hash") == request_hash}


def disagreement(entry):
    if entry["verdict"] == "good" and not entry["passed"]:
        return "false_rejection"
    if entry["verdict"] == "bad" and entry["passed"]:
        return "false_acceptance"
    return None


def export(work, output=None, *, include_agreements=False):
    """Write every take where a person disagreed with verification; return ``(path, counts)``."""
    from .common import file_hash, read_json, regular_file, write_json
    work = Path(work)
    items, counts = [], {"false_rejection": 0, "false_acceptance": 0, "agreed": 0, "audio_changed": 0}
    for entry in sorted(load(work)["takes"].values(), key=lambda value: (value["request_hash"], value["attempt"])):
        kind = disagreement(entry)
        if kind is None:
            counts["agreed"] += 1
            if not include_agreements:
                continue
        directory = work / "clips" / entry["request_hash"]
        attempt_path = directory / f"attempt-{entry['attempt']:03d}.json"
        try:
            record = read_json(regular_file(attempt_path))
        except (OSError, ValueError):
            record = None
        audio = directory / record["audio_file"] if isinstance(record, dict) and record.get("audio_file") else None
        current = file_hash(audio) if audio is not None and audio.is_file() and not audio.is_symlink() else None
        if current != entry.get("audio_sha256"):
            counts["audio_changed"] += 1
        report = (record or {}).get("report") or {}
        chunk = ((record or {}).get("request") or {}).get("chunk") or {}
        items.append({
            "kind": kind or "agreed", "verdict": entry["verdict"], "recorded_at": entry.get("recorded_at"),
            "request_hash": entry["request_hash"], "attempt": entry["attempt"],
            "chunk_id": chunk.get("chunk_id"), "voice_id": (chunk.get("segment") or {}).get("voice_id"),
            "audio_file": str(audio) if audio is not None else None,
            "audio_matches_judgement": current == entry.get("audio_sha256"),
            "verification": {
                "passed": report.get("passed") is True, "reasons": report.get("reasons") or [],
                "text_confirmation_method": report.get("text_confirmation_method"),
                "metrics": report.get("metrics"), "recognized_by": report.get("recognized_by"),
                "expected_text": report.get("expected_text"), "transcript": report.get("transcript"),
            },
        })
        if kind is not None:
            counts[kind] += 1
    path = Path(output) if output is not None else work / EXPORT_FILENAME
    write_json(path, {"format": "audiobook-take-feedback-review", "version": 1,
                      "about": "Takes where a person listening disagreed with automatic verification. "
                               "false_rejection: rejected but judged good; false_acceptance: passed but judged bad.",
                      "counts": counts, "takes": items})
    return path, counts
