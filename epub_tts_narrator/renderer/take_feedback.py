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
here changes verification policy. A take judged bad is never used again:
``judged_bad_audio`` names those takes' audio, and ``take_review`` retires a
passage's acceptance when its used take is one of them, so the passage renders
again.
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
# "truncated": what the take says is right, but it stops before the passage
# ends. It is never used as it is; its passage is finished from it
# (take_completion), keeping what it said and generating only the rest.
# "redo": set aside by a person to render again with fresh history (a whole
# stream, ``history_review``), not because verification was wrong about it;
# never used or heard again, never counted as a disagreement.
VERDICTS = ("good", "bad", "truncated", "redo")
UNUSABLE = ("bad", "truncated", "redo")
# The verdicts that say verification passed a take it should not have.
WRONGLY_PASSED = ("bad", "truncated")
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
        raise ValueError("A take verdict is good, bad, truncated, redo or none")
    take_record = row.get("record") or {}
    report = take_record.get("report") or {}
    with _locked(work):
        record = load(work)
        key = _key(request_hash, row["attempt"])
        if verdict is None:
            record["takes"].pop(key, None)
        else:
            record["takes"][key] = {
                "request_hash": request_hash, "attempt": row["attempt"], "verdict": verdict,
                "audio_sha256": take_record.get("audio_sha256"),
                "passed": report.get("passed") is True, "reasons": list(report.get("reasons") or []),
                "text_confirmation_method": report.get("text_confirmation_method"),
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        write_json(Path(work) / FILENAME, record)


def verdicts(work, request_hash):
    """``{attempt: verdict}`` for one passage's takes."""
    takes = load(work)["takes"]
    return {entry["attempt"]: entry["verdict"] for entry in takes.values() if entry.get("request_hash") == request_hash}


def render_work(clip_directory):
    """The render work holding a passage's take folder, or None.

    A passage's takes are in ``render-work/clips/HASH``; a passage split into
    shorter parts keeps each part's takes in ``render-work/clips/HASH/parts/PART``.
    """
    directory = Path(clip_directory)
    if directory.parent.name == "clips":
        return directory.parent.parent
    if directory.parent.name == "parts" and directory.parent.parent.parent.name == "clips":
        return directory.parent.parent.parent.parent
    return None


def clip_folder(work, request_hash):
    """The take folder of one passage or split part in this render work, or None."""
    clips = Path(work) / "clips"
    direct = clips / request_hash
    if direct.is_dir() and not direct.is_symlink():
        return direct
    for path in sorted(clips.glob(f"*/parts/{request_hash}")):
        if path.is_dir() and not path.is_symlink():
            return path
    return None


def judged_bad_audio(clip_directory):
    """Audio hashes of every take a person marked bad, truncated or set aside (redo) in this take folder's render work.

    Neither is ever used as it is. A folder outside a render work's
    ``clips`` has no person verdicts.
    """
    work = render_work(clip_directory)
    if work is None:
        return frozenset()
    return frozenset(entry["audio_sha256"] for entry in load(work)["takes"].values()
                     if entry.get("verdict") in UNUSABLE and isinstance(entry.get("audio_sha256"), str))


def judged_truncated(clip_directory, request_hash):
    """``[(attempt, audio_sha256)]`` of this passage's takes a person marked truncated, newest mark first."""
    work = render_work(clip_directory)
    if work is None:
        return []
    marked = [entry for entry in load(work)["takes"].values()
              if entry.get("request_hash") == request_hash and entry.get("verdict") == "truncated"
              and isinstance(entry.get("audio_sha256"), str) and type(entry.get("attempt")) is int]
    marked.sort(key=lambda entry: (str(entry.get("recorded_at") or ""), entry["attempt"]), reverse=True)
    return [(entry["attempt"], entry["audio_sha256"]) for entry in marked]


def disagreement(entry):
    if entry["verdict"] == "good" and not entry["passed"]:
        return "false_rejection"
    if entry["verdict"] in WRONGLY_PASSED and entry["passed"]:
        return "false_acceptance"
    return None


def export(work, output=None, *, include_agreements=False):
    """Write every take where a person disagreed with verification; return ``(path, counts)``."""
    from .common import read_json, regular_file, write_json
    from .stored_audio import audio_exists, audio_hash
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
        current = audio_hash(audio) if audio is not None and audio_exists(audio) else None
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
