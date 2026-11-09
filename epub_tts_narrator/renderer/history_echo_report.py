"""Find, and optionally retire, published history-echo takes in a book's render work.

Best-effort exhaustion choices are the only accepted takes that can be echoes:
ordinary acceptance requires the requested words. This scan reads saved take
records and codes files only (no models, no network). A choice is an echo when
its original transcript lines up with a history turn it heard (the texts of
its own request's continuity takes) far better than with its own passage.

``--retire`` renames each echo choice's files with a ``history-echo-<hash>``
suffix (``history_echo.archive``; nothing is deleted) under the same locks as
render and cleanup. The next render regenerates those lines through the
history-echo ladder and rebuilds their chapters from the other cached takes.
Render rejects echo choices on its own as well; this command reports the
affected chapters up front.
"""

from pathlib import Path
import re

from .common import read_json, regular_file
from .history_echo import ECHO_CHOICE_FILES, archive, plain_text, rejected_pick


def _segments(chunk):
    turns = chunk.get("paragraph_turns")
    return [turn["segment"] for turn in turns] if turns else [chunk.get("segment") or {}]


def _heard(clips, request):
    """``(request_hash, text)`` turns the take heard, from saved codes files."""
    turns, missing = [], 0
    for take in (request.get("continuity") or {}).get("takes") or []:
        request_hash, codes = take.get("request_hash"), take.get("codes_sha256")
        if not (isinstance(request_hash, str) and re.fullmatch(r"[0-9a-f]{64}", request_hash)
                and isinstance(codes, str) and re.fullmatch(r"[0-9a-f]{64}", codes)):
            missing += 1
            continue
        path = clips / request_hash / f"{codes}.codes.json"
        try:
            text = read_json(regular_file(path)).get("text")
        except (OSError, ValueError):
            text = None
        if isinstance(text, str):
            turns.append((request_hash, text))
        else:
            missing += 1
    return turns, missing


def scan(work, chapter_numbers):
    """Echo choices in ``work``: findings plus counts. ``chapter_numbers`` maps chapter id to number."""
    clips = Path(work) / "clips"
    findings, choices, unavailable = [], 0, 0
    if not clips.is_dir():
        return {"best_effort_choices": 0, "echo_choices": [], "history_unavailable": 0}
    for directory in sorted(clips.iterdir()):
        path = directory / "accepted.json"
        if directory.is_symlink() or not path.is_file() or path.is_symlink():
            continue
        try:
            record = read_json(regular_file(path))
        except (OSError, ValueError):
            continue
        if (not isinstance(record, dict)
                or not {"exhaustion_selection", "manual_selection"}.intersection(
                    record.get("engine_metadata") or {})):
            continue
        choices += 1
        request = record.get("request") or {}
        heard, missing = _heard(clips, request)
        if missing and not heard:
            unavailable += 1
            continue
        evidence = rejected_pick(record, heard)
        if evidence is None:
            continue
        chunk = request.get("chunk") or {}
        segments = _segments(chunk)
        chapter_id = next((segment.get("chapter_id") for segment in segments if segment.get("chapter_id")), None)
        report = record.get("report") or {}
        original = report.get("original_failed_report") or {}
        echoed = next((text for request_hash, text in heard if request_hash == evidence["heard_request_hash"]), "")
        findings.append({
            "chapter_id": chapter_id, "chapter_number": chapter_numbers.get(chapter_id),
            "segment_ids": [segment.get("id") for segment in segments],
            "voice_ids": sorted({segment.get("voice_id") for segment in segments if segment.get("voice_id")}),
            "text": chunk.get("spoken_text") or report.get("expected_text", ""),
            "transcript": (original.get("transcript") or report.get("transcript") or "").strip(),
            "echoed_line": plain_text(echoed), "evidence": evidence,
            "take_directory": directory.name,
        })
    findings.sort(key=lambda item: (item["chapter_number"] or 0, item["segment_ids"]))
    return {"best_effort_choices": choices, "echo_choices": findings, "history_unavailable": unavailable}


def retire(work, chapter_numbers, *, book_lock=None):
    """Retire every echo choice in ``work`` while holding its render lock."""
    from .run_locks import acquire_directory_lock
    from .workflow import RenderBlocked
    work = Path(work)
    if not work.is_dir():
        return {**scan(work, chapter_numbers), "retired": True}
    lock = work / ".render.lock"
    try:
        lock = acquire_directory_lock(lock)
    except FileExistsError as exc:
        raise RenderBlocked("Another render, assembly, or cleanup may be running; "
                            f"retry after it finishes: {lock}") from exc
    held = None
    try:
        held = book_lock() if book_lock is not None else None
        result = scan(work, chapter_numbers)
        for finding in result["echo_choices"]:
            finding["retired_files"] = archive(work / "clips" / finding["take_directory"],
                                               ECHO_CHOICE_FILES, finding["evidence"])
        return {**result, "retired": True}
    finally:
        if held is not None:
            held.rmdir()
        lock.rmdir()


def history_echo_report(bundle, config, *, retire_choices=False):
    """Scan this book's render work; with ``retire_choices``, retire every echo choice found."""
    from .book_context import resolve_book_config
    from .workflow import RenderBlocked, _lock_book_audio
    config = resolve_book_config(bundle, config)
    work = Path(config["render"]["work_dir"])
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise RenderBlocked("The render work directory must not use symlinks")
    numbers = {chapter["entry"]["chapter_id"]: index
               for index, chapter in enumerate(bundle["chapters"], 1)}
    if not retire_choices:
        return {**scan(work, numbers), "retired": False}
    return retire(work, numbers, book_lock=lambda: _lock_book_audio(bundle))
