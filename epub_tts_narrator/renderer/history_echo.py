"""History echo: a take that replays an earlier line of its chapter history.

Drama 3's history layout is Fish's take-revision conversation: earlier takes
sit before the new user turn as completed turns, and the model was trained to
redo a take when the next turn reads like a note on it. A short line, a speech
tag ("Sal asked carefully.") or a heading can read like such a note, so the
model speaks the previous turn again instead of the requested text. Retrying
the same prompt with another seed rarely helps, because the prompt is the cause.

Recognition evidence decides it after the fact: a failed take whose transcript
lines up with a heard history turn far better than with its requested text is
an echo. Its next attempt changes the prompt along a fixed ladder instead of
re-rolling it (Fish's history layout stays the default for every other take):

1. ``role_tag``  - the same history, with ``[narration]`` or ``[dialogue]``
   leading the new line so it reads as a new line rather than a note;
2. ``streaming`` - the same history audio in Drama 3's streaming-continuation
   layout, with the reference recording as its first turn (single voice, no
   bracketed controls, and an engine that builds that layout);
3. ``none``      - no history; the reference still sets the voice.

An echo never becomes an exhaustion candidate or published audio, and a saved
exhaustion choice that was an echo is not reused.
"""

from difflib import SequenceMatcher
from pathlib import Path
import re

from .common import digest, read_json, regular_file, write_json


VERSION = 1
LADDER = ("role_tag", "streaming", "none")
# Files of a published best-effort choice and the repair that produced it.
ECHO_CHOICE_FILES = ("accepted.json", "exhaustion-selection.json", "clip-repair.json", "clip-repair-parts")
RUNG_DESCRIPTIONS = {
    "role_tag": "the same history with a role tag leading the new line",
    "streaming": "the same history in Drama 3's streaming-continuation layout",
    "none": "no chapter history (reference voice only)",
}
SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_LEADING_TAG = re.compile(r"^(\s*<\|speaker:\d+\|>\s*)")
_LEADING_BRACKET = re.compile(r"^\[([^\]]*)\]\s*")
# Most of what was heard must come from one history turn, and clearly more
# than from the requested text. Calibrated on recorded Drama 3 failures.
MIN_HEARD_SHARE = 0.5
MIN_MARGIN = 0.25


def plain_text(text):
    """Script words of a wire turn: no speaker tags or bracketed controls."""
    return " ".join(_BRACKETS.sub(" ", SPEAKER_TAG.sub(" ", text or "")).split())


def _shared(said, words):
    matcher = SequenceMatcher(None, said, words, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def detect(expected, transcript, heard):
    """Echo evidence, or None. ``heard`` lists ``(request_hash, wire_text)`` turns."""
    from .quality import normalize_words
    said = normalize_words(transcript or "")
    if len(said) < 2:
        return None
    expected_share = _shared(said, normalize_words(expected or "")) / len(said)
    best = None
    for request_hash, text in heard:
        words = normalize_words(plain_text(text))
        if len(words) < 2:
            continue
        share = _shared(said, words) / len(said)
        if (share >= MIN_HEARD_SHARE and share >= expected_share + MIN_MARGIN
                and (best is None or share > best["heard_share"])):
            best = {"version": VERSION, "heard_request_hash": request_hash,
                    "heard_share": round(share, 3), "expected_share": round(expected_share, 3)}
    return best


def heard_turns():
    """The current chapter transaction's history turns as ``(request_hash, text)``."""
    from .continuity import current
    scope = current()
    if scope is None or not scope.get("history"):
        return []
    return [(take["identity"]["request_hash"], take["marker"]["text"]) for take in scope["takes"]]


def attempt_evidence(report, metadata, expected, heard=None):
    """Echo evidence for one generated attempt that heard history and failed."""
    if not isinstance(report, dict) or report.get("passed") is True:
        return None
    if isinstance(report.get("history_echo"), dict):
        return report["history_echo"]
    if "text_mismatch" not in (report.get("reasons") or []):
        return None
    if not (metadata or {}).get("history_turns"):
        return None
    return detect(expected, report.get("transcript"), heard_turns() if heard is None else heard)


def candidate_echo(record, expected, heard=None):
    """Echo evidence for any failed retained take, including a repair join."""
    report = record.get("report") or {}
    if report.get("passed") is True:
        return None
    if isinstance(report.get("history_echo"), dict):
        return report["history_echo"]
    return detect(expected, report.get("transcript"), heard_turns() if heard is None else heard)


def rejected_pick(record, heard=None):
    """Echo evidence when ``record`` is a published exhaustion choice of an echo."""
    metadata = record.get("engine_metadata") or {}
    if "exhaustion_selection" not in metadata:
        return None
    report = record.get("report") or {}
    original = report.get("original_failed_report") or {}
    if isinstance(original.get("history_echo"), dict):
        return original["history_echo"]
    transcript = original.get("transcript") or report.get("transcript")
    return detect(report.get("expected_text", ""), transcript, heard_turns() if heard is None else heard)


def applied_mode(record):
    marker = (record.get("engine_metadata") or {}).get("history_recovery")
    return marker.get("applied") if isinstance(marker, dict) else None


def _attempt_records(directory, request_hash):
    records = []
    for path in sorted(Path(directory).glob("attempt-*.json")):
        if not re.fullmatch(r"attempt-\d+\.json", path.name):
            continue
        record = read_json(regular_file(path))
        if isinstance(record, dict) and record.get("request_hash") == request_hash:
            records.append(record)
    return sorted(records, key=lambda record: record.get("attempt", 0))


def next_mode(records, expected, heard=None):
    """The ladder rung for the next attempt: one past the furthest rung that echoed."""
    heard = heard_turns() if heard is None else heard
    rung = -1
    for record in records:
        applied = applied_mode(record)
        position = LADDER.index(applied) if applied in LADDER else -1
        echoed = attempt_evidence(record.get("report"), record.get("engine_metadata"), expected, heard)
        rung = max(rung, position + 1 if echoed else position)
    return LADDER[min(rung, len(LADDER) - 1)] if rung >= 0 else None


def ladder_mode(directory, request_hash, expected):
    """Next rung for a take directory; None while no attempt of it has echoed."""
    heard = heard_turns()
    if not heard:
        return None
    return next_mode(_attempt_records(directory, request_hash), expected, heard)


def _single_speaker(text, history):
    tags = set(SPEAKER_TAG.findall(text))
    for turn_text, _ in history:
        tags.update(SPEAKER_TAG.findall(turn_text))
    return tags <= {"0"}


def role_tagged(text, kind):
    """Lead the first turn of ``text`` with its role, merged into any leading direction."""
    role = "dialogue" if kind == "dialogue" else "narration"
    match = _LEADING_TAG.match(text)
    prefix, body = (match.group(1), text[match.end():]) if match else ("", text)
    if prefix and not prefix.endswith(" "):
        prefix += " "
    direction = _LEADING_BRACKET.match(body)
    if direction:
        inner = direction.group(1).strip()
        return f"{prefix}[{role}; {inner}] {body[direction.end():]}" if inner else f"{prefix}[{role}] {body[direction.end():]}"
    return f"{prefix}[{role}] {body.lstrip()}"


def apply_recovery(requested, text, history, layouts, *, kind):
    """``(applied, text, history, layout)`` for the requested rung or the first usable one after it."""
    if requested is None or not history:
        return None, text, history, "directing"
    if requested not in LADDER:
        raise ValueError("Unknown history recovery mode")
    single = _single_speaker(text, history)
    for mode in LADDER[LADDER.index(requested):]:
        if mode == "role_tag" and single:
            return mode, role_tagged(text, kind), history, "directing"
        if (mode == "streaming" and single and "streaming" in layouts
                and "[" not in SPEAKER_TAG.sub("", text)):
            return mode, text, history, "streaming"
        if mode == "none":
            return mode, text, [], "directing"
    raise AssertionError("unreachable")


def archive(directory, names, evidence):
    """Retire files of a rejected echo choice under evidence-bound names; nothing is deleted."""
    directory = Path(directory)
    suffix = digest(evidence)[:12]
    moved = []
    for name in names:
        path = directory / name
        if not path.exists() and not path.is_symlink():
            continue
        stem, dot, extension = name.partition(".")
        target = directory / (f"{stem}.history-echo-{suffix}{dot}{extension}" if dot else f"{stem}.history-echo-{suffix}")
        if target.exists() or target.is_symlink():
            raise ValueError(f"A retired history-echo record already exists: {target.name}")
        path.rename(target)
        moved.append(target.name)
    if moved:
        write_json(directory / f"history-echo-{suffix}.json",
                   {"version": VERSION, "evidence": evidence, "retired": moved})
    return moved
