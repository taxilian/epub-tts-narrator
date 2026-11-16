"""What chapter history the next take of a failing passage hears: one decision, one precedence.

A take hears its chapter's history (``continuity``): the voice's first turn
and its latest, the latest chosen to say the line's chosen pronunciations
right. When takes keep failing because of what they heard, the next takes
hear less. ``plan`` decides how much from the passage's saved attempt records,
reading only their saved fields, and both ordinary takes (``workflow``) and
clip-repair parts (``clip_recovery``) use it:

1. A take carrying a steady tone (``steady_tone``), repeated takes that said
   nothing, or repeated takes that ran on until Drama 3's length limit
   without finishing their text (a short line such as "Barak said," running
   on in its history's voice; ``runaway``) mean the history itself is the
   trouble. A line with chosen
   pronunciations then hears only the turn saying them right (``"reading"``):
   without it Drama 3 tends to say a familiar look-alike word in a name's
   place. Its takes may carry the tone again (``steady_tone.filtered_fallback``
   then cleans one), but once a take made with only that turn replayed it or
   said nothing, that turn is the trouble too and no history is heard
   (``"none"``). A line without chosen pronunciations hears none at once.
2. Otherwise a take that replayed a history turn (``history_echo``) moves the
   passage one rung along the echo ladder (role tag, streaming layout, none).
   Ladder rungs keep the whole history, so they never follow a tone. A line
   with chosen pronunciations hears the turn saying them right
   (``"reading"``) where the ladder would leave it no history: with no
   history Drama 3 misread a chosen name in most takes (a short line led by
   the name, every take), with a turn saying it right in very few. Only once
   a take made with that turn replayed it or said nothing does the line hear
   none.

Whatever the mode, a take completion keeps its kept start
(``continuity.compose``, ``history_echo.apply_recovery``).
"""

from __future__ import annotations

SILENT_REASONS = frozenset({"audio_too_quiet", "excessive_silence", "silent_audio"})
SILENT_LIMIT = 2  # takes that said (next to) nothing, or ran away, before history is cut


def _reasons(record):
    return set((record.get("report") or {}).get("reasons") or [])


def _applied(record):
    return ((record.get("engine_metadata") or {}).get("history_recovery") or {}).get("applied")


def runaway(record):
    """Whether a take ran on to Drama 3's length limit without finishing its text.

    Not a take the silence watch stopped: that one said its start and stopped
    (``take_completion`` finishes it).
    """
    from .generation_failures import drama3_truncation
    report = record.get("report") or {}
    return report.get("generation_stopped") != "silence" and drama3_truncation(report)


def plan(records, expected, *, has_readings, ladder=True, heard=None):
    """``(mode, reason)`` for the next take: ``mode`` None (ordinary), a ladder rung, ``"reading"`` or ``"none"``."""
    from .history_echo import RUNG_DESCRIPTIONS, attempt_evidence, next_mode
    from .steady_tone import REASON as TONE
    records = list(records)
    tonal = any(TONE in _reasons(record) for record in records)
    silent = sum(bool(SILENT_REASONS & _reasons(record)) for record in records)
    ran_away = sum(runaway(record) for record in records)

    def heard_mode(index):
        # A take that failed in the engine saved no metadata; it heard what
        # this plan chose from the takes before it.
        applied = _applied(records[index])
        if applied is None and not (records[index].get("engine_metadata") or {}):
            applied = plan(records[:index], expected, has_readings=has_readings, ladder=ladder, heard=heard)[0]
        return applied
    # Once a take made with only the turn saying the readings right replayed
    # it, said nothing or ran away, that turn is part of the trouble.
    reading_failed = any(
        (SILENT_REASONS & _reasons(record) or (record.get("report") or {}).get("history_echo")
         or runaway(record)
         or attempt_evidence(record.get("report"), record.get("engine_metadata"), expected, heard))
        and heard_mode(index) == "reading"
        for index, record in enumerate(records))
    reading = has_readings and not reading_failed
    if tonal or silent >= SILENT_LIMIT or ran_away >= SILENT_LIMIT:
        cause = ("a take carrying a steady tone" if tonal else "takes that said nothing" if silent >= SILENT_LIMIT
                 else "takes that ran on without finishing")
        if reading:
            return "reading", f"after {cause}, only the turn saying its chosen pronunciations right"
        return "none", f"after {cause}, {RUNG_DESCRIPTIONS['none']}"
    if ladder and heard:
        rung = next_mode(records, expected, heard)
        if rung == "none" and reading:
            return "reading", "after its history was replayed, only the turn saying its chosen pronunciations right"
        if rung is not None:
            return rung, RUNG_DESCRIPTIONS[rung]
    return None, None


def for_request(directory, request_hash, expected, request, *, ladder=True):
    """``plan`` for one take folder's saved attempts of ``request``."""
    from .continuity import _chosen_readings
    from .history_echo import _attempt_records, heard_turns
    records = _attempt_records(directory, request_hash)
    return plan(records, expected, has_readings=bool(_chosen_readings(request)), ladder=ladder,
                heard=heard_turns())
