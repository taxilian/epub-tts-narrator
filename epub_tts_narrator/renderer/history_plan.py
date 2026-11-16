"""What chapter history the next take of a failing passage hears: one decision, one precedence.

A take hears its chapter's history (``continuity``): the voice's first turn
and its latest, the latest chosen to say the line's chosen pronunciations
right. When takes keep failing because of what they heard, the next takes
hear less. ``plan`` decides how much from the passage's saved attempt records,
reading only their saved fields, and both ordinary takes (``workflow``) and
clip-repair parts (``clip_recovery``) use it:

1. A take carrying a steady tone (``steady_tone``) or repeated takes that said
   nothing mean the history itself is the trouble. A line with chosen
   pronunciations then hears only the turn saying them right (``"reading"``):
   without it Drama 3 tends to say a familiar look-alike word in a name's
   place. Its takes may carry the tone again (``steady_tone.filtered_fallback``
   then cleans one), but once a take made with only that turn replayed it or
   said nothing, that turn is the trouble too and no history is heard
   (``"none"``). A line without chosen pronunciations hears none at once.
2. Otherwise a take that replayed a history turn (``history_echo``) moves the
   passage one rung along the echo ladder (role tag, streaming layout, none).
   Ladder rungs keep the whole history, so they never follow a tone.

Whatever the mode, a take completion keeps its kept start
(``continuity.compose``, ``history_echo.apply_recovery``).
"""

from __future__ import annotations

SILENT_REASONS = frozenset({"audio_too_quiet", "excessive_silence", "silent_audio"})
SILENT_LIMIT = 2  # takes that said (next to) nothing before history is cut


def _reasons(record):
    return set((record.get("report") or {}).get("reasons") or [])


def _applied(record):
    return ((record.get("engine_metadata") or {}).get("history_recovery") or {}).get("applied")


def plan(records, expected, *, has_readings, ladder=True, heard=None):
    """``(mode, reason)`` for the next take: ``mode`` None (ordinary), a ladder rung, ``"reading"`` or ``"none"``."""
    from .history_echo import RUNG_DESCRIPTIONS, attempt_evidence, next_mode
    from .steady_tone import REASON as TONE
    tonal = any(TONE in _reasons(record) for record in records)
    silent = sum(bool(SILENT_REASONS & _reasons(record)) for record in records)
    if tonal or silent >= SILENT_LIMIT:
        cause = "a take carrying a steady tone" if tonal else "takes that said nothing"
        reading_failed = any(
            _applied(record) == "reading"
            and (SILENT_REASONS & _reasons(record) or (record.get("report") or {}).get("history_echo")
                 or attempt_evidence(record.get("report"), record.get("engine_metadata"), expected, heard))
            for record in records)
        if has_readings and not reading_failed:
            return "reading", f"after {cause}, only the turn saying its chosen pronunciations right"
        return "none", f"after {cause}, {RUNG_DESCRIPTIONS['none']}"
    if ladder and heard:
        rung = next_mode(records, expected, heard)
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
