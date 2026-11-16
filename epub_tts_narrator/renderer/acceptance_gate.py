"""The one check every take passes before it becomes its passage's accepted take.

Takes are accepted on many paths: an ordinary verified take, a learned
recognizer spelling or a skipped-word acceptance turning a word mismatch into a
pass, a speech resolver's vote, a retained or saved take rechecked, a clip
repair (new or restored), the best-effort chooser and a tone-filtered copy.
Whatever path proposes a take, ``refusal`` says why it must never be published:

- a person listened and marked it bad (or truncated);
- it misread a pronunciation a person chose (``name_shape.misread``; on paths
  that skip ordinary word verification a chosen name that is missing counts
  too);
- it replays a heard history turn instead of its passage (``history_echo``);
- it carries a steady high-pitched tone, unless it is the filtered copy.

A take a person chose themselves (``manual_selection``) is their decision and
is never refused. Each caller decides what a refusal means on its path: a new
take counts as a failed attempt (``refused_report``), a saved or restored take
is simply not republished.
"""

from __future__ import annotations

# Acceptances that bypass ordinary word verification, so a chosen name must be
# shown present, not merely not misread.
RESCUED_METHODS = frozenset({"heard_variant", "skipped_source_word", "recurring_resolution"})
REFUSED = "acceptance_refused"


def rescued(record):
    """Whether a record was accepted without ordinary word verification."""
    report = record.get("report") or {}
    metadata = record.get("engine_metadata") or {}
    return (report.get("text_confirmation_method") in RESCUED_METHODS
            or "acceptance_policy" in report or "exhaustion_selection" in metadata)


def refusal(record, directory):
    """Why ``record`` must not become its passage's accepted take, or ``None``."""
    metadata = record.get("engine_metadata") or {}
    if "manual_selection" in metadata:
        return None
    from .take_feedback import judged_bad_audio
    if record.get("audio_sha256") in judged_bad_audio(directory):
        return "a person marked it bad"
    from .name_shape import misread
    words = misread(record, require_heard=rescued(record))
    if words:
        return f"it misread a chosen pronunciation ({words})"
    report = record.get("report") or {}
    if report.get("history_echo"):
        return "it replays a heard history turn instead of its passage"
    from .steady_tone import MARKER, REASON as TONE
    if TONE in (report.get("reasons") or []) and MARKER not in metadata:
        return "it carries a steady high-pitched tone"
    return None


def refused_report(report, why):
    """``report`` as a failure for ``why`` (a misread name keeps its own reason)."""
    from .name_shape import REASON as MISREAD
    reason = MISREAD if why.startswith("it misread") else REFUSED
    return {**report, "passed": False, "reasons": [reason], "acceptance_refused": why,
            "text_confirmation_method": None}
