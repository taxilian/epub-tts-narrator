"""A take whose only fault is one long pause is checked for its words.

The internal-silence gate rejects a take before recognition, so a take that
spoke every word but paused for three seconds between sentences used to cost a
new synthesis. Assembly already sets every interior quiet stretch of an
accepted take to the sentence/clause target (``take_pauses.boundary_edits``,
which caps a stall anywhere in the take), so the published audio never carries
the stall. Such a take is verified again with the silence limit raised to
``LIMIT_FACTOR`` times the configured one: recognition, comparison and judges
all run exactly as for any take, and every other acoustic gate is unchanged.

Recovery is offered only when assembly is certain to shorten the pause: pause
normalization on, a nonzero pause target, no censorship (masked clips keep
their own timing), no simultaneous lanes, and no planned pause inside the
take. A recovered take never becomes chapter history (``continuity``), so its
long silence cannot teach later takes to stall.
"""

from __future__ import annotations

from copy import deepcopy

STALL_RECOVERY_VERSION = 1
LIMIT_FACTOR = 2.0
MARKER = "silence_stall_recovery"


def limit_seconds(config, request, text, performance=None):
    """The raised silence limit when assembly will shorten the pause, else ``None``."""
    from .quality import validate_policy
    render = config.get("render", {})
    if not (render.get("normalize_pauses", True) and render.get("normalize_take_pauses", True)):
        return None
    if max(render.get("sentence_pause_ms", 500), render.get("clause_pause_ms", 250)) <= 0:
        return None
    if config.get("censorship", {}).get("enabled", False):
        return None
    chunk = request.get("chunk") if isinstance(request, dict) else None
    if not isinstance(request, dict) or "performer" in request or (
            isinstance(chunk, dict) and "speaker_lanes" in chunk):
        return None
    if performance is not None:
        from .assembly import _interior_pause
        if _interior_pause(performance, text):
            return None
    return validate_policy(config.get("quality"))["max_internal_silence_seconds"] * LIMIT_FACTOR


def stalled_only(report):
    """A finished report rejected for one long interior pause and nothing else."""
    return (isinstance(report, dict) and report.get("passed") is not True
            and report.get("reasons") == ["internal_silence_stall"] and MARKER not in report)


def recover(report, wav, text, verifier, config, request, *, performance=None, notify=print):
    """Return the report for a stalled take verified for its words, or ``report`` unchanged."""
    if not stalled_only(report):
        return report
    limit = limit_seconds(config, request, text, performance)
    if limit is None:
        return report
    longest = (report.get("metrics") or {}).get("longest_internal_silence_seconds")
    marker = {"version": STALL_RECOVERY_VERSION, "method": "assembly_pause_cap",
              "longest_internal_silence_seconds": longest, "limit_seconds": limit,
              "stalled_report": deepcopy(report)}
    if not isinstance(longest, (int, float)) or longest > limit:
        # A pause this long is more likely a stalled model than a slow reader.
        return {**report, MARKER: {**marker, "decision": "too_long"}}
    from .quality import validate_audio
    policy = {**config.get("quality", {}), "max_internal_silence_seconds": limit}
    options = {"performance": performance} if performance is not None else {}
    checked = validate_audio(wav, text, verifier, policy, purpose="speech", **options)
    checked[MARKER] = {**marker, "decision": "accepted" if checked.get("passed") is True else "words_checked"}
    if checked.get("passed") is True:
        notify(f"  The take paused {longest:.1f} seconds but every word verified; "
               "assembly shortens the pause to the usual length.")
    return checked


def recovered(report):
    """Whether an accepted report was accepted through stall recovery."""
    return isinstance(report, dict) and isinstance(report.get(MARKER), dict)
