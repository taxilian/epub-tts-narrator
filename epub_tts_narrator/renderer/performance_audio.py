"""Locate intentional pauses without weakening ordinary acoustic checks."""

from ..performance_plan import validate_performance


PERFORMANCE_AUDIO_VERSION = 1
_TIMING_TOLERANCE = .12
_DURATION_TOLERANCE = .25


def pause_edges(record, text):
    """Preserve only explicitly requested outer silence; accepted WAVs stay exact."""
    if record is None:
        return {"leading_pause_ms": 0, "trailing_pause_ms": 0}
    controls = validate_performance(record, text)["controls"]
    if any(control["kind"] == "pause" for control in controls):
        # The model removes outer quotation/formatting characters. A pause just
        # inside an opening quote is consequently at the audible clip edge.
        from .performance_compiler import trim_performance
        text, trimmed = trim_performance(text, record)
        controls = trimmed["controls"]
    return {"leading_pause_ms": max((control["duration_ms"] for control in controls
                                      if control["kind"] == "pause" and control["offset"] == 0), default=0),
            "trailing_pause_ms": max((control["duration_ms"] for control in controls
                                       if control["kind"] == "pause" and control["offset"] == len(text)), default=0)}


def planned_pause_adjustment(record, text, words, metrics):
    """Credit measured quiet only at an exact, complete ASR word boundary.

    Semantic transcript confirmation is not enough to guess an event's time.
    Missing/ambiguous timing leaves all normal silence limits in force. Each
    located interval is bounded by that requested pause plus 250ms; a pause
    elsewhere in the clip cannot explain a stall at a different word.
    """
    from .quality import normalize_words
    canonical = validate_performance(record, text)
    pauses = [control for control in canonical["controls"] if control["kind"] == "pause"]
    result = {"version": PERFORMANCE_AUDIO_VERSION, "matched_pauses": [],
              "unmatched_pause_offsets": [control["offset"] for control in pauses],
              "timing_status": "unavailable", "inline_vocal_events": "not_acoustically_confirmed",
              "silence_fraction": metrics["silence_fraction"],
              "longest_internal_silence_seconds": metrics["longest_internal_silence_seconds"]}
    if not pauses or not words:
        return result
    expected = normalize_words(text)
    timed_tokens = [(token, word) for word in words for token in normalize_words(word["word"])]
    if [token for token, _ in timed_tokens] != expected:
        result["timing_status"] = "lexical_alignment_unavailable"
        return result
    result["timing_status"] = "available"
    gaps = []
    for control in pauses:
        offset = control["offset"]
        before, after = normalize_words(text[:offset]), normalize_words(text[offset:])
        if before + after != expected:
            # Reject a split inside a lexical word or across number syntax.
            continue
        index = len(before)
        left = timed_tokens[index - 1][1] if index else None
        right = timed_tokens[index][1] if index < len(timed_tokens) else None
        if left is not None and left is right:
            continue
        start = left["end"] if left else 0.0
        end = right["start"] if right else metrics["duration_seconds"]
        if end > start:
            gaps.append((control, start, end))
    matched, used, credited, remaining_internal = [], set(), 0.0, []
    for interval in metrics.get("silence_intervals", []):
        start, end = interval["start"], interval["end"]
        duration = end - start
        found = None
        for index, (control, gap_start, gap_end) in enumerate(gaps):
            if (index not in used and start >= gap_start - _TIMING_TOLERANCE
                    and end <= gap_end + _TIMING_TOLERANCE
                    and duration <= control["duration_ms"] / 1000 + _DURATION_TOLERANCE):
                found = (index, control)
                break
        if found is None:
            if interval["internal"]:
                remaining_internal.append(duration)
            continue
        index, control = found
        used.add(index)
        credited += duration
        matched.append({"offset": control["offset"], "start": start, "end": end,
                        "requested_seconds": control["duration_ms"] / 1000,
                        "observed_seconds": duration})
    result["matched_pauses"] = matched
    result["unmatched_pause_offsets"] = [control["offset"] for control in pauses
                                         if control["offset"] not in {item["offset"] for item in matched}]
    if credited:
        duration = metrics["duration_seconds"]
        remaining = duration - credited
        silence = max(0.0, duration * metrics["silence_fraction"] - credited)
        result["silence_fraction"] = silence / remaining if remaining > 0 else 1.0
        result["longest_internal_silence_seconds"] = max(remaining_internal, default=0.0)
    return result
