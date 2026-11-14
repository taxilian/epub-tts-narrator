"""Finish a take that said the start of its passage and stopped, instead of discarding it.

Drama 3 sometimes speaks the first part of a passage correctly and then ends,
or goes silent, before the last words. Such a take is rejected for its missing
words, and until now every word of it was generated again. When the take's
transcript is the start of the passage, this keeps that start: it cuts the
take just after the last word heard (``completion_point``, from the
recognizer's word timings), generates only the missing words with the kept
start as the latest chapter history turn, so they continue the same reading,
and joins the two. The joined take is then verified in full like any take, so
a seam that changes or loses words is rejected and ordinary recovery follows.

A completed take is accepted speech but not one generation of its text, so it
never becomes chapter history (``continuity._NOT_HISTORY``); its record keeps
``engine_metadata.completion`` naming the take it finished, where it was cut,
the words generated and the seed.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import tempfile

COMPLETION_VERSION = 1
# A take must have said at least this many words for its start to be kept.
MIN_HEARD_WORDS = 3
# Recognizer word ends are approximate: the cut moves forward to the first
# quiet window within this span after the last word, then keeps a short tail.
CUT_SEARCH_SECONDS = 0.6
QUIET_DBFS = -50.0
WINDOW_SECONDS = 0.025
CUT_TAIL_SECONDS = 0.08
# Metadata of generations whose text is not the passage's own wording.
_NOT_COMPLETABLE = {"speech_resolution", "delivery_recovery", "direction_repair", "recovery",
                    "clip_repair", "fragment_recovery", "exhaustion_selection", "completion"}
_APOSTROPHES = str.maketrans({mark: "'" for mark in "’‘ʼ`´′＇"})
_LEADING_CONTROLS = re.compile(r"^(<\|speaker:\d+\|>\s*)((?:\[[^\[\]]*\]\s*)*)")


def _word(token):
    """A token as comparable letters and digits: case, punctuation and apostrophe forms ignored."""
    return re.sub(r"[^\w]", "", token.translate(_APOSTROPHES).casefold().replace("'", ""))


def completable(metadata):
    """Whether a take's generation spoke the passage's own text (not a rewrite, join or completion)."""
    return not _NOT_COMPLETABLE.intersection(metadata or {})


def completion_point(report, text):
    """Where the missing words start in ``text`` and where to cut the take, or ``None``.

    Only a take rejected for its words alone, with word timings, whose heard
    words are the start of ``text`` (they end on the two words before the
    first missing one, near the same position, and mostly match in order).
    Returns ``{"start", "cut_seconds", "heard_words", "missing_words"}``.
    """
    if (not isinstance(report, dict) or report.get("passed") is not False
            or report.get("reasons") != ["text_mismatch"]
            or report.get("word_timing_status") != "available" or not isinstance(text, str)):
        return None
    heard = [word for word in report.get("words") or []
             if isinstance(word, dict) and isinstance(word.get("word"), str) and _word(word["word"])
             and isinstance(word.get("end"), (int, float))]
    if len(heard) < MIN_HEARD_WORDS:
        return None
    expected = [(match.start(), _word(match.group())) for match in re.finditer(r"\S+", text)]
    expected = [item for item in expected if item[1]]
    heard_words = [_word(word["word"]) for word in heard]
    last = tuple(heard_words[-2:])
    ends = [index for index in range(2, len(expected))
            if (expected[index - 2][1], expected[index - 1][1]) == last]
    if not ends:
        return None
    count = len(heard_words)
    first_missing = min(ends, key=lambda index: abs(index - count))
    if abs(first_missing - count) > max(2, count // 5):
        return None
    same = sum(a == b for a, b in zip(heard_words, (word for _, word in expected[:first_missing])))
    if same < 0.6 * min(count, first_missing):
        return None
    return {"start": expected[first_missing][0], "cut_seconds": float(heard[-1]["end"]),
            "heard_words": first_missing, "missing_words": len(expected) - first_missing}


def _samples(path):
    from .stored_audio import open_wave
    with open_wave(path) as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError("Take completion needs mono 16-bit takes")
        return stream.getframerate(), stream.readframes(stream.getnframes())


def cut_sample(pcm, rate, cut_seconds):
    """The sample after which a take is kept: past the last word's end, at the next quiet window."""
    import array
    samples = array.array("h", pcm)
    window = max(1, round(rate * WINDOW_SECONDS))
    threshold = (10 ** (QUIET_DBFS / 20) * 32768) ** 2
    start = min(len(samples), max(0, round(cut_seconds * rate)))
    limit = min(len(samples), start + round(CUT_SEARCH_SECONDS * rate))
    position = limit
    for offset in range(start, limit, window):
        chunk = samples[offset:offset + window]
        if chunk and sum(value * value for value in chunk) / len(chunk) < threshold:
            position = offset
            break
    return min(len(samples), position + round(CUT_TAIL_SECONDS * rate))


def history_turn(source_metadata, directory, cut, total, text_prefix):
    """The kept start as a history turn: ``(tagged text, codes cut at the same point, speakers)``."""
    from .continuity_codes import load_codes
    marker = (source_metadata or {}).get("continuity_codes")
    if not isinstance(marker, dict):
        return None
    payload = load_codes(Path(directory) / marker["file"], marker)
    frames = len(payload["codes"][0])
    keep = max(1, min(frames, math.ceil(cut * frames / max(1, total))))
    tag, controls = _LEADING_CONTROLS.match(payload["text"]).groups()
    words = " ".join(text_prefix.split())
    return {"text": f"{tag}{controls}{words}".rstrip(), "codes": [row[:keep] for row in payload["codes"]],
            "speakers": list(payload["speakers"])}


def complete(source, generate_tail, raw, seed, *, text, directory, config):
    """Generate the missing words after ``source``'s kept start and write the joined take to ``raw``.

    ``source`` is ``{"attempt", "wav", "metadata", "point"}`` for the rejected
    take; ``generate_tail(start, path, seed)`` speaks ``text[start:]`` in the
    passage's voice, delivery and controls. Returns the engine metadata for
    the joined take.
    """
    import wave
    from .assembly import normalize_clip
    from .continuity import continuation_turn
    from .stored_audio import audio_hash
    point = source["point"]
    rate, pcm = _samples(source["wav"])
    total = len(pcm) // 2
    cut = cut_sample(pcm, rate, point["cut_seconds"])
    turn = history_turn(source["metadata"], directory, cut, total, text[:point["start"]])
    with tempfile.TemporaryDirectory(prefix=".completion-", dir=Path(raw).parent) as temporary:
        name = Path(raw).name.removesuffix(".wav")
        tail_raw = Path(temporary) / f"{name}-completion-tail-raw.wav"
        tail = Path(temporary) / f"{name}-completion-tail.wav"
        with continuation_turn(turn):
            metadata = dict(generate_tail(point["start"], tail_raw, seed))
        normalize_clip(tail_raw, tail, sample_rate=rate, ffmpeg=config["render"]["ffmpeg"])
        tail_rate, tail_pcm = _samples(tail)
        if tail_rate != rate:
            raise ValueError("Take completion produced audio at another sample rate")
        tail_sha = audio_hash(tail)
        with wave.open(str(raw), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(pcm[:cut * 2] + tail_pcm)
    metadata.pop("generated_codes", None)
    metadata["completion"] = {
        "version": COMPLETION_VERSION, "source_attempt": source["attempt"],
        "source_audio_sha256": audio_hash(source["wav"]), "kept_seconds": round(cut / rate, 3),
        "tail_start": point["start"], "tail_text": text[point["start"]:], "tail_seed": seed,
        "tail_audio_sha256": tail_sha, "history_turn": turn is not None,
        "tail_synthesis_text": metadata.get("synthesis_text")}
    metadata["synthesis_text"] = text
    return metadata


def tail_generator(text, synthesize_part, *, offset=0):
    """``generate_tail(start, path, seed)`` from a recovery part synthesizer, or ``None``.

    A synthesizer that needs the span (``needs_span``, scoped controls) gets
    the tail's exact span of the line, ``text`` starting at ``offset`` in it
    (a sentence part's start); its ``check_span`` refuses a tail that would
    cut a control.
    """
    if synthesize_part is None:
        return None
    spanned = getattr(synthesize_part, "needs_span", False)

    def generate_tail(start, path, seed):
        if spanned:
            return synthesize_part(text[start:], path, seed, span=(offset + start, offset + len(text)))
        return synthesize_part(text[start:], path, seed)

    def supports(start):
        if not 0 < start < len(text) or not text[start:].strip():
            return False
        check = getattr(synthesize_part, "check_span", None)
        if spanned and callable(check):
            try:
                check((offset + start, offset + len(text)))
            except ValueError:
                return False
        return True
    generate_tail.supports = supports
    return generate_tail
