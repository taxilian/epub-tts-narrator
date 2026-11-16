"""Bring each take of a chapter to about the chapter's speech level, keeping intended contrast.

Drama 3 takes of one chapter differ by about 8 dB in speech level for no
reason in the text (measured on one book), and chapter loudness
normalization moves the whole chapter, not one take against another. At
assembly each take's speech level (``speech_level``: the mean of its voiced
50 ms frames) is compared with where it should sit: the chapter's typical
take (the median of takes with an ordinary delivery), moved down
``SOFT_DB`` for a whispered, soft or murmured line and up ``LOUD_DB`` for a
shouted one (``expected_offset``, from the take's own directions). A take
within ``tolerance_db`` of that is left alone, so ordinary contrast stays;
one further off is brought back to the edge of that band (``gains``),
never by more than ``MAX_GAIN_DB`` and never past ``PEAK_CEILING_DBFS``.

Only the assembled chapter changes: saved takes are never rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path

VERSION = 1
SOFT_DB = -6.0
LOUD_DB = 4.0
MAX_GAIN_DB = 12.0
PEAK_CEILING_DBFS = -1.0
_FRAME_SECONDS = 0.05
_VOICED_DBFS = -45.0
_SOFT = re.compile(r"\b(whisper\w*|soft(ly)?|quiet(ly)?|murmur\w*|hush\w*|under (his|her|their) breath|"
                   r"barely audible|faint(ly)?|breath(y|ed)?|mutter\w*)\b")
_LOUD = re.compile(r"\b(shout\w*|yell\w*|scream\w*|bellow\w*|roar\w*|loud(ly)?|cr(y|ies|ied) out|"
                   r"thunder\w*|bark\w*)\b")


def speech_level(path):
    """``(level_db, peak_dbfs)`` of a take's speech, or ``None`` when it has no voiced audio."""
    import numpy as np
    from .stored_audio import open_wave
    with open_wave(Path(path)) as audio:
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    size = max(1, int(rate * _FRAME_SECONDS))
    count = len(samples) // size
    if not count:
        return None
    frames = samples[:count * size].reshape(count, size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    voiced = rms[rms > 10 ** (_VOICED_DBFS / 20)]
    if not len(voiced):
        return None
    peak = float(np.max(np.abs(samples)))
    return float(20 * np.log10(np.mean(voiced))), float(20 * np.log10(max(peak, 1e-9)))


def _directions(chunk):
    """The words that direct a chunk's delivery: its direction(s) and any control covering most of it."""
    texts = [str((chunk.get("segment") or {}).get("delivery", {}).get("direction") or ""),
             str(chunk.get("expressive_direction") or "")]
    length = max(1, len(chunk.get("spoken_text") or ""))
    for control in (chunk.get("expressive_performance") or {}).get("controls") or []:
        if not isinstance(control, dict) or control.get("kind") != "delivery":
            continue
        span = (control.get("end") or 0) - (control.get("start") or 0)
        if span >= length / 2:
            texts += [str(control.get("tag") or ""), str(control.get("direction") or "")]
    return " ".join(texts).lower()


def expected_offset(clip):
    """How far from the chapter's typical level this take should sit (dB): soft below, shouted above."""
    turns = clip.get("paragraph_turns") or [clip]
    offsets = set()
    for turn in turns:
        words = _directions(turn)
        offsets.add(SOFT_DB if _SOFT.search(words) else LOUD_DB if _LOUD.search(words) else 0.0)
    # A paragraph mixing deliveries keeps no particular expectation.
    return offsets.pop() if len(offsets) == 1 else 0.0


def gains(clips, *, tolerance_db):
    """``{index: gain_db}`` for the clips (``wav_path`` set) that should be moved, with what was measured.

    Returns ``(gains, report)``; ``report`` lists every measured take's
    level, expected offset and applied gain for the chapter record.
    """
    measured = {}
    for index, clip in enumerate(clips):
        if clip.get("wav_path") is None or "speaker_lanes" in clip:
            continue
        level = speech_level(clip["wav_path"])
        if level is not None:
            measured[index] = (level[0], level[1], expected_offset(clip))
    ordinary = sorted(level for level, _, offset in measured.values() if offset == 0.0)
    pool = ordinary or sorted(level for level, _, _ in measured.values())
    if not pool:
        return {}, {"version": VERSION, "reference_db": None, "takes": []}
    reference = pool[len(pool) // 2]
    result, rows = {}, []
    for index, (level, peak, offset) in sorted(measured.items()):
        deviation = level - (reference + offset)
        gain = 0.0
        if abs(deviation) > tolerance_db:
            gain = -(deviation - tolerance_db) if deviation > 0 else -(deviation + tolerance_db)
            gain = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain, PEAK_CEILING_DBFS - peak))
            gain = round(gain, 2)
        if gain:
            result[index] = gain
        rows.append({"chunk_id": clips[index].get("chunk_id"), "level_db": round(level, 2),
                     "expected_offset_db": offset, "gain_db": gain})
    return result, {"version": VERSION, "reference_db": round(reference, 2), "tolerance_db": tolerance_db,
                    "takes": rows}


def gained_copy(source, target, gain_db):
    """Write ``source`` scaled by ``gain_db`` to ``target`` (same length and format); the target path."""
    import wave
    import numpy as np
    from .stored_audio import open_wave
    with open_wave(Path(source)) as audio:
        params = audio.getparams()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)
    scaled = np.clip(np.round(samples * 10 ** (gain_db / 20)), -32768, 32767).astype("<i2")
    with wave.open(str(target), "wb") as output:
        output.setparams(params)
        output.writeframes(scaled.tobytes())
    return Path(target)
