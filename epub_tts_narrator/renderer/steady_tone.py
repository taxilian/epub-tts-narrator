"""Reject a take carrying a steady high-pitched tone, and retire takes in use that carry one.

Drama 3 sometimes turns a voice's sibilance into a steady, narrow tone far
above the voice (measured at 7, 10 and 13 kHz on one narrator recording whose
own reference has none): a whine that rides over the speech, and often
through its pauses. A chapter take hears the takes before it, so once a take
carries the tone, later takes copy it. Speech has no steady narrow band up
there: its sibilants come and go and are broad.

``measure`` averages the take's voiced spectrum between ``LOW_HZ`` and
``HIGH_HZ`` and finds the narrow band standing furthest above its neighbours;
``presence`` is the share of voiced frames in which that band alone stands
out. A take fails (``REASON``) when both reach ``MIN_LEVEL_DB`` and
``MIN_PRESENCE``. Calibrated on belgariad-5: no other voice's take reached
9 dB in 200 sampled, while the reported narrator takes measured 11 and 20 dB.
A rejected take is never chapter history, so the tone stops spreading.
"""

from __future__ import annotations

import time
from pathlib import Path

VERSION = 1
REASON = "steady_tone"
LOW_HZ, HIGH_HZ = 5000, 16000
MIN_LEVEL_DB = 9.0
MIN_PRESENCE = 0.5
# The band must be audible: within this many dB of the take's loudest
# frequency (the measured tones sat 7 to 20 dB below it). Below that, a ratio
# of two near-silent bands means nothing.
AUDIBLE_DB = 50.0
SWEEP_FILENAME = "steady-tone-sweep.json"
_FRAME = 4096
_VOICED_RMS = 0.01
_MTIME_SLACK = 120  # seconds: an acceptance written while a sweep ran is read by the next one


def measure(path):
    """``{"level_db", "frequency_hz", "presence"}`` for a take's strongest steady high band, or ``None``.

    ``None`` when the take has too little voiced audio to judge (it is then
    never rejected for a tone).
    """
    import numpy as np
    from .stored_audio import open_wave
    with open_wave(Path(path)) as audio:
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    window = np.hanning(_FRAME)
    frames = [samples[start:start + _FRAME] for start in range(0, len(samples) - _FRAME, _FRAME // 2)]
    voiced = [frame for frame in frames if np.sqrt(np.mean(frame ** 2)) > _VOICED_RMS]
    if len(voiced) < 4:
        return None
    spectra = np.abs(np.fft.rfft(np.stack(voiced) * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(_FRAME, 1 / rate)

    def masks(centre):
        mid = (freqs >= centre - 100) & (freqs < centre + 100)
        side = (((freqs >= centre - 800) & (freqs < centre - 400))
                | ((freqs >= centre + 400) & (freqs < centre + 800)))
        return mid, side

    mean = spectra.mean(axis=0)
    audible = mean.max() * 10 ** (-AUDIBLE_DB / 10)
    level, centre = -99.0, 0
    for candidate in range(LOW_HZ, HIGH_HZ, 50):
        mid, side = masks(candidate)
        if mean[mid].mean() < audible:
            continue
        ratio = 10 * np.log10(mean[mid].mean() / max(mean[side].mean(), 1e-20))
        if ratio > level:
            level, centre = float(ratio), candidate
    if centre == 0:
        return {"version": VERSION, "level_db": None, "frequency_hz": None, "presence": 0.0}
    mid, side = masks(centre)
    per_frame = 10 * np.log10(spectra[:, mid].mean(axis=1) / np.maximum(spectra[:, side].mean(axis=1), 1e-20))
    return {"version": VERSION, "level_db": round(level, 1), "frequency_hz": centre,
            "presence": round(float(np.mean(per_frame > 6)), 2)}


def found(measured):
    """Whether a measurement is a steady tone this check rejects."""
    return (isinstance(measured, dict) and (measured.get("level_db") or 0) >= MIN_LEVEL_DB
            and measured.get("presence", 0) >= MIN_PRESENCE)


def check(path):
    """The measurement when ``path`` carries a steady tone, else ``None``."""
    measured = measure(path)
    return measured if found(measured) else None


def describe(report):
    """``a steady 7.1 kHz tone`` for a report rejected for one, else ``""``."""
    measured = (report or {}).get(REASON)
    if not isinstance(measured, dict):
        return ""
    return f"a steady {measured['frequency_hz'] / 1000:.1f} kHz tone ({measured['level_db']:.0f} dB above its neighbours)"


def retire_tonal(work, *, notify=None):
    """Retire every take in use that carries a steady tone; their take folders.

    Each acceptance is measured once (``SWEEP_FILENAME`` remembers how far a
    sweep got under this ``VERSION``), so later renders read only takes
    accepted since. A take a person chose or marked good is kept. The take
    folder's acceptance is renamed, never deleted (``take_review.retire_acceptance``).
    """
    from .common import read_json, write_json
    from .take_feedback import load
    from .take_review import retire_acceptance
    work = Path(work)
    clips = work / "clips"
    if not clips.is_dir() or clips.is_symlink():
        return []
    good = {entry.get("audio_sha256") for entry in load(work)["takes"].values() if entry.get("verdict") == "good"}
    marker = work / SWEEP_FILENAME
    saved = read_json(marker) if marker.is_file() and not marker.is_symlink() else {}
    since = saved.get("checked_through", 0) if saved.get("version") == VERSION else 0
    started = time.time()
    paths = [path for path in [*clips.glob("*/accepted.json"), *clips.glob("*/parts/*/accepted.json")]
             if not path.is_symlink() and path.is_file() and path.stat().st_mtime >= since - _MTIME_SLACK]
    if notify is not None and len(paths) > 200:
        notify(f"Checking {len(paths)} take(s) in use for a steady high-pitched tone (once).")
    retired = []
    for path in sorted(paths):
        record = read_json(path)
        if not isinstance(record, dict) or not isinstance(record.get("report"), dict):
            continue
        metadata = record.get("engine_metadata") or {}
        if (record["report"].get("passed") is not True or "manual_selection" in metadata
                or record.get("audio_sha256") in good or not isinstance(record.get("audio_file"), str)
                or not isinstance(record.get("request"), dict)
                or record["request"].get("kind") not in {"speech", "paragraph_speech", "speech_recovery_part"}):
            continue
        try:
            measured = check(path.parent / record["audio_file"])
        except (OSError, EOFError, ValueError):
            continue
        if measured is not None:
            retire_acceptance(path.parent, record, "steady-tone", {"reason": REASON, **measured})
            retired.append(path.parent)
    write_json(marker, {"version": VERSION, "checked_through": started})
    return retired
