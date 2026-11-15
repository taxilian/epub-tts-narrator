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

from contextlib import contextmanager
from contextvars import ContextVar
import time
import wave
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
# A short take of one held sound (an "Oh…") keeps steady resonances that are
# not a tone; only takes with this much voiced audio are judged.
MIN_VOICED_SECONDS = 2.0
_VOICED_RMS = 0.01
_MTIME_SLACK = 120  # seconds: an acceptance written while a sweep ran is read by the next one


def measure(path, *, exclude=()):
    """``{"level_db", "frequency_hz", "presence"}`` for a take's strongest steady high band, or ``None``.

    ``None`` when the take has too little voiced audio to judge (it is then
    never rejected for a tone). Bands within ``EXCLUDE_HZ`` of a frequency in
    ``exclude`` (ones already filtered out) are not considered: a notch dips
    its neighbours, which would make the next band look like a tone.
    """
    import numpy as np
    from .stored_audio import open_wave
    with open_wave(Path(path)) as audio:
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    window = np.hanning(_FRAME)
    frames = [samples[start:start + _FRAME] for start in range(0, len(samples) - _FRAME, _FRAME // 2)]
    voiced = [frame for frame in frames if np.sqrt(np.mean(frame ** 2)) > _VOICED_RMS]
    if len(voiced) * (_FRAME // 2) / rate < MIN_VOICED_SECONDS:
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
        if any(abs(candidate - frequency) < EXCLUDE_HZ for frequency in exclude):
            continue
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


_FILTERED = ContextVar("steady_tone_filtered", default=())


@contextmanager
def filtered(frequencies):
    """While a filtered take is verified, its notched bands are not judged again (``check``)."""
    token = _FILTERED.set(tuple(frequencies))
    try:
        yield
    finally:
        _FILTERED.reset(token)


def check(path, *, exclude=()):
    """The measurement when ``path`` carries a steady tone, else ``None``."""
    measured = measure(path, exclude=tuple(exclude) + _FILTERED.get())
    return measured if found(measured) else None


def history_mode(directory, request_hash):
    """The history a request's next take hears after a take carrying the tone (``None``: ordinary).

    The tone grows through chapter history (takes made without it measured
    clean), so after one the next takes hear only the turns saying the line's
    chosen readings right (``"reading"``; a line without such turns hears no
    history). Without that turn Drama 3 says a similar familiar word instead
    (Ce'Nedra → Sendaria, every time on belgariad-5), so it is kept even when
    a take made with it carries the tone too: such takes are candidates for
    ``filtered_fallback``. Once a take made with only that turn replayed it
    (a history echo, ``report.history_echo``), the next takes hear no history
    (``"none"``).
    """
    from .common import read_json
    tonal = echoed = False
    for path in sorted(Path(directory).glob("attempt-*.json")):
        try:
            record = read_json(path)
        except (ValueError, OSError):
            continue
        report = record.get("report") if isinstance(record, dict) else None
        if not isinstance(report, dict) or record.get("request_hash") != request_hash:
            continue
        tonal = tonal or REASON in (report.get("reasons") or [])
        recovery = (record.get("engine_metadata") or {}).get("history_recovery") or {}
        echoed = echoed or (recovery.get("applied") == "reading" and bool(report.get("history_echo")))
    return ("none" if echoed else "reading") if tonal else None


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
        except (OSError, EOFError, ValueError, wave.Error):
            # Unreadable audio is the speech cache's to refuse ("audio changed").
            continue
        if measured is not None:
            retire_acceptance(path.parent, record, "steady-tone", {"reason": REASON, **measured})
            retired.append(path.parent)
    write_json(marker, {"version": VERSION, "checked_through": started})
    return retired


FILTER_VERSION = 1
MAX_NOTCHES = 4
EXCLUDE_HZ = 1200
NOTCH_WIDTH_HZ = 1000
# Once a take is being filtered, every steady band this strong is removed,
# not only one strong enough to reject a take.
FILTER_MIN_LEVEL_DB = 6.0
FILTER_MIN_PRESENCE = 0.4
MAX_FILTERED_TRIES = 3
MARKER = "tone_filter"


def _notched(source, target, frequencies, ffmpeg):
    """Write ``source`` with a narrow band-reject at each frequency to ``target`` (PCM16 mono 44.1 kHz)."""
    import subprocess
    chain = ",".join(f"bandreject=f={frequency}:width_type=h:width={NOTCH_WIDTH_HZ}" for frequency in frequencies)
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-af", chain, "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(target)],
                   check=True, capture_output=True)


def filtered_fallback(directory, request, text, verifier, config, *, notify=print):
    """When every take of a passage carries a steady tone: filter the faintest and use it if its words verify.

    Takes rejected only for the tone are tried faintest first (at most
    ``MAX_FILTERED_TRIES``). Each gets a narrow band-reject at every tone
    found (re-measured after each, at most ``MAX_NOTCHES``), is verified in
    full like any take, and on a pass becomes the passage's acceptance with
    the filtering recorded (``MARKER``). The original take is kept. A
    filtered take is never chapter history (its codes still carry the tone).
    Returns ``(audio path, record)``, or ``None`` when none verified.
    """
    from .audio_naming import speech_take_filename
    from .common import digest, read_json, write_json
    from .performance_runtime import verification_options
    from .quality import validate_audio, validation_key
    from .stored_audio import audio_hash, audio_present, materialize
    directory = Path(directory)
    request_hash = digest(request)
    tonal = []
    for path in sorted(directory.glob("attempt-*.json")):
        try:
            record = read_json(path)
        except (ValueError, OSError):
            continue
        report = record.get("report") if isinstance(record, dict) else None
        name = record.get("audio_file") if isinstance(record, dict) else None
        if (record.get("request_hash") != request_hash or not isinstance(report, dict)
                or report.get("reasons") != [REASON] or not isinstance(name, str)
                or not audio_present(directory / name)):
            continue
        tonal.append((float((report.get(REASON) or {}).get("level_db") or 99), record))
    if not tonal:
        return None
    options = verification_options(request)
    key = validation_key(text, verifier, config["quality"], purpose="speech", **options)
    ffmpeg = config["render"]["ffmpeg"]
    for _, record in sorted(tonal, key=lambda item: item[0])[:MAX_FILTERED_TRIES]:
        attempt = record["attempt"]
        output = directory / speech_take_filename(request, attempt, role="tone-filtered")
        frequencies = []
        with materialize(directory / record["audio_file"]) as source:
            current = source
            while len(frequencies) < MAX_NOTCHES:
                measured = measure(current, exclude=frequencies)
                if (not measured or (measured.get("level_db") or 0) < FILTER_MIN_LEVEL_DB
                        or measured["presence"] < FILTER_MIN_PRESENCE):
                    break
                frequencies.append(measured["frequency_hz"])
                _notched(source, output, frequencies, ffmpeg)
                current = output
        if not frequencies or check(output, exclude=frequencies) is not None:
            continue
        with filtered(frequencies):
            report = validate_audio(output, text, verifier, config["quality"], purpose="speech", **options)
        notice = {"version": FILTER_VERSION, "source_attempt": attempt,
                  "source_audio_sha256": record.get("audio_sha256"), "notches_hz": frequencies,
                  "width_hz": NOTCH_WIDTH_HZ, "source_tone": (record["report"] or {}).get(REASON)}
        if report.get("passed") is not True:
            notify(f"  Take {attempt} with its tone filtered out still failed: {', '.join(report.get('reasons') or [])}.")
            continue
        accepted = {"request_hash": request_hash, "request": record["request"], "attempt": attempt,
                    "seed": record.get("seed"), "validation_key": key,
                    "report": {**report, MARKER: notice},
                    "engine_metadata": {**(record.get("engine_metadata") or {}), MARKER: notice},
                    "audio_file": output.name, "raw_file": None, "audio_sha256": audio_hash(output)}
        write_json(directory / "accepted.json", accepted)
        notify(f"  Every take carried a steady high-pitched tone; using take {attempt} with "
               f"{', '.join(f'{hz / 1000:.1f} kHz' for hz in frequencies)} filtered out (its words verified).")
        return output, accepted
    return None
