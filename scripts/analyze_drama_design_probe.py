#!/usr/bin/env python3
"""CPU-only acoustic measurements for the local Drama 3 prompt experiment.

Run with the renderer's Python (numpy, scipy and soundfile are already installed):
    .venv-renderer/bin/python scripts/analyze_drama_design_probe.py --manifest FILE

Pitch is supporting acoustic evidence, not a gender or speaker-identity classifier.
Whispered/creaky/noisy speech can have no reliable pitch, and octave errors remain
possible. Listen to the recordings and check their transcripts separately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import correlate, resample_poly
import soundfile as sf


ANALYSIS_RATE = 16000
FRAME_SECONDS = 0.064
HOP_SECONDS = 0.010
MIN_F0 = 70.0
MAX_F0 = 450.0
YIN_THRESHOLD = 0.15
YIN_FALLBACK_THRESHOLD = 0.25


def _db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def _yin(frame: np.ndarray, rate: int) -> tuple[float, float] | None:
    """Fixed-window YIN difference, CMND trough selection, quadratic refinement."""
    min_lag = math.floor(rate / MAX_F0)
    max_lag = math.ceil(rate / MIN_F0)
    window = len(frame) - max_lag
    if window < max_lag:
        return None
    centered = frame - frame.mean()
    base = centered[:window]
    squares = np.concatenate(([0.0], np.cumsum(centered * centered)))
    shifted_energy = squares[window:] - squares[: max_lag + 1]
    cross = correlate(centered, base, mode="valid", method="fft")
    difference = np.maximum(float(np.dot(base, base)) + shifted_energy - 2 * cross, 0)
    cumulative = np.cumsum(difference[1:])
    cmnd = np.ones(max_lag + 1)
    cmnd[1:] = np.divide(
        difference[1:] * np.arange(1, max_lag + 1), cumulative,
        out=np.ones(max_lag), where=cumulative > 1e-15,
    )
    lag = min_lag
    while lag <= max_lag and cmnd[lag] >= YIN_THRESHOLD:
        lag += 1
    if lag <= max_lag:
        while lag < max_lag and cmnd[lag + 1] < cmnd[lag]:
            lag += 1
    else:
        lag = min_lag + int(np.argmin(cmnd[min_lag:]))
        if cmnd[lag] >= YIN_FALLBACK_THRESHOLD:
            return None
    refined = float(lag)
    if 0 < lag < max_lag:
        left, middle, right = cmnd[lag - 1 : lag + 2]
        denominator = left - 2 * middle + right
        if abs(denominator) > 1e-12:
            refined += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    frequency = rate / refined
    if not MIN_F0 <= frequency <= MAX_F0:
        return None
    return float(frequency), float(1 - cmnd[lag])


def analyze_samples(audio: np.ndarray, rate: int) -> dict:
    """Measure amplitude, approximate silence and reliable periodic frame F0."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or rate <= 0:
        raise ValueError("Audio must contain finite samples at a positive sample rate")
    duration = len(audio) / rate
    rms = float(np.sqrt(np.mean(audio * audio)))
    if rate != ANALYSIS_RATE:
        divisor = math.gcd(rate, ANALYSIS_RATE)
        audio = resample_poly(audio, ANALYSIS_RATE // divisor, rate // divisor)
    hop = round(HOP_SECONDS * ANALYSIS_RATE)
    # Disjoint 10 ms bins make silence durations additive rather than overlapping.
    bins = [audio[start : start + hop] for start in range(0, len(audio), hop)]
    amplitudes = np.array([np.sqrt(np.mean(part * part)) for part in bins])
    durations = np.array([len(part) / ANALYSIS_RATE for part in bins])
    threshold = max(10 ** (-60 / 20), float(np.max(amplitudes)) * 10 ** (-35 / 20))
    quiet = amplitudes < threshold
    active = np.flatnonzero(~quiet)
    leading = float(durations[: active[0]].sum()) if len(active) else duration
    trailing = float(durations[active[-1] + 1 :].sum()) if len(active) else duration

    frame_length = round(FRAME_SECONDS * ANALYSIS_RATE)
    pitches, confidences, pitch_frames = [], [], []
    frame_count = max(0, 1 + (len(audio) - frame_length) // hop)
    for start in range(0, max(0, len(audio) - frame_length + 1), hop):
        frame = audio[start : start + frame_length]
        if float(np.sqrt(np.mean(frame * frame))) < threshold:
            continue
        estimate = _yin(frame, ANALYSIS_RATE)
        if estimate is not None:
            pitch, confidence = estimate
            pitches.append(pitch)
            confidences.append(confidence)
            pitch_frames.append({
                "time_seconds": round((start + frame_length / 2) / ANALYSIS_RATE, 4),
                "f0_hz": round(pitch, 2), "periodicity_confidence": round(confidence, 4),
            })
    return {
        "duration_seconds": duration,
        "rms_dbfs": _db(rms),
        "peak_dbfs": _db(float(np.max(np.abs(audio)))),
        "silence_threshold_dbfs": _db(threshold),
        "estimated_quiet_seconds": float(durations[quiet].sum()),
        "estimated_leading_quiet_seconds": leading,
        "estimated_trailing_quiet_seconds": trailing,
        "analyzed_frame_count": frame_count,
        "periodic_frame_count": len(pitches),
        "periodic_frame_fraction": len(pitches) / frame_count if frame_count else 0,
        "f0_median_hz": float(np.median(pitches)) if pitches else None,
        "f0_p10_hz": float(np.percentile(pitches, 10)) if pitches else None,
        "f0_p90_hz": float(np.percentile(pitches, 90)) if pitches else None,
        "periodicity_confidence_median": float(np.median(confidences)) if pitches else None,
        "pitch_frames": pitch_frames,
    }


def synthetic_check() -> dict:
    """Check pitch scale/octaves and rejection of silence/noise without a model."""
    time = np.arange(ANALYSIS_RATE) / ANALYSIS_RATE
    cases = {}
    for frequency in (75, 110, 150, 220, 330, 440):
        for label, signal in (
            ("sine", np.sin(2 * np.pi * frequency * time)),
            ("harmonics", np.sin(2 * np.pi * frequency * time)
             + 0.7 * np.sin(4 * np.pi * frequency * time)
             + 0.4 * np.sin(6 * np.pi * frequency * time)),
        ):
            result = analyze_samples(signal * 0.15, ANALYSIS_RATE)
            measured = result["f0_median_hz"]
            if measured is None or abs(measured / frequency - 1) > 0.02:
                raise AssertionError(f"{label} {frequency} Hz measured as {measured!r}")
            cases[f"{label}_{frequency}_hz"] = round(measured, 3)
    for label, signal in (
        ("silence", np.zeros_like(time)),
        ("white_noise", np.random.default_rng(1729).normal(0, 0.1, len(time))),
    ):
        result = analyze_samples(signal, ANALYSIS_RATE)
        if result["periodic_frame_count"]:
            raise AssertionError(f"{label} unexpectedly yielded periodic frames")
        cases[label] = "no periodic frames"
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_check(), indent=2))
    if args.manifest is None:
        if args.self_test:
            return
        parser.error("--manifest is required unless only --self-test is requested")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("results"), list):
        parser.error("Manifest must contain a results array")
    results = []
    print(f"{'sample':40} {'seconds':>7} {'F0 Hz':>8} {'periodic':>9} {'quiet s':>8} {'RMS dBFS':>9}")
    for entry in manifest["results"]:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest result must be an object")
        identifier = str(entry.get("id", "unnamed"))
        raw_path = entry.get("audio_path", entry.get("wav_path"))
        row = {"id": identifier, "audio_path": raw_path}
        try:
            if not isinstance(raw_path, str):
                raise ValueError("Result has no audio_path")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = manifest_path.parent / path
            audio, rate = sf.read(path, dtype="float64")
            row["audio_path"] = str(path.resolve())
            row.update(analyze_samples(audio, rate))
            pitch = row["f0_median_hz"]
            rendered_pitch = "n/a" if pitch is None else f"{pitch:.1f}"
            print(f"{identifier[:40]:40} {row['duration_seconds']:7.2f} {rendered_pitch:>8} "
                  f"{row['periodic_frame_fraction']:8.1%} {row['estimated_quiet_seconds']:8.2f} "
                  f"{row['rms_dbfs']:9.1f}")
        except (OSError, RuntimeError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{identifier[:40]:40} ERROR {row['error']}")
        results.append(row)
    output = {
        "manifest": str(manifest_path),
        "method": "fixed-window YIN CMND, quadratic interpolation, amplitude gate",
        "pitch_range_hz": [MIN_F0, MAX_F0],
        "frame_seconds": FRAME_SECONDS, "hop_seconds": HOP_SECONDS,
        "limitations": [
            "Pitch is not a gender or speaker-identity classifier.",
            "Whispers, creaky speech, music and noise may yield missing or incorrect pitch; octave errors remain possible.",
            "Quiet duration is an amplitude estimate, not a speech-activity detector.",
            "Transcription correctness and prompt compliance require separate assessment and listening.",
        ],
        "results": results,
    }
    output_path = manifest_path.with_name("acoustic-analysis.json")
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
