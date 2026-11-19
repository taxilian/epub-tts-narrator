"""Mix independently verified speaker lanes into one passage (lane recipes).

Several voices saying the same words (``simultaneous_speakers`` today, and
collective voices in ``COLLECTIVE_VOICES.md``) are synthesized and verified one
lane at a time. This module only places those verified takes on one timeline:

* every lane's first speech is aligned (a take's own lead-in no longer makes
  it sound late), then a recipe adds deterministic offsets: a little spread for
  unison, whole entrances for staggered speech;
* each lane gets a fixed gain from its recipe;
* the sum is scaled so the passage is as loud as its lanes are on their own
  (gated RMS), never above a fixed peak ceiling. Averaging at ``1/N`` made a
  two-voice line about 3 dB, and an eight-voice line about 9 dB, quieter than
  the solo lines around it.

Recipes are a closed, versioned set like the voice-effect presets. The plan
(offsets, gains, filters, scale) is recorded with the mix, so censorship can
rebuild exactly the same arithmetic over masked lanes. Lanes are never time
stretched, and the mix is never put through recognition as proof.
"""

from __future__ import annotations

import math
import statistics
import sys
from array import array

from .common import digest

RECIPE_VERSION = 1

# Offsets in milliseconds after onset alignment. ``jitter``: each lane after the
# first starts a deterministic 0..jitter ms later. ``step``: lane k (k >= 1)
# enters k steps after the first (a (low, high) pair draws each step from the
# range; with ``each`` every later lane is one step behind the first instead). ``gain_db``: per lane level, the first lane at 0 dB; "step" lowers
# each later lane by the value, "others" sets every later lane. ``darken``: a
# one-pole low-pass per later step (Hz for step 1, halving per further step).
RECIPES = {
    # Unison: one passage said together.
    "tight": {"jitter": 15},
    "loose": {"jitter": 35},
    "crowd": {"jitter": 120},
    # Staggered: the same words entering one after another.
    "swell": {"step": 250, "jitter": 20},
    "babble": {"step": (150, 400), "jitter": 0},
    "echo": {"step": 300, "jitter": 0, "gain_db": ("step", -4.0), "darken": 3200},
    # Layered: one clear lead with the others under it.
    "lead": {"jitter": 0, "step": (20, 60), "each": True, "gain_db": ("others", -12.0)},
}

DEFAULT_RECIPE = "tight"
MAX_LANES = 8

_BLOCK_SECONDS = 0.01
_SPEECH_GATE = 328          # about -40 dBFS: a block this loud is speech, not noise
_LOUDNESS_GATE = 184        # about -45 dBFS: blocks counted towards a lane's loudness
_PEAK_CEILING = 31785       # about -0.3 dBFS


def recipe_names():
    return tuple(RECIPES)


def _blocks(samples, size):
    for start in range(0, len(samples), size):
        block = samples[start:start + size]
        if block:
            yield start, math.sqrt(sum(value * value for value in block) / len(block))


def onset_frame(samples, sample_rate):
    """The first frame of the first 10 ms block that sounds like speech."""
    size = max(1, int(sample_rate * _BLOCK_SECONDS))
    levels = list(_blocks(samples, size))
    loudest = max((level for _, level in levels), default=0.0)
    gate = max(_SPEECH_GATE, 0.1 * loudest)
    for start, level in levels:
        if level >= gate:
            return start
    return 0


def gated_rms(samples, sample_rate):
    """Loudness of the speech in a lane: RMS over its non-quiet 10 ms blocks."""
    size = max(1, int(sample_rate * _BLOCK_SECONDS))
    total = count = 0
    for start in range(0, len(samples), size):
        block = samples[start:start + size]
        energy = sum(value * value for value in block)
        if block and math.sqrt(energy / len(block)) >= _LOUDNESS_GATE:
            total += energy
            count += len(block)
    if not count:
        count = len(samples)
        total = sum(value * value for value in samples)
    return math.sqrt(total / count) if count else 0.0


def _fraction(*parts):
    """A deterministic number in [0, 1) from the lanes' verified identity."""
    return int(digest(list(parts))[:12], 16) / float(16 ** 12)


def plan_lanes(lanes, sources, recipe, sample_rate):
    """Offsets, gains and filters for verified lanes (in their given order).

    ``lanes`` carry their ``audio_sha256``; ``sources`` are their PCM16 sample
    arrays. The first lane is the lead: it is never delayed beyond alignment.
    """
    if recipe not in RECIPES:
        raise ValueError(f"Unknown lane mix recipe {recipe!r}; use one of {', '.join(RECIPES)}")
    if not 2 <= len(lanes) <= MAX_LANES or len(lanes) != len(sources):
        raise ValueError(f"A lane mix needs 2 to {MAX_LANES} verified lanes")
    spec = RECIPES[recipe]
    ms = sample_rate / 1000.0
    onsets = [onset_frame(source, sample_rate) for source in sources]
    latest = max(onsets)
    identity = [lane["audio_sha256"] for lane in lanes]
    offsets, gains, filters = [], [], []
    entry = 0.0
    for index, onset in enumerate(onsets):
        offset = latest - onset
        if index:
            step = spec.get("step", 0)
            if isinstance(step, tuple):
                low, high = step
                step = low + (high - low) * _fraction(recipe, "step", index, identity)
            entry = step if spec.get("each") else entry + step
            jitter = spec.get("jitter", 0) * _fraction(recipe, "jitter", index, identity)
            offset += int(round((entry + jitter) * ms))
        offsets.append(offset)
        kind, value = spec.get("gain_db", ("step", 0.0))
        decibels = 0.0 if not index else (value * index if kind == "step" else value)
        gains.append(10 ** (decibels / 20))
        darken = spec.get("darken")
        filters.append({"lowpass_hz": darken / (2 ** (index - 1))} if darken and index else None)
    return {"recipe": recipe, "recipe_version": RECIPE_VERSION, "offsets": offsets,
            "gains": gains, "filters": filters}


def _lowpass(samples, hz, sample_rate):
    """A one-pole low-pass, deterministic and length preserving."""
    alpha = 1 - math.exp(-2 * math.pi * hz / sample_rate)
    out, state = [], 0.0
    for value in samples:
        state += alpha * (value - state)
        out.append(state)
    return out


def mix_arrays(sources, plan, sample_rate, *, scale=None):
    """Mix PCM16 sample arrays by ``plan``; returns (samples, scale, peak).

    ``scale`` None measures it (loudness match under the peak ceiling); a
    recorded scale is applied as given, which censorship uses so masked lanes
    mix with exactly the arithmetic of the accepted mix.
    """
    frames = max(offset + len(source) for offset, source in zip(plan["offsets"], sources))
    total = array("d", bytes(8 * frames))
    for source, offset, gain, flt in zip(sources, plan["offsets"], plan["gains"], plan["filters"]):
        values = _lowpass(source, flt["lowpass_hz"], sample_rate) if flt else source
        for index, value in enumerate(values):
            total[offset + index] += value * gain
    if scale is None:
        loudest = max(plan["gains"])
        placed = [gated_rms(source, sample_rate) for source, gain in zip(sources, plan["gains"])
                  if gain == loudest]
        target = statistics.median(placed)
        level = gated_rms(total, sample_rate)
        scale = target / level if level else 1.0
        peak = max((abs(value) for value in total), default=0.0)
        if peak * scale > _PEAK_CEILING:
            scale = _PEAK_CEILING / peak
        # Recorded as a short decimal so a JSON round trip reproduces it exactly.
        scale = float(f"{scale:.9g}")
    mixed = array("h", (max(-32768, min(32767, round(value * scale))) for value in total))
    return mixed, scale, max((abs(value) for value in mixed), default=0)


def to_bytes(samples):
    if sys.byteorder != "little":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def from_bytes(data):
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples
