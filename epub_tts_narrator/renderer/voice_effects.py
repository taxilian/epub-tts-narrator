"""Equipment effects heard over a character's unchanged voice.

A segment's ``voice_effect.preset`` (helmet, radio, vocoder, pa, phone,
underwater) is applied to that line's accepted, verified audio while the
chapter is composed. The take itself stays dry: verification, censorship,
silence trimming and chapter continuity history all use the original audio,
and the effected copy keeps exactly the same number of samples so chapter
timings and masks stay valid.

The presets are a closed set of FFmpeg filter chains. ``RECIPE_VERSION`` and
the chain text are part of each effected clip's chapter identity, so changing a
chain re-composes only the chapters that use it.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts import VOICE_EFFECT_PRESETS
from .common import digest

RECIPE_VERSION = 1

# Mono PCM chains. Echo tails beyond the take are cut back to its length.
CHAINS = {
    # Enclosed and close: muffled highs, a short reflection inside the visor.
    "helmet": "highpass=f=180,lowpass=f=3200,aecho=0.85:0.6:9:0.35,volume=1.15",
    # Band-limited, compressed, slightly crunchy two-way radio.
    "radio": "highpass=f=380,lowpass=f=2900,acompressor=threshold=0.1:ratio=6:attack=5:release=60,"
             "acrusher=bits=9:mix=0.3,volume=1.3",
    # Robotized voice: phase discarded per frame, then lightly band-shaped.
    "vocoder": "afftfilt=real='hypot(re,im)':imag='0':win_size=512:overlap=0.75,highpass=f=120,volume=0.9",
    # Public address: band-limited with a large-space slapback.
    "pa": "highpass=f=280,lowpass=f=4200,aecho=0.8:0.7:70|140:0.35|0.2,volume=1.1",
    # Telephone line: classic 300–3400 Hz band with compression.
    "phone": "highpass=f=300,lowpass=f=3400,acompressor=threshold=0.125:ratio=4:attack=5:release=50,volume=1.3",
    # Submerged: dull, wobbling, with a close echo.
    "underwater": "lowpass=f=650,aecho=0.8:0.5:28:0.45,vibrato=f=3.5:d=0.12,volume=1.2",
}

if set(CHAINS) != set(VOICE_EFFECT_PRESETS):
    raise ImportError("voice effect chains must match the contract's presets")


def recipe(preset: str) -> dict:
    """The identity of one preset's processing (version and exact chain)."""
    if preset not in CHAINS:
        raise ValueError(f"Unknown voice effect preset: {preset!r}")
    return {"version": RECIPE_VERSION, "preset": preset, "chain_sha256": digest(CHAINS[preset])}


def apply_effect(source: str | Path, target: str | Path, preset: str, *, ffmpeg: str,
                 sample_rate: int, run=None) -> Path:
    """Write ``preset`` applied to ``source`` at ``target`` with the same frame count."""
    from .assembly import _run, _wave_info
    source, target = Path(source), Path(target)
    if preset not in CHAINS:
        raise ValueError(f"Unknown voice effect preset: {preset!r}")
    frames = _wave_info(source, sample_rate)["frames"]
    if target.exists() or target.is_symlink():
        raise ValueError(f"Output already exists: {target}")
    chain = f"{CHAINS[preset]},apad,atrim=end_sample={frames},aformat=sample_fmts=s16:channel_layouts=mono"
    (run or _run)([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0",
                   "-af", chain, "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target)])
    result = _wave_info(target, sample_rate)["frames"]
    if result != frames:
        raise ValueError(f"The {preset} voice effect changed the clip length ({frames} -> {result} frames)")
    return target
