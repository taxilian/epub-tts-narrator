"""Explicit TOML configuration; no environment-variable settings."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import tomllib


DEFAULTS = {
    "engine": {
        "type": "drama3_mlx", "checkpoint": None, "experts": "q8",
        "temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_new_tokens": 2048, "options": {},
    },
    "verification": {
        "type": "mlx_whisper", "model": None, "language": "en",
        "reference_model": None,
    },
    "voice_design": {
        "type": "auto", "checkpoint": None, "language": "English",
        "temperature": 0.7, "max_tokens": 2048, "options": {},
    },
    "gemini": {
        "api_key": None, "model": "gemini-3.8-flash-tts",
        "language_code": "en-US", "timeout_seconds": 180.0,
    },
    "voices": {
        "directory": None, "series_id": None,
        "reference_text": (
            "The morning light fell across the quiet room. "
            "I opened the door and listened to the wind in the trees. "
            "There was still time to decide what to do next."
        ),
    },
    "casting": {
        "player": ["/usr/bin/afplay"], "normalize_samples": True,
        "sample_loudness_lufs": -20.0, "sample_true_peak_db": -1.5,
    },
    "render": {
        "work_dir": None, "output_dir": None,
        "chapter_format": "m4a", "max_chunk_chars": 320, "max_attempts": 3,
        "seed": 1729, "sample_rate": 44100, "paragraph_pause_ms": 250,
        "blank_pause_ms": 500, "bracket_pause_ms": 120, "allow_review": False, "chapter_numbers": [],
        "normalize_loudness": True, "target_lufs": -20.0, "true_peak_db": -1.5,
        "aac_bitrate": "128k", "mp3_bitrate": "128k",
        "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "epub": None,
    },
    "quality": {},
}


def load_config(path=None):
    from .quality import validate_policy
    from epub_audiobook.config import DEFAULTS as ANALYZER_DEFAULTS, DEFAULT_FILENAME
    path = Path(path or Path.cwd() / DEFAULT_FILENAME).resolve()
    try:
        with path.open("rb") as stream:
            supplied = tomllib.load(stream)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid TOML configuration.") from None
    result = deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if section == "voices" and isinstance(values, dict) and "aliases" in values:
            raise ValueError("voices.aliases is book-specific and cannot be in global configuration; "
                             "use audiobook-render voice-aliases BOOK --set SOURCE TARGET")
        if section in ANALYZER_DEFAULTS:
            if not isinstance(values, dict) or set(values) - set(ANALYZER_DEFAULTS[section]):
                raise ValueError("Invalid analyzer configuration section or keys")
            continue
        if section not in result or not isinstance(values, dict):
            raise ValueError("Unknown configuration section or invalid table")
        if section != "quality" and set(values) - set(result[section]):
            raise ValueError(f"Unknown configuration key in {section}")
        result[section].update(values)
    # Preserve explicit legacy shared-library configurations. New configurations
    # leave book locations unset until the input bundle is known.
    if result["voices"]["series_id"] is not None:
        for section, key, default in (("voices", "directory", "./voices"),
                                      ("render", "work_dir", "./render-work"),
                                      ("render", "output_dir", "./audiobook")):
            if result[section][key] is None:
                result[section][key] = default
    result["quality"] = validate_policy(result["quality"])
    for section, keys in {
        "engine": ("type", "experts"),
        "verification": ("type", "model", "language"),
        "voice_design": ("type", "language"),
        "gemini": ("model", "language_code"),
        "voices": ("reference_text",),
        "render": ("chapter_format", "ffmpeg", "ffprobe"),
    }.items():
        for key in keys:
            value = result[section][key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty string")
    for section, key in (("voices", "directory"), ("voices", "series_id"),
                         ("render", "work_dir"), ("render", "output_dir")):
        value = result[section][key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{section}.{key} must be a nonempty string when configured")
    render = result["render"]
    for key, low, high in (
        ("max_chunk_chars", 40, 1000), ("max_attempts", 1, 10),
        ("seed", 0, 2**32 - 1), ("sample_rate", 44100, 44100),
        ("paragraph_pause_ms", 0, 5000), ("blank_pause_ms", 0, 5000),
        ("bracket_pause_ms", 0, 5000),
    ):
        value = render[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid render.{key}")
    if type(render["allow_review"]) is not bool:
        raise ValueError("render.allow_review must be boolean")
    if type(render["normalize_loudness"]) is not bool:
        raise ValueError("render.normalize_loudness must be boolean")
    for key, low, high in (("target_lufs", -40, -10), ("true_peak_db", -9, 0)):
        value = render[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high):
            raise ValueError(f"render.{key} must be a finite number from {low} to {high}")
    for key in ("aac_bitrate", "mp3_bitrate"):
        if not isinstance(render[key], str) or not re.fullmatch(r"[1-9][0-9]{0,2}k", render[key]):
            raise ValueError(f"render.{key} must be a positive bitrate such as 128k")
    numbers = render["chapter_numbers"]
    if not isinstance(numbers, list) or any(type(n) is not int or n < 1 for n in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError("render.chapter_numbers must contain distinct positive integers")
    if render["chapter_format"] not in {"m4a", "mp3"}:
        raise ValueError("render.chapter_format must be m4a or mp3")
    if result["verification"]["type"] != "mlx_whisper":
        raise ValueError("Unsupported verifier type")
    reference_model = result["verification"]["reference_model"]
    if reference_model is not None and (
            not isinstance(reference_model, str) or not reference_model.strip()):
        raise ValueError("verification.reference_model must be a nonempty local model path")
    if not isinstance(result["engine"]["options"], dict):
        raise ValueError("engine.options must be a table")
    design = result["voice_design"]
    gemini = result["gemini"]
    if gemini["api_key"] is not None and (
            not isinstance(gemini["api_key"], str)
            or any(ord(char) < 32 or ord(char) == 127 for char in gemini["api_key"])):
        raise ValueError("gemini.api_key must be a string without control characters")
    timeout = gemini["timeout_seconds"]
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 1 <= timeout <= 600):
        raise ValueError("gemini.timeout_seconds must be a finite number from 1 to 600")
    if not isinstance(design["options"], dict):
        raise ValueError("voice_design.options must be a table")
    if (isinstance(design["temperature"], bool)
            or not isinstance(design["temperature"], (int, float))
            or not 0 < design["temperature"] <= 2):
        raise ValueError("voice_design.temperature must be positive and at most 2")
    if type(design["max_tokens"]) is not int or not 128 <= design["max_tokens"] <= 4096:
        raise ValueError("voice_design.max_tokens must be an integer from 128 through 4096")
    if design["type"] == "qwen3_mlx" and not design["checkpoint"]:
        raise ValueError("voice_design.checkpoint is required for Qwen3 voice design")
    player = result["casting"]["player"]
    if (not isinstance(player, list) or not player or
            any(not isinstance(item, str) or not item.strip() or "\0" in item for item in player)):
        raise ValueError("casting.player must be a nonempty array of executable and argument strings")
    if type(result["casting"]["normalize_samples"]) is not bool:
        raise ValueError("casting.normalize_samples must be boolean")
    for key, low, high in (("sample_loudness_lufs", -40, -10), ("sample_true_peak_db", -9, 0)):
        value = result["casting"][key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f"casting.{key} must be a finite number from {low} to {high}")
    if result["engine"]["type"] == "drama3_mlx" and not result["engine"]["checkpoint"]:
        raise ValueError("engine.checkpoint is required for Drama 3")
    for section, key in (
        ("engine", "checkpoint"), ("verification", "model"), ("verification", "reference_model"),
        ("voice_design", "checkpoint"), ("voices", "directory"),
        ("render", "work_dir"), ("render", "output_dir"), ("render", "epub"),
    ):
        value = result[section][key]
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid {section}.{key}")
            result[section][key] = str((path.parent / Path(value).expanduser()).resolve())
    model_paths = [Path(result[s][k]) for s, k in
                   (("engine", "checkpoint"), ("verification", "model"),
                    ("verification", "reference_model"), ("voice_design", "checkpoint"))
                   if result[s][k] is not None]
    stores = [Path(result[s][k]) for s, k in
              (("voices", "directory"), ("render", "work_dir"), ("render", "output_dir"))
              if result[s][k] is not None]
    protected = model_paths + stores
    if len(set(protected)) != len(protected):
        raise ValueError("Model, voice, work, and output directories must be distinct")
    # Work directories must not contain other writable stores (or vice versa).
    if any(a in b.parents or b in a.parents for i, a in enumerate(stores) for b in stores[i + 1:]):
        raise ValueError("Voice, work, and output directories must not be nested")
    if any(a in b.parents or b in a.parents for a in model_paths for b in stores):
        raise ValueError("Model directories must not overlap voice, work, or output stores")
    return result


def resolve_voice_design(config):
    """Prefer configured Gemini for creation; never change the speech engine.

    Auto selection is local configuration inspection, not an API probe. A
    provider error stays an error instead of silently creating a different voice.
    """
    settings = deepcopy(config.get("voice_design", {"type": "auto"}))
    kind = settings.get("type", "auto")
    gemini = {**DEFAULTS["gemini"], **config.get("gemini", {})}
    if kind == "auto":
        key = gemini["api_key"]
        if not isinstance(key, str) or not key.strip():
            return {"type": "engine"}
        return {"type": "gemini", **gemini}
    if kind == "gemini":
        # Explicit per-action settings already include credentials. A persisted
        # type=gemini setting instead reads them from the global [gemini] table.
        return {"type": "gemini", **gemini,
                **{key: settings[key] for key in gemini if key in settings}}
    return settings


def select_voice_designer(config, choice):
    """Apply a cast-only override without changing the saved global settings."""
    result = deepcopy(config)
    if choice is None:
        return result
    if choice == "local":
        result["voice_design"] = {"type": "engine"}
    elif choice == "gemini":
        settings = {**DEFAULTS["gemini"], **result.get("gemini", {})}
        key = settings["api_key"]
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Gemini casting requires gemini.api_key in your global TOML configuration")
        result["voice_design"] = {"type": "gemini", **settings}
    else:
        raise ValueError("Voice designer must be local or gemini")
    return result
