"""Load explicit TOML settings without consulting environment variables.

Credentials are literal file values. Error messages intentionally omit supplied
values and paths, so a malformed file cannot spill credentials into CLI output.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tomllib
from typing import Any

from .pipeline import DEFAULT_POLICY
from .jev import DEFAULTS as JEV_DEFAULTS, validate_settings as validate_jev_settings


DEFAULT_FILENAME = "epub-tts-narrator.toml"
LEGACY_FILENAME = "epub-audiobook.toml"


def resolve_config_path(path: Path | str | None = None) -> Path:
    """Select one global file, retaining the original filename as a fallback.

    An explicit path always wins. A present canonical file, including a broken
    symlink or invalid TOML, must report its own error instead of silently using
    a different configuration.
    """
    if path is not None:
        if not isinstance(path, (str, Path)):
            raise ValueError("Configuration path must be a string or Path.")
        return Path(path)
    directory = Path.cwd()
    for name in (DEFAULT_FILENAME, LEGACY_FILENAME):
        candidate = directory / name
        try:
            candidate.lstat()
        except FileNotFoundError:
            continue
        return candidate
    return directory / DEFAULT_FILENAME


DEFAULTS = {
    "provider": {
        "model": None,
        "base_url": "https://api.openai.com/v1",
        "api_key": None,
        "timeout": 120.0,
        "json_mode": True,
        "max_output_tokens": None,
        "token_limit_field": "max_completion_tokens",
        "reasoning_effort": None,
        "max_retries": 0,
        "retry_initial_delay": 5.0,
        "retry_max_delay": 60.0,
    },
    "analysis": {
        "chunk_chars": 12000,
        "max_request_chars": 200000,
        "repair_attempts": 1,
        "cache_dir": None,
        "whole_book_discovery": False,
        "identity_consensus": True,
        "attribution_review": False,
        "attribution_review_chunk_chars": 60000,
        "attribution_review_notes": "",
        "attribution_workers": 1,
        "cast_review_notes": "",
        "cast_reconciliation_rounds": 1,
        "cast_reconciliation_notes": [],
        "series_registry": None,
    },
    "extraction": {"include_auxiliary": False},
    "performance_policy": dict(DEFAULT_POLICY),
    "jev": deepcopy(JEV_DEFAULTS),
    "huggingface": {"api_key": ""},
}


def validate_huggingface_settings(settings: dict[str, Any]) -> dict[str, str]:
    """Validate the file-only download credential without echoing its value."""
    if not isinstance(settings, dict) or set(settings) - {"api_key"}:
        raise ValueError("Unknown key in configuration huggingface section.")
    api_key = settings.get("api_key", "")
    if not isinstance(api_key, str) or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in api_key):
        raise ValueError("Configuration huggingface.api_key must be a string without whitespace or control characters.")
    return {"api_key": api_key}


def _nonempty_string(value: Any, section: str, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration {section}.{key} must be a nonempty string.")


def _integer(value: Any, section: str, key: str, *, minimum: int = 1, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        description = f"an integer of at least {minimum}"
        if maximum is not None:
            description = f"an integer from {minimum} to {maximum}"
        raise ValueError(f"Configuration {section}.{key} must be {description}.")


def _validate(section: str, key: str, value: Any) -> None:
    if section == "performance_policy":
        _nonempty_string(value, section, key)
    elif section == "provider":
        if key in {"model", "base_url", "api_key"}:
            _nonempty_string(value, section, key)
        elif key == "json_mode":
            if type(value) is not bool:
                raise ValueError("Configuration provider.json_mode must be a boolean.")
        elif key == "max_output_tokens":
            _integer(value, section, key)
        elif key == "max_retries":
            _integer(value, section, key, minimum=0, maximum=10)
        elif key in {"retry_initial_delay", "retry_max_delay"}:
            if type(value) not in {int, float} or not 1 <= value <= 60:
                raise ValueError(f"Configuration provider.{key} must be a number from 1 to 60 seconds.")
        elif key == "token_limit_field":
            if not isinstance(value, str) or value not in {"max_completion_tokens", "max_tokens"}:
                raise ValueError("Configuration provider.token_limit_field must be max_completion_tokens or max_tokens.")
        elif key == "reasoning_effort":
            if not isinstance(value, str) or value not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError("Configuration provider.reasoning_effort must be a supported reasoning level.")
        elif key == "timeout":
            if type(value) not in {int, float}:
                raise ValueError("Configuration provider.timeout must be a finite positive number.")
            try:
                valid = math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("Configuration provider.timeout must be a finite positive number.")
    elif section == "analysis":
        if key in {"cast_review_notes", "attribution_review_notes"}:
            if not isinstance(value, str):
                raise ValueError(f"Configuration analysis.{key} must be a string.")
        elif key in {"whole_book_discovery", "identity_consensus", "attribution_review"}:
            if type(value) is not bool:
                raise ValueError(f"Configuration analysis.{key} must be a boolean.")
        elif key == "attribution_workers":
            _integer(value, section, key, minimum=1, maximum=8)
        elif key == "cast_reconciliation_rounds":
            _integer(value, section, key, minimum=1, maximum=3)
        elif key == "cast_reconciliation_notes":
            if not isinstance(value, list) or any(not isinstance(note, str) for note in value):
                raise ValueError("Configuration analysis.cast_reconciliation_notes must be a list of strings.")
        elif key in {"cache_dir", "series_registry"}:
            _nonempty_string(value, section, key)
        elif key in {"chunk_chars", "attribution_review_chunk_chars"}:
            _integer(value, section, key, minimum=256)
        elif key == "repair_attempts":
            _integer(value, section, key, minimum=0, maximum=5)
        else:
            _integer(value, section, key)
    elif section == "extraction" and type(value) is not bool:
        raise ValueError("Configuration extraction.include_auxiliary must be a boolean.")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Return defaults merged with a strictly validated TOML configuration.

    With no path, read ``epub-tts-narrator.toml`` in the current directory, falling
    back to ``epub-audiobook.toml`` when absent. Explicit paths must exist.
    Relative cache and series paths are based on
    the configuration file's directory; strings are never interpolated or read
    from environment variables. Returned dictionaries can be edited independently.
    """
    configuration_path = resolve_config_path(path)
    try:
        with configuration_path.open("rb") as stream:
            supplied = tomllib.load(stream)
    except FileNotFoundError:
        if path is None and not configuration_path.is_symlink():
            return deepcopy(DEFAULTS)
        raise ValueError("The requested configuration file does not exist.") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid TOML configuration.") from None
    except OSError:
        raise ValueError("Cannot read configuration file.") from None

    result = deepcopy(DEFAULTS)
    from epub_tts_narrator.renderer.config import DEFAULTS as RENDERER_DEFAULTS
    if any(section not in DEFAULTS and section not in RENDERER_DEFAULTS for section in supplied):
        raise ValueError("Unknown configuration section.")
    for section, settings in supplied.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Configuration {section} must be a table.")
        if section == "jev":
            # This optional evaluator serves both speech verification and the
            # analyzer's interactive speaker shortlist from the same file.
            result[section] = validate_jev_settings(settings)
            continue
        if section == "huggingface":
            result[section] = validate_huggingface_settings(settings)
            continue
        if section in RENDERER_DEFAULTS:
            if section == "voices" and "aliases" in settings:
                raise ValueError("voices.aliases is book-specific and cannot be in global configuration; "
                                 "use audiobook-render voice-aliases BOOK --set SOURCE TARGET")
            if section != "quality" and any(key not in RENDERER_DEFAULTS[section] for key in settings):
                raise ValueError("Unknown key in renderer configuration section.")
            continue
        if any(key not in DEFAULTS[section] for key in settings):
            raise ValueError(f"Unknown key in configuration {section} section.")
        for key, value in settings.items():
            _validate(section, key, value)
            result[section][key] = value
    if len(result["analysis"]["cast_reconciliation_notes"]) > result["analysis"]["cast_reconciliation_rounds"]:
        raise ValueError("Configuration analysis.cast_reconciliation_notes must have at most cast_reconciliation_rounds entries.")
    result["provider"]["timeout"] = float(result["provider"]["timeout"])
    for key in ("retry_initial_delay", "retry_max_delay"):
        result["provider"][key] = float(result["provider"][key])
    if result["provider"]["retry_initial_delay"] > result["provider"]["retry_max_delay"]:
        raise ValueError("Configuration provider.retry_initial_delay must not exceed provider.retry_max_delay.")
    for key in ("cache_dir", "series_registry"):
        value = result["analysis"][key]
        if value is None:
            continue
        try:
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = configuration_path.parent / resolved
            result["analysis"][key] = str(resolved.resolve())
        except (OSError, RuntimeError, ValueError):
            raise ValueError(f"Configuration analysis.{key} cannot be resolved.") from None
    return result
