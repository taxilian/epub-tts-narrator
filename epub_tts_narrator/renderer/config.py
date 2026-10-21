"""Explicit TOML configuration; no environment-variable settings."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit

from epub_tts_narrator.jev import DEFAULTS as JEV_DEFAULTS, validate_settings as validate_jev_settings


DEFAULTS = {
    "engine": {
        "type": "drama3_mlx", "checkpoint": None, "experts": "q8",
        "temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_new_tokens": 2048, "options": {},
    },
    "verification": {
        "type": "mlx_whisper", "model": None, "language": "en",
        "reference_model": None,
        "device": "cpu", "compute_type": "int8", "cpu_threads": 4,
    },
    "voice_design": {
        "type": "auto", "checkpoint": None, "language": "English",
        "temperature": 0.7, "max_tokens": 2048, "options": {},
    },
    "gemini": {
        "api_key": None, "model": "gemini-3.8-flash-tts",
        "language_code": "en-US", "timeout_seconds": 180.0,
    },
    "elevenlabs": {
        "api_key": None, "model": "eleven_ttv_v3", "timeout_seconds": 180.0,
        "guidance_scale": 5.0, "loudness": 0.0,
    },
    "jev": deepcopy(JEV_DEFAULTS),
    "huggingface": {"api_key": ""},
    "render_pool": {
        "workers": [],
        "model_revision": "b2b5ee998034b2975720906c815e7fb0e9f007bd",
    },
    "server": {
        "host": "127.0.0.1", "port": 8080, "api_key": "",
        "max_request_bytes": 100 * 1024 * 1024,
        "max_pending_requests": 4, "request_timeout_seconds": 300.0,
    },
    "remote_chapters": {
        "url": "", "api_key": "", "timeout_seconds": 60.0,
        "poll_seconds": 2.0, "websocket": True,
    },
    "chapter_server": {
        "data_dir": "./chapter-server-data", "host": "127.0.0.1", "port": 8081,
        "api_key": "", "idle_timeout_seconds": 600.0, "assembly_workers": 2,
        "max_pending_jobs": 256, "max_request_bytes": 100 * 1024 * 1024,
        "request_timeout_seconds": 60.0, "idle_action": "none",
        "persistent_data": False, "runpod_pod_id": "", "runpod_api_key": "",
    },
    "chapter_storage": {
        "bucket": "", "endpoint_url": "", "region": "", "access_key_id": "",
        "secret_access_key": "", "prefix": "epub-tts-narrator",
        "download_expiry_seconds": 3600,
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
        "paragraph_synthesis": True, "max_paragraph_chars": 1000,
        "seed": 1729, "sample_rate": 44100, "paragraph_pause_ms": 250,
        "blank_pause_ms": 500, "bracket_pause_ms": 120, "allow_review": False, "chapter_numbers": [],
        "normalize_pauses": True, "sentence_pause_ms": 220, "clause_pause_ms": 100,
        "keep_intermediate_audio": False,
        "normalize_loudness": True, "target_lufs": -20.0, "true_peak_db": -1.5,
        "aac_bitrate": "128k", "mp3_bitrate": "128k",
        "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "epub": None,
    },
    "quality": {},
}


def _credential(value, label):
    if not isinstance(value, str) or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} must be a string without whitespace or control characters")


def _http_url(value, label, *, optional=False):
    if optional and value == "":
        return
    try:
        parsed = urlsplit(value) if isinstance(value, str) else None
        valid = (parsed is not None and parsed.scheme in {"http", "https"}
                 and bool(parsed.hostname) and not parsed.username and not parsed.password
                 and not parsed.query and not parsed.fragment
                 and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value))
        if parsed is not None:
            parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be an HTTP(S) URL without credentials, query or fragment")


def validate_chapter_settings(config):
    """Validate chapter jobs without reading environment settings or model files."""
    remote, server, storage = (config[key] for key in
                               ("remote_chapters", "chapter_server", "chapter_storage"))
    _http_url(remote["url"], "remote_chapters.url", optional=True)
    _credential(remote["api_key"], "remote_chapters.api_key")
    if type(remote["websocket"]) is not bool:
        raise ValueError("remote_chapters.websocket must be boolean")
    for table, key, low, high in (
        ("remote_chapters", "timeout_seconds", 1, 3600),
        ("remote_chapters", "poll_seconds", 0.1, 60),
        ("chapter_server", "request_timeout_seconds", 1, 3600),
        ("chapter_server", "idle_timeout_seconds", 1, 86400),
    ):
        value = config[table][key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{table}.{key} must be from {low} through {high}")
    for key, low, high in (("port", 1, 65535), ("assembly_workers", 1, 16),
                          ("max_pending_jobs", 1, 10000),
                          ("max_request_bytes", 1024, 512 * 1024 * 1024)):
        if type(server[key]) is not int or not low <= server[key] <= high:
            raise ValueError(f"chapter_server.{key} must be an integer from {low} through {high}")
    for key in ("data_dir", "host"):
        if (not isinstance(server[key], str) or not server[key].strip()
                or any(ord(char) < 32 or ord(char) == 127 for char in server[key])):
            raise ValueError(f"chapter_server.{key} must be a nonempty string without control characters")
    if any(char.isspace() for char in server["host"]):
        raise ValueError("chapter_server.host must be a hostname or IP address")
    for key in ("api_key", "runpod_api_key", "runpod_pod_id"):
        _credential(server[key], f"chapter_server.{key}")
    if server["idle_action"] not in ("none", "runpod_stop"):
        raise ValueError("chapter_server.idle_action must be none or runpod_stop")
    if type(server["persistent_data"]) is not bool:
        raise ValueError("chapter_server.persistent_data must be boolean")
    _http_url(storage["endpoint_url"], "chapter_storage.endpoint_url", optional=True)
    for key in ("bucket", "region", "access_key_id", "secret_access_key", "prefix"):
        _credential(storage[key], f"chapter_storage.{key}")
    if (type(storage["download_expiry_seconds"]) is not int
            or not 60 <= storage["download_expiry_seconds"] <= 604800):
        raise ValueError("chapter_storage.download_expiry_seconds must be from 60 through 604800")
    if server["idle_action"] == "runpod_stop" and not (
            server["persistent_data"] and storage["bucket"]
            and server["runpod_pod_id"] and server["runpod_api_key"]):
        raise ValueError("Runpod idle shutdown requires persistent_data, a chapter_storage.bucket, "
                         "runpod_pod_id and runpod_api_key")


def validate_distributed_settings(config):
    """Validate file-only worker/server settings without importing model packages."""
    pool = config["render_pool"]
    if not isinstance(pool["model_revision"], str) or not pool["model_revision"].strip():
        raise ValueError("render_pool.model_revision must be a nonempty string")
    workers = pool["workers"]
    if not isinstance(workers, list) or len(workers) > 16:
        raise ValueError("render_pool.workers must be an array of at most 16 worker tables")
    names, normalized = set(), []
    for index, supplied in enumerate(workers, 1):
        label = f"render_pool.workers[{index}]"
        if not isinstance(supplied, dict):
            raise ValueError(f"{label} must be a table")
        name, kind = supplied.get("name"), supplied.get("type")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError(f"{label}.name must be a short name containing letters, numbers, dots, dashes or underscores")
        if name in names:
            raise ValueError("Renderer worker names must be unique")
        names.add(name)
        if not isinstance(kind, str) or kind not in {"local", "fish_http"}:
            raise ValueError(f"{label}.type must be local or fish_http")
        allowed = {"name", "type", "concurrency"}
        if kind == "fish_http":
            allowed |= {"url", "api_key", "model", "timeout_seconds", "multi_speaker"}
        if set(supplied) - allowed:
            raise ValueError(f"Unknown configuration key in {label}")
        worker = {"concurrency": 1, **supplied}
        concurrency = worker["concurrency"]
        if type(concurrency) is not int or not 1 <= concurrency <= 16:
            raise ValueError(f"{label}.concurrency must be an integer from 1 to 16")
        if kind == "local":
            if concurrency != 1 or any(item["type"] == "local" for item in normalized):
                raise ValueError("Use at most one local renderer, with concurrency = 1")
        else:
            worker = {"api_key": "", "model": "drama-3", "timeout_seconds": 300.0,
                      "multi_speaker": True, **worker}
            _credential(worker["api_key"], f"{label}.api_key")
            url = worker.get("url")
            try:
                parsed = urlsplit(url) if isinstance(url, str) else None
                valid = (parsed is not None and parsed.scheme in {"http", "https"}
                         and bool(parsed.hostname) and not parsed.username and not parsed.password
                         and not parsed.query and not parsed.fragment and bool(parsed.path.strip("/"))
                         and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url))
                if parsed is not None:
                    parsed.port  # Validate malformed/out-of-range ports without exposing the URL.
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"{label}.url must be a full HTTP(S) TTS URL without credentials, query or fragment")
            if (not isinstance(worker["model"], str)
                    or worker["model"] not in {"drama-3", "fishaudio/drama-3"}):
                raise ValueError(f"{label}.model must be drama-3 or fishaudio/drama-3")
            timeout = worker["timeout_seconds"]
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 3600:
                raise ValueError(f"{label}.timeout_seconds must be from 1 through 3600")
            if type(worker["multi_speaker"]) is not bool:
                raise ValueError(f"{label}.multi_speaker must be boolean")
        normalized.append(worker)
    if sum(worker["concurrency"] for worker in normalized) > 32:
        raise ValueError("Renderer worker concurrency must total at most 32")
    if normalized and config["engine"]["type"] != "drama3_mlx":
        raise ValueError("The render pool currently requires engine.type = drama3_mlx")
    if normalized:
        # Remote-only workers do not instantiate the local adapter, so validate
        # its shared sampling contract here before any generation/API calls.
        engine = config["engine"]
        for key, maximum in (("temperature", 2), ("top_p", 1)):
            value = engine[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"engine.{key} must be positive and at most {maximum} for the render pool")
        for key, maximum in (("top_k", 1000), ("max_new_tokens", 2048)):
            if type(engine[key]) is not int or not 1 <= engine[key] <= maximum:
                raise ValueError(f"engine.{key} must be an integer from 1 through {maximum} for the render pool")
        if not isinstance(engine["experts"], str) or engine["experts"] not in {"q8", "q4", "fp8", "bf16"}:
            raise ValueError("engine.experts must be q8, q4, fp8, or bf16")
        if engine["options"]:
            raise ValueError("Drama 3 settings belong directly in the engine table")
    pool["workers"] = normalized
    server = config["server"]
    if (not isinstance(server["host"], str) or not server["host"].strip()
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in server["host"])):
        raise ValueError("server.host must be a hostname or IP address")
    _credential(server["api_key"], "server.api_key")
    for key, low, high in (("port", 1, 65535), ("max_request_bytes", 1024, 512 * 1024 * 1024),
                           ("max_pending_requests", 1, 32)):
        if type(server[key]) is not int or not low <= server[key] <= high:
            raise ValueError(f"server.{key} must be an integer from {low} through {high}")
    timeout = server["request_timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 3600:
        raise ValueError("server.request_timeout_seconds must be from 1 through 3600")


def load_config(path=None, *, require_verifier=True):
    from .quality import validate_policy
    from epub_tts_narrator.config import (
        DEFAULTS as ANALYZER_DEFAULTS, resolve_config_path, validate_huggingface_settings,
    )
    path = resolve_config_path(path).resolve()
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
        if section in ANALYZER_DEFAULTS and section not in DEFAULTS:
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
    result["jev"] = validate_jev_settings(result["jev"])
    result["huggingface"] = validate_huggingface_settings(result["huggingface"])
    validate_distributed_settings(result)
    validate_chapter_settings(result)
    remote_chapters = bool(result["remote_chapters"]["url"])
    for section, keys in {
        "engine": ("type", "experts"),
        "verification": ("type", "model", "language"),
        "voice_design": ("type", "language"),
        "gemini": ("model", "language_code"),
        "elevenlabs": ("model",),
        "voices": ("reference_text",),
        "render": ("chapter_format", "ffmpeg", "ffprobe"),
    }.items():
        for key in keys:
            value = result[section][key]
            if (section == "verification" and key == "model" and value is None
                    and (not require_verifier or remote_chapters)):
                continue
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
        ("max_paragraph_chars", 320, 1500),
        ("seed", 0, 2**32 - 1), ("sample_rate", 44100, 44100),
        ("paragraph_pause_ms", 0, 5000), ("blank_pause_ms", 0, 5000),
        ("bracket_pause_ms", 0, 5000),
        ("sentence_pause_ms", 0, 5000), ("clause_pause_ms", 0, 5000),
    ):
        value = render[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid render.{key}")
    if type(render["allow_review"]) is not bool:
        raise ValueError("render.allow_review must be boolean")
    if type(render["paragraph_synthesis"]) is not bool:
        raise ValueError("render.paragraph_synthesis must be boolean")
    if type(render["normalize_loudness"]) is not bool:
        raise ValueError("render.normalize_loudness must be boolean")
    if type(render["normalize_pauses"]) is not bool:
        raise ValueError("render.normalize_pauses must be boolean")
    if type(render["keep_intermediate_audio"]) is not bool:
        raise ValueError("render.keep_intermediate_audio must be boolean")
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
    if result["verification"]["type"] not in {"mlx_whisper", "faster_whisper"}:
        raise ValueError("Unsupported verifier type")
    verification = result["verification"]
    if verification["device"] not in {"cpu", "cuda"}:
        raise ValueError("verification.device must be cpu or cuda")
    if verification["compute_type"] not in {"default", "auto", "int8", "int8_float16", "int8_float32", "float16", "float32", "bfloat16", "int8_bfloat16"}:
        raise ValueError("Unsupported verification.compute_type")
    if type(verification["cpu_threads"]) is not int or not 1 <= verification["cpu_threads"] <= 128:
        raise ValueError("verification.cpu_threads must be an integer from 1 through 128")
    reference_model = result["verification"]["reference_model"]
    if reference_model is not None and (
            not isinstance(reference_model, str) or not reference_model.strip()):
        raise ValueError("verification.reference_model must be a nonempty local model path")
    if not isinstance(result["engine"]["options"], dict):
        raise ValueError("engine.options must be a table")
    design = result["voice_design"]
    for provider in ("gemini", "elevenlabs"):
        remote = result[provider]
        if remote["api_key"] is not None and (
                not isinstance(remote["api_key"], str)
                or any(ord(char) < 32 or ord(char) == 127 for char in remote["api_key"])):
            raise ValueError(f"{provider}.api_key must be a string without control characters")
        timeout = remote["timeout_seconds"]
        if (type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 1 <= timeout <= 600):
            raise ValueError(f"{provider}.timeout_seconds must be a finite number from 1 to 600")
    for key, low, high in (("guidance_scale", 0, 100), ("loudness", -1, 1)):
        value = result["elevenlabs"][key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high):
            raise ValueError(f"elevenlabs.{key} must be a finite number from {low} to {high}")
    if result["elevenlabs"]["model"] not in {"eleven_ttv_v3", "eleven_multilingual_ttv_v2"}:
        raise ValueError("elevenlabs.model must be eleven_ttv_v3 or eleven_multilingual_ttv_v2")
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
    workers = result["render_pool"]["workers"]
    needs_local_engine = not remote_chapters and (not workers or any(worker["type"] == "local" for worker in workers))
    if (result["engine"]["type"] == "drama3_mlx" and needs_local_engine
            and not result["engine"]["checkpoint"]):
        raise ValueError("engine.checkpoint is required for Drama 3")
    for section, key in (
        ("engine", "checkpoint"), ("verification", "model"), ("verification", "reference_model"),
        ("voice_design", "checkpoint"), ("voices", "directory"),
        ("render", "work_dir"), ("render", "output_dir"), ("render", "epub"),
        ("chapter_server", "data_dir"),
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
              (("voices", "directory"), ("render", "work_dir"), ("render", "output_dir"),
               ("chapter_server", "data_dir"))
              if result[s][k] is not None]
    protected = model_paths + stores
    if len(set(protected)) != len(protected):
        raise ValueError("Model, voice, work, output, and chapter server data directories must be distinct")
    # Work directories must not contain other writable stores (or vice versa).
    if any(a in b.parents or b in a.parents for i, a in enumerate(stores) for b in stores[i + 1:]):
        raise ValueError("Voice, work, output, and chapter server data directories must not be nested")
    if any(a in b.parents or b in a.parents for a in model_paths for b in stores):
        raise ValueError("Model directories must not overlap voice, work, output, or chapter server data stores")
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
    if kind in {"gemini", "elevenlabs"}:
        # Explicit per-action settings already include credentials. A persisted
        # provider choice instead reads them from its global provider table.
        provider = {**DEFAULTS[kind], **config.get(kind, {})}
        resolved = {"type": kind, **provider,
                    **{key: settings[key] for key in provider if key in settings}}
        if kind == "elevenlabs" and (
                not isinstance(resolved["api_key"], str) or not resolved["api_key"].strip()):
            raise ValueError("ElevenLabs casting requires elevenlabs.api_key in your global TOML configuration")
        return resolved
    return settings


def select_voice_designer(config, choice):
    """Apply a cast-only override without changing the saved global settings."""
    result = deepcopy(config)
    if choice is None:
        return result
    if choice == "local":
        result["voice_design"] = {"type": "engine"}
    elif choice in {"gemini", "elevenlabs"}:
        settings = {**DEFAULTS[choice], **result.get(choice, {})}
        key = settings["api_key"]
        if not isinstance(key, str) or not key.strip():
            name = {"gemini": "Gemini", "elevenlabs": "ElevenLabs"}[choice]
            raise ValueError(f"{name} casting requires {choice}.api_key in your global TOML configuration")
        result["voice_design"] = {"type": choice, **settings}
    else:
        raise ValueError("Voice designer must be local, gemini, or elevenlabs")
    return result
