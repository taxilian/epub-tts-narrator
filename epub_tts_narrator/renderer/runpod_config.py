"""File-only deployment settings and private, minimal worker configuration.

The normal renderer configuration remains local. A deployment override changes
remote settings without teaching the Mac renderer about Linux model paths.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import tomllib
from urllib.parse import urlsplit

from epub_tts_narrator.config import resolve_config_path
from .config import DEFAULTS, load_config
from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model


DEPLOYMENT_DEFAULTS = {
    "name": "epub-tts-narrator",
    "image": "",
    "gpu_type_id": "NVIDIA H100 PCIe",
    "data_center_id": "",
    "volume_id": "",
    "volume_size_gb": 150,
    "container_disk_gb": 20,
    "max_hourly_price": 3.50,
    "state_dir": "./dist/runpod",
    "startup_timeout_seconds": 900,
    "service_port": 8082,
    "ssh_user": "root",
    "app_dir": "/workspace/epub-tts-narrator",
    "fish_source": "/workspace/fish-speech",
    "sglang_source": "/workspace/sglang-omni",
    "tts_backend": "native",
    "tts_concurrency": 4,
    "remote_config": "/workspace/epub-tts-narrator.remote.toml",
    "model_dir": "/workspace/models/drama-3",
    "whisper_dir": "/workspace/models/whisper-ct2",
    "data_dir": "/workspace/chapter-server-data",
    "github_url": "https://github.com/taxilian/epub-tts-narrator.git",
    "git_ref": "main",
    "idle_timeout_seconds": 600,
    "pod_id": "",
}

_REMOTE_SECTIONS = frozenset({
    "engine", "render_pool", "verification", "quality", "render", "jev",
    "provider", "huggingface", "chapter_server", "chapter_storage",
})
_PATH_KEYS = (
    ("engine", "checkpoint"), ("verification", "model"),
    ("verification", "reference_model"), ("voice_design", "checkpoint"),
    ("voices", "directory"), ("render", "work_dir"), ("render", "output_dir"),
    ("render", "epub"), ("chapter_server", "data_dir"),
)


def _read_toml(path):
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid TOML deployment configuration") from None


def _merge(base, supplied):
    result = deepcopy(base)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _toml_value(value):
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{json.dumps(key)} = {_toml_value(item)}" for key, item in value.items()
            if item is not None
        ) + " }"
    raise ValueError("Configuration contains an unsupported TOML value")


def toml_text(config):
    """Serialize renderer settings without adding a runtime TOML dependency."""
    lines = []
    for section, values in config.items():
        if not isinstance(section, str) or not isinstance(values, dict):
            raise ValueError("Configuration must contain named tables")
        lines.append(f"[{json.dumps(section)}]")
        for key, value in values.items():
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings")
            if value is not None:
                lines.append(f"{json.dumps(key)} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def write_private_toml(path, config):
    """Replace one file atomically, with private permissions even on first write."""
    path = Path(path)
    content = toml_text(config).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def _absolute_paths(config, directory):
    result = deepcopy(config)
    # The default data directory has the same origin as the source config.
    result.setdefault("chapter_server", {}).setdefault(
        "data_dir", DEFAULTS["chapter_server"]["data_dir"])
    for section, key in _PATH_KEYS:
        value = result.get(section, {}).get(key)
        if isinstance(value, str) and value:
            result[section][key] = str((directory / Path(value).expanduser()).resolve())
    return result


def _validated_renderer(config, directory):
    # The existing loader is the single validation contract. Temporary settings
    # are private and removed before this function returns, including on error.
    with tempfile.TemporaryDirectory(prefix="epub-runpod-config-") as folder:
        path = write_private_toml(Path(folder) / "config.toml", _absolute_paths(config, directory))
        return load_config(path, require_verifier=False)


def _deployment_settings(values, directory):
    if not isinstance(values, dict) or set(values) - DEPLOYMENT_DEFAULTS.keys():
        raise ValueError("Unknown key or invalid table in runpod configuration")
    settings = _merge(DEPLOYMENT_DEFAULTS, values)
    if settings["tts_backend"] not in ("native", "sglang"):
        raise ValueError("runpod.tts_backend must be native or sglang")
    for key in ("name", "image", "gpu_type_id", "data_center_id", "volume_id", "pod_id", "state_dir"):
        value = settings[key]
        if not isinstance(value, str) or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError(f"runpod.{key} must be a string without control characters")
        if key in {"name", "gpu_type_id", "state_dir"} and not value.strip():
            raise ValueError(f"runpod.{key} must not be empty")
    for key in ("name", "data_center_id", "volume_id", "pod_id"):
        value = settings[key]
        if value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError(f"runpod.{key} must contain only letters, numbers, dots, dashes or underscores")
    if settings["image"] and (
            any(c.isspace() for c in settings["image"]) or settings["image"].startswith("-")):
        raise ValueError("runpod.image must be a container image reference")
    if not isinstance(settings["ssh_user"], str) or not re.fullmatch(
            r"[a-z_][a-z0-9_-]{0,31}", settings["ssh_user"]):
        raise ValueError("runpod.ssh_user must be a Linux account name")
    for key, low, high in (
        ("volume_size_gb", 100, 4000), ("container_disk_gb", 10, 1000),
        ("startup_timeout_seconds", 60, 7200), ("idle_timeout_seconds", 60, 86400),
        ("service_port", 1, 65535),
        ("tts_concurrency", 1, 8),
    ):
        if type(settings[key]) is not int or not low <= settings[key] <= high:
            raise ValueError(f"runpod.{key} must be an integer from {low} through {high}")
    if settings["service_port"] in {22, 8080}:
        raise ValueError("runpod.service_port must differ from SSH port 22 and native speech port 8080")
    price = settings["max_hourly_price"]
    if type(price) not in (int, float) or not math.isfinite(price) or not 0 < price <= 100:
        raise ValueError("runpod.max_hourly_price must be a positive hourly limit at most 100")
    for key in ("app_dir", "fish_source", "sglang_source", "remote_config", "model_dir", "whisper_dir", "data_dir"):
        value = settings[key]
        if (not isinstance(value, str) or not value.startswith("/workspace/")
                or ".." in PurePosixPath(value).parts or any(ord(c) < 32 or ord(c) == 127 for c in value)):
            raise ValueError(f"runpod.{key} must be an absolute path below /workspace")
        settings[key] = str(PurePosixPath(value))
    try:
        parsed = urlsplit(settings["github_url"])
        valid = (parsed.scheme == "https" and parsed.hostname == "github.com"
                 and not parsed.username and not parsed.password and not parsed.query
                 and not parsed.fragment and parsed.port is None
                 and re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", parsed.path))
    except (ValueError, TypeError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("runpod.github_url must be a public HTTPS GitHub repository URL")
    ref = settings["git_ref"]
    if (not isinstance(ref, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,199}", ref)
            or ".." in ref or "//" in ref or ref.endswith(("/", "."))):
        raise ValueError("runpod.git_ref must be a Git branch, tag or commit without options")
    settings["state_dir"] = str((directory / Path(settings["state_dir"]).expanduser()).resolve())
    return settings


def load_settings(config_path=None, overrides_path=None):
    """Load the normal config and optional deployment-only override file.

    An explicit missing override is an error. The default runpod.local.toml is
    optional. Relative base paths retain their original file's directory;
    deployment state is relative to the file that sets it.
    """
    base_path = resolve_config_path(config_path).expanduser().resolve()
    base = _read_toml(base_path)
    base_deployment = base.pop("runpod", {})
    local = _validated_renderer(base, base_path.parent)
    override_path = (Path(overrides_path).expanduser().resolve() if overrides_path is not None
                     else base_path.parent / "runpod.local.toml")
    if overrides_path is not None or override_path.exists():
        override = _read_toml(override_path)
    else:
        override = {}
    override_deployment = override.pop("runpod", {})
    if not isinstance(base_deployment, dict) or not isinstance(override_deployment, dict):
        raise ValueError("runpod configuration must be a table")
    if set(override) - _REMOTE_SECTIONS or any(not isinstance(v, dict) for v in override.values()):
        raise ValueError("Runpod overrides may contain only deployment and remote rendering tables")
    merged_deployment = _merge(base_deployment, override_deployment)
    if "state_dir" in base_deployment and "state_dir" not in override_deployment:
        value = base_deployment["state_dir"]
        if isinstance(value, str):
            merged_deployment["state_dir"] = str((base_path.parent / Path(value).expanduser()).resolve())
    deployment = _deployment_settings(merged_deployment, override_path.parent)
    key = override.get("chapter_server", {}).get(
        "runpod_api_key", local["chapter_server"]["runpod_api_key"])
    if not isinstance(key, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in key):
        raise ValueError("chapter_server.runpod_api_key must be a file-configured API key")
    return {"config_path": base_path, "overrides_path": override_path,
            "renderer_config": local, "override_raw": override,
            "deployment": deployment, "runpod_api_key": key}


def build_remote_config(local_renderer_config, override_raw, pod_id, auth_key, *,
                        deployment=None, idle_timeout_seconds=None):
    """Project verification/render intent onto the single-GPU Linux worker.

    Runtime locations come from [runpod]; remote overrides may change shared
    speech/render settings. Provider credentials are included only when needed
    by pronunciation or recurring-failure recovery. Voice-design keys, book
    directories and analysis settings never cross to the worker.
    """
    settings = _deployment_settings(deployment or {}, Path.cwd())
    if not isinstance(override_raw, dict) or set(override_raw) - _REMOTE_SECTIONS:
        raise ValueError("Unsupported remote configuration override")
    config = _merge({key: deepcopy(value) for key, value in local_renderer_config.items()
                     if key in _REMOTE_SECTIONS}, override_raw)
    native = get_drama3_model(DEFAULT_DRAMA3_MODEL)
    config.setdefault("engine", {}).update(
        type="drama3_mlx", model=DEFAULT_DRAMA3_MODEL, checkpoint=settings["model_dir"])
    batched = settings["tts_backend"] == "sglang"
    config["render_pool"] = {
        "model_revision": native.revision,
        "workers": [{"name": "drama3-cuda", "type": "fish_http",
                     "protocol": "fish_sglang_drama3" if batched else "fish_native_drama3",
                     "model": "drama-3", "url": "http://127.0.0.1:8080/" + (
                         "v1/audio/speech" if batched else "v1/tts"),
                     "api_key": "", "concurrency": settings["tts_concurrency"] if batched else 1,
                     "multi_speaker": not batched, "timeout_seconds": 600.0}],
    }
    if batched:
        # One bound across all chapter coordinators, including verification.
        config["render_pool"]["max_inflight"] = 4 * settings["tts_concurrency"]
    remote_verification = override_raw.get("verification", {})
    device = remote_verification.get("device", "cpu")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise ValueError("Remote verification.device must be cpu or cuda")
    config.setdefault("verification", {}).update(
        type="faster_whisper", model=settings["whisper_dir"], device=device, reference_model=None)
    # Remote hardware choices must be explicit; never inherit the Mac's type.
    # Supported types on the actual GPU are checked before starting inference.
    if "compute_type" not in remote_verification:
        config["verification"]["compute_type"] = "int8_float16" if device == "cuda" else "int8"
    server = config.setdefault("chapter_server", {})
    server.update(host="0.0.0.0", port=settings["service_port"], api_key=auth_key, data_dir=settings["data_dir"],
                  persistent_data=True, runpod_pod_id=pod_id, idle_action="runpod_stop",
                  idle_timeout_seconds=(settings["idle_timeout_seconds"]
                                        if idle_timeout_seconds is None else idle_timeout_seconds))
    render = config.setdefault("render", {})
    for key in ("work_dir", "output_dir", "epub", "chapter_numbers"):
        render.pop(key, None)
    render.update(chapter_format="m4a", ffmpeg="ffmpeg", ffprobe="ffprobe")
    verification = config["verification"]
    if all(verification.get(key, "analysis") == "off"
           for key in ("phonetic_fallback", "recurring_resolution")):
        config.pop("provider", None)
    # Validate before uploading secrets or starting a paid worker.
    _validated_renderer(config, Path("/"))
    return config


def build_client_config(local_renderer_config, server_url, auth_key, override_raw=None):
    """Select remote chapters for this operation without editing the user's file."""
    from .chapter_protocol import SAFE_RENDER_KEYS

    config = deepcopy(local_renderer_config)
    if override_raw is not None:
        if not isinstance(override_raw, dict) or not isinstance(override_raw.get("render", {}), dict):
            raise ValueError("Remote render overrides must be a table")
        # The client sends passage and assembly settings in each job. Apply
        # these same overrides here so a local default cannot silently undo a
        # remote timing choice. Executables and output paths remain local.
        keys = SAFE_RENDER_KEYS | {"chapter_numbers", "allow_review"}
        config["render"].update({key: deepcopy(value) for key, value in
                                 override_raw.get("render", {}).items() if key in keys})
    config["render"]["chapter_format"] = "m4a"
    config["remote_chapters"].update(url=server_url, api_key=auth_key)
    _validated_renderer(config, Path.cwd())
    return config
