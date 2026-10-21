"""Offer explicit, resumable downloads for known local speech checkpoints.

This module checks files without importing MLX or loading speech models. It is
called only when a command needs a component, so inspecting casts or assembling
saved chapters does not require speech checkpoints or network access.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import shlex
import sys

from .diagnostics import configured_secrets, safe_error_text
from .engines import DRAMA3_MODEL_REVISION


class ModelSetupError(RuntimeError):
    """Local inference prerequisites remain unavailable; saved work is intact."""


@dataclass(frozen=True)
class _Model:
    kind: str
    repo: str
    revision: str
    label: str
    size: str


DRAMA = _Model("drama", "fishaudio/drama-3", DRAMA3_MODEL_REVISION,
               "Fish Audio Drama 3", "about 41.5 GB")
WHISPER = _Model("whisper", "mlx-community/whisper-large-v3-turbo",
                 "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
                 "MLX Whisper large-v3-turbo", "about 1.6 GB")
PARAKEET = _Model("parakeet", "mlx-community/parakeet-tdt-0.6b-v3",
                  "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
                  "MLX Parakeet", "about 2.5 GB")
QWEN = _Model("qwen", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
              "7d3824abff87e49756bb0f83fb5411de75d160c4",
              "Qwen3 voice design", "about 4.5 GB")


def _snapshot_target(path, model):
    """Recognize the Hub's repository cache layout without guessing a new model."""
    if path.parent.name != "snapshots" or not path.parent.parent.name.startswith("models--"):
        return model, None
    repository = path.parent.parent.name
    if repository != "models--" + model.repo.replace("/", "--"):
        raise ModelSetupError(
            f"The Hugging Face cache path {path} belongs to a different repository. "
            f"It will not be filled with {model.repo}; choose the correct model directory in the global config.")
    if re.fullmatch(r"[0-9a-f]{40}", path.name) is None:
        raise ModelSetupError(
            f"The Hugging Face snapshot path {path} needs a full 40-character commit revision. "
            "Choose the snapshot directory or use a local model directory in the global config.")
    # Snapshot names identify immutable commits. Keep the requested revision,
    # and let the Hub manage shared blobs/symlinks instead of using local_dir.
    return replace(model, revision=path.name), path.parents[2]


def _file(path):
    return path.is_file() and path.stat().st_size > 0


def _json(path):
    if not _file(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def faster_whisper_model_files(path):
    """Inspect an offline CTranslate2 directory without loading ASR or the Hub.

    A local tokenizer is mandatory: faster-whisper otherwise falls back to a
    Hub tokenizer even when the supplied weights are already on disk.
    """
    path = Path(path)
    return {
        "config": _json(path / "config.json") is not None,
        "weights": _file(path / "model.bin"),
        "tokenizer": _json(path / "tokenizer.json") is not None,
    }


def _weight_files(path, *, index_required=False):
    """List missing weight assets, including every shard in an existing index."""
    index_path = path / "model.safetensors.index.json"
    if index_path.exists() or index_required:
        index = _json(index_path)
        weights = index.get("weight_map") if index else None
        if not isinstance(weights, dict) or not weights:
            return ["model.safetensors.index.json (missing or invalid)"]
        names = set()
        for value in weights.values():
            if (not isinstance(value, str) or not value
                    or Path(value).name != value or value in {".", ".."}
                    or not value.endswith(".safetensors")):
                raise ModelSetupError(f"Unsafe model shard path in {index_path}")
            names.add(value)
        return [name for name in sorted(names) if not _file(path / name)]
    return [] if any(_file(p) for p in path.glob("*.safetensors")
                     if p.name != "codec.safetensors") else ["model safetensors weights"]


def _missing(path, model):
    if path.exists() and not path.is_dir():
        raise ModelSetupError(f"Model checkpoint path is not a directory: {path}")
    metadata = _json(path / "config.json")
    model_type = metadata.get("model_type") if metadata else None
    expected = {"drama": {"fish_omni"}, "whisper": {"whisper"},
                "qwen": {"qwen3_tts"}, "parakeet": {"parakeet"}}[model.kind]
    if model_type is not None and (not isinstance(model_type, str) or model_type not in expected):
        raise ModelSetupError(
            f"The checkpoint at {path} identifies a different model ({model_type}). "
            f"It will not be overwritten with {model.repo}; choose the correct directory in the global config.")
    architectures = metadata.get("architectures") if metadata else None
    expected_architecture = {"drama": "FishOmniModel", "whisper": "WhisperForConditionalGeneration",
                             "qwen": "Qwen3TTSForConditionalGeneration"}.get(model.kind)
    if (expected_architecture and isinstance(architectures, list) and architectures
            and expected_architecture not in architectures):
        raise ModelSetupError(
            f"The checkpoint at {path} has incompatible architecture metadata. "
            f"It will not be overwritten with {model.repo}; choose the correct model directory.")
    if (model.kind == "qwen" and metadata
            and metadata.get("tts_model_type", "voice_design") != "voice_design"):
        raise ModelSetupError(f"The Qwen checkpoint at {path} is not a voice-design model; it will not be overwritten.")
    missing = [] if metadata is not None else ["config.json (missing or invalid)"]
    if model.kind == "drama":
        if not _file(path / "tokenizer.json"):
            missing.append("tokenizer.json")
        if not any(_file(path / name) for name in ("codec.pth", "codec.safetensors")):
            missing.append("codec.pth or codec.safetensors")
        converted = bool(metadata and "drama3_mlx" in metadata)
        missing.extend(_weight_files(path, index_required=not converted))
        if converted and missing:
            raise ModelSetupError(
                f"Converted Drama 3 checkpoint is incomplete at {path}: {', '.join(missing)}. "
                "Restore its converted files or select another directory; official release files will not be mixed into it.")
    elif model.kind == "whisper":
        if not any(_file(path / name) for name in ("weights.safetensors", "weights.npz")):
            missing.append("weights.safetensors or weights.npz")
    else:
        missing.extend(_weight_files(path))
        if model.kind == "qwen":
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "speech_tokenizer/config.json"):
                if not _file(path / name):
                    missing.append(name)
            missing.extend("speech_tokenizer/" + name
                           for name in _weight_files(path / "speech_tokenizer"))
    return missing


class ModelSetup:
    """Prepare only requested components, using the file-configured HF token.

    ``ensure(component, settings=...)`` accepts engine, verification, or
    voice_design. An optional settings table supports a temporary cast designer.
    Refusals and download failures are remembered for this instance; installing
    the files manually makes the next call succeed without another prompt.
    """

    def __init__(self, config, *, input_stream=None, output_stream=None, downloader=None):
        self.config = config
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.downloader = downloader
        self._stopped = {}

    def __call__(self, component, *, settings=None):
        return self.ensure(component, settings=settings)

    def ensure(self, component, *, settings=None):
        if component not in {"engine", "verification", "voice_design"}:
            raise ValueError(f"Unknown model component: {component}")
        settings = self.config.get(component, {}) if settings is None else settings
        requested = []
        if component == "engine" and settings.get("type") == "drama3_mlx":
            requested.append((settings.get("checkpoint"), DRAMA, True, "engine.checkpoint"))
        elif component == "verification" and settings.get("type", "mlx_whisper") == "mlx_whisper":
            requested.append((settings.get("model"), WHISPER, False, "verification.model"))
            if settings.get("reference_model"):
                requested.append((settings["reference_model"], PARAKEET, False, "verification.reference_model"))
        elif component == "verification" and settings.get("type") == "faster_whisper":
            for key in ("model", "reference_model"):
                value = settings.get(key)
                if key == "reference_model" and value is None:
                    continue
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ModelSetupError(f"Set verification.{key} to a local CTranslate2 Whisper directory in the global config.")
                path = Path(value).expanduser().absolute()
                files = faster_whisper_model_files(path)
                names = {"config": "config.json (valid JSON)", "weights": "model.bin (nonempty)",
                         "tokenizer": "tokenizer.json (valid JSON)"}
                missing = [names[name] for name, present in files.items() if not present]
                if missing:
                    raise ModelSetupError(
                        f"Local faster-whisper model is incomplete at {path}: {', '.join(missing)}. "
                        f"Set verification.{key} to an existing CTranslate2 Whisper model directory. "
                        "No model or tokenizer will be downloaded automatically.")
            return
        elif component == "voice_design" and settings.get("type") == "qwen3_mlx":
            requested.append((settings.get("checkpoint"), QWEN, True, "voice_design.checkpoint"))
        for value, model, identifiable, setting in requested:
            if not isinstance(value, (str, Path)) or not str(value):
                raise ModelSetupError(f"Set {setting} to a local model directory in the global config.")
            path = Path(value).expanduser().absolute()
            missing = _missing(path, model)
            if not missing:
                continue
            model, cache_dir = _snapshot_target(path, model)
            if cache_dir is None and not identifiable and path.name != model.repo.rsplit("/", 1)[-1]:
                raise ModelSetupError(
                    f"Local model is incomplete at {path}: {', '.join(missing)}. "
                    f"The repository for custom {setting} cannot be inferred safely. "
                    "Restore that model manually, or use the documented "
                    f"{model.repo.rsplit('/', 1)[-1]} directory for automatic download.")
            self._ensure(path, model, missing, cache_dir=cache_dir)

    def _manual(self, path, model, *, cache_dir=None):
        destination = ["--cache-dir", str(cache_dir)] if cache_dir is not None else ["--local-dir", str(path)]
        command = shlex.join([".venv-renderer/bin/hf", "download", model.repo,
                              "--revision", model.revision, *destination])
        return ("Rerun the same audiobook command from an interactive terminal to retry, "
                "or download manually:\n" + command)

    def _message(self, value):
        return safe_error_text(value, secrets=configured_secrets(self.config))

    def _confirm(self, path, model, missing):
        with ExitStack() as stack:
            source = self.input_stream if self.input_stream is not None else sys.stdin
            output = self.output_stream if self.output_stream is not None else sys.stderr
            if self.input_stream is None and not getattr(source, "isatty", lambda: False)():
                try:
                    source = stack.enter_context(open("/dev/tty", "r", encoding="utf-8"))
                    if self.output_stream is None:
                        output = stack.enter_context(open("/dev/tty", "w", encoding="utf-8"))
                except OSError:
                    return False, "No interactive terminal is available"
            if not getattr(source, "isatty", lambda: False)():
                return False, "No interactive terminal is available"
            print(self._message(
                f"\n{model.label} is missing or incomplete.\n"
                f"Repository: {model.repo}\nRevision: {model.revision}\n"
                f"Download: {model.size}\nDirectory: {path}\n"
                f"Missing: {', '.join(missing)}\n"
                "Completed download files will be reused if you rerun this command."), file=output, flush=True)
            print("Download this model now? [y/N] ", end="", file=output, flush=True)
            try:
                answer = source.readline()
            except EOFError:
                answer = ""
            return answer.strip().casefold() in {"y", "yes"}, "Model download declined"

    def _ensure(self, path, model, missing, *, cache_dir=None):
        key = (str(path), model.repo, model.revision)
        if key in self._stopped:
            raise ModelSetupError(self._stopped[key])
        manual = self._manual(path, model, cache_dir=cache_dir)
        try:
            accepted, reason = self._confirm(path, model, missing)
            if not accepted:
                raise ModelSetupError(f"{reason}: {model.label} is not ready at {path}.\n{manual}")
            downloader = self.downloader
            if downloader is None:
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as exc:
                    raise ModelSetupError(
                        "The Hugging Face downloader is not installed. Install renderer.requirements.txt "
                        "into the renderer environment, then rerun the same command.") from exc
                downloader = snapshot_download
            token = self.config.get("huggingface", {}).get("api_key") or False
            print(self._message(f"Downloading {model.repo} to {path}…"),
                  file=self.output_stream if self.output_stream is not None else sys.stderr, flush=True)
            destination = {"cache_dir": str(cache_dir)} if cache_dir is not None else {"local_dir": str(path)}
            downloader(repo_id=model.repo, revision=model.revision,
                       token=token, max_workers=5, **destination)
            remaining = _missing(path, model)
            if remaining:
                raise ModelSetupError(f"Download returned but the checkpoint is still incomplete: {', '.join(remaining)}")
        except KeyboardInterrupt:
            message = self._message(f"Model download interrupted. Downloaded files remain at {path}.\n{manual}")
            self._stopped[key] = message
            print(message, file=self.output_stream if self.output_stream is not None else sys.stderr, flush=True)
            raise
        except Exception as exc:
            if isinstance(exc, ModelSetupError) and ("declined" in str(exc) or "No interactive" in str(exc)):
                message = self._message(exc)
            else:
                message = self._message(
                    f"Could not prepare {model.label}: {exc}\n"
                    f"Downloaded and partial files remain at {path}; they can be reused on retry.\n{manual}")
            self._stopped[key] = message
            raise ModelSetupError(message) from exc
        print(self._message(f"{model.label} is ready at {path}."),
              file=self.output_stream if self.output_stream is not None else sys.stderr, flush=True)
