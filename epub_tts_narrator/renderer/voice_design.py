"""Prompt-conditioned reference creation, separate from book synthesis."""

from __future__ import annotations

from copy import deepcopy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile

from .engines import EngineError, _checkpoint_fingerprints, _seed


QWEN3_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
QWEN3_DESIGN_REVISION = "7d3824abff87e49756bb0f83fb5411de75d160c4"


class Qwen3VoiceDesigner:
    """Create one reusable voice from a description using Qwen3 VoiceDesign.

    A design prompt conditions the model; it is not an objective guarantee of
    every perceptual trait. Speech verification and listening serve different
    purposes. Once saved, this clip is cloned instead of re-running design.
    """

    capabilities = {"voice_design": True}

    def __init__(self, checkpoint, *, language="English", temperature=0.7,
                 max_tokens=2048):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        try:
            config = json.loads((self.checkpoint / "config.json").read_text())
        except (OSError, ValueError) as exc:
            raise EngineError("Voice design needs a local Qwen3 VoiceDesign checkpoint") from exc
        if config.get("tts_model_type") != "voice_design":
            raise EngineError("The configured Qwen3 model is not a VoiceDesign checkpoint")
        if not isinstance(language, str) or not language.strip():
            raise EngineError("voice_design.language must be nonempty text")
        if (isinstance(temperature, bool) or not isinstance(temperature, (float, int))
                or not math.isfinite(temperature) or not 0 < temperature <= 2):
            raise EngineError("voice_design.temperature must be greater than zero and at most 2")
        if type(max_tokens) is not int or not 128 <= max_tokens <= 4096:
            raise EngineError("voice_design.max_tokens must be from 128 through 4096")
        self.language = language
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self._model = None
        self._identity = None

    @property
    def identity(self):
        if self._identity is None:
            extra = ("vocab.json", "merges.txt", "generation_config.json", "preprocessor_config.json")
            hashes, revision = _checkpoint_fingerprints(self.checkpoint, extra_names=extra)
            codec_hashes, _ = _checkpoint_fingerprints(
                self.checkpoint / "speech_tokenizer", extra_names=extra)
            hashes.update({f"speech_tokenizer/{name}": value
                           for name, value in codec_hashes.items()})
            packages = {}
            for name in ("mlx-audio", "mlx", "mlx-lm", "numpy", "soundfile"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            self._identity = {
                "engine": "qwen3_mlx_voice_design", "adapter_version": 1,
                "model": QWEN3_DESIGN_MODEL, "recommended_model_revision": QWEN3_DESIGN_REVISION,
                "model_revision": revision, "checkpoint": str(self.checkpoint),
                "checkpoint_files_sha256": hashes, "packages": packages,
                "language": self.language, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "top_p": 1.0, "top_k": 50,
                "repetition_penalty": 1.05, "output_encoding": "wav-float32-mono",
            }
        return deepcopy(self._identity)

    def _load(self):
        if self._model is None:
            try:
                from mlx_audio.tts.utils import load_model
                model = load_model(str(self.checkpoint))
                if model.config.tts_model_type != "voice_design":
                    raise EngineError("Loaded model does not support descriptive voice design")
                self._model = model
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"Could not load local Qwen3 VoiceDesign: {exc}") from exc
        return self._model

    def create_voice(self, description, text, output_path, *, seed):
        for value, label, limit in ((description, "Voice description", 4000),
                                    (text, "Sample text", 1000)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise EngineError(f"{label} must contain 1 through {limit} characters")
            if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
                raise EngineError(f"{label} contains control characters")
            if any(marker in value for marker in ("<|", "|>")):
                raise EngineError(f"{label} contains model control syntax")
        _seed(seed)
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        try:
            import mlx.core as mx
            import numpy as np
            import soundfile as sf
            model = self._load()
            mx.random.seed(seed)
            iterator = iter(model.generate_voice_design(
                text=text, instruct=description, language=self.language,
                temperature=self.temperature, max_tokens=self.max_tokens,
                top_p=1.0, top_k=50, repetition_penalty=1.05,
                stream=False, verbose=False,
            ))
            result = next(iterator, None)
            if result is None or next(iterator, None) is not None:
                raise EngineError("Qwen3 VoiceDesign must return exactly one audio segment")
            # The MLX result has no finish_reason. Reaching its token cap is
            # therefore rejected before independent transcript verification.
            if result.token_count >= self.max_tokens:
                raise EngineError("Qwen3 VoiceDesign reached the token limit")
            rate = result.sample_rate
            if type(rate) is not int or rate <= 0:
                raise EngineError("Qwen3 VoiceDesign returned an invalid sample rate")
            audio = np.asarray(result.audio, dtype=np.float32)
            if (audio.ndim != 1 or len(audio) < math.ceil(rate * 0.1)
                    or not np.isfinite(audio).all()):
                raise EngineError("Qwen3 VoiceDesign returned empty or malformed audio")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".design-", suffix=".wav", dir=output_path.parent, delete=False,
                ) as stage:
                    temp_name = stage.name
                sf.write(temp_name, audio, rate, subtype="FLOAT")
                os.link(temp_name, output_path)
            finally:
                if temp_name is not None:
                    Path(temp_name).unlink(missing_ok=True)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Qwen3 voice design failed: {exc}") from exc
        return {
            "creation_method": "prompt_conditioned_voice_design",
            "description_applied": True, "requested_description": description,
            "designer": self.identity, "reference_text": text,
            "sample_rate": rate, "sample_count": len(audio),
            "duration_seconds": len(audio) / rate, "seed": seed,
            "token_count": result.token_count,
        }

    def close(self):
        loaded = self._model is not None
        self._model = None
        if loaded:
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()


def create_designer(config):
    """Resolve an explicit reference designer without unconditioned fallback."""
    if config.get("type") == "elevenlabs":
        from .elevenlabs_voice_design import ElevenLabsVoiceDesigner
        allowed = {"type", "api_key", "model", "timeout_seconds", "guidance_scale", "loudness"}
        if set(config) - allowed:
            raise EngineError("Unknown ElevenLabs voice-design settings")
        return ElevenLabsVoiceDesigner(**{key: value for key, value in config.items() if key != "type"})
    if config.get("type") == "gemini":
        from .gemini_voice_design import GeminiVoiceDesigner
        allowed = {"type", "api_key", "model", "language_code", "timeout_seconds"}
        if set(config) - allowed:
            raise EngineError("Unknown Gemini voice-design settings")
        return GeminiVoiceDesigner(**{key: value for key, value in config.items() if key != "type"})
    if config.get("type") == "qwen3_mlx":
        allowed = {"type", "checkpoint", "language", "temperature", "max_tokens", "options"}
        if set(config) - allowed or config.get("options"):
            raise EngineError("Unknown Qwen3 voice-design settings")
        return Qwen3VoiceDesigner(**{
            key: value for key, value in config.items() if key not in {"type", "options"}
        })
    raise EngineError(f"Unknown voice designer: {config.get('type')!r}")
