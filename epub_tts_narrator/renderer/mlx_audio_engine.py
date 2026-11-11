"""Reference-conditioned Drama 3 synthesis using the supported MLX-Audio port.

MLX-Audio's public ``generate`` accepts neither prior turns nor returns codes,
so this adapter drives the same model internals ``generate`` uses
(``_voice``, ``build_frames``, ``_frames`` and the codec's streaming decoder)
for one chunk with the chapter history. Fewer frames than ``max_new_tokens``
means the frame loop encountered EOS; the cap is rejected. Transcript
verification remains the next stage. The internals are pinned by the
``mlx-audio`` requirement and checked before use.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .audio_naming import temporary_audio_prefix
from .drama3_models import (
    MLX_AUDIO_DRAMA3_REQUIREMENT, get_drama3_model, supports_mlx_audio_drama3,
)
from .engines import EngineError, HistorySynthesis, _checkpoint_fingerprints
from .fish_http import read_reference

# Same prompt ceiling as the pinned Sogni runtime (drama3_mlx/prompt.py).
MAX_PROMPT_ROWS = 8192


class MLXAudioDrama3Engine(HistorySynthesis):
    """One lazy, local quantized model; saved references retain their identity."""

    def __init__(
        self, checkpoint: str | Path, *, model: str,
        temperature: float = 0.7, top_p: float = 0.7, top_k: int = 30,
        max_new_tokens: int = 2048, min_audio_seconds: float = 0.1,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self._validate_checkpoint()
        try:
            selected = get_drama3_model(model)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        if selected.runtime != "mlx_audio":
            raise EngineError("The selected model requires the Sogni Drama 3 adapter")
        for name, value in (("temperature", temperature), ("top_p", top_p),
                            ("min_audio_seconds", min_audio_seconds)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise EngineError(f"{name} must be a positive finite number")
        if top_p > 1:
            raise EngineError("top_p must be at most 1")
        if type(top_k) is not int or top_k < 1:
            raise EngineError("top_k must be a positive integer")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
            raise EngineError("max_new_tokens must be an integer from 1 through 2048")
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.min_audio_seconds = float(min_audio_seconds)
        self._model: Any = None
        self._identity: dict[str, Any] | None = None

    def _validate_checkpoint(self) -> None:
        try:
            config = json.loads((self.checkpoint / "config.json").read_text())
        except (OSError, ValueError) as exc:
            raise EngineError(
                "MLX-Audio Drama 3 needs a local checkpoint directory with config.json"
            ) from exc
        if not isinstance(config, dict) or config.get("model_type") != "fish_drama3":
            raise EngineError("The configured MLX-Audio checkpoint must have model_type='fish_drama3'")

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "voice_design": False, "reference_voice": True,
            "preset_voice": False, "delivery_control": True,
            "multi_speaker": True, "unconditioned_voice_creation": False,
            "structured_performance": True, "chapter_continuity": True,
        }

    @property
    def voice_binding_identity(self) -> dict[str, str]:
        """Saved Drama 3 references are portable across these quantized ports."""
        return {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    @property
    def identity(self) -> dict[str, Any]:
        return self.get_identity()

    def get_identity(self, *, cache_fingerprints=True) -> dict[str, Any]:
        """Read synthesis identity without writing model metadata during audits."""
        if self._identity is None:
            fingerprints, revision = _checkpoint_fingerprints(
                self.checkpoint, extra_names=("chat_template.jinja",), write_cache=cache_fingerprints,
            )
            packages = {}
            for name in ("mlx-audio", "mlx", "mlx-metal", "mlx-lm", "numpy",
                         "soundfile", "tokenizers", "miniaudio"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            self._identity = {
                "engine": "drama3_mlx_audio", "adapter_version": 1,
                "model": self.model, "runtime": "mlx-audio",
                "required_runtime_version": MLX_AUDIO_DRAMA3_REQUIREMENT,
                "installed_version": packages["mlx-audio"], "packages": packages,
                "recommended_model_revision": get_drama3_model(self.model).revision,
                "model_revision": revision, "checkpoint": str(self.checkpoint),
                "checkpoint_files_sha256": fingerprints,
                "temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "max_new_tokens": self.max_new_tokens,
                "min_audio_seconds": self.min_audio_seconds,
                "stream": False, "chunking": "single-chunk-chapter-history",
                "completion_check": "eos-before-max_new_tokens",
                "reference_decoder": "mlx_audio.audio_io.read-float32-mono",
                "output_encoding": "wav-float32-mono",
            }
        return deepcopy(self._identity)

    def _load(self) -> Any:
        if self._model is None:
            self._validate_checkpoint()
            try:
                version = importlib.metadata.version("mlx-audio")
            except importlib.metadata.PackageNotFoundError:
                version = None
            if not supports_mlx_audio_drama3(version):
                raise EngineError(
                    f"MLX-Audio Drama 3 requires a stable mlx-audio{MLX_AUDIO_DRAMA3_REQUIREMENT} release; "
                    "run uv pip install --python .venv-renderer/bin/python "
                    "-r renderer.requirements.txt. "
                    f"Installed mlx-audio version: {version or 'missing'}"
                )
            try:
                from mlx_audio.tts.utils import load_model
                model = load_model(str(self.checkpoint))
                if getattr(getattr(model, "config", None), "model_type", None) != "fish_drama3":
                    raise EngineError("Loaded MLX-Audio model is not a fish_drama3 checkpoint")
                if type(model.sample_rate) is not int or model.sample_rate <= 0:
                    raise EngineError("MLX-Audio Drama 3 returned an invalid sample rate")
                self._model = model
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"Could not load local MLX-Audio Drama 3: {exc}") from exc
        return self._model

    def warmup(self) -> None:
        self._load()

    def close(self) -> None:
        loaded = self._model is not None
        self._model = None
        if loaded:
            gc.collect()
            try:
                import mlx.core as mx
            except ImportError:
                return
            mx.synchronize()
            mx.clear_cache()

    def create_voice(self, description, text, output_path, *, seed):
        raise EngineError(
            "MLX-Audio Drama 3 cannot design a voice from a description. Configure "
            "a separate [voice_design] provider or select an existing saved reference voice."
        )

    def _generate_wire(self, text, references, history, output_path, *, seed):
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        # Capture and verify every reference before loading the large model.
        blobs = {voice["speaker"]: read_reference(voice) for voice in references}
        from .continuity import candidates
        from .continuity_codes import history_digest
        try:
            import numpy as np
            import soundfile as sf

            for blob, _ in blobs.values():
                original, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                if (type(rate) is not int or rate <= 0 or original.ndim != 2
                        or not original.shape[0] or not original.shape[1]
                        or not np.isfinite(original).all()):
                    raise EngineError("Saved reference audio is empty or malformed")

            model = self._load()
            if not all(callable(getattr(model, name, None)) for name in ("_voice", "build_frames", "_frames")) \
                    or not callable(getattr(model.codec, "streaming_step", None)):
                raise EngineError("The installed mlx-audio Drama 3 internals do not support chapter history")
            import mlx.core as mx
            from mlx_audio.audio_io import read as audio_read

            rate = model.sample_rate
            decoded = {}
            for speaker, (blob, transcript) in blobs.items():
                # Match the runtime's path-based decoder using verified bytes,
                # so a file changed since validation cannot reach conditioning.
                audio, ref_rate = audio_read(
                    io.BytesIO(blob), dtype="float32", sample_rate=rate, nchannels=1,
                )
                audio = np.asarray(audio, dtype=np.float32)
                if (type(ref_rate) is not int or ref_rate != rate or audio.ndim != 1 or not len(audio)
                        or not np.isfinite(audio).all()):
                    raise EngineError("Decoded reference audio is empty or malformed")
                decoded[speaker] = (audio, f"<|speaker:{speaker}|> {transcript}")
            for selected, kept in candidates(text, references, history):
                voice = model._voice([decoded[item["speaker"]][0] for item in selected],
                                     [decoded[item["speaker"]][1] for item in selected])
                turns = [(turn_text, mx.array(np.array(codes, dtype=np.int32))) for turn_text, codes in kept]
                frames, system_rows = model.build_frames(voice.references, turns, text)
                if frames.shape[0] > MAX_PROMPT_ROWS:
                    continue
                mx.synchronize()
                from .role_sampling import effective
                sampling = effective(self)
                mx.random.seed(seed)
                codes = list(model._frames(frames, system_rows, voice, self.max_new_tokens,
                                           sampling["temperature"], sampling["top_k"], sampling["top_p"]))
                if not codes:
                    raise EngineError("MLX-Audio Drama 3 returned no audio frames")
                if len(codes) >= self.max_new_tokens:
                    raise EngineError("MLX-Audio Drama 3 reached the token limit; the take may be incomplete")
                stacked = mx.stack(codes, axis=1)
                model.codec.reset_streaming_state()
                audio = model.codec.streaming_step(stacked[None])[0, 0]
                mx.eval(audio)
                audio = np.asarray(audio, dtype=np.float32).copy()
                generated = np.asarray(stacked, dtype=np.int64)
                mx.synchronize()
                break
            else:
                raise EngineError("MLX-Audio Drama 3 prompt is too long even without chapter history")
            if (audio.ndim != 1 or not np.isfinite(audio).all()
                    or len(audio) < math.ceil(rate * self.min_audio_seconds)):
                raise EngineError("MLX-Audio Drama 3 returned empty, malformed or implausibly short audio")
            if generated.ndim != 2 or generated.shape[0] != model.num_codebooks:
                raise EngineError("MLX-Audio Drama 3 returned invalid generated codec tokens")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            staged_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=temporary_audio_prefix(output_path, "mlx-audio", suffix=".wav"),
                    suffix=".wav", dir=output_path.parent, delete=False,
                ) as staged:
                    staged_name = staged.name
                sf.write(staged_name, audio, rate, subtype="FLOAT")
                os.link(staged_name, output_path)
            finally:
                if staged_name is not None:
                    Path(staged_name).unlink(missing_ok=True)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"MLX-Audio Drama 3 generation failed: {exc}") from exc
        metadata = {
            "finish_reason": "inferred_eos",
            "completion_evidence": "The frame loop returned EOS before max_new_tokens",
            "sample_rate": rate, "sample_count": len(audio),
            "duration_seconds": len(audio) / rate, "seed": seed,
            "chunk_count": 1, "token_counts": [len(codes)],
            "synthesis_text": text, "history_turns": len(kept),
            "history_available": len(history), "history_sha256": history_digest(kept),
        }
        return metadata, generated.tolist(), model.tower.vocab_size
