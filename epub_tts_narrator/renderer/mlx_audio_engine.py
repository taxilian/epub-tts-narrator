"""Reference-conditioned Drama 3 synthesis using the supported MLX-Audio port.

MLX-Audio generates one nonstreaming result per text chunk. A result has no
finish_reason: generating fewer frames than max_tokens means its frame
loop encountered EOS. We reject the cap and require every expected text chunk
before publishing any audio; transcript verification remains the next stage.
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

from .drama3_models import (
    MLX_AUDIO_DRAMA3_REQUIREMENT, get_drama3_model, supports_mlx_audio_drama3,
)
from .engines import (
    EngineError, _checkpoint_fingerprints, _directed_text, _seed, _text,
    format_dialogue,
)
from .fish_http import read_reference, validate_wire_text


class MLXAudioDrama3Engine:
    """One lazy, local quantized model; saved references retain their identity."""

    def __init__(
        self, checkpoint: str | Path, *, model: str,
        temperature: float = 1.0, top_p: float = 0.9, top_k: int = 30,
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
        }

    @property
    def voice_binding_identity(self) -> dict[str, str]:
        """Saved Drama 3 references are portable across these quantized ports."""
        return {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    @property
    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            fingerprints, revision = _checkpoint_fingerprints(
                self.checkpoint, extra_names=("chat_template.jinja",),
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
                "stream": False, "chunking": "mlx-audio-fish_drama3",
                "completion_check": "all-text-chunks-and-each-token-count-below-cap",
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

    def synthesize(
        self, text: str, voice: dict[str, Any], output_path: Path, *,
        seed: int, direction: str = "",
    ) -> dict[str, Any]:
        _seed(seed)
        engine_text, delivery = _directed_text(_text(text, "text"), direction)
        return {
            **self._generate_audio(engine_text, [voice], [None], output_path, seed=seed),
            **delivery,
        }

    def synthesize_dialogue(
        self, turns: list[dict[str, Any]], output_path: Path, *, seed: int,
    ) -> dict[str, Any]:
        _seed(seed)
        text, voices, metadata = format_dialogue(turns)
        return {
            **self._generate_audio(text, voices, list(range(len(voices))), output_path, seed=seed),
            **metadata,
        }

    def synthesize_fish(
        self, text: str, voices: list[dict[str, Any]], output_path: Path, *, seed: int,
    ) -> dict[str, Any]:
        _seed(seed)
        if not isinstance(voices, list) or not voices or any(not isinstance(v, dict) for v in voices):
            raise EngineError("MLX-Audio Drama 3 requires saved reference voices")
        speakers = [voice.get("speaker") for voice in voices]
        if any(s is not None and (type(s) is not int or not 0 <= s <= 15) for s in speakers):
            raise EngineError("Reference speaker IDs must be integers from 0 through 15")
        validate_wire_text(text, set(speakers))
        return self._generate_audio(text, voices, speakers, output_path, seed=seed)

    def _generate_audio(
        self, text: str, voices: list[dict[str, Any]], speakers: list[int | None],
        output_path: Path, *, seed: int,
    ) -> dict[str, Any]:
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        if any(not isinstance(voice, dict) for voice in voices):
            raise EngineError("MLX-Audio Drama 3 requires a saved reference voice")
        # Capture and verify every reference before loading the large model.
        references = [read_reference(voice) for voice in voices]
        try:
            import numpy as np
            import soundfile as sf

            for blob, _ in references:
                original, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                if (type(rate) is not int or rate <= 0 or original.ndim != 2
                        or not original.shape[0] or not original.shape[1]
                        or not np.isfinite(original).all()):
                    raise EngineError("Saved reference audio is empty or malformed")

            model = self._load()
            import mlx.core as mx
            from mlx_audio.audio_io import read as audio_read
            from mlx_audio.tts.models.fish_drama3.chunking import split_text

            rate = model.sample_rate
            audio_refs, text_refs = [], []
            for (blob, transcript), speaker in zip(references, speakers):
                # Match the runtime's path-based decoder using verified bytes,
                # so a file changed since validation cannot reach conditioning.
                audio, ref_rate = audio_read(
                    io.BytesIO(blob), dtype="float32", sample_rate=rate, nchannels=1,
                )
                audio = np.asarray(audio, dtype=np.float32)
                if (type(ref_rate) is not int or ref_rate != rate or audio.ndim != 1 or not len(audio)
                        or not np.isfinite(audio).all()):
                    raise EngineError("Decoded reference audio is empty or malformed")
                audio_refs.append(audio)
                text_refs.append(f"<|speaker:{speaker}|> {transcript}" if speaker is not None else transcript)

            expected_chunks = len(split_text(text))
            if not expected_chunks:
                raise EngineError("MLX-Audio Drama 3 text contains no generation chunks")
            mx.synchronize()
            mx.random.seed(seed)
            iterator = iter(model.generate(
                text=text, ref_audio=audio_refs, ref_text=text_refs,
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                max_tokens=self.max_new_tokens, stream=False,
            ))
            chunks, token_counts = [], []
            try:
                for index, result in enumerate(iterator):
                    if (index >= expected_chunks or type(result.segment_idx) is not int
                            or result.segment_idx != index
                            or type(result.is_final_chunk) is not bool
                            or result.is_final_chunk != (index == expected_chunks - 1)
                            or result.is_streaming_chunk is not False):
                        raise EngineError("MLX-Audio Drama 3 returned incomplete or unordered chunks")
                    if type(result.token_count) is not int or result.token_count <= 0:
                        raise EngineError("MLX-Audio Drama 3 returned invalid token metadata")
                    if result.token_count >= self.max_new_tokens:
                        raise EngineError("MLX-Audio Drama 3 reached the token limit; the take may be incomplete")
                    if type(result.sample_rate) is not int or result.sample_rate != rate:
                        raise EngineError("MLX-Audio Drama 3 returned an invalid sample rate")
                    audio = np.asarray(result.audio, dtype=np.float32)
                    if (audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all()
                            or type(result.samples) is not int or result.samples != len(audio)):
                        raise EngineError("MLX-Audio Drama 3 returned empty or malformed audio")
                    # Keep owned samples if the runtime reuses its output buffer.
                    chunks.append(audio.copy())
                    token_counts.append(result.token_count)
            finally:
                if callable(getattr(iterator, "close", None)):
                    iterator.close()
                mx.synchronize()
            if len(chunks) != expected_chunks:
                raise EngineError("MLX-Audio Drama 3 did not return every text chunk")
            audio = np.concatenate(chunks)
            if len(audio) < math.ceil(rate * self.min_audio_seconds):
                raise EngineError("MLX-Audio Drama 3 returned implausibly short audio")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            staged_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".take-", suffix=".wav", dir=output_path.parent, delete=False,
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
        return {
            "finish_reason": "inferred_eos",
            "completion_evidence": "Every text chunk returned fewer frames than max_new_tokens",
            "sample_rate": rate, "sample_count": len(audio),
            "duration_seconds": len(audio) / rate, "seed": seed,
            "chunk_count": len(chunks), "token_counts": token_counts,
        }
