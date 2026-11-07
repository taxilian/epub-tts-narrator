"""Drama 3 on CUDA through the pinned Fish Speech library, with chapter history.

Fish's stock ``tools/api_server.py`` accepts neither prior turns nor returns
codec tokens, so the Runpod worker runs ``audiobook-render serve`` with this
engine inside Fish's own environment instead. It calls the same unmodified
library functions the stock server uses for Drama 3: ``init_model``,
``load_codec_model``, ``build_drama3_prompt`` (whose ``history`` argument holds
completed user/assistant turns), ``generate_drama3`` and ``set_seed``.
Reference encoding mirrors Fish's ``VQManager.encode_reference``. Every import
is lazy; the coordinator never loads torch.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .audio_naming import temporary_audio_prefix
from .drama3_models import DEFAULT_DRAMA3_MODEL, DRAMA3_MODELS
from .engines import EngineError, HistorySynthesis, _checkpoint_fingerprints
from .fish_http import read_reference


FISH_SPEECH_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"


class FishSpeechCudaEngine(HistorySynthesis):
    """One lazy CUDA model instance; references cached by content and transcript."""

    def __init__(self, checkpoint: str | Path, *, temperature: float = 0.7, top_p: float = 0.7,
                 top_k: int = 30, max_new_tokens: int = 2048, min_audio_seconds: float = 0.1,
                 device: str = "cuda:0") -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise EngineError(f"Drama 3 checkpoint directory does not exist: {self.checkpoint}")
        for name, value in (("temperature", temperature), ("top_p", top_p),
                            ("min_audio_seconds", min_audio_seconds)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise EngineError(f"{name} must be a positive finite number")
        if top_p > 1 or temperature > 1:
            raise EngineError("Fish Speech requires temperature and top_p of at most 1")
        if type(top_k) is not int or top_k < 1:
            raise EngineError("top_k must be a positive integer")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
            raise EngineError("max_new_tokens must be an integer from 1 through 2048")
        if not isinstance(device, str) or not device.startswith("cuda"):
            raise EngineError("fish_speech_cuda requires a CUDA device such as cuda:0")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.min_audio_seconds = float(min_audio_seconds)
        self.device = device
        self._model: Any = None
        self._codec: Any = None
        self._references: dict[tuple[str, str], Any] = {}
        self._identity: dict[str, Any] | None = None

    @property
    def capabilities(self) -> dict[str, bool]:
        return {"voice_design": False, "reference_voice": True, "preset_voice": False,
                "delivery_control": True, "multi_speaker": True, "structured_performance": True,
                "unconditioned_voice_creation": False, "chapter_continuity": True}

    @property
    def voice_binding_identity(self) -> dict[str, str]:
        return {"engine": "drama3_mlx", "model": DEFAULT_DRAMA3_MODEL}

    @property
    def identity(self) -> dict[str, Any]:
        if self._identity is None:
            fingerprints, revision = _checkpoint_fingerprints(self.checkpoint)
            packages = {}
            for name in ("fish-speech", "torch", "torchaudio", "numpy", "soundfile", "tokenizers"):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            self._identity = {
                "engine": "fish_speech_cuda", "adapter_version": 1, "model": DEFAULT_DRAMA3_MODEL,
                "runtime": "fishaudio/fish-speech", "required_runtime_revision": FISH_SPEECH_REVISION,
                "recommended_model_revision": DRAMA3_MODELS[DEFAULT_DRAMA3_MODEL].revision,
                "model_revision": revision, "packages": packages, "checkpoint": str(self.checkpoint),
                "checkpoint_files_sha256": fingerprints, "precision": "bfloat16", "device": self.device,
                "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "max_new_tokens": self.max_new_tokens, "min_audio_seconds": self.min_audio_seconds,
                "single_chunk": True, "output_encoding": "wav-float32-mono",
            }
        return deepcopy(self._identity)

    def warmup(self) -> None:
        self._load()

    def close(self) -> None:
        loaded = self._model is not None
        self._model = self._codec = None
        self._references.clear()
        if loaded:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    def create_voice(self, description, text, output_path, *, seed):
        raise EngineError("fish_speech_cuda renders saved reference voices; create voices with the cast tool")

    def _load(self):
        if self._model is None:
            try:
                import torch
                from fish_speech.models.text2semantic.inference import (
                    init_model, load_codec_model, resolve_codec_path)
                from fish_speech.models.text2semantic.fish_omni.generate import (
                    resolve_drama3_devices, resolve_drama3_dtype)
                # Same choices as the stock server: bfloat16, no compilation,
                # codec on the tower device in the model's dtype.
                model, _ = init_model(str(self.checkpoint), self.device, torch.bfloat16)
                devices, tower = resolve_drama3_devices(self.device)
                precision = resolve_drama3_dtype(torch.bfloat16, [*devices, tower])
                self._codec = load_codec_model(resolve_codec_path(self.checkpoint), tower, precision)
                self._model = model
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"Could not load Fish Speech Drama 3: {exc}") from exc
        return self._model, self._codec

    def _reference_codes(self, voice, codec):
        blob, transcript = read_reference(voice)
        key = (hashlib.sha256(blob).hexdigest(), transcript)
        if key not in self._references:
            import torch
            import torchaudio
            waveform, rate = torchaudio.load(io.BytesIO(blob))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if rate != codec.sample_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=rate, new_freq=codec.sample_rate)(waveform)
            audio = waveform.squeeze().to(device=codec.device, dtype=next(codec.parameters()).dtype)
            if audio.ndim != 1 or not audio.numel():
                raise EngineError("Reference audio is empty or malformed")
            with torch.inference_mode():
                lengths = torch.tensor([audio.shape[0]], device=codec.device, dtype=torch.long)
                codes = codec.encode(audio[None, None, :], lengths)[0][0]
            self._references[key] = (transcript, codes.cpu())
        return self._references[key]

    def _generate_wire(self, text, references, history, output_path, *, seed):
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        from .continuity import candidates
        from .continuity_codes import history_digest
        model, codec = self._load()
        try:
            import numpy as np
            import soundfile as sf
            import torch
            from fish_speech.models.text2semantic.fish_omni.generate import (
                Drama3DecodeGraphs, Drama3PromptTooLongError, build_drama3_prompt, generate_drama3)
            from fish_speech.utils import set_seed
            capacity = Drama3DecodeGraphs.graph_capacity(model)
            for selected, kept in candidates(text, references, history):
                texts, tokens = [], []
                for voice in selected:
                    transcript, codes = self._reference_codes(voice, codec)
                    texts.append(f"<|speaker:{voice['speaker']}|> {transcript}")
                    tokens.append(codes)
                turns = [(turn_text, torch.tensor(codes, dtype=torch.long)) for turn_text, codes in kept]
                try:
                    prompt = build_drama3_prompt(text, model.tokenizer, model.config, texts, tokens, history=turns)
                except Drama3PromptTooLongError:
                    continue
                if capacity - len(prompt) < 1:
                    continue
                # Fish's set_seed clamps above 2**31; fold uint32 attempt seeds.
                set_seed(seed % (1 << 31))
                with torch.inference_mode():
                    frames, finish_reason = generate_drama3(
                        model, prompt, max_new_tokens=min(self.max_new_tokens, capacity - len(prompt)),
                        temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
                    if finish_reason != "eos" or not len(frames):
                        raise EngineError(f"Drama 3 did not finish the text: {finish_reason!r}")
                    generated = frames[:, 1:].T.contiguous()
                    audio = codec.from_indices(generated.to(codec.device)[None])[0].squeeze()
                audio = audio.float().cpu().numpy().astype(np.float32)
                break
            else:
                raise EngineError("Drama 3 prompt is too long even without chapter history")
            rate = codec.sample_rate
            if (audio.ndim != 1 or not np.isfinite(audio).all()
                    or len(audio) < math.ceil(rate * self.min_audio_seconds)):
                raise EngineError("Drama 3 returned empty, malformed or implausibly short audio")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            staged = None
            try:
                with tempfile.NamedTemporaryFile(
                        prefix=temporary_audio_prefix(output_path, "drama3-cuda", suffix=".wav"),
                        suffix=".wav", dir=output_path.parent, delete=False) as handle:
                    staged = handle.name
                sf.write(staged, audio, rate, subtype="FLOAT")
                os.link(staged, output_path)
            finally:
                if staged is not None:
                    Path(staged).unlink(missing_ok=True)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Fish Speech Drama 3 generation failed: {exc}") from exc
        metadata = {"finish_reason": "eos", "sample_rate": rate, "sample_count": len(audio),
                    "duration_seconds": len(audio) / rate, "seed": seed, "synthesis_text": text,
                    "history_turns": len(kept), "history_available": len(history),
                    "history_sha256": history_digest(kept)}
        return metadata, generated.cpu().tolist(), model.config.audio_decoder.vocab_size
