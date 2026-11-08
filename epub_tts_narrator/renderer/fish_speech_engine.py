"""Drama 3 on CUDA through the pinned Fish Speech library, with chapter history.

Fish's stock ``tools/api_server.py`` accepts neither prior turns nor returns
codec tokens, so the Runpod worker runs ``audiobook-render serve`` with this
engine inside Fish's own environment instead. It calls the same unmodified
library functions the stock server uses for Drama 3: ``init_model``,
``load_codec_model``, ``build_drama3_prompt`` (whose ``history`` argument holds
completed user/assistant turns) and ``set_seed``. Decoding is Fish's
``generate_drama3`` loop reproduced step for step in ``fish_drama3_decode`` so
``silence_stop`` can end a take that has gone silent; takes that end on their
own get Fish's exact frames. Reference encoding mirrors Fish's
``VQManager.encode_reference``. Every import is lazy; the coordinator never
loads torch.

With ``batch_rows`` above 1 (``serve``'s ``server.batch_rows``), prompts are
still built per request with Fish's ``build_drama3_prompt`` and each request
still returns its own WAV and generated codes, but decoding runs in
``drama3_batch``: up to ``batch_rows`` requests advance one frame per step on
the one loaded model.
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
import threading
from typing import Any

from .drama3_models import DEFAULT_DRAMA3_MODEL, DRAMA3_MODELS
from .engines import EngineError, HistorySynthesis, _checkpoint_fingerprints
from .fish_http import read_reference
from .frame_budget import frame_budget
from .silence_stop import SilenceWatch
from .request_sampling import current_sampling

MAX_BATCH_ROWS = 32


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
        self.batch_rows = 1
        self._scheduler: Any = None
        # Concurrent requests share the model load, the reference cache and
        # the codec; decoding itself runs on the scheduler's one thread.
        self._load_lock = threading.Lock()
        self._codec_lock = threading.Lock()

    def set_batch_rows(self, rows: int) -> None:
        """Decode up to ``rows`` concurrent requests per step; set before loading."""
        if type(rows) is not int or not 1 <= rows <= MAX_BATCH_ROWS:
            raise EngineError(f"batch_rows must be an integer from 1 through {MAX_BATCH_ROWS}")
        if self._model is not None:
            raise EngineError("Set batch_rows before the model loads")
        self.batch_rows = rows
        self._identity = None

    @property
    def concurrent_requests(self) -> int:
        return self.batch_rows

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
            if self.batch_rows > 1:
                # Batched takes are not bit-identical to single-request takes.
                from .drama3_batch import DECODER_VERSION, SAMPLER
                self._identity.update(decoder="batched", decoder_version=DECODER_VERSION,
                                      batch_rows=self.batch_rows, sampler=SAMPLER)
        return deepcopy(self._identity)

    def warmup(self) -> None:
        self._load()

    def close(self) -> None:
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
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
        with self._load_lock:
            return self._load_locked()

    def _load_locked(self):
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
                codec = load_codec_model(resolve_codec_path(self.checkpoint), tower, precision)
                if self.batch_rows > 1:
                    from .drama3_batch import BatchedDrama3Backend
                    from .drama3_batch_scheduler import Drama3BatchScheduler
                    with torch.inference_mode():
                        backend = BatchedDrama3Backend(model, rows=self.batch_rows)
                    self._scheduler = Drama3BatchScheduler(backend)
                self._codec = codec
                self._model = model
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"Could not load Fish Speech Drama 3: {exc}") from exc
        return self._model, self._codec

    def _reference_codes(self, voice, codec):
        with self._codec_lock:
            return self._reference_codes_locked(voice, codec)

    def _reference_codes_locked(self, voice, codec):
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

    def _silence_watch(self, text, codec):
        """Ends a take whose audio has gone silent (``silence_stop``)."""
        def decode(frames):
            import torch
            with self._codec_lock, torch.inference_mode():
                codes = (torch.stack(list(frames)) if torch.is_tensor(frames[0])
                         else torch.tensor(frames, dtype=torch.long))
                codes = codes.to(codec.device).T.contiguous()
                return codec.from_indices(codes[None])[0].squeeze().float().cpu().numpy()
        return SilenceWatch(text, decode, sample_rate=codec.sample_rate)

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
                Drama3DecodeGraphs, Drama3PromptTooLongError, build_drama3_prompt)
            from .fish_drama3_decode import generate_drama3_watched
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
                max_new_tokens = frame_budget(text, min(self.max_new_tokens, capacity - len(prompt)))
                sampling = current_sampling() or {}
                temperature = sampling.get("temperature", self.temperature)
                top_k = sampling.get("top_k", self.top_k)
                top_p = sampling.get("top_p", self.top_p)
                if self._scheduler is not None:
                    # Shared decode loop; this request's own prompt, seed and
                    # sampling, and its own frames back.
                    future = self._scheduler.submit(
                        prompt, max_new_tokens, temperature=temperature, top_k=top_k,
                        top_p=top_p, seed=seed, watch=self._silence_watch(text, codec))
                    rows, finish_reason = future.result()
                    batch = getattr(future, "batch_stats", None)
                    frames = torch.tensor(rows, dtype=torch.long).reshape(len(rows), -1)
                else:
                    # Fish's set_seed clamps above 2**31; fold uint32 attempt seeds.
                    set_seed(seed % (1 << 31))
                    with torch.inference_mode():
                        frames, finish_reason = generate_drama3_watched(
                            model, prompt, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_k=top_k, top_p=top_p,
                            watch=self._silence_watch(text, codec))
                if finish_reason != "eos" or not len(frames):
                    raise EngineError(f"Drama 3 did not finish the text: {finish_reason!r}")
                with self._codec_lock, torch.inference_mode():
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
                with tempfile.NamedTemporaryFile(prefix=".take-", suffix=".wav", dir=output_path.parent,
                                                 delete=False) as handle:
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
        if self._scheduler is not None and isinstance(batch, dict):
            metadata["batch"] = dict(batch)  # payload-free batching facts for serve's log
        return metadata, generated.cpu().tolist(), model.config.audio_decoder.vocab_size
