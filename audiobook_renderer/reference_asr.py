"""Independent, local Parakeet transcription for voice-reference checks."""

from __future__ import annotations

import gc
from importlib import metadata
from pathlib import Path

from .common import file_hash


_TOKENIZER_FILES = ("tokenizer.model", "tokenizer.vocab", "vocab.txt")
_OPTIONS = {
    "decoding": "greedy",
    "dtype": "bfloat16",
    "stream": False,
    "chunk_duration": None,
}


class MlxParakeetVerifier:
    """Transcribe reference audio without receiving its intended words.

    The configured checkpoint must already exist locally. Fingerprinting is
    eager, but imports, model loading, and GPU allocation wait until an actual
    transcription. This adapter provides no word-alignment contract and is not
    the verifier for rendered audiobook chunks.
    """

    def __init__(self, model_path):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError(f"local Parakeet model directory does not exist: {self.model_path}")
        self._model = None
        model_files = self._model_files()
        if not (self.model_path / "config.json").is_file() or not list(self.model_path.glob("*.safetensors")):
            raise ValueError("local Parakeet model needs config.json and MLX safetensors weights")
        if any(not path.is_file() for path in model_files):
            raise ValueError("Parakeet model assets must be regular files")
        self._model_file_state = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in model_files
        }
        packages = {}
        for name in ("mlx-audio", "mlx", "numpy"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        self.identity = {
            "backend": "mlx-parakeet",
            "adapter_version": 1,
            "model_path": str(self.model_path),
            "model_sha256": {path.name: file_hash(path) for path in model_files},
            "packages": packages,
            "options": dict(_OPTIONS),
        }

    def _model_files(self):
        files = [self.model_path / "config.json", *sorted(self.model_path.glob("*.safetensors"))]
        files.extend(self.model_path / name for name in _TOKENIZER_FILES
                     if (self.model_path / name).exists())
        return files

    def _check_model_files(self):
        try:
            current = {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                for path in self._model_files()
            }
        except OSError as exc:
            raise RuntimeError("Parakeet model changed during this run; recreate the verifier") from exc
        if current != self._model_file_state:
            raise RuntimeError("Parakeet model changed during this run; recreate the verifier")

    def _load(self):
        if self._model is None:
            try:
                from mlx_audio.stt.utils import load_model
            except ImportError as exc:
                raise RuntimeError("mlx-audio is required for local Parakeet verification") from exc
            # Passing a Path bypasses the library's repository-name resolution.
            # Local files were checked immediately before this call.
            self._model = load_model(self.model_path, strict=True)
        return self._model

    def transcribe_reference(self, path) -> dict:
        self._check_model_files()
        # No text, prompt, vocabulary hint, or expected answer enters the model.
        result = self._load().generate(str(path))
        if not isinstance(getattr(result, "text", None), str):
            raise ValueError("Parakeet result must contain a text string")
        sentences = getattr(result, "sentences", None)
        if not isinstance(sentences, (list, tuple)):
            raise ValueError("Parakeet result must contain sentences")
        segments = []
        for sentence in sentences:
            if not isinstance(getattr(sentence, "text", None), str):
                raise ValueError("Parakeet sentence must contain a text string")
            segment = {"text": sentence.text}
            for name in ("start", "end"):
                value = getattr(sentence, name, None)
                if value is not None:
                    segment[name] = value
            segments.append(segment)
        return {"text": result.text, "segments": segments}

    def close(self):
        loaded = self._model is not None
        self._model = None
        if loaded:
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                # Cleanup should not hide a synthesis/verification error.
                pass
