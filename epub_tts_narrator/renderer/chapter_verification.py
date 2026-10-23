"""Server verification with local MLX or CPU/CUDA CTranslate2 Whisper weights."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from threading import RLock

from .common import file_hash
from .verification import assemble_verifier


class FasterWhisperVerifier:
    """Lazy, audio-only recognition; never downloads models implicitly.

    A single model is shared by paragraph workers. The lock includes consuming
    the segment generator, since faster-whisper does its inference lazily.
    """

    def __init__(self, model_path, language="en", *, device="cpu", compute_type="int8",
                 cpu_threads=4):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError("faster-whisper requires a local model directory")
        required = [self.model_path / name for name in ("config.json", "model.bin", "tokenizer.json")]
        if not all(path.is_file() for path in required):
            raise ValueError("faster-whisper model needs config.json, model.bin, and tokenizer.json")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("ASR language must be a nonempty language code")
        if device not in {"cpu", "cuda"}:
            raise ValueError("faster-whisper device must be cpu or cuda")
        if compute_type not in {"default", "auto", "int8", "int8_float16", "int8_float32",
                                "int8_bfloat16", "int16", "float16", "float32", "bfloat16"}:
            raise ValueError("Unsupported faster-whisper compute_type")
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 128:
            raise ValueError("faster-whisper cpu_threads must be from 1 through 128")
        optional = self.model_path / "preprocessor_config.json"
        if optional.is_file():
            required.append(optional)
        self._model_file_state = {
            path: (path.stat().st_size, path.stat().st_mtime_ns) for path in required
        }
        self.language, self.device = language, device
        self.compute_type, self.cpu_threads = compute_type, cpu_threads
        self._model, self._closed, self._lock = None, False, RLock()
        packages = {}
        for name in ("faster-whisper", "ctranslate2", "tokenizers", "av"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        self.identity = {
            "backend": "faster-whisper", "version": 1, "packages": packages,
            "model_sha256": {path.name: file_hash(path) for path in required},
            "language": language, "device": device, "compute_type": compute_type,
            "options": self._options(True), "reference_options": self._options(False),
        }

    @staticmethod
    def _options(timed):
        return {"task": "transcribe", "beam_size": 5, "temperature": 0.0,
                "condition_on_previous_text": False, "initial_prompt": None,
                "word_timestamps": timed, "vad_filter": False}

    def _transcribe(self, path, timed):
        with self._lock:
            if self._closed:
                raise RuntimeError("The faster-whisper verifier is closed")
            for filename, state in self._model_file_state.items():
                current = filename.stat()
                if (current.st_size, current.st_mtime_ns) != state:
                    raise RuntimeError("ASR model changed during this run; recreate the verifier")
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError("Install faster-whisper for server speech verification") from exc
                self._model = WhisperModel(
                    str(self.model_path), device=self.device, compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads, num_workers=1, local_files_only=True)
            segments, info = self._model.transcribe(
                str(path), language=self.language, **self._options(timed))
            result = []
            for segment in segments:
                item = {name: getattr(segment, name) for name in
                        ("text", "start", "end", "avg_logprob", "no_speech_prob", "compression_ratio")}
                if timed:
                    item["words"] = [
                        {name: getattr(word, name) for name in ("word", "start", "end", "probability")}
                        for word in segment.words or []]
                result.append(item)
            return {"text": "".join(segment["text"] for segment in result).strip(),
                    "language": info.language, "segments": result}

    def transcribe(self, path):
        return self._transcribe(path, True)

    def transcribe_reference(self, path):
        return self._transcribe(path, False)

    def close(self):
        with self._lock:
            self._closed = True
            self._model = None


def create_chapter_verifier(settings, jev_settings=None, provider_settings=None):
    """Retain the same secondary recognition and JEV policy on a Linux worker."""
    from .phonetic_match import require_phonetic_matcher
    require_phonetic_matcher()
    if settings.get("type", "mlx_whisper") == "mlx_whisper":
        from .verification import create_verifier
        return create_verifier(settings, jev_settings, provider_settings)
    if settings.get("type") != "faster_whisper":
        raise ValueError("Unsupported chapter verifier type")

    def create(model):
        return FasterWhisperVerifier(
            model, language=settings.get("language", "en"),
            device=settings.get("device", "cpu"), compute_type=settings.get("compute_type", "int8"),
            cpu_threads=settings.get("cpu_threads", 4))

    return assemble_verifier(create(settings["model"]), settings, jev_settings, provider_settings,
                             lambda: create(settings["reference_model"]))
