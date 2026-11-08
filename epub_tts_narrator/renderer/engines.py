"""TTS adapters. Model packages are imported only when speech is requested.

Drama 3 uses the pinned, unofficial Sogni MLX port on Apple Silicon. Its native
``accept`` callback returns the last failed take after exhausting retries, so
this adapter deliberately performs one generation and leaves all acceptance,
retries, and publication to the renderer.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import unicodedata
from typing import Any, Protocol

from ..spoken_text import MAX_DELIVERY_CHARACTERS
from .audio_naming import temporary_audio_prefix
from .drama3_design import (
    DESIGN_TEMPLATE_SHA256, DESIGN_TEMPLATE_VERSION,
    generate_scene_voice, validate_scene_fields,
)
from .drama3_models import DEFAULT_DRAMA3_MODEL, DRAMA3_MODELS, get_drama3_model
from .frame_budget import frame_budget
from .speech_text import speech_text_parts, trim_speech_text
from .performance_compiler import compile_performance, trim_performance


DRAMA3_RUNTIME_VERSION = "0.1.1+epub.1"
DRAMA3_RUNTIME_REVISION = "3b76bf84714d3766d7872f3409639a438e296cc8"
DRAMA3_MODEL_REVISION = DRAMA3_MODELS[DEFAULT_DRAMA3_MODEL].revision
DRAMA3_MAX_SPEAKERS = 16
DRAMA3_INSTALL = "uv pip install --python .venv-renderer/bin/python -r renderer.requirements.txt"


class EngineError(RuntimeError):
    """A provider could not safely produce the requested audio."""


class TTSEngine(Protocol):
    # Engines may additionally advertise multi_speaker=True and implement
    # synthesize_dialogue(turns, output_path, *, seed). It is optional so
    # existing single-voice plugins remain compatible.
    @property
    def identity(self) -> dict[str, Any]: ...

    @property
    def capabilities(self) -> dict[str, bool]: ...

    def create_voice(
        self, description: str, text: str, output_path: Path, *, seed: int
    ) -> dict[str, Any]: ...

    def synthesize(
        self, text: str, voice: dict[str, Any], output_path: Path, *,
        seed: int, direction: str = "", performance=None, traits: str = "",
    ) -> dict[str, Any]: ...


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EngineError(f"{label} must be nonempty text")
    # Book text must not become a model instruction. The planner supplies the
    # bracket-free spoken_text field; it must resolve any remaining syntax.
    if "[" in value or "]" in value or re.search(r"<[^>\n]*>", value):
        raise EngineError(
            f"{label} contains Drama 3 control syntax; provide literal spoken "
            "text without bracket or angle tags"
        )
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise EngineError(f"{label} contains control characters")
    return value


def _seed(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise EngineError("seed must be an integer from 0 through 4294967295")


def _plain_direction(direction: str) -> str:
    """Keep a description inside one non-nested Drama 3 style control."""
    if not isinstance(direction, str):
        raise EngineError("direction must be text")
    direction = direction.strip()
    if len(direction) > MAX_DELIVERY_CHARACTERS or any(char in direction for char in "[]<>\r\n"):
        raise EngineError(
            "direction must be a single plain-text phrase of at most "
            f"{MAX_DELIVERY_CHARACTERS} characters"
        )
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
           for char in direction):
        raise EngineError("direction contains control characters")
    return direction


def _directed_text(text: str, direction: str) -> tuple[str, dict[str, Any]]:
    """Apply a non-spoken style prefix while retaining reference conditioning.

    Drama 3 documents free natural-language ``[style] line`` controls in
    https://arxiv.org/html/2610.18010v1#A3. The older exact controls remain
    byte-identical for compatibility with verified takes. Applying a direction
    records conditioning, not a claim that the model followed it perceptually.
    """
    direction = _plain_direction(direction)
    control = {
        "whisper": "whisper", "whispering": "whisper", "whispered": "whisper",
        "nervous": "nervous", "sigh": "sighs", "sighs": "sighs",
        "laugh": "laughs", "laughing": "laughs", "laughs": "laughs",
    }.get(direction.casefold())
    if control == "whisper":
        engine_text = f"<whisper>{text}</whisper>"
    elif control:
        engine_text = f"[{control}] {text}"
    elif direction:
        control = "style_prefix"
        engine_text = f"[{direction}] {text}"
    else:
        engine_text = text
    return engine_text, {
        "direction": direction,
        "direction_applied": control is not None,
        "applied_control": control,
        "direction_note": "",
    }


def prepare_synthesis(text, direction="", *, performance=None, traits=""):
    """Compile the same exact audible inputs for rendering and cache identity."""
    text = _text(text, "text")
    if performance is None:
        if traits:
            raise EngineError("Separate habitual traits require a structured performance record")
        prepared = _text(trim_speech_text(text), "text")
        directed, metadata = _directed_text(prepared, direction)
        return prepared, directed, metadata
    try:
        prepared, record = trim_performance(text, performance)
        compiled = compile_performance(prepared, record, traits=traits, direction=direction)
    except ValueError as exc:
        raise EngineError(str(exc)) from exc
    return prepared, compiled["text"], {
        "direction": compiled["direction"], "direction_applied": bool(direction or traits or record["controls"]),
        "applied_control": "structured_performance", "direction_note": "",
        "expressive_performance": compiled,
    }


def _runtime_details() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("drama3-mlx")
    except importlib.metadata.PackageNotFoundError:
        return {"installed_version": None, "installed_revision": None}
    revision = None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            revision = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except (ValueError, AttributeError):
            pass
    return {
        "installed_version": distribution.version,
        "installed_revision": revision,
    }


def _checkpoint_fingerprints(checkpoint: Path, *, extra_names=(), write_cache=True) -> tuple[dict[str, str], str | None]:
    """Fingerprint actual model content, reusing hashes only for unchanged files.

    The cache is an optimization for repeated CLI invocations, not an assertion
    that arbitrary local files are the official release. Hugging Face download
    metadata supplies a revision when it is available for every model file.
    """
    cache_path = checkpoint / ".renderer-checkpoint-hashes-v1.json"
    try:
        cached = json.loads(cache_path.read_text())
        entries = cached.get("files", {}) if cached.get("version") == 1 else {}
    except (OSError, ValueError, AttributeError):
        entries = {}
    if not isinstance(entries, dict):
        entries = {}
    paths = {p.name: p for p in checkpoint.glob("*.safetensors") if p.is_file()}
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json",
                 "tokenizer_config.json", "special_tokens_map.json", "codec.pth", *extra_names):
        path = checkpoint / name
        if path.is_file():
            paths[name] = path
    fingerprints: dict[str, str] = {}
    current: dict[str, Any] = {}
    revisions: list[str | None] = []
    for name, path in sorted(paths.items()):
        before = path.stat()
        signature = {
            "size": before.st_size, "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns, "inode": before.st_ino,
        }
        entry = entries.get(name, {})
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or any(entry.get(key) != value for key, value in signature.items())):
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino) != (
                before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino
            ):
                raise EngineError(f"Model file changed during fingerprinting: {name}")
            digest = hasher.hexdigest()
        fingerprints[name] = digest
        current[name] = {**signature, "sha256": digest}
        metadata = checkpoint / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        try:
            revision = metadata.read_text().splitlines()[0]
        except (OSError, IndexError):
            revision = None
        revisions.append(revision if isinstance(revision, str)
                         and re.fullmatch(r"[a-f0-9]{40}", revision) else None)
    if write_cache and current != entries:
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=".checkpoint-hashes-", suffix=".json",
                dir=checkpoint, delete=False, encoding="utf-8",
            ) as stream:
                temp_name = stream.name
                json.dump({"version": 1, "files": current}, stream, sort_keys=True)
                stream.write("\n")
            os.replace(temp_name, cache_path)
        except OSError:
            # Read-only model directories work too; they will be hashed again
            # in the next process instead of weakening the content identity.
            pass
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
    revision = revisions[0] if revisions and len(set(revisions)) == 1 else None
    return fingerprints, revision


class HistorySynthesis:
    """Shared request composition for every engine that carries chapter history.

    Requests are always speaker-tagged. ``continuity.compose`` adds the
    chapter's history turns and their voices; subclasses implement only
    ``_generate_wire``, which local runtimes run directly and HTTP clients
    send to a worker's ``synthesize_fish``.

    ``history_layouts`` names the prompt layouts ``_generate_wire`` builds:
    ``directing`` (Fish's history conversation, always) and optionally
    ``streaming`` (Drama 3's streaming continuation, used only by a
    history-echo retry; see ``history_echo``).
    """

    history_layouts = frozenset({"directing"})

    def synthesize(self, text, voice, output_path, *, seed, direction="", performance=None, traits=""):
        # Validate the full request before encoding a voice/loading a model.
        _, directed, delivery = prepare_synthesis(text, direction, performance=performance, traits=traits)
        _seed(seed)
        metadata = self._compose_and_generate(f"<|speaker:0|> {directed}", [voice], output_path, seed=seed)
        return {**metadata, "synthesis_text": directed, **delivery}

    def synthesize_dialogue(self, turns, output_path, *, seed):
        """Speak a paragraph's ordered turns with request-local speaker references.

        It is sequential dialogue, not simultaneous voices. The planner must
        split oversized paragraphs; an incomplete take is never published.
        """
        _seed(seed)
        if not isinstance(turns, list) or not turns:
            raise EngineError("dialogue turns must be a nonempty list")
        if Path(output_path).exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        engine_text, voices, metadata = format_dialogue(turns)
        return {**self._compose_and_generate(engine_text, voices, output_path, seed=seed), **metadata}

    def _compose_and_generate(self, text, voices, output_path, *, seed):
        from .continuity import compose, current, generated_codes, recovery_mode
        from .history_echo import VERSION, apply_recovery
        references, history = compose(text, voices)
        scope = current()
        requested = recovery_mode() if scope is not None else None
        applied, text, history, layout = apply_recovery(
            requested, text, history, self.history_layouts, kind=scope.get("kind") if scope else None)
        options = {"layout": layout} if layout != "directing" else {}
        metadata, codes, codebook_size = self._generate_wire(text, references, history, output_path,
                                                             seed=seed, **options)
        if requested is not None:
            metadata = {**metadata, "history_recovery": {"version": VERSION, "requested": requested,
                                                         "applied": applied}}
        if scope is None:
            # Casting and preview takes are never chapter history.
            return metadata
        return {**metadata, **generated_codes(codes, codebook_size, text, voices)}

    def synthesize_fish(self, text, voices, output_path, *, seed, history=(), layout="directing"):
        """Serve one composed wire request; the client owns chapter history."""
        from .fish_http import trim_wire_text
        _seed(seed)
        if layout not in self.history_layouts:
            raise EngineError(f"This engine does not build the {layout!r} history layout")
        text = trim_wire_text(text, {voice.get("speaker") for voice in voices})
        if len(voices) == 1 and voices[0].get("speaker") is None:
            voices = [dict(voices[0], speaker=0)]
            if not text.lstrip().startswith("<|speaker:"):
                text = f"<|speaker:0|> {text}"
        options = {"layout": layout} if layout != "directing" else {}
        metadata, codes, codebook_size = self._generate_wire(text, voices, list(history), output_path,
                                                             seed=seed, **options)
        return {**metadata, "codes": codes, "codebook_size": codebook_size}


class Drama3MLXEngine(HistorySynthesis):
    """One lazy model instance, with content-addressed reference conditioning.

    A described, single-character scene creates a reference voice. Subsequent
    audiobook synthesis clones the saved reference through the ordinary runtime
    TTS path; it does not repeatedly redesign the character.
    """

    def __init__(
        self, checkpoint: str | Path, *, experts: str = "q8",
        temperature: float = 0.7, top_p: float = 0.7, top_k: int = 30,
        max_new_tokens: int = 2048, min_audio_seconds: float = 0.1,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_dir():
            raise EngineError(f"Drama 3 checkpoint directory does not exist: {self.checkpoint}")
        if experts not in {"q8", "q4", "fp8", "bf16"}:
            raise EngineError("Drama 3 experts must be q8, q4, fp8, or bf16")
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
        self.experts = experts
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.min_audio_seconds = float(min_audio_seconds)
        self._model: Any = None
        self._soundfile: Any = None
        self._reference_cache: dict[tuple[str, str], Any] = {}
        self._identity: dict[str, Any] | None = None
        # Batched decoding (set_batch_rows): every MLX operation runs on one
        # private runtime thread, because MLX streams belong to their thread.
        self.batch_rows = 1
        self._scheduler: Any = None
        self._mlx_runtime: Any = None
        self.batch_audio_decode_seconds = 0.0
        # This thread's latest batched take's scheduler facts, for its metadata.
        self._batch_stats = threading.local()
        self._load_lock = threading.RLock()

    def set_batch_rows(self, rows: int) -> None:
        """Decode up to ``rows`` concurrent requests per step; set before loading."""
        from .mlx_drama3_batch import MAX_BATCH_ROWS
        if type(rows) is not int or not 1 <= rows <= MAX_BATCH_ROWS:
            raise EngineError(f"batch_rows must be an integer from 1 through {MAX_BATCH_ROWS}")
        if self._model is not None:
            raise EngineError("Set batch_rows before the model loads")
        self.batch_rows = rows
        self._identity = None

    @property
    def concurrent_requests(self) -> int:
        return self.batch_rows

    def _mlx(self, function, *args, **kwargs):
        """Run MLX work on the batched engine's own thread (directly when unbatched)."""
        if self.batch_rows == 1:
            return function(*args, **kwargs)
        with self._load_lock:
            if self._mlx_runtime is None:
                from .local_runtime import LocalRuntime
                self._mlx_runtime = LocalRuntime(name="drama3-mlx-batch")
        return self._mlx_runtime.run(function, *args, initialize_mlx=True, **kwargs)

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "voice_design": True,
            "reference_voice": True,
            "preset_voice": False,
            "delivery_control": True,
            "structured_performance": True,
            "multi_speaker": True,
            "chapter_continuity": True,
            "unconditioned_voice_creation": False,
        }

    @property
    def voice_binding_identity(self) -> dict[str, str]:
        """Reference WAVs remain compatible across supported Drama 3 ports."""
        return {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

    def check_continuity(self):
        details = _runtime_details()
        if (details["installed_version"] != DRAMA3_RUNTIME_VERSION
                or details["installed_revision"] not in (None, DRAMA3_RUNTIME_REVISION)):
            raise EngineError("Chapter continuity requires the pinned vendored Drama 3 runtime")

    @property
    def identity(self) -> dict[str, Any]:
        return self.get_identity()

    def get_identity(self, *, cache_fingerprints=True) -> dict[str, Any]:
        """Read synthesis identity without writing model metadata during audits."""
        if self._identity is not None:
            return deepcopy(self._identity)
        file_fingerprints, model_revision = _checkpoint_fingerprints(
            self.checkpoint, write_cache=cache_fingerprints)
        packages = {}
        for name in ("mlx", "mlx-metal", "mlx-lm", "mlx-audio", "numpy", "soundfile", "torch", "tokenizers"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self._identity = {
            "engine": "drama3_mlx",
            "adapter_version": 1,
            "packages": packages,
            "model": "fishaudio/drama-3",
            "runtime": "Sogni-AI/drama3-mlx",
            "required_runtime_version": DRAMA3_RUNTIME_VERSION,
            "required_runtime_revision": DRAMA3_RUNTIME_REVISION,
            "recommended_model_revision": DRAMA3_MODEL_REVISION,
            "model_revision": model_revision,
            **_runtime_details(),
            "checkpoint": str(self.checkpoint),
            "checkpoint_files_sha256": file_fingerprints,
            "experts": self.experts,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "min_audio_seconds": self.min_audio_seconds,
            "single_chunk": True,
            "output_encoding": "wav-float32-mono",
        }
        if self.batch_rows > 1:
            # Per-row counter-hash noise samples differently from the vendored
            # global random stream, so batched takes are never cache hits for it.
            from .mlx_drama3_batch import DECODER_VERSION, SAMPLER
            self._identity.update(decoder="batched", decoder_version=DECODER_VERSION,
                                  batch_rows=self.batch_rows, sampler=SAMPLER)
        return deepcopy(self._identity)

    @property
    def voice_design_identity(self) -> dict[str, Any]:
        """Version design requests without invalidating reference-clone caches."""
        return {
            **self.identity,
            "voice_design_template_version": DESIGN_TEMPLATE_VERSION,
            "voice_design_template_sha256": DESIGN_TEMPLATE_SHA256,
            "voice_design_script_lines": 1,
        }

    def close(self) -> None:
        """Release this adapter's model and reference cache between stages."""
        loaded = self._model is not None
        if self._scheduler is not None:
            self._scheduler.close()
            self._scheduler = None
        self._reference_cache.clear()
        self._model = None
        self._soundfile = None
        if loaded:
            gc.collect()
            try:
                import mlx.core as mx
            except ImportError:
                return
            self._mlx(mx.clear_cache)
        if self._mlx_runtime is not None:
            self._mlx_runtime.close()
            self._mlx_runtime = None

    def warmup(self) -> None:
        """Load the model before a server begins accepting rendering jobs."""
        self._load()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                model = self._mlx(self._load_model)
                if self.batch_rows > 1:
                    from .drama3_batch_scheduler import Drama3BatchScheduler
                    from .mlx_drama3_batch import BatchedMLXBackend
                    backend = BatchedMLXBackend(model.model, rows=self.batch_rows, run=self._mlx)
                    self._scheduler = Drama3BatchScheduler(backend)
                self._model = model
        return self._model

    def _load_model(self) -> Any:
        """Load the pinned runtime's model (on the MLX thread when batched)."""
        details = _runtime_details()
        if details["installed_version"] != DRAMA3_RUNTIME_VERSION:
            raise EngineError(
                "The pinned Drama 3 MLX runtime is required. From the project "
                f"directory, run {DRAMA3_INSTALL}"
            )
        revision = details["installed_revision"]
        if revision is not None and revision != DRAMA3_RUNTIME_REVISION:
            raise EngineError(
                f"Drama 3 runtime revision differs from the pinned revision: {revision}"
            )
        try:
            import soundfile
            from drama3_mlx import Drama3
            self._soundfile = soundfile
            return Drama3.load(self.checkpoint, experts=self.experts)
        except Exception as exc:
            raise EngineError(f"Could not load local Drama 3: {exc}") from exc

    def _reference(self, voice: dict[str, Any]) -> Any:
        if voice.get("kind") != "reference":
            raise EngineError("Drama 3 requires a saved reference voice")
        value = voice.get("reference_audio")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise EngineError("reference_audio must be an absolute local file path")
        path = Path(value)
        transcript = _text(voice.get("reference_text"), "reference_text")
        try:
            if path.stat().st_size > 100 * 1024 * 1024:
                raise EngineError("Reference audio is larger than 100 MiB")
            blob = path.read_bytes()
        except OSError as exc:
            raise EngineError(f"Could not read reference audio: {path}") from exc
        checksum = hashlib.sha256(blob).hexdigest()
        # VoiceStore binds immutable references with reference_sha256. Keep the
        # former adapter field as a fallback for externally supplied bindings.
        declared_checksum = voice.get("reference_sha256", voice.get("reference_audio_sha256"))
        if declared_checksum is not None and declared_checksum != checksum:
            raise EngineError("Saved reference audio checksum does not match")
        key = (checksum, transcript)
        model = self._load()
        if key not in self._reference_cache:
            try:
                audio, rate = self._soundfile.read(
                    io.BytesIO(blob), dtype="float32", always_2d=True,
                )
                if audio.shape[0] == 0:
                    raise EngineError("Reference audio is empty")
                self._reference_cache[key] = self._mlx(
                    model.reference, audio.mean(axis=1), rate, transcript,
                )
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"Could not encode reference audio: {exc}") from exc
        return self._reference_cache[key]

    history_layouts = frozenset({"directing", "streaming"})

    def _streaming_segment(self, model: Any, text: str, voice: dict[str, Any], kept: list, seed: int) -> Any:
        """One take in the streaming-continuation layout (see ``drama3_layouts``)."""
        import mlx.core as mx
        from drama3_mlx.model import Segment
        from .drama3_layouts import streaming_rows
        from .mlx_drama3_decode import generate_frames_watched, silence_watch
        rows = streaming_rows(text, self._reference(voice), kept, model.tokenizer, model.model.config)
        if self._scheduler is not None:
            return self._batched_segment(model, text, rows, seed, frame_budget(text, self.max_new_tokens or 2048))
        mx.random.seed(seed)
        frames, finish_reason = generate_frames_watched(
            model.model, rows, max_new_tokens=frame_budget(text, self.max_new_tokens or 2048),
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            watch=silence_watch(model, text))
        if not len(frames):
            raise EngineError("Drama 3 returned no audio frames")
        codes = frames[:, 1:].T.copy()
        return Segment(text=text, codes=codes, audio=model.codec.decode(codes), finish_reason=finish_reason,
                       prompt_rows=len(rows), seconds=0.0)

    def _directing_segment(self, model: Any, text: str, references: list, previous: list, seed: int) -> Any:
        """One take in Fish's history layout: the vendored ``Drama3.generate(single_chunk=True)``
        on our watched copy of its loop (``mlx_drama3_decode``), ended early once silent."""
        from .mlx_drama3_decode import generate_segment, silence_watch
        if self._scheduler is not None:
            from drama3_mlx.prompt import build_prompt
            from .mlx_drama3_batch import prefix_ends
            texts, codes = [reference.text for reference in references], [reference.codes for reference in references]
            prompt = build_prompt(text, model.tokenizer, model.model.config, texts, codes, list(previous))
            # The references, then the pinned turn, repeat for this voice's later takes.
            hints = prefix_ends(model.tokenizer, model.model.config, text, texts, codes, list(previous))
            return self._batched_segment(model, text, prompt, seed, frame_budget(text, self.max_new_tokens),
                                         prefix_hints=hints)
        return generate_segment(
            model, text, references=references, previous_takes=previous,
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            max_new_tokens=frame_budget(text, self.max_new_tokens), seed=seed,
            watch=silence_watch(model, text))

    def _batched_segment(self, model: Any, text: str, prompt: Any, seed: int, max_new_tokens: int,
                         prefix_hints=()) -> Any:
        """One take decoded beside other requests (``mlx_drama3_batch``); None without frames.

        This request keeps its own prompt, seed and sampling (per-request
        sampling from ``serve`` wins over the engine's), and is watched for
        silence like an unbatched take.
        """
        import numpy as np
        from drama3_mlx.model import Segment
        from .request_sampling import current_cancellation, current_sampling
        from .silence_stop import SilenceWatch
        sampling = current_sampling() or {}

        def decode(frames):
            return self._mlx(model.codec.decode, np.array(frames, dtype=np.int64).T)

        watch = SilenceWatch(text, decode, sample_rate=model.sample_rate)
        future = self._scheduler.submit(
            prompt, max_new_tokens, temperature=sampling.get("temperature", self.temperature),
            top_k=sampling.get("top_k", self.top_k), top_p=sampling.get("top_p", self.top_p),
            seed=seed, watch=watch, prefix_hints=prefix_hints, cancel=current_cancellation())
        frames, finish_reason = future.result()
        stats = getattr(future, "batch_stats", None)
        self._batch_stats.value = dict(stats) if isinstance(stats, dict) else None
        if not frames:
            return None
        codes = np.array(frames, dtype=np.int64)[:, 1:].T.copy()
        began = time.perf_counter()
        audio = self._mlx(model.codec.decode, codes)
        # Shares the MLX thread with decode steps, so it delays every row.
        self.batch_audio_decode_seconds += time.perf_counter() - began
        return Segment(text=text, codes=codes, audio=audio,
                       finish_reason=finish_reason, prompt_rows=len(prompt), seconds=0.0)

    def _generate_wire(
        self, text: str, references: list[dict[str, Any]], history: list, output_path: Path,
        *, seed: int, layout: str = "directing",
    ) -> tuple[dict[str, Any], list[list[int]], int]:
        """Generate tagged ``text`` after ``history`` turns with tagged references.

        Prompt overflow follows Fish's fallback order (drop the latest turn,
        then the pinned one). The metadata records how many turns were used.
        ``layout="streaming"`` (a history-echo retry of one voice) sends the
        same turns in Drama 3's streaming-continuation layout instead.
        """
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        if layout not in self.history_layouts:
            raise EngineError(f"Unknown Drama 3 history layout: {layout!r}")
        model = self._load()
        from .continuity import candidates
        from .continuity_codes import history_digest
        encoded = {}
        overflow = None
        try:
            import numpy as np
            for selected, kept in candidates(text, references, history):
                if layout == "streaming":
                    if len(selected) != 1:
                        raise EngineError("The streaming history layout serves exactly one voice")
                    try:
                        segment = self._streaming_segment(model, text, selected[0], kept, seed)
                    except Exception as exc:
                        if type(exc).__name__ != "PromptTooLongError":
                            raise
                        overflow = exc
                        continue
                else:
                    prompt_references = []
                    for voice in selected:
                        key = voice["speaker"]
                        if key not in encoded:
                            reference = self._reference(voice)
                            encoded[key] = replace(reference, text=f"<|speaker:{key}|> {reference.text}")
                        prompt_references.append(encoded[key])
                    previous = [(turn_text, np.array(codes, dtype=np.int64)) for turn_text, codes in kept]
                    try:
                        segment = self._directing_segment(model, text, prompt_references, previous, seed)
                    except Exception as exc:
                        # The pinned runtime validates the full prompt before sampling.
                        if type(exc).__name__ != "PromptTooLongError":
                            raise
                        overflow = exc
                        continue
                    if segment is None:
                        raise EngineError("Drama 3 must return exactly one audio segment")
                codes = segment.codes
                if (getattr(codes, "ndim", None) != 2 or getattr(codes.dtype, "kind", None) not in ("i", "u")
                        or codes.shape[0] != model.model.config.audio_decoder.num_codebooks):
                    raise EngineError("Drama 3 returned invalid generated codec tokens")
                metadata = self._save_segment(segment, model, text, output_path)
                metadata.update(seed=seed, synthesis_text=text, history_turns=len(kept),
                                history_available=len(history), history_sha256=history_digest(kept))
                batch = getattr(self._batch_stats, "value", None) if self._scheduler is not None else None
                if batch:
                    # Payload-free: rows sharing its steps, waits, prompt and cache sizes.
                    metadata["batch"] = batch
                    self._batch_stats.value = None
                if layout != "directing":
                    metadata["history_layout"] = layout
                return metadata, codes.tolist(), model.model.config.audio_decoder.vocab_size
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Drama 3 generation failed: {exc}") from exc
        raise EngineError(f"Drama 3 prompt is too long even without chapter history: {overflow}")

    def _save_segment(self, segment: Any, model: Any, expected_text: str,
                      output_path: Path) -> dict[str, Any]:
        """Shared design/clone failure contract before exclusive publication."""
        if segment.finish_reason != "eos":
            raise EngineError(
                f"Drama 3 did not finish the text: {segment.finish_reason!r}"
            )
        if segment.text != expected_text:
            raise EngineError("Drama 3 returned metadata for different text")
        rate = model.sample_rate
        if type(rate) is not int or rate <= 0:
            raise EngineError("Drama 3 returned an invalid sample rate")
        samples = memoryview(segment.audio)
        if samples.ndim != 1 or samples.format != "f" or not samples.contiguous:
            raise EngineError("Drama 3 returned invalid mono float32 audio")
        if len(samples) < math.ceil(rate * self.min_audio_seconds):
            raise EngineError("Drama 3 returned empty or implausibly short audio")
        if not all(math.isfinite(sample) for sample in samples):
            raise EngineError("Drama 3 returned non-finite audio")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=temporary_audio_prefix(output_path, "drama3-mlx", suffix=".wav"),
                suffix=".wav", dir=output_path.parent, delete=False,
            ) as staged:
                temp_name = staged.name
            self._soundfile.write(temp_name, segment.audio, rate, subtype="FLOAT")
            # Exclusive publication: do not overwrite an accepted asset on a race.
            os.link(temp_name, output_path)
        except Exception as exc:
            raise EngineError(f"Could not save generated audio: {exc}") from exc
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
        return {
            "finish_reason": segment.finish_reason,
            "sample_rate": rate,
            "sample_count": len(samples),
            "duration_seconds": len(samples) / rate,
        }

    def create_voice(
        self, description: str, text: str, output_path: Path, *, seed: int,
    ) -> dict[str, Any]:
        text = _text(text, "Sample text")
        _seed(seed)
        try:
            applied_description, _ = validate_scene_fields(description, text)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        model = self._load()
        try:
            segment, prompt_audit = self._mlx(
                generate_scene_voice, model, description, text, seed=seed,
                temperature=self.temperature, top_p=self.top_p,
                top_k=self.top_k, max_new_tokens=self.max_new_tokens,
            )
            metadata = self._save_segment(segment, model, text, output_path)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Drama 3 voice design failed: {exc}") from exc
        return {
            **metadata,
            "seed": seed,
            "creation_method": "prompt_conditioned_voice_design",
            # This records actual conditioning, not a perceptual quality claim.
            "description_applied": True,
            "requested_description": description,
            "applied_description": applied_description,
            "description_conditioning": "system_scene_character",
            "designer": self.voice_design_identity,
            "design_template_version": DESIGN_TEMPLATE_VERSION,
            "design_template_sha256": DESIGN_TEMPLATE_SHA256,
            "design_prompt": prompt_audit,
        }

def format_dialogue(turns: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Build identical request-local speaker tags for local and HTTP rendering."""
    if not isinstance(turns, list) or not turns:
        raise EngineError("dialogue turns must be a nonempty list")
    voices: list[dict[str, Any]] = []
    speaker_keys: dict[tuple[Any, ...], int] = {}
    speaker_mapping: list[dict[str, Any]] = []
    turn_directions: list[dict[str, Any]] = []
    rendered_turns: list[dict[str, Any]] = []
    # Validate the entire paragraph before loading or encoding any voice.
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or not isinstance(turn.get("voice"), dict):
            raise EngineError(f"dialogue turn {index + 1} must contain a saved voice")
        label = f"dialogue turn {index + 1} text"
        source_text = _text(turn.get("text"), label)
        text, leading, trailing = speech_text_parts(source_text)
        text = _text(text, label)
        _, directed, delivery = prepare_synthesis(source_text, turn.get("direction", ""),
            performance=turn.get("performance"), traits=turn.get("traits", ""))
        voice = turn["voice"]
        if voice.get("kind") != "reference":
            raise EngineError("Drama 3 dialogue requires saved reference voices")
        transcript = _text(voice.get("reference_text"), "reference_text")
        reference_path = voice.get("reference_audio")
        if not isinstance(reference_path, str) or not Path(reference_path).is_absolute():
            raise EngineError("reference_audio must be an absolute local file path")
        asset_hash = voice.get("asset_hash")
        checksum = voice.get("reference_sha256", voice.get("reference_audio_sha256"))
        if any(value is not None and not isinstance(value, str)
               for value in (asset_hash, checksum)):
            raise EngineError("saved voice reference identifiers must be text")
        # Conditioning follows the reference content, not profile or selection
        # metadata. Aliases and revised descriptions of identical references
        # must retain the same speaker tag and natural adjacent-turn joins.
        key = (checksum or reference_path, transcript)
        if key not in speaker_keys:
            if len(voices) >= DRAMA3_MAX_SPEAKERS:
                raise EngineError("Drama 3 dialogue supports at most 16 distinct voices per request")
            speaker_id = speaker_keys[key] = len(voices)
            voices.append(voice)
            speaker_mapping.append({
                "speaker_id": speaker_id,
                "voice_id": voice.get("voice_id"),
                "asset_hash": asset_hash,
            })
        speaker_id = speaker_keys[key]
        turn_directions.append({"speaker_id": speaker_id, **delivery})
        # Segment boundaries often separate narration from quotations by
        # the same character. Preserve a single natural performance unless
        # the actual voice or supported delivery control changes. Retain word
        # boundaries removed by edge trimming, while actual adjoining fragments
        # ("look" + "ed") remain one word.
        if (rendered_turns
                and "expressive_performance" not in delivery
                and "expressive_performance" not in rendered_turns[-1]
                and rendered_turns[-1]["speaker_id"] == speaker_id
                and rendered_turns[-1]["applied_control"] == delivery["applied_control"]
                and (delivery["applied_control"] != "style_prefix"
                     or rendered_turns[-1]["direction"] == delivery["direction"])):
            previous = rendered_turns[-1]
            boundary = previous["trailing_boundary"] + leading
            separate = (any(char.isspace() for char in boundary)
                        or bool(boundary and previous["text"][-1].isalnum() and text[0].isalnum()))
            previous["text"] += (" " if separate else "") + text
            previous["trailing_boundary"] = trailing
        else:
            rendered_turns.append({"speaker_id": speaker_id, "text": text,
                                   "trailing_boundary": trailing, "compiled_text": directed, **delivery})
    lines = []
    for turn in rendered_turns:
        # Adjacent literal fragments must not form a control tag when
        # concatenated, even if neither fragment contained a full tag.
        text = _text(turn["text"], "dialogue rendered turn text")
        if "expressive_performance" in turn:
            directed = turn["compiled_text"]
        else:
            directed, _ = _directed_text(text, turn["direction"])
        lines.append(f"<|speaker:{turn['speaker_id']}|> {directed}")
    return "\n".join(lines), voices, {
        "speaker_mapping": speaker_mapping,
        "speaker_count": len(voices),
        "turn_count": len(turns),
        "rendered_turn_count": len(rendered_turns),
        "turn_directions": turn_directions,
    }


def engine_entry_points(name: str) -> list[importlib.metadata.EntryPoint]:
    """Find a configured plugin, preferring its current namespace by name.

    Preserve duplicates for the caller to reject instead of hiding an ambiguous
    installation behind a legacy registration. A legacy plugin remains usable
    when only unrelated names are registered in the current namespace.
    """
    for group in ("epub_tts_narrator.tts", "audiobook_renderer.tts"):
        matches = list(importlib.metadata.entry_points(group=group).select(name=name))
        if matches:
            return matches
    return []


def create_engine(config: Mapping[str, Any]) -> TTSEngine:
    """Resolve an explicitly configured provider without a silent fallback."""
    name = config.get("type", config.get("name", "drama3_mlx"))
    if name == "fish_speech_cuda":
        # The Runpod inference service: Fish's library on CUDA, run by serve.
        from .fish_speech_engine import FishSpeechCudaEngine
        allowed = {"type", "checkpoint", "temperature", "top_p", "top_k", "max_new_tokens",
                   "min_audio_seconds", "device"}
        settings = {key: value for key, value in config.items() if value is not None and key in allowed}
        unknown = {key for key, value in config.items() if value not in (None, {}, "") and key not in allowed
                   and key not in {"name", "model", "options", "experts"}}
        if unknown:
            raise EngineError(f"Unknown fish_speech_cuda options: {', '.join(sorted(unknown))}")
        if not isinstance(settings.get("checkpoint"), (str, Path)):
            raise EngineError("engine.checkpoint must specify the Drama 3 model directory")
        settings.pop("type")
        return FishSpeechCudaEngine(**settings)
    if name != "drama3_mlx":
        entries = engine_entry_points(name)
        if len(entries) != 1:
            raise EngineError(f"Unknown or ambiguous TTS engine: {name!r}")
        try:
            factory = entries[0].load()
            engine = factory(deepcopy(dict(config.get("options", {}))))
            identity, capabilities = engine.identity, engine.capabilities
            if (not isinstance(identity, dict) or
                    any(not isinstance(identity.get(key), str) or not identity[key] for key in ("engine", "model")) or
                    not isinstance(capabilities, dict) or
                    any(type(capabilities.get(key)) is not bool for key in
                        ("voice_design", "reference_voice", "preset_voice", "delivery_control")) or
                    ("multi_speaker" in capabilities and type(capabilities["multi_speaker"]) is not bool) or
                    (capabilities.get("multi_speaker") and not callable(getattr(engine, "synthesize_dialogue", None))) or
                    not all(callable(getattr(engine, key, None)) for key in ("create_voice", "synthesize"))):
                raise ValueError("Incomplete plugin protocol")
            json.dumps(identity, allow_nan=False)
        except Exception as exc:
            raise EngineError(f"TTS plugin {name!r} could not initialize ({type(exc).__name__})") from exc
        return engine
    allowed = {
        "name", "type", "model", "checkpoint", "experts", "temperature", "top_p", "top_k",
        "max_new_tokens", "min_audio_seconds", "options", "cache_model_alias",
    }
    unknown = set(config) - allowed
    if unknown:
        raise EngineError(f"Unknown Drama 3 options: {', '.join(sorted(unknown))}")
    if config.get("options"):
        raise EngineError("Drama 3 settings belong directly in the engine table")
    if not isinstance(config.get("checkpoint"), (str, Path)):
        raise EngineError("engine.checkpoint must specify a local Drama 3 directory")
    try:
        selected = get_drama3_model(config.get("model", DEFAULT_DRAMA3_MODEL))
    except ValueError as exc:
        raise EngineError(str(exc)) from exc
    options = {key: value for key, value in config.items()
               if key not in {"name", "type", "model", "options", "cache_model_alias"}}
    if selected.runtime == "mlx_audio":
        from .mlx_audio_engine import MLXAudioDrama3Engine
        # Converted checkpoints carry their complete quantization recipe.
        options.pop("experts", None)
        return MLXAudioDrama3Engine(model=selected.repo, **options)
    return Drama3MLXEngine(**options)
