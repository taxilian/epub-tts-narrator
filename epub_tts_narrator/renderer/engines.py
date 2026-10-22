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
from typing import Any, Protocol

from .drama3_design import (
    DESIGN_TEMPLATE_SHA256, DESIGN_TEMPLATE_VERSION,
    generate_scene_voice, validate_scene_fields,
)
from .speech_text import speech_text_parts, trim_speech_text


DRAMA3_RUNTIME_VERSION = "0.1.1+epub.1"
DRAMA3_RUNTIME_REVISION = "3b76bf84714d3766d7872f3409639a438e296cc8"
DRAMA3_MODEL_REVISION = "b2b5ee998034b2975720906c815e7fb0e9f007bd"
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
        seed: int, direction: str = "",
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


def _directed_text(text: str, direction: str) -> tuple[str, dict[str, Any]]:
    """Apply only supported controls to already validated literal book text."""
    if not isinstance(direction, str):
        raise EngineError("direction must be text")
    direction = direction.strip()
    if len(direction) > 240 or any(char in direction for char in "[]<>\r\n"):
        raise EngineError("direction must be a single plain-text phrase of at most 240 characters")
    if any(ord(char) < 32 for char in direction):
        raise EngineError("direction contains control characters")
    control = {
        "whisper": "whisper", "whispering": "whisper", "whispered": "whisper",
        "nervous": "nervous", "sigh": "sighs", "sighs": "sighs",
        "laugh": "laughs", "laughing": "laughs", "laughs": "laughs",
    }.get(direction.casefold())
    if control == "whisper":
        engine_text = f"<whisper>{text}</whisper>"
    elif control:
        engine_text = f"[{control}] {text}"
    else:
        engine_text = text
    return engine_text, {
        "direction": direction,
        "direction_applied": control is not None,
        "applied_control": control,
        "direction_note": (
            "Unsupported free-form direction was not applied"
            if direction and control is None else ""
        ),
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


class Drama3MLXEngine:
    """One lazy model instance, with content-addressed reference conditioning.

    A described, single-character scene creates a reference voice. Subsequent
    audiobook synthesis clones the saved reference through the ordinary runtime
    TTS path; it does not repeatedly redesign the character.
    """

    def __init__(
        self, checkpoint: str | Path, *, experts: str = "q8",
        temperature: float = 1.0, top_p: float = 0.9, top_k: int = 30,
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

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "voice_design": True,
            "reference_voice": True,
            "preset_voice": False,
            "delivery_control": True,
            "multi_speaker": True,
            "unconditioned_voice_creation": False,
        }

    @property
    def voice_binding_identity(self) -> dict[str, str]:
        """Reference WAVs remain compatible across supported Drama 3 ports."""
        return {"engine": "drama3_mlx", "model": "fishaudio/drama-3"}

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
        self._reference_cache.clear()
        self._model = None
        self._soundfile = None
        if loaded:
            gc.collect()
            try:
                import mlx.core as mx
            except ImportError:
                return
            mx.clear_cache()

    def warmup(self) -> None:
        """Load the model before a server begins accepting rendering jobs."""
        self._load()

    def _load(self) -> Any:
        if self._model is None:
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
                self._model = Drama3.load(self.checkpoint, experts=self.experts)
            except Exception as exc:
                raise EngineError(f"Could not load local Drama 3: {exc}") from exc
        return self._model

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
                self._reference_cache[key] = model.reference(
                    audio.mean(axis=1), rate, transcript,
                )
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"Could not encode reference audio: {exc}") from exc
        return self._reference_cache[key]

    def _generate(
        self, text: str, output_path: Path, *, seed: int,
        references: list[Any], direction: str = "",
    ) -> dict[str, Any]:
        text = _text(trim_speech_text(_text(text, "text")), "text")
        _seed(seed)
        engine_text, delivery = _directed_text(text, direction)
        return {
            **self._generate_audio(engine_text, output_path, seed=seed, references=references),
            **delivery,
        }

    def _generate_audio(
        self, engine_text: str, output_path: Path, *, seed: int, references: list[Any],
    ) -> dict[str, Any]:
        """Render adapter-built text; public entry points validate source text first."""
        output_path = Path(output_path)
        if output_path.exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        model = self._load()
        try:
            iterator = iter(model.generate(
                engine_text, references=references, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k,
                max_new_tokens=self.max_new_tokens, single_chunk=True,
                seed=seed, attempts=1,
            ))
            segment = next(iterator, None)
            if segment is None or next(iterator, None) is not None:
                raise EngineError("Drama 3 must return exactly one audio segment")
            metadata = self._save_segment(segment, model, engine_text, output_path)
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Drama 3 generation failed: {exc}") from exc
        return {
            **metadata,
            "seed": seed,
            "synthesis_text": engine_text,
        }

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
                prefix=".take-", suffix=".wav", dir=output_path.parent, delete=False,
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
            segment, prompt_audit = generate_scene_voice(
                model, description, text, seed=seed,
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

    def synthesize(
        self, text: str, voice: dict[str, Any], output_path: Path, *,
        seed: int, direction: str = "",
    ) -> dict[str, Any]:
        text = _text(trim_speech_text(_text(text, "text")), "text")
        _seed(seed)
        return self._generate(
            text, output_path, seed=seed, references=[self._reference(voice)],
            direction=direction,
        )

    def synthesize_dialogue(
        self, turns: list[dict[str, Any]], output_path: Path, *, seed: int,
    ) -> dict[str, Any]:
        """Speak a paragraph's ordered turns with request-local speaker references.

        This uses the installed runtime's ordinary public ``generate`` API and
        one model instance. It is sequential dialogue, not simultaneous voices.
        The planner must split oversized paragraphs; runtime context and output
        limits still apply, and an incomplete take is never published.
        """
        _seed(seed)
        if not isinstance(turns, list) or not turns:
            raise EngineError("dialogue turns must be a nonempty list")
        if Path(output_path).exists():
            raise EngineError(f"Audio output already exists: {output_path}")
        engine_text, voices, metadata = format_dialogue(turns)
        references = []
        for speaker_id, voice in enumerate(voices):
            reference = self._reference(voice)
            references.append(replace(
                reference, text=f"<|speaker:{speaker_id}|> {reference.text}",
            ))
        return {
            **self._generate_audio(engine_text, output_path, seed=seed, references=references),
            **metadata,
        }

    def synthesize_fish(
        self, text: str, voices: list[dict[str, Any]], output_path: Path, *, seed: int,
    ) -> dict[str, Any]:
        """Serve the validated Fish inline-reference rendering subset.

        Only the HTTP adapter uses this entry point. It validates the limited
        control vocabulary before allowing request text into the model.
        """
        from .fish_http import trim_wire_text
        _seed(seed)
        text = trim_wire_text(text, {voice.get("speaker") for voice in voices})
        references = []
        for voice in voices:
            reference = self._reference(voice)
            speaker = voice.get("speaker")
            if speaker is not None:
                reference = replace(reference, text=f"<|speaker:{speaker}|> {reference.text}")
            references.append(reference)
        return self._generate_audio(text, output_path, seed=seed, references=references)


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
        _, delivery = _directed_text(text, turn.get("direction", ""))
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
                and rendered_turns[-1]["speaker_id"] == speaker_id
                and rendered_turns[-1]["applied_control"] == delivery["applied_control"]):
            previous = rendered_turns[-1]
            boundary = previous["trailing_boundary"] + leading
            separate = (any(char.isspace() for char in boundary)
                        or bool(boundary and previous["text"][-1].isalnum() and text[0].isalnum()))
            previous["text"] += (" " if separate else "") + text
            previous["trailing_boundary"] = trailing
        else:
            rendered_turns.append({"speaker_id": speaker_id, "text": text,
                                   "trailing_boundary": trailing, **delivery})
    lines = []
    for turn in rendered_turns:
        # Adjacent literal fragments must not form a control tag when
        # concatenated, even if neither fragment contained a full tag.
        text = _text(turn["text"], "dialogue rendered turn text")
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
    from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
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
