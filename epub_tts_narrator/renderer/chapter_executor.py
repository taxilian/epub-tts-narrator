"""Server-owned chapter synthesis and compressed, independently verified exports.

Synthesis and assembly are separate operations so the service can begin the
next chapter while CPU composition and encoding finish. Assembled chapter PCM
is temporary. Accepted paragraph checkpoints stay private on the worker for
resumption; only explicitly returned compressed artifacts are published.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock, RLock

from .bundle import chunk_speakers
from .chapter_cache import chapter_settings, prepare_chapter
from .chapter_protocol import ARTIFACT_NAMES
from .chapter_records import (
    chapter_record, manifest_record, signed_record, sync_record, verification_record,
)
from .common import digest, file_hash, pcm_sha256, read_json, regular_file, write_json, verify_flac_pcm
from .quality import QUALITY_VERSION
from .render_metrics import METRICS
from .retry_budget import SpeechRetryBudget
from .workflow import _render_chapter_chunks


MAX_PARALLEL_CHAPTERS = 2


def _pcm_data_hash(path):
    return pcm_sha256(path, format_error=ValueError("Chapter master must contain mono 16-bit PCM"))


def _flac_master(wav, target, ffmpeg):
    """Encode and compare decoded sample bytes, independent of WAV headers."""
    expected = _pcm_data_hash(wav)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-protocol_whitelist", "file,pipe"]
    try:
        result = subprocess.run(base + ["-i", str(wav), "-map", "0:a:0", "-vn", "-sn", "-dn",
                                        "-map_metadata", "-1", "-c:a", "flac",
                                        "-compression_level", "8", str(target)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Could not encode or verify the lossless chapter master") from exc
    if result.returncode:
        raise ValueError("FFmpeg could not encode or verify the lossless chapter master")
    try:
        verify_flac_pcm(target, expected, ffmpeg,
                        decode_error=ValueError("FFmpeg could not encode or verify the lossless chapter master"),
                        mismatch_error=ValueError("FLAC chapter master does not preserve the PCM samples"))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Could not encode or verify the lossless chapter master") from exc
    return expected


def _exported_paths(folder, expected_identity):
    """Reuse a complete export only when every artifact is still byte-identical."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Chapter export must be a regular directory")
    manifest = read_json(regular_file(folder / "manifest.json"))
    if (not isinstance(manifest, dict) or manifest.get("identity") != expected_identity
            or manifest.get("record_sha256") != signed_record(manifest)):
        raise ValueError("Chapter artifact manifest failed integrity checks")
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or not {"chapter.json", "sync.json", "verification.json"} <= records.keys():
        raise ValueError("Chapter artifacts are incomplete")
    result = {"manifest.json": folder / "manifest.json"}
    for name, record in records.items():
        if name not in ARTIFACT_NAMES or name == "manifest.json":
            raise ValueError("Unexpected chapter artifact")
        path = regular_file(folder / name)
        if not isinstance(record, dict) or record.get("sha256") != file_hash(path) or record.get("bytes") != path.stat().st_size:
            raise ValueError("Chapter artifact failed integrity checks")
        result[name] = path
    return result


class ChapterExecutor:
    """Reuse existing paragraph quality gates without creating or recasting voices."""

    def __init__(self, config, *, engine=None, verifier=None):
        self.config = deepcopy(config)
        self._engine, self._verifier = engine, verifier
        self._wrapped = False
        self._closed = False
        self._initialization = RLock()
        self._take_locks, self._take_locks_guard = {}, Lock()

    def _components(self):
        with self._initialization:
            if self._closed:
                raise RuntimeError("The chapter executor is closed")
            if self._engine is None:
                from .pool import create_render_engine
                # Concurrent chapter coordinators share one persistent MLX
                # inference thread, including lazy model initialization.
                self._engine = create_render_engine(self.config, shared_inference=True)
            if self._verifier is None:
                from .chapter_verification import create_chapter_verifier
                self._verifier = create_chapter_verifier(
                    self.config["verification"], self.config.get("jev"), self.config.get("provider"))
            if not self._wrapped:
                from .pool import bind_engine_verifier
                self._verifier = bind_engine_verifier(self._engine, self._verifier)
                self._wrapped = True
            return self._engine, self._verifier

    @property
    def parallel_chapters(self):
        from .pool import RenderPool
        engine, _ = self._components()
        # The pool owns global inference slots and model/thread affinity. Other
        # plugins have no concurrency contract and retain serial chapter jobs.
        return MAX_PARALLEL_CHAPTERS if isinstance(engine, RenderPool) else 1

    @property
    def multi_speaker(self):
        engine, _ = self._components()
        return bool(getattr(engine, "capabilities", {}).get("multi_speaker", False))

    @property
    def single_speaker_paragraph(self):
        engine, _ = self._components()
        capabilities = getattr(engine, "capabilities", {})
        return bool(capabilities.get("multi_speaker") or capabilities.get("single_speaker_paragraph"))

    @property
    def identity(self):
        from .chapter_protocol import SAFE_RENDER_KEYS
        engine, verifier = self._components()
        return {"executor_version": 1, "engine": deepcopy(engine.identity),
                "verifier": deepcopy(verifier.identity), "quality_version": QUALITY_VERSION,
                "quality": deepcopy(self.config["quality"]),
                "render_defaults": {key: deepcopy(value) for key, value in self.config["render"].items()
                                    if key in SAFE_RENDER_KEYS}}

    def _settings(self, spec):
        from .chapter_protocol import validate_spec
        validate_spec(spec)
        config = deepcopy(self.config)
        config["render"].update(spec.get("render", {}))
        from .censorship import DEFAULT_POLICY, validate_policy
        # The client owns the edition policy. An absent key must preserve
        # legacy behavior even if the worker's local config enables muting.
        config["censorship"] = validate_policy(spec.get("censorship_policy", DEFAULT_POLICY))
        # Output names are part of the wire contract, independent of local CLI preferences.
        config["render"]["chapter_format"] = "m4a"
        chapter_settings(config["render"], censorship_policy=config["censorship"])
        return config

    def synthesize(self, spec, voices, work, notify=print, retry_budget=None):
        config = self._settings(spec)
        engine, verifier = self._components()
        if retry_budget is None:
            # A restarted service lost its in-memory submission registry; each
            # recovered execution is bounded like a fresh local CLI run.
            retry_budget = SpeechRetryBudget()
        work = Path(work).resolve()
        work.mkdir(parents=True, exist_ok=True)
        chapter = spec["chapter"]
        from .performance_runtime import require_engine_support
        require_engine_support({"chapters": [chapter]}, engine)
        used = {speaker["voice_id"] for chunk in chapter["chunks"] if not chunk["silent_control"]
                for speaker in chunk_speakers(chunk)}
        registered = {}
        for voice_id in sorted(used):
            voice = voices.get(voice_id)
            if (not isinstance(voice, dict) or voice.get("asset_hash") != spec["voice_bindings"].get(voice_id)
                    or not isinstance(voice.get("reference_text"), str) or not voice["reference_text"].strip()):
                raise ValueError(f"Missing or changed registered voice: {voice_id}")
            path = regular_file(voice["reference_audio"]).resolve()
            reference_hash = file_hash(path)
            if voice.get("reference_sha256", reference_hash) != reference_hash:
                raise ValueError(f"Registered voice audio changed: {voice_id}")
            registered[voice_id] = dict(voice, voice_id=voice_id, kind="reference",
                                        reference_audio=str(path), reference_sha256=reference_hash)
        from .paragraphs import paragraph_supported
        if any(not paragraph_supported(getattr(engine, "capabilities", {}), chunk)
               for chunk in chapter["chunks"]):
            raise ValueError("The configured server engine cannot render multi-speaker paragraphs")
        engine_identity, verifier_identity = deepcopy(engine.identity), deepcopy(verifier.identity)
        with METRICS.active("chapter"):
            clips, acceptance = _render_chapter_chunks(
                chapter, registered, engine, engine_identity, verifier, config, work,
                notify=notify, total_chapters=spec["total_chapters"],
                take_locks=self._take_locks, take_locks_guard=self._take_locks_guard,
                retry_budget=retry_budget, paragraph_split_available=False)
        context = {"version": 1, "spec_hash": digest(spec),
                   "source_fingerprint": spec["source_fingerprint"],
                   "total_chapters": spec["total_chapters"],
                   "voice_assets": {key: registered[key]["asset_hash"] for key in sorted(used)},
                   "engine": engine_identity, "verifier": verifier_identity,
                   "quality_version": QUALITY_VERSION, "quality": deepcopy(config["quality"]),
                   "base_seed": config["render"]["seed"], "settings": chapter_settings(
                       config["render"], censorship_policy=config["censorship"])}
        prepared = {"version": 1, "spec_hash": digest(spec), "clips": clips,
                    "acceptance": acceptance, "context": context}
        return dict(prepared, record_sha256=digest(prepared))

    def assemble(self, spec, prepared, work, notify=print):
        """CPU-only stage. This method never calls the engine or verifier."""
        config = self._settings(spec)
        if (not isinstance(prepared, dict) or prepared.get("spec_hash") != digest(spec)
                or prepared.get("record_sha256") != signed_record(prepared)):
            raise ValueError("Prepared chapter does not match the requested job")
        clips, acceptance = prepared["clips"], prepared["acceptance"]
        if (len(clips) != len(spec["chapter"]["chunks"])
                or any(any(clip.get(key) != value for key, value in original.items())
                       for original, clip in zip(spec["chapter"]["chunks"], clips))
                or len(acceptance) != sum(not chunk["silent_control"] for chunk in clips)):
            raise ValueError("Prepared chapter does not cover the complete script")
        expected = {item["chunk_id"]: item for item in acceptance}
        if len(expected) != len(acceptance):
            raise ValueError("Prepared chapter has duplicate verification records")
        for clip in clips:
            if clip.get("wav_path") is None:
                if not clip["silent_control"]:
                    raise ValueError("Spoken passage is missing its verified audio")
                continue
            accepted = expected.get(clip["chunk_id"], {})
            if (accepted.get("report", {}).get("passed") is not True
                    or file_hash(regular_file(clip["wav_path"])) != accepted.get("audio_sha256")):
                raise ValueError("Verified chapter audio changed before assembly")
        work = Path(work).resolve()
        export_identity = {"spec_hash": digest(spec), "context": prepared["context"],
                           "prepared_sha256": prepared["record_sha256"]}
        parent = work / "chapter-exports"
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink():
            raise ValueError("Chapter exports must not use a symlink")
        folder = parent / digest(export_identity)
        if folder.exists() or folder.is_symlink():
            return _exported_paths(folder, export_identity)
        with tempfile.TemporaryDirectory(prefix=".export-", dir=parent) as temporary:
            # Scope the assembled WAV to this invocation. Never invoke global
            # cache cleanup here: another chapter may be reading shared clips.
            composed = prepare_chapter(
                dict(spec["chapter"], _ordinal=spec["chapter"]["number"], _total_chapters=spec["total_chapters"]),
                clips, Path(temporary) / "chapters", book=spec["book"], render_settings=config["render"],
                context=prepared["context"], notify=notify, censorship_policy=config["censorship"])
            stage = Path(temporary) / "ready"
            stage.mkdir()
            pcm_hash = None
            if composed["frames"]:
                notify(f"Compressing lossless chapter master {spec['chapter']['number']} / {spec['total_chapters']}")
                pcm_hash = _flac_master(composed["wav_path"], stage / "chapter.flac", config["render"]["ffmpeg"])
                shutil.copyfile(regular_file(composed["encoded_path"]), stage / "chapter.m4a")
                if file_hash(stage / "chapter.m4a") != composed["encoded_sha256"]:
                    raise ValueError("Encoded chapter changed during export")
            chapter = chapter_record(composed, spec=spec, pcm_data_sha256=pcm_hash)
            write_json(stage / "chapter.json", chapter)
            sync = sync_record(book_id=spec["book"].get("id"), source_fingerprint=spec["source_fingerprint"],
                               spec_hash=digest(spec), chapter_id=chapter["id"], chapter_number=chapter["number"],
                               sample_rate=chapter["sample_rate"], frames=chapter["frames"],
                               precision="paragraph" if any("source_segments" in item for item in chapter["segments"]) else "chunk",
                               audio_file=chapter["audio_file"], audio_sha256=chapter["encoded_sha256"],
                               segments=chapter["segments"])
            write_json(stage / "sync.json", sync)
            write_json(stage / "verification.json", verification_record(
                spec_hash=digest(spec), context=prepared["context"], acceptance=acceptance))
            artifacts = {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
                         for path in stage.iterdir()}
            manifest = manifest_record(identity=export_identity, spec=spec, status=chapter["status"],
                                       artifacts=artifacts)
            write_json(stage / "manifest.json", dict(manifest, record_sha256=signed_record(manifest)))
            try:
                os.rename(stage, folder)
            except OSError:
                if not folder.exists():
                    raise
        return _exported_paths(folder, export_identity)

    def close(self):
        with self._initialization:
            if self._closed:
                return
            self._closed = True
            try:
                close = getattr(self._verifier, "close", None)
                if callable(close):
                    close()
            finally:
                close = getattr(self._engine, "close", None)
                if callable(close):
                    close()
