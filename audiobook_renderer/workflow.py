"""Resumable synthesis; no audio is published after a failed quality gate."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from .assembly import compose_chapter, extract_cover, normalize_clip, package_audio
from .bundle import make_plan
from .common import digest, file_hash, read_json, write_json
from .quality import validation_key as make_validation_key, validate_audio
from .voice_bindings import design_voice, designer_identity, require_designer
from .voice_prompts import default_voice_prompt


class RenderBlocked(ValueError):
    pass


def _seed(base, key, attempt):
    return (base + int(digest(key)[:8], 16) + attempt) % 2**32


def _diagnostic_text(value, limit=60):
    """Keep untrusted recognition text short and inert in terminal output."""
    text = str(value)
    # Remove whole ANSI sequences before removing other terminal controls.
    text = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = "".join(" " if character.isspace() else character
                   for character in text if not unicodedata.category(character).startswith("C")
                   or character.isspace())
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _verification_failure_detail(report):
    """Summarize a few contiguous lexical edits without dumping the script."""
    alignment = report.get("metrics", {}).get("alignment", [])
    groups, expected, actual = [], [], []

    def finish_group():
        if expected or actual:
            left = _diagnostic_text(" ".join(expected)) or "(nothing)"
            right = _diagnostic_text(" ".join(actual)) or "(nothing)"
            groups.append(f"{left} → {right}")
            expected.clear()
            actual.clear()

    for edit in alignment if isinstance(alignment, list) else []:
        if not isinstance(edit, dict):
            continue
        if edit.get("operation") == "equal":
            finish_group()
        elif edit.get("operation") in {"insert", "delete", "substitute"}:
            for key, target in (("expected", expected), ("actual", actual)):
                words = edit.get(key, [])
                if isinstance(words, list):
                    target.extend(word for word in words if isinstance(word, str))
    finish_group()
    details = []
    if groups:
        changes = "; ".join(groups[:3])
        if len(groups) > 3:
            changes += f"; +{len(groups) - 3} more"
        details.append("expected → heard: " + changes)
    corroboration = report.get("corroboration", {})
    decision = corroboration.get("decision") if isinstance(corroboration, dict) else None
    explanation = {
        "mismatch_outside_reference_limits": "secondary check skipped: discrepancy exceeds reference limits",
        "unavailable": "secondary recognizer unavailable",
        "verification_failed": "secondary recognition failed",
        "disagreed": "recognizers disagreed",
    }.get(decision)
    if explanation:
        details.append(explanation)
    return "; ".join(details)


def verified_take(text, generate, directory, request, verifier, config, *, notify=print,
                  purpose="speech"):
    """Reuse only byte-bound, policy-bound accepted audio. Retain failed takes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    validation_key = make_validation_key(text, verifier, config["quality"], purpose=purpose)
    accepted_path = directory / "accepted.json"
    if accepted_path.exists():
        accepted = read_json(accepted_path)
        name = accepted.get("audio_file", "")
        if name != Path(name).name or not name.endswith(".wav"):
            raise RenderBlocked("Invalid cached audio path")
        wav = directory / name
        if accepted.get("request_hash") != digest(request) or file_hash(wav) != accepted.get("audio_sha256"):
            raise RenderBlocked("Cached audio changed; refusing to trust or silently overwrite it")
        if accepted.get("validation_key") == validation_key and accepted.get("report", {}).get("passed") is True:
            return wav, accepted
        # A new verifier or policy must hear the existing audio again.
        report = validate_audio(wav, text, verifier, config["quality"], purpose=purpose)
        accepted.update(validation_key=validation_key, report=report)
        write_json(accepted_path, accepted)
        if report["passed"]:
            return wav, accepted
    attempts = config["render"]["max_attempts"]
    for attempt in range(1, attempts + 1):
        prefix = f"attempt-{attempt:03d}"
        raw, wav = directory / (prefix + "-raw.wav"), directory / (prefix + ".wav")
        result_path = directory / (prefix + ".json")
        seed = _seed(config["render"]["seed"], request, attempt)
        report = None
        metadata = {}
        # Completed rejected attempts remain reviewable; interrupted raw outputs
        # never count as accepted. Recover a normalized candidate by revalidating.
        if result_path.exists():
            previous = read_json(result_path)
            if previous.get("request_hash") != digest(request):
                raise RenderBlocked("Attempt belongs to a different synthesis request")
            if previous.get("validation_key") == validation_key and not previous.get("report", {}).get("passed"):
                continue
            metadata = previous.get("engine_metadata", {})
        try:
            if wav.exists():
                if result_path.exists():
                    expected_hash = read_json(result_path).get("audio_sha256")
                    if expected_hash and file_hash(wav) != expected_hash:
                        raise RenderBlocked("Candidate audio changed on disk")
            else:
                # Orphans from interruption are retained; choose another raw name.
                if raw.exists():
                    raw = directory / (prefix + "-recovery-raw.wav")
                    if raw.exists():
                        raise RenderBlocked("Repeated interrupted generation requires inspecting the work directory")
                metadata = generate(raw, seed)
                normalize_clip(raw, wav, sample_rate=config["render"]["sample_rate"],
                               ffmpeg=config["render"]["ffmpeg"])
            report = validate_audio(wav, text, verifier, config["quality"], purpose=purpose)
        except RenderBlocked:
            raise
        except (ValueError, RuntimeError, OSError) as exc:
            report = {"passed": False, "reasons": ["generation_or_verification_error"],
                      "error": str(exc), "error_type": type(exc).__name__}
        record = {
            "request_hash": digest(request), "request": request, "attempt": attempt, "seed": seed,
            "validation_key": validation_key, "report": report, "engine_metadata": metadata,
            "audio_file": wav.name if wav.exists() else None,
            "audio_sha256": file_hash(wav) if wav.exists() else None,
        }
        write_json(result_path, record)
        if report.get("passed") is True:
            write_json(accepted_path, record)
            return wav, record
        detail = _verification_failure_detail(report)
        suffix = f" ({detail})" if detail else ""
        notify(f"  Attempt {attempt}/{attempts} rejected: {', '.join(report.get('reasons', []))}{suffix}")
    raise RenderBlocked(f"Speech verification failed after {attempts} attempts. Inspect {directory}")


def ensure_voice(store, voice_id, profile, engine, verifier, config, work, *, notify=print,
                 designer=None, bundle=None):
    store.persist_aliases()
    requested_id = voice_id
    voice_id = store.canonical_id(voice_id)
    profile = dict(profile, id=voice_id)
    existing = store.load(voice_id)
    if existing is not None:
        if any(existing["created_with"].get(key) != engine.identity.get(key) for key in ("engine", "model")):
            raise RenderBlocked("Existing voice was created for a different model; use an explicit compatible binding")
        return existing
    draft = store.directory / "casting-drafts" / (digest(voice_id)[:24] + ".json")
    if store.list_candidates(voice_id) or draft.exists() or draft.is_symlink():
        raise RenderBlocked(
            f"Voice {voice_id} has casting edits or saved audition candidates. "
            "Run the cast command to prepare its selected sample before rendering."
        )
    owned_designer = None
    try:
        settings = config.get("voice_design", {"type": "engine"})
        if designer is None and settings["type"] != "engine":
            from .voice_design import create_designer
            owned_designer = designer = create_designer(deepcopy(settings))
        try:
            source = require_designer(engine, designer)
        except ValueError as exc:
            raise RenderBlocked(str(exc)) from exc
        text = config["voices"]["reference_text"]
        sample_source = None
        if bundle is not None:
            from .samples import select_sample
            sample = select_sample(
                bundle, requested_id, canonical_id=store.canonical_id, fallback_text=text)
            text = sample["reference_text"]
            sample_source = sample["source"]
        prompt = default_voice_prompt(profile)
        request = {"kind": "create_voice", "engine": engine.identity, "designer": designer_identity(source),
                   "series_id": store.series_id, "voice_id": voice_id, "profile": profile,
                   "prompt": prompt, "text": text, "base_seed": config["render"]["seed"]}
        if sample_source is not None:
            request["sample_source"] = sample_source
        notify(f"Creating persistent reference voice {voice_id}")
        wav, accepted = verified_take(
            text, lambda path, seed: design_voice(source, prompt, text, path, seed=seed),
            Path(work) / "voice-candidates" / digest(request), request, verifier, config,
            notify=notify, purpose="reference")
        if accepted.get("engine_metadata", {}).get("description_applied") is not True:
            raise RenderBlocked("Saved generation has no evidence that the voice description was applied")
        reference_text = accepted["report"].get("reference_text", text)
        return store.publish(voice_id, profile, wav, reference_text, engine.identity,
                             accepted["report"], {
                                 **accepted.get("engine_metadata", {}),
                                 "description_applied": True, "designer": designer_identity(source),
                                 "requested_description": prompt,
                                 "requested_reference_text": text, "reference_text": reference_text,
                                 **({"sample_source": sample_source} if sample_source is not None else {}),
                                 "verifier": verifier.identity, "quality": config["quality"],
                                 "validation_key": make_validation_key(
                                     reference_text, verifier, config["quality"], purpose="reference"),
                             })
    finally:
        close = getattr(owned_designer, "close", None)
        if callable(close):
            close()


def import_voice(bundle, voice_id, audio, transcript, config, engine, verifier):
    from .book_context import resolve_book_config, voice_store
    config = resolve_book_config(bundle, config)
    expected_series = bundle["manifest"].get("provenance", {}).get("series", {}).get("series_id")
    if expected_series and expected_series != config["voices"]["series_id"]:
        raise ValueError("Configured voices.series_id must match the analyzer's series ID")
    profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
    if voice_id not in profiles:
        raise ValueError("Voice ID is not in the bundle cast")
    if not transcript.strip():
        raise ValueError("A reference recording needs its exact transcript")
    store = voice_store(bundle, config)
    lock = store.lock()
    try:
        store.persist_aliases()
        requested_id = voice_id
        voice_id = store.canonical_id(voice_id)
        profile = dict(profiles.get(voice_id, profiles[requested_id]), id=voice_id)
        existing = store.load(voice_id)
        if existing is not None and not existing.get("inherited"):
            raise ValueError("Voice already exists; immutable bindings cannot be replaced by import")
        with tempfile.TemporaryDirectory(prefix=".import-", dir=store.directory) as temp:
            wav = Path(temp) / "reference.wav"
            normalize_clip(audio, wav, sample_rate=config["render"]["sample_rate"], ffmpeg=config["render"]["ffmpeg"])
            report = validate_audio(wav, transcript, verifier, config["quality"], purpose="reference")
            if not report["passed"]:
                raise RenderBlocked("Reference recording failed verification: " + ", ".join(report["reasons"]))
            reference_text = report.get("reference_text", transcript)
            return store.publish(voice_id, profile, wav, reference_text, engine.identity, report,
                                 {"method": "supplied_reference", "description_applied": False,
                                  "requested_reference_text": transcript, "reference_text": reference_text,
                                  "verifier": verifier.identity, "quality": config["quality"]})
    finally:
        lock.rmdir()


def _publish(stage, destination):
    destination = Path(destination)
    destination.mkdir()  # Exclusive reservation: never overwrite an existing book.
    for source in sorted(stage.rglob("*")):
        if source.is_dir():
            (destination / source.relative_to(stage)).mkdir()
    # render.json is the completion marker. Ignore incomplete output without it.
    files = [p for p in stage.rglob("*") if p.is_file()]
    files.sort(key=lambda p: (p.name == "render.json", str(p)))
    for source in files:
        os.link(source, destination / source.relative_to(stage))


def render(bundle, config, engine, verifier, *, notify=print):
    from .book_context import resolve_book_config, voice_store
    try:
        config = resolve_book_config(bundle, config)
    except ValueError as exc:
        raise RenderBlocked(str(exc)) from exc
    plan = make_plan(bundle, config)
    if plan["errors"]:
        raise RenderBlocked("\n".join(plan["errors"]))
    work = Path(config["render"]["work_dir"])
    output = Path(config["render"]["output_dir"])
    if output.exists():
        raise RenderBlocked("Output already exists; choose a new output directory")
    epub = config["render"]["epub"]
    if epub and file_hash(epub) != bundle["source"]["book"]["id"]:
        raise RenderBlocked("Configured EPUB does not match the analyzed book")
    store = voice_store(bundle, config)
    work.mkdir(parents=True, exist_ok=True)
    lock = work / ".render.lock"
    lock.mkdir()
    voice_lock = None
    try:
        plan_hash = digest(plan)
        run = work / "runs" / plan_hash
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "plan.json", plan)
        write_json(work / "status.json", {"status": "running", "plan_hash": plan_hash})
        profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
        voice_lock = store.lock()
        store.persist_aliases()
        designer = None
        try:
            settings = config.get("voice_design", {"type": "engine"})
            if (settings["type"] != "engine"
                    and any(store.load(voice_id) is None for voice_id in plan["voice_ids"])):
                from .voice_design import create_designer
                designer = create_designer(deepcopy(settings))
            voices = {
                voice_id: ensure_voice(
                    store, voice_id, profiles.get(store.canonical_id(voice_id), profiles[voice_id]),
                    engine, verifier, config, work, notify=notify, designer=designer, bundle=bundle)
                for voice_id in plan["voice_ids"]
            }
        finally:
            close = getattr(designer, "close", None)
            if callable(close):
                close()
        voice_lock.rmdir()
        voice_lock = None
        rendered, acceptance = [], []
        # A new staging directory lets a rerun assemble from accepted clips safely.
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".audiobook-", dir=output.parent) as temporary:
            stage = Path(temporary)
            pcm = stage / "pcm"
            pcm.mkdir()
            for chapter in plan["chapters"]:
                notify(f"Rendering {chapter['id']}: {chapter['title']} ({len(chapter['chunks'])} chunks)")
                clips = []
                for chunk in chapter["chunks"]:
                    if chunk["silent_control"]:
                        clips.append(dict(chunk, wav_path=None))
                        continue
                    voice = voices[chunk["segment"]["voice_id"]]
                    direction = chunk["segment"].get("delivery", {}).get("direction", "")
                    request = {
                        "kind": "speech", "engine": engine.identity, "voice_asset_hash": voice["asset_hash"],
                        "chunk": chunk, "base_seed": config["render"]["seed"],
                        "sample_rate": config["render"]["sample_rate"],
                    }
                    wav, accepted = verified_take(
                        chunk["spoken_text"],
                        lambda path, seed, c=chunk, v=voice, d=direction:
                            engine.synthesize(c["spoken_text"], v, path, seed=seed, direction=d),
                        work / "clips" / digest(request), request, verifier, config, notify=notify)
                    acceptance.append({"chunk_id": chunk["chunk_id"], "audio_sha256": accepted["audio_sha256"],
                                       "request_hash": accepted["request_hash"],
                                       "validation_key": accepted["validation_key"], "report": accepted["report"]})
                    clips.append(dict(chunk, wav_path=str(wav)))
                composed = compose_chapter(clips, pcm / (chapter["id"] + ".wav"),
                                           sample_rate=config["render"]["sample_rate"],
                                           paragraph_pause_ms=config["render"]["paragraph_pause_ms"],
                                           blank_pause_ms=config["render"]["blank_pause_ms"],
                                           bracket_pause_ms=config["render"].get("bracket_pause_ms", 120))
                composed.update({key: chapter[key] for key in ("id", "title", "script_path")})
                rendered.append(composed)
            cover = None
            if config["render"]["epub"]:
                cover = extract_cover(config["render"]["epub"], stage / "artwork",
                                      expected_book_id=bundle["source"]["book"]["id"])
            audio = package_audio(
                rendered, stage, book=bundle["source"]["book"], cover_path=cover,
                chapter_format=config["render"]["chapter_format"],
                ffmpeg=config["render"]["ffmpeg"], ffprobe=config["render"]["ffprobe"],
                source_fingerprint=bundle["manifest"]["source_fingerprint"])
            # Accepted clips stay in the work cache. Package portable source and
            # timing, without duplicating full-book PCM in the published output.
            shutil.rmtree(pcm)
            write_json(stage / "source.json", bundle["source"])
            write_json(stage / "cast.json", bundle["cast"])
            write_json(stage / "verification.json", acceptance)
            result = {
                "format": "audiobook-render", "version": 1, "status": "complete",
                "bundle_hash": bundle["bundle_hash"], "source_fingerprint": plan["source_fingerprint"],
                "partial_book": plan["partial_book"], "series_id": store.series_id,
                "voice_assets": {key: value["asset_hash"] for key, value in voices.items()},
                "engine": engine.identity, "verifier": verifier.identity, "quality": config["quality"],
                "review_items": plan["review_items"], "allow_review": config["render"]["allow_review"],
                "audio": audio, "files_sha256": {
                    p.relative_to(stage).as_posix(): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()
                },
            }
            write_json(stage / "render.json", result)
            _publish(stage, output)
        write_json(work / "status.json", {"status": "complete", "plan_hash": plan_hash, "output": str(output)})
        return result
    except BaseException as exc:
        write_json(work / "status.json", {"status": "blocked", "error": str(exc), "error_type": type(exc).__name__})
        raise
    finally:
        if voice_lock is not None:
            voice_lock.rmdir()
        lock.rmdir()
