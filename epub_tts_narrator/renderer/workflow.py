"""Resumable synthesis; no audio is published after a failed quality gate."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from time import monotonic
import unicodedata

from .assembly import extract_cover, normalize_clip, package_audio
from .chapter_cache import (LANE_FIELDS, chapter_settings, load_chapter, prepare_chapter,
                            mixture_identity, mixture_validation_key, prepare_mixture)
from .bundle import chunk_speakers, make_plan
from .common import digest, file_hash, read_json, regular_file, write_json
from .config import resolve_voice_design
from .diagnostics import configured_secrets, safe_error_text
from .quality import validation_key as make_validation_key, validate_audio, text_confirmation_unavailable
from .progress import ChapterProgress, format_duration, set_phase
from .retry_budget import SpeechRetryBudget, generation_attempts
from .speech_text import trim_speech_text
from .voice_bindings import design_voice, designer_identity, require_designer
from .voice_prompts import default_voice_prompt


class RenderBlocked(ValueError):
    pass


class VerificationUnavailable(RenderBlocked):
    """Stop batch generation while an external verification service is unavailable."""


class SpeechVerificationFailed(RenderBlocked):
    """A bounded set of generated takes failed audio or transcript checks."""

    def __init__(self, message, *, request, report):
        super().__init__(message)
        self.request = request
        self.report = report


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
    judge = report.get("text_judge", {})
    judge_decision = judge.get("decision") if isinstance(judge, dict) else None
    judge_explanation = {
        "different": "JEV judged the transcript unlikely for the requested speech",
        "uncertain": "JEV could not confirm a plausible transcription",
        "unavailable": "JEV transcription check unavailable",
    }.get(judge_decision)
    if judge_explanation and not report.get("text_judge_skipped"):
        details.append(judge_explanation)
    reviews = report.get("phonetic_reviews", [])
    if (isinstance(reviews, list) and reviews and isinstance(reviews[-1], dict)
            and reviews[-1].get("decision") == "rejected"):
        details.append("phonetic reviewers rejected the pronunciation match; full votes are in the attempt report")
    skipped = report.get("text_judge_skipped")
    if isinstance(skipped, str):
        details.append("JEV not called: " + skipped.replace("_", " "))
    error = report.get("provider_error") or report.get("error")
    if error:
        details.append(safe_error_text(error))
    return "; ".join(details)


def _save_elevenlabs_generation(path, payload):
    write_json(path, {**payload, "record_sha256": digest(payload)})


def _load_elevenlabs_generation(path, request, attempt, seed):
    """Recover creation provenance only with its exact original audio and request."""
    try:
        if regular_file(path).stat().st_size > 4 * 1024 * 1024:
            raise ValueError("oversized generation record")
        payload = read_json(path)
        checksum = payload.pop("record_sha256")
        if (checksum != digest(payload) or payload.get("version") != 1
                or payload.get("request_hash") != digest(request)
                or payload.get("attempt") != attempt or payload.get("seed") != seed):
            raise ValueError("generation identity")
        raw_name = payload["raw_file"]
        if (not isinstance(raw_name, str) or not re.fullmatch(
                rf"attempt-{attempt:03d}-(?:raw|recovery-raw|regenerated-[0-9a-f]{{16}})\.wav", raw_name)):
            raise ValueError("generation audio path")
        raw = path.parent / raw_name
        if file_hash(regular_file(raw)) != payload["raw_sha256"]:
            raise ValueError("generation audio checksum")
        metadata = payload["engine_metadata"]
        if (not isinstance(metadata, dict) or metadata.get("description_applied") is not True
                or metadata.get("designer") != request["designer"]
                or metadata.get("reference_text") != request["text"]
                or metadata.get("requested_description") != request["prompt"]):
            raise ValueError("generation description")
        normalized_hash = payload.get("normalized_sha256")
        if normalized_hash is not None and (not isinstance(normalized_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", normalized_hash)):
            raise ValueError("normalized audio checksum")
        return raw, metadata, payload
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        raise RenderBlocked(f"Saved ElevenLabs generation metadata or audio changed; inspect {path}") from None


def verified_take(text, generate, directory, request, verifier, config, *, notify=print,
                  purpose="speech", resume_attempt=None, retry_budget=None, seed_identity=None):
    """Reuse byte-bound, request-bound accepted audio. Retain failed takes.

    Verification settings govern new candidates and pending verification. An existing
    successful verification remains provenance for the exact accepted bytes;
    changing recognizers or policy does not revoke that acceptance.
    Individual speech gets a fresh bounded generation allowance per explicit
    render invocation, separate from cumulative saved take numbers. Paragraphs
    keep their original allowance when a splitting recovery loop is available.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    accepted_path = directory / "accepted.json"
    fresh_speech = (retry_budget is not None and purpose == "speech"
                    and (request.get("kind") == "speech"
                         or (request.get("kind") == "paragraph_speech" and retry_budget.retry_paragraphs)))
    request_hash = digest(request)
    set_phase(notify, "checking saved audio")

    def finish(wav, record):
        # Acceptance remains attached to the normalized bytes. Raw synthesis
        # output is redundant only after acceptance; reference samples stay intact.
        if purpose == "speech" and not config["render"].get("keep_intermediate_audio", False):
            from .cleanup import clean_take_intermediates
            try:
                clean_take_intermediates(directory)
            except (ValueError, OSError) as exc:
                notify("  Intermediate audio cleanup skipped; accepted speech is retained: "
                       + safe_error_text(exc, secrets=configured_secrets(config)))
        return wav, record

    def pending_confirmation(report):
        return text_confirmation_unavailable(report)

    def stop_for_confirmation():
        raise VerificationUnavailable(
            "Speech text confirmation is unavailable. Saved audio was retained; "
            "rerun the same command to retry verification without regenerating it. "
            f"Inspect {directory}"
        )

    if resume_attempt is not None:
        if (purpose != "reference" or type(resume_attempt) is not int
                or not 1 <= resume_attempt <= 10):
            raise RenderBlocked("Invalid pending reference attempt")
        if accepted_path.exists() or accepted_path.is_symlink():
            raise RenderBlocked("A pending reference cannot resume an accepted candidate")
        pending_path = directory / f"attempt-{resume_attempt:03d}.json"
        pending_wav = directory / f"attempt-{resume_attempt:03d}.wav"
        if any(path.is_symlink() or not path.is_file() for path in (pending_path, pending_wav)):
            raise RenderBlocked("Pending reference verification requires the saved attempt and audio")
        pending = read_json(pending_path)
        if (not isinstance(pending, dict) or pending.get("request_hash") != digest(request)
                or not pending_confirmation(pending.get("report", {}))):
            raise RenderBlocked("Saved reference is not awaiting text confirmation")

    accepted = None
    missing_accepted_attempt = None
    demoted_attempt = None
    if accepted_path.exists() or accepted_path.is_symlink():
        try:
            accepted = read_json(regular_file(accepted_path))
        except (ValueError, OSError) as exc:
            raise RenderBlocked("Invalid cached acceptance record") from exc
        if not isinstance(accepted, dict):
            raise RenderBlocked("Invalid cached acceptance record")
        name = accepted.get("audio_file", "")
        if not isinstance(name, str) or name != Path(name).name or not name.endswith(".wav"):
            raise RenderBlocked("Invalid cached audio path")
        wav = directory / name
        if accepted.get("request_hash") != digest(request):
            raise RenderBlocked("Cached audio changed; refusing to trust or silently overwrite it")
        if not (isinstance(accepted.get("report"), dict)
                and accepted["report"].get("passed") is True):
            # Old versions wrote failed revalidation into the acceptance file.
            # Keep that outcome as an attempt diagnostic, never as a cache hit,
            # even if verification is interrupted again before recovery finishes.
            demoted_attempt = accepted.get("attempt")
            if (type(demoted_attempt) is not int or demoted_attempt < 1
                    or (purpose != "speech" and demoted_attempt > 10)
                    or name != f"attempt-{demoted_attempt:03d}.wav"):
                raise RenderBlocked("Invalid failed acceptance attempt")
            diagnostic_path = directory / f"attempt-{demoted_attempt:03d}.json"
            if diagnostic_path.exists() or diagnostic_path.is_symlink():
                previous = read_json(regular_file(diagnostic_path))
                if not isinstance(previous, dict) or previous.get("request_hash") != request_hash:
                    raise RenderBlocked("Attempt belongs to a different synthesis request")
            write_json(diagnostic_path, accepted)
            accepted_path.unlink()
            accepted = None
        elif not wav.exists() and not wav.is_symlink():
            missing_accepted_attempt = accepted.get("attempt")
            if (type(missing_accepted_attempt) is not int or missing_accepted_attempt < 1
                    or (purpose != "speech" and missing_accepted_attempt > 10)
                    or name != f"attempt-{missing_accepted_attempt:03d}.wav"):
                raise RenderBlocked("Invalid cached attempt for missing accepted audio")
            notify("  Saved accepted audio was removed; generating a replacement.")
            accepted_path.unlink()
            accepted = None
        else:
            try:
                checksum = file_hash(regular_file(wav))
            except (ValueError, OSError) as exc:
                raise RenderBlocked("Cached accepted audio is not a readable regular file") from exc
            if checksum != accepted.get("audio_sha256"):
                raise RenderBlocked("Cached audio changed; refusing to trust or silently overwrite it")
            if isinstance(accepted.get("report"), dict) and accepted["report"].get("passed") is True:
                return finish(wav, accepted)
    validation_key = make_validation_key(text, verifier, config["quality"], purpose=purpose)
    attempts = config["render"]["max_attempts"]
    secrets = configured_secrets(config)
    last_report = None
    first_attempt = missing_accepted_attempt or resume_attempt or 1
    if fresh_speech:
        saved_attempts = generation_attempts(directory)
        if missing_accepted_attempt is not None and missing_accepted_attempt not in saved_attempts:
            saved_attempts = sorted([*saved_attempts, missing_accepted_attempt])
        next_attempt = max(saved_attempts, default=0) + 1
        attempt_numbers = ([number for number in saved_attempts if number >= first_attempt]
                           + list(range(next_attempt, next_attempt + attempts)))
    else:
        saved_attempts = []
        attempt_numbers = range(first_attempt, max(attempts, first_attempt) + 1)
    for attempt in attempt_numbers:
        prefix = f"attempt-{attempt:03d}"
        raw, wav = directory / (prefix + "-raw.wav"), directory / (prefix + ".wav")
        result_path = directory / (prefix + ".json")
        seed = _seed(config["render"]["seed"],
                     request if seed_identity is None else seed_identity, attempt)
        report = None
        generation_ordinal = None
        metadata = {}
        elevenlabs = request.get("designer", {}).get("engine") == "elevenlabs_voice_design"
        generation_path = directory / (prefix + "-generation.json")
        generation = None
        # Completed rejected speech remains diagnostic only. Pending verification
        # can resume from its saved waveform, but a definitive failure needs a new take.
        if result_path.exists():
            previous = read_json(result_path)
            if previous.get("request_hash") != digest(request):
                raise RenderBlocked("Attempt belongs to a different synthesis request")
            if (fresh_speech and not wav.exists() and attempt != missing_accepted_attempt
                    and not previous.get("report", {}).get("passed")):
                # No waveform exists to recheck, even if verification settings
                # changed. Preserve the failed request and append a new take.
                last_report = previous.get("report", {})
                continue
            if ((purpose == "speech" or previous.get("validation_key") == validation_key)
                    and previous.get("report", {}).get("passed") is not True
                    and not pending_confirmation(previous.get("report", {}))
                    and not previous.get("report", {}).get("retry_on_resume", False)
                    and not (purpose == "reference" and attempt == demoted_attempt)):
                last_report = previous.get("report", {})
                continue
            metadata = previous.get("engine_metadata", {})
        if (fresh_speech and attempt in saved_attempts and not wav.exists()
                and attempt != missing_accepted_attempt):
            # Interrupted raw-only takes reserve their IDs but cannot be
            # verified. Append a new seed rather than recycling an orphan.
            continue
        try:
            if (elevenlabs and attempt != missing_accepted_attempt
                    and (generation_path.exists() or generation_path.is_symlink())):
                raw, metadata, generation = _load_elevenlabs_generation(
                    generation_path, request, attempt, seed)
            if wav.exists():
                if result_path.exists():
                    expected_hash = read_json(result_path).get("audio_sha256")
                    if expected_hash and file_hash(wav) != expected_hash:
                        raise RenderBlocked("Candidate audio changed on disk")
                if generation is not None:
                    checksum = file_hash(regular_file(wav))
                    if generation.get("normalized_sha256") is None:
                        # A crash may occur between normalization and recording
                        # its checksum. Recreate privately and compare, retaining
                        # the original rather than trusting an unbound WAV.
                        with tempfile.TemporaryDirectory(prefix=".verify-normalized-", dir=directory) as stage:
                            expected = Path(stage) / "audio.wav"
                            normalize_clip(raw, expected, sample_rate=config["render"]["sample_rate"],
                                           ffmpeg=config["render"]["ffmpeg"])
                            if checksum != file_hash(expected):
                                raise RenderBlocked("Candidate audio does not match the saved ElevenLabs generation")
                    elif checksum != generation["normalized_sha256"]:
                        raise RenderBlocked("Candidate audio changed after ElevenLabs generation")
                    generation["normalized_sha256"] = checksum
                    _save_elevenlabs_generation(generation_path, generation)
            else:
                # Orphans from interruption are retained; choose another raw name.
                if attempt == missing_accepted_attempt:
                    raw = directory / (prefix + "-regenerated-" + os.urandom(8).hex() + ".wav")
                elif raw.exists() and generation is None:
                    raw = directory / (prefix + "-recovery-raw.wav")
                    if raw.exists():
                        raise RenderBlocked("Repeated interrupted generation requires inspecting the work directory")
                if generation is None:
                    if fresh_speech:
                        generation_ordinal = retry_budget.claim(directory, request_hash, attempts)
                        if generation_ordinal is None:
                            break
                        # Reserve this take before discarding abandoned raw
                        # audio. Even an interruption before the engine writes
                        # anything must retain the cumulative ID and seed.
                        if attempt != missing_accepted_attempt:
                            write_json(result_path, {
                                "request_hash": request_hash, "request": request,
                                "attempt": attempt, "seed": seed, "validation_key": validation_key,
                                "report": {"passed": False, "reasons": ["generation_incomplete"],
                                           "retry_on_resume": True},
                                "engine_metadata": {}, "audio_file": None, "audio_sha256": None,
                            })
                        from .cleanup import discard_interrupted_raw
                        try:
                            discard_interrupted_raw(directory)
                        except (ValueError, OSError) as exc:
                            raise RenderBlocked(f"Could not remove interrupted raw speech: {exc}") from exc
                        if generation_ordinal == 1 and saved_attempts and missing_accepted_attempt is None:
                            notify(f"  Retrying failed speech with up to {attempts} new attempts; "
                                   "previous rejection reports retained.")
                    kind = "paragraph" if request.get("kind") == "paragraph_speech" else "speech"
                    current_attempt = generation_ordinal if fresh_speech else attempt
                    set_phase(notify, f"generating {kind}, attempt {current_attempt}/{attempts}", generated=True)
                    metadata = generate(raw, seed)
                    if elevenlabs:
                        generation = {"version": 1, "request_hash": digest(request), "attempt": attempt,
                                      "seed": seed, "raw_file": raw.name,
                                      "raw_sha256": file_hash(regular_file(raw)),
                                      "engine_metadata": metadata, "normalized_sha256": None}
                        _save_elevenlabs_generation(generation_path, generation)
                set_phase(notify, "normalizing speech")
                normalize_clip(raw, wav, sample_rate=config["render"]["sample_rate"],
                               ffmpeg=config["render"]["ffmpeg"])
                if generation is not None:
                    generation["normalized_sha256"] = file_hash(regular_file(wav))
                    _save_elevenlabs_generation(generation_path, generation)
            if fresh_speech and generation_ordinal is None:
                set_phase(notify, f"verifying saved take {attempt}")
            else:
                current_attempt = generation_ordinal if fresh_speech else attempt
                set_phase(notify, f"verifying speech, attempt {current_attempt}/{attempts}")
            report = validate_audio(wav, text, verifier, config["quality"], purpose=purpose)
        except RenderBlocked:
            raise
        except (ValueError, RuntimeError, OSError) as exc:
            error = safe_error_text(exc, secrets=secrets)
            report = {"passed": False, "reasons": ["generation_or_verification_error"],
                      "error": error, "error_type": type(exc).__name__}
            from .gemini_voice_design import GeminiVoiceError
            from .elevenlabs_voice_design import ElevenLabsVoiceError
            if isinstance(exc, (GeminiVoiceError, ElevenLabsVoiceError)):
                report["provider_error"] = error
            if getattr(exc, "retryable", True) is False:
                report["retryable"] = False
            if getattr(exc, "transport_error", False):
                # A new invocation may have repaired credentials/connectivity.
                # Permanent HTTP errors still stop immediately within this run.
                report["retry_on_resume"] = True
        record = {
            "request_hash": digest(request), "request": request, "attempt": attempt, "seed": seed,
            "validation_key": validation_key, "report": report, "engine_metadata": metadata,
            "audio_file": wav.name if wav.exists() else None,
            "audio_sha256": file_hash(wav) if wav.exists() else None,
        }
        write_json(result_path, record)
        last_report = report
        if report.get("passed") is True:
            write_json(accepted_path, record)
            judgment = report.get("text_judge", {})
            if judgment.get("decision") == "equivalent":
                notify("  JEV accepted the transcript as plausible recognition of the requested speech.")
            elif report.get("text_confirmation_method") in {"phonetic_primary", "phonetic_secondary"}:
                notify("  Phonetic reviewers confirmed the pronunciation variants after JEV uncertainty.")
            return finish(wav, record)
        if pending_confirmation(report):
            stop_for_confirmation()
        if report.get("retryable") is False:
            raise RenderBlocked(report["provider_error"] if "provider_error" in report else report["error"])
        detail = _verification_failure_detail(report)
        suffix = f" ({detail})" if detail else ""
        if fresh_speech:
            label = (f"Saved take {attempt}" if generation_ordinal is None
                     else f"Attempt {generation_ordinal}/{attempts}")
            if generation_ordinal is not None and attempt != generation_ordinal:
                label += f" (saved take {attempt})"
        else:
            label = f"Attempt {attempt}/{attempts}"
        notify(f"  {label} rejected: {', '.join(report.get('reasons', []))}{suffix}")
    if fresh_speech:
        used = retry_budget.started(directory, request_hash)
        failure = f"Speech verification failed after {used} attempts in this run. Inspect {directory}"
    else:
        failure = f"Speech verification failed after {attempts} attempts. Inspect {directory}"
    error = (last_report or {}).get("provider_error") or (last_report or {}).get("error")
    if error:
        failure += "\nLast error: " + safe_error_text(error, secrets=secrets)
    raise SpeechVerificationFailed(failure, request=request, report=last_report or {})


def ensure_voice(store, voice_id, profile, engine, verifier, config, work, *, notify=print,
                 designer=None, bundle=None):
    store.persist_aliases()
    requested_id = voice_id
    voice_id = store.canonical_id(voice_id)
    profile = dict(profile, id=voice_id)
    binding_identity = getattr(engine, "voice_binding_identity", engine.identity)
    existing = store.load(voice_id)
    if existing is not None:
        if any(existing["created_with"].get(key) != binding_identity.get(key) for key in ("engine", "model")):
            raise RenderBlocked("Existing voice was created for a different model; use an explicit compatible binding")
        return existing
    regeneration = store.regeneration_status(voice_id)
    if regeneration is not None:
        raise RenderBlocked(
            f"Voice {voice_id} regeneration is {regeneration['state']}; no active reference. "
            "Use G, GG, or G11 in the casting menu to retry before rendering."
        )
    draft = store.directory / "casting-drafts" / (digest(voice_id)[:24] + ".json")
    if store.list_candidates(voice_id) or draft.exists() or draft.is_symlink():
        raise RenderBlocked(
            f"Voice {voice_id} has casting edits or saved audition candidates. "
            "Run the cast command to prepare its selected sample before rendering."
        )
    owned_designer = None
    try:
        settings = resolve_voice_design(config)
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
        return store.publish(voice_id, profile, wav, reference_text, binding_identity,
                             accepted["report"], {
                                 **accepted.get("engine_metadata", {}),
                                 "description_applied": True, "designer": designer_identity(source),
                                 "engine": engine.identity,
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
            identity = engine.identity
            binding_identity = getattr(engine, "voice_binding_identity", identity)
            return store.publish(voice_id, profile, wav, reference_text, binding_identity, report,
                                 {"method": "supplied_reference", "description_applied": False,
                                  "engine": identity,
                                  "requested_reference_text": transcript, "reference_text": reference_text,
                                  "verifier": verifier.identity, "quality": config["quality"]})
    finally:
        lock.rmdir()


def _publish(stage, destination, **kwargs):
    """Install a finished book, preserving the previous output until it is ready."""
    from .publication import publish
    try:
        publish(stage, destination, **kwargs)
    except ValueError as exc:
        raise RenderBlocked(str(exc)) from exc


def make_render_plan(bundle, config, *, multi_speaker=True):
    """Group new paragraphs while preserving legacy accepted chunk boundaries."""
    plan = make_plan(bundle, config)
    work = config["render"].get("work_dir")
    requests = []
    if work:
        from .incremental_identity import (accepted_requests, completed_chunks,
                                           reanchor_export_sequences)
        try:
            requests = accepted_requests(work)
            plan = reanchor_export_sequences(
                plan, requests, completed=completed_chunks(work, plan["chapters"]))
        except (ValueError, OSError) as exc:
            raise RenderBlocked(f"Invalid saved synthesis identity: {exc}") from exc
    if not multi_speaker or not config["render"].get("paragraph_synthesis", True):
        return plan
    from .paragraphs import group_paragraphs
    preserved, preserved_groups = set(), []
    if work:
        from .paragraph_recovery import load_paragraph_fallbacks
        try:
            fallback_members = load_paragraph_fallbacks(work)
            preserved.update(fallback_members)
        except (ValueError, OSError) as exc:
            raise RenderBlocked(f"Invalid saved paragraph recovery: {exc}") from exc
        for request in requests:
            # A deleted WAV still preserves its request boundary: only that
            # deliberately removed clip should need regeneration on resume.
            if request.get("kind") == "speech":
                chunk = request.get("chunk", {})
                if isinstance(chunk, dict) and isinstance(chunk.get("chunk_id"), str):
                    preserved.add(chunk["chunk_id"])
            elif request.get("kind") == "paragraph_speech":
                chunk = request.get("chunk", {})
                turns = chunk.get("paragraph_turns", []) if isinstance(chunk, dict) else []
                if isinstance(turns, list) and len(turns) >= 2 and all(isinstance(turn, dict)
                        and isinstance(turn.get("chunk_id"), str) for turn in turns):
                    members = [turn["chunk_id"] for turn in turns]
                    # An older accepted performance can exist under a previous
                    # voice/model. It must not re-form a group that failed with
                    # the currently selected voices.
                    if not fallback_members.intersection(members):
                        preserved_groups.append(members)
    return group_paragraphs(plan, max_chars=config["render"].get("max_paragraph_chars", 1000),
                           preserved_chunk_ids=preserved, preserved_groups=preserved_groups)


def _preflight(bundle, config, *, chapters_only=False, multi_speaker=True):
    from .book_context import resolve_book_config, voice_store
    _check_analysis_update(bundle)
    try:
        config = resolve_book_config(bundle, config)
    except ValueError as exc:
        raise RenderBlocked(str(exc)) from exc
    plan = make_render_plan(bundle, config, multi_speaker=multi_speaker)
    if plan["errors"]:
        raise RenderBlocked("\n".join(plan["errors"]))
    output = Path(config["render"]["output_dir"])
    if not chapters_only:
        from .publication import published_book, check_selection
        try:
            previous = published_book(output, bundle["source"]["book"]["id"])
            check_selection(previous, (c["id"] for c in plan["chapters"]), partial=plan["partial_book"])
        except ValueError as exc:
            raise RenderBlocked(str(exc)) from exc
        config["_publication_expected"] = digest(previous) if previous is not None else None
    epub = config["render"]["epub"]
    if epub and file_hash(epub) != bundle["source"]["book"]["id"]:
        raise RenderBlocked("Configured EPUB does not match the analyzed book")
    return config, plan, voice_store(bundle, config)


def _check_analysis_update(bundle):
    lock = Path(bundle["directory"]) / ".analysis-update.lock"
    if lock.exists() or lock.is_symlink():
        raise RenderBlocked(f"This book's analysis is being updated; retry when it finishes: {lock}")


def _lock_book_audio(bundle):
    """Coordinate analysis revisions even when render work is outside the book."""
    lock = Path(bundle["directory"]) / ".audio-render.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RenderBlocked(
            f"Another process may be rendering or assembling this book. If it was forcibly terminated, "
            f"confirm it has stopped before removing the stale lock: {lock}") from exc
    try:
        # Revision takes its own lock before checking ours; neither may proceed
        # if both processes acquired their locks at almost the same time.
        _check_analysis_update(bundle)
        if digest(read_json(regular_file(Path(bundle["directory"]) / "manifest.json"))) != bundle["bundle_hash"]:
            raise RenderBlocked("This book's analysis changed after it was loaded; rerun the command")
    except BaseException:
        lock.rmdir()
        raise
    return lock


def _cache_engine_identity(engine_identity, config):
    """Attach explicit reuse policy without changing actual model provenance."""
    identity = deepcopy(engine_identity)
    identity.pop("cache_model_alias", None)
    alias = config["engine"].get("cache_model_alias")
    if alias:
        identity["cache_model_alias"] = alias
    return identity


def _speech_request(chunk, voice, engine_identity, config, *, performer=None):
    request = {
        "kind": "speech", "engine": _cache_engine_identity(engine_identity, config),
        "voice_asset_hash": voice["asset_hash"],
        "chunk": chunk, "base_seed": config["render"]["seed"],
        "sample_rate": config["render"]["sample_rate"],
    }
    if performer is not None:
        request["performer"] = dict(performer)
    return request


def _paragraph_request(chunk, voices, engine_identity, config):
    return {"kind": "paragraph_speech", "version": 1,
            "engine": _cache_engine_identity(engine_identity, config),
            "chunk": chunk, "base_seed": config["render"]["seed"],
            "sample_rate": config["render"]["sample_rate"],
            "voice_assets": {speaker["voice_id"]: voices[speaker["voice_id"]]["asset_hash"]
                             for speaker in chunk_speakers(chunk)}}


def _chapter_context(bundle, chapter, config, voices, engine_identity, verifier_identity):
    from .quality import QUALITY_VERSION
    used = {speaker["voice_id"] for chunk in chapter["chunks"] if not chunk["silent_control"]
            for speaker in chunk_speakers(chunk)}
    engine_config = dict(config["engine"])
    # The optional alias is enforced by the engine compatibility comparison;
    # omitting it must preserve old configuration hashes and checkpoints.
    engine_config.pop("cache_model_alias", None)
    engine_identity = _cache_engine_identity(engine_identity, config)
    # The original model was implicit before engine.model existed. Its explicit
    # default must not invalidate already completed chapter checkpoints.
    from .drama3_models import DEFAULT_DRAMA3_MODEL
    if (engine_config.get("type") != "drama3_mlx"
            or engine_config.get("model") == DEFAULT_DRAMA3_MODEL):
        engine_config.pop("model", None)
    if engine_identity.get("engine") == "fish_tts_pool":
        from .pool import pool_sampling
        from .drama3_models import get_drama3_model
        selected = get_drama3_model(config["engine"].get("model", DEFAULT_DRAMA3_MODEL))
        engine_config = {"sampling": pool_sampling(config),
                         "model_revision": config.get("render_pool", {}).get("model_revision", selected.revision)}
        if selected.repo != DEFAULT_DRAMA3_MODEL:
            engine_config["model"] = selected.repo
    return {
        "version": 1, "bundle_hash": bundle["bundle_hash"],
        "source_fingerprint": bundle["manifest"]["source_fingerprint"],
        "chapter_hash": digest(chapter), "total_chapters": len(bundle["chapters"]),
        "voice_assets": {key: voices[key]["asset_hash"] for key in sorted(used)},
        "engine": engine_identity, "verifier": verifier_identity,
        "engine_config_hash": digest(engine_config),
        "verification_config_hash": digest(config["verification"]),
        "quality": config["quality"], "quality_version": QUALITY_VERSION,
        "base_seed": config["render"]["seed"], "settings": chapter_settings(config["render"]),
    }


def _same_chapter_audio_context(saved, current):
    """Compare inputs that affect audio, retaining old verification provenance."""
    # chapter_hash still binds this chapter's full plan. A revision elsewhere
    # changes bundle_hash without changing any of this chapter's audio inputs.
    ignored = {"verifier", "verification_config_hash", "quality", "quality_version", "bundle_hash"}
    from .speech_identity import engines_compatible
    old_engine, new_engine = saved["engine"], current["engine"]
    routing_change = ((old_engine.get("engine") == "fish_tts_pool")
                      != (new_engine.get("engine") == "fish_tts_pool"))
    if routing_change or new_engine.get("cache_model_alias"):
        if not engines_compatible(old_engine, new_engine):
            return False
        ignored.update({"engine", "engine_config_hash"})
    return ({key: value for key, value in saved.items() if key not in ignored}
            == {key: value for key, value in current.items() if key not in ignored})


def _saved_chapter_engines_match(work, record, current_engine):
    """Check actual worker provenance before choosing chapter reuse over rendering."""
    from .speech_identity import engines_compatible
    for accepted in record["acceptance"]:
        for lane in accepted.get("speaker_lanes", [accepted]):
            saved = read_json(regular_file(Path(work) / "clips" / lane["request_hash"] / "accepted.json"))
            if not engines_compatible(saved["request"]["engine"], current_engine,
                                      engine_metadata=saved.get("engine_metadata")):
                return False
    return True


def _has_accepted_verification(record):
    return (isinstance(record.get("report"), dict) and record["report"].get("passed") is True
            and isinstance(record.get("validation_key"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["validation_key"]) is not None)


def _saved_chapter_clips_present(work, record):
    """Honor clip deletion while refusing to overwrite existing changed bytes."""
    acceptance = record.get("acceptance")
    if not isinstance(acceptance, list):
        raise RenderBlocked("Invalid saved chapter verification records")
    present = True
    for entry in acceptance:
        if not isinstance(entry, dict):
            raise RenderBlocked("Invalid saved chapter verification records")
        lanes = entry.get("speaker_lanes", [entry])
        if not isinstance(lanes, list):
            raise RenderBlocked("Invalid saved chapter speaker verification records")
        for lane in lanes:
            key = lane.get("request_hash") if isinstance(lane, dict) else None
            if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                raise RenderBlocked("Invalid saved chapter synthesis request")
            directory = Path(work) / "clips" / key
            path = directory / "accepted.json"
            if directory.is_symlink() or path.is_symlink():
                raise RenderBlocked("Cached accepted audio must not use symlinks")
            if not path.exists():
                present = False
                continue
            try:
                cached = read_json(regular_file(path))
                if not isinstance(cached, dict) or not _has_accepted_verification(cached):
                    present = False
                    continue
                name = cached.get("audio_file") if isinstance(cached, dict) else None
                if (not isinstance(name, str) or name != Path(name).name or not name.endswith(".wav")
                        or cached.get("request_hash") != key):
                    raise ValueError("Invalid accepted clip metadata")
                wav = directory / name
                if not wav.exists() and not wav.is_symlink():
                    present = False
                    continue
                checksum = file_hash(regular_file(wav))
                if checksum != cached.get("audio_sha256") or checksum != lane.get("audio_sha256"):
                    raise ValueError("Cached audio changed")
            except (ValueError, OSError) as exc:
                raise RenderBlocked(f"Invalid saved chapter clip {key}: {exc}") from exc
    return present


def _checkpoint_path(work, chapter):
    return Path(work) / "completed-chapters" / (chapter["id"] + ".json")


def _read_checkpoint(work, chapter):
    path = _checkpoint_path(work, chapter)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        record = read_json(regular_file(path))
        if not isinstance(record, dict):
            raise ValueError("Checkpoint must be a JSON object")
        checksum = record.pop("record_sha256")
        if (checksum != digest(record) or record.get("format") != "audiobook-chapter-checkpoint"
                or record.get("version") != 1
                or not isinstance(record.get("context"), dict)
                or not isinstance(record.get("cache_key"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["cache_key"])):
            raise ValueError("Checkpoint integrity check failed")
        return record
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise RenderBlocked(f"Invalid chapter checkpoint; inspect {path}: {exc}") from exc


def _restore_chapter(work, chapter, record, context, voices, config, *, speech_cache=None):
    if speech_cache is None:
        from .speech_cache import SpeechCache
        speech_cache = SpeechCache(work, voices)

    def matches_request(request, accepted):
        return accepted["request_hash"] == digest(request) or speech_cache.matches(request, accepted)

    if not _same_chapter_audio_context(record["context"], context):
        raise RenderBlocked(
            f"Completed {chapter['id']} is stale: source, selected voices, synthesis model, or "
            "render settings changed. Run render --chapters-only for this chapter again.")
    if not _saved_chapter_clips_present(work, record):
        raise RenderBlocked(f"A saved clip for {chapter['id']} was removed. "
                            "Run render --chapters-only for this chapter to restore it.")
    try:
        composed = load_chapter(Path(work) / "chapters" / record["cache_key"],
                                ffprobe=config["render"]["ffprobe"])
        if composed["context"] != record["context"] or composed["settings"] != context["settings"]:
            raise ValueError("Chapter cache and checkpoint disagree")
        # A checkpoint binds every spoken chunk to a successful verification,
        # including exact expected words and the voice selected for that chunk.
        expected = [chunk for chunk in chapter["chunks"] if not chunk["silent_control"]]
        acceptance = record["acceptance"]
        if not isinstance(acceptance, list) or len(acceptance) != len(expected):
            raise ValueError("Incomplete chapter verification records")
        cache_record = read_json(Path(composed["cache_directory"]) / "chapter.json")
        cached_clips = {clip["chunk_id"]: clip for clip in cache_record["identity"]["clips"]}
        timings = {item["chunk_id"]: item for item in composed["segments"]}
        for chunk, accepted in zip(expected, acceptance):
            if not isinstance(accepted, dict) or not isinstance(accepted.get("report"), dict):
                raise ValueError("Invalid chapter verification record")
            cached = cached_clips[chunk["chunk_id"]]
            if (accepted["chunk_id"] != chunk["chunk_id"] or not _has_accepted_verification(accepted)
                    or accepted["audio_sha256"] != cached["audio_sha256"]):
                raise ValueError("Chapter verification does not match the current plan")
            speakers = chunk_speakers(chunk)
            if "paragraph_turns" in chunk:
                request = _paragraph_request(chunk, voices, context["engine"], config)
                if not matches_request(request, accepted):
                    raise ValueError("Paragraph verification does not match its source or voices")
            elif len(speakers) > 1:
                lanes, mix = accepted.get("speaker_lanes"), accepted.get("mix")
                if (not isinstance(lanes, list) or len(lanes) != len(speakers)
                        or not isinstance(mix, dict)
                        or any(not isinstance(lane, dict) or not isinstance(lane.get("report"), dict)
                               or lane.get("start_frame") != 0 or type(lane.get("end_frame")) is not int
                               or lane["end_frame"] <= 0 for lane in lanes)):
                    raise ValueError("Incomplete simultaneous speaker verification")
                lane_metadata = [{key: lane[key] for key in LANE_FIELDS} for lane in lanes]
                if (cached.get("speaker_lanes") != lane_metadata or cached.get("mix") != mix
                        or mix.get("cache_key") != digest(mixture_identity(lanes, config["render"]["sample_rate"]))
                        or mix.get("method") != "aligned_fixed_average" or mix.get("version") != 1
                        or mix.get("sample_rate") != config["render"]["sample_rate"]
                        or mix.get("frames") != max(lane["end_frame"] for lane in lanes)
                        or mix.get("weight") != 1 / len(lanes)
                        or mix.get("audio_sha256") != accepted["audio_sha256"]
                        or accepted["request_hash"] != mix["cache_key"]
                        or accepted["validation_key"] != mixture_validation_key(lanes)
                        or accepted["report"].get("method") != "independently_verified_simultaneous_lanes"):
                    raise ValueError("Simultaneous mixture does not match its verified speaker lanes")
                timing = timings[chunk["chunk_id"]]
                if ([{key: lane[key] for key in LANE_FIELDS} for lane in timing.get("speaker_lanes", [])]
                        != lane_metadata or timing.get("mix") != mix
                        or timing["chapter_end_frame"] - timing["chapter_start_frame"] != mix["frames"]):
                    raise ValueError("Simultaneous synchronization differs from its verified lanes")
                for speaker, lane in zip(speakers, lanes):
                    request = _speech_request(chunk, voices[speaker["voice_id"]], context["engine"],
                                              config, performer=speaker)
                    if (lane["role_id"] != speaker["role_id"] or lane["voice_id"] != speaker["voice_id"]
                            or not matches_request(request, lane) or not _has_accepted_verification(lane)):
                        raise ValueError("A simultaneous speaker lacks matching verification")
            else:
                request = _speech_request(chunk, voices[speakers[0]["voice_id"]], context["engine"], config)
                if not matches_request(request, accepted):
                    raise ValueError("Chapter verification does not match the current plan")
        if [item["chunk_id"] for item in composed["segments"]] != [c["chunk_id"] for c in chapter["chunks"]]:
            raise ValueError("Chapter timing map does not cover the complete plan")
        return composed, acceptance
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise RenderBlocked(f"Invalid saved chapter {chapter['id']}: {exc}") from exc


def _save_checkpoint(work, chapter, composed, context, acceptance):
    record = {
        "format": "audiobook-chapter-checkpoint", "version": 1,
        "chapter_number": chapter["number"], "title": chapter["title"],
        "cache_key": composed["cache_key"], "context": context, "acceptance": acceptance,
        "audio_file": composed["encoded_path"],
    }
    write_json(_checkpoint_path(work, chapter), dict(record, record_sha256=digest(record)))


def _package(bundle, config, plan, store, voices, rendered, acceptance,
             engine_identity, verifier_identity, *, notify):
    from .publication import published_book, publication_key
    output = Path(config["render"]["output_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous = published_book(output, bundle["source"]["book"]["id"])
    except ValueError as exc:
        raise RenderBlocked(str(exc)) from exc
    expected = config.get("_publication_expected", digest(previous) if previous is not None else None)
    if (digest(previous) if previous is not None else None) != expected:
        raise RenderBlocked("Published output changed during rendering; rerun to use its current state")
    key = publication_key(bundle, config, plan, voices, rendered, acceptance, engine_identity, verifier_identity)
    unchanged = previous is not None and previous.get("publication_key") == key
    if previous is not None and not unchanged:
        # If all actual chapter audio and publication inputs are unchanged,
        # scheduling or an explicit model alias need not re-encode the book.
        from .speech_identity import engines_compatible
        if engines_compatible(previous.get("engine"), _cache_engine_identity(engine_identity, config)):
            unchanged = previous.get("publication_key") == publication_key(
                bundle, config, plan, voices, rendered, acceptance, previous["engine"], verifier_identity)
    if unchanged:
        notify(f"Audiobook unchanged; reusing {output / 'book.m4b'}")
        return previous
    with tempfile.TemporaryDirectory(prefix=".audiobook-", dir=output.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        cover = None
        if config["render"]["epub"]:
            cover = extract_cover(config["render"]["epub"], stage / "artwork",
                                  expected_book_id=bundle["source"]["book"]["id"])
            if cover is None:
                notify("The supplied EPUB has no supported JPEG/PNG cover; assembling without artwork.")
        else:
            notify("No EPUB supplied: assembling without cover artwork (use --epub to include it).")
        audio = package_audio(
            rendered, stage, book=bundle["source"]["book"], cover_path=cover,
            chapter_format=config["render"]["chapter_format"],
            aac_bitrate=config["render"].get("aac_bitrate", "128k"),
            mp3_bitrate=config["render"].get("mp3_bitrate", "128k"),
            ffmpeg=config["render"]["ffmpeg"], ffprobe=config["render"]["ffprobe"],
            source_fingerprint=bundle["manifest"]["source_fingerprint"], notify=notify)
        write_json(stage / "source.json", bundle["source"])
        write_json(stage / "cast.json", bundle["cast"])
        write_json(stage / "verification.json", acceptance)
        result = {
            "format": "audiobook-render", "version": 1, "status": "complete",
            "publication_key": key,
            "bundle_hash": bundle["bundle_hash"], "source_fingerprint": plan["source_fingerprint"],
            "partial_book": plan["partial_book"], "series_id": store.series_id,
            "voice_assets": {key: value["asset_hash"] for key, value in voices.items()},
            "engine": engine_identity, "verifier": verifier_identity, "quality": config["quality"],
            "verification_policy": "preserve_successful_acceptance",
            "verification_provenance": {
                "reports": "verification.json",
                "note": "Verifier and quality metadata do not imply cached audio was rechecked. "
                        "Each clip retains its original successful validation key and report.",
            },
            "review_items": plan["review_items"], "allow_review": config["render"]["allow_review"],
            "audio": audio, "files_sha256": {
                p.relative_to(stage).as_posix(): file_hash(p) for p in sorted(stage.rglob("*")) if p.is_file()
            },
        }
        write_json(stage / "render.json", result)
        _publish(stage, output, expected=expected)
    notify(f"Audiobook ready: {output / 'book.m4b'}")
    return result


def _start_run(work, plan, mode):
    work.mkdir(parents=True, exist_ok=True)
    lock = work / ".render.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RenderBlocked(
            f"Another render/assembly may be running. If it was forcibly terminated, confirm it has "
            f"stopped before removing the stale lock: {lock}") from exc
    try:
        plan_hash = digest(plan)
        write_json(work / "runs" / plan_hash / "plan.json", plan)
        write_json(work / "status.json", {"status": "running", "mode": mode, "plan_hash": plan_hash})
    except BaseException:
        lock.rmdir()
        raise
    return lock


def cleanup_render_work(bundle, config, *, dry_run=False, discard_render_cache=False):
    """Clean only this book's speech caches while excluding concurrent renders."""
    from .book_context import resolve_book_config
    from .cleanup import cleanup_work
    config = resolve_book_config(bundle, config)
    work = Path(config["render"]["work_dir"])
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise RenderBlocked("Cleanup work directory must not use symlinks")
    if not work.exists():
        return cleanup_work(work, dry_run=dry_run, discard_render_cache=discard_render_cache)
    if not work.is_dir():
        raise RenderBlocked("Cleanup work directory must be a directory")
    lock = work / ".render.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RenderBlocked("Another render, assembly, or cleanup may be running; "
                            f"retry after it finishes: {lock}") from exc
    book_lock = None
    try:
        book_lock = _lock_book_audio(bundle)
        return cleanup_work(work, dry_run=dry_run, discard_render_cache=discard_render_cache)
    finally:
        if book_lock is not None:
            book_lock.rmdir()
        lock.rmdir()


def _failed_run(work, exc, notify):
    write_json(work / "status.json", {"status": "blocked", "error": str(exc), "error_type": type(exc).__name__})
    notify(f"Verified clips and completed chapters are retained in {work}. Rerun the same command to resume.")


def _render_chunk(chunk, number, total, voices, engine, engine_identity, verifier,
                  config, work, take_locks, take_locks_guard, *, notify, retry_budget=None,
                  speech_cache=None):
    """Render one complete performance; publication and ordering stay upstream."""
    if chunk["silent_control"]:
        return dict(chunk, wav_path=None), None
    if speech_cache is None:
        from .speech_cache import SpeechCache
        speech_cache = SpeechCache(work, voices)

    def take(text, generate, request):
        identity = speech_cache.identity(request)
        key = digest(identity)
        with take_locks_guard:
            lock = take_locks.setdefault(key, Lock())
        # Duplicate logical jobs must never race over the same attempt files.
        with lock:
            try:
                cached = speech_cache.find(identity, request=request)
                if cached is not None:
                    if not config["render"].get("keep_intermediate_audio", False):
                        from .cleanup import clean_take_intermediates
                        try:
                            clean_take_intermediates(cached[0].parent)
                        except (ValueError, OSError) as exc:
                            notify("  Intermediate audio cleanup skipped; accepted speech is retained: "
                                   + safe_error_text(exc, secrets=configured_secrets(config)))
                    notify("  Reusing verified speech with matching text, voice, and generation settings.")
                    return cached
                # Full original requests remain the IDs of take/provenance
                # directories. They are not the successful audio lookup key.
                # Bind preparation semantics too: an old successful take with
                # different engine input must not bypass the content lookup.
                request = dict(request, synthesis_identity=identity)
                result = verified_take(text, generate, work / "clips" / digest(request), request,
                                       verifier, config, notify=notify, retry_budget=retry_budget,
                                       seed_identity=identity)
                speech_cache.remember(identity, result[1])
                return result
            except RenderBlocked:
                raise
            except ValueError as exc:
                raise RenderBlocked(str(exc)) from exc

    if "paragraph_turns" in chunk:
        request = _paragraph_request(chunk, voices, engine_identity, config)
        directory = work / "clips" / digest(request)
        turns = [{"text": turn["spoken_text"], "voice": voices[turn["segment"]["voice_id"]],
                  "direction": turn["segment"].get("delivery", {}).get("direction", "")}
                 for turn in chunk["paragraph_turns"]]
        cached = (directory / "accepted.json").is_file()
        notify(f"  Chunk {number}/{total}: "
               f"{'checking saved paragraph' if cached else 'preparing paragraph'} "
               f"({len(turns)} passages, {len(request['voice_assets'])} voices)")
        wav, accepted = take(chunk["spoken_text"],
            lambda path, seed: engine.synthesize_dialogue(turns, path, seed=seed), request)
        acceptance = dict(chunk_id=chunk["chunk_id"], **{key: accepted[key] for key in
            ("audio_sha256", "request_hash", "validation_key", "report")})
        return dict(chunk, wav_path=str(wav)), acceptance

    direction = chunk["segment"].get("delivery", {}).get("direction", "")
    # Formatting at the edges is not speech. Keep the original chunk for
    # verification, request/cache identity, and source synchronization.
    synthesis_text = trim_speech_text(chunk["spoken_text"])
    speakers = chunk_speakers(chunk)
    lanes = []
    for lane_number, speaker in enumerate(speakers, 1):
        voice = voices[speaker["voice_id"]]
        request = _speech_request(chunk, voice, engine_identity, config,
                                  performer=speaker if len(speakers) > 1 else None)
        directory = work / "clips" / digest(request)
        cached = (directory / "accepted.json").is_file()
        lane_label = f", simultaneous speaker {lane_number}/{len(speakers)}" if len(speakers) > 1 else ""
        notify(f"  Chunk {number}/{total}{lane_label}: "
               f"{'checking saved take' if cached else 'preparing speech'} ({speaker['voice_id']})")
        def generate(path, seed, v=voice):
            metadata = engine.synthesize(synthesis_text, v, path, seed=seed, direction=direction)
            return {"synthesis_text": synthesis_text, **metadata}

        wav, accepted = take(chunk["spoken_text"], generate, request)
        lanes.append(dict(speaker, wav_path=str(wav), **{key: accepted[key] for key in
            ("audio_sha256", "request_hash", "validation_key", "report")}))
    if len(lanes) > 1:
        mixed = prepare_mixture(lanes, work / "mixes", sample_rate=config["render"]["sample_rate"], notify=notify)
        mix, speaker_lanes = mixed["mix"], mixed["speaker_lanes"]
        acceptance = {
            "chunk_id": chunk["chunk_id"], "audio_sha256": mix["audio_sha256"],
            "request_hash": mix["cache_key"], "validation_key": mixture_validation_key(speaker_lanes),
            "report": {"passed": True, "method": "independently_verified_simultaneous_lanes"},
            "speaker_lanes": speaker_lanes, "mix": mix}
        return dict(chunk, wav_path=mixed["wav_path"], mix=mix,
                    speaker_lanes=[{key: lane[key] for key in LANE_FIELDS} for lane in speaker_lanes]), acceptance
    lane = lanes[0]
    acceptance = dict(chunk_id=chunk["chunk_id"], **{key: lane[key] for key in
        ("audio_sha256", "request_hash", "validation_key", "report")})
    return dict(chunk, wav_path=lane["wav_path"]), acceptance


def _render_chapter_chunks(chapter, voices, engine, engine_identity, verifier, config, work, *, notify,
                           total_chapters=None, retry_budget=None, take_locks=None, take_locks_guard=None,
                           speech_cache=None):
    """Bound in-flight jobs and restore source order before composing a chapter."""
    chunks = chapter["chunks"]
    if retry_budget is None:
        # Standalone chapter workers have no paragraph-splitting recovery loop.
        # Their explicit reruns need a fresh allowance for paragraphs too.
        retry_budget = SpeechRetryBudget(retry_paragraphs=True)
    if (take_locks is None) != (take_locks_guard is None):
        raise ValueError("Shared take locks require both the mapping and its guard")
    if take_locks is None:
        take_locks, take_locks_guard = {}, Lock()
    if speech_cache is None:
        from .speech_cache import SpeechCache
        speech_cache = SpeechCache(work, voices)
    parallel = getattr(engine, "max_inflight", getattr(engine, "max_parallel", 1))
    if type(parallel) is not int or parallel < 1:
        raise RenderBlocked("Render worker concurrency must be a positive integer")

    with ChapterProgress(chapter.get("number", 1), total_chapters or chapter.get("number", 1), chunks,
                         notify=notify, parallel=getattr(engine, "max_parallel", parallel)) as progress:
        def run(index):
            with progress.task(index) as task_notify:
                return _render_chunk(chunks[index], index + 1, len(chunks), voices, engine,
                                     engine_identity, verifier, config, work, take_locks,
                                     take_locks_guard, notify=task_notify, retry_budget=retry_budget,
                                     speech_cache=speech_cache)

        if parallel == 1:
            results = [run(index) for index in range(len(chunks))]
        else:
            results = [None] * len(chunks)
            executor = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="audiobook-render")
            pending, next_index = {}, 0
            try:
                while next_index < len(chunks) and len(pending) < parallel:
                    pending[executor.submit(run, next_index)] = next_index
                    next_index += 1
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    # Inspect every completion before submitting more work, so a
                    # known failure cannot keep a large queue of new jobs running.
                    for future in finished:
                        index = pending.pop(future)
                        results[index] = future.result()
                    while next_index < len(chunks) and len(pending) < parallel:
                        pending[executor.submit(run, next_index)] = next_index
                        next_index += 1
            finally:
                for future in pending:
                    future.cancel()
                # In-flight remote/local jobs may still publish verified clips.
                # Drain before render() can release book/cache locks, even on Ctrl-C.
                executor.shutdown(wait=True, cancel_futures=True)
        return [result[0] for result in results], [result[1] for result in results if result[1] is not None]


def _render_chapter_with_recovery(bundle, chapter, voices, engine, engine_identity,
                                verifier, config, work, *, notify, retry_budget=None, speech_cache=None):
    """Split only exhausted paragraph performances, retaining verified work."""
    from .paragraph_recovery import record_paragraph_fallback

    if retry_budget is None:
        retry_budget = SpeechRetryBudget()
    recoverable = {"text_mismatch", "audio_too_short", "audio_too_long", "silent_audio",
                   "audio_too_quiet", "excessive_silence", "internal_silence_stall",
                   "excessive_clipping"}
    while True:
        try:
            clips, acceptance = _render_chapter_chunks(
                chapter, voices, engine, engine_identity, verifier, config, work,
                notify=notify, total_chapters=len(bundle["chapters"]), retry_budget=retry_budget,
                speech_cache=speech_cache)
            return chapter, clips, acceptance
        except SpeechVerificationFailed as exc:
            request, report = exc.request, exc.report
            reasons = report.get("reasons", [])
            chunk = request.get("chunk", {})
            if (request.get("kind") != "paragraph_speech"
                    or not isinstance(reasons, list) or not reasons
                    or any(reason not in recoverable for reason in reasons)
                    or text_confirmation_unavailable(report)
                    or not any(item == chunk for item in chapter["chunks"])):
                raise
            # Record before any child generation so an interruption cannot
            # re-form the failed group on the next invocation. This changes
            # synthesis boundaries, never the script, voices, or quality gates.
            try:
                record_paragraph_fallback(work, request)
            except (ValueError, OSError) as error:
                raise RenderBlocked(f"Could not save paragraph recovery: {error}") from error
            new_plan = make_render_plan(bundle, config, multi_speaker=True)
            replacement = next(item for item in new_plan["chapters"] if item["id"] == chapter["id"])
            if (new_plan["errors"] or any(item["chunk_id"] == chunk["chunk_id"]
                                         for item in replacement["chunks"])
                    or "".join(item["spoken_text"] for item in chapter["chunks"])
                       != "".join(item["spoken_text"] for item in replacement["chunks"])):
                raise RenderBlocked("Paragraph recovery did not preserve the complete speech plan") from exc
            notify(f"  Paragraph failed verification; retrying its {len(chunk['paragraph_turns'])} "
                   "passages separately with their assigned voices. Verified clips are retained.")
            chapter = replacement


def render(bundle, config, engine, verifier, *, notify=print, chapters_only=False):
    run_started = monotonic()
    retry_budget = SpeechRetryBudget()
    wrap_verifier = getattr(engine, "wrap_verifier", None)
    if callable(wrap_verifier):
        verifier = wrap_verifier(verifier)
    config, plan, store = _preflight(
        bundle, config, chapters_only=chapters_only,
        multi_speaker=getattr(engine, "capabilities", {}).get("multi_speaker", False))
    work = Path(config["render"]["work_dir"])
    lock = _start_run(work, plan, "chapters" if chapters_only else "render")
    book_lock = None
    voice_lock = None
    try:
        book_lock = _lock_book_audio(bundle)
        profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
        voice_lock = store.lock()
        store.persist_aliases()
        designer = None
        try:
            settings = resolve_voice_design(config)
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
        engine_identity, verifier_identity = engine.identity, verifier.identity
        from .speech_cache import SpeechCache
        speech_cache = SpeechCache(work, voices)
        for chapter_index, chapter in enumerate(plan["chapters"]):
            chapter_started = monotonic()
            chapter_label = f"chapter {chapter['number']} / {len(bundle['chapters'])}"
            context = _chapter_context(bundle, chapter, config, voices, engine_identity, verifier_identity)
            saved = _read_checkpoint(work, chapter)
            if (saved is not None and _same_chapter_audio_context(saved["context"], context)
                    and _saved_chapter_clips_present(work, saved)
                    and _saved_chapter_engines_match(work, saved, context["engine"])):
                composed, chapter_acceptance = _restore_chapter(
                    work, chapter, saved, context, voices, config, speech_cache=speech_cache)
                notify(f"Reusing completed {chapter_label}: {chapter['title']} ({composed['encoded_path'] or 'empty'})")
            else:
                notify(f"Rendering {chapter_label}: {chapter['title']} ({len(chapter['chunks'])} blocks)")
                chapter, clips, chapter_acceptance = _render_chapter_with_recovery(
                    bundle, chapter, voices, engine, engine_identity, verifier, config, work,
                    notify=notify, retry_budget=retry_budget, speech_cache=speech_cache)
                if chapter != plan["chapters"][chapter_index]:
                    plan["chapters"][chapter_index] = chapter
                    write_json(work / "runs" / digest(plan) / "plan.json", plan)
                context = _chapter_context(bundle, chapter, config, voices, engine_identity, verifier_identity)
                composed = prepare_chapter(
                    dict(chapter, _ordinal=chapter["number"], _total_chapters=len(bundle["chapters"])),
                    clips, work / "chapters", book=bundle["source"]["book"],
                    render_settings=config["render"], context=context, notify=notify)
                _save_checkpoint(work, chapter, composed, context, chapter_acceptance)
            rendered.append(composed)
            acceptance.extend(chapter_acceptance)
            notify(f"Completed {chapter_label} in {format_duration(monotonic() - chapter_started)}; "
                   f"run elapsed {format_duration(monotonic() - run_started)} "
                   f"({len(rendered)} / {len(plan['chapters'])} selected chapters ready).")
        if chapters_only:
            result = {
                "format": "audiobook-chapters", "version": 1, "status": "chapters_complete",
                "partial_book": plan["partial_book"], "chapter_count": len(rendered),
                "chapters": [{"number": c["number"], "title": c["title"], "audio_file": c["encoded_path"],
                              "cache_directory": c["cache_directory"]} for c in rendered],
            }
            notify(f"Saved {len(rendered)} chapter checkpoint(s) in {work / 'completed-chapters'}. "
                   "Use assemble to create the M4B.")
        else:
            result = _package(bundle, config, plan, store, voices, rendered, acceptance,
                              engine_identity, verifier_identity, notify=notify)
        write_json(work / "status.json", {
            "status": result["status"], "plan_hash": digest(plan),
            "output": None if chapters_only else config["render"]["output_dir"],
            "chapters": [c["number"] for c in rendered]})
        return result
    except BaseException as exc:
        _failed_run(work, exc, notify)
        raise
    finally:
        if voice_lock is not None:
            voice_lock.rmdir()
        if book_lock is not None:
            book_lock.rmdir()
        lock.rmdir()


def assemble(bundle, config, *, notify=print):
    """Package complete verified chapter checkpoints without importing any model."""
    config, plan, store = _preflight(bundle, config)
    work = Path(config["render"]["work_dir"])
    lock = _start_run(work, plan, "assemble")
    book_lock = None
    try:
        book_lock = _lock_book_audio(bundle)
        voices = {voice_id: store.load(voice_id) for voice_id in plan["voice_ids"]}
        missing = [key for key, value in voices.items() if value is None]
        if missing:
            raise RenderBlocked("Selected voices are missing: " + ", ".join(missing))
        records = [_read_checkpoint(work, chapter) for chapter in plan["chapters"]]
        missing = [chapter["number"] for chapter, record in zip(plan["chapters"], records) if record is None]
        if missing:
            raise RenderBlocked("Chapters have not finished rendering: " + ", ".join(map(str, missing))
                                + ". Run render --chapters-only first.")
        rendered, acceptance = [], []
        identities = None
        publication_provenance = None
        from .speech_cache import SpeechCache
        speech_cache = SpeechCache(work, voices)
        for chapter, record in zip(plan["chapters"], records):
            saved = record["context"]
            current_engine = _cache_engine_identity(saved["engine"], config)
            if current_engine.get("cache_model_alias"):
                # Assembly has no live engine identity. Apply the requested
                # model/sampling so alias reuse cannot hide changed settings.
                from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model
                selected = get_drama3_model(config["engine"].get("model", DEFAULT_DRAMA3_MODEL))
                current_engine["model"] = selected.repo
                if current_engine.get("engine") == "fish_tts_pool":
                    from .pool import pool_sampling
                    current_engine["sampling"] = pool_sampling(config)
                else:
                    current_engine["engine"] = "drama3_mlx" if selected.runtime == "sogni" else "drama3_mlx_audio"
                    for key, default in (("temperature", 1.0), ("top_p", .9),
                                         ("top_k", 30), ("max_new_tokens", 2048)):
                        current_engine[key] = config["engine"].get(key, default)
            current = (current_engine, saved["verifier"])
            from .speech_identity import engines_compatible
            if identities is not None and not engines_compatible(identities[0], current[0]):
                raise RenderBlocked("Completed chapters used different synthesis models; rerun render "
                                    "--chapters-only with consistent settings before assembly.")
            identities = current
            publication_provenance = (saved["engine"], saved["verifier"])
            context = _chapter_context(bundle, chapter, config, voices, *identities)
            # Encoding is independent of verified speech and its PCM timeline.
            # Allow repackaging as MP3 or at a different bitrate without models;
            # Gain, pauses, source and voice changes still block; accepted
            # verification remains attached to the unchanged saved audio.
            encoding_keys = {"chapter_format", "aac_bitrate", "mp3_bitrate", "ffmpeg", "ffprobe"}
            restore_context = deepcopy(context)
            for key in encoding_keys:
                restore_context["settings"][key] = saved["settings"][key]
            composed, chapter_acceptance = _restore_chapter(
                work, chapter, record, restore_context, voices, config, speech_cache=speech_cache)
            if any(context["settings"][key] != saved["settings"][key]
                   for key in ("chapter_format", "aac_bitrate", "mp3_bitrate")):
                composed["encoded_path"] = None
                composed["encoded_sha256"] = None
            rendered.append(composed)
            acceptance.extend(chapter_acceptance)
            notify(f"Verified saved chapter {chapter['number']} / {len(bundle['chapters'])}: {chapter['title']}")
        if identities is None:
            raise RenderBlocked("No chapters selected for assembly")
        result = _package(bundle, config, plan, store, voices, rendered, acceptance,
                          *publication_provenance, notify=notify)
        write_json(work / "status.json", {"status": "complete", "plan_hash": digest(plan),
                                          "output": config["render"]["output_dir"]})
        return result
    except BaseException as exc:
        _failed_run(work, exc, notify)
        raise
    finally:
        if book_lock is not None:
            book_lock.rmdir()
        lock.rmdir()
