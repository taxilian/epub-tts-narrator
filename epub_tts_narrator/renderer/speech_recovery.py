"""One bounded sentence-splitting strategy for exhausted speech generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import tempfile
import wave

from .common import digest, file_hash, read_json, regular_file, write_json
from .speech_text import trim_speech_text


_VERSION = 1
_STRATEGY = "sentence_split"
_MARKER = "speech-recovery.json"
_ABBREVIATIONS = frozenset("mr mrs ms dr prof sr jr st vs etc inc ltd co dept fig vol no".split())


def _period_can_end(text, index):
    if index > 0 and text[index - 1] == ".":
        return False
    word = re.search(r"([A-Za-z]+)$", text[:index])
    return word is None or (len(word[1]) > 1 and word[1].casefold() not in _ABBREVIATIONS)


def split_recovery_text(text):
    """Return two to four contiguous exact slices, each containing >=2 words."""
    if not isinstance(text, str) or not text.strip() or len(text) > 100000:
        return None
    candidates = set()
    for match in re.finditer(r"[.!?;][\"'”’)]*\s+|\n+\s*", text):
        punctuation = match.start()
        if text[punctuation] == "." and not _period_can_end(text, punctuation):
            continue
        if text[punctuation] == "\n":
            previous = len(text[:punctuation].rstrip()) - 1
            if previous >= 0 and text[previous] == "." and not _period_can_end(text, previous):
                continue
        if match.end() < len(text):
            candidates.add(match.end())
    parts = _split_at_boundaries(text, candidates)
    if parts is not None:
        return parts  # Existing persisted sentence plans keep their exact identities.
    addresses = [token for token in re.finditer(r"\S+", text)
                 if any(marker in token[0] for marker in ("://", "www.", "@"))]
    candidates = {match.end() for match in re.finditer(r"—[\"'”’)]*\s*", text)
                  if match.end() < len(text)
                  and not any(address.start() <= match.start() < address.end() for address in addresses)}
    return _split_at_boundaries(text, candidates)


def _split_at_boundaries(text, candidates):
    """Choose bounded exact slices without leaving single-word fragments."""
    from .quality import normalize_words

    parts, start = [], 0
    for end in sorted(candidates):
        if len(parts) == 3:
            break
        if len(normalize_words(text[start:end])) < 2 or len(normalize_words(text[end:])) < 2:
            continue
        parts.append({"start": start, "end": end, "text": text[start:end]})
        start = end
    if not parts or len(normalize_words(text[start:])) < 2:
        return None
    parts.append({"start": start, "end": len(text), "text": text[start:]})
    return parts


def _requests(text, request, config, voices_by_asset=None):
    if (not isinstance(request, dict) or request.get("kind") not in {"speech", "paragraph_speech"}
            or "recovery" in request or request.get("chunk", {}).get("spoken_text") != text):
        return None
    if request["kind"] == "paragraph_speech":
        from .speech_identity import paragraph_rewrite_turn
        if paragraph_rewrite_turn(request, voices_by_asset or {}) is None:
            return None
    parts = split_recovery_text(text)
    if parts is None:
        return None
    maximum = config["render"]["max_attempts"]
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise ValueError("Speech recovery requires one to ten attempts per part")
    recovery = {"version": _VERSION, "strategy": _STRATEGY, "parts": parts}
    recovered_request = {**deepcopy(request), "recovery": recovery}
    body = {"format": "audiobook-speech-recovery", "version": _VERSION,
            "parent_request_hash": digest(request), "recovery_request_hash": digest(recovered_request),
            "recovery": recovery}
    return recovered_request, {**body, "record_sha256": digest(body)}


def _directory(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink() or (path.exists() and not path.is_dir()):
        raise ValueError("Speech recovery directories must be nonsymlink directories")
    return path


def _existing_marker(directory, marker):
    path = directory / _MARKER
    if not path.exists() and not path.is_symlink():
        return False
    if read_json(regular_file(path)) != marker:
        raise ValueError("Saved speech recovery does not match the current request and split plan")
    return True


def recovery_available(directory, request, text, verifier, config, *, voices_by_asset=None):
    """Return whether this request has a valid persisted strategy to resume."""
    from .workflow import RenderBlocked

    try:
        directory = _directory(directory)
        path = directory / _MARKER
        if not path.exists() and not path.is_symlink():
            return False
        planned = _requests(text, request, config, voices_by_asset)
        if planned is None or directory.name != digest(request):
            raise ValueError("Saved speech recovery has no matching eligible parent request")
        return _existing_marker(directory, planned[1])
    except (ValueError, OSError) as exc:
        raise RenderBlocked(f"Invalid speech recovery: {exc}") from exc


def _concatenate(parts, target, sample_rate):
    """Publish an exact ordered PCM concatenation only after every input checks."""
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise ValueError("Speech recovery output already exists")
    with tempfile.TemporaryDirectory(prefix=".recovery-join-", dir=target.parent) as temporary:
        staged = Path(temporary) / "joined.wav"
        with wave.open(str(staged), "wb") as output:
            output.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
            for path, record in parts:
                path = regular_file(path)
                if file_hash(path) != record["audio_sha256"]:
                    raise ValueError("Verified recovery part changed before assembly")
                with wave.open(str(path), "rb") as source:
                    if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, sample_rate, "NONE"):
                        raise ValueError("Recovery parts must share the normalized mono PCM16 format")
                    remaining = source.getnframes()
                    if remaining <= 0:
                        raise ValueError("Recovery part is empty")
                    while remaining:
                        count = min(remaining, 32768)
                        data = source.readframes(count)
                        if len(data) != count * 2:
                            raise ValueError("Recovery part has truncated PCM audio")
                        output.writeframesraw(data)
                        remaining -= count
        if any(file_hash(regular_file(path)) != record["audio_sha256"] for path, record in parts):
            raise ValueError("Verified recovery part changed during assembly")
        target.hardlink_to(staged)


def _legacy_accepted_part(directory, request):
    """Keep an existing successful child ID after adding richer input bindings.

    The normal verified-take path checks this again before reuse. A mere passed
    flag is insufficient: the complete request, source text, and saved bytes must
    match. A corrupt old acceptance is an error, never silently bypassed.
    """
    path = directory / "accepted.json"
    if not path.exists() and not path.is_symlink():
        return False
    accepted = read_json(regular_file(path))
    if (not isinstance(accepted, dict) or accepted.get("request") != request
            or accepted.get("request_hash") != digest(request)):
        raise ValueError("Existing recovery part acceptance differs from its original request")
    report = accepted.get("report")
    if not isinstance(report, dict) or report.get("passed") is not True:
        return False
    if report.get("reasons") or report.get("expected_text") != request["part"]["text"]:
        raise ValueError("Existing recovery part acceptance differs from its source speech")
    name = accepted.get("audio_file")
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
        raise ValueError("Invalid existing recovery part audio path")
    wav = directory / name
    if not wav.exists() and not wav.is_symlink():
        return False
    if file_hash(regular_file(wav)) != accepted.get("audio_sha256"):
        raise ValueError("Existing recovery part audio changed")
    return True


def recover_speech(text, synthesize_part, parent_directory, parent_request, verifier, config,
                   *, notify=print, retry_budget=None, resolution_episode=None, voices_by_asset=None,
                   synthesize_plain=None):
    """Verify split parts with the caller's bounded allowance for this invocation.

    The caller selects this strategy only after ordinary speech takes fail.
    Successful parts survive later invocations; failed parts get fresh takes only
    with a fresh caller budget. The single assembly candidate may resume pending
    or unavailable verification, but a definite rejection is final.
    """
    from .workflow import RenderBlocked, SpeechVerificationFailed, VerificationUnavailable, verified_take

    try:
        planned = _requests(text, parent_request, config, voices_by_asset)
        if planned is None:
            raise ValueError("Speech has no safe sentence boundary for automatic recovery")
        request, marker = planned
        parent = _directory(parent_directory)
        if parent.name != digest(parent_request):
            raise ValueError("Speech recovery parent directory does not match its request")
        directory = _directory(parent.parent / marker["recovery_request_hash"])
        parent.mkdir(parents=True, exist_ok=True)
        if not _existing_marker(parent, marker):
            write_json(parent / _MARKER, marker)
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "attempt-001.wav").exists():
            # An interrupted join or normalization has no accepted full waveform.
            # Recreate that cheap assembly from verified children, without keeping
            # successive orphan raw names or disturbing pending normalized audio.
            from .cleanup import discard_interrupted_raw
            discard_interrupted_raw(directory)
    except (ValueError, OSError) as exc:
        raise RenderBlocked(f"Could not prepare speech recovery: {exc}") from exc
    notify(f"  Recovering speech as {len(request['recovery']['parts'])} separately verified parts.")
    from .delivery_recovery import plain_delivery_request
    bind_single_inputs = (parent_request["kind"] == "speech" and callable(synthesize_plain)
                          and plain_delivery_request(parent_request) is not None)

    def assemble(path, seed):
        verified, audit = [], []
        for number, part in enumerate(request["recovery"]["parts"], 1):
            child_request = {"kind": "speech_recovery_part", "version": _VERSION,
                             "parent_request_hash": digest(parent_request),
                             "recovery_request_hash": digest(request), "part": part}
            if parent_request["kind"] == "paragraph_speech":
                from .speech_identity import paragraph_recovery_part
                child_request["synthesis_request"] = paragraph_recovery_part(parent_request, part, voices_by_asset)
            elif bind_single_inputs:
                from .speech_identity import single_speech_recovery_part
                legacy_directory = _directory(directory / "parts" / digest(child_request))
                if not _legacy_accepted_part(legacy_directory, child_request):
                    child_request["synthesis_request"] = single_speech_recovery_part(parent_request, part)
            child_directory = _directory(directory / "parts" / digest(child_request))
            def generate_part(output, child_seed, piece=part["text"]):
                return synthesize_part(piece, output, child_seed)
            notify(f"  Recovery part {number}/{len(request['recovery']['parts'])}: checking saved audio.")
            wav, record = verified_take(part["text"], generate_part, child_directory,
                                        child_request, verifier, config, notify=notify,
                                        purpose="speech", retry_budget=retry_budget,
                                        generate_repaired=synthesize_part,
                                        generate_plain=(synthesize_plain if "synthesis_request" in child_request
                                                        else None),
                                        resolution_episode=resolution_episode,
                                        episode_reserve=len(request["recovery"]["parts"]) - number)
            verified.append((wav, record))
            audit.append({**part, "request_hash": record["request_hash"],
                          "audio_sha256": record["audio_sha256"],
                          "validation_key": record["validation_key"],
                          "accepted_sha256": file_hash(wav.parent / "accepted.json"),
                          "audio_file": str(wav.relative_to(directory))})
            if parent_request["kind"] == "paragraph_speech" or bind_single_inputs:
                # Retain actual child inputs, including any audited rewrite.
                # The joined file is not represented as a one-shot model call.
                audit[-1]["acceptance"] = deepcopy(record)
        _concatenate(verified, path, config["render"]["sample_rate"])
        recovery_audit = {"version": _VERSION, "strategy": _STRATEGY,
                          "parent_request_hash": digest(parent_request), "parts": audit}
        if bind_single_inputs:
            recovery_audit["child_acceptance_version"] = 1
        return {"synthesis_text": trim_speech_text(text), "recovery": recovery_audit}

    assembly_config = deepcopy(config)
    assembly_config["render"]["max_attempts"] = 1
    try:
        return verified_take(text, assemble, directory, request, verifier, assembly_config,
                             notify=notify, purpose="speech", retry_budget=None)
    except SpeechVerificationFailed as exc:
        if exc.request != request:
            raise  # A child exhausted its own allowance; keep that diagnostic.
        reasons = exc.report.get("reasons", [])
        unavailable = (exc.report.get("retryable") is not False and reasons and
                       all(reason == "generation_or_verification_error"
                           or reason.startswith("verification_failed:") for reason in reasons))
        if unavailable:
            attempt_path = directory / "attempt-001.json"
            record = read_json(regular_file(attempt_path))
            if record.get("request_hash") != digest(request) or record.get("report") != exc.report:
                raise RenderBlocked("Speech recovery verification record changed") from exc
            record["report"]["retry_on_resume"] = True
            write_json(attempt_path, record)
            raise VerificationUnavailable(
                "Recovered speech verification was unavailable. Saved audio and verified parts "
                "were retained; rerun to retry verification. " + str(exc)) from exc
        raise SpeechVerificationFailed(
            "The separately verified speech parts failed final full-text verification. "
            "Automatic sentence recovery is exhausted; repeating the same assembly would not "
            f"produce a new take. Inspect {directory}", request=request, report=exc.report) from exc
