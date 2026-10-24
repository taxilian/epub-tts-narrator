"""One bounded cutoff-tail fragmentation level for a failed recovery part.

A failing recovery part that still loses its final word is split once at its
final em-dash boundary; the short exact-source tail and the exact prefix are
generated and verified separately, and only the full concatenation verified
against the original part text can satisfy it. Pieces are never split again.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re

from .common import digest, file_hash, read_json, regular_file, write_json
from .speech_text import trim_speech_text


_VERSION = 1
_STRATEGY = "cutoff_tail"
_MARKER = "speech-fragment-recovery.json"
_MAX_TAIL_WORDS = 3
_MIN_PREFIX_WORDS = 2
_EMDASH = re.compile(r"—[\"'”’)]*\s*")
_BINDING = ("kind", "version", "parent_request_hash", "recovery_request_hash", "part")


def fragment_plan(text):
    """Return two exact slices when the final em-dash leaves a one-to-three-word tail."""
    if not isinstance(text, str) or not text.strip() or len(text) > 100000:
        return None
    addresses = [token for token in re.finditer(r"\S+", text)
                 if any(marker in token[0] for marker in ("://", "www.", "@"))]
    boundary = None
    for match in _EMDASH.finditer(text):
        if match.end() < len(text) and not any(
                address.start() <= match.start() < address.end() for address in addresses):
            boundary = match.end()
    if boundary is None:
        return None
    from .quality import normalize_words

    tail_words, prefix_words = (len(normalize_words(half)) for half in (text[boundary:], text[:boundary]))
    if not 1 <= tail_words <= _MAX_TAIL_WORDS or prefix_words < _MIN_PREFIX_WORDS:
        return None
    return [{"start": 0, "end": boundary, "text": text[:boundary]},
            {"start": boundary, "end": len(text), "text": text[boundary:]}]


def _bound_single_part(request, text):
    """Return whether this is a bound single-speech recovery part for this text."""
    synthesis = request.get("synthesis_request") if isinstance(request, Mapping) else None
    part = request.get("part") if isinstance(request, Mapping) else None
    return (isinstance(request, Mapping) and request.get("kind") == "speech_recovery_part"
            and request.get("version") == 1 and "recovery" not in request
            and isinstance(synthesis, Mapping) and synthesis.get("kind") == "speech"
            and "recovery" not in synthesis
            and synthesis.get("chunk", {}).get("spoken_text") == text
            and isinstance(part, Mapping) and part.get("text") == text)


def fragment_requests(text, child_request, config):
    """Build the fragment-level request and its hash-bound persisted marker."""
    if not _bound_single_part(child_request, text):
        return None
    parts = fragment_plan(text)
    if parts is None:
        return None
    maximum = config["render"]["max_attempts"]
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise ValueError("Speech fragment recovery requires one to ten attempts per piece")
    recovery = {"version": _VERSION, "strategy": _STRATEGY, "parts": parts}
    fragment_request = {**deepcopy(dict(child_request)), "recovery": recovery}
    body = {"format": "audiobook-fragment-recovery", "version": _VERSION,
            "parent_request_hash": digest(child_request), "fragment_request_hash": digest(fragment_request),
            "recovery": recovery}
    return fragment_request, {**body, "record_sha256": digest(body)}


def _existing_marker(directory, marker):
    path = directory / _MARKER
    if not path.exists() and not path.is_symlink():
        return False
    if read_json(regular_file(path)) != marker:
        raise ValueError("Saved speech fragment recovery does not match the current part and plan")
    return True


def fragment_started(directory, child_request, text, config):
    """Return whether this part already has a persisted cutoff-tail strategy."""
    from .speech_recovery import _directory
    from .workflow import RenderBlocked

    try:
        path = _directory(directory) / _MARKER
        if not path.exists() and not path.is_symlink():
            return False
        planned = fragment_requests(text, child_request, config)
        if planned is None:
            raise ValueError("Saved speech fragment recovery has no matching eligible part request")
        return _existing_marker(directory, planned[1])
    except (ValueError, OSError) as exc:
        raise RenderBlocked(f"Invalid speech fragment recovery: {exc}") from exc


def fragment_part_take(text, generate, directory, request, verifier, config, *, notify=print,
                       retry_budget=None, synthesize_part=None, generate_plain=None,
                       resolution_episode=None, episode_reserve=0, allow_fragments=False):
    """Verify one recovery part; engage the bounded cutoff-tail strategy only on a definite tail loss."""
    from .workflow import SpeechVerificationFailed, verified_take

    engaged = (allow_fragments and getattr(verifier, "speech_resolver", None) is not None
               and _bound_single_part(request, text))
    # When the tail strategy is available, the shared episode reserves room for its two pieces.
    fallback = len(fragment_plan(text) or []) if engaged else 0
    if engaged and fragment_started(directory, request, text, config):
        notify("  Resuming the persisted cutoff-tail recovery for this part; "
               "whole-part takes stay retired.")
        return _recover_fragment(text, synthesize_part, generate_plain, directory, request,
                                 verifier, config, notify=notify, retry_budget=retry_budget,
                                 resolution_episode=resolution_episode, episode_reserve=episode_reserve)
    try:
        return verified_take(text, generate, directory, request, verifier, config, notify=notify,
                             purpose="speech", retry_budget=retry_budget,
                             generate_repaired=synthesize_part, generate_plain=generate_plain,
                             resolution_episode=resolution_episode, episode_reserve=episode_reserve,
                             fallback_parts=fallback)
    except SpeechVerificationFailed as exc:
        from .speech_resolver import _ACOUSTIC_REASONS
        reasons = set(exc.report.get("reasons", []))
        episode = resolution_episode or request.get("parent_request_hash", digest(request))
        remaining = (config.get("verification", {}).get("max_resolution_attempts", 12)
                     - retry_budget.episode_started(episode)) if retry_budget is not None else 0
        if (not engaged or exc.request != request
                or not reasons or not reasons <= _ACOUSTIC_REASONS | {"text_mismatch"}
                or remaining < 2 + episode_reserve
                or fragment_requests(text, request, config) is None):
            raise
    notify("  The final words were cut off again. Splitting this part once at its final em-dash "
           "tail; every fragment keeps the original source, voice, and full verification.")
    return _recover_fragment(text, synthesize_part, generate_plain, directory, request,
                             verifier, config, notify=notify, retry_budget=retry_budget,
                             resolution_episode=resolution_episode, episode_reserve=episode_reserve)


def _piece_request(child, fragment_request, piece, *, synthesis=None):
    actual = deepcopy(child["synthesis_request"] if synthesis is None else synthesis)
    actual.pop("synthesis_identity", None)
    actual["chunk"]["spoken_text"] = piece["text"]
    return actual, {"kind": "speech_recovery_part", "version": _VERSION,
                    "parent_request_hash": digest(child),
                    "recovery_request_hash": digest(fragment_request),
                    "part": piece, "synthesis_request": actual}


def _recover_fragment(text, synthesize_part, synthesize_plain, part_directory, child_request,
                      verifier, config, *, notify=print, retry_budget=None, resolution_episode=None,
                      episode_reserve=0):
    """Verify exact prefix and tail pieces, then the complete joined audio."""
    from .retry_budget import generation_attempts
    from .speech_recovery import _concatenate, _directory
    from .workflow import (RenderBlocked, SpeechRetryBudget, SpeechVerificationFailed,
                           VerificationUnavailable, verified_take)
    from .fragment_provenance import reject_exhausted_join, inherited_delivery

    try:
        planned = fragment_requests(text, child_request, config)
        if planned is None:
            raise ValueError("Speech has no bounded cutoff-tail fragment boundary")
        fragment_request, marker = planned
        directory = _directory(part_directory)
        if not _existing_marker(directory, marker):
            write_json(directory / _MARKER, marker)
    except (ValueError, OSError) as exc:
        raise RenderBlocked(f"Could not prepare speech fragment recovery: {exc}") from exc
    reject_exhausted_join(directory, child_request)
    delivery, synthesis = inherited_delivery(directory, child_request, verifier, config, synthesize_plain)
    synthesize = synthesize_plain if delivery is not None else synthesize_part
    episode = resolution_episode or child_request.get("parent_request_hash", digest(child_request))
    pieces = fragment_request["recovery"]["parts"]
    verified, evidence = [], []
    for number, piece in enumerate(pieces, 1):
        actual, piece_request = _piece_request(child_request, fragment_request, piece, synthesis=synthesis)
        piece_directory = _directory(directory / "fragments" / digest(piece_request))

        def generate_piece(output, piece_seed, spoken=piece["text"]):
            return synthesize(spoken, output, piece_seed)

        notify(f"  Fragment {number}/{len(pieces)}: checking saved audio.")
        wav, record = verified_take(piece["text"], generate_piece, piece_directory, piece_request,
                                    verifier, config, notify=notify, purpose="speech",
                                    retry_budget=retry_budget, generate_repaired=synthesize,
                                    generate_plain=synthesize_plain,
                                    resolution_episode=episode,
                                    episode_reserve=len(pieces) - number + episode_reserve)
        verified.append((wav, record))
        evidence.append({**piece, "request_hash": record["request_hash"],
                         "audio_sha256": record["audio_sha256"], "validation_key": record["validation_key"],
                         "accepted_sha256": file_hash(wav.parent / "accepted.json"),
                         "audio_file": f"fragments/{digest(piece_request)}/{wav.name}",
                         "acceptance": deepcopy(record)})

    def assemble(path, seed):
        _concatenate(verified, path, config["render"]["sample_rate"])
        audit = {"version": _VERSION, "strategy": _STRATEGY, "request_hash": digest(child_request),
                 "binding": {key: child_request[key] for key in _BINDING},
                 "parts": deepcopy(pieces), "evidence": evidence, "child_acceptance_version": 1}
        if delivery is not None:
            audit["delivery_recovery"] = delivery
        return {"synthesis_text": trim_speech_text(text), "fragment_recovery": audit}

    # The join only assembles verified pieces; it never shares generation starts.
    assembly_config = deepcopy(config)
    assembly_config["render"]["max_attempts"] = 1
    expected_attempt = max(generation_attempts(directory), default=0) + 1
    try:
        return verified_take(text, assemble, directory, child_request, verifier, assembly_config,
                             notify=notify, purpose="speech", retry_budget=SpeechRetryBudget(),
                             allow_resolution=False)
    except SpeechVerificationFailed as exc:
        reasons = exc.report.get("reasons", [])
        unavailable = (exc.report.get("retryable") is not False and reasons
                       and all(reason == "generation_or_verification_error"
                               or reason.startswith("verification_failed:") for reason in reasons))
        if not unavailable:
            raise SpeechVerificationFailed(
                "The separately verified fragment pieces failed final full-text verification. "
                "Cutoff-tail recovery is exhausted; repeating the same assembly would not produce "
                f"a new take. Inspect {directory}", request=child_request, report=exc.report) from exc
        attempt_path = directory / f"attempt-{expected_attempt:03d}.json"
        record = read_json(regular_file(attempt_path))
        if record.get("request_hash") != digest(child_request) or record.get("report") != exc.report:
            raise RenderBlocked("Speech fragment recovery verification record changed") from exc
        record["report"]["retry_on_resume"] = True
        write_json(attempt_path, record)
        raise VerificationUnavailable(
            "Fragment recovery verification was unavailable. Saved audio and verified pieces "
            "were retained; rerun to retry verification. " + str(exc)) from exc
