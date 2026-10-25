"""Content identity for generated speech, separate from its source provenance.

Callers must resolve saved voice records before requesting an identity. A legacy
asset hash alone cannot establish which reference audio and transcript were used.
Unknown engines retain their own identity and voice inputs conservatively.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import re

from .engines import _directed_text, _text, format_dialogue, prepare_synthesis
from .performance_compiler import performance_kwargs
from .speech_text import trim_speech_text


_DRAMA_ENGINES = frozenset({"drama3_mlx", "drama3_mlx_audio", "fish_http", "fish_tts_pool"})
_ALIAS_ENGINE = "drama_speech_cache_alias"
_ENGINE_PROVENANCE = frozenset({
    "checkpoint", "recommended_model_revision", "required_runtime_version",
    "required_runtime_revision", "packages", "min_audio_seconds", "completion_check",
    "multi_speaker", "runtime", "installed_version", "installed_revision", "adapter_version",
})


def synthesis_engine_identity(identity):
    """Retain generation behavior while ignoring installation and guard metadata.

    Runtime names, versions, and revisions remain request provenance only.
    Actual models, synthesis settings, and unknown fields remain meaningful
    unless the operator explicitly chooses a shared model-cache alias.
    Model content hashes supersede download revision metadata when available.
    Worker locations and scheduling are already omitted by the pool identity.
    """
    if not isinstance(identity, Mapping):
        raise ValueError("A synthesis engine identity must be an object")
    alias = _model_alias(identity)
    if alias is not None:
        return {"engine": _ALIAS_ENGINE, "cache_model_alias": alias,
                "sampling": _common_sampling(identity)}
    return _concrete_engine_identity(identity)


def _concrete_engine_identity(identity):
    result = deepcopy(dict(identity))
    if identity.get("engine") not in _DRAMA_ENGINES:
        return result
    result.pop("cache_model_alias", None)
    for field in _ENGINE_PROVENANCE:
        result.pop(field, None)
    if result.get("checkpoint_files_sha256"):
        result.pop("model_revision", None)
    return result


def _model_alias(engine):
    if engine.get("engine") not in _DRAMA_ENGINES | {_ALIAS_ENGINE}:
        return None
    alias = engine.get("cache_model_alias")
    return alias.strip() if isinstance(alias, str) and alias.strip() else None


def _common_sampling(engine):
    settings = (engine.get("sampling", {})
                if engine.get("engine") in {"fish_tts_pool", "fish_http", _ALIAS_ENGINE}
                else engine)
    if not isinstance(settings, Mapping):
        raise ValueError("Speech sampling settings must be an object")
    return {key: settings.get(key, value) for key, value in (
        ("temperature", 1.0), ("top_p", .9), ("top_k", 30), ("max_new_tokens", 2048))}


def engines_compatible(saved_engine, current_engine, *, engine_metadata=None):
    """Allow transport changes without weakening concrete local cache identity.

    A pool's pinned revision is the operator's existing compatibility contract
    for its workers. Cross-mode reuse uses that contract only for known models
    with matching sampling; local-to-local reuse still requires matching model
    contents and synthesis settings, independently of runtime metadata.
    Recorded concrete workers add stronger checks when returning to local mode.
    An alias on the current engine explicitly accepts model/quantization changes
    for that logical model, while retaining the common speech sampling settings.
    """
    if not isinstance(saved_engine, Mapping) or not isinstance(current_engine, Mapping):
        return False
    alias = _model_alias(current_engine)
    if alias is not None:
        if saved_engine.get("engine") not in _DRAMA_ENGINES | {_ALIAS_ENGINE}:
            return False
        # A saved alias takes precedence over its execution model. Falling back
        # to both would silently make unrelated aliases transitively equivalent.
        saved_label = _model_alias(saved_engine) or saved_engine.get("model")
        return alias == saved_label and _common_sampling(saved_engine) == _common_sampling(current_engine)
    saved, current = _concrete_engine_identity(saved_engine), _concrete_engine_identity(current_engine)
    if saved == current:
        return True
    saved_pool = saved_engine.get("engine") == "fish_tts_pool"
    current_pool = current_engine.get("engine") == "fish_tts_pool"
    if saved_pool == current_pool:
        return False
    pool = saved_engine if saved_pool else current_engine
    local = current_engine if saved_pool else saved_engine
    declared = pool_contract_identity(pool)
    actual = _pool_contract(local, require_revision=False)
    if declared is None or actual is None or not _same_pool_inputs(declared, actual):
        return False

    exact_worker = False
    if saved_pool and isinstance(engine_metadata, Mapping) and "worker_engine" in engine_metadata:
        worker = engine_metadata["worker_engine"]
        if not isinstance(worker, Mapping):
            return False
        if worker.get("engine") in {"drama3_mlx", "drama3_mlx_audio"}:
            exact_worker = _concrete_engine_identity(worker) == _concrete_engine_identity(local)
            worker_contract = _pool_contract(worker, require_revision=False)
            if (not exact_worker or worker_contract is None
                    or not _same_pool_inputs(declared, worker_contract)):
                return False
    return actual["model_revision"] is not None or exact_worker


def _same_pool_inputs(declared, actual):
    return (declared["model"] == actual["model"]
            and declared["sampling"] == actual["sampling"]
            and actual["model_revision"] in {None, declared["model_revision"]})


def _pool_contract(engine, *, require_revision):
    from .drama3_models import DRAMA3_MODELS

    if not isinstance(engine, Mapping):
        return None
    kind = engine.get("engine")
    if kind not in {"fish_tts_pool", "drama3_mlx", "drama3_mlx_audio"}:
        return None
    model = DRAMA3_MODELS.get(engine.get("model"))
    if model is None:
        return None
    if ((kind == "drama3_mlx" and model.runtime != "sogni")
            or (kind == "drama3_mlx_audio" and model.runtime != "mlx_audio")
            or (kind == "fish_tts_pool" and engine.get("contract_version", 1) != 1)):
        return None
    revision = engine.get("model_revision")
    if not isinstance(revision, str) or not revision.strip():
        if require_revision:
            return None
        revision = None
    settings = engine.get("sampling", {}) if kind == "fish_tts_pool" else engine
    if not isinstance(settings, Mapping):
        return None
    sampling = _common_sampling(engine)
    # MLX-Audio quantization is already part of the model ID; the Sogni experts
    # setting changes its model representation. min_audio_seconds is a gate.
    if model.runtime == "sogni":
        sampling["experts"] = settings.get("experts", "q8")
    return {"model": model.repo, "model_revision": revision, "sampling": sampling}


def pool_contract_identity(engine):
    """Project a supported engine onto its explicit pool compatibility contract.

    Unknown actual local revisions return None. Recommended download revisions
    never establish which weights produced a recording.
    """
    return _pool_contract(engine, require_revision=True)


def generation_compatible(saved_identity, current_identity, *, saved_engine=None,
                          current_engine=None, engine_metadata=None):
    """Compare audible inputs, permitting a compatible local/pool transition."""
    if not isinstance(saved_identity, Mapping) or not isinstance(current_identity, Mapping):
        return False
    if ({key: value for key, value in saved_identity.items() if key != "engine"}
            != {key: value for key, value in current_identity.items() if key != "engine"}):
        return False
    return engines_compatible(
        saved_identity.get("engine") if saved_engine is None else saved_engine,
        current_identity.get("engine") if current_engine is None else current_engine,
        engine_metadata=engine_metadata)


def voice_conditioning_identity(voice):
    """Identify the exact immutable reference consumed by known speech engines.

    VoiceStore verifies the checksum against the file when loading it. This
    helper deliberately performs no file access, allowing identities for old
    requests to be compared before a waveform is considered for reuse.
    """
    if not isinstance(voice, Mapping) or voice.get("kind") != "reference":
        raise ValueError("Speech identity requires a saved reference voice")
    checksum = voice.get("reference_sha256", voice.get("reference_audio_sha256"))
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("Speech identity requires the reference audio checksum")
    return {"kind": "reference", "reference_sha256": checksum,
            "reference_text": _text(voice.get("reference_text"), "reference_text")}


def _voice(asset, voices_by_asset):
    if not isinstance(asset, str) or not asset:
        raise ValueError("Speech request is missing its saved voice asset")
    voice = voices_by_asset.get(asset)
    if not isinstance(voice, Mapping):
        raise ValueError("The saved voice asset cannot be resolved for speech reuse")
    if voice.get("asset_hash", asset) != asset:
        raise ValueError("The resolved voice does not match the requested asset")
    return deepcopy(dict(voice))


def _direction(chunk, voice_id):
    segment = chunk.get("segment")
    if not isinstance(segment, Mapping):
        raise ValueError("Speech request is missing its segment")
    if "expressive_performance" in chunk:
        direction = chunk.get("expressive_direction", "")
        if not isinstance(direction, str):
            raise ValueError("Expressive passage direction must be text")
        return direction
    from .speech_traits import performance_direction
    return performance_direction(chunk, voice_id)


def _check_recorded_text(metadata, expected):
    # Earlier successful takes may include now-discarded edge formatting. The
    # same narrow preparation is compatible with that accepted performance;
    # interior punctuation, words, and delivery controls must still match.
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ValueError("Saved engine metadata must be an object")
        if "synthesis_text" in metadata:
            recorded = metadata["synthesis_text"]
            if (not isinstance(recorded, str) or
                    recorded not in expected and trim_speech_text(recorded) not in expected):
                raise ValueError("Recorded synthesis text differs from the prepared speech input")


def paragraph_rewrite_turn(request, voices_by_asset):
    """Return one equivalent turn only for an already uniform performance.

    This is a synthesis input, never a replacement source chunk. Exact voice
    assets and directions must agree, and the normal formatter must prove that
    collapsing the source turns changes neither the text nor the speaker tags.
    """
    if (request.get("kind") != "paragraph_speech"
            or request.get("engine", {}).get("engine") not in _DRAMA_ENGINES):
        return None
    chunk, assets = request.get("chunk", {}), request.get("voice_assets", {})
    turns = chunk.get("paragraph_turns")
    if not isinstance(turns, list) or len(turns) < 2 or not isinstance(assets, Mapping):
        return None
    if any(isinstance(turn, Mapping) and "expressive_performance" in turn for turn in turns):
        # Legacy rewrite callbacks cannot rebind scoped controls after changing
        # the words. The outer fallback can still retry intact original turns.
        return None
    prepared, bindings = [], []
    for turn in turns:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("segment"), Mapping):
            return None
        asset = assets.get(turn["segment"].get("voice_id"))
        direction = _direction(turn, turn["segment"].get("voice_id"))
        bindings.append((asset, direction))
        prepared.append({"text": _text(turn.get("spoken_text"), "text"),
                         "voice": _voice(asset, voices_by_asset), "direction": direction})
    if (any(binding != bindings[0] for binding in bindings[1:])
            or "".join(turn["text"] for turn in prepared) != chunk.get("spoken_text")):
        return None
    collapsed = dict(prepared[0], text=chunk["spoken_text"])
    original, _, original_metadata = format_dialogue(prepared)
    combined, _, combined_metadata = format_dialogue([collapsed])
    if (original != combined or original_metadata["rendered_turn_count"] != 1
            or original_metadata["speaker_count"] != 1
            or combined_metadata["rendered_turn_count"] != 1):
        return None
    return collapsed


def paragraph_recovery_part(request, part, voices_by_asset):
    """Bind one exact sentence slice to a uniform paragraph's saved performer."""
    from .speech_recovery import split_recovery_text

    collapsed = paragraph_rewrite_turn(request, voices_by_asset)
    parts = split_recovery_text(request.get("chunk", {}).get("spoken_text"))
    if collapsed is None or parts is None or part not in parts:
        raise ValueError("Paragraph sentence recovery requires an unchanged uniform performance")
    turn = deepcopy(request["chunk"]["paragraph_turns"][0])
    turn["spoken_text"] = part["text"]
    return {"kind": "speech", "engine": deepcopy(request["engine"]),
            "base_seed": request["base_seed"], "sample_rate": request["sample_rate"],
            "voice_asset_hash": collapsed["voice"]["asset_hash"], "chunk": turn}


def single_speech_recovery_part(request, part):
    """Bind an exact sentence to its original single speaker and generation settings."""
    from .speech_recovery import split_recovery_text

    if (not isinstance(request, Mapping) or request.get("kind") != "speech"
            or "recovery" in request or not isinstance(request.get("chunk"), Mapping)):
        raise ValueError("Single speech recovery requires its original speech request")
    parts = split_recovery_text(request["chunk"].get("spoken_text"))
    if parts is None or part not in parts:
        raise ValueError("Single speech recovery requires an exact original sentence slice")
    actual = deepcopy(dict(request))
    actual.pop("synthesis_identity", None)
    actual["chunk"]["spoken_text"] = part["text"]
    return actual


def _recovered_generation_identity(request, voices_by_asset, metadata):
    """Validate actual child inputs before reusing an independently verified join."""
    import hashlib
    import json
    from .common import digest
    from .speech_recovery import split_recovery_text

    parent = deepcopy(dict(request))
    recovery = parent.pop("recovery", None)
    parts = split_recovery_text(parent["chunk"].get("spoken_text"))
    audit = metadata.get("recovery")
    if (parts is None or recovery != {"version": 1, "strategy": "sentence_split", "parts": parts}
            or (parent["kind"] == "paragraph_speech"
                and paragraph_rewrite_turn(parent, voices_by_asset) is None)
            or (parent["kind"] == "speech" and (not isinstance(audit, Mapping)
                or type(audit.get("child_acceptance_version")) is not int
                or audit["child_acceptance_version"] != 1))
            or not isinstance(audit, Mapping) or audit.get("version") != 1
            or audit.get("strategy") != "sentence_split"
            or audit.get("parent_request_hash") != digest(parent)
            or not isinstance(audit.get("parts"), list) or len(audit["parts"]) != len(parts)
            or metadata.get("synthesis_text") != trim_speech_text(parent["chunk"]["spoken_text"])):
        raise ValueError("Invalid sentence recovery generation audit")
    for part, evidence in zip(parts, audit["parts"]):
        actual = (paragraph_recovery_part(parent, part, voices_by_asset)
                  if parent["kind"] == "paragraph_speech" else single_speech_recovery_part(parent, part))
        child = {"kind": "speech_recovery_part", "version": 1,
                 "parent_request_hash": digest(parent), "recovery_request_hash": digest(request),
                 "part": part}
        accepted = evidence.get("acceptance") if isinstance(evidence, Mapping) else None
        if not isinstance(accepted, Mapping):
            raise ValueError("Sentence recovery is missing child acceptance provenance")
        saved_child = accepted.get("request")
        if parent["kind"] == "paragraph_speech" or (isinstance(saved_child, Mapping)
                                                    and "synthesis_request" in saved_child):
            child["synthesis_request"] = actual
        encoded = (json.dumps(accepted, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        report = accepted.get("report", {})
        actual_metadata = accepted.get("engine_metadata")
        if (any(evidence.get(key) != value for key, value in part.items())
                or evidence.get("request_hash") != digest(child)
                or accepted.get("request") != child or accepted.get("request_hash") != digest(child)
                or evidence.get("accepted_sha256") != hashlib.sha256(encoded).hexdigest()
                or evidence.get("audio_sha256") != accepted.get("audio_sha256")
                or evidence.get("validation_key") != accepted.get("validation_key")
                or evidence.get("audio_file") != f"parts/{digest(child)}/{accepted.get('audio_file')}"
                or not isinstance(report, Mapping) or report.get("passed") is not True
                or report.get("reasons") or report.get("expected_text") != part["text"]
                or not isinstance(actual_metadata, Mapping)
                or (parent["engine"].get("engine") in _DRAMA_ENGINES
                    and not isinstance(actual_metadata.get("synthesis_text"), str))):
            raise ValueError("Sentence recovery child binding changed")
        from .exhaustion_selection import validate_record
        validate_record(accepted)
        generation_identity(actual, voices_by_asset, engine_metadata=actual_metadata)
    return generation_identity(parent, voices_by_asset)


def _resolved_generation_identity(request, voices_by_asset, metadata):
    """Validate an audited pronunciation input without relabelling its source.

    Speech acceptance still belongs to the verifier, against the original text.
    Once accepted, that performance can satisfy the original logical request;
    the engine metadata retains the exact alternate input and its resolver
    audit. Merely recording different synthesis text never enables this path.
    """
    from .speech_resolver import validate_resolution

    marker = metadata["speech_resolution"]
    chunk = request["chunk"]
    if (request["kind"] not in {"speech", "paragraph_speech"} or not isinstance(marker, Mapping)
            or set(marker) != {"version", "original_text", "tts_text", "audit", "failures"}
            or type(marker["version"]) is not int or marker["version"] != 1
            or marker["original_text"] != chunk.get("spoken_text")
            or not isinstance(marker["tts_text"], str) or not marker["tts_text"].strip()
            or marker["tts_text"] == marker["original_text"]
            or not isinstance(marker["audit"], Mapping)
            or not isinstance(metadata.get("synthesis_text"), str)
            or not metadata["synthesis_text"].strip()):
        raise ValueError("Invalid audited speech pronunciation input")
    payload = marker["audit"].get("payload")
    permitted = payload.get("permitted_actions") if isinstance(payload, Mapping) else None
    if not isinstance(permitted, list):
        raise ValueError("Speech pronunciation audit is missing its permitted actions")
    proposal = validate_resolution(
        marker["audit"], marker["original_text"], marker["failures"],
        marker["audit"].get("identity"), can_rewrite=True, can_split="split" in permitted)
    if (not isinstance(proposal, Mapping) or proposal.get("action") != "rewrite"
            or proposal.get("tts_text") != marker["tts_text"]):
        raise ValueError("Speech pronunciation input does not match its resolver audit")

    actual_request = deepcopy(dict(request))
    actual_request["chunk"]["spoken_text"] = marker["tts_text"]
    if request["kind"] == "paragraph_speech":
        collapsed = paragraph_rewrite_turn(request, voices_by_asset)
        if collapsed is None:
            raise ValueError("Paragraph pronunciation input changes its voice or delivery")
        _, _, expected_metadata = format_dialogue([dict(collapsed, text=marker["tts_text"])])
        for field in ("speaker_count", "turn_count", "rendered_turn_count", "turn_directions"):
            if metadata.get(field) != expected_metadata[field]:
                raise ValueError("Recorded paragraph pronunciation formatting differs from its audited input")
        turn = deepcopy(chunk["paragraph_turns"][0])
        turn["spoken_text"] = marker["tts_text"]
        actual_request["chunk"]["paragraph_turns"] = [turn]
    actual_request.pop("synthesis_identity", None)
    actual_metadata = dict(metadata)
    actual_metadata.pop("speech_resolution")
    # The normal engine formatter must independently reproduce the actual
    # synthesis input, including delivery controls and voice conditioning.
    # Do not substitute a logical identity for this check or weaken it to a
    # substring comparison merely because a resolver proposed the rewrite.
    generation_identity(actual_request, voices_by_asset, engine_metadata=actual_metadata)
    return generation_identity(request, voices_by_asset)


def _plain_delivery_generation_identity(request, voices_by_asset, metadata):
    """Check the recorded plain take before retaining the original intention.

    Delivery recovery is the outer layer: an independently audited pronunciation
    rewrite may also be present, and the normal rewrite path checks it against
    this neutral performance. Neither marker excuses changed words or voices.
    """
    from .delivery_recovery import validate_delivery_marker

    actual_request = validate_delivery_marker(request, metadata["delivery_recovery"])
    actual_request.pop("synthesis_identity", None)
    actual_metadata = dict(metadata)
    actual_metadata.pop("delivery_recovery")
    chunk = actual_request["chunk"]
    performer = actual_request.get("performer")
    voice_id = (performer.get("voice_id") if isinstance(performer, Mapping)
                else chunk.get("segment", {}).get("voice_id"))
    expected_direction = _direction(chunk, voice_id)
    _, expected_delivery = _directed_text(
        _text(trim_speech_text(_text(chunk.get("spoken_text"), "text")), "text"),
        expected_direction,
    )
    if (not isinstance(actual_metadata.get("synthesis_text"), str)
            or not actual_metadata["synthesis_text"].strip()
            or actual_metadata.get("direction", "") != expected_delivery["direction"]
            or actual_metadata.get("direction_applied", False) != expected_delivery["direction_applied"]
            or actual_metadata.get("applied_control") != expected_delivery["applied_control"]):
        raise ValueError("Recorded delivery does not confirm the audited plain performance")
    generation_identity(actual_request, voices_by_asset, engine_metadata=actual_metadata)
    return generation_identity(request, voices_by_asset)


def generation_identity(request, voices_by_asset=None, *, engine_metadata=None):
    """Return generation inputs without position, source IDs or cast bookkeeping.

    ``voices_by_asset`` binds request asset hashes to their exact saved records;
    unresolved legacy assets raise ValueError instead of guessing a current voice.
    ``engine_metadata`` optionally checks a legacy take's recorded actual input.
    The original request should remain saved separately as audit provenance.
    """
    if not isinstance(request, Mapping) or request.get("kind") not in {"speech", "paragraph_speech"}:
        raise ValueError("Generation identity requires a speech request")
    if voices_by_asset is None:
        voices_by_asset = {}
    if not isinstance(voices_by_asset, Mapping):
        raise ValueError("Saved voice assets must be a mapping")
    chunk = request.get("chunk")
    if not isinstance(chunk, Mapping):
        raise ValueError("Speech request is missing its chunk")
    if isinstance(engine_metadata, Mapping) and 'exhaustion_selection' in engine_metadata:
        from .exhaustion_selection import selected_generation_identity
        return selected_generation_identity(request, voices_by_asset, engine_metadata)
    if isinstance(engine_metadata, Mapping) and 'clip_repair' in engine_metadata:
        from .clip_repair_contract import confirm_plan_metadata
        confirm_plan_metadata(request, engine_metadata)
        return generation_identity(request, voices_by_asset)
    if (isinstance(engine_metadata, Mapping) and "recovery" in engine_metadata
            and (request["kind"] == "paragraph_speech"
                 or (isinstance(engine_metadata["recovery"], Mapping)
                     and "child_acceptance_version" in engine_metadata["recovery"]))):
        return _recovered_generation_identity(request, voices_by_asset, engine_metadata)
    if isinstance(engine_metadata, Mapping) and "delivery_recovery" in engine_metadata:
        return _plain_delivery_generation_identity(request, voices_by_asset, engine_metadata)
    if isinstance(engine_metadata, Mapping) and "direction_repair" in engine_metadata:
        from .direction_repair import confirm_metadata
        confirm_metadata(request, engine_metadata)
        return generation_identity(request, voices_by_asset)
    if isinstance(engine_metadata, Mapping) and "fragment_recovery" in engine_metadata:
        from .fragment_provenance import fragment_generation_identity

        return fragment_generation_identity(request, voices_by_asset, engine_metadata)
    if isinstance(engine_metadata, Mapping) and "speech_resolution" in engine_metadata:
        return _resolved_generation_identity(request, voices_by_asset, engine_metadata)
    engine = request.get("engine")
    result = {"version": 1, "kind": request["kind"],
              "engine": synthesis_engine_identity(engine),
              "base_seed": request["base_seed"], "sample_rate": request["sample_rate"]}
    known = engine.get("engine") in _DRAMA_ENGINES
    if request["kind"] == "speech":
        voice = _voice(request.get("voice_asset_hash"), voices_by_asset)
        text = _text(trim_speech_text(_text(chunk.get("spoken_text"), "text")), "text")
        performer = request.get("performer")
        voice_id = (performer.get("voice_id") if isinstance(performer, Mapping)
                    else chunk.get("segment", {}).get("voice_id"))
        direction = _direction(chunk, voice_id)
        expressive = performance_kwargs(chunk, voice_id)
        if known:
            text, directed, delivery = prepare_synthesis(chunk.get("spoken_text"), direction, **expressive)
            result.update(text=directed, voice=voice_conditioning_identity(voice))
            if expressive:
                # The request and engine metadata retain the full compiler
                # audit. Cache identity follows actual conditioning: adding a
                # plan that compiles to identical text must reuse accepted audio.
                if (engine_metadata is not None
                        and engine_metadata.get("expressive_performance") != delivery["expressive_performance"]):
                    raise ValueError("Recorded expressive performance differs from the prepared speech input")
            # Local workflow records the trimmed target before controls; HTTP
            # records the final directed text. Both describe this exact input.
            _check_recorded_text(engine_metadata, (text, directed))
            if (engine_metadata is not None and delivery["applied_control"] == "style_prefix"
                    and engine_metadata.get("synthesis_text") != directed
                    and not (engine_metadata.get("applied_control") == "style_prefix"
                             and engine_metadata.get("direction") == direction)):
                raise ValueError("Saved speech does not confirm the requested delivery style")
            if (engine_metadata is not None and "applied_control" in engine_metadata
                    and engine_metadata["applied_control"] != delivery["applied_control"]):
                raise ValueError("Recorded delivery differs from the prepared speech input")
        else:
            if expressive:
                raise ValueError("Generation identity has no structured-performance compiler for this engine")
            result.update(text=text, direction=direction, voice=voice)
            _check_recorded_text(engine_metadata, (text,))
        return result

    chunks, assets = chunk.get("paragraph_turns"), request.get("voice_assets")
    if not isinstance(chunks, list) or not chunks or not isinstance(assets, Mapping):
        raise ValueError("Paragraph identity requires ordered turns and saved voices")
    turns = []
    for turn in chunks:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("segment"), Mapping):
            raise ValueError("Paragraph contains an invalid turn")
        voice = _voice(assets.get(turn["segment"].get("voice_id")), voices_by_asset)
        turns.append({"text": _text(turn.get("spoken_text"), "text"),
                      "direction": _direction(turn, turn["segment"].get("voice_id")), "voice": voice,
                      **performance_kwargs(turn, turn["segment"].get("voice_id"))})
    if known:
        # Use the rendering formatter itself: edge whitespace can affect joins,
        # and request-local speaker numbering depends on first-use voice order.
        text, ordered_voices, formatted = format_dialogue(turns)
        result.update(text=text, references=[voice_conditioning_identity(v) for v in ordered_voices])
        if any("performance" in turn for turn in turns):
            if (engine_metadata is not None
                    and engine_metadata.get("turn_directions") != formatted["turn_directions"]):
                raise ValueError("Recorded dialogue expressive controls differ from the prepared input")
        _check_recorded_text(engine_metadata, (text,))
        for field in ("speaker_count", "rendered_turn_count"):
            if (engine_metadata is not None and field in engine_metadata
                    and engine_metadata[field] != formatted[field]):
                raise ValueError("Recorded dialogue grouping differs from the prepared speech input")
    else:
        if any("performance" in turn for turn in turns):
            raise ValueError("Generation identity has no structured-performance compiler for this engine")
        result["turns"] = turns
    return result
