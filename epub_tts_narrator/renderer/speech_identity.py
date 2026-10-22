"""Content identity for generated speech, separate from its source provenance.

Callers must resolve saved voice records before requesting an identity. A legacy
asset hash alone cannot establish which reference audio and transcript were used.
Unknown engines retain their own identity and voice inputs conservatively.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
import re

from .engines import _directed_text, _text, format_dialogue
from .speech_text import trim_speech_text


_DRAMA_ENGINES = frozenset({"drama3_mlx", "drama3_mlx_audio", "fish_http", "fish_tts_pool"})
_ALIAS_ENGINE = "drama_speech_cache_alias"
_ENGINE_PROVENANCE = frozenset({
    "checkpoint", "recommended_model_revision", "required_runtime_version",
    "required_runtime_revision", "packages", "min_audio_seconds", "completion_check",
    "multi_speaker",
})


def synthesis_engine_identity(identity):
    """Retain generation behavior while ignoring installation and guard metadata.

    Actual runtime/adapter versions and all unknown fields remain meaningful
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
    with matching sampling; local-to-local reuse remains content/runtime exact.
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


def _direction(chunk):
    segment = chunk.get("segment")
    if not isinstance(segment, Mapping):
        raise ValueError("Speech request is missing its segment")
    delivery = segment.get("delivery", {})
    if not isinstance(delivery, Mapping) or not isinstance(delivery.get("direction", ""), str):
        raise ValueError("Speech delivery must contain a direction string")
    return delivery.get("direction", "")


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
    engine = request.get("engine")
    result = {"version": 1, "kind": request["kind"],
              "engine": synthesis_engine_identity(engine),
              "base_seed": request["base_seed"], "sample_rate": request["sample_rate"]}
    known = engine.get("engine") in _DRAMA_ENGINES
    if request["kind"] == "speech":
        voice = _voice(request.get("voice_asset_hash"), voices_by_asset)
        text = _text(trim_speech_text(_text(chunk.get("spoken_text"), "text")), "text")
        direction = _direction(chunk)
        if known:
            directed, delivery = _directed_text(text, direction)
            result.update(text=directed, voice=voice_conditioning_identity(voice))
            # Local workflow records the trimmed target before controls; HTTP
            # records the final directed text. Both describe this exact input.
            _check_recorded_text(engine_metadata, (text, directed))
            if (engine_metadata is not None and "applied_control" in engine_metadata
                    and engine_metadata["applied_control"] != delivery["applied_control"]):
                raise ValueError("Recorded delivery differs from the prepared speech input")
        else:
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
                      "direction": _direction(turn), "voice": voice})
    if known:
        # Use the rendering formatter itself: edge whitespace can affect joins,
        # and request-local speaker numbering depends on first-use voice order.
        text, ordered_voices, formatted = format_dialogue(turns)
        result.update(text=text, references=[voice_conditioning_identity(v) for v in ordered_voices])
        _check_recorded_text(engine_metadata, (text,))
        for field in ("speaker_count", "rendered_turn_count"):
            if (engine_metadata is not None and field in engine_metadata
                    and engine_metadata[field] != formatted[field]):
                raise ValueError("Recorded dialogue grouping differs from the prepared speech input")
    else:
        result["turns"] = turns
    return result
