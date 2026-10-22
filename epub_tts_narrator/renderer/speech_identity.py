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
_ENGINE_PROVENANCE = frozenset({
    "checkpoint", "recommended_model_revision", "required_runtime_version",
    "required_runtime_revision", "packages", "min_audio_seconds", "completion_check",
    "multi_speaker",
})


def synthesis_engine_identity(identity):
    """Retain generation behavior while ignoring installation and guard metadata.

    Actual runtime/adapter versions and all unknown fields remain meaningful.
    Model content hashes supersede download revision metadata when available.
    Worker locations and scheduling are already omitted by the pool identity.
    """
    if not isinstance(identity, Mapping):
        raise ValueError("A synthesis engine identity must be an object")
    result = deepcopy(dict(identity))
    if identity.get("engine") not in _DRAMA_ENGINES:
        return result
    for field in _ENGINE_PROVENANCE:
        result.pop(field, None)
    if result.get("checkpoint_files_sha256"):
        result.pop("model_revision", None)
    return result


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
