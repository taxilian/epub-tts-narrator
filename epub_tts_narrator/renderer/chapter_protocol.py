"""Versioned, path-free inputs for asynchronous chapter rendering."""

from __future__ import annotations

from copy import deepcopy
import json
import re

from .common import digest
from .config import RENDER_CONTRACT


PROTOCOL = "epub-tts-narrator.chapter-jobs.v1"
HASH = re.compile(r"[0-9a-f]{64}\Z")
SAFE_RENDER_KEYS = frozenset(
    key for key, spec in RENDER_CONTRACT.items() if spec.get("remote_safe"))
ARTIFACT_NAMES = frozenset({
    "chapter.m4a", "chapter.flac", "chapter.json", "sync.json", "verification.json", "manifest.json",
})
FAILED_PARAGRAPH_FIELD = "failed_paragraph="
_CHUNK_ID = re.compile(r"[0-9a-f]{32,64}\Z")


def require_hash(value, label="SHA256"):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def validate_voice_manifest(manifest):
    if (not isinstance(manifest, dict) or set(manifest) != {"version", "audio_sha256", "reference_text"}
            or type(manifest.get("version")) is not int or manifest["version"] != 1):
        raise ValueError("Invalid voice reference manifest")
    require_hash(manifest["audio_sha256"], "Voice audio hash")
    text = manifest["reference_text"]
    if not isinstance(text, str) or not text.strip() or len(text) > 20000 or "\0" in text:
        raise ValueError("Voice reference needs its exact, nonempty transcript")
    return deepcopy(manifest)


def failed_paragraph_detail(exc):
    """Hash-only identity of a failed paragraph synthesis, safe for wire errors.

    Book text never enters job records or logs; the client matches these
    identifiers against its own plan to recover the paragraph remotely.
    """
    request = getattr(exc, "request", None)
    if not isinstance(request, dict) or request.get("kind") != "paragraph_speech":
        return None
    chunk = request.get("chunk")
    if not isinstance(chunk, dict) or "paragraph_turns" not in chunk:
        return None
    identifier = chunk.get("chunk_id")
    if not isinstance(identifier, str) or not _CHUNK_ID.fullmatch(identifier):
        return None
    return {"request_hash": digest(request), "chunk_id": identifier}


def failed_paragraph_suffix(exc):
    """Return the durable wire suffix for a paragraph speech failure, if any."""
    detail = failed_paragraph_detail(exc)
    if detail is None:
        return ""
    return "\n" + FAILED_PARAGRAPH_FIELD + json.dumps(detail, sort_keys=True)


def validate_render_settings(settings):
    """Do not allow clients to choose server executables or filesystem paths."""
    from .chapter_cache import chapter_settings
    if not isinstance(settings, dict) or set(settings) - SAFE_RENDER_KEYS:
        raise ValueError("Chapter rendering contains unsupported settings")
    chapter_settings(settings)
    if settings.get("chapter_format", "m4a") != "m4a":
        raise ValueError("Remote chapters use M4A; the final audiobook uses M4B")
    for key, spec in RENDER_CONTRACT.items():
        if key not in settings or spec.get("checked"):
            continue
        kind = spec["kind"]
        if kind == "int":
            low, high = spec["range"]
            if type(settings[key]) is not int or not low <= settings[key] <= high:
                raise ValueError(f"Invalid remote render.{key}")
        elif kind == "bool":
            if type(settings[key]) is not bool:
                raise ValueError(f"Invalid remote render.{key}")
        elif kind == "enum" and settings[key] not in spec["choices"]:
            raise ValueError(f"Invalid remote render.{key}")
        elif kind == "bitrate" and (not isinstance(settings[key], str)
                                    or not re.fullmatch(r"[1-9][0-9]{0,2}k", settings[key])):
            raise ValueError(f"Invalid remote render.{key}")
    return deepcopy(settings)


def _leaf(chunk, bindings):
    from .bundle import chunk_speakers
    from epub_tts_narrator.spoken_text import build_spoken, effective_delivery, spoken_source_ranges
    from .quality import normalize_words
    row = chunk.get("segment")
    if not isinstance(row, dict):
        raise ValueError("A chapter passage needs a source segment")
    for key in ("id", "block_id", "chapter_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Missing source segment {key}")
    text, start, end = row.get("text"), row.get("start"), row.get("end")
    if (not isinstance(text, str) or type(start) is not int or type(end) is not int
            or not 0 <= start <= end or end - start != len(text)):
        raise ValueError("Invalid source segment text or offsets")
    if "speech_normalization" in row and "spoken_text" not in row:
        raise ValueError("Normalized speech requires its saved spoken text")
    spoken, _ = build_spoken(row)
    a, b = chunk.get("spoken_start"), chunk.get("spoken_end")
    if (type(a) is not int or type(b) is not int or not 0 <= a <= b <= len(spoken)
            or chunk["spoken_text"] != spoken[a:b]
            or chunk.get("source_ranges") != spoken_source_ranges(row, a, b)):
        raise ValueError("Chapter speech must map exactly to its source text")
    if chunk.get("silent_control") is not (not normalize_words(chunk["spoken_text"])):
        raise ValueError("Invalid silent passage marker")
    speakers = chunk_speakers(chunk)
    if any(not isinstance(speaker[key], str) or not speaker[key]
           for speaker in speakers for key in ("voice_id", "role_id")):
        raise ValueError("Chapter performers need nonempty role and voice IDs")
    if not chunk["silent_control"] and (not speakers or any(s["voice_id"] not in bindings for s in speakers)):
        raise ValueError("Every spoken passage needs a registered voice")
    traits_by_voice = chunk.get("speech_traits")
    if traits_by_voice is not None:
        speaker_ids = {speaker["voice_id"] for speaker in speakers}
        if (not isinstance(traits_by_voice, dict) or not traits_by_voice
                or any(not isinstance(voice_id, str) or voice_id not in speaker_ids
                       for voice_id in traits_by_voice)):
            raise ValueError("Speech traits must map passage voice IDs to stable traits")
        from ..speech_traits import validate_speech_traits
        for traits in traits_by_voice.values():
            validate_speech_traits(traits)
    if speakers and "expressive_performance" not in chunk:
        from .speech_traits import performance_direction
        for speaker in speakers:
            performance_direction(chunk, speaker["voice_id"])
    elif not speakers:
        effective_delivery(row)
    if not isinstance(row.get("performance", {}), dict):
        raise ValueError("Invalid passage performance")
    expressive = {"expressive_performance", "expressive_direction", "expressive_traits"}
    if expressive.intersection(chunk):
        if not expressive.issubset(chunk):
            raise ValueError("Expressive passages need a bound plan, direction, and traits")
        from ..performance_plan import validate_performance, plain_direction
        validate_performance(chunk["expressive_performance"], chunk["spoken_text"])
        plain_direction(chunk["expressive_direction"])
        scene = deepcopy(row)
        scene.pop("speech_normalization", None)
        if chunk["expressive_direction"] != effective_delivery(scene):
            raise ValueError("Expressive scene direction differs from the source segment")
        if chunk["expressive_traits"] != chunk.get("speech_traits", {}):
            raise ValueError("Expressive traits differ from the bound voice traits")


def _chunk_shape(chunk, identifiers, *, paragraph=False):
    required = {"chunk_id", "segment", "spoken_text", "spoken_start", "spoken_end",
                "source_ranges", "silent_control"}
    allowed = required | {"speech_traits", "expressive_performance", "expressive_direction", "expressive_traits"} | ({"paragraph_turns"} if paragraph else set())
    if (not isinstance(chunk, dict) or set(chunk) - allowed or required - set(chunk)
            or not isinstance(chunk.get("chunk_id"), str)
            or not re.fullmatch(r"[0-9a-f]{32,64}", chunk["chunk_id"])
            or chunk["chunk_id"] in identifiers or not isinstance(chunk.get("spoken_text"), str)
            or len(chunk["spoken_text"]) > 12000):
        raise ValueError("Invalid or duplicate chapter passage")
    identifiers.add(chunk["chunk_id"])


def validate_spec(spec):
    """Validate the transport contract without trusting client-side preflight."""
    required = {"version", "book", "source_fingerprint", "chapter", "total_chapters", "voice_bindings", "render", "renderer_fingerprint"}
    if (not isinstance(spec, dict) or set(spec) - required - {"censorship_policy"}
            or required - set(spec) or type(spec["version"]) is not int or spec["version"] != 1):
        raise ValueError("Invalid chapter job specification")
    if "censorship_policy" in spec:
        from .censorship import validate_policy
        policy = validate_policy(spec["censorship_policy"])
        if not policy["enabled"] or policy != spec["censorship_policy"]:
            raise ValueError("Remote censorship policy must be enabled and canonical")
    # Also rejects non-finite numbers and values that cannot cross the JSON wire.
    digest(spec)
    require_hash(spec["source_fingerprint"], "Source fingerprint")
    require_hash(spec["renderer_fingerprint"], "Renderer fingerprint")
    book = spec["book"]
    if not isinstance(book, dict) or not isinstance(book.get("title"), str) or not book["title"]:
        raise ValueError("Chapter job requires book metadata")
    authors = book.get("authors", [])
    if ("\0" in book["title"] or not isinstance(authors, list)
            or any(not isinstance(author, str) or "\0" in author for author in authors)):
        raise ValueError("Invalid book title or authors")
    bindings = spec["voice_bindings"]
    if (not isinstance(bindings, dict) or len(bindings) > 2048
            or any(not isinstance(key, str) or not key for key in bindings)):
        raise ValueError("Invalid voice bindings")
    for asset_id in bindings.values():
        require_hash(asset_id, "Voice asset")
    chapter = spec["chapter"]
    if not isinstance(chapter, dict):
        raise ValueError("Invalid chapter")
    number, total = chapter.get("number"), spec["total_chapters"]
    if (type(number) is not int or type(total) is not int or not 1 <= number <= total <= 100000
            or chapter.get("id") != f"chapter-{number:04d}"
            or not isinstance(chapter.get("title"), str) or "\0" in chapter["title"]):
        raise ValueError("Invalid chapter number, ID, or title")
    chunks = chapter.get("chunks")
    if not isinstance(chunks, list) or len(chunks) > 50000:
        raise ValueError("Invalid chapter passage list")
    ids = set()
    for chunk in chunks:
        _chunk_shape(chunk, ids, paragraph=True)
        turns = chunk.get("paragraph_turns")
        if turns is None:
            if "paragraph_turns" in chunk:
                raise ValueError("Invalid paragraph performance")
            _leaf(chunk, bindings)
        else:
            if (not isinstance(turns, list) or not 2 <= len(turns) <= 1000
                    or chunk.get("silent_control") is not False):
                raise ValueError("Invalid paragraph performance")
            for turn in turns:
                _chunk_shape(turn, ids)
                _leaf(turn, bindings)
                if (turn["silent_control"] or "simultaneous_speakers" in turn["segment"]
                        or turn["segment"].get("performance", {}).get("kind") == "bracketed"):
                    raise ValueError("A paragraph turn must contain ordinary single-speaker text")
            if len({turn["segment"]["voice_id"] for turn in turns}) > 16:
                raise ValueError("A paragraph performance supports at most 16 voices")
            if "".join(turn["spoken_text"] for turn in turns) != chunk["spoken_text"]:
                raise ValueError("Paragraph text differs from its source turns")
            row = chunk.get("segment")
            if (not isinstance(row, dict) or row.get("text") != chunk["spoken_text"]
                    or len({turn["segment"]["block_id"] for turn in turns}) != 1
                    or len({turn["segment"]["chapter_id"] for turn in turns}) != 1
                    or row.get("block_id") != turns[0]["segment"]["block_id"]):
                raise ValueError("Paragraph crosses source blocks")
            # Reconstruct exactly the planner's paragraph, including flattened
            # source ranges, offsets and grouping identity. Checking only the
            # concatenated spoken text permits plausible speech with false sync.
            from .paragraphs import _group
            expected = _group(turns)
            if len(expected) != 1 or expected[0] != chunk:
                raise ValueError("Paragraph timing and source mapping differ from its source turns")
    validate_render_settings(spec["render"])
    return deepcopy(spec)
