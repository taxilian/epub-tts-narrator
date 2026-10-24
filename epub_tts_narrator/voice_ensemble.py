"""One whole-cast voice design pass over the fully resolved analysis cast.

Runs once in ``analyze`` after identity consensus and series reconciliation,
before attribution, so the pass sees the final cast — including identity splits
and finalized inherited voice ownership — and downstream stages fingerprint one
coherent cast. Established inherited profiles are reference-only: the pass can
coordinate local creative choices but never touches them. The deterministic
contrast audit in :mod:`voice_contrast` supplies advisory findings captured
before the pass and recomputed on the final cast; a cast with fewer than two
distinct voices, or with nothing designable, makes no request.
"""

from copy import deepcopy
import json
import unicodedata

from .pipeline import check_cast, fingerprint
from .speech_traits import validate_speech_traits
from .voice_contrast import VERSION as CONTRAST_VERSION, contrast_findings
from .voice_ensemble_prompts import ENSEMBLE_SYSTEM
from .voice_prompt_refresh import _record_context


VERSION = 1
PROTECTED_ORIGIN = "series_casting_reuse"
UPSERT_FIELDS = {"id", "description", "speechTraits", "traits"}
TRAIT_FIELDS = {"name", "value", "basis", "evidence"}


def _inherited(voice):
    origin = voice.get("series_origin")
    return isinstance(origin, dict) and origin.get("kind") == PROTECTED_ORIGIN


def _trait_is_text(trait):
    if trait.get("basis") == "text":
        return True
    origin = trait.get("series_origin")
    return isinstance(origin, dict) and origin.get("original_basis") == "text"


def _plain_text(value, limit):
    return (isinstance(value, str) and 0 < len(value) <= limit and value == value.strip()
            and not any(character in value for character in "[]<>`")
            and not any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value))


def _plain_reason(value):
    return (isinstance(value, str) and 0 < len(value) <= 1000 and value == value.strip()
            and not any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value))


def _payload(source, cast, policy, findings):
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    return {"mode": "voice_ensemble", "version": VERSION,
            "contrast_version": CONTRAST_VERSION,
            "book": deepcopy(source["book"]), "performance_policy": deepcopy(policy),
            "cast": {"characters": [_record_context(character, blocks) for character in cast["characters"]],
                     "roles": deepcopy(cast["roles"]),
                     # Upserts must echo established text traits byte-exactly,
                     # so their evidence cannot use the refresh truncation.
                     "voices": deepcopy(cast["voices"])},
            "contrast_findings": deepcopy(findings),
            "source_fingerprint": fingerprint(source), "cast_fingerprint": fingerprint(cast)}


def _build_cast(cast, upserts):
    """Merge complete upserts into a deepcopy; unlisted metadata keys survive."""
    updated = deepcopy(cast)
    by_id = {voice["id"]: voice for voice in updated["voices"]}
    for upsert in upserts:
        by_id[upsert["id"]].update({key: deepcopy(value) for key, value in upsert.items() if key != "id"})
    return updated


def _validate_response(source, cast):
    """Fully validate ensemble output, including the rebuilt cast, before any success.

    The returned closure checks the raw response shape, rebuilds the proposed
    cast, and runs the complete cast validation on it. CachedRequests therefore
    writes a success cache entry only after every profile field passes, and a
    cached replay re-runs this validator against the full rebuilt cast.
    """
    known = {voice["id"]: voice for voice in cast["voices"]}

    def validate(response):
        from epub_tts_narrator.renderer.voice_prompts import audiobook_voice_prompt
        if not isinstance(response, dict) or set(response) != {"voices", "reasons"}:
            raise ValueError("Voice ensemble must return exactly a voices array and a reasons object")
        upserts, reasons = response["voices"], response["reasons"]
        if not isinstance(upserts, list) or not isinstance(reasons, dict):
            raise ValueError("Voice ensemble needs a voices array and a reasons object keyed by voice ID")
        seen = set()
        for index, upsert in enumerate(upserts):
            location = f"voices[{index}]"
            if not isinstance(upsert, dict) or not isinstance(upsert.get("id"), str):
                raise ValueError(f"Invalid voice ensemble upsert at {location}")
            voice_id = upsert["id"]
            if voice_id not in known or voice_id in seen:
                raise ValueError(f"Voice ensemble upserts must reference an existing voice exactly once: {voice_id}")
            seen.add(voice_id)
            if _inherited(known[voice_id]):
                raise ValueError(f"Voice {voice_id} is an established series voice with a protected definition; "
                                 "the ensemble pass must leave inherited profiles unchanged")
            if set(upsert) != UPSERT_FIELDS:
                raise ValueError(f"A voice ensemble upsert must provide exactly id, description, traits, and "
                                 f"speechTraits at {location}")
            description = upsert["description"]
            if not _plain_text(description, 4000) or not any(character.isalpha() for character in description):
                raise ValueError(f"Voice ensemble upsert description must be nonempty plain audible direction "
                                 f"without markup or control characters at {location}")
            if description[0] in '"“' and description[-1] in '"”':
                raise ValueError(f"Voice ensemble upsert description must not be wrapped in quotation "
                                 f"marks at {location}")
            audiobook_voice_prompt(description)
            traits = upsert["traits"]
            if not isinstance(traits, list):
                raise ValueError(f"Voice ensemble upsert traits must be an array at {location}")
            established_text = [trait for trait in known[voice_id].get("traits", []) if _trait_is_text(trait)]
            for position, trait in enumerate(traits):
                trait_location = f"{location}.traits[{position}]"
                if not isinstance(trait, dict) or set(trait) != TRAIT_FIELDS:
                    raise ValueError(f"Each trait needs exactly a plain name, value, basis, and evidence "
                                     f"at {trait_location}")
                if trait["basis"] not in {"text", "casting"}:
                    raise ValueError(f"Invalid trait basis at {trait_location}")
                if not _plain_text(trait["name"], 80) or not _plain_text(trait["value"], 400):
                    raise ValueError(f"Trait name and value must be plain bounded text at {trait_location}")
                if not isinstance(trait["evidence"], list):
                    raise ValueError(f"Trait evidence must be an array at {trait_location}")
                if trait["basis"] == "text":
                    if not any(trait == candidate for candidate in established_text):
                        raise ValueError(f"Voice {voice_id} keeps its text-supported traits exactly as "
                                         "established; echo every basis=text trait unchanged, including evidence, "
                                         "and never claim new text-supported traits")
                elif trait["evidence"] != []:
                    raise ValueError(f"Casting traits carry no evidence at {trait_location}; "
                                     "text-supported traits keep theirs unchanged")
            for trait in established_text:
                if not any(isinstance(candidate, dict) and candidate == trait for candidate in traits):
                    raise ValueError(f"Voice {voice_id} keeps its text-supported traits exactly as established; "
                                     "echo every basis=text trait unchanged, including evidence")
            validate_speech_traits(upsert["speechTraits"])
            if not _plain_reason(reasons.get(voice_id)):
                raise ValueError(f"Voice ensemble needs a plain brief reason of at most 1000 characters "
                                 f"for every upserted voice: {voice_id}")
        if set(reasons) - seen:
            raise ValueError("Voice ensemble reasons must cover exactly the upserted voices")
        proposed = _build_cast(cast, upserts)
        check_cast(source, proposed)
        return {"cast": proposed, "reasons": deepcopy(reasons)}

    return validate


def design_ensemble(source, cast, requests, *, policy, progress=lambda message: None):
    """Return the cast after one coordinated design pass plus its provenance."""
    findings_before = contrast_findings(cast["voices"]) if len(cast["voices"]) >= 2 else []
    designable = [voice for voice in cast["voices"] if not _inherited(voice)]
    if len(cast["voices"]) < 2 or not designable:
        reason = ("fewer than two distinct voices; nothing to coordinate" if len(cast["voices"]) < 2
                  else "every voice is an established series profile")
        return deepcopy(cast), {"version": VERSION, "contrast_version": CONTRAST_VERSION,
                                "status": "skipped", "reason": reason,
                                "source_fingerprint": fingerprint(source),
                                "cast_fingerprint_before": fingerprint(cast),
                                "cast_fingerprint_after": fingerprint(cast),
                                "guidance_fingerprint": fingerprint(ENSEMBLE_SYSTEM),
                                "request_characters": 0, "updated": {},
                                "contrast_findings_before": findings_before,
                                "contrast_findings_after": findings_before}
    payload = _payload(source, cast, policy, findings_before)
    size = len(ENSEMBLE_SYSTEM) + len(json.dumps(payload, ensure_ascii=False))
    limit = getattr(requests, "max_request_chars", 200000)
    if size > limit:
        raise ValueError(f"Voice ensemble request is {size} characters, exceeding the configured request budget "
                         f"({limit}). The complete cast is never dropped or trimmed to fit; "
                         "raise analysis.max_request_chars.")
    progress("Designing the voice cast as one ensemble")
    accepted = requests.run(ENSEMBLE_SYSTEM, payload, _validate_response(source, cast))
    updated, reasons = accepted["cast"], accepted["reasons"]
    original = {voice["id"]: voice for voice in cast["voices"]}
    changed = {}
    for voice in updated["voices"]:
        if voice != original[voice["id"]]:
            changed[voice["id"]] = {"reason": reasons[voice["id"]],
                                    "description_fingerprint": fingerprint(voice["description"]),
                                    "speech_traits_fingerprint": fingerprint(voice.get("speechTraits"))}
    return updated, {"version": VERSION, "contrast_version": CONTRAST_VERSION,
                     "status": "applied" if changed else "unchanged",
                     "source_fingerprint": fingerprint(source),
                     "cast_fingerprint_before": fingerprint(cast),
                     "cast_fingerprint_after": fingerprint(updated),
                     "guidance_fingerprint": fingerprint(ENSEMBLE_SYSTEM),
                     "request_characters": size,
                     "updated": changed,
                     "contrast_findings_before": findings_before,
                     "contrast_findings_after": contrast_findings(updated["voices"])}
