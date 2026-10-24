"""Refresh casting directions without rewriting an analyzed book or its voices."""

from copy import deepcopy
from pathlib import Path
import unicodedata

from .attribution_review import _bounded_map
from .pipeline import CachedRequests, fingerprint
from .voice_contrast import contrast_findings
from .voice_refresh_prompts import SYSTEM


VERSION = 1


def validate_description(value):
    """Enforce the output shape and actual voice-designer prompt size limit."""
    from epub_tts_narrator.renderer.voice_prompts import audiobook_voice_prompt
    if not isinstance(value, dict) or set(value) != {"description", "reason"}:
        raise ValueError("Voice prompt refresh requires exactly description and reason")
    description, reason = value["description"], value["reason"]
    if (not isinstance(description, str) or not description.strip()
            or description != description.strip() or not any(char.isalpha() for char in description)
            or any(char in description for char in "[]<>`")
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in description)
            or (description[0] in '\"“' and description[-1] in '\"”')):
        raise ValueError("Voice description must be nonempty plain direction without control characters, markup, or outer quotation marks")
    audiobook_voice_prompt(description)
    if (not isinstance(reason, str) or not reason.strip() or reason != reason.strip()
            or len(reason) > 1000
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in reason)):
        raise ValueError("Voice prompt refresh requires a brief plain-text reason of at most 1000 characters")
    return {"description": description, "reason": reason}


def _evidence(records, blocks):
    """Keep a bounded set of genuine source quotes; earlier-book IDs may be absent."""
    result, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            continue
        block_id, quote = record.get("block_id"), record.get("quote")
        text = blocks.get(block_id, "")
        if (not isinstance(quote, str) or not quote or quote not in text
                or (block_id, quote) in seen):
            continue
        seen.add((block_id, quote))
        result.append({"block_id": block_id, "quote": quote[:600]})
        if len(result) == 12:
            break
    return result


def _record_context(record, blocks):
    result = deepcopy(record)
    if "evidence" in result:
        result["evidence"] = _evidence(result["evidence"], blocks)
    if "traits" in result:
        result["traits"] = [_record_context(trait, blocks) for trait in result["traits"]]
    return result


def _provider_identity(requests):
    identity = getattr(getattr(requests, "client", None), "identity", {})
    # The real client's identity is already credential-free; allowlist audit
    # fields so alternate request implementations cannot accidentally add keys.
    return {key: deepcopy(identity[key]) for key in (
        "model", "base_url", "json_mode", "max_output_tokens", "token_limit_field", "reasoning_effort"
    ) if isinstance(identity, dict) and key in identity}


def refresh_voice_descriptions(bundle, requests, *, canonical_id=lambda value: value,
                               voice_ids=None, current_prompts=None, workers=5,
                               progress=lambda message: None):
    """Propose validated draft descriptions; only CachedRequests may write files."""
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Voice prompt refresh workers must be between one and five")
    profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
    selected = list(profiles) if voice_ids is None else list(dict.fromkeys(voice_ids))
    if any(voice not in profiles for voice in selected):
        raise ValueError("Voice prompt refresh references an unknown voice")
    canonical = {voice: canonical_id(voice) for voice in profiles}
    if any(not isinstance(value, str) or not value for value in canonical.values()):
        raise ValueError("Voice prompt refresh requires nonempty canonical voice IDs")
    current_prompts = {} if current_prompts is None else current_prompts
    if (not isinstance(current_prompts, dict)
            or any(key not in profiles and key not in canonical.values() for key in current_prompts)
            or any(not isinstance(value, str) or not value.strip() for value in current_prompts.values())):
        raise ValueError("Current voice prompts must map known voice IDs to nonempty direction text")
    groups = {}
    for voice in selected:
        groups.setdefault(canonical[voice], []).append(voice)
    blocks = {block["id"]: block["text"] for block in bundle["source"]["blocks"]}
    # One snapshot for every task: each worker sees the same complete cast.
    known_cast = [_record_context(profile, blocks) for profile in profiles.values()]
    findings = contrast_findings(list(profiles.values()), canonical_id=canonical.get)
    tasks = []
    for target, voices in groups.items():
        aliases = [voice for voice in profiles if canonical[voice] == target]
        primary = target if target in profiles else voices[0]
        profile = _record_context(profiles[primary], blocks)
        profile["id"] = target
        characters = [_record_context(character, blocks) for character in bundle["cast"]["characters"]
                      if character["voice_id"] in aliases]
        roles = [deepcopy(role) for role in bundle["cast"]["roles"] if role["voice_id"] in aliases]
        prior = current_prompts.get(target)
        if prior is None:
            prior = next((current_prompts[alias] for alias in aliases if alias in current_prompts),
                         profile["description"])
        evidence = _evidence([item for trait in profile.get("traits", []) for item in trait.get("evidence", [])]
                             + [item for character in characters for item in character.get("evidence", [])], blocks)
        tasks.append({"mode": "voice_prompt_refresh", "version": VERSION,
                      "book": deepcopy(bundle["source"]["book"]), "voice": profile,
                      "alias_voice_ids": aliases, "characters": characters, "roles": roles,
                      "known_cast": deepcopy(known_cast),
                      "contrast_findings": deepcopy([finding for finding in findings
                                                     if any(voice in aliases for voice in finding["voice_ids"])]),
                      "current_prompt": prior, "source_evidence": evidence,
                      "source_fingerprint": bundle["manifest"]["source_fingerprint"]})
    if tasks and requests is None:
        raise ValueError("Voice prompt refresh needs the configured analysis model")

    def refresh(payload, cancelled):
        value = requests.run(SYSTEM, payload, validate_description, cancelled=cancelled)
        progress(f"Refreshed voice prompt: {payload['voice']['id']}")
        return payload["voice"]["id"], value

    prompts = dict(_bounded_map(tasks, refresh, workers))
    return {"version": VERSION, "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "cast_fingerprint": fingerprint(bundle["cast"]), "guidance_fingerprint": fingerprint(SYSTEM),
            "provider": _provider_identity(requests), "prompts": prompts}


def refresh_bundle_prompts(bundle, config_path=None, *, canonical_id=lambda value: value,
                           voice_ids=None, current_prompts=None, workers=5, progress=print):
    """Use the global analyzer configuration, never the configured speech engine."""
    from .config import load_config
    from .provider import OpenAICompatibleClient
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Voice prompt refresh workers must be between one and five")
    config = load_config(config_path)
    settings = config["analysis"]
    client = OpenAICompatibleClient(**config["provider"])
    cache = settings["cache_dir"] or Path(bundle["directory"]).parent / ".epub-audiobook-cache"
    progress(f"Refreshing voice prompts with {client.model} at {client.base_url}; "
             "character descriptions, casting directions, and source evidence are sent to this endpoint.")
    requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    return refresh_voice_descriptions(
        bundle, requests, canonical_id=canonical_id, voice_ids=voice_ids,
        current_prompts=current_prompts, workers=min(workers, 5, settings["attribution_workers"]), progress=progress)
