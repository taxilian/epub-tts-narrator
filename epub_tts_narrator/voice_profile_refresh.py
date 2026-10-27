"""Refresh analyzed voice profiles in place; nothing else in the book may change.

Only cast voice ``description``, ``traits``, and ``speechTraits`` are revised,
plus their hidden ``cast_fingerprint`` bindings in provenance and the book-local
audition script sidecar. Source, characters, roles, voice identity, segments,
spelling edits, and series ownership stay exact; selected voices, reference
audio, and render caches are never touched.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
import unicodedata

from .attribution_review import _bounded_map
from .audition_scripts import FILENAME as AUDITION_SIDECAR, VERSION as AUDITION_SIDECAR_VERSION, validate_script
from .pipeline import fingerprint
from .speech_traits import validate_speech_traits
from .voice_profile_refresh_prompts import SYSTEM
from .voice_prompt_refresh import _evidence, _provider_identity, _record_context


VERSION = 1
PROFILE_FIELDS = ("description", "traits", "speechTraits")


def _plain_text(value, limit):
    return (isinstance(value, str) and 0 < len(value) <= limit and value == value.strip()
            and not any(char in value for char in "[]<>`")
            and not any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value))


def _accent_traits(profile):
    return [trait for trait in (profile or {}).get("traits", [])
            if " ".join(trait["name"].casefold().split()).strip(" :;.,")
            in {"accent", "accents", "dialect", "dialects"}]


def _validate_inherited_accent(value, prior):
    """Guard declared accent continuity, not semantic or acoustic equivalence."""
    inherited = _accent_traits(prior)
    if not inherited:
        return
    expected = sorted(trait["value"] for trait in inherited)
    proposed = sorted(trait["value"] for trait in _accent_traits(value))
    if proposed != expected:
        raise ValueError("Inherited accent/dialect trait values must be repeated exactly from "
                         "prior_series_voice; automatic profile refresh cannot replace or omit them. "
                         "Return those values with casting basis and empty evidence; existing source "
                         "evidence is preserved by the application. Other compatible traits may improve.")
    for field in ("description", "speechTraits"):
        direction = " ".join(value[field].casefold().split())
        for accent in expected:
            phrase = " ".join(accent.casefold().split())
            if not re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", direction):
                raise ValueError(f"Inherited accent wording is missing from {field}; include every "
                                 "prior_series_voice accent/dialect trait value verbatim in this field. "
                                 "Compatible cadence or delivery refinements may accompany it.")


def validate_profile_response(value, *, prior_series_voice=None):
    """Enforce the refresh output contract: the three profile fields plus a reason."""
    from epub_tts_narrator.renderer.voice_prompts import audiobook_voice_prompt
    if not isinstance(value, dict) or set(value) != {"description", "traits", "speechTraits", "reason"}:
        raise ValueError("Voice profile refresh requires exactly description, traits, speechTraits, and reason")
    description = value["description"]
    if (not isinstance(description, str) or not description.strip()
            or description != description.strip() or not any(char.isalpha() for char in description)
            or any(char in description for char in "[]<>`")
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in description)
            or (description[0] in '"“' and description[-1] in '"”')):
        raise ValueError("Voice description must be nonempty plain direction without control characters, "
                         "markup, or outer quotation marks")
    audiobook_voice_prompt(description)
    traits = value["traits"]
    if not isinstance(traits, list) or not 1 <= len(traits) <= 8:
        raise ValueError("Voice profile refresh requires one to eight traits")
    for trait in traits:
        if (not isinstance(trait, dict) or set(trait) != {"name", "value", "basis", "evidence"}
                or trait["basis"] != "casting" or trait["evidence"] != []
                or not _plain_text(trait["name"], 80) or not _plain_text(trait["value"], 400)):
            raise ValueError("Each refreshed trait needs exactly a plain name and audible value with casting "
                             "basis, an empty evidence list, and no book quotations")
    try:
        speech_traits = validate_speech_traits(value["speechTraits"])
    except ValueError as error:
        raise ValueError(f"Refreshed output {error}") from None
    reason = value["reason"]
    if (not isinstance(reason, str) or not reason.strip() or reason != reason.strip()
            or len(reason) > 1000
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in reason)):
        raise ValueError("Voice profile refresh requires a brief plain-text reason of at most 1000 characters")
    accepted = {"description": description, "traits": deepcopy(traits),
                "speechTraits": speech_traits, "reason": reason}
    _validate_inherited_accent(accepted, prior_series_voice)
    return accepted


def _profile_context(profile, blocks):
    result = {"id": profile["id"], "description": profile["description"]}
    if "speechTraits" in profile:
        result["speechTraits"] = profile["speechTraits"]
    traits = []
    for trait in profile.get("traits", []):
        item = deepcopy(trait)
        if "evidence" in item:
            item["evidence"] = _evidence(item["evidence"], blocks)
        traits.append(item)
    result["traits"] = traits
    return result


def refresh_voice_profiles(bundle, requests, *, voice_ids=None, workers=5,
                           progress=lambda message: None):
    """Propose validated profile updates; only CachedRequests may write files."""
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Voice profile refresh workers must be between one and five")
    profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
    selected = list(profiles) if voice_ids is None else list(dict.fromkeys(voice_ids))
    if any(voice not in profiles for voice in selected):
        raise ValueError("Voice profile refresh references an unknown voice")
    blocks = {block["id"]: block["text"] for block in bundle["source"]["blocks"]}
    prior = {voice["id"]: voice for voice in bundle.get("series_context", {}).get("voices", [])}
    for voice_id in selected:
        if voice_id in prior and "speechTraits" not in prior[voice_id]:
            raise ValueError(f"Voice {voice_id} lacks speechTraits in its originating definition; "
                             "refresh its defining book first, then continue in series order")
    known_cast = [_profile_context(profile, blocks) for profile in profiles.values()]
    tasks = []
    for voice_id in selected:
        profile = profiles[voice_id]
        characters = [_record_context(character, blocks) for character in bundle["cast"]["characters"]
                      if character["voice_id"] == voice_id]
        roles = [deepcopy(role) for role in bundle["cast"]["roles"] if role["voice_id"] == voice_id]
        evidence = _evidence([item for trait in profile.get("traits", []) for item in trait.get("evidence", [])]
                             + [item for character in characters for item in character.get("evidence", [])], blocks)
        tasks.append({"mode": "voice_profile_refresh", "version": VERSION,
                      "book": deepcopy(bundle["source"]["book"]),
                      "voice": _profile_context(profile, blocks),
                      "characters": characters, "roles": roles,
                      "current_description": profile["description"],
                      "current_traits": deepcopy(profile.get("traits", [])),
                      "current_speech_traits": profile.get("speechTraits"),
                      "known_cast": known_cast, "prior_series_voice": deepcopy(prior.get(voice_id)),
                      "source_evidence": evidence,
                      "source_fingerprint": bundle["manifest"]["source_fingerprint"]})
    if tasks and requests is None:
        raise ValueError("Voice profile refresh needs the configured analysis model")

    def refresh(payload, cancelled):
        def validate(response):
            return validate_profile_response(response, prior_series_voice=payload["prior_series_voice"])

        value = requests.run(SYSTEM, payload, validate, cancelled=cancelled)
        progress(f"Refreshed voice profile: {payload['voice']['id']}")
        return payload["voice"]["id"], value

    return {"version": VERSION,
            "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "cast_fingerprint_before": fingerprint(bundle["cast"]),
            "guidance_fingerprint": fingerprint(SYSTEM),
            "provider": _provider_identity(requests),
            "profiles": dict(_bounded_map(tasks, refresh, workers))}


def _merged_traits(profile, proposed):
    retained = [trait for trait in profile.get("traits", []) if trait.get("basis") == "text"
                or trait.get("series_origin", {}).get("original_basis") == "text"]
    names = {trait["name"].casefold() for trait in retained}
    return deepcopy(retained + [trait for trait in proposed if trait["name"].casefold() not in names])


def profiles_changed(document, updates):
    return any(voice["description"] != updates[voice["id"]]["description"]
               or voice.get("traits", []) != _merged_traits(voice, updates[voice["id"]]["traits"])
               or voice.get("speechTraits") != updates[voice["id"]]["speechTraits"]
               for voice in document["cast"]["voices"])


def apply_profile_updates(document, updates, *, provider, guidance_fingerprint):
    """Apply validated responses to every existing voice; return a revised deepcopy."""
    if (not isinstance(updates, dict) or not isinstance(provider, dict)
            or set(provider) - {"model", "base_url", "json_mode", "max_output_tokens",
                                "token_limit_field", "reasoning_effort"}
            or not isinstance(guidance_fingerprint, str) or not guidance_fingerprint):
        raise ValueError("Voice profile updates require validated responses, a credential-free provider "
                         "identity, and the guidance fingerprint")
    result = deepcopy(document)
    samples = document.get("provenance", {}).get("voice_samples")
    if isinstance(samples, dict) and samples.get("cast_fingerprint") != fingerprint(document["cast"]):
        raise ValueError("Saved voice_samples does not match the pre-refresh cast; refusing to rebind it")
    voices = result["cast"]["voices"]
    known = [voice["id"] for voice in voices]
    if len(set(known)) != len(known) or set(updates) != set(known):
        raise ValueError("Voice profile updates must cover exactly the existing voice IDs once each; "
                         "no alias collapse, removals, or additions")
    for voice in voices:
        response = updates[voice["id"]]
        voice["description"] = response["description"]
        voice["traits"] = _merged_traits(voice, response["traits"])
        voice["speechTraits"] = response["speechTraits"]
    provenance = result["provenance"]
    samples = provenance.get("voice_samples")
    if isinstance(samples, dict) and {"cast_fingerprint", "scripts", "samples"} & set(samples):
        samples["cast_fingerprint"] = fingerprint(result["cast"])
    provenance["voice_profile_refresh"] = {
        "version": VERSION,
        "source_fingerprint": fingerprint(result["source"]),
        "cast_fingerprint": fingerprint(result["cast"]),
        "guidance_fingerprint": guidance_fingerprint,
        "provider": deepcopy(provider),
        "voices": {voice["id"]: {"reason": updates[voice["id"]]["reason"],
                                 "description_fingerprint": fingerprint(updates[voice["id"]]["description"]),
                                 "speech_traits_fingerprint": fingerprint(updates[voice["id"]]["speechTraits"])}
                   for voice in voices},
    }
    return result


def _read_json_file(path):
    def reject(value):
        raise ValueError("Non-finite numbers are not valid JSON")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def audition_sidecar_rebind(path, *, source_fingerprint, cast_fingerprint_before,
                            cast_fingerprint_after, voice_ids):
    """Return the existing audition sidecar with only its cast binding refreshed.

    Corrupted or stale sidecars are refused rather than silently rebound; the
    saved scripts, samples, and any other records are preserved byte-for-byte
    in content.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Voice profile refresh needs a regular {AUDITION_SIDECAR} sidecar to rebind")
    record = _read_json_file(path)
    if (not isinstance(record, dict) or record.get("version") != AUDITION_SIDECAR_VERSION
            or record.get("source_fingerprint") != source_fingerprint
            or record.get("cast_fingerprint") != cast_fingerprint_before
            or not isinstance(record.get("scripts"), dict)):
        raise ValueError(f"Saved {AUDITION_SIDECAR} does not match this book's pre-refresh source and cast; "
                         "recast or refresh its auditions before refreshing voice profiles")
    known = set(voice_ids)
    for voice, script in record["scripts"].items():
        if voice not in known or not isinstance(script, dict) or script.get("basis") != "casting":
            raise ValueError(f"Invalid saved character audition script for {voice} in {AUDITION_SIDECAR}")
        try:
            validate_script({key: script.get(key) for key in ("text", "reason")}, require_plain_prose=False)
        except ValueError as error:
            raise ValueError(f"Invalid saved character audition script for {voice}: {error}") from None
    bound = deepcopy(record)
    bound["cast_fingerprint"] = cast_fingerprint_after
    return bound


def refresh_book_voice_profiles(directory, config_path=None, *, workers=1, progress=print):
    from .voice_profile_workflow import refresh_book_voice_profiles as refresh
    return refresh(directory, config_path, workers=workers, progress=progress)
