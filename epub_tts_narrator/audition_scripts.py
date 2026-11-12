"""Prepare useful audition text without changing the book or character identities."""

from copy import deepcopy
import json
from pathlib import Path
import re
import unicodedata

from .analysis_progress import analysis_cache_directory
from .attribution_review import _bounded_map
from .audition_prompts import EXTRA_PROMPT_VERSION, EXTRA_SYSTEM, PROMPT_VERSION, SYSTEM
from .pipeline import CachedRequests, fingerprint, write_json
from .provider import ProviderError


VERSION = 1
FILENAME = "audition-scripts.json"
WORDS = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*", re.UNICODE)


def validate_script(value, *, require_plain_prose=True):
    from epub_tts_narrator.renderer.samples import sample_text_problems, has_repeated_speech_wrappers, strip_speech_wrapper
    if not isinstance(value, dict) or set(value) != {"text", "reason"}:
        raise ValueError("Audition script requires exactly text and reason")
    text, reason = value["text"], value["reason"]
    if not isinstance(text, str):
        raise ValueError("Audition text must be a string of plain spoken prose")
    problems = sample_text_problems(text)
    # Formatting whitespace is not speech; normalize it without asking the
    # model for another take. Check controls before normalization can hide them.
    text = strip_speech_wrapper(" ".join(text.split()))
    problems = list(dict.fromkeys([*problems, *sample_text_problems(text)]))
    words = len(WORDS.findall(text))
    if not 25 <= words <= 60:
        problems.append(f"received {words} words; write 35–50 words (allowed range 25–60)")
    if len(text) > 650:
        problems.append(f"received {len(text)} characters; keep the text within 650 characters")
    if problems:
        raise ValueError("Audition text: " + "; ".join(problems))
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("Audition script requires a brief casting rationale")
    if require_plain_prose and has_repeated_speech_wrappers(text):
        raise ValueError("Audition script must use plain spoken prose, not separately quoted statements. "
                         "Remove the surrounding quotation marks from each statement; preserve only "
                         "meaningful quotations within speech and normal apostrophes.")
    return {"text": text, "basis": "casting", "reason": reason}


def _normalized(text):
    return " ".join(WORDS.findall(text.casefold()))


def validate_extra_scripts(value, *, count, existing):
    """Exactly ``count`` new audition-rule scripts, none repeating ``existing`` or each other."""
    if not isinstance(value, dict) or set(value) != {"scripts"} or not isinstance(value["scripts"], list):
        raise ValueError('Extra reference scripts require exactly {"scripts": [{"text": ..., "reason": ...}]}')
    if len(value["scripts"]) != count:
        raise ValueError(f"Write exactly {count} extra reference script(s); received {len(value['scripts'])}")
    seen = {_normalized(text) for text in existing}
    scripts = []
    for index, item in enumerate(value["scripts"], 1):
        try:
            script = validate_script(item)
        except ValueError as error:
            raise ValueError(f"Extra reference script {index}: {error}") from None
        key = _normalized(script["text"])
        if key in seen:
            raise ValueError(f"Extra reference script {index} repeats an existing or earlier script; write different speech")
        seen.add(key)
        scripts.append(script)
    return scripts


def _saved_extras(bundle):
    saved = bundle.get("audition_extra_scripts", {})
    known = {voice["id"] for voice in bundle["cast"]["voices"]}
    if not isinstance(saved, dict):
        raise ValueError("Saved extra reference scripts must be an object keyed by voice ID")
    for voice, scripts in saved.items():
        if voice not in known or not isinstance(scripts, list) or not scripts:
            raise ValueError("Invalid saved extra reference scripts")
        for script in scripts:
            if not isinstance(script, dict) or script.get("basis") != "casting":
                raise ValueError("Invalid saved extra reference script")
            validate_script({key: script.get(key) for key in ("text", "reason")})
    return deepcopy(saved)


def _voice_context(bundle, aliases):
    """The characters, roles and short attributed lines behind one voice group."""
    people = [character for character in bundle["cast"]["characters"] if character["voice_id"] in aliases]
    roles = [role for role in bundle["cast"]["roles"] if role["voice_id"] in aliases]
    examples = []
    for chapter in bundle["chapters"]:
        for row in chapter["rows"]:
            performers = [row.get("voice_id"), *[item["voice_id"] for item in row.get("simultaneous_speakers", [])]]
            attribution = row.get("attribution", {})
            if (any(value in aliases for value in performers)
                    and attribution.get("basis") in {"explicit", "context", "user"}
                    and attribution.get("review_required") is False
                    and len(examples) < 8):
                examples.append(row["text"][:500])
    return people, roles, examples


def _pending(bundle):
    prepared = bundle["manifest"].get("provenance", {}).get("voice_samples", {})
    if not isinstance(prepared, dict):
        raise ValueError("Invalid saved character audition script metadata")
    pending = prepared.get("pending", {})
    known = {voice["id"] for voice in bundle["cast"]["voices"]}
    if (not isinstance(pending, dict) or any(
            voice not in known or not isinstance(row, dict) or set(row) != {"name", "error"}
            or any(not isinstance(row[key], str) or not row[key].strip() or len(row[key]) > limit
                   for key, limit in (("name", 160), ("error", 2000)))
            for voice, row in pending.items())):
        raise ValueError("Invalid pending character audition script metadata")
    return deepcopy(pending)


def audition_preparation_pending(document):
    return bool(_pending({"cast": document["cast"], "manifest": {"provenance": document["provenance"]}}))


def _voice_name(people, roles, voice):
    names = list(dict.fromkeys(person["name"] for person in people if person.get("name")))
    label = ", ".join(names) if names else ("Narrator" if any(role.get("kind") == "narration" for role in roles) else voice)
    return "".join(" " if unicodedata.category(char).startswith("C") else char for char in label)[:160]


def _saved(bundle):
    prepared = bundle["manifest"].get("provenance", {}).get("voice_samples", {})
    if not isinstance(prepared, dict):
        raise ValueError("Invalid saved character audition script metadata")
    inherited = prepared.get("scripts", {})
    sidecar = bundle.get("audition_scripts", {})
    if not isinstance(inherited, dict) or not isinstance(sidecar, dict):
        raise ValueError("Saved character audition scripts must be an object keyed by voice ID")
    replaced = set(_overrides(bundle))
    scripts = {voice["id"]: deepcopy(voice["sample_script"])
               for voice in bundle["cast"]["voices"] if voice.get("sample_script") is not None}
    scripts.update(deepcopy(inherited))
    scripts = {voice: script for voice, script in scripts.items() if voice not in replaced}
    scripts.update(deepcopy(sidecar))
    known = {voice["id"] for voice in bundle["cast"]["voices"]}
    for voice, script in scripts.items():
        if voice not in known or not isinstance(script, dict) or script.get("basis") != "casting":
            raise ValueError("Invalid saved character audition script")
        # Legacy quoted scripts must remain readable so explicit refresh can
        # replace them. New model responses obey the stricter prose contract.
        validate_script({key: script.get(key) for key in ("text", "reason")}, require_plain_prose=False)
    return scripts


def _overrides(bundle):
    replaced = bundle.get("audition_script_overrides", [])
    known = {voice["id"] for voice in bundle["cast"]["voices"]}
    if (not isinstance(replaced, list) or any(not isinstance(voice, str) or voice not in known
                                               for voice in replaced)):
        raise ValueError("Invalid replaced character audition script metadata")
    return list(dict.fromkeys(replaced))


def prepare_samples(bundle, requests, *, canonical_id=lambda value: value, voice_ids=None,
                    workers=5, progress=lambda message: None, refresh=False, allow_incomplete=False,
                    extra_clips=0):
    """Prefer source auditions; explicitly refresh selected groups when requested.

    Refresh drops creative fallbacks only for the selected canonical groups.
    Their replacement scripts (or absence, when source speech suffices) override
    every older inline/provenance copy without rewriting the analysis bundle.

    ``extra_clips`` asks for that many further reference scripts per voice (a
    voice recorded as several joined clips). Book speech supplies them first;
    the model writes the rest in the voice's own style, saved as
    ``extra_scripts``. A failed request leaves that voice with fewer clips.
    """
    from epub_tts_narrator.renderer.samples import (
        MissingSampleScript, select_book_sample, select_extra_samples, select_sample)
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Audition workers must be between one and five")
    if type(extra_clips) is not int or not 0 <= extra_clips <= 2:
        raise ValueError("Extra reference clips must be an integer from zero through two")
    if type(allow_incomplete) is not bool:
        raise ValueError("allow_incomplete must be a boolean")
    scripts = _saved(bundle)
    extras = _saved_extras(bundle)
    prior_pending = _pending(bundle)
    pending = {}
    profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
    selected = list(profiles) if voice_ids is None else list(dict.fromkeys(voice_ids))
    if any(voice not in profiles for voice in selected):
        raise ValueError("Audition references an unknown voice")
    groups = {}
    for voice in selected:
        groups.setdefault(canonical_id(voice), []).append(voice)
    replaced = set(_overrides(bundle))
    if refresh:
        targets = {voice for voice in profiles if canonical_id(voice) in groups}
        replaced.update(targets)
        scripts = {voice: script for voice, script in scripts.items() if voice not in targets}
        extras = {voice: value for voice, value in extras.items() if voice not in targets}
    working = dict(bundle, audition_scripts=scripts, audition_script_overrides=sorted(replaced))
    tasks = []
    for canonical, voices in groups.items():
        voice = canonical if canonical in profiles else voices[0]
        if select_book_sample(working, voice, canonical_id=canonical_id) is not None:
            continue
        aliases = [value for value in profiles if canonical_id(value) == canonical]
        prior = scripts.get(canonical) or next((scripts[value] for value in aliases if value in scripts), None)
        if prior is not None:
            scripts.update({value: deepcopy(prior) for value in aliases})
            continue
        people, roles, examples = _voice_context(bundle, aliases)
        profile = deepcopy(profiles[voice])
        if refresh:
            profile.pop("sample_script", None)
        payload = {"mode": "audition_script", "version": PROMPT_VERSION,
                   "book": bundle["source"]["book"], "voice": profile,
                   "characters": people, "roles": roles, "short_speech_examples": examples,
                   "source_fingerprint": bundle["manifest"]["source_fingerprint"]}
        tasks.append((aliases, payload, _voice_name(people, roles, voice)))
    def create(task, cancelled):
        aliases, payload, name = task
        try:
            script = requests.run(SYSTEM, payload, validate_script, cancelled=cancelled)
        except (ProviderError, ValueError) as error:
            if not allow_incomplete:
                raise
            detail = str(error)[:2000] or type(error).__name__
            progress(f"Audition pending for {name}: {detail}")
            return aliases, None, {"name": name, "error": detail}
        progress(f"Prepared character-specific audition script: {name} ({payload['voice']['id']})")
        return aliases, script, None
    if tasks:
        if requests is None:
            if not allow_incomplete:
                raise ValueError("Character-specific audition scripts need the configured analysis model")
            for aliases, payload, name in tasks:
                for voice in aliases:
                    pending[voice] = prior_pending.get(voice) or {
                        "name": name, "error": "Character-specific audition script needs the configured analysis model"}
        else:
            progress(f"Preparing {len(tasks)} missing character audition script(s)")
            for aliases, script, failure in _bounded_map(tasks, create, workers):
                if script is None:
                    pending.update({value: deepcopy(failure) for value in aliases})
                else:
                    scripts.update({value: deepcopy(script) for value in aliases})
    if extra_clips:
        extras = _prepare_extras(dict(working, audition_scripts=scripts, audition_extra_scripts=extras),
                                 requests, groups, profiles, pending, extras, extra_clips,
                                 canonical_id=canonical_id, workers=workers, progress=progress,
                                 allow_incomplete=allow_incomplete,
                                 selectors=(MissingSampleScript, select_extra_samples, select_sample))
    samples = {voice: select_sample(working, voice, canonical_id=canonical_id, fallback_text="")
               for voice in selected if voice not in pending}
    result = {"version": VERSION, "source_fingerprint": bundle["manifest"]["source_fingerprint"],
              "cast_fingerprint": fingerprint(bundle["cast"]), "scripts": scripts, "samples": samples}
    if replaced:
        result["replaced_voice_ids"] = sorted(replaced)
    if extras:
        result["extra_scripts"] = extras
    if pending:
        result["pending"] = pending
        progress(f"{len(pending)} voice audition(s) still need preparation; analysis can be saved. "
                 "Rerun analyze or open casting to retry the missing auditions.")
    return result


def _prepare_extras(working, requests, groups, profiles, pending, extras, count, *, canonical_id,
                    workers, progress, allow_incomplete, selectors):
    """Ask the model for the extra reference scripts the book's speech cannot supply."""
    MissingSampleScript, select_extra_samples, select_sample = selectors
    extras = deepcopy(extras)
    tasks = []
    for canonical, voices in groups.items():
        voice = canonical if canonical in profiles else voices[0]
        aliases = [value for value in profiles if canonical_id(value) == canonical]
        if any(value in pending for value in aliases):
            continue
        try:
            first = select_sample(working, voice, canonical_id=canonical_id)
        except MissingSampleScript:
            continue
        book = select_extra_samples(working, voice, canonical_id=canonical_id,
                                    exclude_segments=first["source"].get("segment_ids", []), count=count)
        missing = count - len(book)
        saved = next((extras[value] for value in [canonical, *aliases] if extras.get(value)), [])
        if missing <= len(saved):
            continue
        people, roles, examples = _voice_context(working, aliases)
        existing = [first["reference_text"], *[sample["reference_text"] for sample in book]]
        payload = {"mode": "extra_reference_scripts", "version": EXTRA_PROMPT_VERSION,
                   "book": working["source"]["book"], "voice": deepcopy(profiles[voice]),
                   "characters": people, "roles": roles, "short_speech_examples": examples,
                   "existing_scripts": existing, "count": missing,
                   "source_fingerprint": working["manifest"]["source_fingerprint"]}
        tasks.append((aliases, payload, _voice_name(people, roles, voice)))
    if not tasks:
        return extras
    if requests is None:
        if allow_incomplete:
            return extras
        raise ValueError("Character-specific audition scripts need the configured analysis model")

    def create(task, cancelled):
        aliases, payload, name = task
        def validate(value):
            return validate_extra_scripts(value, count=payload["count"], existing=payload["existing_scripts"])
        try:
            created = requests.run(EXTRA_SYSTEM, payload, validate, cancelled=cancelled)
        except (ProviderError, ValueError) as error:
            detail = str(error)[:2000] or type(error).__name__
            progress(f"Extra reference scripts not prepared for {name}; its voice will have fewer clips: {detail}")
            return aliases, None
        progress(f"Prepared {len(created)} extra reference script(s): {name} ({payload['voice']['id']})")
        return aliases, created
    progress(f"Preparing extra reference scripts for {len(tasks)} voice(s) with too little book speech")
    for aliases, created in _bounded_map(tasks, create, workers):
        if created is not None:
            extras.update({value: deepcopy(created) for value in aliases})
    return extras


def prepare_document_samples(document, requests, *, workers=5, progress=lambda message: None,
                             allow_incomplete=False):
    from .cli import script_partitions
    bundle = {"source": document["source"], "cast": document["cast"],
              "manifest": {"source_fingerprint": fingerprint(document["source"]),
                           "provenance": document["provenance"]},
              "chapters": [{"entry": {"title": chapter["title"],
                                           "path": f"scripts/chapter-{number:04d}.jsonl"},
                            "rows": [dict(item, chapter_id=chapter["chapter_id"])
                                     for item in document["segments"] if item["block_id"] in chapter["block_ids"]]}
                           for number, chapter in enumerate(script_partitions(document), 1)]}
    result = deepcopy(document)
    result["provenance"]["voice_samples"] = prepare_samples(
        bundle, requests, workers=workers, progress=progress, allow_incomplete=allow_incomplete)
    return result


def ensure_bundle_scripts(bundle, config_path=None, *, canonical_id=lambda value: value,
                          voice_ids=None, progress=print, refresh=False, extra_clips=0):
    """Prepare or explicitly refresh auditions without editing signed analysis."""
    from .config import load_config
    from .provider import OpenAICompatibleClient
    directory = Path(bundle["directory"])
    path = directory / FILENAME
    expected = {"source_fingerprint": bundle["manifest"]["source_fingerprint"],
                "cast_fingerprint": fingerprint(bundle["cast"])}
    def load_working():
        working = dict(bundle)
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ValueError("Audition script sidecar must be a regular file")
            record = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or record.get("version") != VERSION
                    or any(record.get(key) != value for key, value in expected.items())
                    or not isinstance(record.get("scripts"), dict)):
                raise ValueError("Saved audition scripts do not match this book's source and cast")
            working["audition_scripts"] = record["scripts"]
            working["audition_script_overrides"] = record.get("replaced_voice_ids", [])
            working["audition_extra_scripts"] = record.get("extra_scripts", {})
        _overrides(working)
        return working

    def prepared_bundle(working, prepared):
        return dict(working, audition_scripts=prepared["scripts"],
                    audition_script_overrides=prepared.get("replaced_voice_ids", []),
                    audition_extra_scripts=prepared.get("extra_scripts", {}))

    working = load_working()
    # Fast path: no configuration or network is needed for prepared books.
    # Explicit refresh always persists its overrides, including source-only runs.
    if not refresh:
        try:
            prepared = prepare_samples(working, None, canonical_id=canonical_id, voice_ids=voice_ids,
                                       extra_clips=extra_clips)
        except ValueError as error:
            if str(error) != "Character-specific audition scripts need the configured analysis model":
                raise
        else:
            return prepared_bundle(working, prepared)
    lock = directory / ".audition-scripts.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(f"Audition script preparation is already running or interrupted: {lock}") from error
    try:
        # A preceding writer may have completed between the fast path and lock.
        working = load_working()
        try:
            prepared = prepare_samples(working, None, canonical_id=canonical_id, voice_ids=voice_ids,
                                       refresh=refresh, extra_clips=extra_clips)
        except ValueError as error:
            if str(error) != "Character-specific audition scripts need the configured analysis model":
                raise
            config = load_config(config_path)
            settings = config["analysis"]
            client = OpenAICompatibleClient(**config["provider"])
            cache = analysis_cache_directory(directory, settings["cache_dir"])
            progress(f"Preparing audition scripts with {client.model} at {client.base_url}; character descriptions and excerpts are sent to this endpoint.")
            requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"], progress=progress)
            prepared = prepare_samples(working, requests, canonical_id=canonical_id, voice_ids=voice_ids,
                                       workers=min(5, settings["attribution_workers"]), progress=progress,
                                       refresh=refresh, extra_clips=extra_clips)
        write_json(path, prepared, overwrite=True)
        return prepared_bundle(working, prepared)
    finally:
        lock.rmdir()
