"""Prepare useful audition text without changing the book or character identities."""

from copy import deepcopy
import json
from pathlib import Path
import re
import unicodedata

from .attribution_review import _bounded_map
from .pipeline import CachedRequests, fingerprint, write_json
from .prompts import COMMON


VERSION = 1
PROMPT_VERSION = 2
FILENAME = "audition-scripts.json"
WORDS = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*", re.UNICODE)
SYSTEM = COMMON + """
Create an original character-specific audition script, not audiobook content.
The available book excerpts are too short or unsuitable for a reusable reference.
Write 35–50 words (hard bounds 25–60 words and 650 characters) that this character
could naturally say, using the supplied personality, setting, speech style and
short examples. For a narrator, write a short passage in their narrative style.
Use several complete sentences with varied natural phrasing and normal delivery.
Write plain spoken prose. Do not wrap the script or individual statements in
quotation marks; preserve quotation marks only when the character quotes wording
within their speech, and preserve normal apostrophes.
Do not recite the voice description, explain the task, introduce the speaker by
name, or include acting directions, headings, sound effects, brackets or markup.
Do not imitate shouting, repeated-letter hissing, sobbing or other extreme delivery.
This invented audition will never enter the audiobook or be claimed as a quotation.
Return exactly {"text":"audition speech only","reason":"brief casting rationale"}.
"""


def validate_script(value, *, require_plain_prose=True):
    from epub_tts_narrator.renderer.samples import _safe_text, has_repeated_speech_wrappers, strip_speech_wrapper
    if not isinstance(value, dict) or set(value) != {"text", "reason"}:
        raise ValueError("Audition script requires exactly text and reason")
    text, reason = value["text"], value["reason"]
    if (not isinstance(text, str) or text != text.strip() or not 25 <= len(WORDS.findall(text)) <= 60
            or len(text) > 650 or any(char in text for char in "[]<>")
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in text)
            or not _safe_text(text)):
        raise ValueError("Audition text must be 25–60 words, at most 650 characters, without control syntax")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
        raise ValueError("Audition script requires a brief casting rationale")
    text = strip_speech_wrapper(text)
    if require_plain_prose and has_repeated_speech_wrappers(text):
        raise ValueError("Audition script must use plain spoken prose, not separately quoted statements. "
                         "Remove the surrounding quotation marks from each statement; preserve only "
                         "meaningful quotations within speech and normal apostrophes.")
    return {"text": text, "basis": "casting", "reason": reason}


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
                    workers=5, progress=lambda message: None, refresh=False):
    """Prefer source auditions; explicitly refresh selected groups when requested.

    Refresh drops creative fallbacks only for the selected canonical groups.
    Their replacement scripts (or absence, when source speech suffices) override
    every older inline/provenance copy without rewriting the analysis bundle.
    """
    from epub_tts_narrator.renderer.samples import select_book_sample, select_sample
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Audition workers must be between one and five")
    scripts = _saved(bundle)
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
        profile = deepcopy(profiles[voice])
        if refresh:
            profile.pop("sample_script", None)
        payload = {"mode": "audition_script", "version": PROMPT_VERSION,
                   "book": bundle["source"]["book"], "voice": profile,
                   "characters": people, "roles": roles, "short_speech_examples": examples,
                   "source_fingerprint": bundle["manifest"]["source_fingerprint"]}
        tasks.append((aliases, payload))
    def create(task, cancelled):
        aliases, payload = task
        script = requests.run(SYSTEM, payload, validate_script, cancelled=cancelled)
        progress(f"Prepared character-specific audition script: {payload['voice']['id']}")
        return aliases, script
    if tasks:
        if requests is None:
            raise ValueError("Character-specific audition scripts need the configured analysis model")
        progress(f"Preparing {len(tasks)} missing character audition script(s)")
        for aliases, script in _bounded_map(tasks, create, workers):
            scripts.update({value: deepcopy(script) for value in aliases})
    samples = {voice: select_sample(working, voice, canonical_id=canonical_id, fallback_text="") for voice in selected}
    result = {"version": VERSION, "source_fingerprint": bundle["manifest"]["source_fingerprint"],
              "cast_fingerprint": fingerprint(bundle["cast"]), "scripts": scripts, "samples": samples}
    if replaced:
        result["replaced_voice_ids"] = sorted(replaced)
    return result


def prepare_document_samples(document, requests, *, workers=5, progress=lambda message: None):
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
    result["provenance"]["voice_samples"] = prepare_samples(bundle, requests, workers=workers, progress=progress)
    return result


def ensure_bundle_scripts(bundle, config_path=None, *, canonical_id=lambda value: value,
                          voice_ids=None, progress=print, refresh=False):
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
        _overrides(working)
        return working

    def prepared_bundle(working, prepared):
        return dict(working, audition_scripts=prepared["scripts"],
                    audition_script_overrides=prepared.get("replaced_voice_ids", []))

    working = load_working()
    # Fast path: no configuration or network is needed for prepared books.
    # Explicit refresh always persists its overrides, including source-only runs.
    if not refresh:
        try:
            prepared = prepare_samples(working, None, canonical_id=canonical_id, voice_ids=voice_ids)
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
                                       refresh=refresh)
        except ValueError as error:
            if str(error) != "Character-specific audition scripts need the configured analysis model":
                raise
            config = load_config(config_path)
            settings = config["analysis"]
            client = OpenAICompatibleClient(**config["provider"])
            cache = settings["cache_dir"] or directory.parent / ".epub-audiobook-cache"
            progress(f"Preparing audition scripts with {client.model} at {client.base_url}; character descriptions and excerpts are sent to this endpoint.")
            requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"], progress=progress)
            prepared = prepare_samples(working, requests, canonical_id=canonical_id, voice_ids=voice_ids,
                                       workers=min(5, settings["attribution_workers"]), progress=progress,
                                       refresh=refresh)
        write_json(path, prepared, overwrite=True)
        return prepared_bundle(working, prepared)
    finally:
        lock.rmdir()
