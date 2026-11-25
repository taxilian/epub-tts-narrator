"""Rerunning ``analyze BOOK`` checks each character's gender against the book and fixes wrong voice descriptions.

A voice is designed from its description, which begins with the gender
("Male, ..."). Each character's given gender is checked against paragraphs
that mention them (``rerun_review_prompts.GENDER_CHECK``: the gender the text
shows, with an exact quote), and a person's character notes
(``renderer/character_notes``, gender) win over the model. A wrong gender is
fixed in the description at once when the character has no voice yet, so the
voice is designed right. When a voice already exists the fix cannot change
it: the mismatch is shown and, in a terminal or the run shell, the
description may be fixed for the next recast ([F] fix the description /
[L] leave it; the run shell shows the description before and after,
``rerun_review_tui.GenderScreen``); without a terminal it is listed. Publication goes through the
``voice-profiles`` revision scope with backups (audition bindings follow).
Answers are cached in the book's request cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

GENDERS = ("Male", "Female", "Androgynous")
BATCH = 25


def given_gender(description):
    words = re.findall(r"[A-Za-z]+", description or "")
    return words[0] if words and words[0] in GENDERS else ""


def with_gender(description, gender):
    current = given_gender(description)
    if current:
        return re.sub(r"\b" + current + r"\b", gender, description, count=1)
    return f"{gender}, {description}" if description else gender


def check_genders(document, requests, rows, *, workers=1):
    """``{character_id: (gender, quote)}`` the text shows, for ``rows`` (cast characters)."""
    from .narrator_read import _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import GENDER_CHECK, GENDER_REVIEW_VERSION
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    numbers = _numbers(document["source"])
    items = []
    for index, row in enumerate(rows, 1):
        names = [name for name in [row["name"], *row.get("aliases", [])] if name.strip()]
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b", re.I)
        found = [block for block in document["source"]["blocks"] if pattern.search(block["text"])][:4]
        if not found:
            continue
        items.append({"id": f"g{index}", "who": "|".join([row["name"], ";".join(row.get("aliases", [])),
                                                         given_gender(voices[row["voice_id"]]["description"])]),
                      "text": _render_blocks(found, numbers), "_row": row, "_blocks": {numbers[b["id"]]: b for b in found}})
    batches = [items[start:start + BATCH] for start in range(0, len(items), BATCH)]

    def run(batch):
        payload = {"mode": "gender_review", "v": GENDER_REVIEW_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}
        # A quote may come from any paragraph the request showed: an item's
        # evidence is often in a paragraph listed under a neighbouring character.
        shown = {number: block for item in batch for number, block in item["_blocks"].items()}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,gender,N,quote]]}')
            result = {}
            for answer in answers:
                answer = _item_row(answer, "g")
                if not (isinstance(answer, list) and answer and answer[0] in by_id):
                    raise ValueError("each answer must start with an item id from the request")
                if answer[1:2] == ["?"]:
                    result[answer[0]] = None
                    continue
                if len(answer) < 4 or answer[1] not in GENDERS or type(answer[2]) is not int:
                    raise ValueError(f'{answer[0]}: give [item_id,"Male"|"Female"|"Androgynous",N,quote] or [item_id,"?"]')
                block = shown.get(answer[2])
                if block is None:
                    raise ValueError(f"{answer[0]}: paragraph {answer[2]} is not in this request; cite a paragraph shown here")
                if not isinstance(answer[3], str) or not answer[3].strip() or answer[3].strip() not in block["text"]:
                    raise ValueError(f"{answer[0]}: the quote must be exact text from paragraph {answer[2]}")
                result[answer[0]] = (answer[1], answer[3].strip())
            missing = sorted(set(by_id) - set(result))
            if missing:
                raise ValueError("missing answers for items: " + ", ".join(missing[:20]))
            return result
        return requests.run(GENDER_CHECK, payload, validate)
    answers = {}
    for result in _parallel(run, batches, workers):
        answers.update(result)
    return {item["_row"]["id"]: answers[item["id"]] for item in items if answers.get(item["id"])}


def _ask(question):
    while True:
        print(question, end="", flush=True)
        try:
            answer = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, OSError):
            answer = ""
        if answer == "":
            print()
            return "l"
        answer = answer.strip().casefold()[:1]
        if answer in {"f", "l"}:
            return answer
        print("Please enter F or L.")


def _decide(row, description, current, wanted, why):
    """"f" or "l": on the run shell's own screen when one is open, else on the line."""
    from .rerun_questions import NO_SCREEN, ask_on_screen, gender_view
    answer = ask_on_screen("GenderScreen", gender_view(row, description, current, wanted, why,
                                                       with_gender(description, wanted)))
    if answer is NO_SCREEN:
        return _ask("[F] fix the description now (used when you recast) / [L] leave it: ")
    return answer if answer in {"f", "l"} else "l"


def review_genders(directory, config, config_path, client_factory, progress, *, interactive=None, recheck_cache=None):
    """``analyze BOOK``: fix voice descriptions whose gender the book (or a person's notes) contradicts."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, revise_directories
    from .cast_review import _renderer_context
    from .pipeline import CachedRequests, fingerprint
    from .renderer.character_notes import _gender
    from .renderer.character_merge_command import _voice_state
    directory = Path(directory).resolve()
    interactive = (sys.stdin.isatty() and sys.stdout.isatty()) if interactive is None else interactive
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    rows = document["cast"]["characters"]
    if not rows:
        return False
    if not config["provider"]["model"]:
        raise ValueError("Rerunning analyze checks the cast's genders; set provider.model in epub-tts-narrator.toml, "
                         "or select a TOML file with --config")
    context = _renderer_context(directory, config_path)
    notes = context["notes"] if context else {}
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    settings = config["analysis"]
    client = client_factory(config["provider"])
    print(f"Checking the gender of {len(rows)} character(s) against the book with {client.model} at "
          f"{client.base_url}; passages naming them will be sent to this endpoint.", file=sys.stderr)
    requests = CachedRequests(client, recheck_cache or analysis_cache_directory(directory, settings["cache_dir"]),
                              repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    shown = check_genders(document, requests, rows, workers=min(5, settings["attribution_workers"]))
    fixes = {}
    for row in rows:
        voice = voices[row["voice_id"]]
        current = given_gender(voice["description"])
        wanted = {"female": "Female", "male": "Male"}.get(_gender((notes.get(row["voice_id"]) or {}).get("gender", "")))
        why = "your notes say so" if wanted else ""
        if not wanted and row["id"] in shown:
            wanted, quote = shown[row["id"]]
            why = f"the book says: “{quote}”"
        if not wanted or wanted == current:
            continue
        state = _voice_state(context["store"], row["voice_id"])[0] if context else "own"
        message = (f"Gender check: {row['name']}'s voice description reads {current or 'no gender'}, but "
                   f"{why} ({wanted}).")
        if state == "none":
            print(message + " Fixed in the description; the voice will be designed with it.")
            fixes[voice["id"]] = with_gender(voice["description"], wanted)
            continue
        print(message + " A voice already exists, so recast it (cast BOOK) to change how it sounds.")
        if interactive and _decide(row, voice["description"], current, wanted, why) == "f":
            fixes[voice["id"]] = with_gender(voice["description"], wanted)
    if not fixes:
        print("Gender check: every character's voice description agrees with the book.")
        return False
    revised = json.loads(json.dumps(document))
    for voice in revised["cast"]["voices"]:
        if voice["id"] in fixes:
            voice["description"] = fixes[voice["id"]]
    samples = revised["provenance"].get("voice_samples")
    if isinstance(samples, dict) and "cast_fingerprint" in samples:
        samples["cast_fingerprint"] = fingerprint(revised["cast"])
    sidecars = {}
    from .voice_profile_refresh import AUDITION_SIDECAR, audition_sidecar_rebind
    sidecar = directory / AUDITION_SIDECAR
    if sidecar.exists() or sidecar.is_symlink():
        sidecars = {directory: {AUDITION_SIDECAR: audition_sidecar_rebind(
            sidecar, source_fingerprint=fingerprint(document["source"]),
            cast_fingerprint_before=fingerprint(document["cast"]), cast_fingerprint_after=fingerprint(revised["cast"]),
            voice_ids=[voice["id"] for voice in revised["cast"]["voices"]])}}
    directories = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in directories}
    if hashes[str(directory)] != original_manifest_hash:
        raise ValueError(f"Analysis changed while its cast's genders were being checked: {directory}")
    try:
        result = revise_directories({directory: revised}, directories=directories, expected_manifest_hashes=hashes,
                                    scope="voice-profiles", sidecars=sidecars, progress=print)
    except (OSError, ValueError) as error:
        print(f"Gender check: the corrected descriptions were not saved: {error} Rerun analyze to try again; "
              "its answers are cached.")
        return False
    print(f"Gender check: {len(fixes)} voice description(s) corrected. Original analysis files and revision "
          f"journal: {result['backup_directory']}")
    return True
