"""``analyze BOOK --instructions "..."``: narration read in the voice of the character whose point of view it follows.

A book told in the third person often follows different characters in
different chapters or scenes. A person's notes about how the narration works
(``--instructions``) start this review: the book is split into sections at
chapters, headings and scene breaks (a paragraph without letters, such as
``#`` or ``* * *``), and each section's opening goes to one check
(``rerun_review_prompts.NARRATOR_CHECK``) that names the cast entry whose point
of view the section follows, with an exact quote, or ``-`` for the book's main
narrator. The narration of a section that follows a character then becomes a
narration role attached to that character and read in their voice
(``narrator_<character id>``, appended to the cast); headings, the main
narrator's sections and narration a person chose stay as they are. A terminal
or the run shell shows every section and asks first; without one the result is
only listed. The notes and each section's answer are kept in
``provenance.narrator_review``, so a later rerun checks only sections whose
text changed (with the saved notes) and new ``--instructions`` check them all
again. Publication goes through the ``narrators`` revision scope (backups,
later books follow). Answers are cached in the book's request cache.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

SECTION_CHARS = 6000
BATCH_CHARS = 40000
MAIN = "narrator_main"


def narrator_role_id(character_id):
    return f"narrator_{character_id}"


def sections(document):
    """The book's sections in order: ``{"title", "blocks"}``, split at chapters, headings and scene breaks."""
    from .narrator_read import HEADING_TAGS
    result, current, title = [], None, ""
    for block in document["source"]["blocks"]:
        heading = block["source"].get("tag") in HEADING_TAGS
        if heading:
            title = block["text"].strip()
        breaker = not re.search(r"\w", block["text"])
        if current is None or heading or breaker or block["chapter_id"] != current["chapter_id"]:
            if current is not None and current["chapter_id"] != block["chapter_id"] and not heading:
                title = ""
            current = {"chapter_id": block["chapter_id"], "title": title, "blocks": []}
            result.append(current)
        if not breaker:
            current["blocks"].append(block)
    counts = {}
    for section in result:
        counts[section["title"]] = counts.get(section["title"], 0) + 1
        section["part"] = counts[section["title"]]
        section["key"] = hashlib.sha256(json.dumps([[block["id"], block["text"]] for block in section["blocks"]],
                                                   ensure_ascii=False).encode()).hexdigest()
    return [section for section in result if section["blocks"]]


def _label(section):
    title = section["title"] or "Opening"
    return title if section["part"] == 1 else f"{title}, part {section['part']}"


def check_narrators(document, requests, notes, wanted, *, workers=1):
    """``{section key: (character_id, quote) or None}`` for the ``wanted`` sections."""
    from .narrator_read import _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import NARRATOR_CHECK, NARRATOR_REVIEW_VERSION
    characters = document["cast"]["characters"]
    cast_ids = {f"c{index}": row["id"] for index, row in enumerate(characters, 1)}
    cast = ["|".join([short, row["name"], ";".join(row.get("aliases", []))])
            for short, row in zip(cast_ids, characters)]
    numbers = _numbers(document["source"])
    items = []
    for index, section in enumerate(wanted, 1):
        shown, size = [], 0
        for block in section["blocks"]:
            if shown and size + len(block["text"]) > SECTION_CHARS:
                break
            shown.append(block)
            size += len(block["text"])
        items.append({"id": f"s{index}", "text": _render_blocks(shown, numbers),
                      "_section": section, "_blocks": {numbers[block["id"]]: block for block in shown}})
    batches, batch, size = [], [], 0
    for item in items:
        if batch and size + len(item["text"]) > BATCH_CHARS:
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += len(item["text"])
    if batch:
        batches.append(batch)

    def run(batch):
        payload = {"mode": "narrator_review", "v": NARRATOR_REVIEW_VERSION, "notes": notes, "cast": cast,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[section_id,cast_id,N,quote]]}')
            result = {}
            for answer in answers:
                answer = _item_row(answer, "s")
                if not (isinstance(answer, list) and answer and answer[0] in by_id):
                    raise ValueError("each answer must start with a section id from the request")
                if answer[1:2] == ["-"]:
                    result[answer[0]] = None
                    continue
                if len(answer) < 4 or answer[1] not in cast_ids or type(answer[2]) is not int:
                    raise ValueError(f'{answer[0]}: give [section_id,cast_id,N,quote] or [section_id,"-"]')
                block = by_id[answer[0]]["_blocks"].get(answer[2])
                if block is None or not isinstance(answer[3], str) or not answer[3].strip() \
                        or answer[3].strip() not in block["text"]:
                    raise ValueError(f"{answer[0]}: the quote must be exact text from paragraph {answer[2]}")
                result[answer[0]] = (cast_ids[answer[1]], answer[3].strip())
            missing = sorted(set(by_id) - set(result), key=lambda name: int(name[1:]))
            if missing:
                raise ValueError("missing answers for sections: " + ", ".join(missing[:20]))
            return result
        return requests.run(NARRATOR_CHECK, payload, validate)
    answers = {}
    for result in _parallel(run, batches, workers):
        answers.update(result)
    return {item["_section"]["key"]: answers[item["id"]] for item in items}


def apply_narrators(document, chosen):
    """``document`` with each section's narration read by ``chosen[section key]`` (a character ID, or None for the main narrator)."""
    revised = deepcopy(document)
    cast = revised["cast"]
    characters = {row["id"]: row for row in cast["characters"]}
    roles = {role["id"]: role for role in cast["roles"]}
    narration_roles = {role["id"] for role in cast["roles"] if role["kind"] == "narration"}
    if MAIN not in roles:
        raise ValueError("The book has no main narrator role (narrator_main)")
    narrator_of = {}
    for section in sections(revised):
        if section["key"] not in chosen:
            continue
        character_id = chosen[section["key"]]
        if character_id is None:
            role = roles[MAIN]
        else:
            role_id = narrator_role_id(character_id)
            if role_id not in roles:
                character = characters[character_id]
                roles[role_id] = {"id": role_id, "kind": "narration", "character_id": character_id,
                                  "voice_id": character["voice_id"],
                                  "description": f"Narration of the sections told from {character['name']}'s point of view."}
                cast["roles"].append(roles[role_id])
                narration_roles.add(role_id)
            role = roles[role_id]
        narrator_of.update(dict.fromkeys((block["id"] for block in section["blocks"]), role))
    for segment in revised["segments"]:
        role = narrator_of.get(segment["block_id"])
        if (role is None or segment["kind"] != "narration" or segment.get("role_id") not in narration_roles
                or segment["attribution"].get("basis") == "user"):
            continue
        segment["role_id"], segment["voice_id"] = role["id"], role["voice_id"]
    return revised


def validate_narrator_revision(document, candidate):
    """The ``narrators`` revision scope: narration moves between narration roles, and only narration roles are added."""
    from .analysis_revision import appended_cast
    if candidate["source"] != document["source"] or candidate["issues"] != document["issues"]:
        raise ValueError("Narrator revisions must preserve the source and issues")
    if appended_cast(document["cast"], candidate["cast"]) != [] \
            or candidate["cast"]["characters"] != document["cast"]["characters"]:
        raise ValueError("Narrator revisions may only add narration roles")
    characters = {row["id"]: row for row in candidate["cast"]["characters"]}
    for role in candidate["cast"]["roles"][len(document["cast"]["roles"]):]:
        character = characters.get(role.get("character_id"))
        if role.get("kind") != "narration" or character is None or role.get("voice_id") != character["voice_id"]:
            raise ValueError("A narrator revision's new role must narrate in an existing character's voice")
    narration = {role["id"] for role in candidate["cast"]["roles"] if role["kind"] == "narration"}
    if len(candidate["segments"]) != len(document["segments"]):
        raise ValueError("Narrator revisions must keep every segment")
    for before, after in zip(document["segments"], candidate["segments"]):
        if before == after:
            continue
        if ({key: value for key, value in before.items() if key not in {"role_id", "voice_id"}}
                != {key: value for key, value in after.items() if key not in {"role_id", "voice_id"}}
                or before["kind"] != "narration" or before.get("role_id") not in narration
                or after.get("role_id") not in narration or before["attribution"].get("basis") == "user"):
            raise ValueError("Narrator revisions may only move narration between narration roles")
    others = lambda record: {key: value for key, value in record.items() if key != "narrator_review"}
    if others(candidate["provenance"]) != others(document["provenance"]):
        raise ValueError("Narrator revisions may only change provenance.narrator_review")


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
        if answer in {"a", "l"}:
            return answer
        print("Please enter A or L.")


def review_narrators(directory, config, client_factory, progress, *, instructions=None, interactive=None):
    """Read each section's narration in its point-of-view character's voice; True when the book changed."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, revise_directories
    from .pipeline import CachedRequests
    from .rerun_review_prompts import NARRATOR_REVIEW_VERSION
    directory = Path(directory).resolve()
    interactive = (sys.stdin.isatty() and sys.stdout.isatty()) if interactive is None else interactive
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    saved = document["provenance"].get("narrator_review") or {}
    notes = (instructions or "").strip() or saved.get("notes", "")
    if not notes:
        return False
    if not 1 <= len(notes) <= 2000:
        raise ValueError("--instructions must be 1-2000 characters")
    if not document["cast"]["characters"]:
        print("Narrator check: the book has no cast characters to narrate.")
        return False
    known = ({row["key"]: row["narrator"] for row in saved.get("sections", [])}
             if saved.get("notes") == notes and saved.get("version") == NARRATOR_REVIEW_VERSION else {})
    book_sections = sections(document)
    wanted = [section for section in book_sections if section["key"] not in known]
    answers = {}
    if wanted:
        if not config["provider"]["model"]:
            raise ValueError("The narrator check needs a model; set provider.model in epub-tts-narrator.toml, "
                             "or select a TOML file with --config")
        settings = config["analysis"]
        client = client_factory(config["provider"])
        print(f"Checking whose point of view {len(wanted)} section(s) follow with {client.model} at "
              f"{client.base_url}; the opening of each section and your notes will be sent to this endpoint.",
              file=sys.stderr)
        requests = CachedRequests(client, analysis_cache_directory(directory, settings["cache_dir"]),
                                  repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        answers = check_narrators(document, requests, notes, wanted, workers=min(5, settings["attribution_workers"]))
    chosen = {section["key"]: known[section["key"]] if section["key"] in known
              else (answers[section["key"]][0] if answers[section["key"]] else None) for section in book_sections}
    revised = apply_narrators(document, chosen)
    names = {row["id"]: row["name"] for row in document["cast"]["characters"]}
    if revised["segments"] == document["segments"] and not wanted:
        print("Narrator check: every section's narration already has its narrator.")
        return False
    print("Narrator check: whose point of view each section follows:")
    for section in book_sections:
        reader = names.get(chosen[section["key"]], "main narrator")
        quote = answers.get(section["key"])
        print(f"  {_label(section)}: {reader}" + (f" (“{quote[1]}”)" if quote else ""))
    moved = sum(before != after for before, after in zip(document["segments"], revised["segments"]))
    if moved:
        print(f"Narrator check: {moved} narration passage(s) would be read in a different voice and rendered again.")
    # Saved answers were already accepted; only new ones are asked about.
    if wanted and moved and not interactive:
        print("Narrator check: nothing was saved; rerun analyze in a terminal to apply these narrators.")
        return False
    if wanted and moved and _ask("[A] apply these narrators / [L] leave the narration as it is: ") != "a":
        print("Narrator check: the narration was left as it is.")
        return False
    revised["provenance"]["narrator_review"] = {
        "version": NARRATOR_REVIEW_VERSION, "notes": notes,
        "sections": [{"key": section["key"], "narrator": chosen[section["key"]]} for section in book_sections]}
    directories = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in directories}
    if hashes[str(directory)] != original_manifest_hash:
        raise ValueError(f"Analysis changed while its narrators were being checked: {directory}")
    try:
        result = revise_directories({directory: revised}, directories=directories, expected_manifest_hashes=hashes,
                                    scope="narrators", progress=print)
    except (OSError, ValueError) as error:
        print(f"Narrator check: the narrators were not saved: {error} Rerun analyze to try again; its answers are cached.")
        return False
    print(f"Narrator check: saved; {moved} narration passage(s) now read by their point-of-view character. "
          f"Original analysis files and revision journal: {result['backup_directory']}")
    return True
