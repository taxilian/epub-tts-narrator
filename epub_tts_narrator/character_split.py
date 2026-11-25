"""Rerunning ``analyze BOOK --instructions`` splits a cast entry that holds two characters' lines.

An entry can join two characters: the "third voice" whispering behind a door
given to the eaterling that spoke in an earlier scene, because both are called
the third voice. A merge cannot fix that and the attribution review chooses
only among existing characters, so this step does, and only when a person's
instructions say so:

1. one request reads the cast with the instructions (``SPLIT_NOMINATE``) and
   names each entry the notes say gives one character's lines to another;
2. for each, one request with every paragraph holding the entry's lines
   (``SPLIT_CHECK``) answers which paragraphs' lines are the other
   character's, that character's name and a one-sentence description, and an
   exact quote that shows they are someone else;
3. the person is asked (S split / K keep / L later; in the run shell
   ``rerun_review_tui.SplitScreen``). Splitting gives those lines to the
   character the book already has by that name, else to a new character with
   its own voice (series-stable IDs, like a new character chosen for a
   passage), published through revision scope ``character-split``. Keep is
   remembered in the book's ``character-splits.json`` for those lines; without
   a terminal the split is only listed.

A line a person assigned moves only with ``reconsider_choices``
(``--reconsider-choices``); otherwise it stays and is reported. Answers are
cached in the book's request cache. Stdlib only.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

KEPT_FILENAME = "character-splits.json"
MAX_PARAGRAPHS = 60


def _roles_of(document, character_id):
    return {role["id"] for role in document["cast"]["roles"]
            if role.get("character_id") == character_id and role.get("kind") == "dialogue"}


def _performers(row):
    return [row.get("role_id"), *[item["role_id"] for item in row.get("simultaneous_speakers", [])]]


def lines_of(document, character_id):
    """Block IDs, in book order, of the paragraphs holding the entry's own spoken lines."""
    roles = _roles_of(document, character_id)
    return list(dict.fromkeys(row["block_id"] for row in document["segments"]
                              if row.get("kind") != "narration" and roles & set(_performers(row))))


def nominate(document, requests, instructions):
    """``[(character_id, who)]``: entries the instructions say hold another character's lines."""
    from .character_merge import first_line
    from .narrator_read import _unwrap
    from .rerun_review_prompts import SPLIT_NOMINATE, SPLIT_REVIEW_VERSION
    rows = [row for row in document["cast"]["characters"] if lines_of(document, row["id"])]
    if not rows:
        return []
    labels = {f"c{index}": row for index, row in enumerate(rows, 1)}
    cast = "\n".join("|".join([label, row["name"], ";".join(row.get("aliases", [])),
                               " ".join(row.get("description", "").split()), first_line(document, row["id"]) or ""])
                     for label, row in labels.items())
    payload = {"mode": "character_split_nominate", "v": SPLIT_REVIEW_VERSION, "cast": cast,
               "instructions": instructions}

    def validate(response):
        found = _unwrap(response.get("r")) if isinstance(response, dict) else None
        if not isinstance(found, list):
            raise ValueError('response must be {"r":[[id,"who"]]}')
        result = []
        for row in found:
            if not (isinstance(row, list) and len(row) >= 2 and isinstance(row[1], str) and row[1].strip()):
                raise ValueError('each entry must be [id,"who"]')
            label = f"c{row[0]}" if type(row[0]) is int or (isinstance(row[0], str) and row[0].isdigit()) else row[0]
            if label not in labels:
                raise ValueError(f"{row[0]!r} is not a cast id from the request")
            result.append((labels[label]["id"], " ".join(row[1].split())[:200]))
        return list(dict.fromkeys(result))
    return requests.run(SPLIT_NOMINATE, payload, validate)


def check(document, requests, nominated, instructions, *, workers=1):
    """``{character_id: {"name", "description", "blocks", "evidence"}}`` for entries the text shows hold two characters."""
    from .narrator_read import _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import SPLIT_CHECK, SPLIT_REVIEW_VERSION
    numbers = _numbers(document["source"])
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    characters = {row["id"]: row for row in document["cast"]["characters"]}
    items = []
    for index, (character_id, who) in enumerate(nominated, 1):
        shown = [blocks[block_id] for block_id in lines_of(document, character_id)[:MAX_PARAGRAPHS]]
        row = characters[character_id]
        items.append({"id": f"s{index}", "entry": "|".join([row["name"], ";".join(row.get("aliases", [])),
                                                             " ".join(row.get("description", "").split())]),
                      "who": who, "text": _render_blocks(shown, numbers), "_row": row,
                      "_blocks": {numbers[block["id"]]: block for block in shown}})

    def run(item):
        payload = {"mode": "character_split_check", "v": SPLIT_REVIEW_VERSION, "instructions": instructions,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")}]}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not (isinstance(answers, list) and answers):
                raise ValueError('response must be {"r":[[item_id,name,description,[N...],N,quote]]}')
            answer = _item_row(answers[0], "s")
            if not (isinstance(answer, list) and answer and answer[0] == item["id"]):
                raise ValueError(f"answer for item {item['id']}")
            if answer[1:2] in (["n"], ["?"]):
                return None
            if (len(answer) < 6 or not all(isinstance(value, str) and value.strip() for value in answer[1:3])
                    or not isinstance(answer[3], list) or not answer[3] or type(answer[4]) is not int):
                raise ValueError(f'{item["id"]}: give [item_id,name,description,[N...],N,quote], [item_id,"n"] or [item_id,"?"]')
            moved = []
            for number in answer[3]:
                if type(number) is not int or number not in item["_blocks"]:
                    raise ValueError(f"{item['id']}: paragraph {number!r} is not one shown in this item")
                moved.append(item["_blocks"][number]["id"])
            block = item["_blocks"].get(answer[4])
            if block is None:
                raise ValueError(f"{item['id']}: paragraph {answer[4]} is not one shown in this item")
            quote = answer[5].strip() if isinstance(answer[5], str) else ""
            if not quote or quote not in block["text"]:
                raise ValueError(f"{item['id']}: the quote must be exact text from paragraph {answer[4]}")
            if len(set(moved)) == len(item["_blocks"]):
                raise ValueError(f"{item['id']}: if every line is the other character's, this is a rename, not a "
                                 "split; list only the paragraphs whose lines are theirs")
            name = " ".join(answer[1].split())[:80]
            description = " ".join(answer[2].split())[:400]
            return {"name": name, "description": description, "blocks": list(dict.fromkeys(moved)),
                    "evidence": {"block_id": block["id"], "quote": quote}}
        return item["_row"]["id"], requests.run(SPLIT_CHECK, payload, validate)
    return {character_id: found for character_id, found in _parallel(run, items, workers) if found}


def _key(name):
    return " ".join(name.casefold().split())


def split_document(document, character_id, found, *, reconsider_choices=False):
    """``(candidate, moved segment IDs, kept person-chosen segment IDs, target character, created)``."""
    from .speaker_decisions import _create_voice_records
    candidate = deepcopy(document)
    cast = candidate["cast"]
    source = next(row for row in cast["characters"] if row["id"] == character_id)
    target = next((row for row in cast["characters"] if row["id"] != character_id and _key(found["name"]) in
                   {_key(name) for name in [row["name"], *row.get("aliases", [])]}), None)
    created = target is None
    if created:
        target, _, role = _create_voice_records(candidate, found["name"], found["description"])
    else:
        role = next((row for row in cast["roles"] if row.get("character_id") == target["id"]
                     and row.get("kind") == "dialogue"), None)
        if role is None:
            raise ValueError(f"{target['name']} has no speaking role to take these lines")
    old_roles = _roles_of(document, character_id)
    wanted = set(found["blocks"])
    moved, kept = [], []
    for row in candidate["segments"]:
        if row["block_id"] not in wanted or row.get("kind") == "narration" or not old_roles & set(_performers(row)):
            continue
        if row["attribution"].get("basis") == "user" and not reconsider_choices:
            kept.append(row["id"])
            continue
        reason = f"The book shows {target['name']}, not {source['name']}, says this."
        if row.get("simultaneous_speakers"):
            members = [dict(item, role_id=role["id"], voice_id=role["voice_id"]) if item["role_id"] in old_roles
                       else item for item in row["simultaneous_speakers"]]
            if len({item["voice_id"] for item in members}) < len(members):
                kept.append(row["id"])
                continue
            row["simultaneous_speakers"] = members
        else:
            row["role_id"], row["voice_id"] = role["id"], role["voice_id"]
            for key in ("voice_effect",):
                row.pop(key, None)
        row["attribution"] = {"basis": "context", "reason": reason, "evidence": [deepcopy(found["evidence"])],
                              "candidates": [], "review_required": False}
        moved.append(row["id"])
    return candidate, moved, kept, target, created


def validate_character_split(document, candidate):
    """Refuse a ``character-split`` revision that does more than give some lines to one other character.

    Returns the voice IDs it adds (none, or the new character's one voice).
    """
    from .analysis_revision import appended_cast
    if candidate["source"] != document["source"] or candidate["provenance"] != document["provenance"]:
        raise ValueError("A character split may not change the source or provenance")
    added = appended_cast(document["cast"], candidate["cast"])
    if added is None or len(added) > 1 or len(candidate["cast"]["characters"]) - len(document["cast"]["characters"]) > 1:
        raise ValueError("A character split may only add one character, with its voice and role, after the saved cast")
    before = {row["id"]: row for row in document["segments"]}
    if [row["id"] for row in candidate["segments"]] != [row["id"] for row in document["segments"]]:
        raise ValueError("A character split may not add, remove or reorder lines")
    changed = 0
    for row in candidate["segments"]:
        old = before[row["id"]]
        if row == old:
            continue
        loose = {key: value for key, value in row.items()
                 if key not in {"role_id", "voice_id", "attribution", "simultaneous_speakers", "voice_effect"}}
        old_loose = {key: value for key, value in old.items()
                     if key not in {"role_id", "voice_id", "attribution", "simultaneous_speakers", "voice_effect"}}
        if loose != old_loose or bool(row.get("simultaneous_speakers")) != bool(old.get("simultaneous_speakers")):
            raise ValueError(f"A character split may only change who speaks line {row['id']}")
        changed += 1
    if not changed:
        raise ValueError("A character split must give at least one line to another character")
    return added


def kept(directory):
    """``{character id: [kept paragraph sets]}`` a person chose not to split."""
    path = Path(directory) / KEPT_FILENAME
    if path.is_symlink() or not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in record.get("kept", []):
        result.setdefault(row["character_id"], []).append(set(row["blocks"]))
    return result


def keep(directory, character_id, blocks, name):
    path = Path(directory) / KEPT_FILENAME
    record = {"version": 1, "kept": []}
    if path.is_file() and not path.is_symlink():
        record = json.loads(path.read_text(encoding="utf-8"))
    record["kept"].append({"character_id": character_id, "blocks": sorted(blocks), "name": name})
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        if answer in {"s", "k", "l"}:
            return answer
        print("Please enter S, K or L.")


def _decide(view):
    """"s", "k" or "l": on the run shell's own screen when one is open, else on the line."""
    from .rerun_questions import NO_SCREEN, ask_on_screen
    answer = ask_on_screen("SplitScreen", view)
    if answer is NO_SCREEN:
        print(f"\n{view['title']}\n  {view['summary']}\n  {view['why']}")
        for line in view["lines"]:
            print(f"    {line}")
        for note in view["notes"]:
            print(f"  {note}")
        return _ask("[S] split them / [K] keep them together / [L] decide later: ")
    return answer if answer in {"s", "k", "l"} else "l"


def review_splits(directory, config, client_factory, progress, *, instructions=None, reconsider_choices=False,
                  interactive=None):
    """``analyze BOOK --instructions``: split entries the notes and the text show hold two characters; True when saved."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, revise_directories
    from .pipeline import CachedRequests
    from .rerun_questions import split_view
    from .tui_support import active_screen_host
    instructions = (instructions or "").strip()
    if not instructions:
        return False
    directory = Path(directory).resolve()
    interactive = (sys.stdin.isatty() and sys.stdout.isatty()) if interactive is None else interactive
    if not config["provider"]["model"]:
        return False
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    settings = config["analysis"]
    client = client_factory(config["provider"])
    print(f"Checking whether your instructions name cast entries that hold two characters' lines, with "
          f"{client.model} at {client.base_url}; the cast and those entries' paragraphs are sent to this endpoint.",
          file=sys.stderr)
    requests = CachedRequests(client, analysis_cache_directory(directory, settings["cache_dir"]),
                              repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    nominated = nominate(document, requests, instructions)
    if not nominated:
        return False
    found = check(document, requests, nominated, instructions, workers=min(5, settings["attribution_workers"]))
    remembered = kept(directory)
    changed = False
    for character_id, split in found.items():
        if set(split["blocks"]) in remembered.get(character_id, []):
            continue
        candidate, moved, held, target, created = split_document(document, character_id, split,
                                                                 reconsider_choices=reconsider_choices)
        if not moved:
            continue
        view = split_view(document, character_id, split, target, created, moved, held)
        if not (interactive or active_screen_host() is not None):
            print(f"Character split: {view['summary']} Rerun analyze in a terminal to decide.")
            continue
        answer = _decide(view)
        if answer == "k":
            keep(directory, character_id, split["blocks"], split["name"])
        if answer != "s":
            continue
        directories = related_directories([directory])
        hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in directories}
        if hashes[str(directory)] != original_manifest_hash:
            raise ValueError(f"Analysis changed while cast entries were being checked: {directory}")
        revise_directories({directory: candidate}, directories=directories, expected_manifest_hashes=hashes,
                           scope="character-split", progress=progress)
        print(f"Character split: {view['summary']}" + (" Cast the new voice with cast BOOK." if created else ""))
        changed = True
        # The next split starts from the saved book.
        original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        document, _, _ = _completed_directory(directory)
    return changed
