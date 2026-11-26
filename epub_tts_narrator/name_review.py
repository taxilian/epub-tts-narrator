"""A character listed under a description is called by the personal name the book gives it.

The narrator read names an entry when it first meets the character, often by
a description ("Eosian Beacon", "the old guard"); when the book later gives
its name ("I believe my name is Aurora") the read adds that name as an alias
but keeps the description as the name. Two requests fix it:
``rerun_review_prompts.NAME_NOMINATE`` reads the cast (names, aliases, notes)
and proposes a personal name for entries listed under a description, and
``NAME_CHECK`` shows, for each, the entry's own lines and the paragraphs using
the proposed name: only "y" with an exact quote containing the name, showing it
is this character's own name, renames it.

The fresh narrator read applies the names before the book is published
(``apply_names``: the analysis's own choice, no rename record). Rerunning
``analyze BOOK`` does it for saved books (``review_names``), through
``rename-character``'s plan and publication with ``basis: "analysis"`` on the
record, so later books with the old name follow; earlier books keep theirs
(a character book 1 called "Red Mouse" that book 3 names is Nestor from book 3
on). Characters a person renamed (a rename record without a basis) are never
renamed. A rename changes only words: nothing is rendered again.
"""

from __future__ import annotations

from .analysis_revision import BookBusy
import re
import sys
from pathlib import Path

BATCH = 20
NAME_SHOWN, LINES_SHOWN, CURRENT_SHOWN = 4, 3, 2


def person_renamed(document):
    """Character IDs a person renamed (records without a basis)."""
    return {record.get("character_id") for record in document["provenance"].get("character_renames", [])
            if isinstance(record, dict) and not record.get("basis")}


_DETERMINERS = {"the", "a", "an", "my", "your", "his", "her", "its", "our", "their", "this", "that"}
_NAMED = re.compile(r"\b(?:named|called)\s+([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*)")


def name_on_hand(row):
    """Whether the entry carries a personal name besides its listed one: a capitalized alias sharing no word with
    the listed name ("Aurora" for "Eosian Beacon"), or a note saying it is "named"/"called" someone."""
    listed = {word.casefold() for word in re.findall(r"\w+", row["name"])}
    for alias in row.get("aliases", []):
        words = re.findall(r"[\w'’-]+", alias)
        if (words and all(word[:1].isupper() for word in words) and words[0].casefold() not in _DETERMINERS
                and not listed & {word.casefold() for word in words}):
            return True
    return bool(_NAMED.search(row.get("description", "")))


def eligible(document):
    """The cast characters the analysis may rename: a personal name is on hand (``name_on_hand``) and a person
    never renamed them (narration excepted)."""
    renamed = person_renamed(document)
    narration = {role["voice_id"] for role in document["cast"]["roles"] if role.get("kind") == "narration"
                 and role.get("character_id") is None}
    return [row for row in document["cast"]["characters"]
            if row["id"] not in renamed and row["voice_id"] not in narration and name_on_hand(row)]


def _word(name):
    return re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)")


def _names(row):
    return [name for name in [row["name"], *row.get("aliases", [])] if name and name.strip()]


def _spoken(document, row):
    from .character_merge import character_roles
    roles = {role["id"] for role in character_roles(document, row["id"])}
    return list(dict.fromkeys(segment["block_id"] for segment in document["segments"]
                              if segment.get("role_id") in roles
                              or any(member.get("role_id") in roles
                                     for member in segment.get("simultaneous_speakers", []))))


def _spread(values, count):
    if len(values) <= count:
        return list(values)
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def _passages(document, row, name, numbers):
    """The entry's own lines and listed names, and the paragraphs that use the proposed name."""
    blocks = document["source"]["blocks"]
    by_id = {block["id"]: block for block in blocks}
    found = [block for block in blocks if _word(name).search(block["text"])][:NAME_SHOWN]
    found += [by_id[block_id] for block_id in _spread(_spoken(document, row), LINES_SHOWN)]
    others = [item for item in _names(row) if item != name]
    if others:
        listed = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(item) for item in others) + r")(?!\w)")
        found += [block for block in blocks if listed.search(block["text"])][:CURRENT_SHOWN]
    unique = list({block["id"]: block for block in found}.values())
    return sorted(unique, key=lambda block: numbers[block["id"]])


def nominate(document, rows, run):
    """``{character_id: name}`` the cast's names, aliases and notes suggest for entries listed under a description."""
    from .character_merge import name_key
    from .narrator_read import _item_row, _unwrap
    from .rerun_review_prompts import NAME_NOMINATE, NAME_REVIEW_VERSION
    if not rows:
        return {}
    labels = {f"c{index}": row for index, row in enumerate(rows, 1)}
    cast = "\n".join("|".join([label, row["name"], ";".join(row.get("aliases", [])),
                               " ".join(row.get("description", "").split())]) for label, row in labels.items())
    taken = {name_key(name): row["id"] for row in document["cast"]["characters"] for name in _names(row)}
    text = "\n".join(block["text"] for block in document["source"]["blocks"])

    def validate(response):
        answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
        if not isinstance(answers, list):
            raise ValueError('response must be {"r":[[id,name]]}')
        result = {}
        for answer in answers:
            answer = _item_row(answer, "c")
            if not (isinstance(answer, list) and len(answer) >= 2 and answer[0] in labels
                    and isinstance(answer[1], str)):
                raise ValueError("each answer must be [cast id from the request, name]")
            row, name = labels[answer[0]], " ".join(answer[1].split())
            # Only a name the book uses, that is not this entry's name already or another entry's.
            if (not name or len(name) > 80 or name_key(name) == name_key(row["name"])
                    or taken.get(name_key(name), row["id"]) != row["id"] or not _word(name).search(text)):
                continue
            result[row["id"]] = name
        return result
    return run(NAME_NOMINATE, {"mode": "name_review_nominate", "v": NAME_REVIEW_VERSION, "cast": cast}, validate)


def check_names(document, proposals, run, *, workers=1):
    """``{character_id: (name, quote)}`` for proposals the text shows are the character's own name."""
    from .narrator_read import _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import NAME_CHECK, NAME_REVIEW_VERSION
    if not proposals:
        return {}
    numbers = _numbers(document["source"])
    rows = {row["id"]: row for row in document["cast"]["characters"]}
    items = []
    for index, (character_id, name) in enumerate(proposals.items(), 1):
        row = rows[character_id]
        shown = _passages(document, row, name, numbers)
        items.append({"id": f"n{index}", "who": "|".join([row["name"], ";".join(row.get("aliases", [])),
                                                         " ".join(row.get("description", "").split())]),
                      "name": name, "text": _render_blocks(shown, numbers),
                      "_row": character_id, "_name": name, "_blocks": {numbers[block["id"]]: block for block in shown}})
    batches = [items[start:start + BATCH] for start in range(0, len(items), BATCH)]

    def ask(batch):
        payload = {"mode": "name_review_check", "v": NAME_REVIEW_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}
        shown = {number: block for item in batch for number, block in item["_blocks"].items()}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,"y",N,quote]|[item_id,"n"|"?"]]}')
            result = {}
            for answer in answers:
                answer = _item_row(answer, "n")
                if not (isinstance(answer, list) and len(answer) >= 2 and answer[0] in by_id
                        and answer[1] in {"y", "n", "?"}):
                    continue
                if answer[1] != "y":
                    result[answer[0]] = None
                    continue
                block = shown.get(answer[2]) if len(answer) >= 4 and type(answer[2]) is int else None
                quote = answer[3].strip() if len(answer) >= 4 and isinstance(answer[3], str) else ""
                name = by_id[answer[0]]["_name"]
                if block is None or not quote or quote not in block["text"] or not _word(name).search(quote):
                    raise ValueError(f'{answer[0]}: "y" needs [item_id,"y",N,quote] with an exact quote from a '
                                     f"paragraph shown here that contains {name!r}")
                result[answer[0]] = quote
            missing = sorted(set(by_id) - set(result), key=lambda value: int(value[1:]))
            if missing:
                raise ValueError("missing or invalid answers for items: " + ", ".join(missing[:20]))
            return result
        return run(NAME_CHECK, payload, validate)
    answers = {}
    for result in _parallel(ask, batches, workers):
        answers.update(result)
    return {item["_row"]: (item["_name"], answers[item["id"]]) for item in items if answers.get(item["id"])}


def find_names(document, run, *, workers=1):
    """``{character_id: (name, quote)}``: personal names the book gives entries listed under a description."""
    return check_names(document, nominate(document, eligible(document), run), run, workers=workers)


def apply_names(document, run, *, workers=1, progress=lambda message: None):
    """A fresh analysis: ``(document, [{character_id, from, to, quote}])`` with the names the book gives.

    The analysis's own choice, so no rename record is kept; the description stays an alias when the text uses it.
    """
    from .character_rename import rename_document
    named = []
    for character_id, (name, quote) in find_names(document, run, workers=workers).items():
        old = next(row["name"] for row in document["cast"]["characters"] if row["id"] == character_id)
        try:
            renamed = rename_document(document, character_id, name)
        except ValueError as error:
            progress(f"Names: {old} is called {name} in the book, but it was not renamed: {error}")
            continue
        records = renamed["provenance"]["character_renames"]
        records.pop()
        if not records:
            del renamed["provenance"]["character_renames"]
        document = renamed
        named.append({"character_id": character_id, "from": old, "to": name, "quote": quote})
        progress(f"Names: {old} is called {name} (“{quote}”)")
    return document, named


def review_names(directory, config, client_factory, progress, *, recheck_cache=None):
    """``analyze BOOK``: rename entries listed under a description to the personal name the book gives; True when saved."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .pipeline import CachedRequests
    from .renderer.character_rename_command import apply_rename, plan_rename
    directory = Path(directory).resolve()
    document, _, _ = _completed_directory(directory)
    rows = eligible(document)
    if not rows:
        return False
    if not config["provider"]["model"]:
        print("Names: set provider.model to check whether the book names characters listed under a description.")
        return False
    settings = config["analysis"]
    client = client_factory(config["provider"])
    print(f"Checking whether the book names {len(rows)} character(s) listed under a description, with "
          f"{client.model} at {client.base_url}; cast lines and passages will be sent to this endpoint.",
          file=sys.stderr)
    requests = CachedRequests(client, recheck_cache or analysis_cache_directory(directory, settings["cache_dir"]),
                              repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    found = find_names(document, requests.run, workers=min(5, settings["attribution_workers"]))
    changed = False
    for character_id, (name, quote) in found.items():
        old = next(row["name"] for row in document["cast"]["characters"] if row["id"] == character_id)
        try:
            plan = plan_rename(directory, character_id, name, basis="analysis")
            apply_rename(plan, progress=print)
        except BookBusy:
            raise
        except (ValueError, OSError) as error:
            print(f"Names: {old} is called {name} in the book (“{quote}”), but it was not renamed: {error}")
            continue
        print(f"Names: renamed {old} to {name}, the name the book gives (“{quote}”); its voice and lines stay. "
              f"rename-character changes it back.")
        changed = True
    if not found:
        print("Names: every character the book names is listed by its name.")
    return changed
