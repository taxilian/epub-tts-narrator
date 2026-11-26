"""A line nobody speaks goes to a character the book already knows, named beside it.

Only characters with lines join a book's cast. A character the narrator read
knew (its ledger entry, often matched to an earlier book's character) whose
speech the read took for narration, such as italic speech after an action
beat (*Topa flapped his wings. Not right now!*), is left out; a later check
that finds the speech can then choose only the cast and leaves the line
unassigned ("no Topa dialogue role exists"). The same holds for a character
only earlier books have.

``absent_characters`` lists those known characters: this book's ledger
entries without a cast entry, and characters only earlier books of the series
have. Rerunning ``analyze BOOK`` (``review_known_speakers``, after the speaker
check) asks, for each unassigned line with one of them named in its paragraph
or the ``BEFORE`` paragraphs before it, whether one of the characters named
there says it (``rerun_review_prompts.KNOWN_SPEAKER_CHECK``, an exact quote
naming the character, from the paragraphs shown). The cast characters named
there are choices too, so the model is never forced to an absent one; a line
it gives to a cast character is left for the speaker completion and a person.
A chosen absent character joins the cast: an earlier book's with that book's
records and voice (``earlier_records``, as a merge takes one in), a new one
from the ledger with a designed voice. Publication is scope ``known-speakers``
(``validate_known_speakers``).

A person choosing speakers is offered the earlier books' characters too
(``speaker_decisions``), taken in through the same ``earlier_records``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import re
import sys
import uuid
from pathlib import Path

from .pipeline import fingerprint

BEFORE = 2
BATCH = 12
SPOKEN_KINDS = {"dialogue", "thought", "letter"}


def _names(row):
    return [name for name in [row.get("name", ""), *row.get("aliases", [])] if isinstance(name, str) and name.strip()]


def _proper(name):
    """A name the text writes capitalized: matched case-sensitively as a whole word."""
    return len(name) >= 2 and name[:1].isupper()


def _word(name):
    return re.compile(r"(?<![\w’'])" + re.escape(name) + r"(?![\w])")


def earlier_books(registry, document):
    """The series books before this one (the registry may or may not hold this book yet)."""
    if not registry:
        return []
    here = {fingerprint(document["source"]), (document["provenance"].get("series") or {}).get("source_fingerprint")}
    return [book for book in registry["books"] if book["source_fingerprint"] not in here]


def role_id(character_id):
    return "role_known_" + hashlib.sha256(character_id.encode("utf-8")).hexdigest()[:16]


def earlier_records(registry, document, character_id):
    """``{"character", "voice", "role", "book"}`` an earlier book's character brings into this book, or None.

    The nearest earlier book's character and voice records, as a merge takes
    them in (``character_merge.kept_records_for``), and a dialogue role.
    """
    from .character_merge import carried_voice
    for book in reversed(earlier_books(registry, document)):
        character = next((row for row in book["cast"]["characters"] if row["id"] == character_id), None)
        if character is None:
            continue
        voice = next((row for row in book["cast"]["voices"] if row["id"] == character["voice_id"]), None)
        if voice is None:
            return None
        carried = {key: deepcopy(value) for key, value in character.items() if key not in {"evidence", "series_identity"}}
        carried["evidence"] = []
        return {"character": carried,
                "voice": carried_voice(voice, series_id=registry["series_id"], source_fingerprint=book["source_fingerprint"]),
                "role": {"id": role_id(character_id), "kind": "dialogue", "character_id": character_id,
                         "voice_id": character["voice_id"], "description": f"Dialogue role for {character['name']}."},
                "book": book["book"].get("title") or book["source_fingerprint"][:12]}
    return None


def _ledger(document):
    read = document["provenance"].get("narrator_read") or {}
    return [row for row in (read.get("ledger") or {}).get("characters", []) if isinstance(row, dict)]


def absent_characters(document, registry):
    """Known characters not in the cast, nearest first: ``[{"key", "name", "aliases", "note", "records", "ledger"}]``.

    ``records`` (``earlier_records``) when an earlier book has the character,
    else None and ``ledger`` is this book's ledger entry (a new character).
    """
    from .character_merge import name_key
    cast = document["cast"]["characters"]
    ids = {row["id"] for row in cast}
    names = {name_key(name) for row in cast for name in _names(row)}
    found, seen = [], set(ids)

    def present(row):
        return any(name_key(name) in names for name in _names(row))
    for row in _ledger(document):
        if row.get("series") in ids or present(row):
            continue
        records = (earlier_records(registry, document, row["series"])
                   if row.get("series") and row.get("series_confirmed") else None)
        if records is not None:
            if records["character"]["id"] in seen:
                continue
            seen.add(records["character"]["id"])
            found.append({"key": records["character"]["id"], "name": records["character"]["name"],
                          "aliases": list(records["character"].get("aliases", [])),
                          "note": row.get("note") or records["character"].get("description", ""),
                          "records": records, "ledger": row})
        elif isinstance(row.get("id"), str) and isinstance(row.get("name"), str):
            found.append({"key": "ledger:" + row["id"], "name": row["name"], "aliases": list(row.get("aliases", [])),
                          "note": row.get("note") or "", "records": None, "ledger": row})
    for book in reversed(earlier_books(registry, document)):
        for row in book["cast"]["characters"]:
            if row["id"] in seen or present(row):
                continue
            seen.add(row["id"])
            records = earlier_records(registry, document, row["id"])
            if records is not None:
                found.append({"key": row["id"], "name": row["name"], "aliases": list(row.get("aliases", [])),
                              "note": row.get("description", ""), "records": records, "ledger": None})
    return found


def named_in(text, row):
    """Whether ``text`` names ``row`` by one of its capitalized names."""
    return any(_proper(name) and _word(name).search(text) for name in _names(row))


def _window(document, segment, positions):
    """The line's paragraph and the ``BEFORE`` paragraphs before it in its chapter."""
    blocks = document["source"]["blocks"]
    index = positions[segment["block_id"]]
    chapter = blocks[index]["chapter_id"]
    return [block for block in blocks[max(0, index - BEFORE):index + 1] if block["chapter_id"] == chapter]


def nominate(document, absent):
    """``[(segment, shown blocks, absent rows named there, cast characters named there)]`` for unassigned lines."""
    from .speaker_completion import unassigned_segments
    if not absent:
        return []
    positions = {block["id"]: index for index, block in enumerate(document["source"]["blocks"])}
    speaking = {role["character_id"] for role in document["cast"]["roles"]
                if role["kind"] == "dialogue" and role.get("character_id")}
    cast = [row for row in document["cast"]["characters"] if row["id"] in speaking]
    found = []
    for segment in unassigned_segments(document):
        if segment["kind"] not in SPOKEN_KINDS or "simultaneous_speakers" in segment:
            continue
        shown = _window(document, segment, positions)
        text = "\n".join(block["text"] for block in shown)
        named = [row for row in absent if named_in(text, row)]
        if named:
            found.append((segment, shown, named, [row for row in cast if named_in(text, row)]))
    return found


def _mark(block, segment):
    text = block["text"]
    if block["id"] != segment["block_id"]:
        return text
    return text[:segment["start"]] + "⟦" + text[segment["start"]:segment["end"]] + "⟧" + text[segment["end"]:]


def _about(row):
    """An option's note, saying where the character comes from."""
    note = " ".join((row.get("note") or row.get("description") or "").split())
    if "key" not in row:
        return "in this book's cast: " + note
    if row["records"] is not None:
        return f"from the earlier book {row['records']['book']}: " + note
    return "met in this book: " + note


def check(document, nominated, run, *, workers=1):
    """``{segment_id: (absent row, block_id, quote)}`` for lines the text shows an absent known character says."""
    from .narrator_read import _item_row, _numbers, _parallel, _unwrap
    from .rerun_review_prompts import KNOWN_SPEAKER_CHECK, KNOWN_SPEAKER_VERSION
    numbers = _numbers(document["source"])
    items = []
    for index, (segment, shown, named, cast) in enumerate(nominated, 1):
        options = {f"o{number}": row for number, row in enumerate([*named, *cast], 1)}
        items.append({"id": f"k{index}",
                      "text": "\n".join(f"{numbers[block['id']]}|{_mark(block, segment)}" for block in shown),
                      "options": "\n".join("|".join([label, row["name"], ";".join(row.get("aliases", [])),
                                                     _about(row)]) for label, row in options.items()),
                      "_segment": segment["id"], "_options": options, "_absent": {id(row) for row in named},
                      "_blocks": {numbers[block["id"]]: block for block in shown}})
    batches = [items[start:start + BATCH] for start in range(0, len(items), BATCH)]

    def ask(batch):
        payload = {"mode": "known_speaker_check", "v": KNOWN_SPEAKER_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,option_id,N,quote]|[item_id,"-"]]}')
            result = {}
            for answer in answers:
                answer = _item_row(answer, "k")
                if not (isinstance(answer, list) and len(answer) >= 2 and answer[0] in by_id):
                    continue
                item = by_id[answer[0]]
                if answer[1] in {"-", "?"}:
                    result[answer[0]] = None
                    continue
                row = item["_options"].get(answer[1])
                block = item["_blocks"].get(answer[2]) if len(answer) >= 4 and type(answer[2]) is int else None
                quote = answer[3].strip() if len(answer) >= 4 and isinstance(answer[3], str) else ""
                if (row is None or block is None or not quote or quote not in block["text"]
                        or not any(name.casefold() in quote.casefold() for name in _names(row))):
                    raise ValueError(f'{answer[0]}: an option needs [item_id,option_id,N,quote] with an option id from '
                                     'the item and an exact quote from paragraph N that contains its name')
                result[answer[0]] = (row, block["id"], quote) if id(row) in item["_absent"] else None
            missing = sorted(set(by_id) - set(result), key=lambda value: int(value[1:]))
            if missing:
                raise ValueError("missing or invalid answers for items: " + ", ".join(missing[:20]))
            return result
        return run(KNOWN_SPEAKER_CHECK, payload, validate)
    answers = {}
    for result in _parallel(ask, batches, workers):
        answers.update(result)
    return {item["_segment"]: answers[item["id"]] for item in items if answers.get(item["id"])}


def new_records(document, row):
    """The character, voice (pending design) and role for ``row``, a ledger character new to the series."""
    series = document["provenance"].get("series") or {}
    try:
        namespace = uuid.UUID(series["series_id"]) if series.get("series_id") else uuid.NAMESPACE_URL
    except (ValueError, TypeError):
        namespace = uuid.NAMESPACE_URL
    base = f"{series.get('source_fingerprint') or fingerprint(document['source'])}:known:{row['id']}"
    character_id = "series_char_" + uuid.uuid5(namespace, base + ":character").hex
    voice_id = "series_voice_" + uuid.uuid5(namespace, base + ":voice").hex
    blocks = {block["id"]: block["text"] for block in document["source"]["blocks"]}
    evidence = [{"block_id": item["block_id"], "quote": item["quote"]} for item in row.get("voice", [])
                if isinstance(item, dict) and item.get("quote") and item["quote"] in blocks.get(item.get("block_id"), "")][:4]
    return {"character": {"id": character_id, "name": row["name"], "aliases": list(row.get("aliases", [])),
                          "description": row.get("note") or "Character in the book.", "evidence": evidence,
                          "voice_id": voice_id},
            "voice": {"id": voice_id, "description": "Voice pending design.", "traits": []},
            "role": {"id": role_id(character_id), "kind": "dialogue", "character_id": character_id,
                     "voice_id": voice_id, "description": f"Dialogue role for {row['name']}."}}


def design_new_voices(document, rows, requests, client, *, workers=1, progress=lambda message: None):
    """Fill the voices of ``rows`` (``[(ledger row, records)]``) with the narrator read's voice design."""
    from .narrator_read import Meter, design_voices
    if not rows:
        return
    read = document["provenance"]["narrator_read"]["ledger"]
    ledger = {"characters": [deepcopy(row) for row, _ in rows], "narration": deepcopy(read.get("narration") or {"mode": None}),
              "questions": []}
    placeholder = {"id": "narrator_placeholder", "description": "Narrator.", "traits": [], "series_origin": {}}
    cast = {"characters": [], "voices": [placeholder], "roles": [
        {"id": "narrator_main", "kind": "narration", "voice_id": placeholder["id"], "description": "Narrator."}]}
    for row, records in rows:
        character = dict(deepcopy(records["character"]), series_origin={"local_character_id": f"char_{row['id']}"})
        cast["characters"].append(character)
        cast["voices"].append(deepcopy(records["voice"]))
        cast["roles"].append(deepcopy(records["role"]))
    meter = Meter(client)
    try:
        designed, _, _ = design_voices(document["source"], requests, meter, ledger, cast, {}, workers=workers,
                                       progress=progress)
    finally:
        meter.close()
    by_id = {voice["id"]: voice for voice in designed["voices"]}
    for _, records in rows:
        voice = by_id[records["voice"]["id"]]
        records["voice"].update({key: deepcopy(value) for key, value in voice.items() if key != "id"})


def apply_answers(document, answers, *, design=lambda rows: None):
    """``document`` with the answered lines given to their known characters, the absent ones taken into the cast.

    ``design(rows)`` fills the voices of characters new to the series.
    """
    result = deepcopy(document)
    cast = result["cast"]
    roles, new = {}, []
    for segment_id, (row, block_id, quote) in answers.items():
        if row["key"] in roles:
            continue
        records = deepcopy(row["records"]) if row["records"] is not None else new_records(result, row["ledger"])
        if row["records"] is None:
            new.append((row["ledger"], records))
        roles[row["key"]] = records
    design(new)
    for records in roles.values():
        cast["characters"].append(records["character"])
        cast["voices"].append(records["voice"])
        cast["roles"].append(records["role"])
    segments = {segment["id"]: segment for segment in result["segments"]}
    assigned = []
    for segment_id, (row, block_id, quote) in answers.items():
        role = roles[row["key"]]["role"]
        segment = segments[segment_id]
        origin = f" (from {row['records']['book']})" if row["records"] is not None else ""
        segment.update(role_id=role["id"], voice_id=role["voice_id"])
        segment["attribution"] = {"basis": "context",
                                  "reason": f"{row['name']}{origin} is named beside the line as its speaker.",
                                  "evidence": [{"block_id": block_id, "quote": quote}], "candidates": [],
                                  "review_required": False}
        assigned.append({"segment_id": segment_id, "character_id": role["character_id"], "name": row["name"],
                         "block_id": block_id, "quote": quote})
    from .speaker_completion import ATTRIBUTION_ISSUE_KINDS, unassigned_segments
    remaining = {segment["block_id"] for segment in unassigned_segments(result)}
    covered = set(answers) | {segments[segment_id]["block_id"] for segment_id in answers
                              if segments[segment_id]["block_id"] not in remaining}
    for issue in result["issues"]:
        if not issue["resolved"] and issue["kind"] in ATTRIBUTION_ISSUE_KINDS and issue["target"] in covered:
            issue.update(resolved=True, resolution={"method": "known_speaker", "basis": "analysis"})
    result["provenance"].setdefault("known_speakers", []).extend(assigned)
    from .contracts import readiness
    result["readiness"] = readiness(result)
    return result


def validate_known_speakers(document, candidate, registry):
    """Refuse a ``known-speakers`` revision that does more than give unassigned lines to known characters.

    Returns the voice IDs it adds.
    """
    from .analysis_revision import appended_cast
    if candidate["source"] != document["source"]:
        raise ValueError("Known speakers may not change the source")
    provenance = {key: value for key, value in candidate["provenance"].items() if key != "known_speakers"}
    if provenance != {key: value for key, value in document["provenance"].items() if key != "known_speakers"}:
        raise ValueError("Known speakers may change only provenance.known_speakers")
    added = appended_cast(document["cast"], candidate["cast"])
    if added is None:
        raise ValueError("Known speakers may only add cast records after the saved ones")
    earlier_ids = {row["id"] for book in earlier_books(registry, document) for row in book["cast"]["characters"]}
    new_characters = candidate["cast"]["characters"][len(document["cast"]["characters"]):]
    voices = {row["id"]: row for row in candidate["cast"]["voices"]}
    for character in new_characters:
        if character["id"] in earlier_ids:
            records = earlier_records(registry, document, character["id"])
            if records is None or records["character"] != character or voices.get(character["voice_id"]) != records["voice"]:
                raise ValueError(f"{character['name']} must be taken in with its earlier book's records")
        elif not voices.get(character["voice_id"], {}).get("description", "").strip():
            raise ValueError(f"{character['name']} needs a voice description")
    before = {row["id"]: row for row in document["segments"]}
    if [row["id"] for row in candidate["segments"]] != [row["id"] for row in document["segments"]]:
        raise ValueError("Known speakers may not add, remove or reorder lines")
    from .contracts import has_assigned_speakers
    for row in candidate["segments"]:
        old = before[row["id"]]
        if row == old:
            continue
        loose = {key: value for key, value in row.items() if key not in {"role_id", "voice_id", "attribution"}}
        if (loose != {key: value for key, value in old.items() if key not in {"role_id", "voice_id", "attribution"}}
                or has_assigned_speakers(old)):
            raise ValueError(f"Known speakers may only give unassigned line {row['id']} a speaker")
    if len(candidate["issues"]) != len(document["issues"]) or any(
            {key: value for key, value in new.items() if key not in {"resolved", "resolution"}}
            != {key: value for key, value in old.items() if key not in {"resolved", "resolution"}}
            for new, old in zip(candidate["issues"], document["issues"])):
        raise ValueError("Known speakers may only resolve issues")
    return added


def review_known_speakers(directory, config, client_factory, progress, *, recheck_cache=None):
    """``analyze BOOK``: give unassigned lines to the known characters named beside them; True when saved."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, revise_directories
    from .pipeline import CachedRequests
    directory = Path(directory).resolve()
    original_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, registry, _ = _completed_directory(directory)
    nominated = nominate(document, absent_characters(document, registry))
    if not nominated:
        return False
    if not config["provider"]["model"]:
        print("Known speakers: set provider.model to check whether characters the book already knows say its "
              "unassigned lines.")
        return False
    settings = config["analysis"]
    client = client_factory(config["provider"])
    print(f"Checking whether {len(nominated)} unassigned line(s) belong to characters the book already knows "
          f"(from this book's reading or earlier books), with {client.model} at {client.base_url}; their passages "
          "will be sent to this endpoint.", file=sys.stderr)
    requests = CachedRequests(client, recheck_cache or analysis_cache_directory(directory, settings["cache_dir"]),
                              repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    workers = min(5, settings["attribution_workers"])
    answers = check(document, nominated, requests.run, workers=workers)
    if not answers:
        print("Known speakers: none of the unassigned lines is shown to be said by a character the book already "
              "knows; they stay for the speaker choice.")
        return False
    candidate = apply_answers(document, answers, design=lambda rows: design_new_voices(
        document, rows, requests, client, workers=workers, progress=progress))
    directories = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in directories}
    if hashes[str(directory)] != original_hash:
        raise ValueError(f"Analysis changed while its unassigned lines were being checked: {directory}")
    result = revise_directories({directory: candidate}, directories=directories, expected_manifest_hashes=hashes,
                                scope="known-speakers", progress=print)
    joined = {}
    for segment_id, (row, _, quote) in answers.items():
        joined.setdefault(row["name"], (row, []))[1].append(quote)
    for name, (row, quotes) in joined.items():
        origin = (f"{name} from {row['records']['book']}, with that book's voice" if row["records"] is not None
                  else f"{name}, new to the cast (choose its voice in the cast screen before rendering)")
        print(f"Known speakers: {len(quotes)} line(s) given to {origin} (“{quotes[0]}”).")
    print(f"Original analysis files and revision journal: {result['backup_directory']}")
    return True
