"""Rerunning ``analyze BOOK`` asks who says a line that several characters say together.

A line whose own tag says several people speak ("the voices all shouted as
one") but which went to one character is flagged by ``group_cues``
(``plural_tag``). When the scene has other speakers, the line is usually the
scene's characters speaking together: three voices behind three doors each
speak, then all three shout the same word. For each such line this asks the
book's model, with the scene's last paragraphs and the characters who spoke in
them (``rerun_review_prompts.JOINT_CHECK``): which of them say these words
together? An answer needs at least two of them and an exact quote that shows
them speaking together; the line then becomes a ``simultaneous_speakers`` line
(``basis: "explicit"``, that quote as evidence; ``simultaneous_mode`` from the
tag), published through revision scope ``attribution``. "n" (the line's
speaker is right, or someone else said it) and "?" change nothing.

A line a person assigned (``basis: "user"``) is never changed by the model:
when the model finds it said together, the person is asked (J join / K keep /
L later; in the run shell ``rerun_review_tui.JointLineScreen``), and a kept
line is remembered in the book's ``joint-lines.json`` (by its text) and never
asked again. Without a terminal it is only listed. Nothing is asked when no
line is flagged. Answers are cached in the book's request cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

BATCH = 20
KEPT_FILENAME = "joint-lines.json"
SCENE = 12  # paragraphs before the line whose speakers may say it with its speaker
MAX_SPEAKERS = 8
_STAGGER = re.compile(r"\b(?:one after another|in turn|overlapping|echo(?:ed|ing)|babble|babbling)\b", re.IGNORECASE)


def _text_key(row):
    return hashlib.sha256(f"{row['id']}\n{row['text']}\n{row.get('role_id')}".encode()).hexdigest()


def kept(directory):
    """``{segment id: text key}`` of lines a person kept with their one speaker."""
    path = Path(directory) / KEPT_FILENAME
    if path.is_symlink() or not path.is_file():
        return {}
    record = json.loads(path.read_text(encoding="utf-8"))
    return {row["segment_id"]: row["key"] for row in record.get("kept", [])}


def keep(directory, row, speakers):
    """Remember that a person keeps ``row`` with its one speaker (never asked again while unchanged)."""
    path = Path(directory) / KEPT_FILENAME
    record = {"version": 1, "kept": []}
    if path.is_file() and not path.is_symlink():
        record = json.loads(path.read_text(encoding="utf-8"))
    record["kept"] = [item for item in record["kept"] if item["segment_id"] != row["id"]]
    record["kept"].append({"segment_id": row["id"], "key": _text_key(row), "not_with": speakers})
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidates(document, kept_lines=None):
    """``[(segment, clause, [role ids], [blocks])]``: flagged lines with the speakers of their scene."""
    from .group_cues import group_warnings
    flags = {warning["segment_id"]: warning for warning in group_warnings(document)
             if warning["kind"] == "plural_tag"}
    if not flags:
        return []
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    blocks = document["source"]["blocks"]
    position = {block["id"]: index for index, block in enumerate(blocks)}
    by_block = {}
    for row in document["segments"]:
        by_block.setdefault(row["block_id"], []).append(row)
    result = []
    for row in document["segments"]:
        warning = flags.get(row["id"])
        if warning is None or (kept_lines or {}).get(row["id"]) == _text_key(row):
            continue
        if voices.get(row.get("voice_id"), {}).get("ensemble"):
            continue
        index = position[row["block_id"]]
        chapter = blocks[index]["chapter_id"]
        scene = [block for block in blocks[max(0, index - SCENE):index + 1] if block["chapter_id"] == chapter]
        speakers = []
        for block in reversed(scene):
            for other in by_block.get(block["id"], []):
                role = roles.get(other.get("role_id") or "")
                if (role is not None and role.get("character_id") and role.get("kind") == "dialogue"
                        and role["id"] not in speakers and len(speakers) < MAX_SPEAKERS):
                    speakers.append(role["id"])
        if row["role_id"] not in speakers:
            speakers.insert(0, row["role_id"])
        if len({roles[role_id]["voice_id"] for role_id in speakers}) >= 2:
            result.append((row, warning["clause"], speakers, scene))
    return result


def check_lines(document, requests, found, *, workers=1):
    """``{segment_id: ([role ids], paragraph number, quote, block)}`` for lines the model says are joint."""
    from .narrator_read import _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import JOINT_CHECK, JOINT_REVIEW_VERSION
    numbers = _numbers(document["source"])
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    names = {character["id"]: character["name"] for character in document["cast"]["characters"]}
    items = []
    for index, (row, clause, speakers, scene) in enumerate(found, 1):
        choices = {number: role_id for number, role_id in enumerate(speakers, 1)}
        items.append({"id": f"j{index}", "line": row["text"], "tag": clause,
                      "speakers": [f"{number}|{names[roles[role_id]['character_id']]}"
                                   for number, role_id in choices.items()],
                      "text": _render_blocks(scene, numbers), "_row": row, "_choices": choices,
                      "_blocks": {numbers[block["id"]]: block for block in scene}})
    batches = [items[start:start + BATCH] for start in range(0, len(items), BATCH)]

    def run(batch):
        payload = {"mode": "joint_line_review", "v": JOINT_REVIEW_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,[speaker numbers],N,quote]|[item_id,"n"]|[item_id,"?"]]}')
            result = {}
            for answer in answers:
                answer = _item_row(answer, "j")
                if not (isinstance(answer, list) and answer and answer[0] in by_id):
                    raise ValueError("each answer must start with an item id from the request")
                item = by_id[answer[0]]
                if answer[1:2] in (["n"], ["?"]):
                    result[answer[0]] = None
                    continue
                if len(answer) < 4 or not isinstance(answer[1], list) or type(answer[2]) is not int:
                    raise ValueError(f'{answer[0]}: give [item_id,[speaker numbers],N,quote], [item_id,"n"] or [item_id,"?"]')
                chosen = list(dict.fromkeys(answer[1]))
                if (len(chosen) < 2 or any(type(number) is not int or number not in item["_choices"] for number in chosen)
                        or len({roles[item["_choices"][number]]["voice_id"] for number in chosen}) < len(chosen)):
                    raise ValueError(f"{answer[0]}: name two or more different speakers by their numbers in this item")
                block = item["_blocks"].get(answer[2])
                if block is None:
                    raise ValueError(f"{answer[0]}: paragraph {answer[2]} is not in this item; cite one of its paragraphs")
                if not isinstance(answer[3], str) or not answer[3].strip() or answer[3].strip() not in block["text"]:
                    raise ValueError(f"{answer[0]}: the quote must be exact text from paragraph {answer[2]}")
                result[answer[0]] = ([item["_choices"][number] for number in chosen], answer[2], answer[3].strip(), block)
            missing = sorted(set(by_id) - set(result))
            if missing:
                raise ValueError("missing answers for items: " + ", ".join(missing[:20]))
            return result
        return requests.run(JOINT_CHECK, payload, validate)
    answers = {}
    for result in _parallel(run, batches, workers):
        answers.update(result)
    return {item["_row"]["id"]: answers[item["id"]] for item in items if answers.get(item["id"])}


def joined(document, confirmed, clauses, *, chosen=()):
    """``document`` with each confirmed line said together by its speakers (``chosen``: a person said so)."""
    revised = json.loads(json.dumps(document))
    roles = {role["id"]: role for role in revised["cast"]["roles"]}
    for row in revised["segments"]:
        if row["id"] not in confirmed:
            continue
        speakers, _, quote, block = confirmed[row["id"]]
        row["role_id"] = row["voice_id"] = None
        row["simultaneous_speakers"] = [{"role_id": role_id, "voice_id": roles[role_id]["voice_id"]}
                                        for role_id in speakers]
        if _STAGGER.search(clauses[row["id"]]):
            row["simultaneous_mode"] = "staggered"
        for key in ("ensemble", "voice_effect"):
            row.pop(key, None)
        by_person = row["id"] in chosen
        row["attribution"] = {"basis": "user" if by_person else "explicit",
                              "reason": ("A person chose these characters to say it together." if by_person else
                                         "The text says these characters speak together."),
                              "evidence": [{"block_id": block["id"], "quote": quote}], "candidates": [],
                              "review_required": False}
    return revised


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
        if answer in {"j", "k", "l"}:
            return answer
        print("Please enter J, K or L.")


def _decide(view):
    """"j", "k" or "l": on the run shell's own screen when one is open, else on the line."""
    from .rerun_questions import NO_SCREEN, ask_on_screen
    answer = ask_on_screen("JointLineScreen", view)
    if answer is NO_SCREEN:
        print(f"\n{view['title']}\n  {view['line']}\n  {view['paragraph']}\n  Now: {view['now']}\n"
              f"  Together: {view['together']}\n  {view['why']}")
        return _ask("[J] say it together / [K] keep it as it is / [L] decide later: ")
    return answer if answer in {"j", "k", "l"} else "l"


def review_joint_lines(directory, config, client_factory, progress, *, interactive=None, reconsider_choices=False):
    """``analyze BOOK``: give lines the scene's characters say together to all of them; True when saved.

    A line a person assigned is only changed when that person says so here, or
    with ``reconsider_choices`` (``--reconsider-choices``), like any other line.
    """
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, revise_directories
    from .pipeline import CachedRequests
    from .rerun_questions import joint_view
    from .tui_support import active_screen_host
    directory = Path(directory).resolve()
    interactive = (sys.stdin.isatty() and sys.stdout.isatty()) if interactive is None else interactive
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    found = candidates(document, kept(directory))
    if not found:
        return False
    if not config["provider"]["model"]:
        print(f"Joint-line check: {len(found)} line(s) say several people speak but went to one character; set "
              "provider.model to ask who says them together.")
        return False
    settings = config["analysis"]
    client = client_factory(config["provider"])
    print(f"Checking who says {len(found)} line(s) together with {client.model} at {client.base_url}; "
          "the scenes around them will be sent to this endpoint.", file=sys.stderr)
    requests = CachedRequests(client, analysis_cache_directory(directory, settings["cache_dir"]),
                              repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    confirmed = check_lines(document, requests, found, workers=min(5, settings["attribution_workers"]))
    if not confirmed:
        print("Joint-line check: no flagged line is said by several of the scene's characters.")
        return False
    rows = {row["id"]: row for row, _, _, _ in found}
    chosen = set()
    for segment_id in [segment_id for segment_id in confirmed
                       if rows[segment_id]["attribution"].get("basis") == "user" and not reconsider_choices]:
        # A person chose this line's speaker: only that person changes it.
        speakers = confirmed[segment_id][0]
        if not (interactive or active_screen_host() is not None):
            print(f"Joint-line check: {segment_id}, which you assigned, reads as said together; rerun analyze in "
                  "a terminal to decide.")
            del confirmed[segment_id]
            continue
        answer = _decide(joint_view(document, rows[segment_id], speakers, confirmed[segment_id][2]))
        if answer == "j":
            chosen.add(segment_id)
        else:
            if answer == "k":
                keep(directory, rows[segment_id], speakers)
            del confirmed[segment_id]
    if not confirmed:
        return False
    revised = joined(document, confirmed, {row["id"]: clause for row, clause, _, _ in found}, chosen=chosen)
    directories = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in directories}
    if hashes[str(directory)] != original_manifest_hash:
        raise ValueError(f"Analysis changed while joint lines were being checked: {directory}")
    try:
        revise_directories({directory: revised}, directories=directories, expected_manifest_hashes=hashes,
                           scope="attribution", progress=progress)
    except (OSError, ValueError) as error:
        print(f"Joint-line check: the changes were not saved: {error} Rerun analyze to try again; its answers are cached.")
        return False
    names = {role["id"]: role for role in document["cast"]["roles"]}
    characters = {character["id"]: character["name"] for character in document["cast"]["characters"]}
    for segment_id, (speakers, _, _, _) in confirmed.items():
        who = " & ".join(characters.get(names[role_id].get("character_id"), role_id) for role_id in speakers)
        print(f"Joint-line check: {who} now say line {segment_id} together.")
    return True
