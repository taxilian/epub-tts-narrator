"""Rerunning ``analyze BOOK`` asks the model about groups the speaker tags leave unclear.

``group_proposals`` proposes a group only when its own tags clearly say
several people speak. A character whose name does not name several people
("Arendish noble") and whose lines carry a single plural tag ("the nobles
asked") stays undecided: the tag may show a group, or the line may belong to
someone else (``group_cues`` flags it). For each such character this asks the
book's model, with the paragraphs of its tagged lines
(``rerun_review_prompts.GROUP_CHECK``): is this entry several people speaking
as one? A "y" needs an exact quote that shows it; the character is then
proposed as a group (``basis: "analysis"``, that quote as evidence) through
revision scope ``group-proposals``, and a person decides in the cast screen
(``C``). "n" and "?" change nothing. A person's name is never asked about,
and nothing is asked when no character is unclear. Answers are cached in the
book's request cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

BATCH = 25


def unclear(document):
    """``[(character, [block ids])]``: characters with a plural tag that did not qualify as groups."""
    from .group_proposals import _effective, _personal, _plural_name, _tags
    tags = _tags(document)
    roles = {role["id"]: role.get("character_id") for role in document["cast"]["roles"]}
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    lines = {}
    for row in document["segments"]:
        character = roles.get(row.get("role_id") or "")
        if character and row["id"] in tags:
            lines.setdefault(character, []).append(row)
    result = []
    for character in document["cast"]["characters"]:
        if ("group" in character or "ensemble" in voices.get(character["voice_id"], {})
                or _personal(character["name"]) or _plural_name(character["name"])):
            continue
        own = set()
        plural = [row["block_id"] for row in lines.get(character["id"], [])
                  if any(_effective(kind, head, own) == "plural" for kind, head, _ in tags[row["id"]])]
        if plural:
            result.append((character, list(dict.fromkeys(plural))[:4]))
    return result


def check_groups(document, requests, candidates, *, workers=1):
    """``{character_id: (paragraph number, quote, block)}`` for entries the model says are groups."""
    from .narrator_read import _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import GROUP_CHECK, GROUP_REVIEW_VERSION
    numbers = _numbers(document["source"])
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    items = []
    for index, (character, block_ids) in enumerate(candidates, 1):
        found = [blocks[block_id] for block_id in block_ids]
        items.append({"id": f"k{index}", "who": "|".join([character["name"], ";".join(character.get("aliases", []))]),
                      "text": _render_blocks(found, numbers), "_row": character,
                      "_blocks": {numbers[block["id"]]: block for block in found}})
    batches = [items[start:start + BATCH] for start in range(0, len(items), BATCH)]

    def run(batch):
        payload = {"mode": "group_review", "v": GROUP_REVIEW_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}
        # A quote may come from any paragraph the request showed.
        shown = {number: block for item in batch for number, block in item["_blocks"].items()}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,"y",N,quote]|[item_id,"n"]|[item_id,"?"]]}')
            result = {}
            for answer in answers:
                answer = _item_row(answer, "k")
                if not (isinstance(answer, list) and answer and answer[0] in by_id):
                    raise ValueError("each answer must start with an item id from the request")
                if answer[1:2] in (["n"], ["?"]):
                    result[answer[0]] = None
                    continue
                if len(answer) < 4 or answer[1] != "y" or type(answer[2]) is not int:
                    raise ValueError(f'{answer[0]}: give [item_id,"y",N,quote], [item_id,"n"] or [item_id,"?"]')
                block = shown.get(answer[2])
                if block is None:
                    raise ValueError(f"{answer[0]}: paragraph {answer[2]} is not in this request; cite a paragraph shown here")
                if not isinstance(answer[3], str) or not answer[3].strip() or answer[3].strip() not in block["text"]:
                    raise ValueError(f"{answer[0]}: the quote must be exact text from paragraph {answer[2]}")
                result[answer[0]] = (answer[2], answer[3].strip(), block)
            missing = sorted(set(by_id) - set(result))
            if missing:
                raise ValueError("missing answers for items: " + ", ".join(missing[:20]))
            return result
        return requests.run(GROUP_CHECK, payload, validate)
    answers = {}
    for result in _parallel(run, batches, workers):
        answers.update(result)
    return {item["_row"]["id"]: answers[item["id"]] for item in items if answers.get(item["id"])}


def review_group_tags(directory, config, client_factory, progress):
    """``analyze BOOK``: propose groups the model confirms from the book's text; True when saved."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .analysis_revision import related_directories, revise_directories
    from .ensembles import proposed_voices
    from .group_proposals import _scale, _size
    from .pipeline import CachedRequests
    directory = Path(directory).resolve()
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    candidates = unclear(document)
    if not candidates:
        return False
    if not config["provider"]["model"]:
        print(f"Group check: {len(candidates)} character(s) have a tag saying several people speak; set "
              "provider.model to ask whether they are groups.")
        return False
    settings = config["analysis"]
    client = client_factory(config["provider"])
    print(f"Checking whether {len(candidates)} character(s) are groups with {client.model} at {client.base_url}; "
          "passages with their lines will be sent to this endpoint.", file=sys.stderr)
    requests = CachedRequests(client, analysis_cache_directory(directory, settings["cache_dir"]),
                              repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    confirmed = check_groups(document, requests, candidates, workers=min(5, settings["attribution_workers"]))
    if not confirmed:
        print("Group check: no unclear character is a group.")
        return False
    revised = json.loads(json.dumps(document))
    names = []
    for character in revised["cast"]["characters"]:
        if character["id"] not in confirmed:
            continue
        _, quote, block = confirmed[character["id"]]
        size = _size([quote])
        scale = _scale([quote], size)
        character["group"] = {"size": size, "scale": scale, "members": [],
                              "proposed_voices": proposed_voices(size, scale), "proposed_mode": "unison",
                              "basis": "analysis", "evidence": [{"block_id": block["id"], "quote": quote}]}
        names.append(character["name"])
    directories = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in directories}
    if hashes[str(directory)] != original_manifest_hash:
        raise ValueError(f"Analysis changed while groups were being checked: {directory}")
    try:
        revise_directories({directory: revised}, directories=directories, expected_manifest_hashes=hashes,
                           scope="group-proposals", progress=progress)
    except (OSError, ValueError) as error:
        print(f"Group check: the proposals were not saved: {error} Rerun analyze to try again; its answers are cached.")
        return False
    print(f"Group check: {', '.join(names)} proposed as group(s); set them up in the cast screen (C).")
    return True
