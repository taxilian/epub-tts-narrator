"""Validate and load the book-local, source-bound annotation checkpoint."""

import json
from pathlib import Path

from .name_script import FILENAME, VERSION, annotate, binding, candidates, identity_table, parse_text
from .pipeline import fingerprint, write_json
from .pronunciations import _regular_json


MAX_BYTES = 16 * 1024 * 1024


def check_size_budget(document, state):
    """Bound encoded checkpoints using longest markers plus every unresolved index."""
    maximum = dict(state, status="pending", windows=[])
    for target in windows(document, state["names"]):
        occurrences = target["candidates"]
        maximum["windows"].append({key: target[key] for key in ("block_id", "start", "end")} | {
            "text": annotate(target["text"], occurrences, [max(item["names"], key=len) for item in occurrences]),
            "unresolved": list(range(len(occurrences)))})
    size = 1
    for part in json.JSONEncoder(indent=2, ensure_ascii=False, allow_nan=False).iterencode(maximum):
        size += len(part.encode("utf-8"))
        if size > MAX_BYTES:
            raise ValueError(f"Name script worst-case encoded size exceeds its {MAX_BYTES} byte capacity; "
                             "no model work was started")


def windows(document, names):
    for block in document["source"]["blocks"]:
        text, start = block["text"], 0
        while start < len(text):
            end = min(len(text), start + 2000)
            while end < len(text) and (text[end].isalnum() or text[end] in "'’\u2010\u2011-"):
                end += 1
            part = text[start:end]
            yield {"block_id": block["id"], "start": start, "end": end, "text": part,
                   "candidates": candidates(part, names)}
            start = end


def read_state(directory):
    path = Path(directory) / FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    value = _regular_json(path)
    fields = {"format", "version", "binding", "ids", "names", "windows", "status"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["format"] != "epub-tts-name-script" or type(value["version"]) is not int or value["version"] != VERSION
            or not isinstance(value["status"], str) or value["status"] not in {"pending", "ready"}
            or not isinstance(value["ids"], list)
            or any(not isinstance(item, str) or not item for item in value["ids"])
            or len(set(value["ids"])) != len(value["ids"])
            or not isinstance(value["names"], dict) or not isinstance(value["windows"], list)):
        raise ValueError("Invalid name-script checkpoint")
    return value


def save_state(directory, state):
    path = Path(directory) / FILENAME
    if path.is_symlink():
        raise ValueError("Name script must not be a symlink")
    write_json(path, state, overwrite=True)


def new_state(document, previous=None):
    identities = identity_table(document)
    ids = list(previous["ids"]) if previous else []
    ids.extend(key for key in identities if key not in ids)
    names = {f"N{ids.index(key) + 1}": value for key, value in identities.items()}
    return {"format": "epub-tts-name-script", "version": VERSION, "binding": binding(document),
            "ids": ids, "names": names, "windows": [], "status": "pending"}


def checked_windows(state, document):
    expected = new_state(document, state)
    if state["binding"] != expected["binding"] or state["names"] != expected["names"]:
        raise ValueError("Name script is stale; run prepare-name-script BOOK")
    targets = list(windows(document, state["names"]))
    if len(state["windows"]) > len(targets):
        raise ValueError("Name script contains excess windows")
    result = []
    for saved, target in zip(state["windows"], targets):
        if (not isinstance(saved, dict) or set(saved) != {"block_id", "start", "end", "text", "unresolved"}
                or type(saved["start"]) is not int or type(saved["end"]) is not int
                or any(saved[key] != target[key] for key in ("block_id", "start", "end"))
                or not isinstance(saved["unresolved"], list)
                or any(type(index) is not int or not 0 <= index < len(target["candidates"])
                       for index in saved["unresolved"])
                or len(set(saved["unresolved"])) != len(saved["unresolved"])):
            raise ValueError("Invalid name script window or candidate coverage")
        mentions = parse_text(saved["text"], target["text"], state["names"])
        spans = {(item["start"], item["end"]): item for item in target["candidates"]}
        if any((item["start"], item["end"]) not in spans for item in mentions):
            raise ValueError("Name script marked a noncandidate word")
        resolved = {(item["start"], item["end"]) for item in mentions}
        if any((target["candidates"][i]["start"], target["candidates"][i]["end"]) in resolved
               for i in saved["unresolved"]):
            raise ValueError("Name script candidate is both resolved and unresolved")
        result.append((saved, target, mentions))
    if state["status"] == "ready" and len(result) != len(targets):
        raise ValueError("Ready name script lacks complete source coverage")
    return result


def mention_id(state, block_id, start, end):
    return fingerprint([state["binding"], block_id, start, end])


def ordinary_span(snapshot, block_id, start, end):
    return not any(item["start"] < end and start < item["end"]
                   for item in snapshot["mentions"].get(block_id, [])) and not any(
        item["block_id"] == block_id and item["start"] < end and start < item["end"]
        for item in snapshot["unresolved"])


def load_name_script(directory, document):
    state = read_state(directory)
    if state is None:
        raise ValueError("Scoped names require prepare-name-script BOOK")
    checked = checked_windows(state, document)
    if state["status"] != "ready":
        raise ValueError("Name script preparation is pending; rerun prepare-name-script BOOK")
    mentions, unresolved = {}, []
    for saved, target, spans in checked:
        block_id, offset = saved["block_id"], saved["start"]
        mentions.setdefault(block_id, []).extend(dict(item, start=item["start"] + offset,
                                                     end=item["end"] + offset,
                                                     word=target["text"][item["start"]:item["end"]]) for item in spans)
        for index in saved["unresolved"]:
            item = target["candidates"][index]
            start, end = item["start"] + offset, item["end"] + offset
            unresolved.append(dict(item, block_id=block_id, start=start, end=end,
                                   mention_id=mention_id(state, block_id, start, end)))
    return {"binding": state["binding"], "names": state["names"], "mentions": mentions, "unresolved": unresolved}
