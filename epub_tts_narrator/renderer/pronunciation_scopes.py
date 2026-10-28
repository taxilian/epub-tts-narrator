from ..name_script_state import load_name_script, ordinary_span
from ..pronunciation_groups import find_pronunciation, group_pronunciations
from ..pronunciations import word_id


def scoped_review_rows(bundle, scope, rows, *, snapshot=None):
    if scope["class"] != "name":
        from ..pronunciations import load_pronunciations
        additions = [dict(row, scope=scope) for row in load_pronunciations(bundle["directory"])
                     if row["status"] == "pending" and row["kind"] not in {"name", "place"}
                     and find_pronunciation(rows, row["word"]) is None]
        return [*rows, *additions]
    snapshot = snapshot if snapshot is not None else load_name_script(bundle["directory"], bundle)
    name_id = next(key for key, value in snapshot["names"].items() if value["character_id"] == scope["character_id"])
    result = list(rows)
    current = set()
    for block in bundle["source"]["blocks"]:
        for mention in snapshot["mentions"].get(block["id"], []):
            if mention["name_id"] != name_id:
                continue
            word = block["text"][mention["start"]:mention["end"]]
            current.add(word)
            if find_pronunciation(result, word) is None:
                result.append({"id": word_id(word), "scope": scope, "word": word, "kind": "name",
                               "status": "pending", "reason": f"Name {name_id}: pronunciation is entity-scoped.",
                               "evidence": [{"block_id": block["id"], "quote": word}]})
    return group_pronunciations(result, current, contextual=True)


def scoped_matches(bundle, scope, row, matches, *, snapshot=None):
    from pathlib import Path
    from ..name_script import FILENAME
    if not (Path(bundle["directory"]) / FILENAME).exists() and scope["class"] == "special":
        return matches
    snapshot = snapshot if snapshot is not None else load_name_script(bundle["directory"], bundle)
    names = snapshot["names"]
    spans = snapshot["mentions"].get(row["block_id"], [])
    result = []
    for match in matches:
        start, end = row["start"] + match.start(), row["start"] + match.end()
        marked = next((item for item in spans if item["start"] == start and item["end"] == end), None)
        if scope["class"] == "name":
            if marked and names[marked["name_id"]]["character_id"] == scope["character_id"]:
                result.append(match)
        elif ordinary_span(snapshot, row["block_id"], start, end):
            result.append(match)
    return result
