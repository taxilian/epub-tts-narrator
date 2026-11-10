"""Bind tagged name uses to the engine's sanctioned pronunciation controls.

Only spans the book's ``name-pronunciations.json`` marks as the name receive
the confirmed reading (phonemes, or a say-as spelling spoken in place of the
word); homonyms and untagged words are untouched. The controls
travel through ``merge_row_pronunciations`` exactly like registry choices, so
Drama 3 phoneme tags are emitted only by the performance compiler.
"""

import re

from ..pronunciations import WORD_PATTERN


def prepare_name_pronunciations(bundle, config_path, *, notify=print):
    from ..name_pronunciations import pending_tagging
    pending = pending_tagging(bundle["directory"], bundle["source"])
    if not any(state["pending"] for _, state in pending):
        return
    from ..config import load_config
    from ..name_pronunciation_model import tag_defined_names
    tag_defined_names(bundle["directory"], load_config(config_path), progress=notify, document=bundle)


def named_pronunciation_snapshot(bundle):
    from ..name_pronunciations import render_snapshot
    return render_snapshot(bundle["directory"], bundle["source"])


def with_tagged_names(row, legacy, matches, snapshot):
    """Add tagged spans to a row's pronunciation matches; tags win overlaps.

    ``matches`` is the existing explicit match list, or None for the ordinary
    registry behaviour, which is reproduced here before tagged spans are added.
    """
    if snapshot is None:
        return matches
    inside = []
    for item in snapshot["spans"].get(row["block_id"], []):
        if item["end"] <= row["start"] or item["start"] >= row["end"]:
            continue
        if item["start"] < row["start"] or item["end"] > row["end"]:
            raise ValueError("A tagged name crosses a script segment boundary; rerun pronounce-name after resolving the segmentation")
        inside.append(item)
    if not inside:
        return matches
    base = matches if matches is not None else [
        (match, legacy[match.group()]) for match in WORD_PATTERN.finditer(row["text"]) if match.group() in legacy]
    tagged = []
    for item in inside:
        match = re.compile(re.escape(item["word"])).match(row["text"], item["start"] - row["start"])
        if match is None or match.end() != item["end"] - row["start"]:
            raise ValueError("A tagged name no longer matches the script text; rerun pronounce-name for this book")
        pronunciation = ({"alphabet": "respelling", "value": item["say_as"]} if "say_as" in item
                         else {"alphabet": "cmu_arpabet", "value": item["phonemes"]})
        tagged.append((match, {"word": item["word"], "status": "confirmed", "pronunciation": pronunciation}))
    kept = [pair for pair in base
            if not any(pair[0].start() < other.end() and other.start() < pair[0].end() for other, _ in tagged)]
    return sorted([*kept, *tagged], key=lambda pair: pair[0].start())
