"""Conservative front-matter detection for fresh narrator-read analyses."""

from copy import deepcopy
import re

from .quote_spans import quote_spans

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_ISBN = re.compile(
    r"\b(?:e?ISBN)\b(?:\s*[-:]\s*(?:10|13))?\s*:?\s*"
    r"(?:(?:\d[\s-]*){13}|(?:\d[\s-]*){9}[\dXx])(?![\dXx])",
    re.IGNORECASE,
)
_COPYRIGHT = re.compile(r"©|\bCopyright\b|\bAll\s+rights\s+reserved\b", re.IGNORECASE)
_OTHER_SIGNALS = {
    "published_by": re.compile(r"\bPublished\s+by\b", re.IGNORECASE),
    "printed_in": re.compile(r"\bPrinted\s+in\b", re.IGNORECASE),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE),
    "us_address": re.compile(r"^\s*\d+\s+[^\n]*[A-Za-z][^\n]*\d{5}(?:-\d{4})?\s*$", re.MULTILINE),
}


def is_front_matter_text(text: str) -> bool:
    """Return whether publisher signals justify dropping one pre-story block."""
    if _ISBN.search(text) or _COPYRIGHT.search(text):
        return True
    return sum(bool(pattern.search(text)) for pattern in _OTHER_SIGNALS.values()) >= 2


def drop_front_matter(source):
    """Remove matching blocks only from the first chapter's conservative window.

    A heading closes the window. Without a heading, the first detected dialogue
    block closes it; without either boundary, no blocks are removed. The input
    source is never mutated, and omitted blocks are recorded as exclusions.
    """
    if not source["chapters"]:
        return source
    first_chapter = source["chapters"][0]["id"]
    chapter_blocks = [block for block in source["blocks"] if block["chapter_id"] == first_chapter]
    heading = next((index for index, block in enumerate(chapter_blocks)
                    if block["source"]["tag"] in _HEADING_TAGS), None)
    if heading is not None:
        window = chapter_blocks[:heading]
    else:
        dialogue = {span["block_id"] for span in quote_spans(source)}
        first_dialogue = next((index for index, block in enumerate(chapter_blocks)
                               if block["id"] in dialogue), None)
        if first_dialogue is None:
            return source
        window = chapter_blocks[:first_dialogue]
    dropped = {block["id"] for block in window if is_front_matter_text(block["text"])}
    if not dropped:
        return source
    result = deepcopy(source)
    result["blocks"] = [block for block in result["blocks"] if block["id"] not in dropped]
    for chapter in result["chapters"]:
        chapter["block_ids"] = [identifier for identifier in chapter["block_ids"] if identifier not in dropped]
    result["exclusions"].extend(
        {"href": block["source"]["href"], "block_id": block["id"],
         "reason": "front matter omitted by narrator-read analysis"}
        for block in source["blocks"] if block["id"] in dropped
    )
    return result
