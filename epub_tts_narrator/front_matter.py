"""Conservative publisher-apparatus detection for fresh narrator-read analyses."""

from copy import deepcopy
import re

from .quote_spans import quote_spans


_ISBN = re.compile(
    r"\b(?:e?ISBN)\b(?:\s*[-:]\s*(?:10|13))?\s*:?\s*"
    r"(?:(?:\d[\s-]*){13}|(?:\d[\s-]*){9}[\dXx])(?![\dXx])",
    re.IGNORECASE,
)
_STRONG = re.compile(r"©|\bAll\s+rights\s+reserved\b", re.IGNORECASE)
_WEAK_SIGNALS = {
    "published_by": re.compile(r"\bPublished\s+by\b", re.IGNORECASE),
    "printed_in": re.compile(r"\bPrinted\s+in\b", re.IGNORECASE),
    "copyright": re.compile(r"\bCopyright\b", re.IGNORECASE),
    "email": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE),
    "web_url": re.compile(r"\b(?:https?://|www\.)[\w.-]+", re.IGNORECASE),
    "us_address": re.compile(r"^\s*\d+\s+[^\n]*[A-Za-z][^\n]*\d{5}(?:-\d{4})?\s*$", re.MULTILINE),
    "imprint": re.compile(r"(?:^|\n)\s*An?\s+[A-Z][\w&.'®™-]*(?:\s+[A-Z][\w&.'®™-]*){0,3}\s+Books?\b"),
    "trademark": re.compile(r"\btrademarks?\b", re.IGNORECASE),
    "catalog_title": re.compile(r"^[^a-z]{3,60}(?:\n[^a-z]{3,60})*$", re.MULTILINE),
}
_MAX_CLUSTER_BLOCKS = 120
_MAX_CLUSTER_CHARACTERS = 24000
_MAX_BLOCK_CHARACTERS = 700


def is_front_matter_text(text: str) -> bool:
    """Return whether publisher signals make one block apparatus-like."""
    if _ISBN.search(text) or _STRONG.search(text):
        return True
    return any(pattern.search(text) for pattern in _WEAK_SIGNALS.values())


def _block_signals(text: str) -> set[str]:
    return {name for name, pattern in _WEAK_SIGNALS.items() if pattern.search(text)}


def _qualifies(blocks) -> bool:
    if any(_ISBN.search(block["text"]) or _STRONG.search(block["text"]) for block in blocks):
        return True
    carriers = [signals for signals in (_block_signals(block["text"]) for block in blocks) if signals]
    types = set().union(*carriers) if carriers else set()
    return len(types) >= 2 and len(carriers) >= 2


def _cluster_drops(blocks):
    """Signal-carrying blocks of a qualifying cluster; silent neighbors stay."""
    if not _qualifies(blocks):
        return []
    return [block for block in blocks if is_front_matter_text(block["text"])]


def _clusters(blocks, eligible_ids):
    cluster = []
    for block in blocks:
        if block["id"] in eligible_ids:
            cluster.append(block)
            continue
        yield cluster
        cluster = []
    yield cluster


def drop_front_matter(source):
    """Remove publisher-apparatus blocks wherever they appear in the book.

    Blocks group into maximal runs with no detected dialogue, each block short
    enough not to be narrative prose. A run qualifies when it carries a strong
    signal (ISBN with number, copyright mark, all-rights notice) or two
    distinct weaker publisher signals in two different blocks; only the
    signal-carrying blocks of a qualifying run are dropped, so story prose
    adjacent to back matter survives. Dialogue, long prose, and oversized runs
    always break or disqualify a run. The input source is never mutated, and
    omitted blocks are recorded as exclusions.
    """
    blocks_by_id = {block["id"]: block for block in source["blocks"]}
    spans = quote_spans(source)
    dialogue = {span["block_id"] for span in spans}
    # Real dialogue lines open with the quotation; a quoted word mid-sentence
    # ("type \"subscribe\"") is narration and must not shield publisher text.
    openers = {span["block_id"] for span in spans
               if span["start"] <= len(blocks_by_id[span["block_id"]]["text"])
               - len(blocks_by_id[span["block_id"]]["text"].lstrip())}
    eligible_ids = {block["id"] for block in source["blocks"]
                    if block["id"] not in openers and len(block["text"]) <= _MAX_BLOCK_CHARACTERS}
    dropped = set()
    for cluster in _clusters(source["blocks"], eligible_ids):
        if (len(cluster) > _MAX_CLUSTER_BLOCKS
                or sum(len(block["text"]) for block in cluster) > _MAX_CLUSTER_CHARACTERS):
            continue
        dropped.update(block["id"] for block in _cluster_drops(cluster)
                       if block["id"] not in dialogue)
    if not dropped:
        return source
    result = deepcopy(source)
    result["blocks"] = [block for block in result["blocks"] if block["id"] not in dropped]
    for chapter in result["chapters"]:
        chapter["block_ids"] = [identifier for identifier in chapter["block_ids"] if identifier not in dropped]
    result["exclusions"].extend(
        {"href": block["source"]["href"], "block_id": block["id"],
         "reason": "publisher apparatus omitted by narrator-read analysis"}
        for block in source["blocks"] if block["id"] in dropped)
    return result
