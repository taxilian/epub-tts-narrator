"""Shared display chapter boundaries without changing source coordinates."""

import re


_CHAPTER_NUMBER = (
    r"(?:[0-9]+|[IVXLCDM]+|"
    r"ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|"
    r"THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|"
    r"(?:TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY)"
    r"(?:[ -](?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE))?)"
)
_CHAPTER_LABEL = re.compile(rf"CHAPTER\s+{_CHAPTER_NUMBER}\s*[:.]?", re.IGNORECASE)


def script_partitions(document, *, prefer_source_titles=True):
    """Find conservative display chapters without changing source coordinates.

    Source chapters are EPUB resources, which can contain several story chapters.
    Only standalone numbered headings with structural/annotation support create
    extra boundaries. Resources resembling a contents list remain one section.
    """
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    kinds = {}
    for segment in document["segments"]:
        kinds.setdefault(segment["block_id"], set()).add(segment["kind"])

    def is_heading(block):
        return (re.fullmatch(r"h[1-6]", block["source"]["tag"], re.IGNORECASE) is not None
                or kinds.get(block["id"]) == {"heading"})

    def is_label(block):
        return _CHAPTER_LABEL.fullmatch(block["text"].strip()) is not None

    def has_prose(items):
        return any(block["text"].strip() and not is_label(block) and not is_heading(block)
                   for block in items)

    result = []
    for chapter in document["source"]["chapters"]:
        chapter_blocks = [blocks[block_id] for block_id in chapter["block_ids"]]
        labels = [index for index, block in enumerate(chapter_blocks)
                  if is_label(block) and is_heading(block)]
        list_like = len(labels) >= 2 and (
            len(labels) / len(chapter_blocks) >= 0.75
            or not any(has_prose(chapter_blocks[left + 1:right])
                       for left, right in zip(labels, labels[1:])))
        starts = [0]
        headings = {}
        if labels and not list_like:
            first = labels[0]
            if first and has_prose(chapter_blocks[:first]):
                starts.append(first)
                headings[first] = chapter_blocks[first]
            else:
                headings[0] = chapter_blocks[first]
            for left, right in zip(labels, labels[1:]):
                if has_prose(chapter_blocks[left + 1:right]):
                    starts.append(right)
                    headings[right] = chapter_blocks[right]
        for part, start in enumerate(starts):
            end = starts[part + 1] if part + 1 < len(starts) else len(chapter_blocks)
            heading = headings.get(start)
            title = chapter["title"]
            if list_like:
                title = "Contents"
            elif prefer_source_titles and part == 0 and title.strip().casefold() not in {"", "unknown", "untitled"}:
                heading = None
            elif heading is not None:
                title = heading["text"]
            elif title.strip().casefold() in {"", "unknown", "untitled"}:
                first_heading = next((block for block in chapter_blocks[start:end]
                                      if is_heading(block)), None)
                if first_heading is not None:
                    heading = first_heading
                    title = heading["text"]
            result.append({
                "chapter_id": chapter["id"], "title": title,
                "heading_block_id": heading["id"] if heading is not None else None,
                "block_ids": [block["id"] for block in chapter_blocks[start:end]],
            })
    return result
