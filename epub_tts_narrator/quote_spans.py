"""Deterministic quotation spans and explicit speech tags for the narrator read.

No model calls. Quote spans let the model label short IDs instead of copying
source anchors; tags give local evidence and a disagreement check. A book's
dialogue style is detected once (``dialogue_style``): double quotation marks
(straight or curly), single quotation marks (British style, telling quotes from
apostrophes by position and punctuation), or a leading dash (one paragraph per
turn, speech split from its tag). Unquoted speech is left to the model's
explicit-anchor operation.
"""

import re

OPEN_CURLY, CLOSE_CURLY = "“", "”"

SPEECH_VERBS = (
    "said|says|asked|asks|replied|replies|answered|answers|shouted|shouts|yelled|yells|whispered|whispers|"
    "muttered|mutters|murmured|murmurs|called|calls|cried|cries|added|adds|continued|continues|"
    "explained|explains|admitted|admits|agreed|agrees|snapped|snaps|growled|growls|hissed|hisses|"
    "demanded|demands|insisted|insists|offered|offers|suggested|suggests|repeated|repeats|"
    "sighed|sighs|laughed|laughs|chuckled|chuckles|groaned|groans|grumbled|grumbles|stammered|stammers|"
    "corrected|corrects|interrupted|interrupts|declared|declares|announced|announces|countered|counters|"
    "breathed|breathes|told|tells|exclaimed|exclaims|responded|responds|began|begins|finished|finishes|"
    "noted|notes|observed|observes|warned|warns|pointed out|teased|teases|joked|jokes|protested|protests"
)
_NAME = r"((?:[A-Z][\w'’-]+)(?:\s+[A-Z][\w'’-]+){0,2})"
_PRONOUN = r"(he|she|they|I|we|it)"
_VERB = rf"(?:{SPEECH_VERBS})"
# Tag immediately following a quotation: `, Mara said` / ` said Mara` / ` she asked`.
_AFTER = [
    re.compile(rf"^[\s,]*{_NAME}\s+(?:\w+ly\s+)?{_VERB}\b"),
    re.compile(rf"^[\s,]*{_VERB}\s+{_NAME}\b"),
    re.compile(rf"^[\s,]*{_PRONOUN}\s+(?:\w+ly\s+)?{_VERB}\b", re.IGNORECASE),
]
# Tag immediately before a quotation: `Mara said, "` / `Mara said: "`.
_BEFORE = re.compile(rf"{_NAME}\s+(?:\w+ly\s+)?{_VERB}[,:]?\s*$")
_NOT_NAMES = {"The", "A", "An", "Then", "But", "And", "So", "When", "After", "Before", "As", "If", "Now",
              "Still", "Yet", "Once", "Even", "Finally", "Instead", "Meanwhile", "Suddenly",
              "He", "She", "They", "I", "We", "It", "You", "His", "Her", "Their"}


def _straight_quote_spans(text):
    spans, start = [], None
    for index, char in enumerate(text):
        if char == '"':
            if start is None:
                start = index
            else:
                spans.append((start, index + 1, True))
                start = None
    if start is not None:
        spans.append((start, len(text), False))
    return spans


def _curly_quote_spans(text):
    spans, start = [], None
    for index, char in enumerate(text):
        if char == OPEN_CURLY:
            if start is not None:
                # A new opening mark without a close: the earlier speech ran to here.
                spans.append((start, index, False))
            start = index
        elif char == CLOSE_CURLY and start is not None:
            spans.append((start, index + 1, True))
            start = None
    if start is not None:
        spans.append((start, len(text), False))
    return spans


def block_quote_spans(text):
    """Return [(start, end, closed)] double-quotation spans in one paragraph."""
    if OPEN_CURLY in text or CLOSE_CURLY in text:
        return _curly_quote_spans(text)
    return _straight_quote_spans(text)


# Single quotes double as apostrophes ("don't", "James' book", "'em"). An opening
# quote follows a space, a line start or an opening mark and precedes a word; a
# closing quote follows the punctuation that ends a spoken phrase.
_OPEN_BEFORE = " \t\n([{\u2014\u2013-\u201c\""
_CLOSE_AFTER = ",.!?…;:\u2014\u2013-"
_ELISIONS = ("em", "tis", "twas", "cause", "til", "bout", "n ", "n'", "round", "nuff")


def _single_quote_spans(text):
    spans, start = [], None
    for index, char in enumerate(text):
        if char not in "‘'’":
            continue
        before = text[index - 1] if index else " "
        after = text[index + 1] if index + 1 < len(text) else ""
        if start is None:
            if (char in "‘'" and before in _OPEN_BEFORE and after and not after.isspace()
                    and not after.isdigit() and not text[index + 1:index + 6].casefold().startswith(_ELISIONS)):
                start = index
        elif char in "’'" and before in _CLOSE_AFTER and not (after.isalnum()):
            spans.append((start, index + 1, True))
            start = None
        elif char == "‘" and before in _OPEN_BEFORE:
            # A new opening mark without a close: the earlier speech ran to here.
            spans.append((start, index, False))
            start = index
    if start is not None:
        spans.append((start, len(text), False))
    return spans


_DASH_OPEN = re.compile(r"\s*(?:—|–|―|--)\s*(?=\S)")
_DASH_RESUME = re.compile(r"\s+(?:—|–|―|--)\s*(?=\S)")


def _dash_spans(text):
    """A paragraph opening with a dash is one turn; a tag (", she said,") ends the
    speech, and a later dash resumes it."""
    opening = _DASH_OPEN.match(text)
    if not opening:
        return []
    spans, start, cursor = [], opening.start(), opening.end()
    while True:
        cut = None
        for mark in re.finditer(r"[,.!?…]", text[cursor:]):
            at = cursor + mark.end()
            if any(pattern.match(text[at:at + 80]) for pattern in _AFTER):
                cut = at
                break
        if cut is None:
            spans.append((start, len(text), True))
            return spans
        spans.append((start, cut, True))
        resume = _DASH_RESUME.search(text, cut)
        if resume is None:
            return spans
        start, cursor = resume.start() + len(resume.group(0)) - len(resume.group(0).lstrip()), resume.end()


_FINDERS = {"double": block_quote_spans, "single": _single_quote_spans, "dash": _dash_spans}


def dialogue_style(source):
    """The book's dialogue style: the finder that marks speech in the most paragraphs.

    Double quotation marks win ties; a style needs at least five paragraphs.
    """
    counts = {style: sum(1 for block in source["blocks"] if finder(block["text"]))
              for style, finder in _FINDERS.items()}
    best = max(("double", "single", "dash"), key=lambda style: counts[style])
    return best if best == "double" or counts[best] >= max(5, counts["double"] + 1) else "double"


def _tag_name(match):
    name = match.group(1)
    if name is None:
        return None
    if name.split()[0] in _NOT_NAMES:
        return None
    return name


def find_tag(text, start, end):
    """Return {"quote", "name"|None, "pronoun"|None} for a tag adjacent to text[start:end]."""
    after = text[end:end + 80]
    for pattern in _AFTER:
        match = pattern.match(after)
        if match:
            value = match.group(1)
            quote = after[:match.end()].strip(" ,")
            if pattern is _AFTER[2]:
                return {"quote": quote, "name": None, "pronoun": value.lower()}
            name = _tag_name(match)
            if name:
                return {"quote": quote, "name": name, "pronoun": None}
    before = text[max(0, start - 80):start]
    match = _BEFORE.search(before)
    if match:
        name = _tag_name(match)
        if name:
            return {"quote": match.group(0).rstrip(" ,:"), "name": name, "pronoun": None}
    return None


_BEAT_AFTER = re.compile(rf"^[\s,]*{_NAME}(?:'s)?\s+[a-z]")
_BEAT_START = re.compile(rf"^{_NAME}(?:'s)?\s+[a-z]")


def find_beat(text, start, end, first_in_block):
    """Name of an adjacent action beat: `"..." Upgrade deadpanned` or a paragraph opening
    `Divinity rolled her eyes. "..."`. A beat nominates a speaker for verification only;
    it is not attribution evidence."""
    match = _BEAT_AFTER.match(text[end:end + 60])
    if match and match.group(1).split()[0] not in _NOT_NAMES:
        return match.group(1)
    if first_in_block and start > 0:
        match = _BEAT_START.match(text[:start])
        if match and match.group(1).split()[0] not in _NOT_NAMES:
            return match.group(1)
    return None


def quote_spans(source, style=None):
    """Return ordered quote spans for every block with stable short IDs.

    Each span: {"id": "q12", "block_id", "start", "end", "text", "closed",
    "continues": bool (opens a paragraph after an unclosed speech), "tag": {...}|None}.
    ``style`` defaults to the book's ``dialogue_style``.
    """
    finder = _FINDERS[style or dialogue_style(source)]
    spans, previous_unclosed, previous_chapter = [], False, None
    for block in source["blocks"]:
        text = block["text"]
        if block["chapter_id"] != previous_chapter:
            previous_unclosed, previous_chapter = False, block["chapter_id"]
        found = finder(text)
        for index, (start, end, closed) in enumerate(found):
            if not text[start:end].strip('"“”‘’\'—–―- \t'):
                continue
            spans.append({"id": f"q{len(spans) + 1}", "block_id": block["id"], "start": start, "end": end,
                          "text": text[start:end], "closed": closed,
                          "continues": index == 0 and start == 0 and previous_unclosed,
                          "tag": find_tag(text, start, end),
                          "beat": find_beat(text, start, end, index == 0)})
        previous_unclosed = bool(found) and not found[-1][2]
    return spans


def spans_by_block(spans):
    grouped = {}
    for span in spans:
        grouped.setdefault(span["block_id"], []).append(span)
    return grouped
