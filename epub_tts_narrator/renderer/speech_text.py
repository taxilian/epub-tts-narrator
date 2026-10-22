"""Target-only text preparation; source and saved voice transcripts stay exact."""


_EDGE_QUOTES = frozenset("\"'‘’‚‛“”„‟«»‹›")
_PAREN_EDGES = _EDGE_QUOTES | frozenset("()")
_SENTENCE_ENDS = frozenset(".,;:!?…")


def speech_text_parts(value: str) -> tuple[str, str, str]:
    """Prepare a target and retain its removed leading/trailing boundaries.

    Source segmentation can leave an opening parenthesis on one speaker's
    fragment and its closing mark on another. Strip those edge marks, including
    a closing mark before sentence punctuation, and pairs enclosing the whole
    target. Balanced interior parentheticals remain intact. Original indices
    identify the edge boundaries even when the prepared text is not contiguous.
    """
    leading_end, trailing_start = 0, len(value)
    while leading_end < len(value) and (value[leading_end].isspace()
                                        or value[leading_end] in _PAREN_EDGES):
        leading_end += 1
    while trailing_start > 0 and (value[trailing_start - 1].isspace()
                                  or value[trailing_start - 1] in _PAREN_EDGES | _SENTENCE_ENDS):
        trailing_start -= 1
    stack, pairs, unmatched = [], [], []
    for index, character in enumerate(value):
        if character == "(":
            stack.append(index)
        elif character == ")":
            if stack:
                pairs.append((stack.pop(), index))
            else:
                unmatched.append(index)
    removed = {index for index in unmatched + stack
               if index < leading_end or index >= trailing_start}
    for opening, closing in pairs:
        if opening < leading_end and closing >= trailing_start:
            removed.update((opening, closing))
    indices = [index for index in range(len(value)) if index not in removed]
    start, end = 0, len(indices)
    while start < end and (value[indices[start]].isspace() or value[indices[start]] in _EDGE_QUOTES):
        start += 1
    while end > start and (value[indices[end - 1]].isspace() or value[indices[end - 1]] in _EDGE_QUOTES):
        end -= 1
    kept = indices[start:end]
    if not kept:
        return "", value, ""
    return "".join(value[index] for index in kept), value[:kept[0]], value[kept[-1] + 1:]


def trim_speech_text(value: str) -> str:
    """Remove nonspoken edge formatting from disposable synthesis input only.

    Never apply this to source offsets, reference transcripts, or descriptions.
    """
    return speech_text_parts(value)[0]
