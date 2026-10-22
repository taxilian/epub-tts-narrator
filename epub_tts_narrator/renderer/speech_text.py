"""Target-only text preparation; source and saved voice transcripts stay exact."""


_EDGE_QUOTES = frozenset("\"'‘’‚‛“”„‟«»‹›")


def trim_speech_text(value: str) -> str:
    """Remove wrapping quotes and Unicode whitespace, leaving the interior exact.

    This is for the disposable synthesis input, never source offsets, reference
    transcripts, or voice descriptions. Alternating whitespace/quotation marks
    are removed together, including unmatched marks at segment boundaries.
    """
    start, end = 0, len(value)
    while start < end and (value[start].isspace() or value[start] in _EDGE_QUOTES):
        start += 1
    while end > start and (value[end - 1].isspace() or value[end - 1] in _EDGE_QUOTES):
        end -= 1
    return value[start:end]
