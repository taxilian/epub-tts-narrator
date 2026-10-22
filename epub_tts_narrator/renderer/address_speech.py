"""Scoped URL/email spelling edges for the spoken-coverage gate.

This only makes a complete address eligible for the independent sound judge.
It never changes source text or accepts audio. Ordinary prose does not acquire
optional punctuation words, letter spelling, or missing-word allowances.
"""

import re

from .letter_names import _NAME_TO_LETTER


_ADDRESS = re.compile(
    r"(?<![\w@])(?:[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"
    r"|(?:https?://|www\.)[^\s<>\"\[\]{}]+)", re.IGNORECASE)
_SPACE = r"(?>\s*)"
_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_SYMBOL_NAMES = {"@": ("at",), ".": ("dot", "point"), "/": ("slash", "forward slash"),
                 ":": ("colon",), "-": ("dash", "hyphen"), "_": ("underscore",),
                 "+": ("plus",), "?": ("question mark",), "=": ("equals", "equal"),
                 "&": ("ampersand", "and"), "#": ("hash", "pound"), "%": ("percent",)}


def _spoken_word(word):
    return r"\s+".join(re.escape(part) for part in word.split())


def _component(value):
    """An unchanged component, or its complete ordered character spelling."""
    letters = []
    for char in value.casefold():
        glyphs = [char]
        names = []
        if char.isdigit():
            names.append(_DIGITS[int(char)])
            if char == "0":
                names.append("oh")
                glyphs.append("o")
        else:
            names.extend(name for name, letter in _NAME_TO_LETTER.items() if letter.casefold() == char)
            if char == "w":
                names.extend(("double u", "double you"))
        # Letter names are complete spoken words, never arbitrary prefixes
        # inside a component. Atomic choices and separators prevent exponential
        # reconsideration of repeated ambiguous names after a late mismatch.
        options = [r"(?<![a-z0-9])" + _spoken_word(name) + r"(?![a-z0-9])"
                   for name in sorted(set(names), key=len, reverse=True)]
        options.extend(re.escape(glyph) for glyph in glyphs)
        letters.append("(?>" + "|".join(options) + ")")
    spelled = r"(?>[\s.\-]*)".join(letters)
    return "(?>" + re.escape(value) + r"(?![a-z0-9])|" + spelled + ")"


def _symbol(value, *, path=False):
    options = [re.escape(value)]
    options.extend(_spoken_word(name) for name in _SYMBOL_NAMES[value])
    if value == "/" and path:
        # ASR often writes a read path as a dotted web address. All actual
        # components still have to match, and JEV must judge the full passage.
        options.append(r"\.")
    return _SPACE + "(?>" + "|".join(options) + ")" + _SPACE


def _pattern(address):
    email = "@" in address and not re.match(r"https?://", address, re.IGNORECASE)
    scheme = re.match(r"(https?)://", address, re.IGNORECASE)
    prefix = ""
    if scheme:
        # Reading a website without its transport prefix is conventional.
        # If a prefix is read, all its letters and separators must be covered.
        prefix = "(?:" + _component(scheme[1]) + _symbol(":") + _symbol("/") + _symbol("/") + ")?"
        address = address[scheme.end():]
    tokens = re.findall(r"[a-z0-9]+|[^a-z0-9]", address, re.IGNORECASE)
    if any(not token.isalnum() and token not in _SYMBOL_NAMES for token in tokens):
        return None
    parts = []
    for index, token in enumerate(tokens):
        if token.isalnum():
            parts.append(_component(token))
        else:
            part = _symbol(token, path=not email)
            # A terminal URL slash has no spoken payload. Interior separators
            # remain required; missing path components never become optional.
            if token == "/" and index == len(tokens) - 1 and not email:
                part = "(?:" + part + ")?"
            parts.append(part)
    # The caller checks full lexeme boundaries; a trailing spoken separator
    # may also have consumed whitespace up to the following prose word.
    return re.compile(prefix + "".join(parts), re.IGNORECASE)


def make_address_matcher(expected_source, expected, expected_spans,
                         actual_source, actual, actual_spans):
    """Yield whole-address edges using the existing lexeme offset coordinates."""
    addresses = {}
    starts = {span[0]: index for index, span in enumerate(expected_spans)}
    for match in _ADDRESS.finditer(expected_source):
        raw = match.group().rstrip(".,;!)]}")
        if not raw or len(raw) > 1024:
            continue
        end = match.start() + len(raw)
        begin_index = starts.get(match.start())
        if begin_index is None:
            continue
        end_index = begin_index
        while end_index < len(expected_spans) and expected_spans[end_index][1] <= end:
            end_index += 1
        pattern = _pattern(raw)
        if pattern is not None and end_index > begin_index:
            addresses[begin_index] = (end_index, pattern)

    def matches(i, j):
        address = addresses.get(i)
        if address is None:
            return
        end_i, pattern = address
        match = pattern.match(actual_source, actual_spans[j][0])
        if match is None:
            return
        end_j = j
        while end_j < len(actual_spans) and actual_spans[end_j][1] <= match.end():
            end_j += 1
        if end_j == j or (end_j < len(actual_spans) and actual_spans[end_j][0] < match.end()):
            return
        yield end_i, end_j, {
            "operation": "address_pronunciation", "rule": "complete_address_components",
            "expected_start": i, "expected_end": end_i, "actual_start": j, "actual_end": end_j,
            "expected": expected[i:end_i], "actual": actual[j:end_j],
        }
    return matches
