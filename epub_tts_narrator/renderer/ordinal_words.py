"""Bounded deterministic ordinal equivalents for speech verification.

A voice reads "7th" and "seventh" as the same spoken word, so verification
may accept either spelling of one spoken ordinal. A changed value ("8th"),
a cardinal ("seven"), a malformed suffix ("7st"), a leading zero ("07th"),
a signed form ("-7th"), a decimal ("7.5th") and values at or above one
million stay literal so incorrect speech is never excused. The numeric range
mirrors the bounded cardinal expansion in ``quality.normalize_words``.
"""

import re

_ORDINAL = re.compile(
    r"(0|[1-9][0-9]{0,5}|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(st|nd|rd|th)"
)
_BOUNDED_DIGITS = re.compile(r"0|[1-9][0-9]{0,5}")
_ORDINAL_WORDS = {
    "zero": "zeroth", "one": "first", "two": "second", "three": "third",
    "four": "fourth", "five": "fifth", "six": "sixth", "seven": "seventh",
    "eight": "eighth", "nine": "ninth", "ten": "tenth", "eleven": "eleventh",
    "twelve": "twelfth", "thirteen": "thirteenth", "fourteen": "fourteenth",
    "fifteen": "fifteenth", "sixteen": "sixteenth", "seventeen": "seventeenth",
    "eighteen": "eighteenth", "nineteen": "nineteenth", "twenty": "twentieth",
    "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth",
    "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth",
    "ninety": "ninetieth", "hundred": "hundredth", "thousand": "thousandth",
}


def _canonical_suffix(value: int) -> str:
    # The teens keep "th": 11th, 12th, 13th, 111th, 112th, 113th, ...
    if value % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def ordinal_words(token: str, integer_words) -> list[str] | None:
    """Expand one casefolded lexeme such as "7th" or "1,000th" to spoken words.

    Returns the ordinal word list built from the caller's bounded cardinal
    expander, or None when the token is not a well-formed, correctly
    suffixed, in-range ordinal digit form. Only the final spoken word takes
    the ordinal form: "21st" reads "twenty first" like its cardinal "21".
    """
    match = _ORDINAL.fullmatch(token)
    if match is None:
        return None
    # Like the cardinal path, grouped digits shed commas before the range
    # check, so "7,000,000th" and every value >= one million stays literal.
    digits = match.group(1).replace(",", "")
    if not _BOUNDED_DIGITS.fullmatch(digits):
        return None
    value = int(digits)
    if _canonical_suffix(value) != match.group(2):
        return None
    words = integer_words(value)
    return words[:-1] + [_ORDINAL_WORDS[words[-1]]]
