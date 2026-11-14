"""Conservative, complete word coverage required before a text-only judgment.

Passing this check means a transcript is eligible for pronunciation review,
not that it has been approved. Every alignment edge consumes words on both
sides. There is deliberately no percentage allowance for missing speech.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
import re
import unicodedata

from .letter_names import letter_name_spellings, make_letter_name_matcher
from .phonetic_match import (
    _ALPHABETIC_WORD, _COLLOQUIAL_EXPANSIONS, _FUNCTION_WORDS, _PROTECTED, _PROTECTED_NUMERIC,
    _REVIEW_COLLOQUIAL_EXPANSIONS, _load_double_metaphone, phonetic_equivalence,
)


SPEECH_COVERAGE_VERSION = 16
_MAX_ALIGNMENT_STATES = 20_000
# Non-lexical vocal sounds a performance may add or leave out ("Oh,", "Ha!").
# They carry no words, so omitting, adding or changing one is a performance
# difference for review, never a missing or extra word. Bounded per passage.
_VOCALIZATIONS = frozenset("""
ah aah ahh oh ohh ooh uh uhh um umm er erm eh hm hmm mm mmm huh heh ha hah
""".split())
# A closed-mouth hum the text asks for is a deliberate audible beat; a take
# may add one, but leaving it out stays a missing sound no judge can excuse.
_HUMS = frozenset("hm hmm mm mmm".split())
_MAX_SPOKEN_GAPS = 2
# Symbols with no fixed spoken form: a reader may say them or pass over them.
_SKIPPABLE_SYMBOLS = frozenset({"/", "*"})
_SYMBOL_NAMES = {"/": frozenset({"slash", "stroke"})}
_DEGREE_UNITS = {"c": frozenset({"celsius", "centigrade", "c"}),
                 "f": frozenset({"fahrenheit", "f"}), "k": frozenset({"kelvin", "k"})}
_CONTRACTION_SUFFIXES = {"s": ("is", "has"), "ve": ("have",), "ll": ("will",),
                         "d": ("had", "would"), "re": ("are",), "m": ("am",)}
_GENERAL_CONTRACTION = re.compile(r"([a-z]+)'(s|ve|ll|d|re|m)\Z")
_ROMAN_NUMERAL = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\Z")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_VOWEL_PHONES = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_CERTAIN_OPERATIONS = frozenset({"colloquial", "contraction", "spelling_boundary", "letter_names",
                                 "address_pronunciation", "roman_numeral", "symbol_reading"})


def contraction_expansions(word):
    """Spelled-out forms of a contraction, for any host word.

    The fixed table keeps its pronoun-specific readings; otherwise a word plus
    's, 've, 'll, 'd, 're or 'm is its host followed by the auxiliary
    ("else's" → "else is"/"else has", "to've" → "to have").
    """
    if word in _CONTRACTIONS:
        return _CONTRACTIONS[word]
    match = _GENERAL_CONTRACTION.fullmatch(word)
    if match is None or "n't" in word:
        return ()
    return tuple((match.group(1), auxiliary) for auxiliary in _CONTRACTION_SUFFIXES[match.group(2)])


def roman_numeral_words(spelling):
    """Cardinal words for an uppercase Roman numeral of two or more letters, else None."""
    from .quality import _integer_words
    if (not isinstance(spelling, str) or len(spelling) < 2 or not spelling.isupper()
            or _ROMAN_NUMERAL.fullmatch(spelling) is None):
        return None
    values = [_ROMAN_VALUES[letter] for letter in spelling]
    total = sum(-value if index + 1 < len(values) and value < values[index + 1] else value
                for index, value in enumerate(values))
    return tuple(_integer_words(total))


@lru_cache(maxsize=4096)
def _dictionary_phones(word):
    from ..pronunciation_dictionary import dictionary_readings
    return tuple(re.sub(r"\d", "", reading) for reading in dictionary_readings(word))


def _non_rhotic(phones):
    """A reading as a non-rhotic accent says it: no R before a consonant or pause."""
    phones = phones.split()
    kept = []
    for index, phone in enumerate(phones):
        if phone == "R" and (index + 1 == len(phones) or phones[index + 1] not in _VOWEL_PHONES):
            continue
        kept.append("AH" if phone == "ER" else phone)
    return " ".join(kept)


def _one_edit_per_four(first, second):
    """Levenshtein distance within one edit per four sounds of the longer code."""
    allowed = max(len(first), len(second)) // 4
    if allowed < 1 or abs(len(first) - len(second)) > allowed:
        return False
    row = list(range(len(second) + 1))
    for index, left in enumerate(first, 1):
        previous, row[0] = row[0], index
        for column, right in enumerate(second, 1):
            previous, row[column] = row[column], min(row[column] + 1, row[column - 1] + 1,
                                                     previous + (left != right))
    return row[-1] <= allowed


def dictionary_homophone(left, right):
    """Two different dictionary words that share a reading ("high"/"hi", "aloud"/"allowed").

    Quantities stay protected here; a long passage's "two" heard as "to" is a
    close transcript (``close_match``) instead.
    """
    return (left != right and left not in _PROTECTED_NUMERIC and right not in _PROTECTED_NUMERIC
            and bool(set(_dictionary_phones(left)) & set(_dictionary_phones(right))))


def non_rhotic_homophone(left, right, metaphone):
    """Words that sound alike once R is dropped before a consonant or pause.

    Many English accents never sound such an R, so "floor" and "flaw" or an
    invented "Killa" and "killer" are said alike. Dictionary readings decide
    when both words have one; otherwise their Double Metaphone codes after
    dropping a written r not followed by a vowel must agree.
    """
    if left == right or not ("r" in left or "r" in right):
        return False
    if any(len(word) < 3 or not re.fullmatch(r"[a-z]+", word) or word in _FUNCTION_WORDS
           or word in _PROTECTED for word in (left, right)):
        return False
    readings = [_dictionary_phones(word) for word in (left, right)]
    if all(readings):
        return bool({_non_rhotic(reading) for reading in readings[0]}
                    & {_non_rhotic(reading) for reading in readings[1]})
    codes = [{key for key in metaphone(re.sub(r"r+(?![aeiouy])", "", word)) if len(key) >= 2}
             for word in (left, right)]
    return bool(codes[0] & codes[1])
_SPELLING_VARIANTS: dict[frozenset[str], str] = {
    frozenset(("qi", "chi")): "standard_transliteration",
}
_TITLE_ABBREVIATIONS = frozenset({
    frozenset(("mister", "mr")),
    frozenset(("doctor", "dr")),
    frozenset(("professor", "prof")),
    frozenset(("saint", "st")),
    frozenset(("captain", "capt")),
    frozenset(("sergeant", "sgt")),
})
_CONTRACTIONS = {
    "i'm": (("i", "am"),), "let's": (("let", "us"),),
    "can't": (("can", "not"),), "cannot": (("can", "not"),),
    "won't": (("will", "not"),), "shan't": (("shall", "not"),),
    **{f"{word}n't": ((word, "not"),) for word in (
        "are", "is", "was", "were", "do", "does", "did", "have", "has", "had",
        "could", "would", "should", "must", "need", "dare",
    )},
    **{f"{word}'re": ((word, "are"),) for word in ("you", "we", "they", "who")},
    **{f"{word}'ve": ((word, "have"),) for word in ("i", "you", "we", "they", "could", "would", "should", "must")},
    **{f"{word}'ll": ((word, "will"),) for word in ("i", "you", "he", "she", "it", "we", "they", "who", "that", "there")},
    **{f"{word}'d": ((word, "had"), (word, "would")) for word in ("i", "you", "he", "she", "it", "we", "they", "who", "that", "there")},
    **{f"{word}'s": ((word, "is"), (word, "has")) for word in ("he", "she", "it", "who", "what", "that", "there", "here")},
}


def stretched_word_equivalent(expected, actual):
    """Recognize shortening a visibly elongated word, never a missing word.

    One source run must contain at least three repeated letters. Once that
    identifies deliberate elongation, other doubled runs may shorten too:
    ``Wheerrre`` becomes ``Where``. Ordinary ``dessert``/``desert`` does not
    qualify. The target preserves every distinct letter and apostrophe in
    order, cannot lengthen any run, and cannot add or remove a word boundary.
    This permits sound review; it does not approve a take by itself.
    """
    if not isinstance(expected, str) or not isinstance(actual, str):
        return False
    expected, actual = (value.casefold().replace("’", "'") for value in (expected, actual))
    if (not _ALPHABETIC_WORD.fullmatch(expected) or not _ALPHABETIC_WORD.fullmatch(actual)
            or len(actual.replace("'", "")) < 2 or expected == actual):
        return False
    left, right = ([match.group() for match in re.finditer(r"(.)\1*", value)]
                   for value in (expected, actual))
    return (len(left) == len(right)
            and any(run[0].isalpha() and len(run) >= 3 for run in left)
            and all(a[0] == b[0] and len(b) <= len(a) for a, b in zip(left, right)))


def _near_phonetic_word(left, right, left_keys, right_keys):
    """Admit a small within-word sound change for review, never acceptance.

    Metaphone keys are rough sound encodings, not evidence that speech is
    correct. Permit one insertion, deletion, or substitution within complete
    keys only when at least three encoded sounds remain and the edit accounts
    for at most a quarter of the longer key. Whole-word alignment still has
    no insertion/deletion edges. Grammar words, quantities, and negations do
    not gain this tolerance, and the sound judge must approve the candidate.
    """
    if any(len(word.replace("'", "")) < 4 or word in _FUNCTION_WORDS
           or word.replace("'", "") in _PROTECTED or "n't" in word
           for word in (left, right)):
        return False
    for first in left_keys:
        for second in right_keys:
            if min(len(first), len(second)) < 3 or max(len(first), len(second)) < 4:
                continue
            if len(first) == len(second):
                if sum(a != b for a, b in zip(first, second)) == 1:
                    return True
            else:
                shorter, longer = sorted((first, second), key=len)
                if len(longer) == len(shorter) + 1 and any(
                        longer[:index] + longer[index + 1:] == shorter
                        for index in range(len(longer))):
                    return True
    return False


def speech_coverage(expected_text, transcript):
    """Return an auditable eligibility result without making any API calls.

    In addition to established complete phonetic matches, permit exact
    contractions and spelling boundaries, and bounded pronunciation candidates
    such as ``admitted`` / ``admit it`` or ``libary`` / ``library``. These extra
    edges still require an independent sound judgment; they cannot approve audio.
    """
    from .quality import _lexeme_matches, normalize_words, word_spacing_equivalence

    result = {"passed": False, "version": SPEECH_COVERAGE_VERSION,
              "reason": "unmatched_spoken_content", "alignment": []}
    if (not isinstance(expected_text, str) or not isinstance(transcript, str)
            or len(expected_text) > 100_000 or len(transcript) > 100_000):
        return dict(result, reason="invalid_text")

    def tokens(text):
        source = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
        words, spans = [], []
        for match in _lexeme_matches(source):
            normalized = normalize_words(match.group())
            words.extend(normalized)
            spans.extend([(match.start(), match.end(), match.group().casefold())] * len(normalized))
        return source, words, spans

    expected_source, expected, expected_spans = tokens(expected_text)
    actual_source, actual, actual_spans = tokens(transcript)
    if not expected or not actual:
        return dict(result, reason="empty_spoken_content")
    if len(expected) > 2000 or len(actual) > 4000:
        return dict(result, reason="comparison_limit")
    if expected == actual:
        return dict(result, passed=True, reason="exact_spoken_content")
    spacing = word_spacing_equivalence(expected_text, transcript)
    if spacing is not None:
        return dict(result, passed=True, reason="exact_spelling_boundaries",
                    alignment=spacing["alignment"])
    if Counter(expected) == Counter(actual):
        return dict(result, reason="reordered_spoken_content")

    complete = phonetic_equivalence(expected_text, transcript)
    if complete is not None:
        return dict(result, passed=True, reason="complete_phonetic_coverage",
                    alignment=complete["alignment"])
    metaphone = _load_double_metaphone()
    letter_matches = make_letter_name_matcher(
        expected_source, expected, expected_spans, actual_source, actual, actual_spans,
        protected=_PROTECTED)
    from .address_speech import make_address_matcher
    address_matches = make_address_matcher(
        expected_source, expected, expected_spans, actual_source, actual, actual_spans)

    @lru_cache(maxsize=None)
    def keys(spelling):
        return tuple(dict.fromkeys(key for key in metaphone(spelling) if len(key) >= 2))

    def eligible_word(word):
        bare = word.replace("'", "")
        return (bool(_ALPHABETIC_WORD.fullmatch(word)) and len(bare) >= 2
                and bare not in _PROTECTED and "n't" not in word)

    def contiguous(side, start, size, *, hyphen=False):
        source, words, spans = ((expected_source, expected, expected_spans)
                                if side == "expected" else
                                (actual_source, actual, actual_spans))
        if start + size > len(words):
            return False
        return all((bool(re.fullmatch(r"[\s\-‐‑]+", source[spans[index - 1][1]:spans[index][0]]))
                    if hyphen else source[spans[index - 1][1]:spans[index][0]].isspace())
                   for index in range(start + 1, start + size))

    def exact_compound(i, j):
        end_i, end_j = i + 1, j + 1
        left, right = expected[i].replace("'", ""), actual[j].replace("'", "")
        while left != right:
            if len(left) < len(right):
                if not right.startswith(left) or end_i == len(expected):
                    return None
                left += expected[end_i].replace("'", "")
                end_i += 1
            else:
                if not left.startswith(right) or end_j == len(actual):
                    return None
                right += actual[end_j].replace("'", "")
                end_j += 1
        left_text = expected_source[expected_spans[i][0]:expected_spans[end_i - 1][1]]
        right_text = actual_source[actual_spans[j][0]:actual_spans[end_j - 1][1]]
        if word_spacing_equivalence(left_text, right_text) is not None:
            return end_i, end_j
        return None

    def span_words(side, start, size):
        source, words, spans = ((expected_source, expected, expected_spans)
                                if side == "expected" else
                                (actual_source, actual, actual_spans))
        end = start + size
        if end > len(words):
            return None
        selected = words[start:end]
        if (not all(eligible_word(word) or (size > 1 and word == span[2] and letter_name_spellings(span[2]))
                    for word, span in zip(selected, spans[start:end]))
                or any(not _ALPHABETIC_WORD.fullmatch(span[2]) for span in spans[start:end])
                or not contiguous(side, start, size, hyphen=True)):
            return None
        return selected

    @lru_cache(maxsize=None)
    def group_spellings(side, start, size):
        words = span_words(side, start, size)
        if words is None:
            return ()
        spans = expected_spans if side == "expected" else actual_spans
        options = [letter_name_spellings(spans[start + index][2]) or (word,)
                   for index, word in enumerate(words)]
        return tuple(tuple(word.replace("'", "").replace(" ", "") for word in spellings)
                     for spellings in product(*options))

    @lru_cache(maxsize=None)
    def group_keys(side, start, size):
        found = set()
        for spoken in group_spellings(side, start, size):
            full = set(keys("".join(spoken)))
            if size > 1:
                # Every complete component, including a spoken letter, must contribute.
                for index in range(size):
                    remainder = "".join(spoken[:index] + spoken[index + 1:])
                    full.difference_update(keys(remainder))
            found.update(full)
        return tuple(sorted(found))

    def one_repeated_consonant(left, right):
        shorter, longer = sorted((left, right), key=len)
        if len(shorter) < 3 or len(longer) != len(shorter) + 1 or shorter[0] != longer[0]:
            return False
        return any(longer[index] == longer[index - 1]
                   and longer[:index] + longer[index + 1:] == shorter
                   for index in range(1, len(longer)))

    def one_spelling_letter(left, right):
        shorter, longer = sorted((left.replace("'", ""), right.replace("'", "")), key=len)
        return len(longer) == len(shorter) + 1 and any(
            longer[:index] + longer[index + 1:] == shorter for index in range(len(longer)))

    def spelling_variant(left, right):
        folded = ["".join(char for char in unicodedata.normalize("NFD", word)
                          if not unicodedata.combining(char)) for word in (left, right)]
        if not all(eligible_word(word) for word in folded):
            return None
        if folded[0] == folded[1]:
            return "accent_spelling"
        # Dialects differ on retaining silent e after soft dg before -ment(s).
        regional = [re.sub(r"(?<=dg)e(?=ments?$)", "", word) for word in folded]
        if regional[0] == regional[1]:
            return "regional_spelling"
        return _SPELLING_VARIANTS.get(frozenset(folded))

    def bounded_pronunciation(left, right, left_keys, right_keys, i, j):
        # All rules cover a whole token. They confer eligibility for the model
        # to judge the sound, never permission to remove a neighboring word.
        if all(re.fullmatch(r"h?m{2,}", word) for word in (left, right)):
            return "nasal_hum"
        runs = [re.findall(r"(.)\1*", word) for word in (left, right)]
        if runs[0] == runs[1]:
            groups = [re.findall(r"((.)\2*)", word) for word in (left, right)]
            lengths = [[len(run) for run, _ in parts] for parts in groups]
            if (any(a != b for a, b in zip(*lengths))
                    and all(a == b or max(a, b) >= 3 for a, b in zip(*lengths))):
                return "stretched_spelling"
        for longer, shorter in ((left, right), (right, left)):
            if longer.endswith("ing") and len(longer) >= 5 and shorter == longer[:-1]:
                return "colloquial_ing"
        if any(a[0] in {"J", "X"} and b[0] in {"J", "X"} and a[1:] == b[1:]
               for a in left_keys for b in right_keys):
            return "affricate_voicing"
        if one_spelling_letter(left, right) and any(
                one_repeated_consonant(a, b) for a in left_keys for b in right_keys):
            return "repeated_phonetic_consonant"
        if (not any(source[span[0]:span[1]].isupper()
                    for source, span in ((expected_source, expected_spans[i]),
                                         (actual_source, actual_spans[j])))
                and _near_phonetic_word(left, right, left_keys, right_keys)):
            return "near_phonetic_word"
        return None

    def terminal_s_variation(left, right, i, j):
        # A weak final /s/ or /z/ can disappear in connected speech or its
        # transcription. Retain BOTH complete words and require a sound judge;
        # this is not stemming, grammatical correction, or local acceptance.
        if (any(word in _CONTRACTIONS or word.replace("'", "") in _PROTECTED
                or word in _FUNCTION_WORDS for word in (left, right))):
            return None
        base, inflected = sorted((left, right), key=len)
        if (not re.fullmatch(r"[a-z]{3,}", base) or not re.search(r"[aeiouy]", base)
                or inflected not in {base + "s", base + "'s"}):
            return None
        for source, span in ((expected_source, expected_spans[i]),
                             (actual_source, actual_spans[j])):
            if source[span[0]:span[1]].isupper():
                # Check both sides: ASR can lowercase a shortened initialism.
                # A final spoken letter is not a weak grammatical ending.
                return None
        return "terminal_s_variation"

    def change_record(operation, rule, i, end_i, j, end_j):
        return {"operation": operation, "rule": rule, "expected_start": i, "expected_end": end_i,
                "actual_start": j, "actual_end": end_j,
                "expected": expected[i:end_i], "actual": actual[j:end_j]}

    def raw(side, index):
        source, spans = ((expected_source, expected_spans) if side == "expected"
                         else (actual_source, actual_spans))
        return source[spans[index][0]:spans[index][1]]

    def elided(form, span, source, words, start, side):
        """End positions where an apostrophe-elided spelling is read in full.

        A leading apostrophe ("'em") or an inner one that is not a
        contraction ending ("g'night") marks one to three left-out letters.
        The full form keeps every written letter in order around them.
        """
        if (not _ALPHABETIC_WORD.fullmatch(form) or "n't" in form or contraction_expansions(form)
                or form in _COLLOQUIAL_EXPANSIONS):
            return  # Contractions and listed colloquial forms have their own rules.
        if "'" in form:
            head, _, tail = form.partition("'")
            if "'" in tail or not head or len(tail) < 2:
                return
        elif span[0] > 0 and source[span[0] - 1] == "'" and len(form) >= 2:
            head, tail = "", form
        else:
            return
        for size in (1, 2):
            if start + size > len(words) or (size > 1 and not contiguous(side, start, size)):
                continue
            pieces = words[start:start + size]
            if not all(re.fullmatch(r"[a-z]+", piece) for piece in pieces):
                continue
            joined = "".join(pieces)
            if (joined.startswith(head) and joined.endswith(tail)
                    and 1 <= len(joined) - len(head) - len(tail) <= 3):
                yield start + size

    def written_readings(i, j):
        """Bounded readings of a written form that compare words, never drop one."""
        left, right = expected[i], actual[j]
        for reverse in (False, True):
            form, span, source, words, start, side = (
                (actual[j], actual_spans[j], actual_source, expected, i, "expected") if reverse else
                (expected[i], expected_spans[i], expected_source, actual, j, "actual"))
            for end in elided(form, span, source, words, start, side):
                yield ((end, j + 1) if reverse else (i + 1, end)) + (change_record(
                    "pronunciation_candidate", "elided_spelling",
                    i, end if reverse else i + 1, j, j + 1 if reverse else end),)
            numeral = roman_numeral_words(raw("actual" if reverse else "expected", j if reverse else i))
            if numeral is not None and tuple(words[start:start + len(numeral)]) == numeral:
                end = start + len(numeral)
                yield ((end, j + 1) if reverse else (i + 1, end)) + (change_record(
                    "roman_numeral", "uppercase_roman_numeral",
                    i, end if reverse else i + 1, j, j + 1 if reverse else end),)
        units = _DEGREE_UNITS.get(left)
        before = expected_source[:expected_spans[i][0]].rstrip()
        if units is not None and before.endswith("°"):
            for size in (1, 2):
                spoken = actual[j:j + size]
                if (len(spoken) == size and spoken[-1] in units
                        and (size == 1 or spoken[0] in {"degree", "degrees"})):
                    yield i + 1, j + size, change_record("symbol_reading", "degree_unit", i, i + 1, j, j + size)
        names = _SYMBOL_NAMES.get(left)
        if names is not None and right in names:
            yield i + 1, j + 1, change_record("symbol_reading", "symbol_name", i, i + 1, j, j + 1)
        if not all(_ALPHABETIC_WORD.fullmatch(word) for word in (left, right)) or left == right:
            return
        if any(source[span[0]:span[1]].isupper() and len(word) > 1
               for source, span, word in ((expected_source, expected_spans[i], left),
                                          (actual_source, actual_spans[j], right))):
            return  # An initialism is spelled letter by letter, not read as a word.
        if dictionary_homophone(left, right):
            yield i + 1, j + 1, change_record("pronunciation_candidate", "dictionary_homophone", i, i + 1, j, j + 1)
        elif non_rhotic_homophone(left, right, metaphone):
            yield i + 1, j + 1, change_record("pronunciation_candidate", "non_rhotic", i, i + 1, j, j + 1)

    def invented_name_sounds(i, j):
        """A name no dictionary knows, heard as nearby sounds across new word breaks.

        A recognizer cannot spell an invented name, so it splits, joins or
        respells it ("Killashandra Ree" heard as "Kila Chandra Rhee"). One or
        two source words, at least one of them a capitalized word outside the
        dictionary, may match one to three heard words when the Double
        Metaphone codes of the joined spellings differ by at most one edit per
        four sounds. Heard grammar words may sit inside the name; negations
        and quantities may not, and a heard word equal to a source word
        around the span is never absorbed. The sound judges still decide.
        """
        for left_size in (1, 2):
            source = expected[i:i + left_size]
            if (len(source) < left_size or not contiguous("expected", i, left_size)
                    or not all(re.fullmatch(r"[a-z]{2,}", word) and word not in _PROTECTED
                               and word not in _FUNCTION_WORDS for word in source)):
                continue
            invented = [word for index, word in enumerate(source)
                        if len(word) >= 4 and not _dictionary_phones(word)
                        and raw("expected", i + index)[:1].isupper()
                        and not raw("expected", i + index).isupper()]
            if not invented:
                continue
            neighbours = set(expected[max(0, i - 1):i]) | set(expected[i + left_size:i + left_size + 1])
            left_codes = {key for key in metaphone("".join(source)) if len(key) >= 3}
            for right_size in (1, 2, 3):
                if left_size == right_size == 1:
                    continue  # One-word respellings keep their established rules.
                heard = actual[j:j + right_size]
                if (len(heard) < right_size or not contiguous("actual", j, right_size)
                        or not all(re.fullmatch(r"[a-z]+", word) and word not in _PROTECTED
                                   for word in heard)
                        or set(heard) & (neighbours | set(source))):
                    continue
                right_codes = {key for key in metaphone("".join(heard)) if len(key) >= 3}
                if any(first[0] == second[0] and _one_edit_per_four(first, second)
                       for first in left_codes for second in right_codes):
                    yield i + left_size, j + right_size, change_record(
                        "pronunciation_candidate", "invented_name_sound",
                        i, i + left_size, j, j + right_size)

    def attached_hyphen(side, index):
        # "-4" in "23-4-3308" or "-43NW" after "F42NW": a joining hyphen read
        # as nothing, not a spoken minus sign. A spaced "-4" keeps its sign.
        words, spans, source = ((expected, expected_spans, expected_source) if side == "expected"
                                else (actual, actual_spans, actual_source))
        start = spans[index][0]
        return (words[index] == "minus" and source[start:start + 1] == "-" and start > 0
                and source[start - 1].isalnum()
                and (index == 0 or spans[index - 1][:2] != spans[index][:2]))

    def gap_matches(i, j, gaps):
        """Edges that consume words on one side only: unspoken symbols and vocal sounds."""
        for side, index in (("expected", i), ("actual", j)):
            words = expected if side == "expected" else actual
            if index >= len(words):
                continue
            step = (i + 1, j) if side == "expected" else (i, j + 1)
            change = (change_record("symbol_reading", None, i, step[0], j, step[1]))
            if words[index] in _SKIPPABLE_SYMBOLS:
                yield step + (gaps, dict(change, rule="unspoken_symbol"))
            elif attached_hyphen(side, index):
                yield step + (gaps, dict(change, rule="joining_hyphen"))
        if gaps >= _MAX_SPOKEN_GAPS:
            return
        if i < len(expected) and j < len(actual) and expected[i] != actual[j] and {
                expected[i], actual[j]} <= _VOCALIZATIONS:
            yield i + 1, j + 1, gaps + 1, change_record("vocalization", "changed_vocalization", i, i + 1, j, j + 1)
        if i < len(expected) and expected[i] in _VOCALIZATIONS - _HUMS:
            yield i + 1, j, gaps + 1, change_record("vocalization", "omitted_vocalization", i, i + 1, j, j)
        if j < len(actual) and actual[j] in _VOCALIZATIONS:
            yield i, j + 1, gaps + 1, change_record("vocalization", "added_vocalization", i, i, j, j + 1)
        if (j + 1 < len(actual) and re.fullmatch(r"[a-z]+", actual[j])
                and (len(actual[j]) <= 3 or actual[j] == actual[j + 1])
                and actual[j + 1].startswith(actual[j])
                and re.fullmatch(r"[\-‐‑]", actual_source[actual_spans[j][1]:actual_spans[j + 1][0]])):
            # "T-Trag", "Or-or": a hyphen-joined restart of the following word.
            yield i, j + 1, gaps + 1, change_record("stammer", "hyphenated_restart", i, i, j, j + 1)

    def next_matches(i, j):
        yield from address_matches(i, j)
        if expected[i] == actual[j]:
            yield i + 1, j + 1, None
            yield from letter_matches(i, j)
            return
        yield from letter_matches(i, j)
        pair = (expected[i], actual[j])
        review_rule = None
        if all(_ALPHABETIC_WORD.fullmatch(word) for word in pair):
            if frozenset(pair) in _TITLE_ABBREVIATIONS:
                review_rule = "title_abbreviation"
            elif frozenset(pair) == {"i", "l"} and all(len(word) == 1 for word in pair):
                review_rule = "single_letter_homoglyph"
        if review_rule is not None:
            yield i + 1, j + 1, {"operation": "pronunciation_candidate", "rule": review_rule,
                "expected_start": i, "expected_end": i + 1, "actual_start": j,
                "actual_end": j + 1, "expected": [pair[0]], "actual": [pair[1]]}
            return
        boundary = exact_compound(i, j)
        if boundary is not None:
            end_i, end_j = boundary
            yield end_i, end_j, {"operation": "spelling_boundary", "expected_start": i,
                "expected_end": end_i, "actual_start": j, "actual_end": end_j,
                "expected": expected[i:end_i], "actual": actual[j:end_j]}
        if stretched_word_equivalent(expected[i], actual[j]):
            yield i + 1, j + 1, {"operation": "pronunciation_candidate", "rule": "stretched_spelling",
                "expected_start": i, "expected_end": i + 1, "actual_start": j,
                "actual_end": j + 1, "expected": expected[i:i + 1], "actual": actual[j:j + 1]}
        variant = spelling_variant(expected[i], actual[j])
        if variant is not None:
            yield i + 1, j + 1, {"operation": "pronunciation_candidate", "rule": variant,
                "expected_start": i, "expected_end": i + 1, "actual_start": j,
                "actual_end": j + 1, "expected": expected[i:i + 1], "actual": actual[j:j + 1]}
        left_expansions, right_expansions = contraction_expansions(expected[i]), contraction_expansions(actual[j])
        if set(left_expansions).intersection(right_expansions):
            yield i + 1, j + 1, {"operation": "contraction", "expected_start": i,
                "expected_end": i + 1, "actual_start": j, "actual_end": j + 1,
                "expected": expected[i:i + 1], "actual": actual[j:j + 1]}
        for reverse in (False, True):
            expansions = right_expansions if reverse else left_expansions
            words, start, side = (expected, i, "expected") if reverse else (actual, j, "actual")
            for expansion in expansions:
                size = len(expansion)
                if tuple(words[start:start + size]) != expansion or not contiguous(side, start, size):
                    continue
                end_i, end_j = (i + size, j + 1) if reverse else (i + 1, j + size)
                yield end_i, end_j, {"operation": "contraction", "expected_start": i,
                    "expected_end": end_i, "actual_start": j, "actual_end": end_j,
                    "expected": expected[i:end_i], "actual": actual[j:end_j]}
        for reverse in (False, True):
            form = actual[j] if reverse else expected[i]
            expansion = _COLLOQUIAL_EXPANSIONS.get(form)
            if expansion is None:
                continue
            source, words, spans, start = ((expected_source, expected, expected_spans, i)
                                          if reverse else
                                          (actual_source, actual, actual_spans, j))
            colloquial_source, colloquial_span = ((actual_source, actual_spans[j]) if reverse
                                                 else (expected_source, expected_spans[i]))
            if (form == "cause" and (colloquial_span[0] == 0
                                     or colloquial_source[colloquial_span[0] - 1] != "'")):
                continue
            size = len(expansion)
            if (tuple(words[start:start + size]) != expansion
                    or any(not source[spans[index - 1][1]:spans[index][0]].isspace()
                           for index in range(start + 1, start + size))):
                continue
            end_i, end_j = (i + size, j + 1) if reverse else (i + 1, j + size)
            review_only = form in _REVIEW_COLLOQUIAL_EXPANSIONS
            change = {"operation": "pronunciation_candidate" if review_only else "colloquial",
                "expected_start": i, "expected_end": end_i, "actual_start": j, "actual_end": end_j,
                "expected": expected[i:end_i], "actual": actual[j:end_j]}
            if review_only:
                change.update(rule="colloquial_expansion", expansion=list(expansion))
            yield end_i, end_j, change
        yield from written_readings(i, j)
        yield from invented_name_sounds(i, j)
        for left_size, right_size in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1)):
            if letter_matches.blocks_fuzzy(i, j, left_size, right_size):
                continue
            left = span_words("expected", i, left_size)
            right = span_words("actual", j, right_size)
            if left is None or right is None:
                if left_size != 1 or right_size != 1:
                    continue
                pair = (expected[i], actual[j])
                if (not any(word in _PROTECTED for word in pair)
                        or any(not _ALPHABETIC_WORD.fullmatch(word)
                               or len(word.replace("'", "")) < 2
                               or word in _PROTECTED_NUMERIC or "n't" in word for word in pair)
                        or any(not _ALPHABETIC_WORD.fullmatch(span[2])
                               or source[span[0]:span[1]].isupper()
                               for source, span in ((expected_source, expected_spans[i]),
                                                    (actual_source, actual_spans[j])))):
                    continue
                left_keys = tuple(key for key in metaphone(pair[0].replace("'", "")) if key)
                right_keys = tuple(key for key in metaphone(pair[1].replace("'", "")) if key)
                if set(left_keys).intersection(right_keys):
                    # Short full codes may nominate a protected homophone, but
                    # never establish approval or consume an absent word.
                    yield i + 1, j + 1, {"operation": "pronunciation_candidate",
                        "rule": "protected_homophone", "expected_start": i, "expected_end": i + 1,
                        "actual_start": j, "actual_end": j + 1,
                        "expected": [pair[0]], "actual": [pair[1]],
                        "expected_keys": list(left_keys), "actual_keys": list(right_keys)}
                continue
            multi = left_size != right_size
            if multi and set(left).intersection(right):
                continue
            if any(sum(bool(letter_name_spellings(word)) for word in words) > 1
                   for words in (left, right)):
                # Phonetic encoding alone can collapse distinct spoken letters.
                written_left = {"".join(words) for words in group_spellings("expected", i, left_size)}
                written_right = {"".join(words) for words in group_spellings("actual", j, right_size)}
                if written_left.isdisjoint(written_right):
                    continue
            left_keys, right_keys = group_keys("expected", i, left_size), group_keys("actual", j, right_size)
            common = sorted(set(left_keys).intersection(right_keys))
            candidate = None
            if not common:
                candidate = (spelling_variant("".join(left), "".join(right)) if multi else
                             terminal_s_variation(left[0], right[0], i, j)
                             or bounded_pronunciation(left[0], right[0], left_keys, right_keys, i, j))
            if not common and candidate is None and not multi:
                pair = (left[0], right[0])
                if (all(_ALPHABETIC_WORD.fullmatch(word)
                        and len(word.replace("'", "")) >= 2
                        and word not in _PROTECTED_NUMERIC and "n't" not in word
                        and word not in _COLLOQUIAL_EXPANSIONS for word in pair)
                        and not any(source[span[0]:span[1]].isupper()
                                    for source, span in ((expected_source, expected_spans[i]),
                                                         (actual_source, actual_spans[j])))):
                    full_left = tuple(key for key in metaphone(pair[0].replace("'", "")) if key)
                    full_right = tuple(key for key in metaphone(pair[1].replace("'", "")) if key)
                    if set(full_left).intersection(full_right):
                        candidate = "single_word_homophone"
                        left_keys, right_keys = full_left, full_right
            if not common and candidate is None:
                continue
            end_i, end_j = i + left_size, j + right_size
            yield end_i, end_j, {
                "operation": "pronunciation_candidate" if candidate else "phonetic_group",
                "rule": candidate or "complete_group_sound",
                "expected_start": i, "expected_end": end_i,
                "actual_start": j, "actual_end": end_j,
                "expected": left, "actual": right,
                "expected_keys": list(left_keys), "actual_keys": list(right_keys),
            }

    # A state also counts the vocal sounds and restarts consumed on one side
    # only, so no path can excuse more than _MAX_SPOKEN_GAPS of them.
    pending, parents = [(0, 0, 0)], {(0, 0, 0): None}
    while pending:
        state = pending.pop()
        i, j, gaps = state
        if (i, j) == (len(expected), len(actual)):
            changes = []
            while parents[state] is not None:
                previous, change = parents[state]
                if change is not None:
                    changes.append(change)
                state = previous
            changes.reverse()
            uncertain = [change for change in changes if change["operation"] not in _CERTAIN_OPERATIONS]
            if ({word for change in uncertain for word in change["expected"]}
                    & {word for change in uncertain for word in change["actual"]}):
                return dict(result, reason="reordered_spoken_content")
            return dict(result, passed=True, reason="complete_reviewable_coverage", alignment=changes)
        edges = list(gap_matches(i, j, gaps))
        if i < len(expected) and j < len(actual):
            # Two-sided matches come first, so they are explored first.
            edges = [(end_i, end_j, gaps, change) for end_i, end_j, change in next_matches(i, j)] + edges
        for end_i, end_j, end_gaps, change in reversed(edges):
            following = (end_i, end_j, end_gaps)
            if following not in parents:
                if len(parents) >= _MAX_ALIGNMENT_STATES:
                    return dict(result, reason="comparison_limit")
                parents[following] = (state, change)
                pending.append(following)
    return result
