"""Validate authored voice descriptions without adding performance context."""

import re


MAX_PROMPT_CHARACTERS = 4000
# Historical application text, retained only to recognize saved prompts.
# Context for new voices belongs in the authored description, not a wrapper.
AUDIOBOOK_VOICE_CONTEXT = (
    "Adult voice mimicking a character that sounds like"
)
_LEGACY_CONTEXT_PREFIX = AUDIOBOOK_VOICE_CONTEXT + " "
_LEGACY_CONTEXT_PREFIXES = (
    _LEGACY_CONTEXT_PREFIX,
    "Adult audiobook narrator portraying a character in a book. "
    "Create an adult performer's imitation of the following vocal qualities: ",
    "audiobook narrator portraying a character in a book: ",
)


def voice_prompt_description(prompt):
    """Remove only known application wrappers for legacy provenance comparisons."""
    # One released wrapper accidentally supplied two separator spaces. Treat
    # that exact boundary as legacy, but do not lstrip arbitrary description
    # whitespace (three or more spaces can include deliberate custom text).
    legacy_double = _LEGACY_CONTEXT_PREFIX + " "
    while True:
        if prompt.startswith(legacy_double) and not prompt[len(legacy_double):].startswith(" "):
            prompt = prompt[len(legacy_double):]
            continue
        for prefix in _LEGACY_CONTEXT_PREFIXES:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):]
                break
        else:
            return prompt


def audiobook_voice_prompt(prompt):
    """Pass through the complete description, removing historical wrappers only."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Voice prompt must be nonempty text")
    description = voice_prompt_description(prompt)
    if not description.strip():
        raise ValueError("Voice prompt must include a nonempty description")
    if len(description) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"Voice prompt must be at most {MAX_PROMPT_CHARACTERS} characters; "
            f"shorten the description by {len(description) - MAX_PROMPT_CHARACTERS} characters")
    return description


# Hosted voice designers refuse to design a minor's voice, and read some words
# as sexual. A voice description says how a character sounds; youth is a sound
# (light, bright, high), never an age or a word for a child. Rewrites are
# general vocabulary, applied without adding any prefix.
_YOUNG_VOICE = re.compile(
    r"\b(?:(?:very\s+|quite\s+)?(?:young|little|small)\s+)?(?:teen(?:age)?\s+(?P<gender>boy|girl)|(?P<gender2>boy|girl)"
    r"|child|kid)(?:'s|s'|s)?\s+voice\b", re.I)
# Gendered words for the young keep their gender: a boy is a male voice.
_YOUNG_MALE = re.compile(r"\b(?:(?:very\s+|quite\s+)?(?:young|little|small)\s+)?(?:teen(?:age)?\s+)?"
                         r"(?:boys?|lads?|schoolboys?)\b", re.I)
_YOUNG_FEMALE = re.compile(r"\b(?:(?:very\s+|quite\s+)?(?:young|little|small)\s+)?(?:teen(?:age)?\s+)?"
                           r"(?:girls?|lass(?:es)?|schoolgirls?)\b", re.I)
_YOUNG_ADJECTIVE = re.compile(r"\b(?:boyish|girlish|childlike|childish|kiddish)\b", re.I)
_YOUNG_NOUN = re.compile(
    r"\b(?:(?:very\s+|quite\s+)?(?:young|little|small)\s+)?(?:teen(?:ager)?s?"
    r"|child(?:ren)?|kids?|kiddos?|adolescents?|toddlers?|infants?|bab(?:y|ies)|pre-?teens?|tweens?"
    r"|youngsters?|minors?|juveniles?)\b", re.I)
_YOUNG_STATE = re.compile(r"\b(?:pre-?pubescent|pubescent|teenaged?|underage|(?:early|mid|late)[- ]teens)\b", re.I)
_UNDER_EIGHTEEN = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen"
                   "|sixteen|seventeen")
_AGE_PHRASES = (
    re.compile(r"\b(?:about|around|roughly|approximately|aged?|nearly|almost|maybe|perhaps|some)\s+(\d{1,2})"
               r"(?:\s*(?:years?|yrs?)(?:\s+old)?)?(?![\w%])", re.I),
    re.compile(r"\b(\d{1,2})[- ]?(?:years?|yrs?)[- ]?old\b", re.I),
)
_AGE_WORDS = re.compile(rf"\b(?:{_UNDER_EIGHTEEN})[- ](?:years?[- ]old|year)\b", re.I)
# "adult" on its own describes a grown voice and is kept; beside a word for an
# entertainer it reads as sexual.
_PERFORMER = re.compile(
    r"\badult\s+(?:voice\s+)?(?:performers?|actors?|actress(?:es)?|entertainers?|stars?|models?)\b"
    r"|\bperformers?\b", re.I)
_SEXUAL_READING = re.compile(
    r"\b(?:sexy|seductive|sultry|sensual|erotic|intimate|explicit|lustful|provocative)\b", re.I)


def safe_voice_description(text):
    """A voice description a hosted designer can accept for any character.

    Minor-age wording becomes sound and keeps its gender ("a young boy's voice"
    -> "a light, bright male voice"; "teen boy" -> "male"; "boyish" -> "light,
    bright"; an ungendered "child" -> "character"), ages
    under 18 and words with a sexual reading are removed ("adult performer"
    becomes "voice actor"), and the punctuation left behind is tidied. Adult
    ages and every acoustic quality are kept.
    """
    def young_voice(match):
        gender = (match.group("gender") or match.group("gender2") or "").casefold()
        return {"boy": "light, bright male voice", "girl": "light, bright female voice"}.get(gender, "light, bright voice")

    result = _YOUNG_VOICE.sub(young_voice, text)
    result = _YOUNG_ADJECTIVE.sub("light, bright", result)

    def under_age(match):
        return "" if int(match.group(1)) < 18 else match.group(0)

    for pattern in _AGE_PHRASES:
        result = pattern.sub(under_age, result)
    result = _AGE_WORDS.sub("", result)
    result = _YOUNG_STATE.sub("", result)
    result = _YOUNG_MALE.sub("male", result)
    result = _YOUNG_FEMALE.sub("female", result)
    result = _YOUNG_NOUN.sub("character", result)
    result = _PERFORMER.sub("voice actor", result)
    result = _SEXUAL_READING.sub("", result)
    # Tidy what removal leaves behind: empty brackets, doubled or leading punctuation.
    result = re.sub(r"\(\s*[,;:]?\s*\)", "", result)
    result = re.sub(r"\s+([,.;:])", r"\1", result)
    result = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", result)
    result = re.sub(r"(^|[.!?]\s+)[,;:]\s*", r"\1", result)
    result = re.sub(r"\b(a|an)\s+(?=[,.;:]|$)", "", result, flags=re.I)
    # "a"/"an" before the words inserted above.
    result = re.sub(r"\b([Aa])n(\s+(?:light|character|voice|male)\b)", r"\1\2", result)
    result = re.sub(r"\b([Aa])(\s+(?:female)\b)", r"\1\2", result)
    result = re.sub(r"\s{2,}", " ", result).strip(" ,;:")
    return result[:1].upper() + result[1:] if result else "A clear, natural speaking voice."


def designer_prompt(prompt, designer):
    """The prompt a designer receives: hosted Gemini design gets ``safe_voice_description``.

    A retired application prefix on a saved prompt is kept verbatim (so older
    requests still resume as they were made); only the description is rewritten.
    """
    if (designer or {}).get("engine") != "gemini_voice_design":
        return prompt
    description = voice_prompt_description(prompt)
    prefix = prompt[:len(prompt) - len(description)] if prompt.endswith(description) else ""
    return prefix + safe_voice_description(description)


def default_voice_prompt(profile):
    """Use the complete prose design; trait records remain evidence metadata."""
    # Never silently trim a distinctive quality out of an overlong default.
    return audiobook_voice_prompt(profile["description"])
