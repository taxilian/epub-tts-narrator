"""How the narration says a line is spoken, for that line's own speaker.

Books say how lines sound in the tag or beat beside them: `"Robbers?" Wolf
laughed.`, `Silk sighed. "Durnik," he said`, `"Run!" she shouted.`, `he said
softly`. A model asked to mark these across a long request misses many, so
code finds them, using only structure the analysis already has: each spoken
row's speaker, the narration around it, and the speaker's names and pronoun.

A clause belongs to a line when it touches the line (the first sentence after
it, or the last sentence before it) and its subject is that line's speaker: one
of the speaker's names or aliases, or a pronoun that fits the speaker. A clause
about anyone else, or with an unknown subject, gives nothing. Negated words
("without laughing", "didn't sigh") give nothing.

The cue classes follow Drama 3's documented controls: point sounds
(``[laughs]``, ``[sighs]`` ...), scoped ``<whisper>``/``<slow>``, and tone words
for the line's direction (shouting, screaming, soft, hurried). The clause itself
is also handed to the director model, so its direction agrees with the book.
"""

import re

from .quote_spans import SPEECH_VERBS


SPOKEN = frozenset({"dialogue", "thought", "letter"})
REASON = "The narration beside this line says how it is spoken."
MAX_CLAUSE = 160

# Sound words -> performance_plan.VOCAL_SOUNDS.
_SOUNDS = (
    ("throat_clear", r"clear(?:s|ed|ing)? (?:his|her|their|my|its) throat"),
    ("chuckle", r"chuckl\w*|chortl\w*|snicker\w*|snigger\w*"),
    ("laugh", r"laugh(?:s|ed|ing)?|laughter|guffaw\w*|cackl\w*|giggl\w*"),
    ("sigh", r"sigh(?:s|ed|ing)?"),
    ("gasp", r"gasp(?:s|ed|ing)?"),
    ("sob", r"sob(?:s|bed|bing)?|wept|weep(?:s|ing)?"),
    ("groan", r"groan(?:s|ed|ing)?|moan(?:s|ed|ing)?"),
    ("pant", r"pant(?:s|ed|ing)|panting|wheez\w*"),
    ("yawn", r"yawn(?:s|ed|ing)?"),
    ("cough", r"cough(?:s|ed|ing)?"),
)
# A manner adverb counts only beside a speech verb: "said softly", not "whistled softly".
def _spoken(adverbs):
    return rf"(?:{SPEECH_VERBS})\s+(?:\w+\s+)?(?:{adverbs})|(?:{adverbs})\s+(?:{SPEECH_VERBS})"


# Tone words: scoped tags where Drama 3 documents one, else a direction word.
# A speech-tag "breathed" (line-final or with a recipient) is cautiously soft;
# breathing actions such as "breathed deeply" or "breathed in" give no tone.
_TONES = (
    ("whisper", r"whisper\w*"),
    ("screaming", r"scream\w*|shriek\w*|screech\w*|wail(?:s|ed|ing)?|howl(?:s|ed|ing)?"),
    ("shouting", r"shout\w*|yell\w*|roar(?:s|ed|ing)?|bellow\w*|bawl(?:s|ed|ing)?|holler\w*|"
                 r"thunder(?:s|ed|ing)|called out|cried out|bark(?:s|ed|ing)?|" + _spoken("loudly")),
    ("soft", r"murmur\w*|breathed(?=\s*(?:$|[,.;:!?])|\s+(?:at|to|towards?)\b)|"
             r"under (?:his|her|their|my) breath|" + _spoken("softly|quietly|gently")),
    ("hurried", r"blurt\w*|gabbl\w*|babbl\w*|" + _spoken("hurriedly|rapidly|quickly|breathlessly")),
    ("slow", r"drawl\w*|" + _spoken("slowly")),
)
SCOPED_TONES = {"whisper": "whisper", "slow": "slow"}
# A direction word for the other tones, and the words that show a direction already has it.
TONE_WORDS = {"screaming": ("screaming", r"scream|shriek"),
              "shouting": ("shouting", r"shout|yell|roar|bellow|loud|bark|call"),
              "soft": ("soft and quiet", r"soft|quiet|gentl|hush|murmur|low"),
              "hurried": ("hurried, quick", r"hurr|quick|fast|rush|blurt|rapid|breathless")}
_SOUND = [(name, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)) for name, pattern in _SOUNDS]
_TONE = [(name, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)) for name, pattern in _TONES]
_NEGATION = re.compile(r"\b(?:not|never|no|without|stopped|ceased|refused to|tried not to)\s+(?:\w+\s+){0,2}$|"
                       r"\bwithout\s+(?:so\s+much\s+as|any)\s+(?:\w+\s+){0,2}$|n['’]t\s+(?:\w+\s+){0,1}$",
                       re.IGNORECASE)
_UNREALIZED = re.compile(r"\b(?:almost|nearly|about\s+to|started\s+to|began\s+to)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)
_SILENT = re.compile(r"^\s*,?\s*(?:silently|in\s+silence|without\s+a\s+sound|noiselessly)\b", re.IGNORECASE)
_INDIRECT = re.compile(r"\b(?:hear(?:s|d)?|see(?:s|d)?|saw|watch(?:es|ed)?|listen(?:s|ed)?|notic\w+|observ\w+|remember\w+|recall\w+|report\w+|describ\w+)\b",
                       re.IGNORECASE)
_INDIRECT_BREAK = re.compile(r"\b(?:and|but|then|so|or|nor)\b|[;,]", re.IGNORECASE)
_LAUGHTER = re.compile(r"^\s+with\s+(?:[\w'’-]+\s+){0,2}laughter\b", re.IGNORECASE)
_MANNER_BEFORE = re.compile(rf"(?:{SPEECH_VERBS})\s*,\s*$", re.IGNORECASE)
_MANNER_AFTER = re.compile(rf"^\s*,\s*(?:{SPEECH_VERBS})\b", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?…])[\"”’']?\s+")
_LEAD = re.compile(r"^[\s,;:\-—–]*(?:(?:and|then|but|so)\s+)?", re.IGNORECASE)
_INVERTED_TAG = re.compile(rf"^(?:{SPEECH_VERBS})\s+", re.IGNORECASE)
_PRONOUNS = {"he": "m", "him": "m", "his": "m", "she": "f", "her": "f", "i": "i", "my": "i"}
_GENDER = (("m", re.compile(r"\b(?:male|man|boy|masculine)\b", re.IGNORECASE)),
           ("f", re.compile(r"\b(?:female|woman|girl|feminine)\b", re.IGNORECASE)))


def _sex(description):
    found = [(match.start(), sex) for sex, pattern in _GENDER for match in [pattern.search(description or "")] if match]
    return min(found)[1] if found else None


def speakers(cast):
    """role id -> {"names": casefolded names and aliases, "tokens": their words, "sex", "narrator"}."""
    characters = {character["id"]: character for character in cast.get("characters", [])}
    voices = {voice["id"]: voice for voice in cast.get("voices", [])}
    narrating = next((role.get("character_id") for role in cast.get("roles", []) if role["id"] == "narrator_main"), None)
    found = {}
    for role in cast.get("roles", []):
        if role.get("kind") == "narration":
            continue
        character = characters.get(role.get("character_id"), {})
        names = [name for name in [character.get("name", ""), *character.get("aliases", [])] if name]
        voice = voices.get(role.get("voice_id"), {})
        found[role["id"]] = {
            "names": {name.casefold() for name in names},
            "tokens": {token.casefold() for name in names for token in re.findall(r"[^\W\d_][\w'’-]*", name)
                       if token[:1].isupper()},
            "sex": _sex(voice.get("description") or character.get("description")),
            "narrator": narrating is not None and role.get("character_id") == narrating,
        }
    return found


def _about(clause, speaker):
    """True when the clause's subject is this speaker."""
    text = _INVERTED_TAG.sub("", _LEAD.sub("", clause), count=1)
    words = re.findall(r"[^\W\d_][\w'’-]*", text)
    if not words:
        return False
    first = words[0]
    pronoun = _PRONOUNS.get(first.casefold())
    if pronoun is not None:
        if pronoun == "i":
            return speaker["narrator"]
        return speaker["sex"] in (None, pronoun)
    if first.casefold() == "the":
        # "the old man laughed": a descriptive alias.
        phrase = " ".join(words[:4]).casefold()
        return any(phrase.startswith(name) for name in speaker["names"] if name.startswith("the "))
    if not first[:1].isupper():
        return False
    run = []
    for word in words:
        if not word[:1].isupper() or word.casefold() in _PRONOUNS:
            break
        run.append(word.casefold().removesuffix("'s").removesuffix("’s"))
    return bool(run) and (" ".join(run) in speaker["names"] or all(token in speaker["tokens"] for token in run))


def _own(clause):
    """The clause up to another subject: "Faldor laughed and Durnik sighed" -> "Faldor laughed"."""
    match = re.search(r"(?:;|,? (?:and|while|as) (?=[A-Z][a-z]|(?:he|she|they) ))", clause[1:])
    return clause[:match.start() + 1] if match else clause


# A bare tag ("Mara said.", "he asked her") says nothing about how a line sounds.
_PLAIN = frozenset("""said says asked asks replied replies answered answers told tells added adds continued
continues went on agreed agrees to him her them me us you it the a an and then""".split())


def _informative(clause, speaker):
    words = [word.casefold() for word in re.findall(r"[^\W\d_][\w'’-]*", clause)]
    return any(word not in _PLAIN and word not in _PRONOUNS and word not in speaker["tokens"] for word in words)


def _produced(prefix, speaker):
    """True when the speaker performs the cue, rather than perceiving or reporting it."""
    sentence = re.split(r"[.!?…]", prefix)[-1]
    for match in _INDIRECT.finditer(sentence):
        if not _INDIRECT_BREAK.search(sentence[match.end():]):
            return False
    named = re.search(r"([A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*){0,2})(?:\s+(?:had|has|was|is|could|would|should|might|must))?\s+$", prefix)
    return not named or all(word.casefold().removesuffix("'s").removesuffix("’s") in speaker["tokens"]
                            for word in named.group(1).split())


def _find(patterns, clause, speaker):
    for name, pattern in patterns:
        for match in pattern.finditer(clause):
            prefix, suffix = clause[:match.start()], clause[match.end():]
            if _NEGATION.search(prefix) or _UNREALIZED.search(prefix) or not _produced(prefix, speaker):
                continue
            if patterns is _SOUND and _SILENT.match(suffix):
                continue
            if name == "pant" and match.group().casefold().endswith("ing") and (_MANNER_BEFORE.search(prefix) or _MANNER_AFTER.match(suffix)):
                continue
            if name in {"screaming", "shouting"} and _LAUGHTER.match(suffix):
                continue
            return name
    return None


def _clauses(text, rows):
    """id(row) -> candidate clauses [(clause, side)] from the narration touching each spoken row."""
    spoken = [row for row in rows if row["kind"] in SPOKEN]
    found = {id(row): [] for row in spoken}
    for index in range(len(spoken) + 1):
        before = spoken[index - 1] if index else None
        after = spoken[index] if index < len(spoken) else None
        start = before["end"] if before else 0
        end = after["start"] if after else len(text)
        gap = text[start:end]
        sentences = [piece.strip() for piece in _SENTENCES.split(gap) if piece.strip(" ,;:—–-\"“”")]
        if not sentences:
            continue
        if before is not None:
            found[id(before)].append((sentences[0][:MAX_CLAUSE], "after"))
        if after is not None and (before is None or len(sentences) > 1):
            found[id(after)].append((sentences[-1][-MAX_CLAUSE:], "before"))
        elif after is not None:
            # One sentence between two lines: it belongs to whichever speaker it is about.
            found[id(after)].append((sentences[0][:MAX_CLAUSE], "between"))
    return found


def line_cues(rows, text, people):
    """id(row) -> {"clause", "sound", "tone", "side"} for spoken rows whose own speaker the narration describes.

    ``rows`` are all of one block's segments in order, with offsets into ``text``;
    ``people`` comes from ``speakers(cast)``. ``side`` locates the clause that
    produced the sound, defaulting to ``before`` when there is no sound.
    """
    cues = {}
    clauses = _clauses(text, rows)
    claimed = set()
    for row in [row for row in rows if row["kind"] in SPOKEN]:
        group = row.get("simultaneous_speakers")
        if group or row.get("role_id") not in people:
            continue
        speaker = people[row["role_id"]]
        kept = []
        for clause, side in clauses.get(id(row), []):
            if side == "between" and clause in claimed:
                continue
            if _about(clause, speaker):
                kept.append((_own(clause), side))
                if side == "after":
                    claimed.add(clause)
        kept = [(clause, side) for clause, side in kept if _informative(clause, speaker)]
        if not kept:
            continue
        joined = " ".join(clause for clause, _ in kept)
        sound = _find(_SOUND, joined, speaker) if row["kind"] == "dialogue" else None
        tone = _find(_TONE, joined, speaker) if row["kind"] == "dialogue" else None
        side = next((side for clause, side in kept if _find(_SOUND, clause, speaker) == sound), "before") if sound else "before"
        cues[id(row)] = {"clause": " / ".join(clause for clause, _ in kept), "sound": sound, "tone": tone, "side": side}
    return cues


def tone_direction(direction, tone):
    """The line's direction with a tone word the narration gives, unless it already says so."""
    if tone not in TONE_WORDS:
        return direction
    if tone == "soft":
        direction = ", ".join(part for part in (piece.strip() for piece in direction.split(","))
                              if part and not re.search(r"\bwhisper\w*\b", part, re.IGNORECASE))
    word, present = TONE_WORDS[tone]
    if re.search(rf"\b(?:{present})", direction, re.IGNORECASE):
        return direction
    return f"{word}, {direction}" if direction else word
