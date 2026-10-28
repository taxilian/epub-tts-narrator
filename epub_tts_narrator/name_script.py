"""Compact derived name markup; canonical source is never markup input."""

import re

from .pronunciations import WORD_PATTERN
from .pipeline import fingerprint


FILENAME = "name-script.json"
VERSION = 1
_MARKER = re.compile(r"\(N([1-9][0-9]*):([^()\\]+)\)")


def escape_text(text):
    """Escape all parentheses and backslashes, including source marker lookalikes."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def alias_words(name):
    return {token.group().lower() for alias in name["aliases"] for token in WORD_PATTERN.finditer(alias)}


def parse_text(annotated, source, names):
    """Unwrap exact source and derive Unicode offsets, rejecting arbitrary markup."""
    if not isinstance(annotated, str):
        raise ValueError("Name script text must be a string")
    plain, mentions, position, length = [], [], 0, 0
    while position < len(annotated):
        char = annotated[position]
        if char == "\\":
            if position + 1 == len(annotated) or annotated[position + 1] not in "\\()":
                raise ValueError("Invalid name script escape")
            plain.append(annotated[position + 1])
            length += 1
            position += 2
        elif char == "(":
            match = _MARKER.match(annotated, position)
            if match is None:
                raise ValueError("Malformed or nested name marker")
            name_id, word = "N" + match[1], match[2]
            if name_id not in names or WORD_PATTERN.fullmatch(word) is None or word.lower() not in alias_words(names[name_id]):
                raise ValueError("Unknown name ID or non-alias name marker")
            mentions.append({"start": length, "end": length + len(word), "name_id": name_id})
            plain.append(word)
            length += len(word)
            position = match.end()
        elif char == ")":
            raise ValueError("Unescaped name script parenthesis")
        else:
            plain.append(char)
            length += 1
            position += 1
    if "".join(plain) != source:
        raise ValueError("Name script must unwrap to exact canonical source")
    words = {(match.start(), match.end()) for match in WORD_PATTERN.finditer(source)}
    if any((item["start"], item["end"]) not in words for item in mentions):
        raise ValueError("Name markers must cover whole source words")
    return mentions


def identity_table(document):
    """Character identity context includes description/evidence, never voice profiles."""
    return {character["id"]: {"character_id": character["id"],
                              "aliases": sorted(set([character["name"], *character.get("aliases", [])])),
                              "description": character.get("description", ""),
                              "evidence": character.get("evidence", [])}
            for character in document["cast"].get("characters", [])}


def binding(document):
    return {"source": fingerprint(document["source"]), "identities": fingerprint(identity_table(document)),
            "version": VERSION}


def candidates(text, names):
    """Enumerate every alias-token occurrence without deciding its meaning."""
    words = {key: alias_words(value) for key, value in names.items()}
    return [{"start": token.start(), "end": token.end(), "word": token.group(), "names": possible}
            for token in WORD_PATTERN.finditer(text)
            if (possible := [key for key, aliases in words.items() if token.group().lower() in aliases])]


def annotate(text, occurrences, decisions):
    result, cursor = [], 0
    for item, decision in zip(occurrences, decisions, strict=True):
        result.append(escape_text(text[cursor:item["start"]]))
        word = text[item["start"]:item["end"]]
        result.append(f"({decision}:{word})" if decision in item["names"] else escape_text(word))
        cursor = item["end"]
    result.append(escape_text(text[cursor:]))
    return "".join(result)
