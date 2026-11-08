"""Deterministic stretched-letter spelling proposals for narrator-read segments.

The narrator read's model notes remain the source of dialect and performance
proposals. This pass only collapses unmistakable alphabetic run-length stretches
in spoken rows, using the book itself to choose an attested spelling when possible.
Every attached edit goes through the shared inline-performance collection and
publication machinery; protected or otherwise non-mechanical candidates are omitted.
"""

from collections import Counter
import re

from .attribution_performance import _edit_outcome


_WORDS = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*", re.UNICODE)
_STRETCH = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)
_ROMAN_NUMERAL = re.compile(r"[IVXLCDM]+")
_SPOKEN_KINDS = {"dialogue", "thought", "letter"}
_DIRECTION = "drawn-out delivery"
_REASON = "Deterministic normalization of an elongated letter run."


def _collapse(word: str, length: int) -> str:
    return _STRETCH.sub(lambda match: match.group(1) * length, word)


def _fallback(word: str) -> str:
    return _STRETCH.sub(lambda match: match.group(1) * (2 if len(match.group()) == 3 else 1), word)


def _replacement(word: str, corpus: Counter[str], names: list[str]) -> str | None:
    candidates = (_collapse(word, 1), _collapse(word, 2))
    # Do not let a less-collapsed variant evade a guard that the ordinary
    # spelling trips (notably an elongated established character name).
    if any(_edit_outcome({"original": word, "replacement": candidate}, names) == "protected"
           for candidate in candidates):
        return None
    attested = [candidate for candidate in candidates if corpus[candidate.casefold()]]
    if attested:
        replacement = min(attested, key=lambda candidate: (
            -corpus[candidate.casefold()], len(word) - len(candidate)))
    else:
        replacement = _fallback(word)
    spec = {"original": word, "replacement": replacement}
    return replacement if _edit_outcome(spec, names) == "mechanical" else None


def _covered_edit_spans(row: dict) -> list[tuple[int, int]]:
    spans = []
    payload = row.get("inline_proposals") or {}
    for spec in payload.get("edits", []):
        composed = spec["left"] + spec["original"] + spec["right"]
        at = row["text"].find(composed)
        if at != -1:
            start = at + len(spec["left"])
            spans.append((start, start + len(spec["original"])))
    return spans


def _locator(text: str, start: int, end: int) -> tuple[str, str]:
    original = text[start:end]
    if text.count(original) == 1:
        return "", ""
    # The complete row is a guaranteed unique composed locator. Repeated tokens
    # are uncommon, and mechanical proposals never send this context to a model.
    return text[:start], text[end:]


def attach_stretch_proposals(document: dict) -> dict:
    """Merge valid deterministic edits into narrator-read inline proposals.

    Existing model edit spans win, including spans later rejected by review, so
    the deterministic pass never duplicates or overrides a model proposal.
    """
    corpus = Counter(match.group().casefold()
                     for block in document["source"]["blocks"]
                     for match in _WORDS.finditer(block["text"]))
    names = [name for character in document.get("cast", {}).get("characters", [])
             for name in ([character.get("name", ""), *character.get("aliases", [])]) if name]
    for row in document["segments"]:
        if row["kind"] not in _SPOKEN_KINDS:
            continue
        covered = _covered_edit_spans(row)
        edits = []
        for match in _WORDS.finditer(row["text"]):
            word = match.group()
            if _ROMAN_NUMERAL.fullmatch(word) or not _STRETCH.search(word):
                continue
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in covered):
                continue
            replacement = _replacement(word, corpus, names)
            if replacement is None:
                continue
            left, right = _locator(row["text"], start, end)
            edits.append({"original": word, "replacement": replacement,
                          "direction": _DIRECTION, "reason": _REASON,
                          "left": left, "right": right})
        if edits:
            payload = row.setdefault("inline_proposals", {"edits": [], "controls": []})
            payload["edits"].extend(edits)
    return document
