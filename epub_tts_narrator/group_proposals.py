"""Propose groups and their line cues from speaker tags (``COLLECTIVE_VOICES.md`` §5.2).

Code reads the tags, as ``speech_cues`` does for sounds: a model asked to mark
groups across a long request misses many, and the tags say it plainly ("the
cultists intoned in unison", "one of the others observed", "the two dozen
shaven-headed men ... intoned", "half of the kneeling eunuchs murmured").

A character is proposed as a group (``cast.characters[i].group``, basis
``analysis``, the tags quoted as evidence) when its own lines' tags say
several people or one unnamed member speak more often than they name it as one
person. A tag naming the group's own word ("the Dryad offered" for "Dryad
girls") is one of its members. Tags decide, never a name: a character with a
personal name (every word capitalized, none plural, e.g. "Beltira") is never
proposed, so a plural tag on one twin stays a ``group_cues`` flag for a person;
a name that itself names several ("Dryad girls", "eunuch chorus") needs its
group tags to outnumber the others, and any other name ("stout farmer", whose
"the farmer said" is that one man) needs repeated plural tags.

From the same tags come the proposal's size ("the seven", "two dozen"), scale
("the throng" → crowd), voice count and mode (in unison → unison; mostly "one
of them" → pool; "repeating", "echoed back" → staggered), and per-line
overrides with the tag as evidence: ``lanes`` for "half of them" / "several of
them", ``mode`` where one line differs, and pool ``slot``s ("one of the others"
opens a new individual; "the Dryad" or "she" right after keeps the same one).

A proposal renders nothing differently until a person casts the group's
ensemble; characters that already have a group, and lines that already have an
override, are never touched. Deterministic and idempotent. Stdlib only.
"""

from __future__ import annotations

from copy import deepcopy
import re

from .ensembles import MAX_VOICES, proposed_voices
from .group_cues import COLLECTIVE_NOUNS, _WORD, _plural_noun, subject_kind, tag_subject
from .speech_cues import SPOKEN, _clauses

VERSION = 1
NUMBERS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "dozen": 12, "both": 2, "pair": 2, "twins": 2}
_CROWD = re.compile(r"\b(?:crowd|throng|multitude|legions?|hundreds|thousands|host|army|everyone|everybody|"
                    r"dozens|mob|assembl(?:y|ed)|gathered)\b", re.IGNORECASE)
_SEVERAL = re.compile(r"\b(?:several|some of|a few|many)\b", re.IGNORECASE)
_HALF = re.compile(r"\bhalf of\b", re.IGNORECASE)
_STAGGER = re.compile(r"\b(?:repeating|repeated|babble|babbling|echo(?:ed|ing) (?:back|again)|swell(?:ed|ing)|"
                      r"overlapping|one after another)\b", re.IGNORECASE)
_UNISON = re.compile(r"\b(?:in unison|as one|in chorus|together|chorused|intoned|chanted|all at once)\b",
                     re.IGNORECASE)
_PRONOUN = re.compile(r"^(?:he|she|it)$", re.IGNORECASE)


def _head_noun(name):
    """The noun a name is about: its last word, or the word before "of" ("Chorus of ancestors")."""
    before = re.split(r"\bof\b", name, maxsplit=1)[0]
    words = _WORD.findall(before) or _WORD.findall(name)
    return words[-1].split("-")[-1] if words else ""


def _plural_name(name):
    """A name that itself names several ("Dryad girls", "eunuch chorus", "Bear-cultists", "the seven")."""
    words = _WORD.findall(name)
    head = _head_noun(name)
    common = head.islower() or any(word.islower() for word in words)
    return bool(head) and (head.casefold() in COLLECTIVE_NOUNS or (common and _plural_noun(head.casefold()))
                           or any(NUMBERS.get(word.casefold(), 0) > 1 for word in words))


def _personal(name):
    """A person's name ("Beltira", "Taur Urgas"): every word capitalized and no plural head noun."""
    words = _WORD.findall(name)
    return (bool(words) and all(word[:1].isupper() for word in words) and words[0].casefold() != "the"
            and not _plural_name(name) and not _head_noun(name).islower())


def _size(heads):
    """A number the tags give for the group ("the seven", "two dozen"), else None."""
    for head in heads:
        words = [word.casefold() for word in _WORD.findall(head)]
        for index, word in enumerate(words):
            if word == "dozen":
                before = NUMBERS.get(words[index - 1]) if index else None
                return (before or 1) * 12
            if word in NUMBERS and word not in {"both", "pair", "twins"}:
                return NUMBERS[word]
            if word.isdigit() and int(word) >= 2:
                return int(word)
    return None


def _scale(heads, size):
    text = " ".join(heads)
    if size is not None:
        return "pair" if size == 2 else "few" if size == 3 else "several" if size <= 5 else "many" if size <= 12 else "crowd"
    if re.search(r"\b(?:both|the two|the pair|the twins)\b", text, re.IGNORECASE):
        return "pair"
    if _CROWD.search(text):
        return "crowd"
    if _SEVERAL.search(text):
        return "several"
    return None


def _tags(document):
    """``{segment_id: [(kind, head, clause)]}``: each spoken line's own speech tags."""
    by_block = {}
    for row in document["segments"]:
        by_block.setdefault(row["block_id"], []).append(row)
    texts = {block["id"]: block["text"] for block in document["source"]["blocks"]}
    found = {}
    for block_id, rows in by_block.items():
        clauses = _clauses(texts[block_id], rows)
        for row in rows:
            if row.get("kind") not in SPOKEN or row.get("simultaneous_speakers") or not row.get("role_id"):
                continue
            for clause, _ in clauses.get(id(row), []):
                kind = subject_kind(clause)
                if kind is not None:
                    found.setdefault(row["id"], []).append((kind, tag_subject(clause)[0], clause))
    return found


def _quote(text, clause):
    """The clause as it stands in the block (evidence must be an exact substring)."""
    piece = " ".join(clause.split()).strip(" ,;:—–-")
    piece = piece[:160]
    return piece if piece and piece in text else None


def propose_groups(document):
    """``(candidate, changes)``: ``document`` with analysis's group proposals and line cues added."""
    candidate = deepcopy(document)
    cast = candidate["cast"]
    roles = {role["id"]: role for role in cast["roles"]}
    voices = {voice["id"]: voice for voice in cast["voices"]}
    texts = {block["id"]: block["text"] for block in candidate["source"]["blocks"]}
    tags = _tags(candidate)
    lines = {}
    for row in candidate["segments"]:
        role = roles.get(row.get("role_id") or "")
        if role and role.get("character_id") and row["id"] in tags:
            lines.setdefault(role["character_id"], []).append(row)
    changes = []
    for character in cast["characters"]:
        if "group" in character or "ensemble" in voices.get(character["voice_id"], {}):
            continue
        rows = lines.get(character["id"], [])
        if not rows or _personal(character["name"]):
            continue
        plural_name = _plural_name(character["name"])
        own = _own_words(character) if plural_name else set()
        counted = {"plural": 0, "member": 0, "singular": 0}
        evidence, heads = [], []
        for row in rows:
            for kind, head, clause in tags[row["id"]]:
                kind = _effective(kind, head, own)
                counted[kind] += 1
                if kind == "singular":
                    continue
                heads.append(head)
                quote = _quote(texts[row["block_id"]], clause)
                # Plural tags are the strongest evidence; a member tag is kept only when no other is found.
                if quote and len(evidence) < 3 and (kind == "plural" or not evidence):
                    evidence.append({"block_id": row["block_id"], "quote": quote})
        grouped = counted["plural"] + counted["member"]
        if plural_name:
            # Its own word in a tag ("the Dryad") is one of its members; the tags must still say so.
            qualifies = grouped > counted["singular"] and (counted["plural"] >= 1 or counted["member"] >= 2)
        else:
            # A one-person name needs repeated plural tags: one is more likely a misattribution (a flag).
            qualifies = counted["plural"] >= 2 and counted["plural"] > counted["singular"]
        if not evidence or not qualifies:
            continue
        size = _size(heads)
        scale = _scale(heads, size)
        joined = " ".join(clause for row in rows for _, _, clause in tags[row["id"]])
        if _STAGGER.search(joined):
            mode = "staggered"
        elif counted["member"] > counted["plural"] and not _UNISON.search(joined):
            mode = "pool"
        else:
            mode = "unison"
        group = {"size": size, "scale": scale, "members": [], "proposed_voices": proposed_voices(size, scale),
                 "proposed_mode": mode, "basis": "analysis", "evidence": evidence[:3]}
        character["group"] = group
        changes.append({"character_id": character["id"], "name": character["name"], "group": deepcopy(group)})
        _line_cues(rows, tags, texts, group, own, changes)
    return candidate, changes


def _own_words(character):
    return {word.casefold().removesuffix("s") for name in [character["name"], *character.get("aliases", [])]
            for word in _WORD.findall(name)}


def _effective(kind, head, own):
    """A tag naming the group's own word ("the Dryad" for "Dryad girls") is one of its members."""
    words = {word.casefold().removesuffix("s") for word in _WORD.findall(head)}
    return "member" if kind == "singular" and words & own else kind


def _line_cues(rows, tags, texts, group, own, changes):
    """Per-line overrides for one proposed group's lines, each with its tag as evidence."""
    slot, previous = 0, None
    voices = group.get("proposed_voices")
    for row in rows:
        if "ensemble" in row:
            previous = None
            continue
        kind, head, clause = tags[row["id"]][0]
        kind = _effective(kind, head, own)
        quote = _quote(texts[row["block_id"]], clause)
        if quote is None:
            continue
        override = {}
        text = f"{head} {clause}"
        if _HALF.search(head) and voices:
            override["lanes"] = max(1, voices // 2)
        elif _SEVERAL.search(head) and voices and voices > 3:
            override["lanes"] = 3
        if group["proposed_mode"] == "pool":
            if kind == "plural" and _UNISON.search(text):
                override["mode"] = "unison"
            elif kind == "member" and re.match(r"^(?:one|another|someone|somebody|a voice)", head, re.IGNORECASE):
                slot += 1
                override["slot"] = slot
            elif slot and previous is not None and (kind == "member" or _PRONOUN.match(head)):
                # "the Dryad offered", "she said" right after: the same individual.
                override["slot"] = slot
        elif kind == "member" and re.match(r"^(?:one of|another|someone|somebody)", head, re.IGNORECASE):
            override["mode"] = "pool"
            slot += 1
            override["slot"] = slot
        previous = row["id"]
        if override:
            row["ensemble"] = dict(override, basis="explicit",
                                   evidence=[{"block_id": row["block_id"], "quote": quote}])
            changes.append({"segment_id": row["id"], "ensemble": deepcopy(row["ensemble"])})


def review_groups(directory, progress=print):
    """``analyze BOOK``: propose the groups a saved book's tags show; True when the book changed.

    Published through ``analysis_revision`` scope ``group-proposals``, which
    accepts only new analysis groups and new line cues. Proposals change no
    rendering; the cast screen (``C``) is where a person casts or declines them.
    """
    import hashlib
    from pathlib import Path
    from .analysis_revision import _completed_directory, related_directories, revise_directories
    directory = Path(directory).resolve()
    document = _completed_directory(directory)[0]
    candidate, changes = propose_groups(document)
    groups = [change["name"] for change in changes if "character_id" in change]
    if not changes:
        return False
    paths = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths}
    revise_directories({directory: candidate}, directories=paths, expected_manifest_hashes=hashes,
                       scope="group-proposals", progress=progress)
    shown = ", ".join(groups[:6]) + (f" and {len(groups) - 6} more" if len(groups) > 6 else "")
    if groups:
        progress(f"The book's tags show {len(groups)} group(s) that could be heard as several voices: {shown}. "
                 "They speak in their own voice until you set them up in the cast screen (C).")
    return True
