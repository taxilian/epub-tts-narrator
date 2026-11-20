"""Guards for plural speakers (``COLLECTIVE_VOICES.md`` §6.1): flags only, never fixes.

A line's speaker tag says who speaks: "the twins said in unison", "one of the
others observed", "the voice promised". This module reads only the tag (a
clause with a speech verb touching the line, as ``speech_cues`` finds it) and
its subject, never a character's name alone, and flags three mistakes:

* ``plural_tag``: the tag's subject is several people ("the Murgos replied in
  unison", "they all answered", "several of them") but the line went to one
  ordinary character, not a group or a ``simultaneous_speakers`` row;
* ``singular_tag``: a group's line whose tag names another character ("the
  voice in his mind said" on a group's line);
* ``invented_member``: the tag says only "one of them" / "the other" but the
  line went to a specific named character (twins whose halves the text never
  tells apart).

A flagged line is reported (an issue, ``verify-attribution``'s risk set) for a
person or a jury to look at again; nothing here changes attribution.
"""

from __future__ import annotations

import re

from .quote_spans import SPEECH_VERBS
from .speech_cues import SPOKEN, _clauses, _LEAD

# Tag verbs groups are often given beyond the shared list (kept local: that list is analysis behaviour).
_GROUP_VERBS = ("roared|roars|bellowed|bellows|boomed|booms|promised|promises|chorused|choruses|chanted|chants|"
                "intoned|intones|echoed|echoes|squealed|squeals|screamed|screams|completed|completes|chimed in")
_VERB = re.compile(rf"\b(?:{SPEECH_VERBS}|{_GROUP_VERBS})\b", re.IGNORECASE)
_INVERTED = re.compile(rf"^(?:{SPEECH_VERBS}|{_GROUP_VERBS})\s+", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_JOINT = re.compile(r"\b(?:in unison|as one|in chorus|together|all at once|simultaneously)\b", re.IGNORECASE)
_PLURAL_START = re.compile(
    r"^(?:they|we|those|these|both|all|everyone|everybody|several|many|some of|a few|half of|the others|the rest|"
    r"the (?:two|three|four|five|six|seven|eight|nine|ten|dozen|pair|twins?|whole|assembled|gathered))(?=[\s,.;:!?]|$)",
    re.IGNORECASE)
_MEMBER_START = re.compile(r"^(?:one of|another|someone|somebody|a voice|one\b|the (?:first|second|third|other)\b)",
                           re.IGNORECASE)
_SINGULAR_PRONOUN = re.compile(r"^(?:he|she|i)\b", re.IGNORECASE)
COLLECTIVE_NOUNS = frozenset("""crowd throng chorus choir council assembly mob congregation company group gathering
audience herd flock pack legion legions men women people children folk crew guards soldiers band troop troupe
horde squad retinue entourage""".split())
_ARTICLES = frozenset({"the", "a", "an"})


MAX_SUBJECT_WORDS = 6
_AFTER_VERB_JOINT = re.compile(r"^\s*(?:[\w'’-]+\s+){0,2}?(?:in unison|as one|in chorus|together|all at once|"
                               r"simultaneously|at the same time)\b", re.IGNORECASE)
_NAME_PAIR = re.compile(r"^(?:both\s+)?(?:the\s+)?[\w'’-]+(?:\s+[\w'’-]+)?\s+and\s+(?:the\s+)?[\w'’-]+(?:\s+[\w'’-]+)?$",
                        re.IGNORECASE)
_BARE_MEMBER = re.compile(r"^(?:one of (?:them|the two|the pair)|the other|the second|the first)$", re.IGNORECASE)


_TAG_NEXT = frozenset("""him her them us me you back again together in as at with to then quickly almost finally
once aloud loudly softly quietly""".split())
_ONGOING = frozenset({"began", "begins", "continued", "continues", "finished", "finishes",
                      "agreed", "agrees", "offered", "offers"})


def _tag_ending(verb, after):
    """What follows a speech verb in a tag: nothing, punctuation, an adverb, a manner or an addressee."""
    rest = after.strip()
    if not rest or rest[0] in ",.;:!?—–-\"“”'’":
        return True
    first = _WORD.findall(rest)[:1]
    if not first:
        return True
    word = first[0].casefold()
    if verb.casefold() in _ONGOING and (word == "to" or word.endswith("ing")):
        return False
    return word.endswith("ly") or word in _TAG_NEXT


def tag_subject(clause):
    """``(subject, after_verb)`` of a speech tag clause, or None when it is not a plain tag.

    The subject is the short noun phrase before the speech verb ("the twins",
    "one of the others", "Mara and Lena") or after it in an inverted tag
    ("said the twins"). A clause whose verb follows other actions ("Kal Torak
    arose and declared") or a long description is no plain tag and gives None.
    """
    text = _LEAD.sub("", clause.strip()).strip(" ,;:—–-\"“”")
    verb = _VERB.search(text)
    if verb is None:
        return None
    inverted = _INVERTED.match(text)
    if inverted:
        head = re.split(r"[,.;!?]|\b(?:in|as|with|to|and then)\b", text[inverted.end():], maxsplit=1)[0]
        after = text[inverted.end() + len(head):]
    else:
        head, after = text[:verb.start()], text[verb.end():]
        head = head.split(",")[0] if _MEMBER_START.match(head) or _PLURAL_START.match(head) else head
        if "," in head:
            return None
    if not inverted and not _tag_ending(text[verb.start():verb.end()], after):
        return None  # "They began pulling him": the verb is an action here, not a speech tag
    head = " ".join(head.split()).strip(" ,")
    words = _WORD.findall(head)
    if not words or len(words) > MAX_SUBJECT_WORDS:
        return None
    if " and " in f" {head} " and not _NAME_PAIR.match(head):
        return None  # "looked at him and laughed": actions, not two speakers
    return head, after


def subject_kind(clause):
    """``plural``, ``member`` (one unnamed member), ``singular``, or None (not a tag / unknown)."""
    found = tag_subject(clause)
    if found is None:
        return None
    head, after = found
    if _MEMBER_START.match(head):
        return "member"
    words = [word.casefold() for word in _WORD.findall(head)]
    if not _PLURAL_START.match(head):
        # "the Father of the Gods", "the man with white eyes": the head noun comes first.
        words = [word.casefold() for word in _WORD.findall(
            re.split(r"\b(?:of|with|from|in|on|at|who|that)\b", head, flags=re.IGNORECASE)[0])]
    if (_PLURAL_START.match(head) or _NAME_PAIR.match(head) or _AFTER_VERB_JOINT.match(after)
            or (words and words[0] in _ARTICLES and (words[-1] in COLLECTIVE_NOUNS or _plural_noun(words[-1])))):
        return "plural"
    return "singular" if words else None


def _plural_noun(word):
    return (len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is", "'s", "’s"))
            and word not in {"boss", "princess", "witness", "mistress", "actress"})


def _names(character):
    return [name for name in [character.get("name", ""), *character.get("aliases", [])] if name]


def _content(words):
    return [word for word in words if word not in _ARTICLES]


def group_warnings(document):
    """``[{segment_id, block_id, kind, clause, reason}]`` for lines whose tag contradicts their speaker."""
    cast = document["cast"]
    characters = {character["id"]: character for character in cast["characters"]}
    roles = {role["id"]: role for role in cast["roles"]}
    name_words = {cid: [set(_content([word.casefold() for word in _WORD.findall(name)])) for name in _names(row)]
                  for cid, row in characters.items()}
    by_block = {}
    for row in document["segments"]:
        by_block.setdefault(row["block_id"], []).append(row)
    texts = {block["id"]: block["text"] for block in document["source"]["blocks"]}
    found = []
    for block_id, rows in by_block.items():
        spoken = [row for row in rows if row.get("kind") in SPOKEN]
        if not spoken:
            continue
        clauses = _clauses(texts[block_id], rows)
        for row in spoken:
            if row.get("simultaneous_speakers") or not row.get("role_id"):
                continue
            character = characters.get(roles.get(row["role_id"], {}).get("character_id"))
            if character is None:
                continue
            for clause, _ in clauses.get(id(row), []):
                kind = subject_kind(clause)
                if kind is None:
                    continue
                head = tag_subject(clause)[0]
                head_words = set(_content([word.casefold() for word in _WORD.findall(head)]))
                group = isinstance(character.get("group"), dict)
                warning = None
                if kind == "plural" and not group:
                    warning = ("plural_tag", f"The tag says several people speak, but the line went to "
                                             f"{character['name']} alone.")
                elif kind == "singular" and group and head_words:
                    own = set().union(*name_words[character["id"]]) if name_words[character["id"]] else set()
                    members = set((character["group"] or {}).get("members", []))
                    other = next((characters[cid]["name"] for cid, names in name_words.items()
                                  if cid != character["id"] and cid not in members
                                  and any(name_set and head_words <= name_set for name_set in names)
                                  and not head_words <= own), None)
                    if other is not None:
                        warning = ("singular_tag", f"The tag names {other}, but the line went to the group "
                                                   f"{character['name']}.")
                elif kind == "member" and not group and _BARE_MEMBER.match(head):
                    # Only a personal name (every word capitalized) is an individual the text did not name here.
                    personal = all(word[:1].isupper() for word in _WORD.findall(character.get("name", "")))
                    if personal and character.get("name"):
                        warning = ("invented_member", f"The tag says only “{head}”, but the line went to "
                                                      f"{character['name']} by name.")
                if warning is not None:
                    found.append({"segment_id": row["id"], "block_id": block_id, "kind": warning[0],
                                  "clause": clause[:160], "reason": warning[1]})
                    break
    return found
