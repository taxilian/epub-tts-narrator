"""User-defined character-name pronunciations and their tagged uses.

One book-local sidecar, ``name-pronunciations.json``, holds two things:

- ``definitions``: a name's confirmed CMU Arpabet reading, chosen by a person
  through ``audiobook-render pronounce-name``; they inherit live from
  continuation ancestors, nearest book first, like the pronunciation registry.
  ``drafts`` checkpoint an unconfirmed choice while listening and never affect
  rendering.
- ``mentions``: this book's per-occurrence decisions saying whether a source
  word really is that name (``name``) or a homonym/ordinary word (``other``).
  Each decision is bound to its block's exact text, so a book update retags
  only the blocks whose text changed. Ancestor mentions never apply here.

Inflections are derived from the confirmed base reading by general English
rules (possessive/plural S, Z or IH0 Z by the final sound); any other
apostrophe or hyphen continuation keeps its suffix in plain text and tags only
the name part. Rendering uses tagged ``name`` spans only and fails closed while
any current occurrence of an effective name is undecided.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .performance_plan import validate_phonemes
from .pronunciations import WORD_PATTERN, validate_word


FILENAME = "name-pronunciations.json"
FORMAT = "epub-tts-name-pronunciations"
VERSION = 1
DECISIONS = frozenset({"name", "other", "unsure"})
BASES = frozenset({"model", "user"})
HISTORY_SOURCES = frozenset({"model", "model_revision", "manual", "inherited", "command_line"})
MAX_HISTORY = 64
VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_VOICELESS = frozenset({"P", "T", "K", "F", "TH"})
_SIBILANTS = frozenset({"S", "Z", "SH", "ZH", "CH", "JH"})
_APOSTROPHES = "'’"
_HYPHENS = "-‐‑"
_RESPELLING = re.compile(r"[^\W\d_]+(?:[ '’\-‐‑][^\W\d_]+)*", re.UNICODE)


def name_key(name):
    """One definition per spelling regardless of case: shouted names share it."""
    return validate_word(name).casefold()


def normalize_phonemes(text):
    """Accept typed input in any case/spacing; stored values stay canonical."""
    if not isinstance(text, str):
        raise ValueError("Phonemes must be text")
    return " ".join(text.upper().split())


def validate_name_phonemes(value):
    """CMUdict-style Arpabet: every vowel stressed, at least one primary."""
    validate_phonemes(value)
    tokens = value.split(" ")
    if any(token in VOWELS for token in tokens):
        raise ValueError("Name phonemes must mark every vowel with stress 0, 1 or 2 (for example AO1)")
    if not any(token[:-1] in VOWELS and token[-1] == "1" for token in tokens):
        raise ValueError("Name phonemes need one vowel with primary stress 1")
    return value


def validate_respelling(value):
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 80 or _RESPELLING.fullmatch(value) is None:
        raise ValueError("A pronunciation respelling hint must be short plain letters, hyphens or spaces")
    return value


def _suffix(phonemes):
    last = phonemes.split(" ")[-1].rstrip("012")
    if last in _SIBILANTS:
        return "IH0 Z"
    return "S" if last in _VOICELESS else "Z"


def plural_spelling(name):
    lower = name.casefold()
    return name + ("es" if lower.endswith(("s", "x", "z", "ch", "sh")) else "s")


def form_spellings(name):
    """Casefolded whole-word inflections that take a derived reading."""
    validate_word(name)
    forms = {name.casefold(): "base", plural_spelling(name).casefold(): "plural"}
    for apostrophe in _APOSTROPHES:
        forms[(name + apostrophe + "s").casefold()] = "possessive"
    return forms


def form_phonemes(name, phonemes):
    """Map each whole-word inflection (casefolded) to its derived reading."""
    validate_name_phonemes(phonemes)
    suffixed = phonemes + " " + _suffix(phonemes)
    return {spelling: (form, phonemes if form == "base" else suffixed)
            for spelling, form in form_spellings(name).items()}


def _parts(token, offset):
    """Split one source word at hyphens, keeping exact offsets."""
    start = 0
    for index, character in enumerate(token + "-"):
        if index == len(token) or character in _HYPHENS:
            if index > start:
                yield offset + start, token[start:index]
            start = index + 1


def occurrences(text, name):
    """Every span in ``text`` that could be the name, without judging meaning.

    Case-insensitive; matches the whole name, its possessive or plural, or the
    name part of an apostrophe/hyphen continuation (``Salvatore'll``,
    ``Salvatore-like``). Longer words that merely start with it never match.
    """
    key = name_key(name)
    forms = form_spellings(name)
    result = []
    for token in WORD_PATTERN.finditer(text):
        for start, part in _parts(token.group(), token.start()):
            lower = part.casefold()
            if lower in forms:
                result.append({"start": start, "end": start + len(part), "word": part, "form": forms[lower]})
            elif lower.startswith(key) and len(lower) > len(key) and lower[len(key)] in _APOSTROPHES:
                result.append({"start": start, "end": start + len(key), "word": part[:len(key)], "form": "base"})
    return result


def block_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def book_occurrences(source, name):
    found = []
    for block in source["blocks"]:
        digest = None
        for item in occurrences(block["text"], name):
            digest = digest or block_sha256(block["text"])
            found.append(dict(item, block_id=block["id"], block_sha256=digest))
    return found


def mention_key(item):
    return (item["block_id"], item["start"], item["end"], item["word"], item["block_sha256"])


def _plain(value, maximum):
    return (isinstance(value, str) and 0 < len(value) <= maximum and value == value.strip()
            and not any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value))


def _regular_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Name pronunciation metadata must be stored in a regular file")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Name pronunciation metadata exceeds its size limit")
    def reject(value):
        raise ValueError("Name pronunciation metadata contains a non-finite value")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _book_id(directory):
    source = _regular_json(Path(directory) / "source.json")
    if not isinstance(source, dict) or not isinstance(source.get("book"), dict) or not isinstance(source["book"].get("id"), str):
        raise ValueError("Name pronunciations require this book's source")
    return source["book"]["id"]


def _validate_definition(key, value):
    fields = {"name", "phonemes", "respelling", "history"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Name pronunciation definition has missing or unsupported fields")
    if name_key(value["name"]) != key:
        raise ValueError("Name pronunciation definition identity is invalid")
    validate_name_phonemes(value["phonemes"])
    validate_respelling(value["respelling"])
    history = value["history"]
    if not isinstance(history, list) or not 1 <= len(history) <= MAX_HISTORY:
        raise ValueError("Name pronunciation history must be a bounded list")
    for item in history:
        if (not isinstance(item, dict) or not {"phonemes", "source"} <= set(item)
                or set(item) - {"phonemes", "source", "instruction", "respelling"}
                or item["source"] not in HISTORY_SOURCES):
            raise ValueError("Name pronunciation history entry is invalid")
        validate_name_phonemes(item["phonemes"])
        validate_respelling(item.get("respelling"))
        if "instruction" in item and not _plain(item["instruction"], 1000):
            raise ValueError("Name pronunciation instruction must be bounded plain text")


def _validate_mention(value):
    fields = {"block_id", "start", "end", "word", "block_sha256", "decision", "basis"}
    if (not isinstance(value, dict) or set(value) != fields or not isinstance(value["block_id"], str)
            or type(value["start"]) is not int or type(value["end"]) is not int
            or not 0 <= value["start"] < value["end"] or not isinstance(value["word"], str)
            or len(value["word"]) != value["end"] - value["start"]
            or not isinstance(value["block_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["block_sha256"]) is None
            or value["decision"] not in DECISIONS or value["basis"] not in BASES
            or value["decision"] == "unsure" and value["basis"] != "model"):
        raise ValueError("Name pronunciation mention is invalid")


def load_local(directory):
    """This book's record, or an empty one; never an ancestor's mentions."""
    directory = Path(directory)
    path = directory / FILENAME
    if not path.exists() and not path.is_symlink():
        return {"format": FORMAT, "version": VERSION, "book_id": _book_id(directory), "definitions": {},
                "drafts": {}, "mentions": {}}
    record = _regular_json(path)
    if (not isinstance(record, dict) or set(record) != {"format", "version", "book_id", "definitions", "drafts", "mentions"}
            or record["format"] != FORMAT or type(record["version"]) is not int or record["version"] != VERSION
            or record["book_id"] != _book_id(directory) or not isinstance(record["definitions"], dict)
            or not isinstance(record["drafts"], dict) or not isinstance(record["mentions"], dict)):
        raise ValueError("Name pronunciation record is invalid or belongs to a different book")
    for section in ("definitions", "drafts"):
        for key, value in record[section].items():
            _validate_definition(key, value)
    for key, values in record["mentions"].items():
        name_key(key)
        if not isinstance(values, list) or len(values) > 1_000_000:
            raise ValueError("Name pronunciation mentions must be a bounded list")
        seen = set()
        for value in values:
            _validate_mention(value)
            if mention_key(value) in seen:
                raise ValueError("Name pronunciation mentions repeat one occurrence")
            seen.add(mention_key(value))
    return record


def _owners(directory):
    directory = Path(directory).resolve()
    manifest = directory / "manifest.json"
    if manifest.exists() and isinstance(_regular_json(manifest).get("continuation"), dict):
        from .analysis_directory import continuation_directories
        return [directory, *continuation_directories(directory)]
    return [directory]


def effective_definitions(directory):
    """Confirmed definitions visible here, nearest book first."""
    result = {}
    owners = _owners(directory)
    for owner in owners:
        if not (owner / FILENAME).exists() and not (owner / FILENAME).is_symlink():
            continue
        for key, value in load_local(owner)["definitions"].items():
            if key not in result:
                result[key] = dict(deepcopy(value), origin_directory=str(owner), inherited=owner != owners[0])
    return result


def _mutate(directory, operation):
    from .pipeline import write_json
    from .renderer.run_locks import acquire_directory_lock
    directory = Path(directory).resolve()
    if directory.is_symlink():
        raise ValueError("Name pronunciation directory must not be a symlink")
    lock = acquire_directory_lock(directory / ".name-pronunciations.lock")
    try:
        record = load_local(directory)
        result = operation(record)
        write_json(directory / FILENAME, record, overwrite=True)
        return result
    finally:
        lock.rmdir()


def save_definition(directory, name, phonemes, *, status, source, respelling=None, instruction=None):
    """Checkpoint a draft or confirm a choice; history keeps every listened step."""
    key = name_key(name)
    validate_name_phonemes(phonemes)
    validate_respelling(respelling)
    if status not in {"draft", "confirmed"} or source not in HISTORY_SOURCES:
        raise ValueError("Invalid name pronunciation status or source")
    step = {"phonemes": phonemes, "source": source}
    if respelling is not None:
        step["respelling"] = respelling
    if instruction is not None:
        if not _plain(instruction, 1000):
            raise ValueError("Name pronunciation instruction must be bounded plain text")
        step["instruction"] = instruction
    def update(record):
        # A draft never replaces the confirmed choice that rendering uses.
        previous = record["drafts"].get(key) or record["definitions"].get(key)
        history = list(previous["history"]) if previous else []
        if not history or history[-1] != step:
            history.append(step)
        value = {"name": name, "phonemes": phonemes, "respelling": respelling, "history": history[-MAX_HISTORY:]}
        if status == "draft":
            record["drafts"][key] = value
        else:
            record["definitions"][key] = value
            record["drafts"].pop(key, None)
        return deepcopy(value)
    return _mutate(directory, update)


def discard_draft(directory, name):
    key = name_key(name)
    def update(record):
        record["drafts"].pop(key, None)
    if (Path(directory) / FILENAME).exists():
        _mutate(directory, update)


def local_draft(directory, name):
    value = load_local(directory)["drafts"].get(name_key(name))
    return deepcopy(value) if value else None


def local_definition(directory, name):
    value = load_local(directory)["definitions"].get(name_key(name))
    return deepcopy(value) if value else None


def tagging_state(directory, source, name):
    """Current occurrences split by saved decision; stale decisions are ignored."""
    saved = {mention_key(item): item for item in load_local(directory)["mentions"].get(name_key(name), [])}
    current = book_occurrences(source, name)
    decided, pending, unsure = [], [], []
    for item in current:
        decision = saved.get(mention_key(item))
        if decision is None:
            pending.append(item)
        elif decision["decision"] == "unsure":
            unsure.append(dict(item, decision="unsure", basis="model"))
        else:
            decided.append(dict(item, decision=decision["decision"], basis=decision["basis"]))
    return {"occurrences": current, "decided": decided, "pending": pending, "unsure": unsure,
            "ready": not pending and not unsure}


def save_decisions(directory, source, name, decisions):
    """Merge decisions for current occurrences; drop ones whose text changed."""
    key = name_key(name)
    current = {mention_key(item): item for item in book_occurrences(source, name)}
    checked = []
    for value in decisions:
        _validate_mention(value)
        if mention_key(value) not in current:
            raise ValueError("Name tag decision refers to a stale or unknown occurrence")
        checked.append(deepcopy(value))
    def update(record):
        merged = {mention_key(item): item for item in record["mentions"].get(key, []) if mention_key(item) in current}
        for value in checked:
            merged[mention_key(value)] = value
        record["mentions"][key] = sorted(merged.values(), key=lambda item: (item["block_id"], item["start"]))
    _mutate(directory, update)


def pending_tagging(directory, source):
    """Effective names whose current uses are not all decided in this book."""
    result = []
    for _, definition in sorted(effective_definitions(directory).items()):
        state = tagging_state(directory, source, definition["name"])
        if not state["ready"]:
            result.append((definition, state))
    return result


def render_snapshot(directory, source):
    """Tagged spans per block for rendering, or None when nothing is defined.

    Fails closed: an effective name with any undecided or unsure current use
    raises, because rendering it with the default reading would reintroduce
    the inconsistency the user fixed.
    """
    definitions = effective_definitions(directory)
    if not definitions:
        return None
    spans, blocked = {}, []
    for key, definition in sorted(definitions.items()):
        state = tagging_state(directory, source, definition["name"])
        if not state["ready"]:
            origin = " (inherited)" if definition["inherited"] else ""
            blocked.append(f"{definition['name']}{origin}: {len(state['pending'])} untagged, {len(state['unsure'])} unsure")
            continue
        forms = form_phonemes(definition["name"], definition["phonemes"])
        for item in state["decided"]:
            if item["decision"] != "name":
                continue
            lower = item["word"].casefold()
            phonemes = forms[lower][1] if lower in forms else definition["phonemes"]
            spans.setdefault(item["block_id"], []).append(
                {"start": item["start"], "end": item["end"], "word": item["word"], "phonemes": phonemes, "name": key})
    if blocked:
        raise ValueError("Name pronunciations need this book's name uses tagged before rendering ("
                         + "; ".join(blocked) + "). Run: audiobook-render pronounce-name BOOK "
                         "(or just pronounce-name BOOK) to tag them; nothing was rendered")
    for block_id, items in spans.items():
        items.sort(key=lambda item: item["start"])
        if any(a["end"] > b["start"] for a, b in zip(items, items[1:])):
            raise ValueError(f"Two defined name pronunciations claim overlapping words in block {block_id}")
    return {"spans": spans}
