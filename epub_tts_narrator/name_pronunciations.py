"""User-defined pronunciations of names and invented words, and their tagged uses.

One book-local sidecar, ``name-pronunciations.json``, holds two things:

- ``definitions``: a word's confirmed reading and the parts of speech it
  applies to. The reading is CMU Arpabet ``phonemes`` (optionally in
  consecutive groups, ``M AY1 L | K IY1``, sent to the engine as separate
  phoneme tags), or a ``say_as`` spelling the engine reads instead of the
  word (``mile-key``; ``phonemes`` is then null). The parts of speech are (``parts_of_speech``: proper noun, common noun,
  adjective, verb, adverb, interjection), chosen by a person through
  ``audiobook-render pronounce``; they inherit live from continuation
  ancestors, nearest book first, like the pronunciation registry. ``drafts``
  checkpoint an unconfirmed choice while listening and never affect rendering.
- ``mentions``: this book's per-occurrence evidence. A model ``label`` is the
  part of speech of that use in context (or ``unsure``), stamped with the
  labelling version that produced it; a person's decision is ``use`` true or
  false and always wins. Each entry is bound to its block's exact text, so a
  book update relabels only the blocks whose text changed. Ancestor mentions
  never apply here.

A use gets the special reading when a person said so, or when its label is one
of the definition's parts of speech. Labels do not depend on the chosen parts,
so changing them needs no new requests. When the spelling is not in the CMU
dictionary and every part of speech applies, no other word can share it, so
every use is tagged without labels (``needs_labels``).

Inflections are derived from the confirmed base reading by general English
rules (possessive/plural S, Z or IH0 Z by the final sound; a ``say_as``
reading keeps the source word's own ending, ``mile-key's``); any other
apostrophe or hyphen continuation keeps its suffix in plain text and tags only
the word part. Rendering fails closed while any current use of an effective
word is unlabelled or unsure.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .performance_plan import phoneme_groups, validate_phonemes
from .pronunciations import WORD_PATTERN, validate_word


FILENAME = "name-pronunciations.json"
FORMAT = "epub-tts-name-pronunciations"
VERSION = 2
PARTS_OF_SPEECH = ("proper_noun", "common_noun", "adjective", "verb", "adverb", "interjection")
PART_LABELS = {"proper_noun": "proper noun", "common_noun": "common noun", "adjective": "adjective",
               "verb": "verb", "adverb": "adverb", "interjection": "interjection"}
LABELS = frozenset(PARTS_OF_SPEECH) | {"unsure"}
# Bump when the labelling prompt's meaning changes: older model labels then
# count as unlabelled and are relabelled; a person's decisions never are.
LABEL_VERSION = 1
BASES = frozenset({"model", "user"})
_PART_ALIASES = {"proper": "proper_noun", "proper_noun": "proper_noun", "name": "proper_noun",
                 "propernoun": "proper_noun", "common": "common_noun", "common_noun": "common_noun",
                 "noun": "common_noun", "improper": "common_noun", "improper_noun": "common_noun",
                 "adjective": "adjective", "adj": "adjective", "verb": "verb", "adverb": "adverb", "adv": "adverb",
                 "interjection": "interjection"}
HISTORY_SOURCES = frozenset({"model", "model_revision", "manual", "inherited", "command_line", "imported"})
MAX_HISTORY = 64
VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
_VOICELESS = frozenset({"P", "T", "K", "F", "TH"})
_SIBILANTS = frozenset({"S", "Z", "SH", "ZH", "CH", "JH"})
_APOSTROPHES = "'’"
_HYPHENS = "-‐‑"
_RESPELLING = re.compile(r"[^\W\d_]+(?:[ '’\-‐‑][^\W\d_]+)*", re.UNICODE)
_SAY_AS = re.compile(r"[^\W\d_]+(?:['’\-‐‑][^\W\d_]+)*", re.UNICODE)
MAX_SAY_AS = 80


def name_key(name):
    """One definition per spelling regardless of case: shouted names share it."""
    return validate_word(name).casefold()


def normalize_phonemes(text):
    """Accept typed input in any case/spacing; stored values stay canonical."""
    if not isinstance(text, str):
        raise ValueError("Phonemes must be text")
    return " ".join(text.upper().replace("|", " | ").split())


def validate_name_phonemes(value):
    """CMUdict-style Arpabet: every vowel stressed, at least one primary.

    ``|`` splits the reading into consecutive groups (``M AY1 L | K IY1``);
    the primary stress may be in any group.
    """
    validate_phonemes(value)
    tokens = [token for group in phoneme_groups(value) for token in group.split(" ")]
    if any(token in VOWELS for token in tokens):
        raise ValueError("Name phonemes must mark every vowel with stress 0, 1 or 2 (for example AO1)")
    if not any(token[:-1] in VOWELS and token[-1] == "1" for token in tokens):
        raise ValueError("Name phonemes need one vowel with primary stress 1")
    return value


def validate_parts(value):
    if (not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value)
            or any(part not in PARTS_OF_SPEECH for part in value)):
        raise ValueError("Parts of speech must be a non-empty list from: " + ", ".join(PARTS_OF_SPEECH))
    return [part for part in PARTS_OF_SPEECH if part in value]


def parse_parts(text):
    """Typed parts of speech ("proper noun, adjective", "all") in canonical order."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Name at least one part of speech, or all")
    pieces = [" ".join(piece.split()) for piece in re.split(r"[,;/+]|\band\b", text.casefold())]
    pieces = [piece for piece in pieces if piece]
    if any(piece in {"all", "any", "everything"} for piece in pieces):
        return list(PARTS_OF_SPEECH)
    parts = []
    for piece in pieces:
        key = piece.replace("-", " ").replace(" ", "_")
        part = _PART_ALIASES.get(key) or _PART_ALIASES.get(key.removesuffix("s"))
        if part is None:
            raise ValueError(f"Unknown part of speech {piece!r}; use proper noun, common noun, adjective, "
                             "verb, adverb, interjection or all")
        if part not in parts:
            parts.append(part)
    return validate_parts(parts)


def describe_parts(parts):
    if list(parts) == list(PARTS_OF_SPEECH):
        return "every part of speech"
    return ", ".join(PART_LABELS[part] for part in parts)


def in_dictionary(word):
    """Whether an ordinary English word can share this spelling."""
    from .pronunciation_dictionary import dictionary_readings
    return bool(dictionary_readings(word))


def default_parts(word):
    """A dictionary spelling is a name only by default; an invented one is every use."""
    return ["proper_noun"] if in_dictionary(word) else list(PARTS_OF_SPEECH)


def needs_labels(word, parts):
    return in_dictionary(word) or list(parts) != list(PARTS_OF_SPEECH)


def validate_respelling(value):
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 80 or _RESPELLING.fullmatch(value) is None:
        raise ValueError("A pronunciation respelling hint must be short plain letters, hyphens or spaces")
    return value


def normalize_say_as(text):
    """Typed say-as text: spaces between parts become hyphens (``mile key`` → ``mile-key``)."""
    if not isinstance(text, str):
        raise ValueError("Say-as text must be text")
    return "-".join(text.split())


def validate_say_as(value):
    """A spelling the engine reads in place of the word: letters, joined by hyphens or apostrophes."""
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_SAY_AS or _SAY_AS.fullmatch(value) is None:
        raise ValueError(f"Say-as text must be up to {MAX_SAY_AS} letters, with parts joined by hyphens "
                         "(for example mile-key)")
    return value


def reading_of(value):
    """The reading of a definition, draft or history step: ``{"phonemes": …}`` or ``{"say_as": …}``."""
    if value.get("say_as") is not None:
        return {"say_as": value["say_as"]}
    return {"phonemes": value["phonemes"]}


def validate_reading(reading):
    if not isinstance(reading, dict) or len(reading) != 1 or not {"phonemes", "say_as"} >= set(reading):
        raise ValueError("A reading is either phonemes or a say-as spelling")
    if "phonemes" in reading:
        return {"phonemes": validate_name_phonemes(reading["phonemes"])}
    return {"say_as": validate_say_as(reading["say_as"])}


def describe_reading(reading):
    return f"said as {reading['say_as']}" if "say_as" in reading else reading["phonemes"]


def word_pronunciation(name, reading, word):
    """The engine pronunciation for one tagged source word (base or inflected).

    Phonemes derive the possessive/plural ending by sound; a say-as spelling
    keeps the word's own ending (``Milekey's`` → ``mile-key's``) and is
    capitalized throughout for a word written in capitals.
    """
    # Any hyphen character the book uses in a hyphenated name is the same name.
    lower = word.casefold().translate(_HYPHEN_FOLD)
    if "phonemes" in reading:
        forms = {spelling.translate(_HYPHEN_FOLD): value
                 for spelling, value in form_phonemes(name, reading["phonemes"]).items()}
        return {"alphabet": "cmu_arpabet", "value": forms[lower][1] if lower in forms else reading["phonemes"]}
    value = validate_say_as(reading["say_as"])
    spellings = {spelling.translate(_HYPHEN_FOLD): form for spelling, form in form_spellings(name).items()}
    if spellings.get(lower, "base") != "base":
        value += word[len(name):]
    if len(word) > 1 and word.isupper():
        value = value.upper()
    return {"alphabet": "respelling", "value": validate_say_as(value)}


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


_HYPHEN_FOLD = str.maketrans({hyphen: "-" for hyphen in _HYPHENS})


def occurrences(text, name):
    """Every span in ``text`` that could be the name, without judging meaning.

    Case-insensitive; matches the whole name, its possessive or plural, or the
    name part of an apostrophe/hyphen continuation (``Salvatore'll``,
    ``Salvatore-like``). Longer words that merely start with it never match.
    A hyphenated name (``Cho-Hag``) matches as a whole: the same number of
    consecutive hyphen-joined parts, whichever hyphen character the book uses,
    never one of its parts alone.
    """
    key = name_key(name).translate(_HYPHEN_FOLD)
    forms = {spelling.translate(_HYPHEN_FOLD): form for spelling, form in form_spellings(name).items()}
    width = key.count("-") + 1
    result = []
    for token in WORD_PATTERN.finditer(text):
        parts = list(_parts(token.group(), token.start()))
        for index in range(len(parts) - width + 1):
            start = parts[index][0]
            last_start, last = parts[index + width - 1]
            part = text[start:last_start + len(last)]
            lower = part.casefold().translate(_HYPHEN_FOLD)
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
    fields = {"name", "phonemes", "respelling", "parts_of_speech", "history"}
    # ``say_as`` is present only for a say-as reading, so phoneme records keep
    # their exact bytes and older readers refuse say-as records.
    if not isinstance(value, dict) or set(value) not in (fields, fields | {"say_as"}):
        raise ValueError("Name pronunciation definition has missing or unsupported fields")
    if name_key(value["name"]) != key:
        raise ValueError("Name pronunciation definition identity is invalid")
    if "say_as" in value:
        if value["phonemes"] is not None or value["respelling"] is not None:
            raise ValueError("A say-as pronunciation has no phonemes or respelling hint")
        validate_say_as(value["say_as"])
    else:
        validate_name_phonemes(value["phonemes"])
    validate_respelling(value["respelling"])
    if validate_parts(value["parts_of_speech"]) != value["parts_of_speech"]:
        raise ValueError("Name pronunciation parts of speech must be in canonical order")
    history = value["history"]
    if not isinstance(history, list) or not 1 <= len(history) <= MAX_HISTORY:
        raise ValueError("Name pronunciation history must be a bounded list")
    for item in history:
        if (not isinstance(item, dict) or "source" not in item or len({"phonemes", "say_as"} & set(item)) != 1
                or set(item) - {"phonemes", "say_as", "source", "instruction", "respelling"}
                or item["source"] not in HISTORY_SOURCES):
            raise ValueError("Name pronunciation history entry is invalid")
        validate_reading(reading_of({"phonemes": None, **item}))
        validate_respelling(item.get("respelling"))
        if "instruction" in item and not _plain(item["instruction"], 1000):
            raise ValueError("Name pronunciation instruction must be bounded plain text")


_SPAN_FIELDS = ("block_id", "start", "end", "word", "block_sha256")


def _validate_mention(value):
    """A model label (part of speech + labelling version) or a person's use decision."""
    if not isinstance(value, dict) or value.get("basis") not in BASES:
        raise ValueError("Name pronunciation mention is invalid")
    labelled = {"label", "labeler"} <= set(value)
    fields = {*_SPAN_FIELDS, "basis", *(("label", "labeler") if labelled else ())}
    if value["basis"] == "user":
        fields.add("use")
    if (set(value) != fields or not isinstance(value["block_id"], str)
            or type(value["start"]) is not int or type(value["end"]) is not int
            or not 0 <= value["start"] < value["end"] or not isinstance(value["word"], str)
            or len(value["word"]) != value["end"] - value["start"]
            or not isinstance(value["block_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["block_sha256"]) is None
            or value["basis"] == "model" and not labelled
            or value["basis"] == "user" and type(value["use"]) is not bool
            or labelled and (value["label"] not in LABELS or type(value["labeler"]) is not int or value["labeler"] < 1)):
        raise ValueError("Name pronunciation mention is invalid")


def _upgrade(record):
    """Read a version 1 record: keep a person's name/other decisions as use
    decisions; drop model name/other verdicts, which the labelling replaces."""
    for section in ("definitions", "drafts"):
        for value in record[section].values():
            if isinstance(value, dict) and "parts_of_speech" not in value and isinstance(value.get("name"), str):
                value["parts_of_speech"] = default_parts(value["name"])
    mentions = {}
    for key, values in record["mentions"].items():
        if not isinstance(values, list):
            raise ValueError("Name pronunciation mentions must be a bounded list")
        kept = []
        for value in values:
            if not isinstance(value, dict) or value.get("decision") not in {"name", "other", "unsure"}:
                raise ValueError("Name pronunciation mention is invalid")
            if value.get("basis") == "user" and value["decision"] in {"name", "other"}:
                kept.append({**{field: value.get(field) for field in _SPAN_FIELDS}, "basis": "user",
                             "use": value["decision"] == "name"})
        mentions[key] = kept
    return dict(record, version=VERSION, mentions=mentions)


def load_local(directory):
    """This book's record, or an empty one; never an ancestor's mentions."""
    directory = Path(directory)
    path = directory / FILENAME
    if not path.exists() and not path.is_symlink():
        return {"format": FORMAT, "version": VERSION, "book_id": _book_id(directory), "definitions": {},
                "drafts": {}, "mentions": {}}
    record = _regular_json(path)
    if (not isinstance(record, dict) or set(record) != {"format", "version", "book_id", "definitions", "drafts", "mentions"}
            or record["format"] != FORMAT or type(record["version"]) is not int or record["version"] not in {1, VERSION}
            or record["book_id"] != _book_id(directory) or not isinstance(record["definitions"], dict)
            or not isinstance(record["drafts"], dict) or not isinstance(record["mentions"], dict)):
        raise ValueError("Name pronunciation record is invalid or belongs to a different book")
    if record["version"] == 1:
        record = _upgrade(record)
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


def save_definition(directory, name, phonemes, *, status, source, respelling=None, instruction=None,
                    parts_of_speech=None, say_as=None):
    """Checkpoint a draft or confirm a choice; history keeps every listened step.

    Give ``phonemes`` or ``say_as`` (with ``phonemes`` None), not both.
    ``parts_of_speech`` None keeps the draft's or confirmed choice's parts, or
    the default for the spelling (``default_parts``).
    """
    key = name_key(name)
    if (phonemes is None) == (say_as is None):
        raise ValueError("A pronunciation is either phonemes or a say-as spelling")
    reading = validate_reading({"say_as": say_as} if say_as is not None else {"phonemes": phonemes})
    validate_respelling(respelling)
    if say_as is not None and respelling is not None:
        raise ValueError("A say-as pronunciation has no separate respelling hint")
    if parts_of_speech is not None:
        parts_of_speech = validate_parts(parts_of_speech)
    if status not in {"draft", "confirmed"} or source not in HISTORY_SOURCES:
        raise ValueError("Invalid name pronunciation status or source")
    step = {**reading, "source": source}
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
        parts = parts_of_speech or (previous["parts_of_speech"] if previous else default_parts(name))
        value = {"name": name, "phonemes": phonemes, "respelling": respelling, "parts_of_speech": parts,
                 "history": history[-MAX_HISTORY:], **({"say_as": say_as} if say_as is not None else {})}
        if status == "draft":
            record["drafts"][key] = value
        else:
            record["definitions"][key] = value
            record["drafts"].pop(key, None)
        return deepcopy(value)
    return _mutate(directory, update)


def import_definitions(directory, entries, *, replace=False):
    """Confirm imported choices here in one write; returns {outcome: [words]}.

    ``entries`` are ``{"name", "reading", "respelling", "parts"}`` (already
    validated, e.g. by ``name_pronunciation_yaml``). A word whose effective
    choice here is already the same is ``unchanged``. A different choice this
    book made itself is kept as a ``conflict`` unless ``replace``; one inherited
    from an earlier book is overridden here (``overrode_inherited``), as a
    choice made in this book would be. Each imported step joins the history.
    Uses are tagged later, as for any confirmed word (render or analyze).
    """
    keys = [name_key(entry["name"]) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("The import names one word more than once")
    effective = effective_definitions(directory)
    outcome = {"added": [], "overrode_inherited": [], "replaced": [], "unchanged": [], "conflict": []}

    def same(value, entry):
        return (reading_of(value) == entry["reading"] and value["parts_of_speech"] == entry["parts"]
                and value.get("respelling") == entry["respelling"])

    def update(record):
        for key, entry in zip(keys, entries):
            local, seen = record["definitions"].get(key), effective.get(key)
            if seen is not None and same(seen, entry):
                outcome["unchanged"].append(entry["name"])
                continue
            if local is not None and not replace:
                outcome["conflict"].append(entry["name"])
                continue
            kind = "replaced" if local is not None else "overrode_inherited" if seen is not None else "added"
            previous = local or record["drafts"].get(key)
            step = {**entry["reading"], "source": "imported",
                    **({"respelling": entry["respelling"]} if entry["respelling"] is not None else {})}
            history = [*(previous["history"] if previous else []), step][-MAX_HISTORY:]
            say_as = entry["reading"].get("say_as")
            record["definitions"][key] = {"name": entry["name"], "phonemes": entry["reading"].get("phonemes"),
                                          "respelling": entry["respelling"], "parts_of_speech": list(entry["parts"]),
                                          "history": history, **({"say_as": say_as} if say_as is not None else {})}
            _validate_definition(key, record["definitions"][key])
            record["drafts"].pop(key, None)
            outcome[kind].append(entry["name"])
        return outcome
    return _mutate(directory, update)


def discard_draft(directory, name):
    key = name_key(name)
    def update(record):
        record["drafts"].pop(key, None)
    if (Path(directory) / FILENAME).exists():
        _mutate(directory, update)


def remove_definition(directory, name):
    """Remove this book's own confirmed choice and draft for a name.

    Returns the removed confirmed definition (or None when only a draft was
    removed). This book's labels and decisions stay: they record what each use
    is, not how it sounds, and apply again if the word is defined in an
    ancestor. Books continued from this one stop inheriting the removed choice.
    """
    key = name_key(name)
    if not (Path(directory) / FILENAME).exists() or (
            key not in load_local(directory)["definitions"] and key not in load_local(directory)["drafts"]):
        inherited = effective_definitions(directory).get(key)
        where = (f"; it is inherited from {inherited['origin_directory']}, so remove it there"
                 if inherited is not None else "")
        raise ValueError(f"{name} has no pronunciation defined in this book{where}")
    def update(record):
        record["drafts"].pop(key, None)
        return record["definitions"].pop(key, None)
    return _mutate(directory, update)


def local_draft(directory, name):
    value = load_local(directory)["drafts"].get(name_key(name))
    return deepcopy(value) if value else None


def local_definition(directory, name):
    value = load_local(directory)["definitions"].get(name_key(name))
    return deepcopy(value) if value else None


def parts_for(directory, name):
    """The parts of speech that apply here: the effective choice, else the default."""
    definition = effective_definitions(directory).get(name_key(name))
    return definition["parts_of_speech"] if definition is not None else default_parts(name)


def tagging_state(directory, source, name, parts=None):
    """Current uses split into decided, unlabelled (pending) and unsure.

    A decided use carries ``use`` (whether it gets the special reading),
    ``basis`` (``user``, ``model`` or ``rule``) and its ``label`` when known.
    Labels from an older labelling version, or for text that changed, are
    ignored, so those uses are pending again.
    """
    parts = validate_parts(parts) if parts is not None else parts_for(directory, name)
    labels_needed = needs_labels(name, parts)
    saved = {mention_key(item): item for item in load_local(directory)["mentions"].get(name_key(name), [])}
    current = book_occurrences(source, name)
    decided, pending, unsure, counts = [], [], [], {}
    for item in current:
        mention = saved.get(mention_key(item))
        label = (mention["label"] if mention is not None and mention.get("labeler") == LABEL_VERSION else None)
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
        if mention is not None and mention["basis"] == "user":
            decided.append(dict(item, use=mention["use"], basis="user", label=label))
        elif not labels_needed:
            decided.append(dict(item, use=True, basis="rule", label=label))
        elif label is None:
            pending.append(item)
        elif label == "unsure":
            unsure.append(dict(item, use=None, basis="model", label="unsure"))
        else:
            decided.append(dict(item, use=label in parts, basis="model", label=label))
    return {"occurrences": current, "decided": decided, "pending": pending, "unsure": unsure,
            "ready": not pending and not unsure, "parts_of_speech": parts, "labels_needed": labels_needed,
            "label_counts": counts}


def save_decisions(directory, source, name, decisions):
    """Merge model labels or a person's decisions for current uses.

    Entries whose text changed are dropped. A person's decision keeps the
    model label it overrides (for display); a model label never replaces a
    person's decision.
    """
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
            old = merged.get(mention_key(value))
            if value["basis"] == "model" and old is not None and old["basis"] == "user":
                value = dict(old, label=value["label"], labeler=value["labeler"])
            elif value["basis"] == "user" and "label" not in value and old is not None and "label" in old:
                value = dict(value, label=old["label"], labeler=old["labeler"])
            merged[mention_key(value)] = value
        record["mentions"][key] = sorted(merged.values(), key=lambda item: (item["block_id"], item["start"]))
    _mutate(directory, update)


def pending_tagging(directory, source):
    """Effective names whose current uses are not all decided in this book."""
    result = []
    for _, definition in sorted(effective_definitions(directory).items()):
        state = tagging_state(directory, source, definition["name"], definition["parts_of_speech"])
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
        state = tagging_state(directory, source, definition["name"], definition["parts_of_speech"])
        if not state["ready"]:
            origin = " (inherited)" if definition["inherited"] else ""
            blocked.append(f"{definition['name']}{origin}: {len(state['pending'])} untagged, {len(state['unsure'])} unsure")
            continue
        reading = reading_of(definition)
        for item in state["decided"]:
            if not item["use"]:
                continue
            pronunciation = word_pronunciation(definition["name"], reading, item["word"])
            field = "phonemes" if pronunciation["alphabet"] == "cmu_arpabet" else "say_as"
            spans.setdefault(item["block_id"], []).append(
                {"start": item["start"], "end": item["end"], "word": item["word"], field: pronunciation["value"],
                 "name": key})
    if blocked:
        raise ValueError("Name pronunciations need this book's name uses tagged before rendering ("
                         + "; ".join(blocked) + "). Run: audiobook-render pronounce BOOK "
                         "(or just pronounce BOOK) to tag them; nothing was rendered")
    for block_id, items in spans.items():
        items.sort(key=lambda item: item["start"])
        if any(a["end"] > b["start"] for a, b in zip(items, items[1:])):
            raise ValueError(f"Two defined pronunciations claim overlapping words in block {block_id}")
    return {"spans": spans}
