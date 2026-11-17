"""A person's own knowledge about a character, kept beside the book (``character-notes.json``).

A listener often knows more than a book has revealed so far: that a hooded
figure is later revealed as a named villain, is a woman, and comes from a
particular land. These notes record that knowledge per voice, as a person's
choice (never written by a model):

    {"names": ["Zandramas"], "gender": "female", "origin": "Mallorea",
     "notes": "Revealed later as Zandramas; a more exotic accent than other Malloreans"}

Every field is optional. An empty record (``{}``) is an explicit "no notes"
that shadows notes inherited from an earlier book; removing this book's own
record inherits again. Notes are inherited nearest book first, like the
other per-voice choices (``book_voice_settings``).

They reach the two places that decide how a character sounds:

- accent assignment (``accent_assignment.request_assignment``): sent as
  ``known`` beside the book's own clues, and the extra names also find the
  passages that mention the character by those names. Each saved
  assignment records the notes it was made with
  (``ASSIGNMENT_NOTES_FILENAME``), so changing a character's notes makes
  that character's accent undecided again (``stale_assignments``);
- voice design: description revisions (``description_revision.revise``)
  and refreshes (``voice_prompt_refresh``) receive them, and the cast
  screen offers to revise a voice's description to agree after notes are
  saved. ``description_conflict`` flags a description that contradicts the
  notes' gender.

They are story data describing the character, never instructions to a
model; the prompts say so. The designer never receives them directly: it
receives the description, which ``voice_prompts.designer_prompt`` makes safe.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata


NOTES_FILENAME = "character-notes.json"
ASSIGNMENT_NOTES_FILENAME = "voice-accent-assignment-notes.json"
FIELDS = ("names", "gender", "origin", "notes")
MAX_NAMES = 8
MAX_NAME = 80
MAX_GENDER = 40
MAX_ORIGIN = 120
MAX_NOTES = 1500
_MARKUP = set("[]<>{}`")


def _check_text(value, label, limit, *, single_line=True):
    if not isinstance(value, str):
        raise ValueError(f"Character notes {label} must be text")
    if value != value.strip():
        raise ValueError(f"Character notes {label} must not start or end with spaces")
    if len(value) > limit:
        raise ValueError(f"Character notes {label} must be at most {limit} characters")
    if any(character in _MARKUP for character in value):
        raise ValueError(f"Character notes {label} must be plain words without brackets or markup")
    for character in value:
        if character == "\n" and not single_line:
            continue
        if unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            raise ValueError(f"Character notes {label} must not contain control characters")


def validate_notes(record):
    """Raise ``ValueError`` saying what is wrong with one voice's notes record; return it."""
    if not isinstance(record, dict) or set(record) - set(FIELDS):
        raise ValueError(f"Character notes must be a record with only {', '.join(FIELDS)}")
    names = record.get("names", [])
    if "names" in record and (not isinstance(names, list) or not names):
        raise ValueError("Character notes names must be a nonempty list (leave it out for none)")
    if len(names) > MAX_NAMES:
        raise ValueError(f"Character notes may give at most {MAX_NAMES} other names")
    for name in names:
        _check_text(name, "names", MAX_NAME)
        if not name:
            raise ValueError("Character notes names must not be empty")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Character notes names must not repeat")
    for field, limit in (("gender", MAX_GENDER), ("origin", MAX_ORIGIN), ("notes", MAX_NOTES)):
        if field in record:
            _check_text(record[field], field, limit, single_line=field != "notes")
            if not record[field]:
                raise ValueError(f"Character notes {field} must not be empty (leave it out for none)")
    return record


def _setting(field="character_notes"):
    from .book_voice_settings import VoiceSetting
    if field == "character_notes":
        return VoiceSetting(
            filename=NOTES_FILENAME, record_format="audiobook-book-character-notes", field=field,
            validate=validate_notes,
            invalid_message="Invalid character notes, or they belong to another book/series",
            unknown_voice_message="Character notes must use voice IDs from this book",
            busy_message="Character notes are being edited; retry after the current edit")
    return VoiceSetting(
        filename=ASSIGNMENT_NOTES_FILENAME, record_format="audiobook-book-voice-accent-assignment-notes",
        field="notes_sha256", validate=_validate_digest,
        invalid_message="Invalid accent assignment notes record, or it belongs to another book/series",
        unknown_voice_message="Accent assignment notes records must use voice IDs from this book",
        busy_message="Accent assignment notes are being saved; retry after the current save")


def _validate_digest(value):
    if not (value == "" or isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)):
        raise ValueError("An accent assignment notes record must be a SHA-256 hex digest or empty")


def character_notes(bundle, config, *, include_cleared=False):
    """Each voice's notes, nearest book first; ``{}`` (cleared here) is left out unless ``include_cleared``."""
    from .book_voice_settings import nearest_book_settings
    values = nearest_book_settings(_setting(), bundle, config)
    return values if include_cleared else {voice_id: value for voice_id, value in values.items() if value}


def own_character_notes(bundle, config):
    """This book's own saved notes (not inherited)."""
    from .book_context import resolve_book_config
    from .book_voice_settings import read_book_setting
    config = resolve_book_config(bundle, config)
    return read_book_setting(_setting(), bundle, config["voices"]["series_id"])


def save_character_notes(bundle, config, voice_id, record):
    """Save this book's notes for one voice; ``{}`` clears inherited notes here, ``None`` inherits again."""
    from .book_context import resolve_book_config
    from .book_voice_settings import save_book_setting
    profile = next((voice for voice in bundle["cast"]["voices"] if voice["id"] == voice_id), None)
    if profile is not None and isinstance(profile.get("derived_from"), dict):
        raise ValueError("A changed voice (a character's later state) shares its main voice's notes; "
                         "add notes to the main voice")
    if record is not None:
        validate_notes(record)
    save_book_setting(_setting(), bundle, resolve_book_config(bundle, config), voice_id, record)


def notes_digest(record):
    """The identity of a notes record: "" for none, else SHA-256 of its canonical JSON."""
    if not record:
        return ""
    text = json.dumps({key: record[key] for key in FIELDS if key in record}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Accent assignments remember the notes they were made with ----------------------


def save_assignment_notes(bundle, config, voice_ids, notes, *, replace=False):
    """Record, for each of ``voice_ids``, the notes its new accent assignment was made with."""
    from .book_context import resolve_book_config
    from .book_voice_settings import save_book_settings
    save_book_settings(_setting("bindings"), bundle, resolve_book_config(bundle, config),
                       {voice_id: notes_digest(notes.get(voice_id)) for voice_id in voice_ids}, replace=replace)


def _recorded_notes(path):
    """Whether a book's assignment notes record names any notes at all (read without loading its cast)."""
    from .book_voice_settings import _present
    from .common import read_json
    if not _present(path):
        return False
    record = read_json(path)
    values = record.get("notes_sha256") if isinstance(record, dict) else None
    # An unreadable record is checked in full (and refused) by the per-book reading below.
    return not isinstance(values, dict) or any(values.values())


def stale_assignments(bundle, config):
    """Voices whose accent assignment was made with other notes than they have now.

    Each voice's assignment comes from the nearest book that assigned it;
    the notes it was made with are that book's record (an assignment saved
    before notes were recorded was made with none).
    """
    from pathlib import Path
    from .accent_assignment import _setting as assignment_setting
    from .book_context import resolve_book_config
    from .book_voice_settings import _parent_context, _present, read_book_setting
    config = resolve_book_config(bundle, config)
    current = character_notes(bundle, config)
    if not current and not any(_recorded_notes(Path(directory) / ASSIGNMENT_NOTES_FILENAME)
                               for directory in [bundle["directory"], *config.get("_voice_parents", [])]):
        return set()  # No notes now, and none behind any assignment: every assignment matches.
    series_id = config["voices"]["series_id"]
    books = [bundle] + [_parent_context(directory) for directory in config.get("_voice_parents", [])]
    allowed = {voice["id"] for voice in bundle["cast"]["voices"]}
    stale, seen = set(), set()
    for book in books:
        assigned = read_book_setting(assignment_setting(), book, series_id)
        bindings = read_book_setting(_setting("bindings"), book, series_id) if assigned else {}
        for voice_id in assigned:
            if voice_id in seen or voice_id not in allowed:
                continue
            seen.add(voice_id)
            if bindings.get(voice_id, "") != notes_digest(current.get(voice_id)):
                stale.add(voice_id)
    return stale


# The text form a person edits ----------------------------------------------------

_LABELS = {"names": ("names", "also", "aliases", "also known as", "later names", "name"),
           "gender": ("gender", "sex"), "origin": ("origin", "from", "comes from"),
           "notes": ("notes", "note")}


def to_text(record):
    """The notes as the lines a person edits: "names: …", "gender: …", "origin: …", then free notes."""
    record = record or {}
    return "\n".join([f"names: {', '.join(record.get('names', []))}", f"gender: {record.get('gender', '')}",
                      f"origin: {record.get('origin', '')}", f"notes: {record.get('notes', '')}"])


def from_text(text):
    """A notes record from the edited lines; lines without a known label are free notes. ``{}`` when empty."""
    fields = {"names": [], "gender": [], "origin": [], "notes": []}
    for line in str(text or "").splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z ]{0,20}?)\s*:\s?(.*)$", line)
        label = match.group(1).casefold() if match else ""
        field = next((name for name, labels in _LABELS.items() if label in labels), None)
        # A line without a known label is a free note.
        fields[field or "notes"].append(match.group(2) if field else line)
    record = {}
    names = [" ".join(part.split()) for line in fields["names"] for part in re.split(r"[,;]", line)]
    names = list(dict.fromkeys(name for name in names if name))
    if names:
        record["names"] = names
    for field in ("gender", "origin"):
        value = " ".join(" ".join(fields[field]).split())
        if value:
            record[field] = value
    notes = "\n".join(line.rstrip() for line in fields["notes"]).strip()
    notes = re.sub(r"\n{3,}", "\n\n", notes)
    if notes:
        record["notes"] = notes
    return record


def summary(record, limit=None):
    """One line saying what the notes say."""
    if not record:
        return ""
    parts = []
    if record.get("names"):
        parts.append("also " + ", ".join(record["names"]))
    parts += [record[field] for field in ("gender", "origin") if record.get(field)]
    if record.get("notes"):
        parts.append(" ".join(record["notes"].split()))
    text = "; ".join(parts)
    return text if limit is None or len(text) <= limit else text[:limit - 1].rstrip() + "…"


def model_row(record):
    """The notes as one compact model input row: [names, gender, origin, notes]."""
    return [record.get("names", []), record.get("gender", ""), record.get("origin", ""),
            " ".join(record.get("notes", "").split())]


# A description that contradicts the notes --------------------------------------

_FEMALE = re.compile(r"\b(?:female|woman|women|woman's|feminine|lady|girl|she|her)\b", re.I)
_MALE = re.compile(r"\b(?:male|man|men|man's|masculine|gentleman|boy|he|his|him)\b", re.I)


def _gender(text):
    text = text or ""
    female, male = bool(_FEMALE.search(text)), bool(_MALE.search(text))
    return "female" if female and not male else "male" if male and not female else ""


def description_conflict(description, record):
    """A sentence saying how ``description`` contradicts the notes' gender, else ""."""
    wanted = _gender((record or {}).get("gender", ""))
    if not wanted:
        return ""
    found = _gender(description)
    if found and found != wanted:
        return (f"Your notes say {record['gender']}, but the voice description reads {found}: revise it "
                "before generating the voice (i asks the model to, with your notes; e edits it by hand).")
    return ""
