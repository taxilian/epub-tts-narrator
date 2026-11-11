"""Confirmed pronunciation choices as a small, hand-editable YAML file.

Export writes every confirmed choice a book uses (its own and, unless asked
otherwise, those it inherits), so the choices can move to a re-analyzed book,
another series or another checkout. Import confirms them in a book in one
write (``name_pronunciations.import_definitions``); drafts and per-use
decisions never travel, because uses belong to one book's paragraphs and are
labelled again there.

The file is a strict subset of YAML, read and written here without a YAML
library (the analyzer core is stdlib-only)::

    format: epub-tts-pronunciations
    version: 1
    pronunciations:
      - word: Barak
        phonemes: "B EH1 R EH0 K"
        parts_of_speech: [proper noun]
      - word: Taur
        say_as: Tor
        parts_of_speech: [all]

Each entry has ``word``, exactly one of ``phonemes`` (CMU Arpabet with stress
digits, ``|`` between groups) or ``say_as`` (a spelling read instead), optional
``respelling`` (a hint shown with phonemes), and optional ``parts_of_speech``
(default: the spelling's usual default). Scalars may be plain, "double" or
'single' quoted; lists may be ``[a, b]`` or one ``- item`` per line; ``#``
starts a comment. Anything else is refused with its line number.
"""

import json
from pathlib import Path

from .name_pronunciations import (PART_LABELS, PARTS_OF_SPEECH, default_parts, effective_definitions,
                                  import_definitions, load_local, name_key, normalize_phonemes, normalize_say_as,
                                  parse_parts, reading_of, validate_reading, validate_respelling)


FORMAT = "epub-tts-pronunciations"
VERSION = 1
FIELDS = ("word", "phonemes", "say_as", "respelling", "parts_of_speech")
MAX_BYTES = 4 * 1024 * 1024


def _quote(value):
    return json.dumps(value, ensure_ascii=False)


def _parts_text(parts):
    if list(parts) == list(PARTS_OF_SPEECH):
        return "[all]"
    return "[" + ", ".join(PART_LABELS[part] for part in parts) + "]"


def dump(definitions, *, title=None):
    """YAML text for ``{key: definition}`` (definition records as stored)."""
    lines = []
    if title:
        lines.append(f"# Pronunciations used by {' '.join(title.split())}.")
    lines += ["# Import with: audiobook-render pronounce BOOK --import FILE",
              f"format: {FORMAT}", f"version: {VERSION}"]
    if not definitions:
        lines.append("pronunciations: []")
        return "\n".join(lines) + "\n"
    lines.append("pronunciations:")
    for key in sorted(definitions):
        value = definitions[key]
        lines.append(f"  - word: {_quote(value['name'])}")
        reading = reading_of(value)
        if "say_as" in reading:
            lines.append(f"    say_as: {_quote(reading['say_as'])}")
        else:
            lines.append(f"    phonemes: {_quote(reading['phonemes'])}")
        if value.get("respelling"):
            lines.append(f"    respelling: {_quote(value['respelling'])}")
        lines.append(f"    parts_of_speech: {_parts_text(value['parts_of_speech'])}")
    return "\n".join(lines) + "\n"


def _strip_comment(line):
    quote, escaped = None, False
    for index, char in enumerate(line):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index].rstrip()
    return line.rstrip()


def _scalar(text, where):
    text = text.strip()
    if text.startswith('"'):
        try:
            value, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            raise ValueError(f"{where}: unterminated or invalid double-quoted text") from None
        if text[end:].strip() or not isinstance(value, str):
            raise ValueError(f"{where}: unexpected text after a quoted value")
        return value
    if text.startswith("'"):
        if len(text) < 2 or not text.endswith("'"):
            raise ValueError(f"{where}: unterminated single-quoted text")
        return text[1:-1].replace("''", "'")
    if text[:1] in set("[]{}&*!|>%@`"):
        raise ValueError(f"{where}: unsupported YAML value {text[:20]!r}; quote it")
    return text


def _flow_list(text, where):
    inner = text.strip()[1:-1]
    items, current, quote = [], "", None
    for char in inner:
        if quote:
            current += char
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            current += char
        elif char == ",":
            items.append(current)
            current = ""
        else:
            current += char
    if current.strip() or items:
        items.append(current)
    return [_scalar(item, where) for item in items if item.strip()]


def parse(text):
    """The file's entries as plain dicts, or ValueError naming the line."""
    top, entries, entry, list_key = {}, [], None, None
    section = False
    for number, raw in enumerate(text.splitlines(), 1):
        where = f"line {number}"
        if "\t" in raw[:len(raw) - len(raw.lstrip())]:
            raise ValueError(f"{where}: indent with spaces, not tabs")
        line = _strip_comment(raw)
        if not line.strip() or line.strip() == "---":
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if indent == 0:
            entry, list_key = None, None
            key, separator, value = body.partition(":")
            if not separator or key not in {"format", "version", "pronunciations"} or key in top:
                raise ValueError(f"{where}: expected format, version or pronunciations")
            if key == "pronunciations":
                section = True
                if value.strip() not in {"", "[]"}:
                    raise ValueError(f"{where}: list the pronunciations on the lines below")
                top[key] = True
            else:
                top[key] = _scalar(value, where)
            continue
        if not section:
            raise ValueError(f"{where}: indented line outside pronunciations")
        if body.startswith("- ") or body == "-":
            item = body[1:].strip()
            if list_key is not None and indent > entry_indent:
                entry[list_key].append(_scalar(item, where))
                continue
            entry_indent, list_key = indent, None
            entry = {}
            entries.append((number, entry))
            if not item:
                continue
            body, indent = item, indent + 2
        if entry is None:
            raise ValueError(f"{where}: start each pronunciation with '- word: ...'")
        key, separator, value = body.partition(":")
        key = key.strip()
        if not separator or key not in FIELDS:
            raise ValueError(f"{where}: unknown field {key[:30]!r}; use " + ", ".join(FIELDS))
        if key in entry:
            raise ValueError(f"{where}: {key} given twice in one pronunciation")
        value = value.strip()
        list_key = None
        if key == "parts_of_speech":
            if not value:
                entry[key], list_key = [], key
            elif value.startswith("[") and value.endswith("]"):
                entry[key] = _flow_list(value, where)
            else:
                entry[key] = [_scalar(value, where)]
        else:
            entry[key] = _scalar(value, where)
    if top.get("format") != FORMAT:
        raise ValueError(f"Not a pronunciation file: expected 'format: {FORMAT}'")
    if str(top.get("version")) != str(VERSION):
        raise ValueError(f"Unsupported pronunciation file version; expected {VERSION}")
    return entries


def entries_from(text):
    """Validated import entries: {name, reading, respelling, parts}."""
    found = []
    for number, entry in parse(text):
        where = f"pronunciation at line {number}"
        try:
            name = entry["word"]
            name_key(name)
            if ("phonemes" in entry) == ("say_as" in entry):
                raise ValueError("give exactly one of phonemes or say_as")
            reading = validate_reading({"phonemes": normalize_phonemes(entry["phonemes"])} if "phonemes" in entry
                                       else {"say_as": normalize_say_as(entry["say_as"])})
            respelling = validate_respelling(entry.get("respelling") or None)
            if "say_as" in reading and respelling is not None:
                raise ValueError("a say_as reading has no separate respelling")
            parts = (parse_parts(", ".join(entry["parts_of_speech"])) if entry.get("parts_of_speech")
                     else default_parts(name))
        except KeyError:
            raise ValueError(f"{where}: missing word") from None
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        found.append({"name": name, "reading": reading, "respelling": respelling, "parts": parts})
    return found


def export_file(directory, path, *, inherited=True):
    """Write the book's confirmed choices; returns how many were written."""
    directory = Path(directory)
    definitions = (effective_definitions(directory) if inherited else load_local(directory)["definitions"])
    title = None
    source = directory / "source.json"
    if source.is_file():
        title = (json.loads(source.read_text(encoding="utf-8")).get("book") or {}).get("title")
    path = Path(path)
    if path.is_symlink() or path.is_dir():
        raise ValueError(f"Export target must be a regular file path: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(dump(definitions, title=title), encoding="utf-8")
    temporary.replace(path)
    return len(definitions)


def import_file(directory, path, *, replace=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Pronunciation file must be a regular file: {path}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Pronunciation file exceeds its size limit")
    return import_definitions(directory, entries_from(path.read_text(encoding="utf-8")), replace=replace)
