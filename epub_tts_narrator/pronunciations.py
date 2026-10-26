"""Book-local pronunciation decisions with live, read-only series inheritance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .performance_plan import validate_phonemes
from .pipeline import fingerprint, write_json


FORMAT = "epub-tts-pronunciations"
VERSION = 1
DISCOVERY_VERSION = 1
FILENAME = "pronunciations.json"
KINDS = frozenset({"name", "place", "invented", "foreign", "ambiguous", "other"})
STATUSES = frozenset({"pending", "confirmed", "ignored"})
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’\-‐‑][^\W_]+)*", re.UNICODE)


def validate_word(word):
    if (not isinstance(word, str) or not 1 <= len(word) <= 128
            or WORD_PATTERN.fullmatch(word) is None or not any(character.isalpha() for character in word)):
        raise ValueError("Pronunciation word must be one lexical word, optionally joined by apostrophes or hyphens")
    return word


def word_id(word):
    """Case and exact spelling are meaningful; similar names never collapse."""
    return "pron_" + hashlib.sha256(validate_word(word).encode("utf-8")).hexdigest()


def validate_pronunciation(value):
    if not isinstance(value, dict) or set(value) != {"alphabet", "value"}:
        raise ValueError("Pronunciation requires alphabet and value")
    if value["alphabet"] == "respelling":
        validate_word(value["value"])
    elif value["alphabet"] == "cmu_arpabet":
        validate_phonemes(value["value"])
    else:
        raise ValueError("Pronunciation alphabet must be respelling or cmu_arpabet")
    return deepcopy(value)


def _plain(value, maximum):
    return (isinstance(value, str) and 0 < len(value) <= maximum and value == value.strip()
            and not any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value))


def _word_in(word, text):
    return any(match.group() == word for match in WORD_PATTERN.finditer(text))


def _evidence_contains_word(word, quote, source_text):
    """At least one exact quotation occurrence must contain a complete source word."""
    source_positions = [match.start() for match in WORD_PATTERN.finditer(source_text) if match.group() == word]
    for token in WORD_PATTERN.finditer(quote):
        if token.group() != word:
            continue
        for position in source_positions:
            quote_start = position - token.start()
            if quote_start >= 0 and source_text[quote_start:quote_start + len(quote)] == quote:
                return True
    return False


def validate_entry(entry, source):
    """Validate stored/proposed metadata against its own book, not a descendant."""
    fields = {"id", "word", "status", "kind", "reason", "evidence"}
    if not isinstance(entry, dict) or set(entry) - fields - {"pronunciation"} or not fields <= set(entry):
        raise ValueError("Pronunciation entry has missing or unsupported fields")
    word = validate_word(entry["word"])
    if (entry["id"] != word_id(word) or not isinstance(entry["status"], str) or entry["status"] not in STATUSES
            or not isinstance(entry["kind"], str) or entry["kind"] not in KINDS or not _plain(entry["reason"], 1000)):
        raise ValueError("Pronunciation entry identity, status, kind, or reason is invalid")
    evidence = entry["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 32:
        raise ValueError("Pronunciation entry needs bounded exact source evidence")
    blocks = {block["id"]: block["text"] for block in source["blocks"]}
    for citation in evidence:
        if (not isinstance(citation, dict) or set(citation) != {"block_id", "quote"}
                or not isinstance(citation["block_id"], str) or citation["block_id"] not in blocks
                or not isinstance(citation["quote"], str) or not 1 <= len(citation["quote"]) <= 2000
                or not _evidence_contains_word(word, citation["quote"], blocks[citation["block_id"]])):
            raise ValueError("Pronunciation evidence must quote the exact whole word from a referenced source block")
    if entry["status"] == "confirmed":
        validate_pronunciation(entry.get("pronunciation"))
    elif "pronunciation" in entry:
        raise ValueError("Only confirmed pronunciation entries may contain a pronunciation")
    return deepcopy(entry)


def _regular_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Pronunciation metadata must be stored in a regular file")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Pronunciation metadata exceeds its size limit")
    def reject(value):
        raise ValueError("Pronunciation metadata contains a non-finite value")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _source(directory):
    source = _regular_json(directory / "source.json")
    if (not isinstance(source, dict) or not isinstance(source.get("book"), dict)
            or not isinstance(source["book"].get("id"), str) or not isinstance(source.get("blocks"), list)):
        raise ValueError("Pronunciation metadata requires this book's source")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = _regular_json(manifest_path)
        if manifest.get("source_fingerprint") != fingerprint(source):
            raise ValueError("Pronunciation source differs from the book manifest")
    return source


def _empty(source):
    return {"format": FORMAT, "version": VERSION, "book_id": source["book"]["id"],
            "source_fingerprint": fingerprint(source), "entries": []}


def _load_local(directory, source=None):
    path = directory / FILENAME
    if not path.exists() and not path.is_symlink():
        return _empty(source) if source is not None else None
    source = _source(directory) if source is None else source
    record = _regular_json(path)
    required = {"format", "version", "book_id", "source_fingerprint", "entries"}
    if (not isinstance(record, dict) or set(record) - required - {"discovery"} or not required <= set(record)
            or record["format"] != FORMAT or type(record["version"]) is not int or record["version"] != VERSION
            or record["book_id"] != source["book"]["id"] or record["source_fingerprint"] != fingerprint(source)
            or not isinstance(record["entries"], list) or len(record["entries"]) > 100000):
        raise ValueError("Pronunciation registry is invalid or belongs to different source text")
    seen = set()
    for entry in record["entries"]:
        validate_entry(entry, source)
        if entry["id"] in seen:
            raise ValueError("Pronunciation registry contains a duplicate exact word")
        seen.add(entry["id"])
    if "discovery" in record:
        marker = record["discovery"]
        if (not isinstance(marker, dict) or set(marker) != {"version", "source_fingerprint", "cast_fingerprint"}
                or type(marker["version"]) is not int or marker["version"] < 1
                or marker["source_fingerprint"] != record["source_fingerprint"]
                or not isinstance(marker["cast_fingerprint"], str) or re.fullmatch(r"[a-f0-9]{64}", marker["cast_fingerprint"]) is None):
            raise ValueError("Pronunciation discovery marker is invalid")
    return record


def _directory(directory):
    path = Path(directory)
    if path.is_symlink():
        raise ValueError("Pronunciation directory must not be a symlink")
    return path.resolve()


def _document_source(directory, document):
    if not isinstance(document, dict) or not isinstance(document.get("source"), dict):
        raise ValueError("Pronunciation changes require a loaded analysis document")
    source = _source(directory)
    if source != document["source"]:
        raise ValueError("Pronunciation changes require this book's exact source")
    return source


def load_pronunciations(directory):
    """Return effective decisions, rereading ancestors so late approvals are visible."""
    directory = _directory(directory)
    manifest_path = directory / "manifest.json"
    ancestors = []
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = _regular_json(manifest_path)
        if manifest.get("continuation") is not None:
            from .analysis_directory import continuation_directories
            ancestors = continuation_directories(directory)
    found = []
    for owner in [directory, *ancestors]:
        record = _load_local(owner)
        if record is not None:
            found.extend(dict(deepcopy(entry), origin_directory=str(owner), inherited=owner != directory)
                         for entry in record["entries"])
    effective = {}
    # An unresolved descendant proposal must never shadow a resolved ancestor.
    for status_group in (("confirmed", "ignored"), ("pending",)):
        for entry in found:
            if entry["status"] in status_group:
                effective.setdefault(entry["id"], entry)
    if not effective:
        return []
    current_words = {match.group() for block in _source(directory)["blocks"] for match in WORD_PATTERN.finditer(block["text"])}
    return sorted((entry for entry in effective.values() if entry["word"] in current_words),
                  key=lambda entry: (entry["word"].casefold(), entry["word"]))


def pronunciation_counts(entries):
    result = {status: 0 for status in ("pending", "confirmed", "ignored")}
    for entry in entries:
        result[entry["status"]] += 1
    return result


def _discovery_marker(document):
    return {"version": DISCOVERY_VERSION, "source_fingerprint": fingerprint(document["source"]),
            "cast_fingerprint": fingerprint(document.get("cast", {}))}


def discovery_pending(directory, document):
    directory = _directory(directory)
    source = _document_source(directory, document)
    return _load_local(directory, source).get("discovery") != _discovery_marker(document)


def _mutate(directory, operation):
    from .renderer.run_locks import acquire_directory_lock
    lock = acquire_directory_lock(directory / ".pronunciations.lock")
    try:
        record = operation()
        write_json(directory / FILENAME, record, overwrite=True)
    finally:
        lock.rmdir()
    return record


def save_candidates(directory, document, candidates):
    """Merge completed discovery without replacing any local human decision."""
    directory = _directory(directory)
    source = _document_source(directory, document)
    if not isinstance(candidates, list):
        raise ValueError("Pronunciation candidates must be a list")
    checked = [validate_entry(entry, source) for entry in candidates]
    if any(entry["status"] != "pending" for entry in checked):
        raise ValueError("Discovery may only propose pending pronunciation candidates")
    if len({entry["id"] for entry in checked}) != len(checked):
        raise ValueError("Pronunciation candidates contain duplicate exact words")
    def update():
        record = _load_local(directory, source)
        entries = {entry["id"]: entry for entry in record["entries"]}
        for entry in checked:
            if entries.get(entry["id"], {}).get("status", "pending") == "pending":
                entries[entry["id"]] = entry
        record["entries"] = sorted(entries.values(), key=lambda entry: entry["word"])
        record["discovery"] = _discovery_marker(document)
        return record
    _mutate(directory, update)
    return load_pronunciations(directory)


def set_pronunciation(directory, document, word, pronunciation=None, status="confirmed"):
    directory = _directory(directory)
    source = _document_source(directory, document)
    validate_word(word)
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError("Pronunciation status must be pending, confirmed, or ignored")
    evidence = next(({"block_id": block["id"], "quote": word} for block in source["blocks"] if _word_in(word, block["text"])), None)
    if evidence is None:
        raise ValueError("Pronunciation word must occur exactly in this book's source")
    if status == "confirmed":
        pronunciation = validate_pronunciation(pronunciation)
    elif pronunciation is not None:
        raise ValueError("Only confirmed entries may contain a pronunciation")
    def update():
        record = _load_local(directory, source)
        entries = {entry["id"]: entry for entry in record["entries"]}
        entry = deepcopy(entries.get(word_id(word), {"id": word_id(word), "word": word, "kind": "other",
                                                   "reason": "Manually selected for pronunciation review.", "evidence": [evidence]}))
        entry["status"] = status
        entry.pop("pronunciation", None)
        if pronunciation is not None:
            entry["pronunciation"] = pronunciation
        entries[entry["id"]] = validate_entry(entry, source)
        record["entries"] = sorted(entries.values(), key=lambda item: item["word"])
        return record
    _mutate(directory, update)
    return next(entry for entry in load_pronunciations(directory) if entry["id"] == word_id(word))


def reset_pronunciation(directory, word):
    """Remove only this book's entry, exposing the nearest ancestor decision."""
    directory = _directory(directory)
    identity = word_id(word)
    if not (directory / FILENAME).exists() and not (directory / FILENAME).is_symlink():
        return load_pronunciations(directory)
    source = _source(directory)
    def update():
        record = _load_local(directory, source)
        record["entries"] = [entry for entry in record["entries"] if entry["id"] != identity]
        return record
    _mutate(directory, update)
    return load_pronunciations(directory)
