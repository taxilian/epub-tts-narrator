"""A book's title and authors as its audiobook names them.

The EPUB's ``dc:title``/``dc:creator`` are extracted into ``source.book``.
A placeholder a conversion tool writes when it does not know the author
(Calibre's ``Unknown``) is no author: ``real_authors`` drops it at extraction
and wherever saved books are read, so a book saved with ``["Unknown"]`` reads
as having none.

A later choice of title or authors, a person's (``review-ui`` ``b``) or the
analysis's (``author_review``: the book's own front matter names the author),
is kept in ``provenance.book_details``, never written into ``source.book``:
the source fingerprint binds chunk IDs, series history, journals and
source-bound sidecars, and none of those change when the book is renamed.
``effective_book`` lays the choice over ``source.book``; the spoken ending,
the M4B tags and a fresh ``.m4b`` basename read it, so a change is used by the
next ``render`` or ``assemble``. Publication goes through
``analysis_revision.revise_directories`` scope ``book-details``
(``validate_book_details_revision``), with manifest-hash receipts and later
books' series history rebound.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
import unicodedata

FIELDS = ("title", "authors")
BASES = frozenset({"user", "analysis"})
MAX_TITLE_CHARS = 500
MAX_AUTHOR_CHARS = 200
MAX_AUTHORS = 20

# What conversion tools write when they do not know a book's author. A real
# attribution such as "Anonymous" is not among them.
_PLACEHOLDERS = frozenset({
    "unknown", "unknown author", "unknown authors", "author unknown", "authors unknown",
    "unknown writer", "no author", "none", "null", "nil", "n/a", "na", "tbd", "tba", "-", "?",
})


def _key(name):
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def placeholder(name):
    """Whether ``name`` is a stand-in for a missing author (``Unknown``, ``[unknown]``, ``N/A``), not a name."""
    if not isinstance(name, str):
        return True
    key = _key(name).strip(" ()[]{}<>.,;:'\"")
    return not key or key in _PLACEHOLDERS


def clean_name(name):
    return " ".join(unicodedata.normalize("NFC", name).split()) if isinstance(name, str) else ""


def real_authors(authors):
    """The authors with placeholders and blank or repeated names dropped, in order."""
    result, seen = [], set()
    for name in authors if isinstance(authors, list) else []:
        name = clean_name(name)
        if placeholder(name) or _key(name) in seen:
            continue
        seen.add(_key(name))
        result.append(name)
    return result


def split_authors(text):
    """A typed author list (``A; B`` or one name per line) as names; placeholders are refused."""
    if not isinstance(text, str):
        raise ValueError("Authors must be text")
    names = [clean_name(part) for part in re.split(r"[;\n]", text)]
    names = [name for name in names if name]
    for name in names:
        if placeholder(name):
            raise ValueError(f"{name!r} is not an author's name; leave the authors empty when the book has none")
    return check_authors(names)


def check_title(title):
    title = clean_name(title)
    if not title:
        raise ValueError("The title must not be empty")
    if len(title) > MAX_TITLE_CHARS:
        raise ValueError(f"The title must be at most {MAX_TITLE_CHARS} characters")
    return title


def check_authors(authors):
    if not isinstance(authors, list) or not all(isinstance(name, str) for name in authors):
        raise ValueError("Authors must be a list of names")
    names = [clean_name(name) for name in authors]
    if any(not name or placeholder(name) for name in names):
        raise ValueError("Each author must be a name, not blank or a placeholder such as Unknown")
    if len({_key(name) for name in names}) != len(names):
        raise ValueError("An author is listed twice")
    if len(names) > MAX_AUTHORS or any(len(name) > MAX_AUTHOR_CHARS for name in names):
        raise ValueError(f"At most {MAX_AUTHORS} authors of at most {MAX_AUTHOR_CHARS} characters each")
    return names


def validate_record(record):
    """Raise ValueError unless ``record`` is a valid ``provenance.book_details``."""
    if not isinstance(record, dict) or record.get("version") != 1 or not set(record) <= {"version", *FIELDS}:
        raise ValueError("provenance.book_details must be {version: 1, title?, authors?}")
    for field in FIELDS:
        if field not in record:
            continue
        entry = record[field]
        keys = {"value", "basis"} | ({"evidence"} if isinstance(entry, dict) and entry.get("basis") == "analysis"
                                     else set())
        if not isinstance(entry, dict) or not {"value", "basis"} <= set(entry) <= keys or entry["basis"] not in BASES:
            raise ValueError(f"provenance.book_details.{field} must be {{value, basis: user|analysis}}")
        if field == "title":
            if not isinstance(entry["value"], str) or check_title(entry["value"]) != entry["value"]:
                raise ValueError("provenance.book_details.title.value must be a clean, nonempty title")
        elif check_authors(entry["value"]) != entry["value"]:
            raise ValueError("provenance.book_details.authors.value must be clean author names")
        if field == "authors" and entry["basis"] == "analysis" and not entry["value"]:
            raise ValueError("The analysis records authors only when the book names them")
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
            raise ValueError("provenance.book_details evidence must be a list of quotes")


def details(provenance):
    """The saved ``provenance.book_details`` (validated), or ``{}``."""
    record = provenance.get("book_details") if isinstance(provenance, dict) else None
    if record is None:
        return {}
    validate_record(record)
    return record


def basis(provenance, field):
    """Who chose ``field`` (``"user"``, ``"analysis"``), or None when it is the EPUB's."""
    entry = details(provenance).get(field)
    return entry["basis"] if entry else None


def effective_book(book, provenance=None):
    """``source.book`` with a chosen title/authors laid over it and placeholder authors dropped."""
    result = deepcopy(book) if isinstance(book, dict) else {}
    record = details(provenance)
    for field in FIELDS:
        if field in record:
            result[field] = deepcopy(record[field]["value"])
    result["authors"] = real_authors(result.get("authors"))
    return result


def document_book(document):
    return effective_book(document["source"]["book"], document.get("provenance"))


def bundle_book(bundle):
    """The renderer's view: a loaded bundle carries the manifest provenance."""
    provenance = bundle.get("provenance")
    if provenance is None:
        provenance = bundle.get("manifest", {}).get("provenance")
    return effective_book(bundle["source"]["book"], provenance)


_UNSET = object()


def with_details(document, *, title=_UNSET, authors=_UNSET, basis="user", evidence=None):
    """``document`` with a new title and/or authors recorded; a person's value equal to the EPUB's clears it."""
    if basis not in BASES:
        raise ValueError("basis must be user or analysis")
    candidate = deepcopy(document)
    record = deepcopy(details(document["provenance"])) or {"version": 1}
    book = document["source"]["book"]
    for field, value in (("title", title), ("authors", authors)):
        if value is _UNSET:
            continue
        value = check_title(value) if field == "title" else check_authors(value)
        original = book.get("title") if field == "title" else real_authors(book.get("authors"))
        # A person's choice equal to the EPUB's is no choice; "no author" for an EPUB without one is
        # kept, so the book is not asked about again.
        if value == original and basis == "user" and value:
            record.pop(field, None)
            continue
        entry = {"value": value, "basis": basis}
        if basis == "analysis" and evidence:
            entry["evidence"] = list(evidence)
        record[field] = entry
    if set(record) == {"version"}:
        candidate["provenance"].pop("book_details", None)
    else:
        validate_record(record)
        candidate["provenance"]["book_details"] = record
    return candidate


def validate_book_details_revision(document, candidate):
    """A ``book-details`` revision changes nothing but ``provenance.book_details``."""
    before = {key: value for key, value in document.items() if key != "provenance"}
    after = {key: value for key, value in candidate.items() if key != "provenance"}
    rest = {key: value for key, value in document["provenance"].items() if key != "book_details"}
    if before != after or rest != {key: value for key, value in candidate["provenance"].items()
                                   if key != "book_details"}:
        raise ValueError("A title or author change may change only provenance.book_details")
    details(candidate["provenance"])


def describe(book):
    authors = book.get("authors") or []
    return f"{book.get('title', '')} — " + (", ".join(authors) if authors else "author unknown")


def save_details(directory, *, title=_UNSET, authors=_UNSET, basis="user", evidence=None, document=None,
                 expected_manifest_hashes=None, progress=print, revise=None):
    """Publish a title/author change for the book in ``directory``; returns the revision result (None: no change).

    ``document`` and ``expected_manifest_hashes`` are the caller's receipt (the
    book as it was read); without them the book is read now.
    """
    from .analysis_directory import _completed_directory
    from .analysis_revision import related_directories, revise_directories
    directory = Path(directory).resolve()
    paths = related_directories([directory])
    if document is None:
        document = _completed_directory(directory)[0]
    if expected_manifest_hashes is None:
        expected_manifest_hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                                    for path in paths}
    candidate = with_details(document, title=title, authors=authors, basis=basis, evidence=evidence)
    if candidate == document:
        return None
    return (revise or revise_directories)(
        {directory: candidate}, directories=paths, expected_manifest_hashes=expected_manifest_hashes,
        scope="book-details", progress=progress)
