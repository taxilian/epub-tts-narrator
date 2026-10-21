"""Identify explicit author narration without inventing a fictional identity."""

from __future__ import annotations

import re
import unicodedata


def author_name_key(name):
    """Exact name matching permits only Unicode, spacing, and case normalization."""
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def author_narrator_name(role, book):
    """Return the sole book author's name for an explicit author narration scope.

    A generic narrator, editor, guest contributor, or character is not the author.
    This identifies a casting scope, not the author's real speaking voice or gender.
    """
    if (not isinstance(role, dict) or not isinstance(book, dict)
            or role.get("kind") != "narration" or role.get("character_id") is not None
            or role.get("id") == "narrator_main"):
        return None
    authors = book.get("authors")
    if not isinstance(authors, list) or not authors or any(not isinstance(name, str) or not name.strip() for name in authors):
        return None
    names = {author_name_key(name): " ".join(unicodedata.normalize("NFKC", name).split()) for name in authors}
    if len(names) != 1:
        return None
    key, name = next(iter(names.items()))
    if (key in {"unknown", "anonymous", "various", "various authors", "multiple authors"}
            or re.search(r"\b(?:and|et al)\b|[&;]", key)):
        return None
    description = role.get("description", "")
    if not isinstance(description, str):
        return None
    description = unicodedata.normalize("NFKC", description).casefold().replace("’", "'")
    if re.search(r"\b(?:guest|translator|editorial|editor)\b|author\s*/\s*editor", description):
        return None
    if re.search(r"\b(?:not|never|no)\s+(?:the\s+)?author\b|\bnon[ -]author\b", description):
        return None
    author_scope = (
        r"\bauthor(?:'s)?\s+(?:afterword|foreword|acknowledg(?:e)?ments|note\b|notes\b|narration\b|"
        r"back[ -]matter|front[ -]matter|direct\s+address|commentary\b)"
        r"|\b(?:afterword|foreword|acknowledg(?:e)?ments|direct\s+address)\b.{0,60}\b(?:by|from)\s+(?:the\s+)?author\b"
        r"|\bfrom\s+the\s+author\b"
    )
    if not re.search(author_scope, description):
        return None
    # A byline naming someone else overrides a generic "author afterword"
    # label; the book metadata alone cannot make that contributor the author.
    byline = re.search(r"\b(?:afterword|foreword|acknowledg(?:e)?ments|narration|direct\s+address|notes?)"
                       r"\s+(?:narration\s+)?by\s+(.+)", description)
    if byline:
        named = author_name_key(byline[1])
        if not (re.match(r"(?:the\s+)?author\b", named)
                or named == key or named.startswith(key + ":")
                or named.startswith(key + ";") or named.startswith(key + ".") or named.startswith(key + ",")):
            return None
    return name
