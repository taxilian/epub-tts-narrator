"""The spoken ending: one narrator line after the last chapter.

"This has been <title> by <authors>. Thank you for listening!" is planned as
one extra chapter (``spoken-ending``) at the end of a whole-book render plan,
built only from the bundle's ``source.book`` metadata and the narrator role.
It never changes the analysis, its source text, or any block, segment, chunk
or chapter ID, so every existing chapter keeps its checkpoint and audio; the
ending is rendered and verified like any other narration take.
"""

from __future__ import annotations

import re
from string import Formatter

from .common import digest


ENDING_ID = "spoken-ending"
# Changing how the line is built or bound changes its chunk identity.
VERSION = 1
FIELDS = frozenset({"title", "authors"})
MAX_TEXT_CHARS = 400
MAX_TITLE_CHARS = 120
SETTINGS = {
    "ending": True,
    "ending_title": "End Credits",
    "ending_text": "This has been {title} by {authors}. Thank you for listening!",
    "ending_text_without_authors": "This has been {title}. Thank you for listening!",
}


def _fields(template):
    try:
        return [(field, conversion, spec) for _, field, spec, conversion in Formatter().parse(template)
                if field is not None]
    except ValueError as exc:
        raise ValueError(f"has unbalanced braces ({exc})") from None


def validate_settings(render):
    """Check the ``[render]`` ending settings; errors name the setting."""
    if type(render.get("ending", True)) is not bool:
        raise ValueError("render.ending must be boolean")
    title = render.get("ending_title", SETTINGS["ending_title"])
    if (not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE_CHARS
            or any(ord(char) < 32 for char in title)):
        raise ValueError(f"render.ending_title must be a nonempty single line of at most {MAX_TITLE_CHARS} characters")
    for key, allowed in (("ending_text", FIELDS), ("ending_text_without_authors", FIELDS - {"authors"})):
        template = render.get(key, SETTINGS[key])
        if not isinstance(template, str) or not template.strip() or len(template) > MAX_TEXT_CHARS:
            raise ValueError(f"render.{key} must be nonempty text of at most {MAX_TEXT_CHARS} characters")
        try:
            fields = _fields(template)
        except ValueError as exc:
            raise ValueError(f"render.{key} {exc}; write a literal brace as {{{{ or }}}}") from None
        for field, conversion, spec in fields:
            if field not in allowed or conversion or spec:
                raise ValueError(f"render.{key} may use only " + " and ".join(
                    "{" + name + "}" for name in sorted(allowed)) + " as placeholders")
        if any(char in template for char in "[]<>") or any(ord(char) < 32 for char in template):
            raise ValueError(f"render.{key} must be plain spoken text without brackets or control characters")


def _plain(value):
    """One line of speakable text: brackets are never spoken, so they become commas."""
    return re.sub(r"\s+", " ", str(value)).strip()


def spoken_title(title):
    """A trailing parenthetical (often the series) is read as an aside:
    ``Pawn of Prophecy (The Belgariad Book 1)`` -> ``Pawn of Prophecy, The Belgariad Book 1,``.
    Other brackets are dropped, as the renderer never speaks bracketed text.
    """
    title = _plain(title)
    match = re.fullmatch(r"(.*?\S)\s*[(\[]([^()\[\]]*\S[^()\[\]]*)[)\]]", title)
    if match:
        title = f"{match.group(1)}, {_plain(match.group(2))},"
    return _plain(re.sub(r"[()\[\]<>]", " ", title))


def spoken_authors(authors):
    """``A``, ``A and B``, ``A, B, and C``; blank and repeated names are dropped."""
    names, seen = [], set()
    for author in authors if isinstance(authors, list) else []:
        name = _plain(re.sub(r"[()\[\]<>]", " ", author)) if isinstance(author, str) else ""
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def ending_text(book, render):
    """The ending line for ``book`` (``source.book``), or None when the book has no title."""
    title = spoken_title(book.get("title", "")) if isinstance(book.get("title"), str) else ""
    if not title.strip(" ,"):
        return None
    authors = spoken_authors(book.get("authors"))
    template = render.get("ending_text", SETTINGS["ending_text"])
    if not authors and "authors" in {field for field, _, _ in _fields(template)}:
        template = render.get("ending_text_without_authors", SETTINGS["ending_text_without_authors"])
    text = template.format(title=title, authors=authors)
    # A title read as an aside ends in a comma; drop it before other punctuation.
    text = re.sub(r",\s*([,.!?;:])", r"\1", text)
    return _plain(text)


def narrator_role(cast):
    """The main narration role (``narrator_main`` first), or None when the cast has none."""
    roles = [role for role in cast.get("roles", []) if isinstance(role, dict)
             and role.get("kind") == "narration" and isinstance(role.get("voice_id"), str)]
    return next((role for role in roles if role.get("id") == "narrator_main"), roles[0] if roles else None)


def is_ending(chapter):
    return isinstance(chapter, dict) and chapter.get("ending") is True


def enabled(config):
    """The ending is on unless ``[render] ending = false``."""
    return config.get("render", {}).get("ending", SETTINGS["ending"]) is True


def ending_chapter(bundle, plan, render):
    """The plan chapter that speaks the ending, or None when there is nothing to end.

    Only a whole-book plan with speech gets one: a partial render, a book
    whose chapters are all ignored, or a book whose narrator voice speaks no
    line (or that has no narrator) has none.
    """
    if plan["partial_book"] or not any(chapter["chunks"] for chapter in plan["chapters"]):
        return None
    role = narrator_role(bundle["cast"])
    text = ending_text(bundle["source"]["book"], render)
    # Only a narrator voice the book already uses: the ending never designs a voice.
    if role is None or text is None or role["voice_id"] not in plan["voice_ids"]:
        return None
    from .bundle import split_text
    row = {"id": ENDING_ID, "block_id": ENDING_ID, "chapter_id": ENDING_ID,
           "start": 0, "end": len(text), "text": text, "kind": "narration",
           "role_id": role["id"], "voice_id": role["voice_id"],
           "delivery": {"direction": "", "evidence": []}, "break_after": "paragraph"}
    chunks = []
    for start, end in split_text(text, render.get("max_chunk_chars", 320)):
        chunk = {"segment": row, "spoken_text": text[start:end], "spoken_start": start, "spoken_end": end,
                 "source_ranges": [{"start": start, "end": end}], "silent_control": False,
                 "chunk_id": digest({"spoken_ending": VERSION, "role_id": role["id"],
                                     "voice_id": role["voice_id"], "start": start, "end": end,
                                     "spoken_text": text[start:end]})[:32]}
        chunks.append(chunk)
    return {"number": len(bundle["chapters"]) + 1, "id": ENDING_ID, "title": _plain(render.get("ending_title", SETTINGS["ending_title"])),
            "script_path": None, "audio_stem": ENDING_ID, "chunks": chunks, "ending": True}
