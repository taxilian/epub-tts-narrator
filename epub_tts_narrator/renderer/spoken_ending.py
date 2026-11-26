"""The spoken opening and ending: one narrator line before the first chapter and one after the last.

"You are listening to <title>, by <authors>." is planned as one extra chapter
(``spoken-opening``, "Opening Credits") at the start of a whole-book render
plan, and "This has been <title> by <authors>. Thank you for listening!" as
one (``spoken-ending``, "End Credits") at its end. Both are built only from
the book's title and authors (``book_details.bundle_book``) and the narrator
role. They never change the analysis, its source text, or any block,
segment, chunk or chapter ID, so every existing chapter keeps its checkpoint
and audio; each is rendered and verified like any other narration take.

Chapter numbers are positive and distinct, and every source chapter keeps its
own, so the ending is number last+1 and the opening last+2 although it plays
first: play order is the plan's order, never the number.
"""

from __future__ import annotations

import re
from string import Formatter

from .common import digest


ENDING_ID = "spoken-ending"
OPENING_ID = "spoken-opening"
# Changing how a line is built or bound changes its chunk identity.
VERSION = 1
FIELDS = frozenset({"title", "authors"})
MAX_TEXT_CHARS = 400
MAX_TITLE_CHARS = 120
SETTINGS = {
    "ending": True,
    "ending_title": "End Credits",
    "ending_text": "This has been {title} by {authors}. Thank you for listening!",
    "ending_text_without_authors": "This has been {title}. Thank you for listening!",
    "opening": True,
    "opening_title": "Opening Credits",
    "opening_text": "You are listening to {title}, by {authors}.",
    "opening_text_without_authors": "You are listening to {title}.",
}
KINDS = ("opening", "ending")


def _fields(template):
    try:
        return [(field, conversion, spec) for _, field, spec, conversion in Formatter().parse(template)
                if field is not None]
    except ValueError as exc:
        raise ValueError(f"has unbalanced braces ({exc})") from None


def validate_settings(render):
    """Check the ``[render]`` opening and ending settings; errors name the setting."""
    for kind in KINDS:
        _validate_kind(render, kind)


def _validate_kind(render, kind):
    if type(render.get(kind, True)) is not bool:
        raise ValueError(f"render.{kind} must be boolean")
    title = render.get(f"{kind}_title", SETTINGS[f"{kind}_title"])
    if (not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE_CHARS
            or any(ord(char) < 32 for char in title)):
        raise ValueError(f"render.{kind}_title must be a nonempty single line of at most {MAX_TITLE_CHARS} characters")
    for key, allowed in ((f"{kind}_text", FIELDS), (f"{kind}_text_without_authors", FIELDS - {"authors"})):
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
    """``A``, ``A and B``, ``A, B, and C``; blank, placeholder (``Unknown``) and repeated names are dropped."""
    from ..book_details import real_authors
    names, seen = [], set()
    for author in real_authors(authors):
        name = _plain(re.sub(r"[()\[\]<>]", " ", author)) if isinstance(author, str) else ""
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            names.append(name)
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def ending_text(book, render):
    """The ending line for ``book`` (``source.book``), or None when the book has no title."""
    return credit_text(book, render, "ending")


def opening_text(book, render):
    """The opening line for ``book`` (``source.book``), or None when the book has no title."""
    return credit_text(book, render, "opening")


def credit_text(book, render, kind):
    title = spoken_title(book.get("title", "")) if isinstance(book.get("title"), str) else ""
    if not title.strip(" ,"):
        return None
    authors = spoken_authors(book.get("authors"))
    template = render.get(f"{kind}_text", SETTINGS[f"{kind}_text"])
    if not authors and "authors" in {field for field, _, _ in _fields(template)}:
        template = render.get(f"{kind}_text_without_authors", SETTINGS[f"{kind}_text_without_authors"])
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


def is_opening(chapter):
    return isinstance(chapter, dict) and chapter.get("opening") is True


def is_credits(chapter):
    """The spoken opening or ending: a plan chapter that is not in the bundle."""
    return is_opening(chapter) or is_ending(chapter)


def label(chapter):
    return "the spoken opening" if is_opening(chapter) else "the spoken ending"


def enabled(config, kind=None):
    """The opening and ending are each on unless ``[render] opening``/``ending = false``; no ``kind``: either."""
    render = config.get("render", {})
    return any(render.get(name, SETTINGS[name]) is True for name in ((kind,) if kind else KINDS))


def ending_chapter(bundle, plan, render):
    """The plan chapter that speaks the ending, or None when there is nothing to end.

    Only a whole-book plan with speech gets one: a partial render, a book
    whose chapters are all ignored, or a book whose narrator voice speaks no
    line (or that has no narrator) has none.
    """
    return _credit_chapter(bundle, plan, render, "ending")


def opening_chapter(bundle, plan, render):
    """The plan chapter that speaks the opening, or None (as for ``ending_chapter``)."""
    return _credit_chapter(bundle, plan, render, "opening")


def _credit_chapter(bundle, plan, render, kind):
    if plan["partial_book"] or not any(chapter["chunks"] for chapter in plan["chapters"]
                                       if not is_credits(chapter)):
        return None
    role = narrator_role(bundle["cast"])
    from ..book_details import bundle_book
    text = credit_text(bundle_book(bundle), render, kind)
    # Only a narrator voice the book already uses: the credits never design a voice.
    if role is None or text is None or role["voice_id"] not in plan["voice_ids"]:
        return None
    from .bundle import split_text
    chapter_id = OPENING_ID if kind == "opening" else ENDING_ID
    row = {"id": chapter_id, "block_id": chapter_id, "chapter_id": chapter_id,
           "start": 0, "end": len(text), "text": text, "kind": "narration",
           "role_id": role["id"], "voice_id": role["voice_id"],
           "delivery": {"direction": "", "evidence": []}, "break_after": "paragraph"}
    chunks = []
    for start, end in split_text(text, render.get("max_chunk_chars", 320)):
        chunk = {"segment": row, "spoken_text": text[start:end], "spoken_start": start, "spoken_end": end,
                 "source_ranges": [{"start": start, "end": end}], "silent_control": False,
                 "chunk_id": digest({f"spoken_{kind}": VERSION, "role_id": role["id"],
                                     "voice_id": role["voice_id"], "start": start, "end": end,
                                     "spoken_text": text[start:end]})[:32]}
        chunks.append(chunk)
    # The ending keeps the number it always had; the opening takes the next.
    number = len(bundle["chapters"]) + (2 if kind == "opening" else 1)
    return {"number": number, "id": chapter_id,
            "title": _plain(render.get(f"{kind}_title", SETTINGS[f"{kind}_title"])),
            "script_path": None, "audio_stem": chapter_id, "chunks": chunks, kind: True}
