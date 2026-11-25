"""What the questions ``analyze BOOK`` reruns ask look like on screen (stdlib only, no Textual).

Inside the run shell each review's question opens its own screen
(``rerun_review_tui``) instead of the generic line-prompt dialog: the cast
review's "are these one character?" (M/K/L), the gender check's "fix the
description?" (F/L), and the narrator check's "apply these narrators?" (A/L).
The functions here turn what the review already worked out into a plain
``view`` dict the screen draws, so they can be tested offline. The answer
comes back as the same letter the line prompt takes and goes through the same
code; on a plain terminal the line prompt is unchanged.
"""

from __future__ import annotations

NO_SCREEN = object()


def ask_on_screen(screen_name, view):
    """The letter answered on the open full-screen UI's ``rerun_review_tui.<screen_name>``.

    ``NO_SCREEN`` when no full-screen UI is open (ask on the line instead).
    """
    from .tui_support import active_screen_host
    show = active_screen_host()
    if show is None:
        return NO_SCREEN

    def make():
        from . import rerun_review_tui
        return getattr(rerun_review_tui, screen_name)(view)
    return show(make)


def _capital(text):
    return text[:1].upper() + text[1:]


def _lines(count):
    return f"{count} line" + ("" if count == 1 else "s")


def merge_view(plan, why):
    """The cast review's question about one pair, from ``character_merge_command.plan_merge``'s plan."""
    from .character_merge import name_key
    kept, absorbed = plan.kept, plan.absorbed
    same = name_key(kept["name"]) == name_key(absorbed["name"])
    kept_label = f"{kept['name']} on the left" if same else kept["name"]
    absorbed_label = f"{absorbed['name']} on the right" if same else absorbed["name"]
    this_book = plan.books[0]

    def card(entry, heading, voice, lines, first):
        return {"heading": heading, "name": entry["name"], "aliases": list(entry.get("aliases", [])),
                "description": " ".join(entry.get("description", "").split()), "first_line": first or "",
                "lines": lines, "voice": _capital(voice),
                "from": (f"From an earlier book: {plan.origins[entry['id']]}" if entry["id"] in plan.origins
                         else f"New in {this_book.title}")}

    moved = this_book.moved_lines
    # The cards name the earlier book; an entry from one always stays.
    reason = "an entry from an earlier book always stays" if plan.fixed else plan.reason
    summary = f"Merging keeps {kept_label}, because {reason}. " + (
        f"{_capital(absorbed_label)}'s {_lines(moved)} {'moves' if moved == 1 else 'move'} to {kept_label}'s voice "
        f"and will be rendered again."
        if moved else f"{_capital(absorbed_label)} has no lines in this book, so nothing is rendered again.")
    later = plan.books[1:]
    notes = []
    if this_book.joined_lines:
        notes.append(f"{this_book.joined_lines} passage(s) they speak together become one voice.")
    if plan.absorbed_voice.startswith("accepted"):
        notes.append(f"The recording accepted for {absorbed_label} will no longer be used.")
    if later:
        notes.append(f"Later books follow: {sum(book.moved_lines for book in later)} more line(s) in "
                     + ", ".join(book.title for book in later) + ".")
    if plan.earlier:
        notes.append(f"Earlier books keep {absorbed['name']} as a separate entry: " + ", ".join(plan.earlier) + ".")
    others = [name for name in plan.names if name_key(name) != name_key(plan.name)]
    return {"title": "Are these the same character?", "book": this_book.title, "why": _capital(why) + ".",
            "entries": [card(kept, "Stays", plan.kept_voice, plan.kept_lines, plan.kept_first),
                        card(absorbed, "Merged into it", plan.absorbed_voice, plan.absorbed_lines,
                             plan.absorbed_first)],
            "summary": summary,
            "afterwards": f"Afterwards: {plan.name}" + (", also called " + ", ".join(others) if others else ""),
            "notes": notes, "details": plan.text}


def gender_view(row, description, current, wanted, why, fixed):
    """The gender check's question about one character whose voice already exists."""
    aliases = [name for name in row.get("aliases", []) if name.strip()]
    return {"title": f"Is {row['name']}'s voice the right gender?", "name": row["name"], "aliases": aliases,
            "current": current or "no gender", "wanted": wanted, "why": _capital(why) + ".",
            "description": " ".join(description.split()), "fixed": " ".join(fixed.split()),
            "voice": "A voice already exists. Fixing the description changes how it sounds only when you "
                     "recast it (cast BOOK); leaving it keeps the description as it is."}


def narrator_view(document, revised, book_sections, chosen, answers, notes, *, label, own_voices):
    """The narrator check's question: whose point of view each section follows, and what changes."""
    names = {row["id"]: row["name"] for row in document["cast"]["characters"]}
    after = {segment["id"]: segment for segment in revised["segments"]}
    changed_by_block = {}
    for segment in document["segments"]:
        if after.get(segment["id"]) != segment:
            changed_by_block[segment["block_id"]] = changed_by_block.get(segment["block_id"], 0) + 1
    rows, moved = [], 0
    for section in book_sections:
        count = sum(changed_by_block.get(block["id"], 0) for block in section["blocks"])
        moved += count
        quote = answers.get(section["key"])
        rows.append({"label": label(section), "reader": names.get(chosen[section["key"]], "main narrator"),
                     "pov": chosen[section["key"]] is not None, "quote": quote[1] if quote else "",
                     "changed": count})
    summary = (f"{moved} narration passage(s) would be read in a different voice and rendered again."
               if moved else "No narration passage changes voice.")
    if own_voices:
        summary += (" Each point-of-view narrator gets its own voice, linked to the character's recording and "
                    "accent.")
    return {"title": "Read narration in the point-of-view character's voice?", "notes": " ".join(notes.split()),
            "sections": rows, "moved": moved, "summary": summary}


def joint_view(document, row, speakers, quote):
    """The joint-line check's question about a line a person gave to one speaker."""
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    names = {character["id"]: character["name"] for character in document["cast"]["characters"]}

    def name(role_id):
        return names.get(roles.get(role_id, {}).get("character_id"), role_id)
    paragraph = next(block["text"] for block in document["source"]["blocks"] if block["id"] == row["block_id"])
    together = [name(role_id) for role_id in speakers]
    return {"title": "Do they say this together?", "line": row["text"], "paragraph": " ".join(paragraph.split()),
            "now": f"{name(row['role_id'])} alone (you chose)", "together": ", ".join(together),
            "speakers": together, "why": f"The book: “{quote}”.",
            "afterwards": "Saying it together renders the line in every one of these voices at once; only this "
                          "line changes. Keeping it is remembered and not asked again."}


def split_view(document, character_id, split, target, created, moved, held):
    """The character split's question: which of an entry's lines go to which other character."""
    characters = {row["id"]: row for row in document["cast"]["characters"]}
    source = characters[character_id]
    texts = {row["id"]: row["text"] for row in document["segments"]}
    lines = [" ".join(texts[segment_id].split())[:120] for segment_id in moved[:12]]
    if len(moved) > 12:
        lines.append(f"…and {len(moved) - 12} more")
    goes = f"a new character, {target['name']}" if created else target["name"]
    notes = []
    if created:
        notes.append(f"{target['name']}: {split['description']} A new voice is cast for it (cast BOOK).")
    if held:
        notes.append(f"{_lines(len(held))} you chose for {source['name']} {'stays' if len(held) == 1 else 'stay'} "
                     "(add --reconsider-choices to move them too).")
    notes.append(f"{source['name']} keeps its other lines and its voice. Keeping them together is remembered.")
    return {"title": f"Are some of {source['name']}'s lines someone else's?",
            "summary": f"{_capital(_lines(len(moved)))} of {source['name']} would go to {goes}.",
            "why": f"The book: “{split['evidence']['quote']}”.", "from": source["name"], "to": target["name"],
            "lines": lines, "notes": notes}
