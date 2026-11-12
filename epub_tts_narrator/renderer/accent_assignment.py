"""Book-wide accent assignment: group the cast by where they come from, one real accent per group.

A voice description names an accent only when whoever wrote it (usually a
model, one voice at a time) thought to, so descriptions are a poor source of
accents: a whole cast can come out "General American". This asks the text
model once for the whole cast: which characters come from the same place
(from the story, their names, how they speak and who they are with), and a
distinct real-world English accent for each group. A person's guidance
("the Alorns sound Scandinavian") goes with the request.

The result is saved beside the book in ``voice-accent-assignments.json``
(per voice, nearest book first, so a continued book keeps its inherited
characters' accents and only new characters are assigned around them). Each
voice's designed description and speech traits go with it as evidence:
one or the other often already names the right accent. A cue a person
chose (``voice-accents.json``), or an accent named in vocal traits they
set (``voice-traits.json``), always wins and is sent as fixed; ``accent_cues``
falls back to a description's accent only for voices with no assignment.

Renders assign voices that have no accent yet before planning
(``prepare_accents``), through the book's request cache; ``audiobook-render
accents BOOK`` reassigns the book, with guidance, as a fresh run.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import uuid

from .accent_prompts import ASSIGN, NOTES, REVISE


ASSIGNMENTS_FILENAME = "voice-accent-assignments.json"
GUIDANCE_FILENAME = "voice-accent-guidance.json"
GUIDANCE_FORMAT = "audiobook-book-accent-guidance"
GROUPS_FILENAME = "voice-accent-groups.json"
GROUPS_FORMAT = "audiobook-book-accent-groups"
MAX_ORIGIN = 80
MAX_WHY = 200
MAX_EXAMPLES = 6
MAX_NOTES = 2000
MAX_INTRO = 2
INTRO_BEFORE = 80
INTRO_AFTER = 240
MAX_ALIASES = 8
MAX_NAMES = 240
MAX_QUOTE = 160
MAX_ABOUT = 500
SPOKEN_KINDS = {"dialogue", "thought", "letter"}


def _validate_assigned(value):
    # An empty assignment records that the model gave this voice no accent.
    if value != "":
        from .speech_traits import validate_accent
        validate_accent(value)


def _setting():
    from .book_voice_settings import VoiceSetting
    return VoiceSetting(
        filename=ASSIGNMENTS_FILENAME, record_format="audiobook-book-voice-accent-assignments",
        field="assigned_accents", validate=_validate_assigned,
        invalid_message="Invalid book voice accent assignments, or they belong to another book/series",
        unknown_voice_message="Book voice accent assignments must use voice IDs from this book",
        busy_message="Book voice accent assignments are being saved; retry after the current save")


def assigned_accents(bundle, config, *, own=False):
    """Each voice's assigned accent cue ("" = the model gave none), nearest book first, or this book's own."""
    from .book_voice_settings import nearest_book_settings, read_book_setting
    from .book_context import resolve_book_config
    if own:
        config = resolve_book_config(bundle, config)
        return read_book_setting(_setting(), bundle, config["voices"]["series_id"])
    return nearest_book_settings(_setting(), bundle, config)


def save_assignments(bundle, config, accents, *, replace=False):
    from .book_context import resolve_book_config
    from .book_voice_settings import save_book_settings
    save_book_settings(_setting(), bundle, resolve_book_config(bundle, config), accents, replace=replace)


def guidance(bundle, config):
    """The nearest book's saved accent guidance, or ""."""
    from .book_context import resolve_book_config
    from .common import read_json, regular_file
    config = resolve_book_config(bundle, config)
    for directory in [bundle["directory"], *config.get("_voice_parents", [])]:
        path = Path(directory) / GUIDANCE_FILENAME
        if path.exists() or path.is_symlink():
            record = read_json(regular_file(path))
            notes = record.get("notes") if isinstance(record, dict) else None
            if (not isinstance(record, dict) or set(record) != {"format", "version", "notes"}
                    or record.get("format") != GUIDANCE_FORMAT or record.get("version") != 1
                    or not isinstance(notes, str) or len(notes) > MAX_NOTES):
                raise ValueError(f"Invalid accent guidance file: {path}")
            return notes
    return ""


def save_guidance(bundle, notes):
    from .common import write_json
    write_json(Path(bundle["directory"]) / GUIDANCE_FILENAME,
               {"format": GUIDANCE_FORMAT, "version": 1, "notes": notes})


def _origin_key(origin):
    return " ".join(str(origin).casefold().split())


def read_groups(directory, book_id):
    """A book's saved accent groups: [{"origin", "accent", "why", "voice_ids", "names"}], or []."""
    from .common import read_json, regular_file
    path = Path(directory) / GROUPS_FILENAME
    if not (path.exists() or path.is_symlink()):
        return []
    record = read_json(regular_file(path))
    groups = record.get("groups") if isinstance(record, dict) else None
    if (not isinstance(record, dict) or set(record) != {"format", "version", "book_id", "groups"}
            or record.get("format") != GROUPS_FORMAT or record.get("version") != 1
            or record.get("book_id") != book_id or not isinstance(groups, list)
            or not all(isinstance(group, dict) and set(group) == {"origin", "accent", "why", "voice_ids", "names"}
                       and all(isinstance(group[key], str) for key in ("origin", "accent", "why"))
                       and all(isinstance(group[key], list) and all(isinstance(item, str) for item in group[key])
                               for key in ("voice_ids", "names"))
                       for group in groups)):
        raise ValueError(f"Invalid accent groups file: {path}")
    return groups


def save_groups(bundle, groups, *, replace=False):
    """Save this book's accent groups, merging new members into a saved origin unless ``replace``."""
    from .common import write_json
    book_id = bundle["source"]["book"]["id"]
    saved = [] if replace else read_groups(bundle["directory"], book_id)
    by_origin = {_origin_key(group["origin"]): group for group in saved}
    for group in groups:
        current = by_origin.get(_origin_key(group["origin"]))
        if current is None:
            current = {"origin": group["origin"], "accent": group["accent"], "why": group.get("why", ""),
                       "voice_ids": [], "names": []}
            saved.append(current)
            by_origin[_origin_key(group["origin"])] = current
        for voice_id, name in zip(group["voice_ids"], group["names"]):
            if voice_id not in current["voice_ids"]:
                current["voice_ids"].append(voice_id)
                current["names"].append(name)
    write_json(Path(bundle["directory"]) / GROUPS_FILENAME,
               {"format": GROUPS_FORMAT, "version": 1, "book_id": book_id, "groups": saved})


def series_origins(bundle, config, *, own=True):
    """Origins the series already decided, nearest book first: [{"origin", "accent", "why", "names"}].

    ``own=False`` leaves out this book's own groups (when it is being redone).
    """
    from .book_context import resolve_book_config
    from .common import read_json, regular_file
    config = resolve_book_config(bundle, config)
    books = [(bundle["directory"], bundle["source"]["book"]["id"])] if own else []
    for directory in config.get("_voice_parents", []):
        manifest = read_json(regular_file(Path(directory) / "manifest.json"))
        books.append((directory, manifest["book"]["id"]))
    origins, seen = [], set()
    for directory, book_id in books:
        for group in read_groups(directory, book_id):
            key = _origin_key(group["origin"])
            if key in seen or not group["accent"]:
                continue
            seen.add(key)
            origins.append({"origin": group["origin"], "accent": group["accent"], "why": group["why"],
                            "names": group["names"][:MAX_EXAMPLES]})
    return origins


def accent_cue(accent, terms=frozenset()):
    """A model's accent name as a cue ("Welsh" -> "Welsh accent"), or ValueError saying why it is unusable."""
    from .speech_traits import MAX_ACCENT_CHARACTERS, suggest_accent, validate_accent
    text = " ".join(re.sub(r"[\[\]<>{}\"]", " ", accent).split()) if isinstance(accent, str) else ""
    if not text:
        raise ValueError("every group needs an accent name")
    cue = text if text.casefold().endswith("accent") else f"{text} accent"
    if len(cue.split()) > 6 or len(cue) > MAX_ACCENT_CHARACTERS:
        raise ValueError(f"accent {text!r} is too long: give a real-world accent name of at most 5 words")
    # The same rule as a description's accent: a real place or people, not the book's own.
    if suggest_accent("a " + cue, terms).casefold() != cue.casefold():
        raise ValueError(f"accent {text!r} must be a real-world accent from Earth, never a name from the book")
    validate_accent(cue)
    return cue


def _clip(text):
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_QUOTE else text[:MAX_QUOTE - 1].rsplit(" ", 1)[0] + "…"


def _introductions(book, names, limit):
    """The first paragraphs of ``book`` naming the character, each clipped around the mention."""
    names = sorted({name for name in names if len(name) >= 3 and any(letter.isupper() for letter in name)},
                   key=len, reverse=True)
    if not names:
        return []
    pattern = re.compile(r"(?<![\w'’-])(?:" + "|".join(map(re.escape, names)) + r")(?![\w'’-])")
    found = []
    for block in (book.get("source") or {}).get("blocks", []):
        match = pattern.search(block.get("text", ""))
        if match is None:
            continue
        text = block["text"]
        left = max(0, match.start() - INTRO_BEFORE)
        if left:
            space = text.find(" ", left, match.start())
            left = space + 1 if space != -1 else left
        right = min(len(text), match.end() + INTRO_AFTER)
        if right < len(text):
            space = text.rfind(" ", match.end(), right)
            right = space if space != -1 else right
        found.append(("…" if left else "") + " ".join(text[left:right].split()) + ("…" if right < len(text) else ""))
        if len(found) == limit:
            break
    return found


def voice_entries(bundle, parents=()):
    """What the story says about each voice that speaks.

    ``{voice_id: {"label", "names", "about", "intro", "line"}}``: the
    character's names and aliases, the analysis's description (and the one
    from the earliest book they appear in, when it differs), the narration
    where the series first mentions them (where a story usually says who
    they are and where they come from; ``parents`` are earlier books' bundles,
    nearest first), and one of their own lines as a dialect clue. Derived
    voices (a character's later state) are left out; they take their main
    voice's accent.
    """
    cast = bundle["cast"]
    derived = {voice["id"] for voice in cast["voices"] if isinstance(voice.get("derived_from"), dict)}
    used, spoken = set(), {}
    for chapter in bundle.get("chapters", []):
        for row in chapter.get("rows", []):
            voice_id = row.get("voice_id")
            if not voice_id:
                continue
            used.add(voice_id)
            if voice_id not in spoken and row.get("kind") in SPOKEN_KINDS and len(row.get("text", "").split()) >= 4:
                spoken[voice_id] = _clip(row["text"])
    earliest_first = [*reversed(list(parents)), bundle]
    entries = {}
    for profile in cast["voices"]:
        voice_id = profile["id"]
        if voice_id in derived or voice_id not in used:
            continue
        characters = [character for character in cast["characters"] if character.get("voice_id") == voice_id]
        roles = [role for role in cast["roles"] if role.get("voice_id") == voice_id]
        names = [character["name"] + (f" (also {', '.join(character['aliases'][:MAX_ALIASES])})"
                                      if character.get("aliases") else "") for character in characters]
        narrates = any(role.get("kind") == "narration" for role in roles)
        label = ", ".join(character["name"] for character in characters)
        if not names:
            # A voice with no character: the author reading front/back matter, or a storyteller narrator.
            from epub_tts_narrator.narration_identity import author_narrator_name
            authors = list(dict.fromkeys(name for role in roles if role.get("kind") == "narration"
                                         and (name := author_narrator_name(role, bundle["source"]["book"]))))
            if authors:
                label = ", ".join(authors) + " (author)"
            elif narrates:
                label = "Narrator (the story's storyteller, not a character)"
            else:
                label = roles[0]["id"] if roles else voice_id
            names = [label]
        about = "; ".join(character.get("description", "") for character in characters if character.get("description"))
        if not about:
            about = "; ".join(role.get("description", "") for role in roles if role.get("description"))
        intro, first_about = [], ""
        for book in earliest_first:
            here = [character for character in book["cast"]["characters"] if character.get("voice_id") == voice_id]
            if not here:
                continue
            if not first_about:
                first_about = "; ".join(character.get("description", "") for character in here
                                        if character.get("description"))
            words = [name for character in here for name in [character["name"], *character.get("aliases", [])]]
            intro += _introductions(book, words, MAX_INTRO - len(intro))
            if len(intro) >= MAX_INTRO:
                break
        if first_about and first_about != about:
            about = f"{first_about} (when first met); now: {about}" if about else first_about
        if narrates and characters:
            about += "; also narrates the book"
        if not intro:
            intro = [_clip(evidence["quote"]) for character in characters for evidence in character.get("evidence", [])
                     if isinstance(evidence, dict) and isinstance(evidence.get("quote"), str)][:MAX_INTRO]
        entries[voice_id] = {"label": label[:120], "names": ", ".join(names)[:MAX_NAMES], "about": about[:MAX_ABOUT],
                             "intro": intro, "line": spoken.get(voice_id, "")}
    return entries


def parent_bundles(config):
    """Earlier books' bundles, nearest first (for where a returning character was introduced)."""
    from .bundle import load_bundle
    return [load_bundle(path) for path in config.get("_voice_parents", [])]


MAX_VOICE = 300
MAX_TRAITS = 200


def voice_clues(bundle, config=None, *, store=None):
    """Each voice's designed sound: {voice_id: (description, traits)}.

    The description that designed its saved recording (else its profile
    description), and the vocal traits a person set, else the analysis's
    speech traits. Either often already names the accent.
    """
    from ..speech_traits import speech_traits
    from .speech_traits import designed_description, speech_trait_overrides
    chosen = speech_trait_overrides(bundle, config) if config is not None else {}
    clues = {}
    for profile in bundle["cast"]["voices"]:
        try:
            analysed = speech_traits(profile)
        except ValueError:
            analysed = ""
        traits = chosen.get(profile["id"]) or analysed
        clues[profile["id"]] = (" ".join(designed_description(store, profile).split())[:MAX_VOICE],
                                " ".join(traits.split())[:MAX_TRAITS])
    return clues


def request_assignment(bundle, requests, voice_ids, fixed, *, notes="", terms=frozenset(), clues=None,
                       entries=None, origins=(), keep=None, revise=None):
    """One request assigning ``voice_ids`` around ``fixed`` ({voice_id: cue}).

    ``clues`` ({voice_id: (voice description, traits)}, ``voice_clues``) go
    with each voice. ``origins`` (``series_origins``) are the series' earlier
    decisions: a group reusing one must keep its accent. ``keep``
    ({voice_id: cue}, voices among ``voice_ids``) must come back with exactly
    that accent, so only their grouping and reasons are asked for.
    ``revise`` = (groups, instructions) asks for that plan changed as the
    person's instructions say. Returns
    {"groups": [{"origin", "accent", "why", "voice_ids", "names"}],
    "accents": {voice_id: cue}}. A voice the reply leaves out gets no accent.
    """
    keep = keep or {}
    entries = entries if entries is not None else voice_entries(bundle)
    clues = clues if clues is not None else voice_clues(bundle)
    short, rows, fixed_rows = {}, [], []
    for voice_id in voice_ids:
        short[f"v{len(short) + 1}"] = voice_id
        entry = entries[voice_id]
        description, traits = clues.get(voice_id, ("", ""))
        rows.append([f"v{len(short)}", entry["names"], entry["about"], entry["intro"], entry["line"],
                     description, traits])
    for voice_id, cue in fixed.items():
        if voice_id in entries:
            fixed_rows.append([f"f{len(fixed_rows) + 1}", entries[voice_id]["names"], cue])
    payload = {"mode": "accent_assignment", "v": 4, "book": bundle["source"]["book"]["title"], "voices": rows}
    if fixed_rows:
        payload["fixed"] = fixed_rows
    decided = {_origin_key(item["origin"]): item["accent"] for item in origins}
    if origins:
        payload["origins"] = [[item["origin"], item["accent"], item["why"], item["names"]] for item in origins]
    ids = {voice_id: short_id for short_id, voice_id in short.items()}
    if keep:
        payload["keep"] = [[ids[voice_id], cue] for voice_id, cue in keep.items() if voice_id in ids]
    system = ASSIGN
    if notes:
        system, payload = system + NOTES, {**payload, "notes": notes}
    if revise is not None:
        plan, instructions = revise
        system = system + REVISE
        payload = {**payload, "revise": instructions,
                   "plan": [[group["origin"], group["accent"], group.get("why", ""),
                             [ids[voice_id] for voice_id in group["voice_ids"] if voice_id in ids]]
                            for group in plan]}

    def validate(response):
        groups = response.get("g") if isinstance(response, dict) else None
        if not isinstance(groups, list):
            raise ValueError('response must be {"g":[[origin,accent,why,[id,...]]]}')
        accents, found, problems = {}, [], []
        for group in groups:
            if not (isinstance(group, list) and len(group) >= 3 and isinstance(group[0], str)
                    and isinstance(group[-1], list)):
                continue
            why = group[2] if len(group) >= 4 and isinstance(group[2], str) else ""
            try:
                cue = accent_cue(group[1], terms)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            origin = " ".join(group[0].split())[:MAX_ORIGIN]
            series_accent = decided.get(_origin_key(origin))
            if series_accent and series_accent.casefold() != cue.casefold():
                problems.append(f"origin {origin!r} already has the accent {series_accent!r} in this series; "
                                "keep it, or name a different origin")
                continue
            members = [short[item] for item in group[-1] if isinstance(item, str) and item in short
                       and short[item] not in accents]
            for voice_id in members:
                if voice_id in keep and keep[voice_id].casefold() != cue.casefold():
                    problems.append(f"voice {ids[voice_id]} must keep the accent {keep[voice_id]!r}")
            for voice_id in members:
                accents[voice_id] = cue
            if members:
                found.append({"origin": origin, "accent": cue, "why": " ".join(why.split())[:MAX_WHY],
                              "voice_ids": members, "names": [entries[voice_id]["label"] for voice_id in members]})
        if problems:
            raise ValueError("; ".join(dict.fromkeys(problems)))
        return {"groups": found, "accents": accents}

    return requests.run(system, payload, validate)


def _requests(bundle, config_path, cache, *, progress):
    from epub_tts_narrator.config import load_config
    from epub_tts_narrator.pipeline import CachedRequests
    from epub_tts_narrator.provider import OpenAICompatibleClient
    analysis = load_config(config_path)
    if not analysis["provider"]["model"]:
        return None
    client = OpenAICompatibleClient(**analysis["provider"])
    progress(f"Assigning accents with {client.model} at {client.base_url}; character names, descriptions and a "
             "few of their lines are sent to this endpoint.")
    settings = analysis["analysis"]
    return CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                          max_request_chars=settings["max_request_chars"], progress=progress)


def _book_cache(bundle, config_path):
    from epub_tts_narrator.analysis_progress import analysis_cache_directory
    from epub_tts_narrator.config import load_config
    return analysis_cache_directory(Path(bundle["directory"]), load_config(config_path)["analysis"]["cache_dir"])


def _fixed(bundle, config, entries, *, own_only=False):
    """Decided accents the request must keep: a person's cues (or an accent in traits they set), and existing assignments."""
    from .speech_traits import _book_terms, accent_overrides, speech_trait_overrides, trait_accent
    chosen = accent_overrides(bundle, config, include_cleared=True)
    for voice_id, traits in speech_trait_overrides(bundle, config).items():
        accent = trait_accent(traits, _book_terms(bundle)) if traits else ""
        if accent and voice_id not in chosen:
            chosen[voice_id] = accent
    assigned = assigned_accents(bundle, config)
    if own_only:
        # Reassigning this book keeps only what earlier books decided.
        own = assigned_accents(bundle, config, own=True)
        assigned = {voice_id: value for voice_id, value in assigned.items() if voice_id not in own}
    fixed = {voice_id: chosen[voice_id] or "" for voice_id in entries if voice_id in chosen}
    fixed.update({voice_id: value for voice_id, value in assigned.items()
                  if voice_id in entries and voice_id not in fixed})
    return chosen, fixed


def prepare_accents(bundle, config, config_path, *, notify=print):
    """Before a render plans: assign an accent to every voice that has none yet (cached, resumable)."""
    from .book_context import resolve_book_config
    from .speech_traits import _book_terms
    config = resolve_book_config(bundle, config)
    if not config.get("casting", {}).get("assign_accents", True):
        return None
    entries = voice_entries(bundle, parent_bundles(config))
    chosen, fixed = _fixed(bundle, config, entries)
    pending = [voice_id for voice_id in entries if voice_id not in fixed]
    if not pending:
        return None
    requests = _requests(bundle, config_path, _book_cache(bundle, config_path), progress=notify)
    if requests is None:
        notify("Accent assignment skipped: set provider.model in the TOML configuration so voices get accents "
               "(set casting.assign_accents = false to stop asking).")
        return None
    notify(f"Assigning accents to {len(pending)} voice(s) without one, around {sum(1 for v in fixed.values() if v)} "
           "decided accent(s).")
    from .book_context import voice_store
    result = request_assignment(bundle, requests, pending, {k: v for k, v in fixed.items() if v},
                                notes=guidance(bundle, config), terms=_book_terms(bundle),
                                clues=voice_clues(bundle, config, store=voice_store(bundle, config)),
                                entries=entries, origins=series_origins(bundle, config))
    save_groups(bundle, result["groups"])
    save_assignments(bundle, config, {voice_id: result["accents"].get(voice_id, "") for voice_id in pending})
    _report(result, entries, notify, series_origins(bundle, config, own=False))
    return result


def _report(result, entries, notify, origins=()):
    inherited = {_origin_key(item["origin"]) for item in origins}
    for group in result["groups"]:
        names = ", ".join(entries[voice_id]["label"] for voice_id in group["voice_ids"])
        series = " [as earlier in the series]" if _origin_key(group["origin"]) in inherited else ""
        notify(f"  {group['accent']} ({group['origin']}){series}: {names}")
        if group.get("why"):
            notify(f"      why: {group['why']}")


def run_accents_command(bundle, config, args, config_path):
    """``audiobook-render accents BOOK``: review this book's accent plan, change it, then save.

    A book with a saved plan opens it for editing and asks the model nothing
    until a person asks (a revision, a new plan, or ``--instructions``); a
    book without one gets a proposal first. On a terminal the plan opens in
    the full-screen accent screen (``accent_tui.py``); elsewhere it is
    printed, and a terminal can save it, type instructions for the model to
    revise it, ask for a new plan, or leave it.
    """
    from .accent_review import AccentPlanner, ask_line, report
    planner = AccentPlanner(bundle, config, config_path, instructions=args.instructions,
                            keep=getattr(args, "keep", False), resume=args.resume,
                            progress=lambda message: print(message, file=sys.stderr))
    print(f"Accent assignment: {len(planner.pending)} voice(s) to assign; {len(planner.fixed)} kept (cues or trait "
          f"accents a person chose, and accents inherited from earlier books); {len(planner.origins)} origin(s) "
          f"decided earlier in the series; guidance={'yes' if planner.base_notes else 'no'}"
          + (f"; keeping this book's {len(planner.keep)} current accent(s)" if planner.keep else "") + ".")
    if not planner.pending:
        print("Every voice's accent is already decided; nothing to assign.")
        return 0
    # --keep records reasons for accents saved before reasons were, so it asks the model.
    saved = not planner.keep and planner.load_saved()
    if saved:
        print(f"Opening this book's saved plan ({len(planner.groups)} group(s)); the model is asked only when "
              "you ask for a revision or a new plan" + (", starting with --instructions." if args.instructions
                                                        else ".") +
              (f" {len(planner.unassigned())} character(s) added since have no accent yet."
               if planner.unassigned() else ""))
    from epub_tts_narrator.tui_support import wants_tui
    if not args.apply and wants_tui(args):
        from .accent_tui import AccentApp
        from .accent_review import AccentController
        controller = AccentController(planner)
        try:
            AccentApp(controller, title=bundle["source"]["book"]["title"],
                      instructions=args.instructions if saved else None).run()
        finally:
            controller.close()
        print(f"Saved accents for {len(planner.pending)} voice(s)." if planner.saved else "Nothing saved.")
        return 0
    if not saved:
        planner.propose()
    elif args.instructions:
        planner.revise(args.instructions)
    elif args.apply:
        report(planner)
        print("The saved plan is unchanged; nothing to save. Pass --instructions to revise it.")
        return 0
    report(planner)
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    while not args.apply:
        if not interactive:
            print("Report only: nothing saved. Run it in a terminal to review it, or pass --apply.")
            return 0
        choice, instructions = ask_line()
        if choice == "leave":
            print("Nothing saved.")
            return 0
        if choice == "save":
            break
        if choice == "new":
            planner.propose()
        else:
            planner.revise(instructions)
        report(planner)
    count = planner.save()
    print(f"Saved accents for {count} voice(s) in {ASSIGNMENTS_FILENAME}, with each group's origin and reason in "
          f"{GROUPS_FILENAME}" + (f" and the guidance in {GUIDANCE_FILENAME}"
                                  if planner.explicit_guidance is not None or planner.revisions else "") + ".")
    return 0
