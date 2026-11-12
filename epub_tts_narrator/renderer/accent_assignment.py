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

from .accent_prompts import ASSIGN, NOTES


ASSIGNMENTS_FILENAME = "voice-accent-assignments.json"
GUIDANCE_FILENAME = "voice-accent-guidance.json"
GUIDANCE_FORMAT = "audiobook-book-accent-guidance"
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
            label = "Narrator" if narrates else (roles[0]["id"] if roles else voice_id)
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
                       entries=None):
    """One request assigning ``voice_ids`` around ``fixed`` ({voice_id: cue}).

    ``clues`` ({voice_id: (voice description, traits)}, ``voice_clues``) go
    with each voice. Returns {"groups": [{"origin", "accent", "voice_ids"}],
    "accents": {voice_id: cue}}. A voice the reply leaves out gets no accent.
    """
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
    payload = {"mode": "accent_assignment", "v": 3, "book": bundle["source"]["book"]["title"], "voices": rows}
    if fixed_rows:
        payload["fixed"] = fixed_rows
    system = ASSIGN
    if notes:
        system, payload = system + NOTES, {**payload, "notes": notes}

    def validate(response):
        groups = response.get("g") if isinstance(response, dict) else None
        if not isinstance(groups, list):
            raise ValueError('response must be {"g":[[origin,accent,[id,...]]]}')
        accents, found, problems = {}, [], []
        for group in groups:
            if not (isinstance(group, list) and len(group) >= 3 and isinstance(group[0], str)
                    and isinstance(group[2], list)):
                continue
            try:
                cue = accent_cue(group[1], terms)
            except ValueError as exc:
                problems.append(str(exc))
                continue
            members = [short[item] for item in group[2] if isinstance(item, str) and item in short
                       and short[item] not in accents]
            for voice_id in members:
                accents[voice_id] = cue
            if members:
                found.append({"origin": " ".join(group[0].split())[:80], "accent": cue, "voice_ids": members})
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
                                entries=entries)
    save_assignments(bundle, config, {voice_id: result["accents"].get(voice_id, "") for voice_id in pending})
    _report(result, entries, notify)
    return result


def _report(result, entries, notify):
    for group in result["groups"]:
        names = ", ".join(entries[voice_id]["label"] for voice_id in group["voice_ids"])
        notify(f"  {group['accent']} ({group['origin']}): {names}")


def _ask_apply():
    while True:
        print("[A] save these accents / [L] leave the book unchanged: ", end="", flush=True)
        try:
            answer = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, OSError):
            answer = ""
        if answer == "":
            print("\nInput ended; leaving the book unchanged.")
            return False
        answer = answer.strip().casefold()
        if answer in {"a", "l", "q"}:
            return answer == "a"
        print("Please enter A or L.")


def run_accents_command(bundle, config, args, config_path):
    """``audiobook-render accents BOOK``: reassign the book's accents as a fresh run, with optional guidance."""
    from .book_context import resolve_book_config
    from .speech_traits import _book_terms, accent_cues
    config = resolve_book_config(bundle, config)
    instructions = args.instructions
    if instructions is not None and not 1 <= len(instructions) <= MAX_NOTES:
        raise ValueError(f"--instructions must contain from 1 to {MAX_NOTES} characters")
    notes = instructions if instructions is not None else guidance(bundle, config)
    entries = voice_entries(bundle, parent_bundles(config))
    chosen, fixed = _fixed(bundle, config, entries, own_only=True)
    pending = [voice_id for voice_id in entries if voice_id not in fixed]
    print(f"Accent assignment: {len(pending)} voice(s) to assign; {len(fixed)} kept (cues or trait accents a person "
          f"chose, and accents inherited from earlier books); guidance={'yes' if notes else 'no'}.")
    if not pending:
        print("Every voice's accent is already decided; nothing to assign.")
        return 0
    root = _book_cache(bundle, config_path) / "accent-assignment"
    run_id = args.resume if args.resume is not None else uuid.uuid4().hex
    cache = root / run_id
    if args.resume is None:
        cache.mkdir(parents=True, exist_ok=False)
        print(f"Fresh accent run: {run_id} (finish an interrupted run with --resume {run_id})", file=sys.stderr)
    elif re.fullmatch(r"[0-9a-f]{32}", run_id) is None or not cache.is_dir():
        raise ValueError("--resume must be an accent run ID printed by an earlier invocation for this book")
    requests = _requests(bundle, config_path, cache, progress=lambda message: print(message, file=sys.stderr))
    if requests is None:
        raise ValueError("Set provider.model in the TOML configuration to assign accents")
    from .book_context import voice_store
    result = request_assignment(bundle, requests, pending, {k: v for k, v in fixed.items() if v},
                                notes=notes, terms=_book_terms(bundle),
                                clues=voice_clues(bundle, config, store=voice_store(bundle, config)),
                                entries=entries)
    _report(result, entries, print)
    missing = [voice_id for voice_id in pending if voice_id not in result["accents"]]
    if missing:
        print(f"{len(missing)} voice(s) got no accent: " + ", ".join(entries[v]["label"] for v in missing[:8]))
    before = accent_cues(bundle, config)
    changed = sum(1 for voice_id in pending if result["accents"].get(voice_id, "") != before.get(voice_id, ""))
    print(f"{changed} voice(s) would change accent; their lines render again on the next render of this book.")
    write = args.apply or (sys.stdin.isatty() and sys.stdout.isatty())
    if not args.apply:
        if not write:
            print("Report only: nothing saved. Run it in a terminal to be asked, or pass --apply.")
            return 0
        if not _ask_apply():
            print("Nothing saved.")
            return 0
    save_assignments(bundle, config, {voice_id: result["accents"].get(voice_id, "") for voice_id in pending},
                     replace=True)
    if instructions is not None:
        save_guidance(bundle, instructions)
    print(f"Saved accents for {len(pending)} voice(s) in {ASSIGNMENTS_FILENAME}"
          + (f" and the guidance in {GUIDANCE_FILENAME}" if instructions is not None else "") + ".")
    return 0
