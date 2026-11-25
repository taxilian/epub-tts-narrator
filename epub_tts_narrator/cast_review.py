"""Rerunning ``analyze BOOK`` looks for cast entries that are one person, and merges them.

Pairs are found three ways: names that overlap (``narrator_read`` rule: a
shared name, a shared first name, one name inside the other), one whole-cast
request asking which entries may be one person under different names (an
epithet before a reveal, a title, a disguise), and a person's character notes
(``renderer/character_notes``) giving a later name that another entry has.
Notes given with ``analyze BOOK --instructions`` join that whole-cast request
(``rerun_review_prompts.NOMINATE_WITH_INSTRUCTIONS``): pairs they say are one
person are asked about like a character note's, and pairs they say are
different people are not merged in this run. Overlap means the same name or one name's words inside the other's; a form
of address ("your Grace") or a name standing only as a word about someone
else ("Silk's servant", "Cherek sleigh driver") is not one. Each pair is
checked against the book's text (``rerun_review_prompts.SAME_PERSON_CHECK``),
shown up to six of each entry's lines spread over the book: "y" needs an exact
quote showing they are one person and every line being theirs, a resemblance
(people, title, rank, occupation, description) is never evidence, and "p"
(an entry that mixes several people) is reported, never merged.

Characters only earlier books of the series have join the comparison
(``with_earlier_characters``): a character may return under another name the
analysis never matched (book 1's "First Voice" as book 2's "Hissing Shadow"),
and a person's notes about them, kept in the earlier book, are sent too. A
merge with one keeps the earlier book's character and takes it into this book.

A pair the text shows is one person is merged at once through
``merge-characters``' plan and publication (``character_merge_command``:
the earlier book's entry, else the one with a voice, else the one with more
lines is kept; backups, later books follow) when neither entry has a voice
yet. When either has a voice, or only a person's notes say so, the merge
would change a voice a person has, so it is shown and asked about in a
terminal or the run shell ([M] merge, [K] keep apart, [L] decide later; the
run shell shows both entries side by side, ``rerun_review_tui.MergeScreen``);
without a terminal it is listed and left. "Keep apart" is a person's
decision, saved beside the book in ``cast-review.json`` and never asked
again for those two entries. Model answers are cached in the book's request
cache, so a rerun with an unchanged cast asks nothing new.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REVIEW_FILENAME = "cast-review.json"
VERIFY_BATCH = 25
BATCH_CHARS = 60000
NO_VOICE = "no accepted voice yet"


def _names(row):
    return [name for name in [row["name"], *row.get("aliases", [])] if name and name.strip()]


def _sex(document, row):
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    first = (voices.get(row["voice_id"], {}).get("description") or "").split(",")[0].split()
    return first[0] if first and first[0] in {"Male", "Female", "Androgynous"} else ""


# A form of address ("your Grace", "my Lord") is said to many people; it never names one.
_ADDRESS = re.compile(r"^(?:your|her|his|my|our|their|thy|thine)\b", re.I)


def _person_names(row):
    return [name for name in _names(row) if not _ADDRESS.match(name.strip())]


def _describes_another(longer, shorter):
    """True when ``shorter`` (a name) stands in ``longer`` only as a word describing someone else.

    A possessive ("Silk's servant", "Actas' wife") or a name followed by a lowercase noun ("Cherek sleigh
    driver", "Temple guard") names another person: one belonging to, or from, the named one."""
    words = re.findall(r"\w+", shorter)
    if not words:
        return False
    match = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b(['’]s?(?!\w))?", longer, re.I)
    if match is None:
        return False
    if match.group(1):
        return True
    following = re.match(r"\W*(\w+)", longer[match.end():])
    return bool(following and following.group(1)[:1].islower())


def with_earlier_characters(document, registry):
    """``document`` for comparing only (never saved): its cast plus the characters only earlier books have.

    Each added entry's description says which book it is from and that this book's cast lacks it.
    """
    from .renderer.character_merge_command import earlier_characters
    added = earlier_characters(registry, document)
    if not added:
        return document
    voices = {}
    for book in registry["books"][:-1]:
        voices.update({voice["id"]: voice for voice in book["cast"]["voices"]})
    have = {voice["id"] for voice in document["cast"]["voices"]}
    characters, extra_voices = list(document["cast"]["characters"]), list(document["cast"]["voices"])
    for row, title in added:
        description = " ".join(row.get("description", "").split())
        characters.append(dict(row, description=f"from an earlier book ({title}), not in this book's cast"
                                                + (f": {description}" if description else "")))
        if row["voice_id"] in voices and row["voice_id"] not in have:
            have.add(row["voice_id"])
            extra_voices.append(voices[row["voice_id"]])
    return dict(document, cast=dict(document["cast"], characters=characters, voices=extra_voices))


def overlap_pairs(document, earlier):
    """Pairs of entries whose names overlap; two different earlier-book characters are never paired.

    The same name, or one name's words inside the other's ("Mara" and "Mara Venn", "General Varana"), make
    a pair; a form of address, or a name standing only as a word about someone else, does not."""
    from .narrator_ledger import _norm
    rows = document["cast"]["characters"]
    pairs = []

    def overlap(x, y):
        nx, ny = _norm(x), _norm(y)
        if not nx or not ny:
            return False
        if nx == ny:
            return True
        small, large, large_raw = (nx, ny, y) if len(nx.split()) < len(ny.split()) else (ny, nx, x)
        return set(small.split()) < set(large.split()) and not _describes_another(large_raw, small)
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["id"] in earlier and right["id"] in earlier:
                continue
            if any(overlap(x, y) for x in _person_names(left) for y in _person_names(right)):
                pairs.append((left["id"], right["id"]))
    return pairs


def note_pairs(document, notes):
    """Pairs where a person's notes give one entry a name another entry has: ``{(a, b): note text}``."""
    from .narrator_ledger import _norm
    by_voice = {row["voice_id"]: row for row in document["cast"]["characters"]}
    result = {}
    for voice_id, record in notes.items():
        row = by_voice.get(voice_id)
        if row is None:
            continue
        wanted = {_norm(name) for name in record.get("names", [])} - {""}
        for other in document["cast"]["characters"]:
            if other["id"] != row["id"] and wanted & {_norm(name) for name in _names(other)}:
                result[(row["id"], other["id"])] = f"your notes say {row['name']} is also called " + \
                    ", ".join(record.get("names", []))
    return result


LINES_SHOWN = 6


def _spread(values, count):
    """Up to ``count`` of ``values`` spread from the first to the last."""
    if len(values) <= count:
        return list(values)
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def _spoken_blocks(document, row):
    from .character_merge import character_roles
    roles = {role["id"] for role in character_roles(document, row["id"])}
    return list(dict.fromkeys(segment["block_id"] for segment in document["segments"]
                              if segment.get("role_id") in roles
                              or any(member["role_id"] in roles
                                     for member in segment.get("simultaneous_speakers", []))))


def _context(document, row, other, numbers):
    """Paragraphs naming each entry, paragraphs each speaks (spread over the book), and paragraphs naming both."""
    blocks = document["source"]["blocks"]
    by_id = {block["id"]: block for block in blocks}
    found = []
    patterns = []
    for entry in (row, other):
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in _names(entry)) + r")\b", re.I)
        patterns.append(pattern)
        spoken = _spoken_blocks(document, entry)
        found += [block for block in blocks if pattern.search(block["text"])][:2]
        found += [by_id[block_id] for block_id in _spread(spoken, LINES_SHOWN)]
    found += [block for block in blocks if all(pattern.search(block["text"]) for pattern in patterns)][:2]
    unique = list({block["id"]: block for block in found}.values())
    return sorted(unique, key=lambda block: numbers[block["id"]])


def nominate(document, requests, notes, instructions=None):
    """The whole-cast request: pairs of entry IDs that may be one person.

    With a person's ``instructions``: ``(pairs, same, apart)``, adding the pairs the notes say are one
    person and the pairs they say are different people."""
    from .rerun_review_prompts import CAST_REVIEW_VERSION, NOMINATE, NOMINATE_WITH_INSTRUCTIONS
    from .character_merge import first_line
    rows = document["cast"]["characters"]
    if len(rows) < 2:
        return ([], [], []) if instructions else []
    labels = {f"c{index}": row for index, row in enumerate(rows, 1)}
    label_of = {row["id"]: label for label, row in labels.items()}
    cast = "\n".join("|".join([label, row["name"], ";".join(row.get("aliases", [])), _sex(document, row),
                               " ".join(row.get("description", "").split()), first_line(document, row["id"]) or ""])
                     for label, row in labels.items())
    payload = {"mode": "cast_review_nominate", "v": CAST_REVIEW_VERSION, "cast": cast}
    by_voice = {row["voice_id"]: row for row in rows}
    if notes:
        from .renderer.character_notes import model_row
        lines = []
        for voice_id, record in notes.items():
            if voice_id in by_voice and record:
                names, gender, origin, text = model_row(record)
                lines.append("|".join([label_of[by_voice[voice_id]["id"]], ";".join(names), gender, origin, text]))
        if lines:
            payload["notes"] = "\n".join(lines)
    if instructions:
        payload["instructions"] = instructions

    def pairs_of(rows_):
        pairs = []
        for row in rows_:
            if not (isinstance(row, list) and len(row) >= 2):
                raise ValueError("each pair must be [id,id]")
            left, right = (f"c{int(value)}" if type(value) is int or (isinstance(value, str) and value.isdigit())
                           else value for value in row[:2])
            if left not in labels or right not in labels or left == right:
                raise ValueError("each pair must name two different cast ids from the request")
            pairs.append((labels[left]["id"], labels[right]["id"]))
        return pairs

    def validate(response):
        rows_ = response.get("r", [] if instructions else None) if isinstance(response, dict) else None
        if not isinstance(rows_, list):
            raise ValueError('response must be {"r":[[id,id]]}')
        if not instructions:
            return pairs_of(rows_)
        extra = {key: response.get(key, []) for key in ("same", "apart")}
        if not all(isinstance(value, list) for value in extra.values()):
            raise ValueError('"same" and "apart" must be lists of [id,id] pairs')
        return pairs_of(rows_), pairs_of(extra["same"]), pairs_of(extra["apart"])
    return requests.run(NOMINATE_WITH_INSTRUCTIONS if instructions else NOMINATE, payload, validate)


def check_pairs(document, requests, pairs, *, workers=1):
    """``{(a, b): (verdict, quote)}``: "y" with the exact quote showing one person, "p" (an entry mixes
    several people), "n" or "?", from ``rerun_review_prompts.SAME_PERSON_CHECK``."""
    from .narrator_read import _is, _item_row, _numbers, _parallel, _render_blocks, _unwrap
    from .rerun_review_prompts import SAME_PERSON_CHECK, SAME_PERSON_VERSION
    if not pairs:
        return {}
    numbers = _numbers(document["source"])
    rows = {row["id"]: row for row in document["cast"]["characters"]}

    def line(row):
        return "|".join([row["id"], row["name"], ";".join(row.get("aliases", [])), _sex(document, row),
                         " ".join(row.get("description", "").split())])
    items = []
    for index, (a, b) in enumerate(pairs, 1):
        shown = _context(document, rows[a], rows[b], numbers)
        shown_ids = {block["id"] for block in shown}
        items.append({"id": f"m{index}", "a": line(rows[a]), "b": line(rows[b]),
                      "text": _render_blocks(shown, numbers),
                      "a_lines": [numbers[block_id] for block_id in _spoken_blocks(document, rows[a])
                                  if block_id in shown_ids],
                      "b_lines": [numbers[block_id] for block_id in _spoken_blocks(document, rows[b])
                                  if block_id in shown_ids],
                      "_pair": (a, b), "_text": {numbers[block["id"]]: block["text"] for block in shown}})
    batches, batch, size = [], [], 0
    for item in items:
        if batch and (len(batch) >= VERIFY_BATCH or size + len(item["text"]) > BATCH_CHARS):
            batches.append(batch)
            batch, size = [], 0
        batch.append(item)
        size += len(item["text"])
    if batch:
        batches.append(batch)

    def run(batch):
        payload = {"mode": "cast_review_same_person", "v": SAME_PERSON_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        by_id = {item["id"]: item for item in batch}
        # A quote may come from any paragraph the request showed.
        shown = {number: text for item in batch for number, text in item["_text"].items()}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,"y",N,quote]|[item_id,"p"|"n"|"?"]]}')
            result = {}
            for row in answers:
                row = _item_row(row, "m")
                if not (isinstance(row, list) and len(row) >= 2 and _is(row[0], set(by_id))
                        and _is(row[1], {"y", "p", "n", "?"})):
                    continue
                if row[1] != "y":
                    result[row[0]] = (row[1], "")
                    continue
                text = shown.get(row[2]) if len(row) >= 4 and type(row[2]) is int else None
                quote = row[3].strip() if len(row) >= 4 and isinstance(row[3], str) else ""
                if text is None or not quote or quote not in text:
                    raise ValueError(f'{row[0]}: "y" needs [item_id,"y",N,quote] with an exact quote from '
                                     f"paragraph N that shows they are one person")
                result[row[0]] = ("y", quote)
            missing = sorted(set(by_id) - set(result), key=lambda name: int(name[1:]))
            if missing:
                raise ValueError(f"missing or invalid verdicts for items: {', '.join(missing[:20])}")
            return result
        return requests.run(SAME_PERSON_CHECK, payload, validate)
    verdicts = {}
    for result in _parallel(run, batches, workers):
        verdicts.update(result)
    return {item["_pair"]: verdicts[item["id"]] for item in items}


def _key(a, b):
    return "|".join(sorted((a, b)))


def kept_apart(directory):
    path = Path(directory) / REVIEW_FILENAME
    if path.is_symlink() or not path.is_file():
        return set()
    record = json.loads(path.read_text(encoding="utf-8"))
    return {_key(*row["pair"]) for row in record.get("kept_apart", [])}


def keep_apart(directory, a, b, names):
    path = Path(directory) / REVIEW_FILENAME
    record = {"version": 1, "kept_apart": []}
    if path.is_file() and not path.is_symlink():
        record = json.loads(path.read_text(encoding="utf-8"))
    record["kept_apart"].append({"pair": sorted((a, b)), "names": names})
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ask(question):
    while True:
        print(question, end="", flush=True)
        try:
            answer = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, OSError):
            answer = ""
        if answer == "":
            print()
            return "l"
        answer = answer.strip().casefold()[:1]
        if answer in {"m", "k", "l"}:
            return answer
        print("Please enter M, K or L.")


def _decide(plan, why):
    """"m", "k" or "l": on the run shell's own screen when one is open, else on the line."""
    from .rerun_questions import NO_SCREEN, ask_on_screen, merge_view
    answer = ask_on_screen("MergeScreen", merge_view(plan, why))
    if answer is NO_SCREEN:
        return _ask("[M] merge them / [K] keep them apart / [L] decide later: ")
    return answer if answer in {"m", "k", "l"} else "l"


def _renderer_context(directory, config_path):
    """``{"notes", "store"}``: the person's character notes and the book's voice store, or ``None`` when unreadable."""
    try:
        from .renderer.book_context import resolve_book_config, voice_store
        from .renderer.bundle import load_bundle
        from .renderer.character_notes import character_notes
        from .renderer.config import load_config
        bundle = load_bundle(directory)
        config = resolve_book_config(bundle, load_config(config_path, require_verifier=False, require_engine=False))
        # Notes about earlier books' characters too: one may be in this book under another name.
        return {"notes": character_notes(bundle, config, earlier_voices=True), "store": voice_store(bundle, config)}
    except (ValueError, OSError, ImportError) as error:
        print(f"Cast review: this book's voices could not be read ({type(error).__name__}); every merge it "
              "finds is asked about instead of made.")
        return None


def review_cast(directory, config, config_path, client_factory, progress, *, interactive=None, instructions=None,
                recheck_cache=None):
    """``analyze BOOK``: find and merge cast entries that are one person; True when the book changed.

    ``recheck_cache``: a recheck's own request cache (``recheck.Recheck.cache``), so every answer is fresh."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .pipeline import CachedRequests
    from .renderer.character_merge_command import apply_merge, plan_merge
    directory = Path(directory).resolve()
    interactive = (sys.stdin.isatty() and sys.stdout.isatty()) if interactive is None else interactive
    saved, registry, _ = _completed_directory(directory)
    document = with_earlier_characters(saved, registry)
    if len(document["cast"]["characters"]) < 2:
        return False
    if not config["provider"]["model"]:
        raise ValueError("Rerunning analyze reviews the cast; set provider.model in epub-tts-narrator.toml, "
                         "or select a TOML file with --config")
    context = _renderer_context(directory, config_path)
    notes = context["notes"] if context else {}
    earlier = {row["id"] for book in registry["books"][:-1] for row in book["cast"]["characters"]}
    settings = config["analysis"]
    cache = recheck_cache or analysis_cache_directory(directory, settings["cache_dir"])
    client = client_factory(config["provider"])
    added = len(document["cast"]["characters"]) - len(saved["cast"]["characters"])
    print(f"Reviewing the cast ({len(saved['cast']['characters'])} entries"
          + (f", and {added} character(s) only earlier books have" if added else "")
          + ") for entries that are one person, "
          f"with {client.model} at {client.base_url}; cast lines and passages will be sent to this endpoint.",
          file=sys.stderr)
    requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    from_notes = note_pairs(document, notes)
    instructions = (instructions or "").strip() or None
    nominated, said_apart = nominate(document, requests, notes, instructions), set()
    if instructions:
        nominated, same, different = nominated
        names_ = {row["id"]: row["name"] for row in document["cast"]["characters"]}
        for a, b in same:
            from_notes.setdefault((a, b), f"your notes say {names_[a]} and {names_[b]} are one person")
        said_apart = {_key(a, b) for a, b in different}
        for a, b in different:
            print(f"Cast review: your notes say {names_[a]} and {names_[b]} are different people; "
                  f"they are not merged.")
    pairs = list(dict.fromkeys([*overlap_pairs(document, earlier), *nominated, *from_notes]))
    apart = kept_apart(directory) | said_apart
    pairs = [pair for pair in pairs if _key(*pair) not in apart and not (pair[0] in earlier and pair[1] in earlier)]
    unique = list({_key(*pair): pair for pair in pairs}.values())
    verdicts = check_pairs(document, requests, unique, workers=min(5, settings["attribution_workers"]))
    reasons = {_key(*pair): text for pair, text in from_notes.items()}
    names = {row["id"]: row["name"] for row in document["cast"]["characters"]}
    changed, gone, unsure, mixed = False, {}, [], []
    for pair in unique:
        (verdict, quote), note = verdicts[pair], reasons.get(_key(*pair))
        if verdict != "y" and note is None:
            if verdict == "?":
                unsure.append(pair)
            if verdict == "p":
                mixed.append(pair)
            continue
        a, b = (gone.get(value, value) for value in pair)
        if a == b:
            continue
        from .renderer.bundle import load_bundle
        try:
            plan = plan_merge(load_bundle(directory), None, a, b,
                              store=context["store"] if context else _UnknownVoices())
        except ValueError as error:
            print(f"Cast review: {names.get(a, a)} and {names.get(b, b)} may be one person, but they cannot be "
                  f"merged here: {error}")
            continue
        why = f"the book's text shows they are one person: “{quote}”" if verdict == "y" else note
        automatic = (verdict == "y" and context is not None
                     and plan.kept_voice == NO_VOICE and plan.absorbed_voice == NO_VOICE)
        if not automatic:
            print(f"Cast review: {why}.\n{plan.text}")
            if not interactive:
                print(f"  Left as two entries: a voice would change. Rerun analyze in a terminal to decide, or run "
                      f"merge-characters {directory} {plan.absorbed['id']} {plan.kept['id']}.")
                continue
            answer = _decide(plan, why)
            if answer == "k":
                keep_apart(directory, a, b, [plan.kept["name"], plan.absorbed["name"]])
                print(f"Kept {plan.kept['name']} and {plan.absorbed['name']} apart; this is not asked again.")
                continue
            if answer == "l":
                print("Left as two entries for now.")
                continue
        try:
            apply_merge(plan, progress=print)
        except (ValueError, OSError) as error:
            print(f"Cast review: {plan.absorbed['name']} and {plan.kept['name']} were left as two entries: {error}")
            continue
        print(f"Cast review: merged {plan.absorbed['name']} into {plan.name} ({why}; kept because {plan.reason}).")
        gone = {key: (plan.kept["id"] if value == plan.absorbed["id"] else value) for key, value in gone.items()}
        gone[plan.absorbed["id"]] = plan.kept["id"]
        changed = True
    for a, b in mixed:
        print(f"Cast review: some lines of {names[a]} or {names[b]} may be the other's, but not all of them, so "
              f"they are not merged. Check those lines with verify-attribution {directory} or review-ui.")
    for a, b in unsure:
        print(f"Cast review: {names[a]} and {names[b]} may be one person; the text does not say. To merge them: "
              f"merge-characters {directory} {a} {b}")
    if not changed:
        print("Cast review: no cast entries to merge.")
    return changed


class _UnknownVoices:
    """A voice store that cannot be read: every voice counts as chosen, so every merge is asked about."""

    def canonical_id(self, voice_id):
        return voice_id

    def load(self, voice_id):
        return {"inherited": False}
