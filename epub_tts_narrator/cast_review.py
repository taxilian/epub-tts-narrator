"""Rerunning ``analyze BOOK`` looks for cast entries that are one person, and merges them.

Pairs are found three ways: names that overlap (``narrator_read`` rule: a
shared name, a shared first name, one name inside the other), one whole-cast
request asking which entries may be one person under different names (an
epithet before a reveal, a title, a disguise), and a person's character notes
(``renderer/character_notes``) giving a later name that another entry has.
Each pair is checked against the book's text with the first analysis's
same-person check (``narrator_prompts.MERGE_CHECK``: "y" only when the text
shows they are one person).

A pair the text shows is one person is merged at once through
``merge-characters``' plan and publication (``character_merge_command``:
the earlier book's entry, else the one with a voice, else the one with more
lines is kept; backups, later books follow) when neither entry has a voice
yet. When either has a voice, or only a person's notes say so, the merge
would change a voice a person has, so it is shown and asked about in a
terminal or the run shell ([M] merge, [K] keep apart, [L] decide later);
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
NO_VOICE = "no accepted voice yet"


def _names(row):
    return [name for name in [row["name"], *row.get("aliases", [])] if name and name.strip()]


def _sex(document, row):
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    first = (voices.get(row["voice_id"], {}).get("description") or "").split(",")[0].split()
    return first[0] if first and first[0] in {"Male", "Female", "Androgynous"} else ""


def overlap_pairs(document, earlier):
    """Pairs of entries whose names overlap; two different earlier-book characters are never paired."""
    from .narrator_ledger import _norm
    rows = document["cast"]["characters"]
    pairs = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["id"] in earlier and right["id"] in earlier:
                continue
            a = {_norm(name) for name in _names(left)} - {""}
            b = {_norm(name) for name in _names(right)} - {""}
            first = ({name.split()[0] for name in a if " " in name} & b
                     or {name.split()[0] for name in b if " " in name} & a)
            contained = any(set(x.split()) < set(y.split()) or set(y.split()) < set(x.split())
                            for x in a for y in b)
            if a & b or first or contained:
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


def _context(document, row, other, numbers):
    """Paragraphs naming each entry, paragraphs each speaks, and paragraphs naming both."""
    from .character_merge import character_roles
    blocks = document["source"]["blocks"]
    by_id = {block["id"]: block for block in blocks}
    found = []
    patterns = []
    for entry in (row, other):
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in _names(entry)) + r")\b", re.I)
        patterns.append(pattern)
        roles = {role["id"] for role in character_roles(document, entry["id"])}
        spoken = list(dict.fromkeys(segment["block_id"] for segment in document["segments"]
                                    if segment.get("role_id") in roles
                                    or any(member["role_id"] in roles
                                           for member in segment.get("simultaneous_speakers", []))))
        found += [block for block in blocks if pattern.search(block["text"])][:2]
        found += [by_id[block_id] for block_id in spoken[:2]]
    found += [block for block in blocks if all(pattern.search(block["text"]) for pattern in patterns)][:2]
    unique = list({block["id"]: block for block in found}.values())
    return sorted(unique, key=lambda block: numbers[block["id"]])


def nominate(document, requests, notes):
    """The whole-cast request: pairs of entry IDs that may be one person."""
    from .rerun_review_prompts import CAST_REVIEW_VERSION, NOMINATE
    from .character_merge import first_line
    rows = document["cast"]["characters"]
    if len(rows) < 2:
        return []
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

    def validate(response):
        rows_ = response.get("r") if isinstance(response, dict) else None
        if not isinstance(rows_, list):
            raise ValueError('response must be {"r":[[id,id]]}')
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
    return requests.run(NOMINATE, payload, validate)


def check_pairs(document, requests, pairs, *, workers=1):
    """``{(a, b): "y"|"n"|"?"}`` from the first analysis's same-person check, with each pair's paragraphs."""
    from .narrator_prompts import MERGE_CHECK
    from .narrator_read import PROMPT_VERSION, _is, _item_row, _numbers, _parallel, _render_blocks, _unwrap
    if not pairs:
        return {}
    numbers = _numbers(document["source"])
    rows = {row["id"]: row for row in document["cast"]["characters"]}

    def line(row):
        return "|".join([row["id"], row["name"], ";".join(row.get("aliases", [])), _sex(document, row),
                         " ".join(row.get("description", "").split())])
    items = [{"id": f"m{index}", "a": line(rows[a]), "b": line(rows[b]),
              "text": _render_blocks(_context(document, rows[a], rows[b], numbers), numbers), "_pair": (a, b)}
             for index, (a, b) in enumerate(pairs, 1)]
    batches = [items[start:start + VERIFY_BATCH] for start in range(0, len(items), VERIFY_BATCH)]

    def run(batch):
        payload = {"mode": "narrator_duplicate_check", "v": PROMPT_VERSION,
                   "items": [{key: value for key, value in item.items() if not key.startswith("_")} for item in batch]}
        expected = {item["id"] for item in batch}

        def validate(response):
            answers = _unwrap(response.get("r")) if isinstance(response, dict) else None
            if not isinstance(answers, list):
                raise ValueError('response must be {"r":[[item_id,"y"|"n"|"?"]]}')
            result = {}
            for row in answers:
                row = _item_row(row, "m")
                if isinstance(row, list) and len(row) >= 2 and _is(row[0], expected) and _is(row[1], {"y", "n", "?"}):
                    result[row[0]] = row[1]
            missing = sorted(expected - set(result))
            if missing:
                raise ValueError(f"missing or invalid verdicts for items: {', '.join(missing[:20])}")
            return result
        return requests.run(MERGE_CHECK, payload, validate)
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


def _renderer_context(directory, config_path):
    """``{"notes", "store"}``: the person's character notes and the book's voice store, or ``None`` when unreadable."""
    try:
        from .renderer.book_context import resolve_book_config, voice_store
        from .renderer.bundle import load_bundle
        from .renderer.character_notes import character_notes
        from .renderer.config import load_config
        bundle = load_bundle(directory)
        config = resolve_book_config(bundle, load_config(config_path, require_verifier=False, require_engine=False))
        return {"notes": character_notes(bundle, config), "store": voice_store(bundle, config)}
    except (ValueError, OSError, ImportError) as error:
        print(f"Cast review: this book's voices could not be read ({type(error).__name__}); every merge it "
              "finds is asked about instead of made.")
        return None


def review_cast(directory, config, config_path, client_factory, progress, *, interactive=None):
    """``analyze BOOK``: find and merge cast entries that are one person; True when the book changed."""
    from .analysis_directory import _completed_directory
    from .analysis_progress import analysis_cache_directory
    from .pipeline import CachedRequests
    from .renderer.character_merge_command import apply_merge, plan_merge
    directory = Path(directory).resolve()
    interactive = (sys.stdin.isatty() and sys.stdout.isatty()) if interactive is None else interactive
    document, registry, _ = _completed_directory(directory)
    if len(document["cast"]["characters"]) < 2:
        return False
    if not config["provider"]["model"]:
        raise ValueError("Rerunning analyze reviews the cast; set provider.model in epub-tts-narrator.toml, "
                         "or select a TOML file with --config")
    context = _renderer_context(directory, config_path)
    notes = context["notes"] if context else {}
    earlier = {row["id"] for book in registry["books"][:-1] for row in book["cast"]["characters"]}
    settings = config["analysis"]
    cache = analysis_cache_directory(directory, settings["cache_dir"])
    client = client_factory(config["provider"])
    print(f"Reviewing the cast ({len(document['cast']['characters'])} entries) for entries that are one person, "
          f"with {client.model} at {client.base_url}; cast lines and passages will be sent to this endpoint.",
          file=sys.stderr)
    requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                              max_request_chars=settings["max_request_chars"], progress=progress)
    from_notes = note_pairs(document, notes)
    pairs = list(dict.fromkeys([*overlap_pairs(document, earlier), *nominate(document, requests, notes),
                                *from_notes]))
    apart = kept_apart(directory)
    pairs = [pair for pair in pairs if _key(*pair) not in apart and not (pair[0] in earlier and pair[1] in earlier)]
    unique = list({_key(*pair): pair for pair in pairs}.values())
    verdicts = check_pairs(document, requests, unique, workers=min(5, settings["attribution_workers"]))
    reasons = {_key(*pair): text for pair, text in from_notes.items()}
    names = {row["id"]: row["name"] for row in document["cast"]["characters"]}
    changed, gone, unsure = False, {}, []
    for pair in unique:
        verdict, note = verdicts[pair], reasons.get(_key(*pair))
        if verdict != "y" and note is None:
            if verdict == "?":
                unsure.append(pair)
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
        why = "the book's text shows they are one person" if verdict == "y" else note
        automatic = (verdict == "y" and context is not None
                     and plan.kept_voice == NO_VOICE and plan.absorbed_voice == NO_VOICE)
        if not automatic:
            print(f"Cast review: {why}.\n{plan.text}")
            if not interactive:
                print(f"  Left as two entries: a voice would change. Rerun analyze in a terminal to decide, or run "
                      f"merge-characters {directory} {plan.absorbed['id']} {plan.kept['id']}.")
                continue
            answer = _ask("[M] merge them / [K] keep them apart / [L] decide later: ")
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
