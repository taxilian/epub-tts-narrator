"""``merge-characters``: a person says two cast entries are one character, and it is published.

Unlike voice sharing (``voice-aliases.json``, cast ``d``), which only reuses a
recording, this changes the book's identities: ``character_merge`` rewrites
the analysis, and this module plans the change across the book and every later
book that carries the absorbed entry, shows it, and publishes it through the
receipt-bound revision path (``analysis_revision.revise_directories`` scope
``character-merge``, with backups and descendant rebinding). Per-voice choices
the absorbed voice had (sharing, traits, accent cue and assignment, clip count,
audition script) are dropped with it; the kept entry's own choices stay. A
pending speaker-decision journal is restated for the merged cast.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys

from ..character_merge import (carried_voice, character_lines, character_roles, combined_names, first_line,
                               kept_records_for, merge_document, name_key, receipt, remap_decisions,
                               resolve_character, segment_speakers)
from ..pipeline import fingerprint


# Book-local per-voice choices: file -> (value field, what a person calls it).
_VOICE_SETTINGS = {
    "voice-traits.json": ("speech_traits", "speech traits"),
    "voice-accents.json": ("accents", "accent cue"),
    "voice-accent-assignments.json": ("assigned_accents", "assigned accent"),
    "voice-clips.json": ("reference_clips", "reference clip count"),
    "character-notes.json": ("character_notes", "your character notes"),
    "voice-accent-assignment-notes.json": ("notes_sha256", "notes an accent was assigned with"),
}
_ALIASES = "voice-aliases.json"
_GROUPS = "voice-accent-groups.json"
_AUDITIONS = "audition-scripts.json"


@dataclass
class BookChange:
    path: Path
    title: str
    original: dict
    candidate: dict
    sidecars: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    moved_lines: int = 0
    joined_lines: int = 0
    took_in: bool = False


@dataclass
class MergePlan:
    directory: Path
    paths: list
    hashes: dict
    kept: dict
    absorbed: dict
    name: str
    reason: str
    kept_lines: int
    absorbed_lines: int
    kept_voice: str
    absorbed_voice: str
    books: list
    earlier: list
    fixed: str = ""  # the earlier book the kept entry comes from, when the other is new: it cannot be swapped
    kept_first: str = ""
    absorbed_first: str = ""
    origins: dict = field(default_factory=dict)  # {character ID: the earliest book with it}, for the two entries

    @property
    def names(self):
        return combined_names(self.kept, self.absorbed)

    @property
    def text(self):
        return describe(self)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _regular(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{path.name} must be a regular, nonsymlink file")
    return path.is_file()


def _voice_state(store, voice_id):
    """("own" | "shared" | "none", detail) for one voice in this book's store."""
    if store is None:
        return "none", ""
    try:
        canonical = store.canonical_id(voice_id)
        selected = store.load(voice_id)
    except (ValueError, OSError) as error:
        return "none", str(error)
    if selected is None:
        return "none", ""
    if canonical != voice_id:
        return "shared", canonical
    return "own", "inherited" if selected.get("inherited") else ""


def _sidecar_changes(path, original, candidate, absorbed_voice, kept_voice):
    """This book's per-voice files with the absorbed voice dropped: (records, dropped labels, warnings)."""
    records, dropped, warnings = {}, [], []
    for name, (field_name, label) in _VOICE_SETTINGS.items():
        target = path / name
        if not _regular(target):
            continue
        record = _read(target)
        values = record.get(field_name) if isinstance(record, dict) else None
        if not isinstance(values, dict):
            raise ValueError(f"Invalid {name}; fix or remove it before merging characters")
        if absorbed_voice in values:
            updated = deepcopy(record)
            updated[field_name].pop(absorbed_voice)
            records[name] = updated
            dropped.append(label)
    target = path / _ALIASES
    if _regular(target):
        from .voice_bindings import validate_aliases
        record = _read(target)
        aliases = record.get("aliases") if isinstance(record, dict) else None
        if not isinstance(aliases, dict):
            raise ValueError(f"Invalid {_ALIASES}; fix or remove it before merging characters")
        updated = {}
        for source, destination in aliases.items():
            if source == absorbed_voice:
                dropped.append("voice sharing")
                continue
            if destination == absorbed_voice:
                if source == kept_voice:
                    warnings.append("the kept entry used the absorbed entry's recording; it goes back to its own voice")
                    continue
                destination = kept_voice
            if source != destination:
                updated[source] = destination
        if updated != aliases:
            try:
                validate_aliases(updated)
            except ValueError as error:
                raise ValueError(f"Merging would leave {target} with invalid voice sharing ({error}); "
                                 "change it with voice-aliases first") from None
            records[_ALIASES] = dict(record, aliases=updated)
    target = path / _GROUPS
    if _regular(target):
        record = _read(target)
        groups = record.get("groups") if isinstance(record, dict) else None
        if not isinstance(groups, list):
            raise ValueError(f"Invalid {_GROUPS}; fix or remove it before merging characters")
        if any(absorbed_voice in group.get("voice_ids", []) for group in groups):
            updated = deepcopy(record)
            for group in updated["groups"]:
                pairs = [(voice, name) for voice, name in zip(group["voice_ids"], group["names"])
                         if voice != absorbed_voice]
                group["voice_ids"], group["names"] = [voice for voice, _ in pairs], [name for _, name in pairs]
            records[_GROUPS] = updated
            dropped.append("accent group membership")
    target = path / _AUDITIONS
    if _regular(target):
        record = _read(target)
        if (isinstance(record, dict) and record.get("source_fingerprint") == fingerprint(original["source"])
                and record.get("cast_fingerprint") == fingerprint(original["cast"])):
            updated = deepcopy(record)
            updated["cast_fingerprint"] = fingerprint(candidate["cast"])
            if isinstance(updated.get("scripts"), dict) and updated["scripts"].pop(absorbed_voice, None) is not None:
                dropped.append("audition script")
            if isinstance(updated.get("extra_scripts"), dict):
                updated["extra_scripts"].pop(absorbed_voice, None)
            if isinstance(updated.get("replaced_voice_ids"), list):
                updated["replaced_voice_ids"] = [voice for voice in updated["replaced_voice_ids"]
                                                 if voice != absorbed_voice]
            records[_AUDITIONS] = updated
    return records, list(dict.fromkeys(dropped)), warnings


def _book_change(path, original, candidate, absorbed_id, *, took_in):
    record = candidate["provenance"]["character_merges"][-1]
    absorbed_roles = {role["id"] for role in character_roles(original, absorbed_id)}
    before = {segment["id"]: segment_speakers(segment) for segment in original["segments"]}
    moved = joined = 0
    for segment_id in record["revoiced_segments"]:
        speakers = before[segment_id]
        if any(role in absorbed_roles for role, _ in speakers):
            moved += 1
            if len(speakers) > 1 and not any(segment["id"] == segment_id and "simultaneous_speakers" in segment
                                             for segment in candidate["segments"]):
                joined += 1
    sidecars, dropped, warnings = _sidecar_changes(path, original, candidate, record["absorbed_voice_id"],
                                                   record["kept_voice_id"])
    return BookChange(path=path, title=original["source"]["book"].get("title") or path.name,
                      original=original, candidate=candidate, sidecars=sidecars, dropped=dropped,
                      warnings=warnings, moved_lines=moved, joined_lines=joined, took_in=took_in)


def earlier_characters(registry, document):
    """``[(character, book title)]``: characters only earlier books of the series have, nearest book first.

    A later book's entry may be one of them under another name (book 1's "First Voice" is
    book 2's "Hissing Shadow"), so a merge may name one; it is then taken into this book.
    """
    here = {row["id"] for row in document["cast"]["characters"]}
    found, seen = [], set()
    for book in reversed(registry["books"][:-1]):
        title = book["book"].get("title") or book["source_fingerprint"][:12]
        for row in book["cast"]["characters"]:
            if row["id"] not in here and row["id"] not in seen:
                seen.add(row["id"])
                found.append((row, title))
    return found


def _earlier_records(registry, character_id):
    """The nearest earlier book's records of ``character_id``, as a book without it takes them in."""
    for book in reversed(registry["books"][:-1]):
        character = next((row for row in book["cast"]["characters"] if row["id"] == character_id), None)
        if character is not None:
            voice = next(row for row in book["cast"]["voices"] if row["id"] == character["voice_id"])
            carried = {key: deepcopy(value) for key, value in character.items()
                       if key not in {"evidence", "series_identity"}}
            carried["evidence"] = []
            return {"character": carried, "voice": carried_voice(voice, series_id=registry["series_id"],
                                                                 source_fingerprint=book["source_fingerprint"])}
    raise ValueError(f"{character_id} is in no earlier book of this series")


def _resolve(document, registry, text):
    """The character a person means: this book's, else one only an earlier book has."""
    try:
        return resolve_character(document, text)
    except ValueError as error:
        earlier = earlier_characters(registry, document)
        if not earlier:
            raise
        try:
            return resolve_character({"cast": {"characters": [row for row, _ in earlier]}}, text)
        except ValueError:
            raise error from None


def plan_merge(bundle, config, first, second, *, keep=None, name=None, store=None):
    """Work out the whole merge without writing anything; ``describe(plan)`` shows it."""
    from ..analysis_directory import _completed_directory
    from ..analysis_revision import related_directories
    directory = Path(bundle["directory"]).resolve()
    paths = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths}
    loaded = {path: _completed_directory(path) for path in paths}
    document, registry = loaded[directory][0], loaded[directory][1]
    one, other = _resolve(document, registry, first), _resolve(document, registry, second)
    here = {row["id"] for row in document["cast"]["characters"]}
    absent = [row for row in (one, other) if row["id"] not in here]
    if len(absent) == 2:
        raise ValueError(f"Neither {one['name']} nor {other['name']} is in this book; merge them in an earlier book")
    if one["id"] == other["id"]:
        raise ValueError(f"{first!r} and {second!r} are the same cast entry ({one['name']})")
    if store is None and config is not None:
        from .book_context import voice_store
        store = voice_store(bundle, config)
    states = {row["id"]: _voice_state(store, row["voice_id"]) for row in (one, other)}
    lines = {row["id"]: character_lines(document, row["id"]) if row["id"] in here else 0 for row in (one, other)}
    # A character an earlier book already has keeps its identity: a new entry merges into it, never the reverse.
    origins = {}
    for book in registry["books"][:-1]:
        for row in book["cast"]["characters"]:
            origins.setdefault(row["id"], book["book"].get("title") or book["source_fingerprint"][:12])
    inherited = [row for row in (one, other) if row["id"] in origins]
    fixed = ""
    chosen = None
    if keep is not None:
        chosen = _resolve(document, registry, keep)
        if chosen["id"] not in {one["id"], other["id"]}:
            raise ValueError(f"--keep must name one of the two characters being merged, not {chosen['name']!r}")
    if len(inherited) == 1:
        kept = inherited[0]
        if chosen is not None and chosen["id"] != kept["id"]:
            raise ValueError(f"{kept['name']} comes from an earlier book ({origins[kept['id']]}), so it must be kept; "
                             f"{chosen['name']} is merged into it")
        fixed = origins[kept["id"]]
        reason = f"it comes from an earlier book ({fixed}), so it must be kept"
    elif chosen is not None:
        kept = chosen
        reason = "you chose it"
    else:
        owned = [row for row in (one, other) if states[row["id"]][0] == "own"]
        if len(owned) == 1:
            kept, reason = owned[0], "it already has an accepted voice"
        elif lines[other["id"]] > lines[one["id"]]:
            kept, reason = other, ("both have accepted voices; " if owned else "") + "it has more lines"
        else:
            kept = one
            reason = (("both have accepted voices; " if owned else "")
                      + ("it has more lines" if lines[one["id"]] > lines[other["id"]] else "it was named first"))
    absorbed = other if kept is one else one
    if absent and absent[0]["id"] != kept["id"]:
        raise ValueError(f"{absent[0]['name']} is not in this book and {kept['name']} comes from an earlier book too; "
                         "merge them in an earlier book that has both")
    # A character only an earlier book has is taken into this book with its voice.
    taken = _earlier_records(registry, kept["id"]) if absent else None
    name = kept["name"] if name is None else name.strip()
    if name_key(name) not in {name_key(item) for item in combined_names(kept, absorbed)}:
        raise ValueError(f"The merged name must be one of: " + ", ".join(combined_names(kept, absorbed)))
    merged = merge_document(document, receipt(kept["id"], absorbed["id"], name, kept_records=taken))
    books = [_book_change(directory, document, merged, absorbed["id"], took_in=taken is not None)]
    for path in paths:
        if path == directory:
            continue
        later = loaded[path][0]
        characters = {row["id"]: row for row in later["cast"]["characters"]}
        if absorbed["id"] not in characters:
            continue
        if kept["id"] in characters:
            later_kept, records = characters[kept["id"]], None
        else:
            records = kept_records_for(merged, kept["id"])
            later_kept = records["character"]
        later_name = (name if name_key(name) in {name_key(item) for item in
                                                 combined_names(later_kept, characters[absorbed["id"]])}
                      else later_kept["name"])
        candidate = merge_document(later, receipt(kept["id"], absorbed["id"], later_name, kept_records=records))
        books.append(_book_change(path, later, candidate, absorbed["id"], took_in=records is not None))
    earlier = [book["book"].get("title") or book["source_fingerprint"][:12] for book in registry["books"][:-1]
               if any(row["id"] == absorbed["id"] for row in book["cast"]["characters"])]
    return MergePlan(directory=directory, paths=paths, hashes=hashes, kept=kept, absorbed=absorbed, name=name,
                     reason=reason, kept_lines=lines[kept["id"]], absorbed_lines=lines[absorbed["id"]],
                     kept_voice=_describe_state(states[kept["id"]]),
                     absorbed_voice=_describe_state(states[absorbed["id"]]), books=books, earlier=earlier, fixed=fixed,
                     kept_first="" if taken else first_line(document, kept["id"]),
                     absorbed_first=first_line(document, absorbed["id"]),
                     origins={row["id"]: origins[row["id"]] for row in (kept, absorbed) if row["id"] in origins})


def _describe_state(state):
    kind, detail = state
    if kind == "own":
        return "accepted voice" + (" (from an earlier book)" if detail == "inherited" else "")
    if kind == "shared":
        return f"uses {detail}'s recording"
    return "no accepted voice yet"


def describe(plan):
    """The change, in the words a person reads before confirming it."""
    kept, absorbed = plan.kept, plan.absorbed
    selected = plan.books[0]
    lines = [f"Merge two cast entries of {selected.title} into one character:"]
    for title, entry, voice, count, first in (("Keep:  ", kept, plan.kept_voice, plan.kept_lines, plan.kept_first),
                                              ("Absorb:", absorbed, plan.absorbed_voice, plan.absorbed_lines,
                                               plan.absorbed_first)):
        amount = (f"from {plan.origins[entry['id']]}, not in this book yet"
                  if entry is kept and selected.took_in else f"{count} line(s)")
        lines.append(f"  {title} {entry['name']}  ({entry['id']}, voice {entry['voice_id']}) — {voice}; {amount}")
        # Two entries often share a name: what each is and says tells them apart.
        description = " ".join(entry["description"].split())
        if description:
            lines.append("          " + (description if len(description) <= 100 else description[:99].rstrip() + "…"))
        if first:
            lines.append(f"          first line: {first}")
    lines += [f"  Kept because {plan.reason}. The kept entry's voice, description and accent stay.",
             f"  Afterwards: {plan.name}" + (", also called " + ", ".join(
                 name for name in plan.names if name_key(name) != name_key(plan.name))
                 if len(plan.names) > 1 else "")]
    if plan.absorbed_voice.startswith("accepted"):
        lines.append(f"  {absorbed['name']}'s accepted recording will no longer be used.")
    for book in plan.books:
        prefix = "  This book" if book is selected else f"  Later book {book.title}"
        if book.took_in:
            prefix += f" (gains {plan.name} in place of {absorbed['name']})"
        detail = f"{book.moved_lines} line(s) move to {plan.name}'s voice and will be rendered again"
        if book.joined_lines:
            detail += f"; {book.joined_lines} passage(s) they spoke together become one voice"
        lines.append(f"{prefix}: {detail}.")
        if book.dropped:
            lines.append(f"    {absorbed['name']}'s " + ", ".join(book.dropped) + " are dropped.")
        lines.extend(f"    Note: {warning}." for warning in book.warnings)
    if plan.earlier:
        lines.append(f"  Earlier book(s) keep {absorbed['name']} as a separate entry: " + ", ".join(plan.earlier)
                     + ". Merge in the earliest book to change the whole series.")
    return "\n".join(lines)


def apply_merge(plan, *, progress=print):
    """Publish a planned merge atomically across the books it changes; returns the revision result."""
    from ..analysis_revision import revise_directories
    from ..voice_profile_guard import require_valid_profiles
    for book in plan.books:
        require_valid_profiles(book.path)
    result = revise_directories({book.path: book.candidate for book in plan.books}, directories=plan.paths,
                                expected_manifest_hashes=plan.hashes, scope="character-merge",
                                sidecars={book.path: book.sidecars for book in plan.books if book.sidecars},
                                progress=progress)
    for book in plan.books:
        _restate_journal(book)
    return result


def _restate_journal(book):
    """Carry an unfinished speaker-decision journal over to the merged cast (pending state, best effort)."""
    from ..analysis_progress import analysis_cache_directory
    from ..speaker_decisions import (_load_decision_checkpoint, _save_decision_checkpoint,
                                     decision_checkpoint_key)
    cache = analysis_cache_directory(book.path)
    journal = _load_decision_checkpoint(cache, decision_checkpoint_key(book.original))
    if journal is None or journal.get("status") == "complete":
        return
    decisions = remap_decisions(journal["decisions"], book.candidate)
    if decisions is not None:
        _save_decision_checkpoint(cache, book.candidate, decisions, journal["status"])


def _ask(prompt, input_stream, output_stream):
    output_stream.write(prompt)
    output_stream.flush()
    answer = input_stream.readline()
    return None if not answer else answer.strip()


def run_merge_command(bundle, config, args, *, input_stream=None, output_stream=None):
    """CLI: show the merge, confirm it (``--yes`` skips, ``--dry-run`` only shows), publish it."""
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    keep, name = args.keep, args.name
    plan = plan_merge(bundle, config, args.first, args.second, keep=keep, name=name)
    while True:
        print(plan.text, file=output_stream)
        if args.dry_run:
            print("Dry run: nothing was changed.", file=output_stream)
            return 0
        if args.yes:
            break
        if not getattr(input_stream, "isatty", lambda: False)():
            print("Nothing was changed. Rerun with --yes to merge, or in a terminal to confirm.", file=output_stream)
            return 0
        answer = _ask("Merge them? [y]es, " + ("" if plan.fixed else "[s]wap which entry is kept, ")
                      + "[c]hoose the name, [n]o: ", input_stream, output_stream)
        choice = (answer or "n").casefold()[:1]
        if choice == "y":
            break
        if choice == "s" and plan.fixed:
            print(f"{plan.kept['name']} comes from an earlier book ({plan.fixed}), so it is the one kept.",
                  file=output_stream)
            continue
        if choice == "s":
            keep = plan.absorbed["id"]
            plan = plan_merge(bundle, config, args.first, args.second, keep=keep, name=name)
            continue
        if choice == "c":
            names = plan.names
            for number, item in enumerate(names, 1):
                print(f"  {number}. {item}", file=output_stream)
            picked = _ask("Name number: ", input_stream, output_stream)
            if picked and picked.isdigit() and 1 <= int(picked) <= len(names):
                name = names[int(picked) - 1]
                keep = plan.kept["id"]
                plan = plan_merge(bundle, config, args.first, args.second, keep=keep, name=name)
            continue
        print("Nothing was changed.", file=output_stream)
        return 0
    result = apply_merge(plan, progress=lambda message: print(message, file=output_stream))
    print(f"Merged {plan.absorbed['name']} into {plan.name}. Original files and the revision journal: "
          f"{result['backup_directory']}", file=output_stream)
    print("Run render again: chapters with moved lines are rendered again with the kept voice.", file=output_stream)
    return 0
