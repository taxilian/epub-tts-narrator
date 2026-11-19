"""``rename-character``: a person gives a cast character a new name (a misspelling fixed, say), and it is published.

The rename (``character_rename.rename_document``) changes the book and every
later book whose entry for the same character still has the old name; earlier
books keep theirs. Book-local files that show the name follow in the same
transaction: the accent groups' names and the audition scripts' cast binding.
Voices, lines and recordings stay, so nothing is rendered again. Publication
goes through ``analysis_revision.revise_directories`` scope
``character-rename`` (backups, later books' series history rebased).
"""

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sys

from ..character_merge import name_key, resolve_character
from ..character_rename import clean_name, rename_document
from ..pipeline import fingerprint
from .character_merge_command import _AUDITIONS, _GROUPS, _read, _regular


@dataclass
class RenameBook:
    path: Path
    title: str
    original: dict
    candidate: dict
    sidecars: dict = field(default_factory=dict)


@dataclass
class RenamePlan:
    directory: Path
    paths: list
    hashes: dict
    character: dict
    name: str
    books: list
    earlier: list
    skipped: list

    @property
    def kept_old_name(self):
        return self.books[0].candidate["provenance"]["character_renames"][-1]["old_name_kept_as_alias"]

    @property
    def text(self):
        return describe(self)


def _sidecars(path, original, candidate, voice_id, name):
    records = {}
    target = path / _GROUPS
    if _regular(target):
        record = _read(target)
        groups = record.get("groups") if isinstance(record, dict) else None
        if not isinstance(groups, list):
            raise ValueError(f"Invalid {_GROUPS}; fix or remove it before renaming a character")
        if any(voice_id in group.get("voice_ids", []) for group in groups):
            updated = deepcopy(record)
            for group in updated["groups"]:
                group["names"] = [name if voice == voice_id else item
                                  for voice, item in zip(group["voice_ids"], group["names"])]
            if updated != record:
                records[_GROUPS] = updated
    target = path / _AUDITIONS
    if _regular(target):
        record = _read(target)
        if (isinstance(record, dict) and record.get("source_fingerprint") == fingerprint(original["source"])
                and record.get("cast_fingerprint") == fingerprint(original["cast"])):
            records[_AUDITIONS] = dict(record, cast_fingerprint=fingerprint(candidate["cast"]))
    return records


def plan_rename(bundle, who, new_name):
    """Work out the rename without writing anything; ``describe(plan)`` shows it."""
    from ..analysis_directory import _completed_directory
    from ..analysis_revision import related_directories
    directory = Path(bundle["directory"] if isinstance(bundle, dict) else bundle).resolve()
    new_name = clean_name(new_name)
    paths = related_directories([directory])
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths}
    loaded = {path: _completed_directory(path) for path in paths}
    document, registry = loaded[directory][0], loaded[directory][1]
    character = resolve_character(document, who)
    old = character["name"]
    candidate = rename_document(document, character["id"], new_name)
    books = [RenameBook(directory, document["source"]["book"].get("title") or directory.name, document, candidate,
                        _sidecars(directory, document, candidate, character["voice_id"], new_name))]
    skipped = []
    for path in paths:
        if path == directory:
            continue
        later = loaded[path][0]
        row = next((item for item in later["cast"]["characters"] if item["id"] == character["id"]), None)
        if row is None:
            continue
        title = later["source"]["book"].get("title") or path.name
        if name_key(row["name"]) != name_key(old):
            skipped.append(f"{title} (calls them {row['name']})")
            continue
        renamed = rename_document(later, row["id"], new_name)
        books.append(RenameBook(path, title, later, renamed, _sidecars(path, later, renamed, row["voice_id"], new_name)))
    earlier = [book["book"].get("title") or book["source_fingerprint"][:12] for book in registry["books"][:-1]
               if any(row["id"] == character["id"] for row in book["cast"]["characters"])]
    return RenamePlan(directory=directory, paths=paths, hashes=hashes, character=character, name=new_name,
                      books=books, earlier=earlier, skipped=skipped)


def describe(plan):
    character = plan.character
    lines = [f"Rename {character['name']} ({character['id']}, voice {character['voice_id']}) to {plan.name}."]
    lines.append(f"  The book's text uses “{character['name']}”, so it stays as an alias." if plan.kept_old_name
                 else f"  “{character['name']}” is not in the book's text, so it is dropped.")
    aliases = plan.books[0].candidate["cast"]["characters"]
    aliases = next(row for row in aliases if row["id"] == character["id"])["aliases"]
    if aliases:
        lines.append("  Also called: " + ", ".join(aliases))
    if len(plan.books) > 1:
        lines.append("  Later books renamed too: " + ", ".join(book.title for book in plan.books[1:]))
    if plan.skipped:
        lines.append("  Later books that already use another name keep it: " + ", ".join(plan.skipped))
    if plan.earlier:
        lines.append("  Earlier books keep the old name: " + ", ".join(plan.earlier)
                     + ". Rename it in the earliest book to change the whole series.")
    lines.append("  The voice, its recording and every line stay; nothing is rendered again.")
    return "\n".join(lines)


def apply_rename(plan, *, progress=print):
    """Publish a planned rename atomically across the books it changes; returns the revision result."""
    from ..analysis_revision import revise_directories
    result = revise_directories({book.path: book.candidate for book in plan.books}, directories=plan.paths,
                                expected_manifest_hashes=plan.hashes, scope="character-rename",
                                sidecars={book.path: book.sidecars for book in plan.books if book.sidecars},
                                progress=progress)
    for book in plan.books:
        _restate_journal(book)
    return result


def _restate_journal(book):
    """An unfinished speaker-decision journal is keyed by the cast; carry it over unchanged (best effort)."""
    from ..analysis_progress import analysis_cache_directory
    from ..speaker_decisions import _load_decision_checkpoint, _save_decision_checkpoint, decision_checkpoint_key
    cache = analysis_cache_directory(book.path)
    journal = _load_decision_checkpoint(cache, decision_checkpoint_key(book.original))
    if journal is not None and journal.get("status") != "complete":
        _save_decision_checkpoint(cache, book.candidate, journal["decisions"], journal["status"])


def run_rename_command(bundle, args, *, input_stream=None, output_stream=None):
    """CLI: show the rename, confirm it (``--yes`` skips, ``--dry-run`` only shows), publish it."""
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    plan = plan_rename(bundle, args.character, args.name)
    print(plan.text, file=output_stream)
    if args.dry_run:
        print("Dry run: nothing was changed.", file=output_stream)
        return 0
    if not args.yes:
        if not getattr(input_stream, "isatty", lambda: False)():
            print("Nothing was changed. Rerun with --yes to rename, or in a terminal to confirm.", file=output_stream)
            return 0
        output_stream.write("Rename? [y]es, [n]o: ")
        output_stream.flush()
        if (input_stream.readline().strip().casefold() or "n")[:1] != "y":
            print("Nothing was changed.", file=output_stream)
            return 0
    result = apply_rename(plan, progress=lambda message: print(message, file=output_stream))
    print(f"Renamed {plan.character['name']} to {plan.name}. Original files and the revision journal: "
          f"{result['backup_directory']}", file=output_stream)
    return 0
