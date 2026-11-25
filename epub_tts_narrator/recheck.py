"""``analyze BOOK --recheck``: every check of a saved book again, with fresh model answers.

A rerun of ``analyze BOOK`` normally checks only what changed: each paragraph's
speakers and directions are recorded as checked by their text and split, and
the cast, gender, group and joint-line reviews replay their cached answers when
nothing they send has changed. A recheck sets those aside once: every step
asks again, in a request cache of its own (``<analysis cache>/recheck/RUN_ID``),
and the speaker and direction checks take every paragraph.

A recheck is pending state beside the book (``<analysis cache>/recheck.json``:
its run ID and the steps already finished), so an interrupted recheck resumes
where it stopped, replaying the answers it already has, and a finished step is
not asked again. It is removed when the last step finishes. A terminal asks
before a rerun whether to recheck (``ask``); ``--recheck`` starts one without
asking.
"""

from __future__ import annotations

import json
from pathlib import Path
import uuid

FILENAME = "recheck.json"
VERSION = 1
# The rerun's checks, in the order ``analyze BOOK`` runs them.
STEPS = ("cast", "names", "genders", "group-tags", "joint-lines", "attribution", "directions")
NAMES = {"cast": "cast entries that are one person", "names": "names the book gives characters listed by a description",
         "genders": "each character's gender",
         "group-tags": "characters that may be groups", "joint-lines": "lines several characters say together",
         "attribution": "every paragraph's speakers and speech splits", "directions": "every line's direction"}


class Recheck:
    def __init__(self, base, run_id, done=()):
        self.base, self.run_id, self.done = Path(base), run_id, list(done)

    @property
    def path(self):
        return self.base / FILENAME

    def wanted(self, step):
        """True while ``step`` has not finished in this recheck."""
        return step not in self.done

    def cache(self, step):
        """This recheck's own request cache for ``step``, or None when it already finished (ordinary rerun)."""
        if not self.wanted(step):
            return None
        directory = self.base / "recheck" / self.run_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def finished(self, step):
        if step not in self.done:
            self.done.append(step)
        if all(item in self.done for item in STEPS):
            self.path.unlink(missing_ok=True)
        else:
            self._save()

    def _save(self):
        self.base.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": VERSION, "run_id": self.run_id, "done": self.done}),
                             encoding="utf-8")
        temporary.replace(self.path)


def _base(directory, settings):
    from .analysis_progress import analysis_cache_directory
    return analysis_cache_directory(Path(directory), settings["cache_dir"])


def pending(directory, settings):
    """The unfinished recheck of this book, or None."""
    base = _base(directory, settings)
    path = base / FILENAME
    if path.is_symlink() or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(record, dict) or record.get("version") != VERSION or not isinstance(record.get("run_id"), str)
            or not isinstance(record.get("done"), list)):
        return None
    return Recheck(base, record["run_id"], [step for step in record["done"] if step in STEPS])


def start(directory, settings):
    """A new recheck of this book (saved at once, so an interruption resumes it)."""
    recheck = Recheck(_base(directory, settings), uuid.uuid4().hex)
    recheck._save()
    return recheck


def describe_due(document):
    """What an ordinary rerun would check now, in words, or "" when nothing changed since it was checked."""
    from .attribution_recheck import unchecked_blocks
    from .direction_review import unreviewed_blocks
    speakers, directions = len(unchecked_blocks(document)), len(unreviewed_blocks(document))
    parts = ([f"{speakers} paragraph(s)' speakers"] if speakers else []) + (
        [f"{directions} paragraph(s)' directions"] if directions else [])
    return " and ".join(parts)


def ask(document, *, input_fn=input, output=print):
    """Ask whether to recheck the whole book; True for yes."""
    due = describe_due(document)
    situation = (f"Only what changed is checked: {due}." if due else
                 "Everything in this book was already checked and has not changed since, so nothing would be "
                 "asked again.")
    output(situation)
    try:
        answer = input_fn("Check the whole book again with fresh answers (" + ", ".join(NAMES[step] for step in STEPS)
                          + ")? [y/N]: ")
    except EOFError:
        return False
    return (answer or "").strip().casefold()[:1] == "y"
