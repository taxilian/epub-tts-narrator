"""Reviewing a book's accent plan before it is saved, independent of any terminal library.

``AccentPlanner`` holds one book's proposed plan: groups of characters by
origin, each with an accent and the model's reason. A person can change a
group's accent, origin or reason, move a character to another group (or a
new one), ask the model for a fresh plan, or type instructions that the
model uses to revise the current plan. Nothing is saved until ``save``.

Every character that speaks is listed. Characters whose accent was decided
elsewhere (a cue chosen in casting, an accent in vocal traits a person set, or
an earlier book's assignment) are shown as kept groups the model works
around; moving one by hand makes it this book's own choice (``released``).

``AccentController`` runs the planner's model calls as background jobs on
one serial ``JobQueue`` for the full-screen screen (``accent_tui.py``); the
line command (``accent_assignment.run_accents_command``) calls the planner
directly.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import re
import sys
import time
import uuid

from epub_tts_narrator.tui_jobs import FAILED, RUNNING, JobQueue

from . import accent_assignment as assignment


class AccentPlanner:
    """One book's accent plan under review."""

    def __init__(self, bundle, config, config_path, *, instructions=None, keep=False, resume=None,
                 progress=print, requests=None):
        from .book_context import resolve_book_config, voice_store
        from .speech_traits import _book_terms, accent_cues
        self.bundle = bundle
        self.config = resolve_book_config(bundle, config)
        self.config_path = config_path
        self.progress = progress
        if instructions is not None and not 1 <= len(instructions) <= assignment.MAX_NOTES:
            raise ValueError(f"--instructions must contain from 1 to {assignment.MAX_NOTES} characters")
        self.explicit_guidance = instructions
        self.base_notes = instructions if instructions is not None else assignment.guidance(bundle, self.config)
        self.entries = assignment.voice_entries(bundle, assignment.parent_bundles(self.config))
        _, self.fixed = assignment._fixed(bundle, self.config, self.entries, own_only=True)
        self.pending = [voice_id for voice_id in self.entries if voice_id not in self.fixed]
        self.locked = self._locked()  # decided elsewhere: {voice_id: {"accent", "source", "origin"}}
        self.released = {}  # kept characters moved by hand: {voice_id: source}
        self.keep = {}
        if keep:
            own = assignment.assigned_accents(bundle, self.config, own=True)
            self.keep = {voice_id: own[voice_id] for voice_id in self.pending if own.get(voice_id)}
        self.origins = assignment.series_origins(bundle, self.config, own=False)
        self.series = {assignment._origin_key(item["origin"]): item["accent"] for item in self.origins}
        self.terms = _book_terms(bundle)
        self.clues = assignment.voice_clues(bundle, self.config, store=voice_store(bundle, self.config))
        self.before = accent_cues(bundle, self.config)
        self.groups = []
        self.from_saved = False  # the plan on screen is the saved one, not a new proposal
        self.revisions = []  # instructions given while reviewing, in order
        self.dirty = False
        self.saved = False
        self._requests = requests
        self.run_id = resume
        self._resume = resume

    def _locked(self):
        from .speech_traits import accent_overrides, speech_trait_overrides
        chosen = accent_overrides(self.bundle, self.config, include_cleared=True)
        traits = speech_trait_overrides(self.bundle, self.config)
        origins = {}
        for directory in self.config.get("_voice_parents", []):
            from .common import read_json, regular_file
            from pathlib import Path
            book_id = read_json(regular_file(Path(directory) / "manifest.json"))["book"]["id"]
            for group in assignment.read_groups(directory, book_id):
                for voice_id in group["voice_ids"]:
                    origins.setdefault(voice_id, group["origin"])
        locked = {}
        for voice_id, accent in self.fixed.items():
            source = "chosen" if voice_id in chosen else "traits" if traits.get(voice_id) else "inherited"
            locked[voice_id] = {"accent": accent or "", "source": source,
                                "origin": origins.get(voice_id, "") if source == "inherited" else ""}
        return locked

    # The saved plan ---------------------------------------------------------

    def load_saved(self):
        """Open this book's saved plan for editing, without asking the model. False when there is none.

        Characters added since it was saved are left without an accent (``unassigned``).
        A book saved before groups were recorded shows one group per accent.
        """
        own = assignment.assigned_accents(self.bundle, self.config, own=True)
        accents = {voice_id: own[voice_id] for voice_id in self.pending if own.get(voice_id)}
        if not accents:
            return False
        groups, placed = [], set()
        for group in assignment.read_groups(self.bundle["directory"], self.bundle["source"]["book"]["id"]):
            members = [voice_id for voice_id in group["voice_ids"]
                       if accents.get(voice_id, "").casefold() == group["accent"].casefold() and voice_id not in placed]
            if members:
                placed.update(members)
                groups.append({"origin": group["origin"], "accent": group["accent"], "why": group["why"],
                               "voice_ids": members, "names": [self.label(voice_id) for voice_id in members]})
        by_accent = {}
        for voice_id, accent in accents.items():
            if voice_id not in placed:
                by_accent.setdefault(accent, []).append(voice_id)
        for accent, members in by_accent.items():
            groups.append({"origin": "(origin not recorded)", "accent": accent, "why": "", "voice_ids": members,
                           "names": [self.label(voice_id) for voice_id in members]})
        self.groups = groups
        self.from_saved = True
        self.dirty = False
        if self.explicit_guidance is not None:
            # On a saved plan, --instructions are a revision of it, kept with the book's earlier guidance.
            self.explicit_guidance = None
            self.base_notes = assignment.guidance(self.bundle, self.config)
        return True

    # Model requests ---------------------------------------------------------

    def requests(self):
        if self._requests is None:
            root = assignment._book_cache(self.bundle, self.config_path) / "accent-assignment"
            if self._resume is None:
                self.run_id = uuid.uuid4().hex
                (root / self.run_id).mkdir(parents=True, exist_ok=False)
                self.progress(f"Fresh accent run: {self.run_id} (finish an interrupted run with --resume {self.run_id})")
            elif re.fullmatch(r"[0-9a-f]{32}", self._resume) is None or not (root / self._resume).is_dir():
                raise ValueError("--resume must be an accent run ID printed by an earlier invocation for this book")
            self._requests = assignment._requests(self.bundle, self.config_path, root / self.run_id,
                                                  progress=self.progress)
            if self._requests is None:
                raise ValueError("Set provider.model in the TOML configuration to assign accents")
        return self._requests

    def _notes(self):
        notes = dict.fromkeys(note for note in [self.base_notes, *self.revisions] if note)
        return "\n".join(notes)[-assignment.MAX_NOTES:]

    def _ask(self, revise=None):
        if not self.pending:
            raise ValueError("Every character's accent was decided elsewhere; move one (m) to give it an accent here")
        return assignment.request_assignment(
            self.bundle, self.requests(), self.pending,
            {voice_id: kept["accent"] for voice_id, kept in self.locked.items() if kept["accent"]},
            notes=self.base_notes, terms=self.terms, clues=self.clues, entries=self.entries,
            origins=self.origins, keep=self.keep, revise=revise)

    def propose(self):
        """Ask for a fresh plan (the current one is replaced)."""
        self._take(self._ask())
        return self.groups

    def revise(self, instructions):
        """Ask the model to change the current plan as ``instructions`` say."""
        instructions = " ".join(str(instructions).split())
        if not instructions:
            raise ValueError("Type what to change")
        if len(instructions) > assignment.MAX_NOTES:
            raise ValueError(f"Instructions must be at most {assignment.MAX_NOTES} characters")
        result = self._ask(revise=(deepcopy(self.groups), instructions))
        self.revisions.append(instructions)
        self._take(result)
        return self.groups

    def _take(self, result):
        self.groups = [dict(group, voice_ids=list(group["voice_ids"]), names=list(group["names"]))
                       for group in result["groups"]]
        self.from_saved = False
        self.dirty = True

    # Reading ----------------------------------------------------------------

    def label(self, voice_id):
        return self.entries[voice_id]["label"]

    def accents(self):
        return {voice_id: group["accent"] for group in self.groups for voice_id in group["voice_ids"]}

    KEPT_SOURCES = {"chosen": "chosen in casting", "traits": "from vocal traits you set",
                    "inherited": "from an earlier book"}

    def kept_groups(self):
        """Characters decided elsewhere, grouped by where and what: [{"origin", "accent", "source", "voice_ids"}]."""
        groups = {}
        for voice_id, kept in self.locked.items():
            origin = kept["origin"] or self.KEPT_SOURCES[kept["source"]]
            key = (kept["source"], kept["accent"].casefold(), origin.casefold())
            groups.setdefault(key, {"origin": origin, "accent": kept["accent"], "source": kept["source"],
                                    "voice_ids": []})["voice_ids"].append(voice_id)
        return list(groups.values())

    def unassigned(self):
        placed = self.accents()
        return [voice_id for voice_id in self.pending if voice_id not in placed]

    def changed(self):
        accents = self.accents()
        return sum(1 for voice_id in self.pending if accents.get(voice_id, "") != self.before.get(voice_id, ""))

    def continues_series(self, group):
        return assignment._origin_key(group["origin"]) in self.series

    # Editing ----------------------------------------------------------------

    def _group(self, index):
        if not 0 <= index < len(self.groups):
            raise ValueError("Choose a group first")
        return self.groups[index]

    def _check_series(self, origin, accent):
        series = self.series.get(assignment._origin_key(origin))
        if series and series.casefold() != accent.casefold():
            raise ValueError(f"{origin!r} already has the {series} in this series; rename the origin to give it "
                             "a different accent")

    def set_accent(self, index, accent):
        group = self._group(index)
        cue = assignment.accent_cue(accent, self.terms)
        self._check_series(group["origin"], cue)
        group["accent"] = cue
        self.dirty = True
        return cue

    def set_origin(self, index, origin):
        group = self._group(index)
        origin = " ".join(str(origin).split())[:assignment.MAX_ORIGIN]
        if not origin:
            raise ValueError("An origin needs a few words")
        self._check_series(origin, group["accent"])
        group["origin"] = origin
        self.dirty = True

    def set_why(self, index, why):
        self._group(index)["why"] = " ".join(str(why).split())[:assignment.MAX_WHY]
        self.dirty = True

    def move(self, voice_id, target):
        """Move one character to group ``target`` (an index), a new group given as (origin, accent), or None (no accent).

        A kept character (decided elsewhere) becomes this book's own choice.
        """
        if voice_id not in self.pending and voice_id not in self.locked:
            raise ValueError("That character has no lines in this book")
        if target is None:
            destination = None
        elif isinstance(target, tuple):
            origin, accent = target
            origin = " ".join(str(origin).split())[:assignment.MAX_ORIGIN]
            if not origin:
                raise ValueError("A new group needs an origin")
            cue = assignment.accent_cue(accent, self.terms)
            self._check_series(origin, cue)
            destination = {"origin": origin, "accent": cue, "why": "", "voice_ids": [], "names": []}
        else:
            destination = self._group(target)
        if voice_id in self.locked:
            self.released[voice_id] = self.locked.pop(voice_id)["source"]
            self.pending.append(voice_id)
        for group in self.groups:
            if voice_id in group["voice_ids"]:
                position = group["voice_ids"].index(voice_id)
                del group["voice_ids"][position], group["names"][position]
        if isinstance(target, tuple):
            self.groups.append(destination)
        if destination is not None:
            destination["voice_ids"].append(voice_id)
            destination["names"].append(self.label(voice_id))
        self.groups = [group for group in self.groups if group["voice_ids"]]
        self.dirty = True

    # Saving -----------------------------------------------------------------

    def save(self):
        accents = self.accents()
        assignment.save_assignments(self.bundle, self.config,
                                    {voice_id: accents.get(voice_id, "") for voice_id in self.pending}, replace=True)
        assignment.save_groups(self.bundle, self.groups, replace=True)
        from .speech_traits import save_accent
        for voice_id, source in self.released.items():
            if source != "inherited":
                # A cue chosen in casting (or traits naming an accent) outranks assignments: the move replaces it.
                save_accent(self.bundle, self.config, voice_id, accents.get(voice_id, ""))
        if self.explicit_guidance is not None or self.revisions:
            # Instructions given here shape later assignments too (new characters, later books).
            assignment.save_guidance(self.bundle, self._notes())
        self.dirty = False
        self.saved = True
        return len(self.pending)


class AccentController:
    """The accent screen's state: the planner plus a log and background model jobs."""

    def __init__(self, planner, *, jobs=None, on_change=None, clock=time.monotonic, describe_error=None):
        self.planner = planner
        self._on_change = on_change or (lambda: None)
        self._clock = clock
        self.messages = deque(maxlen=500)
        if describe_error is None:
            from .diagnostics import configured_secrets, safe_error_text
            secrets = configured_secrets(planner.config or {})
            describe_error = lambda error: safe_error_text(error, secrets=secrets)  # noqa: E731
        self.jobs = jobs or JobQueue(on_change=self._job_changed, describe_error=describe_error, clock=clock)
        planner.progress = lambda message: self.log(str(message))

    def log(self, text, severity="info"):
        self.messages.append((self._clock(), severity, str(text)))

    def busy(self):
        return any(job.active for job in self.jobs.jobs())

    def propose(self):
        return self.jobs.submit("Ask for an accent plan", self._propose)

    def _propose(self):
        self.planner.propose()
        self.log(f"Plan: {len(self.planner.groups)} group(s), {self.planner.changed()} voice(s) would change accent.")
        self._on_change()

    def revise(self, instructions):
        return self.jobs.submit("Revise the plan: " + " ".join(instructions.split())[:60],
                                lambda: self._revise(instructions))

    def _revise(self, instructions):
        self.planner.revise(instructions)
        self.log(f"Revised: {len(self.planner.groups)} group(s), {self.planner.changed()} voice(s) would change "
                 "accent.")
        self._on_change()

    def edit(self, action, *args):
        """Apply one edit now; returns an error message, or None."""
        try:
            action(*args)
        except ValueError as error:
            self.log(str(error), "error")
            return str(error)
        self._on_change()
        return None

    def save(self):
        count = self.planner.save()
        self.log(f"Saved accents for {count} voice(s) and the plan's groups and reasons.")
        self._on_change()
        return count

    def close(self):
        self.jobs.close()

    def _job_changed(self, job):
        if job.state == RUNNING:
            self.log("Started: " + job.label)
        elif job.state == FAILED:
            self.log(f"Failed: {job.label}: {job.error}", "error")
        elif job.state == "done":
            self.log("Done: " + job.label)
        self._on_change()


def report(planner, write=print):
    """The plan as text lines: each group's accent, origin, members and reason."""
    for group in planner.groups:
        series = " [as earlier in the series]" if planner.continues_series(group) else ""
        write(f"  {group['accent']} ({group['origin']}){series}: {', '.join(group['names'])}")
        if group.get("why"):
            write(f"      why: {group['why']}")
    missing = planner.unassigned()
    if missing:
        write(f"{len(missing)} voice(s) got no accent: " + ", ".join(planner.label(v) for v in missing[:8]))
    for group in planner.kept_groups():
        names = [planner.label(voice_id) for voice_id in group["voice_ids"]]
        write(f"  kept: {group['accent'] or 'no accent'} ({group['origin']}): {', '.join(names)}")
    write(f"{planner.changed()} voice(s) would change accent; their lines render again on the next render of "
          "this book.")


def ask_line(read=None, write=print):
    """The line interface's choice: ("save"|"leave"|"new"|"revise", instructions)."""
    read = read or sys.stdin.readline
    while True:
        print("[A] save these accents / [I] type instructions for the model to revise them / "
              "[N] ask the model for a new plan / [L] leave the book unchanged: ", end="", flush=True)
        try:
            answer = read()
        except KeyboardInterrupt:
            write("\nInterrupted; leaving the book unchanged.")
            return "leave", None
        except (EOFError, OSError):
            answer = ""
        if answer == "":
            write("\nInput ended; leaving the book unchanged.")
            return "leave", None
        answer = answer.strip()
        if answer.casefold() in {"a", "l", "q"}:
            return ("save" if answer.casefold() == "a" else "leave"), None
        if answer.casefold() == "n":
            return "new", None
        if answer.casefold() == "i" or answer.casefold().startswith("i "):
            text = answer[1:].strip()
            if not text:
                print("What should change? ", end="", flush=True)
                try:
                    text = read().strip()
                except (KeyboardInterrupt, EOFError, OSError):
                    text = ""
            if text:
                return "revise", text
            continue
        write("Please enter A, I, N or L.")
