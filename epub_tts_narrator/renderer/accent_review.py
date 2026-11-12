"""Reviewing a book's accent plan before it is saved, independent of any terminal library.

``AccentPlanner`` holds one book's proposed plan: groups of characters by
origin, each with an accent and the model's reason. A person can change a
group's accent, origin or reason, move a character to another group (or a
new one), ask the model for a fresh plan, or type instructions that the
model uses to revise the current plan. Nothing is saved until ``save``.

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
        self.revisions = []  # instructions given while reviewing, in order
        self.dirty = False
        self.saved = False
        self._requests = requests
        self.run_id = resume
        self._resume = resume

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
        return "\n".join(note for note in [self.base_notes, *self.revisions] if note)[-assignment.MAX_NOTES:]

    def _ask(self, revise=None):
        return assignment.request_assignment(
            self.bundle, self.requests(), self.pending, {k: v for k, v in self.fixed.items() if v},
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
        self.dirty = True

    # Reading ----------------------------------------------------------------

    def label(self, voice_id):
        return self.entries[voice_id]["label"]

    def accents(self):
        return {voice_id: group["accent"] for group in self.groups for voice_id in group["voice_ids"]}

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
        """Move one character to group ``target`` (an index), or to a new group given as (origin, accent)."""
        if voice_id not in self.pending:
            raise ValueError("Only characters this book is assigning can be moved")
        if isinstance(target, tuple):
            origin, accent = target
            origin = " ".join(str(origin).split())[:assignment.MAX_ORIGIN]
            if not origin:
                raise ValueError("A new group needs an origin")
            cue = assignment.accent_cue(accent, self.terms)
            self._check_series(origin, cue)
            destination = {"origin": origin, "accent": cue, "why": "", "voice_ids": [], "names": []}
        else:
            destination = self._group(target)
        for group in self.groups:
            if voice_id in group["voice_ids"]:
                position = group["voice_ids"].index(voice_id)
                del group["voice_ids"][position], group["names"][position]
        if isinstance(target, tuple):
            self.groups.append(destination)
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
    write(f"{planner.changed()} voice(s) would change accent; their lines render again on the next render of "
          "this book.")


def ask_line(read=None, write=print):
    """The line interface's choice: ("save"|"leave"|"revise", instructions)."""
    read = read or sys.stdin.readline
    while True:
        print("[A] save these accents / [I] type instructions for the model to revise them / "
              "[L] leave the book unchanged: ", end="", flush=True)
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
        write("Please enter A, I or L.")
