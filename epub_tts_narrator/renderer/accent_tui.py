"""Full-screen accent review (Textual): see the plan, edit it, or have the model revise it, then save.

Only the layout and key handling live here; the plan and every change are in
``accent_review.AccentPlanner`` / ``AccentController`` so they can be tested
without a terminal. Import this module only after ``tui_support.wants_tui``
said yes.
"""

from __future__ import annotations

import threading
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from epub_tts_narrator.tui_widgets import (
    BASE_CSS, CapturePrintMixin, ChoiceScreen, ConfirmScreen, JobsTable, KeysScreen, MessageLog, TextEditScreen,
    highlighted_key, sync_table,
)

UNASSIGNED = "unassigned"
NEW_GROUP = "new"
NO_ACCENT = "none"
KEPT = "kept:"


class AccentApp(CapturePrintMixin, App):
    """The accent screen. ``controller`` owns the plan; this class only displays it."""

    TITLE = "Accents"
    CSS = BASE_CSS + """
    #main { height: 1fr; }
    #left { width: 50%; min-width: 40; }
    #right { width: 1fr; }
    #groups { height: 1fr; }
    #detail-scroll { height: 9; border: round $primary-background; padding: 0 1; }
    #members { height: 1fr; border: round $primary-background; }
    #bottom { height: 9; }
    #log { width: 1fr; height: 1fr; border: round $primary-background; }
    #jobs { width: 40%; height: 1fr; border: round $primary-background; }
    """
    BINDINGS = [
        Binding("q", "quit_accents", "Quit"),
        Binding("ctrl+q", "quit_accents", "Quit", show=False, priority=True),
        Binding("question_mark", "keys", "Keys", tooltip="Show every key"),
        Binding("i", "instruct", "Instruct", tooltip="Type what to change; the model revises the whole plan"),
        Binding("a", "edit('accent')", "Accent", tooltip="Change the highlighted group's accent"),
        Binding("o", "edit('origin')", "Origin", tooltip="Rename the highlighted group's origin"),
        Binding("w", "edit('why')", "Why", tooltip="Edit the reason for the highlighted group's accent"),
        Binding("m", "move", "Move", tooltip="Move a character to another group or a new one"),
        Binding("tab", "switch", "Groups/Members", show=False, tooltip="Switch between groups and members"),
        Binding("n", "propose", "New plan", show=False, tooltip="Ask the model for a fresh plan"),
        Binding("s", "save", "Save", tooltip="Save the accents, groups and reasons"),
    ]

    def __init__(self, controller, *, title="Accents", instructions=None):
        super().__init__()
        self.controller = controller
        self.initial_instructions = instructions  # revise the saved plan with these on opening
        self.book_title = title
        self._ui_thread = None
        self._started = time.monotonic()
        controller._on_change = self._changed

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield DataTable(id="groups", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                with VerticalScroll(id="detail-scroll"):
                    yield Static("", id="detail")
                yield DataTable(id="members", cursor_type="row")
        with Horizontal(id="bottom"):
            yield MessageLog(id="log")
            yield JobsTable(id="jobs")
        yield Footer()

    def on_mount(self):
        self._ui_thread = threading.get_ident()
        self.capture_prints()
        self.title = f"Accents — {self.book_title}"
        self.query_one("#groups", DataTable).add_columns("Accent", "Origin", "Voices")
        self.query_one("#members", DataTable).add_columns("Character", "About")
        self.query_one("#members", DataTable).border_title = "Characters (m moves one)"
        self.query_one("#log", MessageLog).border_title = "Log"
        self.query_one("#jobs", JobsTable).border_title = "Model requests"
        if not self.controller.planner.groups and self.controller.planner.pending:
            self.controller.propose()  # no saved plan yet
        elif self.initial_instructions:
            self.controller.revise(self.initial_instructions)
        else:
            self.controller.log("This is the saved plan; nothing is asked of the model until you press i or n.")
        self.render_all()
        self.query_one("#groups", DataTable).focus()
        self.set_interval(0.5, self._tick)

    # Updates ------------------------------------------------------------------

    def on_captured_line(self, text, stderr):
        self.controller.log(text, "error" if stderr and "rror" in text else "info")

    def _changed(self):
        if threading.get_ident() == self._ui_thread:
            self.render_all()
        else:
            try:
                self.call_from_thread(self.render_all)
            except RuntimeError:
                pass  # The app is closing.

    def _tick(self):
        if self.controller.busy():
            self.query_one("#jobs", JobsTable).show(self.controller.jobs.jobs())

    def render_all(self):
        planner = self.controller.planner
        status = f"{len(planner.groups)} group(s) · {planner.changed()} voice(s) would change"
        if self.controller.busy():
            status += " · asking the model…"
        elif (planner.saved or planner.from_saved) and not planner.dirty:
            status += " · saved"
        elif planner.dirty:
            status += " · not saved"
        self.sub_title = status
        rows = []
        for index, group in enumerate(planner.groups):
            origin = Text(group["origin"])
            if planner.continues_series(group):
                origin.append("  (series)", style="dim")
            rows.append((str(index), (group["accent"], origin, str(len(group["voice_ids"])))))
        missing = planner.unassigned()
        if missing:
            rows.append((UNASSIGNED, (Text("no accent", style="yellow"), "left out of the plan", str(len(missing)))))
        for index, group in enumerate(planner.kept_groups()):
            # Decided elsewhere; shown so every character can be found and moved.
            rows.append((f"{KEPT}{index}", (Text(group["accent"] or "no accent", style="dim"),
                                           Text(f"{group['origin']} (kept)", style="dim"),
                                           str(len(group["voice_ids"])))))
        sync_table(self.query_one("#groups", DataTable), rows)
        self.render_detail()
        self.query_one("#log", MessageLog).sync(self.controller.messages, started=self._started)
        self.query_one("#jobs", JobsTable).show(self.controller.jobs.jobs())

    def render_detail(self):
        planner = self.controller.planner
        key = highlighted_key(self.query_one("#groups", DataTable))
        detail = self.query_one("#detail", Static)
        text = Text()
        if key is None:
            text.append("Waiting for the model's plan…" if self.controller.busy() else "No plan yet; press n.")
            members = []
        elif key == UNASSIGNED:
            text.append("Left out of the plan\n", style="bold")
            text.append("These characters get no assigned accent. Press m to place one, or i to ask the model.")
            members = planner.unassigned()
        elif key.startswith(KEPT):
            group = planner.kept_groups()[int(key[len(KEPT):])]
            text.append(group["accent"] or "no accent", style="bold")
            text.append(f" — {group['origin']}\n")
            text.append("Decided elsewhere, so the model keeps it. Tab to a character and press m to give them "
                        "another accent in this book (or none); saving makes it this book's choice.", style="dim")
            members = group["voice_ids"]
        else:
            group = planner.groups[int(key)]
            text.append(f"{group['accent']}", style="bold")
            text.append(f" — {group['origin']}\n")
            if planner.continues_series(group):
                text.append("Continues this origin from an earlier book in the series.\n", style="dim")
            text.append("Why (w): ", style="bold")
            text.append(group.get("why") or "(no reason given)")
            members = group["voice_ids"]
        detail.update(text)
        sync_table(self.query_one("#members", DataTable),
                   [(voice_id, (planner.label(voice_id), planner.entries[voice_id]["about"][:120]))
                    for voice_id in members])

    def on_data_table_row_highlighted(self, event):
        if event.data_table.id == "groups":
            self.render_detail()

    # Helpers ------------------------------------------------------------------

    def _group_index(self):
        key = highlighted_key(self.query_one("#groups", DataTable))
        if key is None or key == UNASSIGNED:
            self.notify("Highlight a group first", severity="warning")
            return None
        if key.startswith(KEPT):
            self.notify("This accent was decided elsewhere; move a character (m) to change theirs",
                        severity="warning")
            return None
        return int(key)

    def _idle(self):
        if self.controller.busy():
            self.notify("Wait for the model to finish", severity="warning")
            return False
        return True

    # Actions ------------------------------------------------------------------

    def action_keys(self):
        self.push_screen(KeysScreen(self.BINDINGS, title="Accent keys"))

    def action_switch(self):
        members = self.query_one("#members", DataTable)
        groups = self.query_one("#groups", DataTable)
        (groups if self.focused is members else members).focus()

    def action_instruct(self):
        if not self._idle():
            return

        def submitted(text):
            if text and text.strip():
                self.controller.revise(text)
                self.render_all()
        self.push_screen(TextEditScreen(
            "What should change?", "",
            help_text="e.g. \"Tolnedrans should sound Italian\", \"Silk is from Drasnia, keep him with Rhodar\", "
                      "\"no upper-class English\". The model revises the whole plan. Ctrl+S sends · Escape cancels"),
            submitted)

    def action_propose(self):
        if not self._idle():
            return
        self._confirm("Ask the model for a fresh plan? The current plan and your edits are replaced.",
                      lambda: (self.controller.propose(), self.render_all()))

    def action_edit(self, field):
        if not self._idle():
            return
        index = self._group_index()
        if index is None:
            return
        planner = self.controller.planner
        group = planner.groups[index]
        action = {"accent": planner.set_accent, "origin": planner.set_origin, "why": planner.set_why}[field]
        title = {"accent": "Accent (a real-world accent, e.g. broad Yorkshire)", "origin": "Origin",
                 "why": "Why this accent fits"}[field]

        def check(text):
            if field == "accent":
                from .accent_assignment import accent_cue
                accent_cue(text, planner.terms)

        def saved(text):
            if text is not None:
                error = self.controller.edit(action, index, text)
                if error:
                    self.notify(error, severity="error")
        self.push_screen(TextEditScreen(title, group[field] or "", single_line=field != "why", validate=check), saved)

    def action_move(self):
        if not self._idle():
            return
        planner = self.controller.planner
        members = self.query_one("#members", DataTable)
        voice_id = highlighted_key(members) if self.focused is members else None
        if voice_id is None:
            self.push_screen(ChoiceScreen("Move which character?",
                                          [(v, planner.label(v)) for v in planner.entries]), self._move_to)
        else:
            self._move_to(voice_id)

    def _move_to(self, voice_id):
        if voice_id is None:
            return
        planner = self.controller.planner
        options = [(index, f"{group['accent']} — {group['origin']}") for index, group in enumerate(planner.groups)]
        options.append((NEW_GROUP, "A new group…"))
        options.append((NO_ACCENT, "No accent"))

        def chosen(target):
            if target is None:
                return
            if target == NO_ACCENT:
                target = None
            elif target == NEW_GROUP:
                self.push_screen(TextEditScreen("New group's origin", "", single_line=True),
                                 lambda origin: origin and self._new_group(voice_id, origin))
                return
            error = self.controller.edit(planner.move, voice_id, target)
            if error:
                self.notify(error, severity="error")
        self.push_screen(ChoiceScreen(f"Move {planner.label(voice_id)} to", options), chosen)

    def _new_group(self, voice_id, origin):
        planner = self.controller.planner

        def check(text):
            from .accent_assignment import accent_cue
            accent_cue(text, planner.terms)

        def chosen(accent):
            if accent:
                error = self.controller.edit(planner.move, voice_id, (origin, accent))
                if error:
                    self.notify(error, severity="error")
        self.push_screen(TextEditScreen(f"Accent for {origin}", "", single_line=True, validate=check), chosen)

    def action_save(self):
        if not self._idle():
            return
        planner = self.controller.planner
        if not planner.groups:
            self.notify("There is no plan to save yet", severity="warning")
            return
        self._confirm(f"Save accents for {len(planner.pending)} voice(s)? {planner.changed()} voice(s) change "
                      "accent, and their lines render again on the next render of this book.",
                      self._save, title="Save accents")

    def _save(self):
        try:
            self.controller.save()
        except (ValueError, OSError) as error:
            self.controller.log(f"Could not save: {error}", "error")
            self.notify(str(error), severity="error")
        self.render_all()

    def action_quit_accents(self):
        planner = self.controller.planner
        if self.controller.busy():
            self._confirm("A model request is still running. Quit without saving?", self.exit)
        elif planner.dirty:
            self._confirm("Quit without saving this plan?", self.exit)
        else:
            self.exit()

    def _confirm(self, message, then, *, title="Confirm"):
        self.push_screen(ConfirmScreen(message, title=title), lambda yes: yes and then())
