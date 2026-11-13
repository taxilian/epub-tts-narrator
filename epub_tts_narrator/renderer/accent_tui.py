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
from textual.widgets import DataTable, Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from epub_tts_narrator.tui_widgets import (
    BASE_CSS, CapturePrintMixin, ChoiceScreen, ConfirmScreen, JobsTable, KeysScreen, MessageLog, TextEditScreen,
    highlighted_key, sync_table,
)

from . import accent_catalog as catalog

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TYPE_OWN = "type-own"
UNASSIGNED = "unassigned"


def shown(accent):
    """An accent as the screen shows it: "broad Scottish", not "broad Scottish accent" on every row."""
    if accent and accent.casefold().endswith(" accent"):
        return accent[:-len(" accent")]
    return accent
NEW_GROUP = "new"
NO_ACCENT = "none"
KEPT = "kept:"


def _close_accents(name, used, unused_style=""):
    """Text for an accent's close neighbours, the ones some group already uses marked with its origin."""
    text = Text()
    for position, other in enumerate(catalog.near(name)):
        if position:
            text.append(", ", style=unused_style)
        origins = used.get(other.casefold())
        text.append(other + ("*" if catalog.ACCENTS[other]["rare"] else ""), style="yellow" if origins else unused_style)
        if origins:
            text.append(f" ({', '.join(origins)})", style="yellow")
    return text


class AccentPickerScreen(ChoiceScreen):
    """Choose an accent from ``accent_catalog``: each row shows its family, its close accents, and the groups using it.

    Filtering matches names, families and close accents, so typing "Scottish"
    lists every accent that sounds near it. The last row types an accent of
    your own (a strength word, or one not on the list).
    """

    def __init__(self, title, current, used, *, own_origin=None):
        self.current = catalog.listed(current)
        self.typed_default = shown(current) or ""
        self.typed = ""
        self.used = used
        self.own_origin = own_origin
        options = [(name, self._plain(name)) for name in catalog.ACCENTS]
        options.append((TYPE_OWN, "Type an accent of your own…"))
        super().__init__(title, options, help_text=(
            "Type to filter by accent, family or close accent · Enter chooses · Escape cancels\n"
            "Accents of one family sound related; yellow ones are already used (by the origin shown). "
            "* seldom heard in recordings: Drama 3 may blur it into its family."))

    def _origins(self, name):
        return [origin for origin in self.used.get(name.casefold(), []) if origin != self.own_origin]

    def _plain(self, name):
        return " ".join([name, catalog.ACCENTS[name]["family"], *catalog.near(name), *self._origins(name)])

    def _label(self, name, *, grouped=False):
        """One accent's row. Under its family heading (``grouped``) only links to other families are listed."""
        entry = catalog.ACCENTS[name]
        origins = self._origins(name)
        text = Text("  " if grouped else "")
        text.append(name + ("*" if entry["rare"] else ""), style="bold yellow" if origins else "bold")
        if name == self.current:
            text.append("  (current)", style="green")
        if origins:
            text.append(f"  used: {', '.join(origins)}", style="yellow")
        used = {key: [o for o in value if o != self.own_origin] for key, value in self.used.items()}
        if grouped:
            links = [other for other in catalog.near(name) if catalog.ACCENTS[other]["family"] != entry["family"]]
            for position, other in enumerate(links):
                text.append("  ≈ " if not position else ", ", style="dim")
                text.append(other, style="yellow" if used.get(other.casefold()) else "dim")
            return text
        text.append(f"  {entry['family']}", style="dim")
        close = _close_accents(name, used, "dim")
        if close:
            text.append("  ≈ ", style="dim")
            text.append_text(close)
        return text

    def _options(self, text):
        words = text.casefold().split()
        own = Option(Text(self.options[-1][1], style="italic"), id=str(len(self.options) - 1))
        if not words:
            # The whole list under family headings: a family's accents sound related.
            found, family = [], None
            for index, (value, _) in enumerate(self.options[:-1]):
                if catalog.ACCENTS[value]["family"] != family:
                    family = catalog.ACCENTS[value]["family"]
                    found.append(Option(Text(family, style="bold underline"), id=f"family:{family}", disabled=True))
                found.append(Option(self._label(value, grouped=True), id=str(index)))
            return [*found, own]
        matches = [(index, value) for index, (value, label) in enumerate(self.options)
                   if value != TYPE_OWN and all(word in label.casefold() for word in words)]
        # Accents whose own name matches come first, then those that only sound close to it.
        matches.sort(key=lambda item: not all(word in item[1].casefold() for word in words))
        return [*(Option(self._label(value), id=str(index)) for index, value in matches), own]

    def _first_choice(self, start=0, step=1):
        choices = self.query_one("#choices", OptionList)
        index = start
        while 0 <= index < choices.option_count:
            if not choices.get_option_at_index(index).disabled:
                return index
            index += step
        return None

    def on_mount(self):
        super().on_mount()
        choices = self.query_one("#choices", OptionList)
        if self.current is not None:
            choices.highlighted = choices.get_option_index(str(list(catalog.ACCENTS).index(self.current)))
        else:
            choices.highlighted = self._first_choice()

    def on_input_changed(self, event):
        self.typed = event.value.strip()  # offered when typing an accent of your own
        super().on_input_changed(event)
        self.query_one("#choices", OptionList).highlighted = self._first_choice()

    def action_move(self, delta):
        choices = self.query_one("#choices", OptionList)
        current = choices.highlighted if choices.highlighted is not None else -1
        target = self._first_choice(current + delta, 1 if delta > 0 else -1)
        if target is not None:
            choices.highlighted = target


class AccentApp(CapturePrintMixin, App):
    """The accent screen. ``controller`` owns the plan; this class only displays it."""

    TITLE = "Accents"
    CSS = BASE_CSS + """
    #busy { display: none; height: auto; padding: 0 1; background: $warning 25%; color: $text; text-style: bold;
            border: heavy $warning; }
    #busy.running { display: block; }
    #main { height: 1fr; }
    #left { width: 50%; min-width: 40; }
    #right { width: 1fr; }
    #groups { height: 1fr; }
    #detail-scroll { height: 12; border: round $primary-background; padding: 0 1; }
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
        yield Static("", id="busy")
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
        self.set_interval(0.25, self._tick)

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
        self.render_busy()

    def render_busy(self):
        """A banner across the screen while a model request runs, so it cannot be missed."""
        from textual.css.query import NoMatches
        try:
            banner = self.query_one("#busy", Static)
            groups = self.query_one("#groups", DataTable)
        except NoMatches:
            return  # The screen is closing; a last timer tick has nothing to draw.
        jobs = [job for job in self.controller.jobs.jobs() if job.active]
        busy = bool(jobs)
        banner.set_class(busy, "running")
        # No plan to show yet: the groups table itself says it is waiting.
        groups.loading = busy and not self.controller.planner.groups
        if not busy:
            return
        job = next((job for job in jobs if job.started is not None), jobs[0])
        now = time.monotonic()
        seconds = int(job.elapsed(now)) if job.started is not None else 0
        frame = SPINNER[int(now * 8) % len(SPINNER)]
        text = Text(f"{frame} Asking the model: {job.label} — {seconds} s", style="bold")
        text.append("\nThe plan appears here when the reply arrives (often a minute or two); editing waits until "
                    "then. Q quits without saving.", style="")
        banner.update(text)

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
            rows.append((str(index), (self._accent_cell(str(index), group), origin, str(len(group["voice_ids"])))))
        missing = planner.unassigned()
        if missing:
            rows.append((UNASSIGNED, (Text("no accent", style="yellow"), "left out of the plan", str(len(missing)))))
        for index, group in enumerate(planner.kept_groups()):
            # Decided elsewhere; shown so every character can be found and moved.
            rows.append((f"{KEPT}{index}", (Text(shown(group["accent"]) or "no accent", style="dim"),
                                           Text(f"{group['origin']} (kept)", style="dim"),
                                           str(len(group["voice_ids"])))))
        sync_table(self.query_one("#groups", DataTable), rows)
        self.render_detail()
        self.query_one("#log", MessageLog).sync(self.controller.messages, started=self._started)
        self.query_one("#jobs", JobsTable).show(self.controller.jobs.jobs())
        self.render_busy()

    def _accent_cell(self, key, group):
        """The accent, yellow when another origin has the same one, with "≈" when another's is close."""
        relations = {relation for relation, _ in self.controller.planner.neighbours(key)}
        text = Text(shown(group["accent"]), style="yellow" if "same" in relations else "")
        if "near" in relations:
            text.append(" ≈", style="dim")
        return text

    def _likeness(self, key, group, text):
        """Which other groups sound the same or close, and the accents near this one to tell them apart."""
        planner = self.controller.planner
        neighbours = planner.neighbours(key)
        same = [other["origin"] for relation, other in neighbours if relation == "same"]
        near = [f"{shown(other['accent'])} ({other['origin']})" for relation, other in neighbours if relation == "near"]
        if same:
            text.append("\nSame accent as: ", style="bold yellow")
            text.append(", ".join(same), style="yellow")
        if near:
            text.append("\nClose to: ", style="bold")
            text.append(", ".join(near))
        name = catalog.listed(group["accent"])
        if name is not None:
            text.append("\nSimilar accents (a chooses): ", style="bold")
            used = {k: [o for o in v if o != group["origin"]] for k, v in planner.used_by().items()}
            text.append_text(_close_accents(name, used))
        else:
            text.append("\nNot on the accent list (a chooses one)", style="dim")

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
            text.append(shown(group["accent"]) or "no accent", style="bold")
            text.append(f" — {group['origin']}\n")
            text.append("Decided elsewhere, so the model keeps it. Tab to a character and press m to give them "
                        "another accent in this book (or none); saving makes it this book's choice.", style="dim")
            if group["accent"]:
                self._likeness(key, group, text)
            members = group["voice_ids"]
        else:
            group = planner.groups[int(key)]
            text.append(f"{shown(group['accent'])}", style="bold")
            text.append(f" — {group['origin']}\n")
            if planner.continues_series(group):
                text.append("Continues this origin from an earlier book in the series.\n", style="dim")
            text.append("Why (w): ", style="bold")
            text.append(group.get("why") or "(no reason given)")
            self._likeness(key, group, text)
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
        title = {"accent": "Accent (one or two words, e.g. Scottish, Cockney, upper-class English)", "origin": "Origin",
                 "why": "Why this accent fits"}[field]

        def check(text):
            if field == "accent":
                from .accent_assignment import accent_cue
                accent_cue(catalog.canonical(text), planner.terms)

        def saved(text):
            if text is not None:
                error = self.controller.edit(action, index, text)
                if error:
                    self.notify(error, severity="error")
        value = shown(group[field]) if field == "accent" else group[field]
        if field == "accent":
            self._pick_accent(f"Accent for {group['origin']}", group["accent"], saved, check, own_origin=group["origin"])
            return
        self.push_screen(TextEditScreen(title, value or "", single_line=field != "why", validate=check), saved)

    def _pick_accent(self, title, current, then, check, *, own_origin=None):
        """The accent list first; its last row (or a filter matching nothing listed) types one instead."""
        picker = AccentPickerScreen(title, current, self.controller.planner.used_by(), own_origin=own_origin)

        def picked(name):
            if name is None:
                return
            if name != TYPE_OWN:
                then(name)
                return
            typed = picker.typed or picker.typed_default
            self.push_screen(TextEditScreen(
                title, typed, single_line=True, validate=check,
                help_text="An accent in one or two words, optionally after a strength word (light, broad, heavy). "
                          "Enter saves · Escape cancels"), then)
        self.push_screen(picker, picked)

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
        options = [(index, f"{shown(group['accent'])} — {group['origin']}") for index, group in enumerate(planner.groups)]
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
            accent_cue(catalog.canonical(text), planner.terms)

        def chosen(accent):
            if accent:
                error = self.controller.edit(planner.move, voice_id, (origin, accent))
                if error:
                    self.notify(error, severity="error")
        self._pick_accent(f"Accent for {origin}", "", chosen, check, own_origin=origin)

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
