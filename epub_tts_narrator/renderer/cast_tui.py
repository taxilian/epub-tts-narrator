"""Full-screen casting (Textual): browse, play, compare, generate, accept, and edit voices.

Only the layout and key handling live here; every decision and action is in
``cast_tui_model.CastController`` so it can be tested without a terminal.
Import this module only after ``tui_support.wants_tui`` said yes.
"""

from __future__ import annotations

import threading
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Label, SelectionList, Static

from epub_tts_narrator.tui_widgets import (
    BASE_CSS, CapturePrintMixin, ChoiceScreen, MultiChoiceScreen, VimKeysMixin, ConfirmScreen, DetailScreen, Dialog, JobsTable, KeysScreen, MessageLog, TextEditScreen, highlighted_key, sync_table,
)

from . import playback_speed
from .cast_tui_model import DESIGNERS, FIELDS, STATUS_FILTERS, CastController, candidate_rows, label, next_status_filter


STATUS_STYLES = {
    "failed": "bold red", "generating": "bold cyan", "queued": "cyan", "needs regeneration": "yellow",
    "missing": "bold black on yellow", "pending": "magenta", "accepted": "green", "shared": "blue",
    "derived": "blue", "inherited": "dim green",
}


def word_changes(before, after):
    """``after`` with the words it added in green and the words it dropped struck through in red."""
    import difflib
    old, new = before.split(), after.split()
    text = Text()
    for tag, first, last, start, end in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            text.append(" ".join(old[first:last]) + " ", style="red strike")
        if tag in ("insert", "replace"):
            text.append(" ".join(new[start:end]) + " ", style="bold green")
        if tag == "equal":
            text.append(" ".join(new[start:end]) + " ")
    return text


class RevisionScreen(Dialog):
    """Review the model's revised descriptions: every change is marked; Space unmarks one; Enter saves the marked.

    Dismisses with ``{voice_id: description}`` to save, or None.
    """

    DEFAULT_CSS = BASE_CSS + """
    RevisionScreen .dialog { width: 90%; height: 90%; }
    RevisionScreen #revision-body { height: 1fr; }
    RevisionScreen #revision-voices { width: 34%; height: 1fr; }
    RevisionScreen #revision-scroll { width: 1fr; height: 1fr; border: round $primary-background; padding: 0 1; }
    """
    BINDINGS = [Binding("escape", "cancel", "Discard", priority=True),
                Binding("enter", "save", "Save marked", priority=True)]

    def __init__(self, changes, names, instructions):
        super().__init__()
        self.changes, self.names, self.instructions = changes, names, instructions

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(f"{len(self.changes)} revised description(s): {self.instructions}"),
                        classes="dialog-title")
            with Horizontal(id="revision-body"):
                yield SelectionList(*[(self.names.get(voice_id, voice_id), voice_id, True) for voice_id in self.changes],
                                    id="revision-voices")
                with VerticalScroll(id="revision-scroll"):
                    yield Static("", id="revision-detail")
            yield Static(Text("Green words are added, red struck words removed · Space unmarks a voice · Enter saves "
                              "the marked descriptions as drafts (recordings are unchanged; regenerate to hear them) · "
                              "Escape discards"), classes="dialog-help")

    def on_mount(self):
        voices = self.query_one("#revision-voices", SelectionList)
        voices.focus()
        if self.changes:
            voices.highlighted = 0
            self.show(next(iter(self.changes)))

    def show(self, voice_id):
        before, after = self.changes[voice_id]
        text = Text(self.names.get(voice_id, voice_id) + "\n", style="bold")
        text.append("Now\n", style="dim")
        text.append(before + "\n\n")
        text.append("Revised\n", style="dim")
        text.append_text(word_changes(before, after))
        self.query_one("#revision-detail", Static).update(text)

    def on_selection_list_selection_highlighted(self, event):
        self.show(event.selection.value)

    def action_save(self):
        marked = set(self.query_one("#revision-voices", SelectionList).selected)
        self.dismiss({voice_id: after for voice_id, (_, after) in self.changes.items() if voice_id in marked})

    def action_cancel(self):
        self.dismiss(None)


REVISE_SCOPES = (("this", "This voice"), ("untaken", "Every voice without an accepted take"),
                 ("all", "Every voice"))


def _described(error):
    """An error as a person reads it: its message, or its kind when it has none."""
    text = str(error).strip()
    return text if isinstance(error, (ValueError, RuntimeError, OSError)) and text else (
        f"{type(error).__name__}: {text}" if text else type(error).__name__)


MIN_COLUMN = 8  # a cut column never gets narrower than this
_CELL_GAPS = 2 * 3 + 2  # one cell of padding each side of the three columns, plus the scrollbar


def fit_columns(width, names, statuses, notes):
    """``(voice, note)`` column widths that share a voice table ``width`` cells wide.

    The status is shown whole. When every name and note fits, each column is
    as wide as its longest value. Otherwise the name, which is how a person
    finds a row, keeps up to two thirds of the room, and the note takes the
    rest; both are cut with "…" (the detail pane shows them whole).
    """
    status = max([len("Status"), *map(len, statuses)])
    room = max(width - _CELL_GAPS - status, 2 * MIN_COLUMN)
    voice = max([len("Voice"), *map(len, names)])
    note = max([len("Note"), *map(len, notes)])
    if voice + note <= room:
        return voice, note
    note_share = min(note, max(MIN_COLUMN, room // 3))
    voice = max(min(voice, room - note_share), MIN_COLUMN)
    return voice, max(room - voice, min(note_share, MIN_COLUMN))


def _cut(text, width):
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


class CastApp(VimKeysMixin, CapturePrintMixin, App):
    """The casting screen. ``controller`` owns all state; this class only displays it."""

    TITLE = "Cast"
    CSS = BASE_CSS + """
    #main { height: 1fr; }
    #left { width: 46%; min-width: 40; }
    #right { width: 1fr; }
    #filter { display: none; }
    #filter.visible { display: block; }
    #voices { height: 1fr; }
    #detail-scroll { height: 1fr; border: round $primary-background; padding: 0 1; }
    #takes { height: 9; border: round $primary-background; }
    #bottom { height: 11; }
    #log { width: 1fr; height: 1fr; border: round $primary-background; }
    #jobs { width: 45%; height: 1fr; border: round $primary-background; }
    """
    BINDINGS = [
        Binding("q", "quit_cast", "Quit"),
        # Textual's own Ctrl-Q would close at once, even mid-generation.
        Binding("ctrl+q", "quit_cast", "Quit", show=False, priority=True),
        Binding("question_mark", "keys", "Keys", tooltip="Show every key"),
        Binding("slash", "filter", "Filter", tooltip="Filter by name, voice ID or description"),
        Binding("f", "status_filter", "Status", tooltip="Cycle the status filter"),
        Binding("o", "sort", "Sort", tooltip="Sort by status (what needs attention first) or cast order"),
        Binding("enter,p", "play", "Play", tooltip="Play the highlighted take in Takes, else the accepted take"),
        Binding("space", "stop", "Stop", show=False, tooltip="Stop playback"),
        Binding("greater_than_sign", "speed", "Speed",
                tooltip="Play at 2x or 1x (pitch unchanged); also in every other listening screen"),
        Binding("g", "generate", "Generate", tooltip="Generate a new voice; choose the designer"),
        Binding("a", "accept", "Accept", tooltip="Accept the highlighted take"),
        Binding("m", "mark", "Mark A", show=False, tooltip="Mark the highlighted take as A for comparison"),
        Binding("x", "compare", "Compare", show=False,
                tooltip="Play the other of take A and the highlighted take"),
        Binding("e", "edit('prompt')", "Description",
                tooltip="Edit the voice description (the clone direction for a derived voice)"),
        Binding("s", "edit('reference_text')", "Script", tooltip="Edit the audition script"),
        Binding("t", "edit('speech_traits')", "Traits", tooltip="Edit speech traits"),
        Binding("A", "edit('accent')", "Accent",
                tooltip="Edit the accent cue said before every line in this voice (empty means none)"),
        Binding("n", "notes", "Notes",
                tooltip="What you know about this character (later names, gender, origin, notes): used for its "
                        "accent and voice description"),
        Binding("v", "accents", "Accents",
                tooltip="Review the book's accent plan (as accents BOOK does); q comes back here"),
        Binding("d", "share", "Share", tooltip="Use another voice's recording for this voice"),
        Binding("D", "stop_sharing", "Unshare", show=False, tooltip="Stop sharing (this book's own link only)"),
        Binding("J", "merge", "Same as", show=False,
                tooltip="This character is the same character as another: merge the two cast entries into one"),
        Binding("c", "clips", "Clips", tooltip="Reference clip count for the next generation"),
        Binding("i", "instruct", "Instruct",
                tooltip="Type how descriptions should change; the model revises them for you to review"),
        Binding("r", "refresh_prompt", "Rewrite", show=False,
                tooltip="Rewrite the description and script with the analysis model"),
        Binding("B,M", "bulk", "Bulk",
                tooltip="Generate many voices: every one without a voice, every one without an accepted take, "
                        "or the ones you choose (accents must be decided first)"),
        Binding("w", "pronounce", "Pronounce",
                tooltip="Choose how one of this character's names is said (opens pronounce for it)"),
        Binding("R", "retry", "Retry", tooltip="Retry a failed job (highlighted in Jobs, else this voice's last)"),
        Binding("X", "cancel_job", "Cancel",
                tooltip="Cancel a queued job (highlighted in Jobs, else this voice's queued jobs)"),
    ]

    def __init__(self, controller: CastController, *, title="Cast", startup=True):
        super().__init__()
        self.controller = controller
        self.book_title = title
        self.run_startup = startup
        self._ui_thread = None
        self._started = time.monotonic()
        self._voice_widths = None
        self._fitted_width = None
        controller._on_change = self._changed

    # Layout --------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Input(placeholder="Filter by name, voice ID or description", id="filter")
                yield DataTable(id="voices", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                with VerticalScroll(id="detail-scroll"):
                    yield Static("", id="detail")
                yield DataTable(id="takes", cursor_type="row")
        with Horizontal(id="bottom"):
            yield MessageLog(id="log")
            yield JobsTable(id="jobs")
        yield Footer()

    def on_mount(self):
        self._ui_thread = threading.get_ident()
        self.capture_prints()
        self.title = f"Cast — {self.book_title}"
        voices = self.query_one("#voices", DataTable)
        self.query_one("#takes", DataTable).add_columns("", "Take", "Created", "Designer", "Clips", "Verified")
        self.query_one("#log", MessageLog).border_title = "Log"
        self.query_one("#jobs", JobsTable).border_title = "Jobs"
        self.query_one("#takes", DataTable).border_title = "Takes"
        if self.run_startup:
            self.controller.startup()
        else:
            self.controller.refresh()
        self.render_all()
        voices.focus()
        self.set_interval(0.5, self._tick)

    # Updates -------------------------------------------------------------------

    def on_captured_line(self, text, stderr):
        from epub_tts_narrator.run_events import classify
        self.controller.log(text, classify(text))

    def _changed(self):
        """Controller callback from any thread: redraw on the UI thread."""
        if threading.get_ident() == self._ui_thread:
            self._schedule_reload()
        else:
            try:
                self.call_from_thread(self._schedule_reload)
            except RuntimeError:
                pass  # The app is closing.

    def _schedule_reload(self):
        self.run_worker(self._reload, thread=True, exclusive=True, group="reload")

    def _reload(self):
        try:
            self.controller.refresh()
        except Exception as error:  # noqa: BLE001 - a worker's error would otherwise close the whole screen
            self.call_from_thread(self.controller.log, f"Could not reread the cast: {_described(error)}", "error")
        self.call_from_thread(self.render_all)

    def _tick(self):
        from textual.css.query import NoMatches
        try:
            width = self.query_one("#voices", DataTable).size.width
        except NoMatches:
            return  # The screen is closing; a last timer tick has nothing to draw.
        if width != self._fitted_width:
            # The screen was resized: fit the voice table's columns to its new width.
            self.render_all()
        if self.controller.reopen and self.controller.jobs.running() is None:
            self.exit()
            return
        if self.controller.playing is not None and not self.controller.poll_playback():
            self.render_all()
        elif self.controller.jobs.running() is not None:
            self.query_one("#jobs", JobsTable).show(self.controller.jobs.jobs())

    def render_all(self):
        controller = self.controller
        counts = controller.counts()
        flt = controller.filter
        summary = f"{controller.ready()}/{len(controller.rows)} ready"
        if counts.get("missing"):
            summary += f" · {counts['missing']} missing (f shows them)"
        if counts.get("failed"):
            summary += f" · {counts['failed']} failed"
        if flt.status or flt.text:
            summary += f" · filter: {flt.status or 'any status'}{(' ' + repr(flt.text)) if flt.text else ''}"
        if flt.by_status:
            summary += " · sorted by status"
        if controller.playing is not None:
            summary += " · playing (space stops)"
        summary += " · " + playback_speed.help_text(controller.player)
        self.sub_title = summary
        visible = [(row, label(row), status, note.splitlines()[0] if note else "")
                   for row, status, note in controller.visible()]
        table = self.query_one("#voices", DataTable)
        # Long names (a changed-voice state) and notes are cut to the table's own width, so it stays
        # usable on any screen; the columns are fitted again when that width changes.
        self._fitted_width = table.size.width
        width = table.size.width or max(40, self.size.width * 46 // 100)
        # Room for "generating" too, so a job starting never refits the table.
        statuses = [status for _, _, status, _ in visible] + ["generating"]
        widths = fit_columns(width, [name for _, name, _, _ in visible], statuses,
                             [note for _, _, _, note in visible])
        fitted = (*widths, max(map(len, statuses)))
        highlighted = line = None
        if fitted != self._voice_widths:
            # Refitting rebuilds the columns; keep the highlighted row on the same line of the screen.
            highlighted = highlighted_key(table)
            if highlighted is not None:
                line = table.cursor_row - round(table.scroll_y)
            table.clear(columns=True)
            table.add_column("Voice", width=widths[0], key="voice")
            table.add_column("Status", key="status")
            table.add_column("Note", width=widths[1], key="note")
            self._voice_widths = fitted
        rows = [(row["voice_id"], (_cut(name, widths[0]), Text(status, style=STATUS_STYLES.get(status, "")),
                                   _cut(note, widths[1])))
                for row, name, status, note in visible]
        keys = sync_table(table, rows)
        if highlighted in keys:
            index = keys.index(highlighted)
            table.move_cursor(row=index, scroll=False)
            table.call_after_refresh(table.scroll_to, y=max(0, index - line), animate=False)
        self.render_detail()
        self.query_one("#log", MessageLog).sync(controller.messages, started=self._started)
        self.query_one("#jobs", JobsTable).show(controller.jobs.jobs())

    def render_detail(self):
        row = self.current_row()
        detail = self.query_one("#detail", Static)
        takes = self.query_one("#takes", DataTable)
        if row is None:
            detail.update("No voice matches the filter." if self.controller.rows else "This book has no voices.")
            sync_table(takes, [])
            return
        text = Text()
        text.append(label(row) + "\n", style="bold")
        for title, value, key in self.controller.details(row):
            text.append(f"{title}", style="bold")
            if key:
                text.append(f" ({key})", style="dim")
            text.append(f": {value}\n")
        detail.update(text)
        marked = self.controller.compare
        rows = []
        for item in candidate_rows(row):
            marker = "●" if item["accepted"] else ""
            if marked == (row["voice_id"], item["candidate_id"]):
                marker += "A"
            rows.append((item["candidate_id"], (marker, item["candidate_id"], item["created"], item["designer"],
                                                str(item["clips"]), item["verified"])))
        sync_table(takes, rows)

    def on_data_table_row_highlighted(self, event):
        if event.data_table.id == "voices":
            self.render_detail()

    def on_data_table_row_selected(self, event):
        """Enter inside a table: play a voice or take, or open a job's full detail."""
        if event.data_table.id in {"voices", "takes"}:
            self.action_play()
        elif event.data_table.id == "jobs":
            job = next((job for job in self.controller.jobs.jobs()
                        if str(job.job_id) == event.row_key.value), None)
            if job is not None:
                lines = [f"State: {job.state}", f"Job: {job.label}"]
                if job.error:
                    lines += ["", f"{job.error_type}: {job.error}"]
                if job.state == "failed":
                    lines += ["", "Press R with this job highlighted to retry it."]
                self.push_screen(DetailScreen(f"Job {job.job_id}", "\n".join(lines)))

    # Helpers -------------------------------------------------------------------

    def current_row(self):
        key = highlighted_key(self.query_one("#voices", DataTable))
        return self.controller.row(key) if key is not None else None

    def current_take(self):
        takes = self.query_one("#takes", DataTable)
        return highlighted_key(takes) if self.focused is takes else None

    def _require_row(self):
        row = self.current_row()
        if row is None:
            self.notify("Select a voice first", severity="warning")
        return row

    def _attempt(self, action):
        try:
            return action()
        except (ValueError, RuntimeError, OSError) as error:
            self.controller.log(str(error), "error")
            self.notify(str(error), severity="error")
            return None

    def _confirm(self, message, then, *, title="Confirm"):
        self.push_screen(ConfirmScreen(message, title=title), lambda yes: yes and then())

    # Actions -------------------------------------------------------------------

    def action_keys(self):
        self.push_screen(KeysScreen(self.BINDINGS, title="Casting keys"))

    def action_filter(self):
        box = self.query_one("#filter", Input)
        box.add_class("visible")
        box.focus()

    def on_input_changed(self, event):
        if event.input.id == "filter":
            self.controller.filter.text = event.value
            self.render_all()

    def on_input_submitted(self, event):
        if event.input.id == "filter":
            if not event.value:
                event.input.remove_class("visible")
            self.query_one("#voices", DataTable).focus()

    def action_status_filter(self):
        self.controller.filter.status = next_status_filter(self.controller.filter.status)
        self.render_all()

    def action_sort(self):
        self.controller.filter.by_status = not self.controller.filter.by_status
        self.render_all()

    def action_play(self):
        row = self._require_row()
        if row is None:
            return
        take = self.current_take()
        self.run_worker(lambda: self._play(row, take), thread=True, exclusive=True, group="play")

    def _play(self, row, take):
        try:
            self.controller.play(row, take)
        except Exception as error:  # noqa: BLE001 - playback trouble is reported, never closes the screen
            text = _described(error)
            self.call_from_thread(self.controller.log, f"Could not play: {text}", "error")
            self.call_from_thread(self.notify, text, severity="warning")
        self.call_from_thread(self.render_all)

    def action_stop(self):
        self.controller.stop()
        self.render_all()

    def action_speed(self):
        try:
            playback_speed.toggle(self.controller.player)  # A playing take starts again at the new speed.
        except (OSError, ValueError) as error:
            self.controller.log(f"Could not play: {_described(error)}", "error")
        self.controller.log(playback_speed.toggled_notice(self.controller.player))
        self.render_all()

    def action_mark(self):
        row, take = self.current_row(), self.current_take()
        if row is None or take is None:
            self.notify("Highlight a take in the takes table (Tab) to mark it", severity="warning")
            return
        self.controller.mark_compare(row, take)
        self.render_detail()

    def action_compare(self):
        row, take = self.current_row(), self.current_take()
        if row is None or take is None:
            self.notify("Highlight a take in the takes table (Tab) to compare", severity="warning")
            return
        partner = self.controller.compare_partner(row, take)
        if partner is None:
            self.notify("Mark another take of this voice with m first", severity="warning")
            return
        playing = self.controller.playing
        other = partner if playing == (row["voice_id"], take) else take
        self.run_worker(lambda: self._play(row, other), thread=True, exclusive=True, group="play")

    def action_generate(self):
        row = self._require_row()
        if row is None:
            return
        default = (getattr(self.controller.session, "config", {}).get("voice_design") or {}).get("type")
        order = ["gemini", "local", "elevenlabs"] if default == "gemini" else ["local", "gemini", "elevenlabs"]
        options = [(name, f"Generate {DESIGNERS[name]}") for name in order]

        def chosen(designer):
            if designer is None:
                return
            warning = self.controller.regenerate_warning(row)
            submit = lambda: self._attempt(lambda: self.controller.regenerate(row, designer))  # noqa: E731
            if warning:
                self._confirm(warning, submit, title=f"Generate {label(row)} {DESIGNERS[designer]}?")
            else:
                submit()
        self.push_screen(ChoiceScreen(f"Generate a new voice for {label(row)}", options, filterable=False), chosen)

    def action_accept(self):
        row, take = self.current_row(), self.current_take()
        if row is None or take is None:
            self.notify("Highlight a take in the takes table (Tab) to accept it", severity="warning")
            return
        self._confirm(f"Accept take {take} as the voice of {label(row)}?",
                      lambda: self._attempt(lambda: self.controller.accept(row, take)))

    def action_edit(self, field):
        row = self._require_row()
        if row is None:
            return
        title = ("Clone direction" if field == "prompt" and row.get("derived_from") else FIELDS[field])
        help_text = "Ctrl+S saves · Escape cancels"
        if field == "speech_traits":
            help_text = "Empty text means no traits for this character. " + help_text
        elif field == "accent":
            help_text = ("A short cue such as \"heavy Brooklyn accent\", said before every line in this voice; "
                         "empty means no cue for this voice. "
                         + ({"assigned": "Now the book's accent for this character's origin. ",
                            "description": "Now taken from the voice description. "}.get(row.get("accent_source"), ""))
                         + ("Pre-filled from the voice description. " if not row.get("accent")
                                                and row.get("accent_suggestion") else "") + help_text)
        elif field in {"prompt", "reference_text"}:
            help_text = "Applies the next time this voice is generated. " + help_text

        current = row.get(field) or ""

        def saved(value):
            if value is not None and value != current:
                self._attempt(lambda: self.controller.save_field(row, field, value))

        self.push_screen(TextEditScreen(f"{title} — {label(row)}", self.controller.field_value(row, field),
                                        help_text=help_text, single_line=field == "accent",
                                        validate=lambda value: self.controller.validate_field(row, field, value)),
                         saved)

    def action_notes(self):
        row = self._require_row()
        if row is None:
            return
        if row.get("derived_from"):
            self.notify("A changed voice shares its main voice's notes; add notes to the main voice",
                        severity="warning")
            return
        current = self.controller.notes_text(row)
        help_text = ("What you know about this character, even beyond what this book has revealed so far. "
                     "names: other or later names, comma-separated · gender · origin: where they come from · "
                     "notes: anything else about who they are or how they sound. Used for this character's accent "
                     "and voice description (sent to the analysis model as a description of the character, never "
                     "to the voice designer). Empty clears notes inherited from an earlier book. "
                     "Ctrl+S saves · Escape cancels")

        def saved(text):
            if text is None or text == current:
                return
            record = self._attempt(lambda: self.controller.parse_notes(row, text))
            if record is None or self._attempt(lambda: self.controller.save_notes(row, text)) is None:
                return
            if record and row.get("accent_source") != "chosen":
                self.controller.log(f"{label(row)}'s accent is reassigned with your notes: by accents BOOK, or "
                                    "before a render when the voice is already recorded.")
            if record and not row.get("derived_from"):
                self._confirm("Ask the analysis model to revise this voice's description so it agrees with your "
                              "notes? You review the proposal before anything is saved; recordings are unchanged.",
                              lambda: self._attempt(lambda: self.controller.follow_notes(
                                  row, record, self._revised("follow your notes"))),
                              title="Revise the description?")

        self.push_screen(TextEditScreen(f"Your notes — {label(row)}", current, help_text=help_text,
                                        validate=lambda text: self.controller.parse_notes(row, text)), saved)

    def action_share(self):
        row = self._require_row()
        if row is None:
            return
        options = [(other["voice_id"], f"{label(other)}  ({other['voice_id']})")
                   for other in self.controller.rows if other["voice_id"] != row["voice_id"]]

        def chosen(target_id):
            if target_id is not None:
                target = self.controller.row(target_id)
                self._confirm(f"{label(row)} will use {label(target)}'s recorded voice. "
                              "No new sample is generated; D undoes this.",
                              lambda: self._attempt(lambda: self.controller.share(row, target)))
        self.push_screen(ChoiceScreen(f"Which voice should {label(row)} use?", options), chosen)

    def action_merge(self):
        row = self._require_row()
        if row is None:
            return
        options = self._attempt(lambda: self.controller.merge_options(row))
        if options is None:
            return
        if not options:
            self.notify("No other character to merge with", severity="warning")
            return

        def chosen(other_id):
            if other_id is None:
                return
            other = self.controller.row(other_id)
            plan = self._attempt(lambda: self.controller.merge_plan(row, other))
            if plan is None:
                return
            if plan.fixed:
                # A character from an earlier book is always the one kept: nothing to choose.
                self._confirm(plan.text + "\n\nSave this merge?",
                              lambda: self._attempt(lambda: self.controller.merge(plan)), title="Merge characters?")
                return
            kept = self.controller.row(plan.kept["voice_id"])
            absorbed = self.controller.row(plan.absorbed["voice_id"])
            survivors = [(kept["voice_id"], f"Keep {label(kept)} — {plan.kept_voice}, {plan.kept_lines} line(s) "
                                            f"({kept['voice_id']}; suggested because {plan.reason})"),
                         (absorbed["voice_id"], f"Keep {label(absorbed)} — {plan.absorbed_voice}, "
                                                f"{plan.absorbed_lines} line(s) ({absorbed['voice_id']})")]

            def kept_one(voice_id):
                if voice_id is None:
                    return
                final = (plan if voice_id == kept["voice_id"] else
                         self._attempt(lambda: self.controller.merge_plan(row, other, keep=absorbed)))
                if final is not None:
                    self._confirm(final.text + "\n\nSave this merge?",
                                  lambda: self._attempt(lambda: self.controller.merge(final)),
                                  title="Merge characters?")
            self.push_screen(ChoiceScreen("Which entry is kept, with its voice, description and accent?",
                                          survivors, filterable=False), kept_one)
        self.push_screen(ChoiceScreen(f"Which character is the same character as {label(row)}?", options), chosen)

    def action_stop_sharing(self):
        row = self._require_row()
        if row is not None:
            self._attempt(lambda: self.controller.stop_sharing(row))

    def action_clips(self):
        row = self._require_row()
        if row is None:
            return
        default = (getattr(self.controller.session, "config", {}).get("casting") or {}).get("reference_clips", 3)
        # Escape dismisses with None, so "use the default" needs its own value.
        options = [("default", f"Default ({default})")] + [(count, f"{count} clip(s)") for count in (1, 2, 3)]

        def chosen(value):
            if value is not None:
                count = None if value == "default" else value
                self._attempt(lambda: self.controller.set_clips(row, count))
        self.push_screen(ChoiceScreen(f"Reference clips for {label(row)} (applies at the next generation)",
                                      options, filterable=False), chosen)

    def action_refresh_prompt(self):
        row = self._require_row()
        if row is not None:
            self._confirm("Ask the analysis model to rewrite this voice's description and audition script? "
                          "Book text about this character is sent to the configured endpoint. "
                          "Recordings are unchanged.",
                          lambda: self._attempt(lambda: self.controller.refresh_prompt(row)))

    def action_instruct(self):
        row = self.current_row()

        def scoped(scope):
            if scope is None:
                return
            rows = {"this": [row] if row else [], "untaken": [r for r in self.controller.rows if r.get("selected") is None],
                    "all": list(self.controller.rows)}[scope]
            rows = self.controller.revisable(rows)
            if not rows:
                self.notify("No designed voice to revise there (a cloned voice keeps its clone direction)",
                            severity="warning")
                return
            self.push_screen(TextEditScreen(
                f"How should {len(rows)} description(s) change?", "",
                help_text="e.g. \"Remove every mention of accent or regional speech; the accent cue adds it\", "
                          "\"make the women's voices lower\". Each voice's accent cue is sent too. Names and "
                          "descriptions go to the analysis model. Ctrl+S sends · Escape cancels"),
                lambda text: text and text.strip() and self._attempt(
                    lambda: self.controller.revise_descriptions(rows, text, self._revised(text))))
        self.push_screen(ChoiceScreen("Revise which descriptions?", REVISE_SCOPES, filterable=False), scoped)

    def _revised(self, instructions):
        def ready(changes):
            def review():
                if not changes:
                    self.notify("The model changed no description", severity="information")
                    return
                names = {voice_id: label(self.controller.row(voice_id) or {"voice_id": voice_id})
                         for voice_id in changes}
                self.push_screen(RevisionScreen(changes, names, instructions),
                                 lambda chosen: chosen and self._attempt(
                                     lambda: self.controller.save_descriptions(chosen)))
            try:
                self.call_from_thread(review)
            except RuntimeError:
                pass  # The screen closed before the reply arrived.
        return ready

    # Many voices at once --------------------------------------------------------

    def _designers(self):
        default = (getattr(self.controller.session, "config", {}).get("voice_design") or {}).get("type")
        order = ["gemini", "local", "elevenlabs"] if default == "gemini" else ["local", "gemini", "elevenlabs"]
        return [(name, f"Generate {DESIGNERS[name]}") for name in order]

    def action_bulk(self):
        controller = self.controller
        missing, untaken = controller.bulk_rows("missing"), controller.bulk_rows("untaken")
        scopes = [("missing", f"Every voice with no voice yet ({len(missing)})"),
                  ("untaken", f"Every voice without an accepted take ({len(untaken)}); saved verified takes are "
                              "used first"),
                  ("chosen", "Voices I choose…")]

        def scoped(scope):
            if scope is None:
                return
            if scope == "chosen":
                options = [(row["voice_id"], f"{label(row)} — {status}") for row, status, _ in controller.visible()]
                self.push_screen(MultiChoiceScreen("Generate which voices?", options,
                                                   help_text="Space marks voices · Enter generates the marked ones "
                                                             "(or the highlighted one) · Escape cancels"),
                                 lambda chosen: chosen and self._bulk_designer("chosen", chosen))
            else:
                self._bulk_designer(scope, ())
        self.push_screen(ChoiceScreen("Generate many voices", scopes, filterable=False), scoped)

    def _bulk_designer(self, scope, voice_ids):
        controller = self.controller
        rows = self._attempt(lambda: controller.bulk_rows(scope, voice_ids))
        if not rows:
            if rows is not None:
                self.notify("No voice to generate there", severity="warning")
            return
        undecided = controller.undecided_accents(rows)
        if undecided:
            self.notify(f"Decide accents first ({len(undecided)} without one): press v", severity="warning")
            controller.log(f"Not generated: {', '.join(undecided[:8])} {'has' if len(undecided) == 1 else 'have'} "
                           "no accent yet. Press v to review the accent plan, then generate.", "warning")
            return
        if scope == "untaken":
            self._confirm(f"Prepare {len(rows)} voice(s) without an accepted take? Saved verified takes are used "
                          "first; the rest are generated with the default designer.",
                          lambda: self._attempt(lambda: controller.bulk_generate(scope)),
                          title="Generate many voices?")
            return

        def chosen(designer):
            if designer is None:
                return
            replaced = [label(row) for row in rows if row.get("selected") is not None]
            warning = (f" The accepted take of {len(replaced)} of them ({', '.join(replaced[:4])}"
                       f"{'…' if len(replaced) > 4 else ''}) is replaced by the new one; old takes stay."
                       if replaced else "")
            self._confirm(f"Generate {len(rows)} voice(s) {DESIGNERS[designer]}?{warning}",
                          lambda: self._attempt(lambda: controller.bulk_generate(scope, voice_ids, designer)),
                          title="Generate many voices?")
        self.push_screen(ChoiceScreen(f"Generate {len(rows)} voice(s) with", self._designers(), filterable=False),
                         chosen)

    # Pronunciations ---------------------------------------------------------------

    TYPE_WORD = "\0type"

    def action_pronounce(self):
        row = self._require_row()
        if row is None:
            return
        options = [(word, f"{word} — {reading}") for word, reading in (row.get("pronunciations") or {}).items()]
        options.append((self.TYPE_WORD, "Another word…"))

        def chosen(word):
            if word is None:
                return
            if word == self.TYPE_WORD:
                self.push_screen(TextEditScreen("Which word?", "", single_line=True,
                                                help_text="One word as the book writes it, e.g. Zandramas. "
                                                          "Enter opens it · Escape cancels",
                                                validate=lambda text: self.controller.pronounce_command(row, text)),
                                 lambda text: text and self._pronounce(row, text))
            else:
                self._pronounce(row, word)
        self.push_screen(ChoiceScreen(f"Pronounce a name of {label(row)}", options, filterable=False), chosen)

    def _pronounce(self, row, word):
        argv = self._attempt(lambda: self.controller.pronounce_command(row, word))
        if argv is None:
            return
        self.controller.log(f"Opening pronounce for {word}; the cast comes back when it closes.")
        code = self.run_external(argv)
        self.controller.log(f"Pronounce for {word} closed" + ("." if code == 0 else f" (exit code {code})."),
                            "info" if code in (0, 2) else "error")
        self._schedule_reload()

    def run_external(self, argv):
        """Run a command on the real terminal with this screen suspended; its exit code."""
        import subprocess
        with self.suspend():
            code = subprocess.run(argv, check=False).returncode
            if code not in (0, 2):
                # Keep its error on screen until read. sys.stdin may be the screen's dialog bridge, so ask the terminal.
                try:
                    with open("/dev/tty", "r+") as terminal:
                        terminal.write(f"\n{argv[3]} exited with code {code}. Press Enter to return to the cast… ")
                        terminal.flush()
                        terminal.readline()
                except (OSError, KeyboardInterrupt):
                    pass
        return code

    def action_retry(self):
        jobs = self.query_one("#jobs", JobsTable)
        key = highlighted_key(jobs) if self.focused is jobs else None
        failures = self.controller.failures()
        if key is not None:
            target = next((job for job in failures if str(job.job_id) == key), None)
        else:
            row = self.current_row()
            target = next((job for job in reversed(failures)
                           if row is not None and job.scope.get("voice_id") == row["voice_id"]), None)
            target = target or (failures[-1] if failures else None)
        if target is None:
            self.notify("No failed job to retry", severity="warning")
            return
        self._attempt(lambda: self.controller.retry(target.job_id))

    def action_cancel_job(self):
        jobs = self.query_one("#jobs", JobsTable)
        key = highlighted_key(jobs) if self.focused is jobs else None
        if key is not None:
            attempt = lambda: self.controller.cancel(job_id=int(key))  # noqa: E731
        else:
            row = self._require_row()
            if row is None:
                return
            attempt = lambda: self.controller.cancel(row=row)  # noqa: E731
        cancelled = self._attempt(attempt)
        if cancelled:
            self.notify(f"Cancelled {len(cancelled)} queued job(s)")
            self.render_all()

    # The accent view ---------------------------------------------------------------

    def _accent_view(self):
        from .accent_tui import AccentScreen
        return next((screen for screen in self.screen_stack if isinstance(screen, AccentScreen)), None)

    def check_action(self, action, parameters):
        """While the accent view is open, the cast's own keys do nothing (its keys are the accent view's)."""
        if action in _CAST_ACTIONS and action != "quit_cast" and self._accent_view() is not None:
            return False
        return True

    def action_accents(self):
        if self._accent_view() is not None:
            return
        self.controller.log("Opening the book's accent plan…")
        self._attempt(lambda: self.controller.open_accents(self._accents_ready))

    def _accents_ready(self, accents):
        def show():
            from .accent_tui import CastAccentScreen
            if self._accent_view() is not None:
                accents.close()
                return
            planner = accents.planner
            title = self.book_title
            screen = CastAccentScreen(accents, title=title)

            def closed(saved):
                accents.close()
                self.controller.log("Saved the accent plan; voices now use its accents." if saved or planner.saved
                                    else "Closed the accent plan without saving.")
                self._schedule_reload()
            self.push_screen(screen, closed)
        try:
            self.call_from_thread(show)
        except RuntimeError:
            accents.close()  # The cast screen closed first.

    def action_quit_cast(self):
        accents = self._accent_view()
        if accents is not None:
            # Close the accent view first; it asks before discarding an unsaved plan.
            if self.screen is accents:
                accents.action_quit_accents()
            return
        running = self.controller.jobs.running()
        if running is None:
            self.exit()
            return
        self._confirm(f"“{running.label}” is still running. Quit when it finishes? Queued jobs are cancelled.",
                      self._quit_after_job, title="Quit casting?")

    def _quit_after_job(self):
        self.controller.jobs.cancel_queued()
        self.notify("Quitting after the running job finishes…")
        self.run_worker(self._wait_then_exit, thread=True, group="quit")

    def _wait_then_exit(self):
        self.controller.jobs.wait_idle()
        self.call_from_thread(self.exit)


# The cast's own actions; vim movement keys keep moving in the accent view.
_CAST_ACTIONS = {binding.action.split("(")[0] for binding in CastApp.BINDINGS} - {"vim_key"}


def run_cast_tui(controller, *, title="Cast"):
    """Open the casting screen; returns after the person quits and jobs settle."""
    from epub_tts_narrator.tui_support import prompt_streams
    from epub_tts_narrator.tui_widgets import PromptBridge
    app = CastApp(controller, title=title)
    # Model download questions are answered in a dialog, never on the raw terminal.
    bridge = PromptBridge(app)
    try:
        with prompt_streams(bridge.input, bridge.output):
            app.run()
    finally:
        controller.close()
    return 0


__all__ = ["CastApp", "run_cast_tui", "STATUS_FILTERS"]
