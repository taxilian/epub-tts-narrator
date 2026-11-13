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
    BASE_CSS, CapturePrintMixin, ChoiceScreen, ConfirmScreen, DetailScreen, Dialog, JobsTable, KeysScreen, MessageLog, TextEditScreen, highlighted_key, sync_table,
)

from .cast_tui_model import DESIGNERS, FIELDS, STATUS_FILTERS, CastController, candidate_rows, label, next_status_filter


STATUS_STYLES = {
    "failed": "bold red", "generating": "bold cyan", "queued": "cyan", "needs regeneration": "yellow",
    "missing": "yellow", "pending": "magenta", "accepted": "green", "shared": "blue",
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


class CastApp(CapturePrintMixin, App):
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
        Binding("L", "show_silent", "No lines", show=False,
                tooltip="Show or hide voices with no lines in this book (characters it only names)"),
        Binding("o", "sort", "Sort", tooltip="Sort by status (what needs attention first) or cast order"),
        Binding("enter,p", "play", "Play", tooltip="Play the highlighted take in Takes, else the accepted take"),
        Binding("space", "stop", "Stop", show=False, tooltip="Stop playback"),
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
        Binding("d", "share", "Share", tooltip="Use another voice's recording for this voice"),
        Binding("D", "stop_sharing", "Unshare", show=False, tooltip="Stop sharing (this book's own link only)"),
        Binding("c", "clips", "Clips", tooltip="Reference clip count for the next generation"),
        Binding("i", "instruct", "Instruct",
                tooltip="Type how descriptions should change; the model revises them for you to review"),
        Binding("r", "refresh_prompt", "Rewrite", show=False,
                tooltip="Rewrite the description and script with the analysis model"),
        Binding("M", "prepare_missing", "All missing", show=False,
                tooltip="Prepare every voice without an accepted take"),
        Binding("R", "retry", "Retry", tooltip="Retry a failed job (highlighted in Jobs, else this voice's last)"),
    ]

    def __init__(self, controller: CastController, *, title="Cast", startup=True):
        super().__init__()
        self.controller = controller
        self.book_title = title
        self.run_startup = startup
        self._ui_thread = None
        self._started = time.monotonic()
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
        voices.add_columns("Voice", "Status", "Note")
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
        except (ValueError, OSError) as error:
            self.call_from_thread(self.controller.log, f"Could not reread the cast: {error}", "error")
        self.call_from_thread(self.render_all)

    def _tick(self):
        if self.controller.playing is not None and not self.controller.poll_playback():
            self.render_all()
        elif self.controller.jobs.running() is not None:
            self.query_one("#jobs", JobsTable).show(self.controller.jobs.jobs())

    def render_all(self):
        controller = self.controller
        counts = controller.counts()
        flt = controller.filter
        summary = f"{controller.ready()}/{len(controller.speaking())} ready"
        silent = controller.silent()
        if silent:
            summary += (f" · {silent} without lines shown (L hides)" if flt.silent
                        else f" · {silent} without lines hidden (L shows)")
        if counts.get("failed"):
            summary += f" · {counts['failed']} failed"
        if flt.status or flt.text:
            summary += f" · filter: {flt.status or 'any status'}{(' ' + repr(flt.text)) if flt.text else ''}"
        if flt.by_status:
            summary += " · sorted by status"
        if controller.playing is not None:
            summary += " · playing (space stops)"
        self.sub_title = summary
        rows = []
        for row, status, note in controller.visible():
            rows.append((row["voice_id"], (label(row), Text(status, style=STATUS_STYLES.get(status, "")),
                                            note.splitlines()[0][:80] if note else "")))
        sync_table(self.query_one("#voices", DataTable), rows)
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

    def action_show_silent(self):
        flt = self.controller.filter
        flt.silent = not flt.silent
        self.notify(("Showing" if flt.silent else "Hiding") + f" {self.controller.silent()} voice(s) with no lines "
                    "in this book")
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
        except (ValueError, RuntimeError, OSError) as error:
            self.call_from_thread(self.controller.log, f"Could not play: {error}", "error")
        self.call_from_thread(self.render_all)

    def action_stop(self):
        self.controller.stop()
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

    def action_prepare_missing(self):
        self._attempt(lambda: self.controller.prepare_pending() or self.notify("Every voice has a take"))

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

    def action_quit_cast(self):
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
