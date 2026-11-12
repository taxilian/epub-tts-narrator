"""The full-screen run shell (Textual) that long-running commands run inside.

``run_in_shell(command, ...)`` opens the screen, runs ``command()`` on a
worker thread with the process's render boards, prints and terminal prompts
routed to the screen, and returns the command's exit code. The screen shows
live chapter status with block counts, ETA and worker activity, a filterable
log, and every failure with its saved detail; after a run that failed or was
interrupted it stays open so the person can read the errors and resume
(rerun the same command, which reuses every checkpoint) with ``R``.

Import this module only after ``tui_support.wants_tui`` said yes.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
import os
import sys
import threading

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static, TabbedContent, TabPane

from .run_events import EventBus
from .run_shell_model import RunModel, next_severity
from .tui_support import prompt_streams
from .tui_widgets import (
    BASE_CSS, CapturePrintMixin, ChoiceScreen, ConfirmScreen, DetailScreen, KeysScreen, PromptBridge,
    SEVERITY_STYLES, format_seconds, sync_table,
)


@contextmanager
def raise_interrupts():
    """Default stop for commands without safe stop boundaries: Ctrl-C as ``KeyboardInterrupt``.

    Yields a ``request`` callable that raises ``KeyboardInterrupt`` in the
    command's thread at its next Python instruction, as Ctrl-C would in the
    line interface (blocking calls finish first).
    """
    thread_id = threading.get_ident()

    def request():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), ctypes.py_object(KeyboardInterrupt))
    yield request


class RunApp(CapturePrintMixin, App):
    """Status, workers, log and errors for one command run (and its resumes)."""

    CSS = BASE_CSS + """
    #rows, #workers, #errors { height: 1fr; }
    #sched { height: auto; color: $text-muted; padding: 0 1; }
    #progress { height: auto; padding: 0 1; }
    #filters { height: auto; color: $text-muted; padding: 0 1; }
    #search { display: none; }
    #search.visible { display: block; }
    #log { height: 1fr; }
    """
    BINDINGS = [
        Binding("q", "quit_run", "Quit", tooltip="Quit; while running, stop at safe boundaries first"),
        Binding("question_mark", "keys", "Keys", tooltip="Show every key"),
        Binding("1", "tab('status')", "Status", tooltip="Live chapters, blocks, timing and active work"),
        Binding("2", "tab('workers')", "Workers", tooltip="What each local and remote worker is doing"),
        Binding("3", "tab('log')", "Log", tooltip="Every message, filtered"),
        Binding("4", "tab('errors')", "Errors", tooltip="Every failure with its saved detail"),
        Binding("b", "pick('book')", "Book", tooltip="Filter by book"),
        Binding("c", "pick('chapter')", "Chapter", tooltip="Filter by chapter"),
        Binding("w", "pick('worker')", "Worker", tooltip="Filter by worker"),
        Binding("v", "severity", "Level", tooltip="Cycle the message level: info, warnings, errors, everything"),
        Binding("slash", "search", "Search", tooltip="Filter messages by text"),
        Binding("x", "clear_filters", "Clear", tooltip="Clear every filter"),
        Binding("R", "resume", "Resume", tooltip="Run the same command again; completed work is reused"),
        Binding("ctrl+c", "interrupt", "Stop", priority=True, show=False,
                tooltip="Stop at safe boundaries (press again to stop immediately)"),
    ]

    def __init__(self, model, bus, command, *, interrupt_scope=raise_interrupts, title="Run",
                 interactive_input=False, resumable=True, bridge=None, resume_command=None):
        super().__init__()
        self.resume_command = resume_command
        self.interactive_input = interactive_input
        self.resumable = resumable
        self.bridge = bridge
        self.model = model
        self.bus = bus
        self.command = command
        self.interrupt_scope = interrupt_scope
        self.run_title = title
        self.request_stop = None
        self.result = None
        self.raised = None
        self._log_seen = 0
        self._log_filter = None
        self._thread = None

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="status", id="tabs"):
            with TabPane("Status", id="status"):
                yield Static("", id="progress")
                yield DataTable(id="rows", cursor_type="row", zebra_stripes=True)
                yield Static("", id="sched")
            with TabPane("Workers", id="workers-tab"):
                yield DataTable(id="workers", cursor_type="row")
            with TabPane("Log", id="log-tab"):
                yield Static("", id="filters")
                yield Input(placeholder="Filter messages by text", id="search")
                yield RichLog(id="log", wrap=True, markup=False, highlight=False)
            with TabPane("Errors", id="errors-tab"):
                yield DataTable(id="errors", cursor_type="row")
        yield Footer()

    def on_mount(self):
        self.title = self.run_title
        if self.interactive_input and self.bridge is not None:
            # Commands that ask only on a terminal keep asking, through dialogs.
            self.capture_prints(stdout_tty=True, tee=self.bridge.output)
        else:
            self.capture_prints()
        self.query_one("#rows", DataTable).add_columns("Book", "Chapter", "Blocks", "Elapsed", "Left", "Active")
        self.query_one("#workers", DataTable).add_columns("Worker", "Book", "Chapter", "Block", "Doing", "For")
        self.query_one("#errors", DataTable).add_columns("Run", "Book", "Chapter", "Kind", "Error")
        self.start_command()
        self.set_interval(0.5, self.refresh_view)
        self.refresh_view(force=True)

    # Command thread ------------------------------------------------------------

    def start_command(self):
        def run():
            result = raised = None
            try:
                with self.interrupt_scope() as request:
                    self.request_stop = request
                    result = self.command()
            except BaseException as error:  # noqa: BLE001 - handed back to the caller after the screen closes
                raised = error
            finally:
                self.request_stop = None
            try:
                self.call_from_thread(self._command_done, result, raised)
            except RuntimeError:
                pass
        self._thread = threading.Thread(target=run, name="run-shell-command", daemon=True)
        self._thread.start()

    def _command_done(self, result, raised):
        self.result, self.raised = result, raised
        interrupted = isinstance(raised, KeyboardInterrupt) or result == 130
        error = None if raised is None or interrupted else f"{type(raised).__name__}: {raised}"
        if error:
            self.bus.log(error, severity="error")
        self.model.finish(130 if interrupted else (result if raised is None else 1), error, interrupted=interrupted)
        if not self.model.should_hold():
            self.exit()
            return
        self.refresh_view(force=True)
        self.action_tab("errors" if self.model.current_failures() else "log")
        self.notify("The run ended with problems. R resumes (completed work is reused); q quits.",
                    severity="warning", timeout=10)

    def on_captured_line(self, text, stderr):
        # Routine notices (cache paths, endpoints) also go to stderr; the text decides.
        self.bus.log(text)

    # Drawing -------------------------------------------------------------------

    def refresh_view(self, force=False):
        if not force and not self.model.changed.is_set() and not self.model.running:
            return
        self.model.changed.clear()
        model = self.model
        self.sub_title = model.summary()
        rows = []
        for key, scope, snapshot, text in model.visible_rows():
            if snapshot:
                active = "; ".join(
                    f"#{task['block']} {(task.get('worker') + ' ') if task.get('worker') else ''}{task['phase']} "
                    f"{format_seconds(task.get('elapsed', 0))}" for task in snapshot.get("active", [])) or "idle"
                cells = (scope.get("book") or "", f"{snapshot['chapter']}/{snapshot['total_chapters']}",
                         f"{snapshot['done']}/{snapshot['blocks']} ({round(100 * snapshot['done'] / max(1, snapshot['blocks']))}%)",
                         format_seconds(snapshot.get("elapsed", 0)),
                         f"~{format_seconds(snapshot['remaining'])}" if snapshot.get("remaining") else "",
                         active)
            else:
                cells = (scope.get("book") or "", "", "", "", "", text)
            rows.append((str(key), cells))
        sync_table(self.query_one("#rows", DataTable), rows)
        self.query_one("#sched", Static).update(self._sched_line())
        progress = model.progress if model.running else None
        self.query_one("#progress", Static).update(
            "" if not progress else "\n".join(line for line in (
                f"Step {progress['step']}/{progress['steps']}: {progress['step_name']} "
                f"{progress['step_fraction']:.0%} · {progress['stage']}",
                progress.get("activity", ""), progress.get("totals", "")) if line))
        workers = []
        for worker, tasks in sorted(model.workers().items()):
            for book, chapter, block, phase, elapsed in tasks:
                workers.append((f"{worker}:{book}:{chapter}:{block}",
                                (worker, book or "", str(chapter or ""), f"#{block}", phase, format_seconds(elapsed))))
        sync_table(self.query_one("#workers", DataTable), workers)
        failures = []
        for index, item in enumerate(model.visible_failures()):
            scope = item["scope"]
            failures.append((str(index), (str(item["attempt"]), scope.get("book") or "", str(scope.get("chapter") or ""),
                                          item["record"].get("error_type") or item["kind"],
                                          (item["error"] or "").splitlines()[0][:160] if item["error"] else "")))
        sync_table(self.query_one("#errors", DataTable), failures)
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.get_tab("errors-tab").label = f"Errors ({len(model.current_failures())})" if model.current_failures() else "Errors"
        self._draw_log()
        self.query_one("#filters", Static).update(
            "Filters: " + (model.filter.describe() or "none") + "   (b book · c chapter · w worker · v level · / text · x clear)")

    def _draw_log(self):
        log = self.query_one("#log", RichLog)
        signature = repr(self.model.filter)
        if signature != self._log_filter:
            log.clear()
            self._log_seen, self._log_filter = 0, signature
        items = self.model.visible_logs(after=self._log_seen)
        if self._log_seen == 0:
            items = items[-3000:]
        for sequence, event in items:
            line = Text(format_seconds(event.time - self.model.started) + " ", style="dim")
            if event.scope.get("book"):
                line.append(f"[{event.scope['book']}] ", style="bold")
            if event.scope.get("chapter") is not None:
                line.append(f"Ch {event.scope['chapter']}: ", style="bold")
            line.append(event.text, style=SEVERITY_STYLES.get(event.severity, ""))
            log.write(line)
            self._log_seen = sequence

    @staticmethod
    def _sched_line():
        try:
            from .renderer.progress import _scheduling_segment
            return _scheduling_segment()
        except Exception:
            return ""

    # Actions -------------------------------------------------------------------

    def action_keys(self):
        self.push_screen(KeysScreen(self.BINDINGS, title="Run shell keys"))

    def action_tab(self, name):
        ids = {"status": "status", "workers": "workers-tab", "log": "log-tab", "errors": "errors-tab"}
        self.query_one("#tabs", TabbedContent).active = ids[name]

    def action_pick(self, dimension):
        values = self.model.choices(dimension)
        if not values:
            self.notify(f"No {dimension} seen yet", severity="warning")
            return
        options = [("__all__", f"Any {dimension}")] + [(value, str(value)) for value in values]

        def chosen(value):
            if value is not None:
                setattr(self.model.filter, dimension, None if value == "__all__" else value)
                self.refresh_view(force=True)
        self.push_screen(ChoiceScreen(f"Show which {dimension}?", options), chosen)

    def action_severity(self):
        self.model.filter.severity = next_severity(self.model.filter.severity)
        self.refresh_view(force=True)

    def action_search(self):
        self.action_tab("log")
        box = self.query_one("#search", Input)
        box.add_class("visible")
        box.focus()

    def on_input_changed(self, event):
        if event.input.id == "search":
            self.model.filter.text = event.value
            self.refresh_view(force=True)

    def on_input_submitted(self, event):
        if event.input.id == "search" and not event.value:
            event.input.remove_class("visible")

    def action_clear_filters(self):
        from .run_shell_model import RunFilter
        self.model.filter = RunFilter()
        box = self.query_one("#search", Input)
        box.value = ""
        box.remove_class("visible")
        self.refresh_view(force=True)

    def on_data_table_row_selected(self, event):
        if event.data_table.id != "errors":
            return
        failures = self.model.visible_failures()
        index = int(event.row_key.value)
        if 0 <= index < len(failures):
            item = failures[index]
            scope = item["scope"]
            title = " ".join(part for part in (f"[{scope['book']}]" if scope.get("book") else "",
                                               f"Chapter {scope['chapter']}" if scope.get("chapter") else "",
                                               item["kind"]) if part)
            detail = json.dumps(item["record"], indent=2, ensure_ascii=False, default=str)
            self.push_screen(DetailScreen(title or "Failure", (item["error"] or "") + "\n\nSaved record:\n" + detail))

    def action_interrupt(self):
        if not self.model.running:
            self.action_quit_run()
            return
        if self.request_stop is None:
            return
        self.model.stopping = True
        self.model.changed.set()
        self.notify("Stopping at safe boundaries; press Ctrl-C again to stop immediately.", severity="warning")
        self.request_stop()

    def action_quit_run(self):
        if not self.model.running:
            self.exit()
            return
        self.push_screen(ConfirmScreen("The command is still running. Stop it at safe boundaries and quit when it "
                                       "has stopped? Rerunning the same command resumes from what was saved.",
                                       title="Stop the run?"),
                         lambda yes: yes and self.action_interrupt())

    def action_resume(self):
        if self.model.running:
            self.notify("Still running; resume is for a run that ended with problems", severity="warning")
            return
        if not self.resumable:
            self.notify("Running this command again would redo finished work; rerun it yourself if you mean to.",
                        severity="warning", timeout=8)
            return
        command = self.resume_command(self.model) if self.resume_command is not None else None
        self.model.restart()
        self.bus.log(f"Resuming: attempt {self.model.attempt}", severity="warning")
        self.action_tab("status")
        if command is not None:
            # Commands that start fresh on rerun resume the run they printed.
            self.command = command
            self.resume_command = None
        self.start_command()


def _render_interrupts(app):
    """Renderer commands: Ctrl-C keys drive ``RenderInterrupts`` exactly like SIGINT would."""
    @contextmanager
    def scope():
        from .renderer.interrupts import keyboard_interrupts

        def force_exit(code):
            # A second Ctrl-C exits at once; give the terminal back first.
            try:
                driver = getattr(app, "_driver", None)
                if driver is not None:
                    driver.stop_application_mode()
            finally:
                os._exit(code)
        with keyboard_interrupts(lambda message: print(message, file=sys.stderr), force_exit=force_exit) as controller:
            yield controller.request
    return scope


def run_in_shell(command, *, title, graceful_render_stop=False, interactive_input=False, resumable=True,
                 resume_command=None, closing_output=None):
    """Run ``command()`` inside the run shell; return its result (or re-raise its error).

    Render boards, prints and terminal prompts go to the screen while it is
    open. With ``graceful_render_stop`` Ctrl-C stops a render at safe
    boundaries (a second press exits at once); otherwise it raises
    ``KeyboardInterrupt`` in the command. With ``interactive_input`` the
    command's own terminal questions (``sys.stdin.readline`` after a printed
    question) are asked in dialogs. ``resumable`` allows ``R`` to run the
    command again; ``resume_command(model)`` may return a different command
    for that (one that resumes the printed run ID). When the screen closes, the final messages and every error
    are printed so the terminal keeps a record.
    """
    from .renderer.progress import shell_board
    bus = EventBus()
    model = RunModel(title)
    bus.subscribe(model.on_event)
    from .run_events import set_active_bus
    bridge = PromptBridge(None, on_answer=lambda answer: bus.log("> " + (answer or "(no answer)")))
    app = RunApp(model, bus, command, title=title, interactive_input=interactive_input,
                 resumable=resumable, bridge=bridge, resume_command=resume_command)
    bridge.app = app
    if graceful_render_stop:
        app.interrupt_scope = _render_interrupts(app)
    previous_bus, previous_stdin = set_active_bus(bus), sys.stdin
    try:
        if interactive_input:
            sys.stdin = bridge.input
        with shell_board(bus), prompt_streams(bridge.input, bridge.output):
            app.run()
    finally:
        sys.stdin = previous_stdin
        set_active_bus(previous_bus)
    output = closing_output or (lambda line: print(line, file=sys.stderr))
    for line in model.closing_lines():
        output(line)
    if app.raised is not None:
        raise app.raised
    if model.running:
        # The screen closed while the command was still running (terminal lost).
        return 130
    return app.result
