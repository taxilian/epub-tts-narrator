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
import os
import sys
import threading

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static, TabbedContent, TabPane

from .run_events import EventBus
from .run_shell_model import RunModel, next_severity, repair_hint
from .tui_support import prompt_streams, screen_host
from .tui_widgets import (
    BASE_CSS, CapturePrintMixin, ChoiceScreen, ConfirmScreen, DetailScreen, KeysScreen, PromptBridge,
    SEVERITY_STYLES, format_seconds, highlighted_key, sync_table,
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
    """At-a-glance health, books, chapters, workers, log and errors for one command run (and its resumes)."""

    CSS = BASE_CSS + """
    #glance { height: 10; grid-size: 4 2; grid-rows: 5 5; grid-gutter: 0 1; padding: 0 1; }
    .tile { height: 5; border: round $primary-background; padding: 0 1; }
    .tile.ok { border: round $success; }
    .tile.busy { border: round $accent; }
    .tile.warning { border: round $warning; }
    .tile.error { border: round $error; }
    #books { height: 1fr; }
    #rows, #workers, #errors { height: 1fr; }
    #coordinators { height: auto; max-height: 18; }
    #node-work { height: auto; max-height: 8; }
    .section-title { color: $text-muted; padding: 0 1; margin-top: 1; }
    #nodes-line { height: auto; padding: 0 1; }
    #stages { height: auto; max-height: 12; }
    #sched { height: auto; color: $text-muted; padding: 0 1; }
    #progress { height: auto; padding: 0 1; }
    #filters { height: auto; color: $text-muted; padding: 0 1; }
    #search { display: none; }
    #search.visible { display: block; }
    #log { height: 1fr; }
    """
    TILES = ("Health", "Books", "Generating", "Verifying", "Accepted", "Rejected takes", "Failures", "Left")
    TABS = {"summary": "summary", "chapters": "chapters", "workers": "workers-tab", "log": "log-tab",
            "errors": "errors-tab", "rejected": "rejected-tab"}
    BINDINGS = [
        Binding("q", "quit_run", "Quit", tooltip="Quit; while running this asks before stopping anything"),
        # Textual's own Ctrl-Q would close the screen at once; it asks here too.
        Binding("ctrl+q", "quit_run", "Quit", show=False, priority=True, tooltip="Same as q"),
        Binding("question_mark", "keys", "Keys", tooltip="Show every key"),
        Binding("1", "tab('summary')", "Summary", tooltip="At a glance: health, books, clips, verification"),
        Binding("2", "tab('chapters')", "Chapters", tooltip="Live chapters, blocks, timing and active work"),
        Binding("3", "tab('workers')", "Nodes", tooltip="Chapter workers, each node's speech and verify slots and "
                                                        "their blocks, and the pipeline's stages"),
        Binding("4", "tab('log')", "Log", tooltip="Every message, filtered"),
        Binding("5", "tab('errors')", "Errors", tooltip="Every failure with its detail and repairs"),
        Binding("6", "tab('rejected')", "Rejected", tooltip="Every rejected take: listen, judge it, or use it"),
        Binding("e", "export_feedback", "Export", show=False,
                tooltip="Write the takes you judged differently from verification, for detection work"),
        Binding("b", "pick('book')", "Book", tooltip="Filter by book"),
        Binding("c", "pick('chapter')", "Chapter", tooltip="Filter by chapter"),
        Binding("w", "pick('worker')", "Worker", tooltip="Filter by worker"),
        Binding("V", "pick('voice')", "Voice", show=False,
                tooltip="Show chapters with a block spoken by this voice"),
        Binding("v", "severity", "Level", tooltip="Cycle the message level: info, warnings, errors, everything"),
        Binding("slash", "search", "Search", tooltip="Filter messages by text"),
        Binding("x", "clear_filters", "Clear", tooltip="Clear every filter"),
        Binding("r", "retry", "Retry", tooltip="Put the highlighted failed chapter or book back at the front of the "
                                               "queue while the run continues"),
        Binding("t", "takes", "Takes", tooltip="Listen to the failed passage's takes and accept a good one"),
        Binding("F", "finish", "Finish", show=False,
                tooltip="Stop waiting for retries; the run reports what still failed (asks first)"),
        Binding("R", "resume", "Resume", tooltip="Run again (only the failed books when one is highlighted); "
                                                 "completed work is reused"),
        Binding("ctrl+c", "interrupt", "Stop", priority=True, show=False,
                tooltip="Stop at safe boundaries, after confirming (again: stop immediately)"),
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
        self.exit_when_done = False
        self.forced_exit = None  # The exit code of a second stop, once the screen has closed.
        self.result = None
        self.raised = None
        self._log_seen = 0
        self._log_filter = None
        self._thread = None
        self._player = None

    PROFILE = "render"
    FILTER_KEYS = "b book · c chapter · w worker · V voice · v level · / text · x clear"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="summary", id="tabs"):
            with TabPane("Summary", id="summary"):
                with Grid(id="glance"):
                    for name in self.TILES:
                        yield Static("", classes="tile", id=_tile_id(name))
                yield from self._compose_summary()
            yield from self._compose_tabs()
        yield Footer()

    def _compose_summary(self):
        yield Static("", id="nodes-line")
        yield Static("", id="progress")
        yield DataTable(id="books", cursor_type="row", zebra_stripes=True)

    def _compose_tabs(self):
        """Every tab after Summary; the render layout adds chapters, nodes and rejected takes."""
        render = self.PROFILE == "render"
        if render:
            with TabPane("Chapters", id="chapters"):
                yield DataTable(id="rows", cursor_type="row", zebra_stripes=True)
                yield Static("", id="sched")
            with TabPane("Nodes", id="workers-tab"):
                yield Static("", id="coordinators-title", classes="section-title")
                yield DataTable(id="coordinators", cursor_type="row")
                yield Static(Text("What each speech node reports doing now (it counts every client, not only "
                                  "this render)"), classes="section-title")
                yield DataTable(id="node-work", cursor_type="row")
                yield Static("", id="nodes-title", classes="section-title")
                yield DataTable(id="workers", cursor_type="row", zebra_stripes=True)
                yield DataTable(id="stages", cursor_type="row")
        with TabPane("Log", id="log-tab"):
            yield Static("", id="filters")
            yield Input(placeholder="Filter messages by text", id="search")
            yield RichLog(id="log", wrap=True, markup=False, highlight=False)
        with TabPane("Errors", id="errors-tab"):
            yield DataTable(id="errors", cursor_type="row")
        if render:
            with TabPane("Rejected", id="rejected-tab"):
                yield Static(Text("Every take verification rejected in this run. Enter opens all takes of the "
                                  "passage: listen, mark each good (wrongly rejected) or bad, or change which one "
                                  "is used. e writes the takes you marked differently from verification to "
                                  "render-work/take-feedback-review.json for later study."), classes="section-title")
                yield DataTable(id="rejected", cursor_type="row", zebra_stripes=True)

    def on_mount(self):
        self.title = self.run_title
        if self.interactive_input and self.bridge is not None:
            # Commands that ask only on a terminal keep asking, through dialogs.
            self.capture_prints(stdout_tty=True, tee=self.bridge.output)
        else:
            self.capture_prints()
        self.query_one("#errors", DataTable).add_columns("Run", "State", "Book", "Chapter", "Kind", "Error")
        self._mount_profile()
        self.start_command()
        self.set_interval(0.5, self.refresh_view)
        self.refresh_view(force=True)

    def _mount_profile(self):
        """The render layout's tables and its node polling."""
        self.query_one("#books", DataTable).add_columns("Done", "Book", "State", "Chapters", "Active", "Blocks", "Failures")
        self.query_one("#coordinators", DataTable).add_columns("Chapter worker", "Book", "Chapter", "Blocks", "Working on")
        self.query_one("#rows", DataTable).add_columns("Book", "Chapter", "Blocks", "Elapsed", "Left", "Active")
        self.query_one("#workers", DataTable).add_columns("Node", "Role", "Slot", "Book", "Chapter", "Block",
                                                          "Since sent")
        self.query_one("#node-work", DataTable).add_columns("Node", "Batch rows", "Decoding", "Prefilling",
                                                            "Waiting on the node", "Requests on the node",
                                                            "Sent by this render")
        self.query_one("#stages", DataTable).add_columns("Pipeline stage", "Active", "Queued")
        self.query_one("#rejected", DataTable).add_columns("When", "Book", "Chapter", "Block", "Take", "Why")
        # Nodes report their own batch state; asking can take a moment, so never on the UI thread.
        self.set_interval(2.0, lambda: self.run_worker(self._probe_nodes, thread=True, exclusive=True,
                                                       group="node-probe"))

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
        if not self.model.should_hold() or self.exit_when_done:
            self.exit()
            return
        self.refresh_view(force=True)
        if self.model.state == "interrupted":
            self.action_tab("errors" if self.model.current_failures() else "summary")
            self.notify("Stopped. R resumes from where it stopped (completed work is reused); q quits.",
                        severity="warning", timeout=10)
            return
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
        self._draw_glance()
        self._draw_profile()
        failures = []
        for index, item in enumerate(model.visible_failures()):
            scope = item["scope"]
            failures.append((str(index), (str(item["attempt"]), item.get("state", "failed"),
                                          scope.get("book") or "", str(scope.get("chapter") or ""),
                                          item["record"].get("error_type") or item["kind"],
                                          (item["error"] or "").splitlines()[0][:160] if item["error"] else "")))
        sync_table(self.query_one("#errors", DataTable), failures)
        tabs = self.query_one("#tabs", TabbedContent)
        current = len(model.current_failures())
        tabs.get_tab("errors-tab").label = f"Errors ({current})" if current else "Errors"
        self._draw_log()
        self.query_one("#filters", Static).update(Text(
            "Filters: " + (model.filter.describe() or "none") + "   (" + self.FILTER_KEYS + ")"))

    def _draw_profile(self):
        """The render layout's chapters, nodes, pipeline stages and rejected takes."""
        model = self.model
        model.sched = self._sched_status()
        rows = []
        for key, scope, snapshot, text in model.visible_rows():
            if snapshot:
                cells = (scope.get("book") or "", f"{snapshot['chapter']}/{snapshot['total_chapters']}",
                         f"{snapshot['done']}/{snapshot['blocks']} ({round(100 * snapshot['done'] / max(1, snapshot['blocks']))}%)",
                         format_seconds(snapshot.get("elapsed", 0)),
                         f"~{format_seconds(snapshot['remaining'])}" if snapshot.get("remaining") else "",
                         _active_text(snapshot))
            else:
                cells = (scope.get("book") or "", "", "", "", "", text)
            rows.append((str(key), cells))
        sync_table(self.query_one("#rows", DataTable), rows)
        self.query_one("#sched", Static).update(self._sched_line())
        self._draw_nodes()
        stages = (model.sched or {}).get("stages") or {}
        sync_table(self.query_one("#stages", DataTable), [
            (name, (STAGE_NAMES.get(name, name), str(gauges.get("active_now", 0)), str(gauges.get("queued_now", 0))))
            for name, gauges in sorted(stages.items()) if name in STAGE_NAMES])
        sync_table(self.query_one("#rejected", DataTable), [
            (f"{item['request_hash']}:{item['attempt']}",
             (format_seconds(item["time"] - model.started), str(item["scope"].get("book") or ""),
              str(item["scope"].get("chapter") or ""), f"#{item['scope']['block']}" if item["scope"].get("block") else "",
              str(item["attempt"]), (", ".join(item["reasons"]) + (f" ({item['detail']})" if item["detail"] else ""))[:160]))
            for item in model.visible_rejections()[:2000]])
        rejected = len(model.rejections)
        self.query_one("#tabs", TabbedContent).get_tab("rejected-tab").label = (
            f"Rejected ({rejected})" if rejected else "Rejected")

    def _draw_nodes(self):
        """Chapter workers (not tied to a node) and every node's speech/verify slots."""
        model = self.model
        nodes, capacity = self._node_status()
        live = [(key, scope, snapshot) for key, scope, snapshot, _ in model.visible_rows() if snapshot]
        title = (f"Chapter workers: {len(live)} of {capacity} busy" if capacity
                 else f"Chapter workers: {len(live)} busy")
        self.query_one("#coordinators-title", Static).update(
            Text(title + " — each runs one chapter, sending its blocks to any free node"))
        coordinator_rows = [(str(key), (f"#{number}", scope.get("book") or "", f"{snapshot['chapter']}/{snapshot['total_chapters']}",
                                        f"{snapshot['done']}/{snapshot['blocks']}", _active_text(snapshot)))
                            for number, (key, scope, snapshot) in enumerate(live, 1)]
        coordinator_rows += [(f"idle-{number}", (f"#{number}", "", "", "", Text("idle", style="dim")))
                             for number in range(len(live) + 1, (capacity or 0) + 1)]
        sync_table(self.query_one("#coordinators", DataTable), coordinator_rows)
        rows = []
        for name, role, slot, job in model.node_rows(nodes):
            label = "speech" if role == "speech" else "verify"
            if job is None:
                cells = (name, label, str(slot), Text("idle", style="dim"), "", "", "")
            else:
                cells = (name, label, str(slot), str(job.get("book") or ""), str(job.get("chapter") or ""),
                         f"#{job['block']}" if job.get("block") else "", format_seconds(job.get("elapsed", 0)))
            rows.append((f"{name}:{role}:{slot}", cells))
        def number(value):
            return "—" if value is None else str(value)
        sync_table(self.query_one("#node-work", DataTable), [
            (name, (name, number(rows), number(decoding), number(prefilling), number(waiting), number(requests),
                    str(sent)))
            for name, rows, decoding, prefilling, waiting, requests, sent in model.node_work_rows(nodes)])
        self.query_one("#nodes-title", Static).update(Text(
            "This render's requests on each node: speech (TTS) and verify (transcription) slots and the block "
            "each is working on, timed from when it was sent (so it includes waiting on the node)"
            if rows else "Render nodes: none reported (no worker pool; speech and verification run in this process)"))
        sync_table(self.query_one("#workers", DataTable), rows)
        summary = model.node_summary(nodes)
        line = Text()
        for name, roles in summary.items():
            line.append(f"{name}: ", style="bold")
            line.append(" · ".join(f"{'speech' if role == 'speech' else 'verify'} {busy}/{slots}"
                                   for role, (busy, slots) in sorted(roles.items(), key=lambda item: item[0] != "speech")))
            line.append("    ")
        self.query_one("#nodes-line", Static).update(line)

    @staticmethod
    def _probe_nodes():
        try:
            from .renderer.node_activity import ACTIVITY
            ACTIVITY.probe_nodes()
        except Exception:
            pass

    @staticmethod
    def _node_status():
        try:
            from .renderer.node_activity import ACTIVITY
            return ACTIVITY.snapshot(), ACTIVITY.chapter_capacity
        except Exception:
            return [], None

    def _glance(self):
        return self.model.glance()

    def _progress_text(self):
        progress = self.model.progress if self.model.running else None
        return "" if not progress else "\n".join(line for line in (
            f"Step {progress['step']}/{progress['steps']}: {progress['step_name']} "
            f"{progress['step_fraction']:.0%} · {progress['stage']}",
            progress.get("activity", ""), progress.get("totals", "")) if line)

    def _draw_glance(self):
        model = self.model
        for name, (value, detail, level) in self._glance().items():
            tile = self.query_one("#" + _tile_id(name), Static)
            text = Text(name + "\n", style="dim")
            text.append(value, style="bold")
            if detail:
                text.append("\n" + detail, style="dim")
            tile.update(text)
            for other in ("ok", "busy", "warning", "error"):
                tile.set_class(other == level, other)
        self._draw_summary_extra()

    def _draw_summary_extra(self):
        model = self.model
        self.query_one("#progress", Static).update(Text(self._progress_text()))
        sync_table(self.query_one("#books", DataTable), [
            (str(book), (Text("" if percent is None else f"{percent:3d}%", style="bold"),
                         str(book or "this run"), Text(state, style=BOOK_STYLES.get(state, "")),
                         f"{ready}/{total}" if total else (str(ready) if ready else ""),
                         str(active) if active else "", blocks, str(failed) if failed else ""))
            for book, percent, state, ready, total, active, blocks, failed in model.book_rows()])

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
            log.write(_log_line(event, self.model.started))
            self._log_seen = sequence

    @staticmethod
    def _sched_status():
        try:
            from .renderer.render_schedule_status import scheduling_status
            return scheduling_status()
        except Exception:
            return None

    @staticmethod
    def _sched_line():
        try:
            from .renderer.progress import _scheduling_segment
            return _scheduling_segment()
        except Exception:
            return ""

    # Navigation and filters ------------------------------------------------------

    def action_keys(self):
        self.push_screen(KeysScreen(self.BINDINGS, title="Run shell keys"))

    def action_tab(self, name):
        self.query_one("#tabs", TabbedContent).active = self.TABS[name]

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
        table = event.data_table.id
        if table == "books":
            # Drill into one book: its chapters, workers, log and errors.
            key = event.row_key.value
            self.model.filter.book = None if key == "None" else key
            self.action_tab("chapters")
            self.refresh_view(force=True)
        elif table == "rows":
            self._chapter_detail(event.row_key.value)
        elif table == "errors":
            item = self._failure_at(event.row_key.value)
            if item is not None:
                self._open_failure(item)
        elif table == "rejected":
            item = self._highlighted_rejection()
            if item is not None:
                self._open_rejection(item)

    def _chapter_detail(self, key):
        row = next((item for item in self.model.visible_rows() if str(item[0]) == key), None)
        if row is None or not row[2]:
            return
        _, scope, snapshot, _ = row
        lines = [f"{snapshot['done']} of {snapshot['blocks']} blocks accepted, {format_seconds(snapshot['elapsed'])} so far"
                 + (f", about {format_seconds(snapshot['remaining'])} left" if snapshot.get("remaining") else "") + ".", ""]
        for task in snapshot.get("active", []):
            lines.append(f"Block {task['block']}: {task['phase']} for {format_seconds(task.get('elapsed', 0))}"
                         + (f" on {task['worker']}" if task.get("worker") else "")
                         + (f" (voices: {', '.join(task['voices'])})" if task.get("voices") else ""))
        recent = [event.text for _, event in self.model.visible_logs()
                  if event.scope.get("chapter") == snapshot["chapter"]
                  and event.scope.get("book") == scope.get("book")][-15:]
        if recent:
            lines += ["", "Recent messages:", *recent]
        title = (f"[{scope['book']}] " if scope.get("book") else "") + f"Chapter {snapshot['chapter']}"
        self.push_screen(DetailScreen(title, "\n".join(lines)))

    # Failures and repairs ------------------------------------------------------

    def _failure_at(self, key):
        failures = self.model.visible_failures()
        try:
            index = int(key)
        except (TypeError, ValueError):
            return None
        return failures[index] if 0 <= index < len(failures) else None

    def _highlighted_failure(self):
        errors = self.query_one("#errors", DataTable)
        return self._failure_at(highlighted_key(errors))

    def _open_failure(self, item):
        from .run_shell_screens import FailureScreen
        actions = []
        if item.get("state", "failed") == "failed" and self.model.running and callable(item.get("retry")):
            actions.append("retry" if item["kind"] == "chapter" else "retry_book")
        if item.get("repair") is not None:
            actions.append("takes")
        if item["scope"].get("book") is not None and self.resumable:
            actions.append("resume_book")

        def chosen(action):
            if action in ("retry", "retry_book"):
                self._retry(item)
            elif action == "takes":
                self._takes(item)
            elif action == "resume_book":
                self._resume(books=[item["scope"]["book"]])
        self.push_screen(FailureScreen(item, repair_hint(item), actions), chosen)

    def _retry(self, item):
        if self.model.retry(item):
            if item["kind"] == "book":
                self.notify(f"{item['scope'].get('book')} is back at the front of the queue.")
            else:
                self.notify(f"Retrying chapter {item['scope'].get('chapter')} first; other chapters keep going.")
        elif self.model.running:
            self.notify("This failure cannot be retried in place; R resumes after the run.", severity="warning")
        else:
            self.notify("The run has ended; R resumes, reusing completed work.", severity="warning")
        self.refresh_view(force=True)

    def action_retry(self):
        item = self._highlighted_failure()
        if item is None:
            self.notify("Highlight a failed chapter or book in Errors (5) to retry it", severity="warning")
            return
        self._retry(item)

    def action_takes(self):
        tab = self.query_one("#tabs", TabbedContent).active
        if tab == "rejected-tab":
            item = self._highlighted_rejection()
            if item is not None:
                self._open_rejection(item)
            return
        item = self._highlighted_failure()
        if item is None or item.get("repair") is None:
            self.notify("Highlight a chapter that failed on one passage in Errors (5), or a take in Rejected (6)",
                        severity="warning")
            return
        self._takes(item)

    def _takes(self, item):
        self._open_takes(item["scope"], item["repair"], failure=item,
                         title_tail="takes of the passage that failed")

    def _open_takes(self, scope, review, *, failure=None, highlight=None, title_tail="every take of this passage"):
        from .run_shell_screens import TakesScreen
        if self._player is None:
            from .renderer.audition import AudioPlayer
            self._player = AudioPlayer(review.player_command)
        title = ((f"[{scope['book']}] " if scope.get("book") else "") +
                 f"Chapter {scope.get('chapter')}" + (f", block {scope['block']}" if scope.get("block") else "") +
                 f": {title_tail}")

        def chosen(choice):
            if choice is not None:
                self.run_worker(lambda: self._use_take(scope, review, choice, failure), thread=True,
                                group="use-take")
        self.push_screen(TakesScreen(title, review, self._player, highlight=highlight), chosen)

    def _use_take(self, scope, review, choice, failure=None):
        chapter = scope.get("chapter")
        try:
            outcome = review.accept(choice["attempt"], learn=choice["learn"])
        except (ValueError, OSError) as error:
            self.call_from_thread(self.notify, f"Could not use take {choice['attempt']}: {error}", severity="error")
            self.bus.log(f"Could not use take {choice['attempt']} for chapter {chapter}: {error}",
                         scope=dict(scope), severity="error")
            return
        learned = "; ".join(f"{expected} → {heard}" for expected, heard in outcome["learned"])
        self.bus.log(f"Using take {outcome['attempt']} for chapter {chapter} after listening "
                     f"({outcome['basis']})" + (f"; the recognizer learned: {learned}" if learned else ""),
                     scope=dict(scope), severity="warning")
        if failure is None:
            # A failed chapter whose passage this is retries with the take; any other chapter is rebuilt.
            failure = next((item for item in self.model.current_failures()
                            if item["record"].get("request_hash") == review.request_hash), None)
        if failure is not None and self.model.retry(failure):
            message = f"Take {outcome['attempt']} is used; chapter {chapter} renders again with it, first in line."
        else:
            when = self.model.request_rebuild(scope.get("book"), chapter)
            message = {"now": f"Take {outcome['attempt']} is used; chapter {chapter} is rebuilt with it, first in line.",
                       "after": f"Take {outcome['attempt']} is used; chapter {chapter} is rebuilt with it as soon as "
                                "it finishes rendering."}.get(
                when, f"Take {outcome['attempt']} is used; R (or the next render) rebuilds chapter {chapter} with it.")
        self.call_from_thread(self.notify, message)
        self.call_from_thread(self.refresh_view, True)

    # Rejected takes ----------------------------------------------------------------

    def _highlighted_rejection(self):
        table = self.query_one("#rejected", DataTable)
        key = highlighted_key(table)
        return next((item for item in self.model.visible_rejections()
                     if f"{item['request_hash']}:{item['attempt']}" == key), None)

    def _open_rejection(self, item):
        if item.get("review") is None:
            self.notify("This take cannot be reviewed here", severity="warning")
            return
        self._open_takes(item["scope"], item["review"], highlight=item["attempt"])

    def action_export_feedback(self):
        works = {}
        for item in self.model.rejections:
            review = item.get("review")
            if review is not None:
                works.setdefault(str(review.work), review.work)
        for failure in self.model.failures:
            review = failure.get("repair")
            if review is not None:
                works.setdefault(str(review.work), review.work)
        if not works:
            self.notify("No reviewed takes yet", severity="warning")
            return
        from .renderer.take_feedback import export
        for work in works.values():
            try:
                path, counts = export(work)
            except (ValueError, OSError) as error:
                self.notify(f"Could not write the feedback review: {error}", severity="error")
                continue
            self.bus.log(f"Feedback review written: {path} ({counts['false_rejection']} wrongly rejected, "
                         f"{counts['false_acceptance']} wrongly passed)", severity="warning")
            self.notify(f"Wrote {path}")

    def action_finish(self):
        if not self.model.holds:
            self.notify("Nothing is waiting for retries", severity="warning")
            return
        self.push_screen(ConfirmScreen(
            "Stop waiting for retries? The run then reports the chapters that still failed (no audiobook "
            "is assembled for those books); R resumes them later.", title="Finish the run?", danger=True,
            yes="Finish"), lambda yes: yes and self.model.release_holds())

    # Stopping (always confirmed) --------------------------------------------------

    def _stop(self, *, then_exit=False):
        """Ask the running command to stop at safe boundaries (a second request stops at once)."""
        self.exit_when_done = self.exit_when_done or then_exit
        if self.model.holds:
            self.model.release_holds()
            return
        if self.request_stop is None:
            return
        self.model.stopping = True
        self.model.changed.set()
        self.request_stop()

    def action_interrupt(self):
        if not self.model.running:
            self.action_quit_run()
            return
        if self.model.stopping:
            self.push_screen(ConfirmScreen(
                "The run is already stopping at safe boundaries. Stop immediately instead? Takes in progress are "
                "lost; everything verified and saved stays and is reused on resume.",
                title="Stop immediately?", danger=True, yes="Stop now"), lambda yes: yes and self._stop())
            return
        self.push_screen(ConfirmScreen(
            "Stop the run? Active takes finish verification and are saved first; R or rerunning the same "
            "command resumes from there.", title="Stop the run?", danger=True, yes="Stop"),
            lambda yes: yes and (self._stop(), self.notify("Stopping at safe boundaries…", severity="warning")))

    def action_quit_run(self):
        if not self.model.running:
            self.exit()
            return
        if self.model.holds:
            message = ("The run is waiting for you to retry failed chapters. Quit? It reports what still failed "
                       "and closes.")
        else:
            message = ("The command is still running. Stop it at safe boundaries and quit when it has stopped? "
                       "Rerunning the same command resumes from what was saved.")
        self.push_screen(ConfirmScreen(message, title="Quit and stop the run?", danger=True, yes="Stop and quit"),
                         lambda yes: yes and self._stop(then_exit=True))

    # Resuming ------------------------------------------------------------------

    def action_resume(self):
        item = self._highlighted_failure() if self.query_one("#tabs", TabbedContent).active == "errors-tab" else None
        books = [item["scope"]["book"]] if item is not None and item["scope"].get("book") else None
        self._resume(books=books)

    def _resume(self, books=None):
        if self.model.running:
            self.notify("Still running; resume is for a run that has ended", severity="warning")
            return
        if not self.resumable:
            self.notify("Running this command again would redo finished work; rerun it yourself if you mean to.",
                        severity="warning", timeout=8)
            return
        command = self.resume_command(self.model, books) if self.resume_command is not None else None
        self.model.restart()
        self.bus.log(f"Resuming: attempt {self.model.attempt}" + (f" ({', '.join(books)})" if books else ""),
                     severity="warning")
        self.action_tab("summary")
        if command is not None:
            # Commands that start fresh on rerun resume the run they printed; a
            # multi-book render resumes only the books that did not finish.
            self.command = command
        self.start_command()


def _tile_id(name):
    return "tile-" + name.lower().replace(" ", "-")


class TaskApp(RunApp, inherit_bindings=False):
    """The run shell for commands that are not renders: analysis, attribution review, assembly, imports.

    Their work is a sequence of steps with model requests, not books of
    chapters on render nodes, so the Summary shows the run's health, its
    current step and live model requests, warnings and errors, and the
    latest output; the Log and Errors views are the same as for a render.
    """

    PROFILE = "task"
    FILTER_KEYS = "v level · / text · x clear"
    TILES = ("Health", "Step", "Model requests", "Warnings", "Errors", "Elapsed")
    TABS = {"summary": "summary", "log": "log-tab", "errors": "errors-tab"}
    CSS = RunApp.CSS + """
    #glance { grid-size: 3 2; }
    #recent { height: 1fr; border: round $primary-background; }
    """
    BINDINGS = [
        Binding("q", "quit_run", "Quit", tooltip="Quit; while running this asks before stopping anything"),
        Binding("ctrl+q", "quit_run", "Quit", show=False, priority=True, tooltip="Same as q"),
        Binding("question_mark", "keys", "Keys", tooltip="Show every key"),
        Binding("1", "tab('summary')", "Summary", tooltip="At a glance: health, step, model requests, latest output"),
        Binding("2", "tab('log')", "Log", tooltip="Every message, filtered"),
        Binding("3", "tab('errors')", "Errors", tooltip="Every failure with its detail"),
        Binding("v", "severity", "Level", tooltip="Cycle the message level: info, warnings, errors, everything"),
        Binding("slash", "search", "Search", tooltip="Filter messages by text"),
        Binding("x", "clear_filters", "Clear", tooltip="Clear every filter"),
        Binding("R", "resume", "Resume", tooltip="Run the same command again; completed work is reused"),
        Binding("ctrl+c", "interrupt", "Stop", priority=True, show=False,
                tooltip="Stop, after confirming (completed work is kept)"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent_seen = 0

    def _compose_summary(self):
        yield Static("", id="progress")
        yield Static(Text("Latest output (2 shows everything, with filters)"), classes="section-title")
        yield RichLog(id="recent", wrap=True, markup=False, highlight=False)

    def _mount_profile(self):
        pass

    def _draw_profile(self):
        pass

    def _glance(self):
        return self.model.task_glance()

    def _draw_summary_extra(self):
        self.query_one("#progress", Static).update(Text(self._progress_text()))
        recent = self.query_one("#recent", RichLog)
        for sequence, event in self.model.visible_logs(after=self._recent_seen)[-500:]:
            recent.write(_log_line(event, self.model.started))
            self._recent_seen = sequence


def _log_line(event, started):
    line = Text(format_seconds(event.time - started) + " ", style="dim")
    if event.scope.get("book"):
        line.append(f"[{event.scope['book']}] ", style="bold")
    if event.scope.get("chapter") is not None:
        line.append(f"Ch {event.scope['chapter']}: ", style="bold")
    line.append(event.text, style=SEVERITY_STYLES.get(event.severity, ""))
    return line


STAGE_NAMES = {"synthesis": "Generating speech", "synthesis_slot": "Waiting for a speech slot",
               "recognition": "Transcribing (verification)", "review": "Reviewing transcripts",
               "preparation": "Preparing audio", "admission": "Waiting to start",
               "review_coordination": "Review coordination", "model_thread": "Model thread"}
BOOK_STYLES = {"rendering": "bold cyan", "assembling": "cyan", "preparing": "cyan", "queued": "dim",
               "done": "green", "failed": "bold red", "stopped": "yellow"}


def _active_text(snapshot):
    return "; ".join(
        f"#{task['block']} {(task.get('worker') + ' ') if task.get('worker') else ''}{task['phase']} "
        f"{format_seconds(task.get('elapsed', 0))}" for task in snapshot.get("active", [])) or "idle"


# How long a stopping screen may take to give the terminal back before the process exits anyway.
FORCED_EXIT_GRACE_SECONDS = 3.0


def _render_interrupts(app, *, exit_process=None, grace=FORCED_EXIT_GRACE_SECONDS):
    """Renderer commands: Ctrl-C keys drive ``RenderInterrupts`` exactly like SIGINT would.

    A second stop closes the screen the ordinary way, so Textual restores the
    terminal (its restore sequences go through a writer thread that an
    immediate ``os._exit`` would cut off), and ``run_in_shell`` then exits the
    process. Should the screen not close within ``grace`` seconds, the process
    exits anyway.
    """
    @contextmanager
    def scope():
        from .renderer.interrupts import hard_exit, keyboard_interrupts
        leave = exit_process or hard_exit

        def force_exit(code):
            app.forced_exit = code
            if not app.is_running:
                leave(code)
                return
            timer = threading.Timer(grace, leave, (code,))
            timer.daemon = True
            timer.start()
            try:
                app.call_from_thread(app.exit)
            except RuntimeError:  # Already on the screen's own thread.
                app.exit()
        with keyboard_interrupts(lambda message: print(message, file=sys.stderr), force_exit=force_exit) as controller:
            yield controller.request
    return scope


def run_in_shell(command, *, title, graceful_render_stop=False, interactive_input=False, resumable=True,
                 resume_command=None, closing_output=None, layout="task"):
    """Run ``command()`` inside the run shell; return its result (or re-raise its error).

    Render boards, prints and terminal prompts go to the screen while it is
    open. With ``graceful_render_stop`` Ctrl-C stops a render at safe
    boundaries (a second press exits at once); otherwise it raises
    ``KeyboardInterrupt`` in the command. With ``interactive_input`` the
    command's own terminal questions (``sys.stdin.readline`` after a printed
    question) are asked in dialogs. ``resumable`` allows ``R`` to run the
    command again; ``resume_command(model)`` may return a different command
    for that (one that resumes the printed run ID). When the screen closes, the final messages and every error
    are printed so the terminal keeps a record. ``layout`` is ``"render"``
    for the render dashboard, else the task layout (step, model requests,
    latest output).
    """
    from .renderer.progress import shell_board
    bus = EventBus()
    model = RunModel(title)
    bus.subscribe(model.on_event)
    from .run_events import set_active_bus
    bridge = PromptBridge(None, on_answer=lambda answer: bus.log("> " + (answer or "(no answer)")))
    # A render's dashboard (books, chapters, nodes, rejected takes) or the plainer task layout.
    app_class = RunApp if layout == "render" else TaskApp
    app = app_class(model, bus, command, title=title, interactive_input=interactive_input,
                 resumable=resumable, bridge=bridge, resume_command=resume_command)
    bridge.app = app
    if graceful_render_stop:
        app.interrupt_scope = _render_interrupts(app)
    previous_bus, previous_stdin = set_active_bus(bus), sys.stdin
    try:
        if interactive_input:
            sys.stdin = bridge.input
        with shell_board(bus), prompt_streams(bridge.input, bridge.output), screen_host(bridge.show):
            app.run()
    finally:
        sys.stdin = previous_stdin
        set_active_bus(previous_bus)
    output = closing_output or (lambda line: print(line, file=sys.stderr))
    for line in model.closing_lines():
        output(line)
    if app.forced_exit is not None:
        # A second stop: the terminal is restored; blocked workers must not delay the exit.
        from .renderer.interrupts import hard_exit
        hard_exit(app.forced_exit)
    if app.raised is not None:
        raise app.raised
    if model.running:
        # The screen closed while the command was still running (terminal lost).
        return 130
    return app.result
