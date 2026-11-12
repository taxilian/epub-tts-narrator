"""Reusable Textual pieces for every full-screen interface in this project.

Import this module only after ``tui_support.wants_tui`` said yes: Textual is an
optional extra, and nothing else in the package imports it at module level.

* ``ConfirmScreen``, ``TextEditScreen``, ``ChoiceScreen``, ``PromptScreen`` —
  modal dialogs that return their answer through ``dismiss``.
* ``PromptBridge`` — file-like input/output streams that turn a worker
  thread's line prompt (``readline``) into a ``PromptScreen``, so code written
  for a terminal (model download confirmation, speaker choices) can run under a
  full-screen UI unchanged.
* ``sync_table`` — refill a ``DataTable`` by row key, keeping the cursor on
  the same row.
* ``MessageLog`` — a scrolling, severity-colored log fed from ``(time,
  severity, text)`` records.
* ``JobsTable`` — a ``DataTable`` showing a ``tui_jobs.JobQueue``.
"""

from __future__ import annotations

import re
import sys
import threading
import time

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option

from .tui_jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING


BASE_CSS = """
.dialog {
    width: 80%;
    max-width: 110;
    height: auto;
    max-height: 90%;
    padding: 1 2;
    border: thick $accent;
    background: $panel;
}
.dialog-title { text-style: bold; margin-bottom: 1; }
.dialog-help { color: $text-muted; margin-top: 1; }
.dialog-error { color: $error; margin-top: 1; }
.dialog-buttons { height: auto; margin-top: 1; align-horizontal: right; }
.dialog-buttons Button { margin-left: 2; }
TextEditScreen TextArea { height: 12; }
ChoiceScreen OptionList { height: auto; max-height: 20; }
ModalScreen { align: center middle; }
"""

SEVERITY_STYLES = {"error": "bold red", "warning": "yellow", "info": "", "debug": "dim"}
JOB_STYLES = {QUEUED: "dim", RUNNING: "bold cyan", DONE: "green", FAILED: "bold red", CANCELLED: "yellow"}


class _LineWriter:
    """A text stream that hands each complete line to ``deliver(line)`` in the writing thread."""

    def __init__(self, deliver, *, tty=False, tee=None):
        self._deliver = deliver
        self._partial = {}
        self._lock = threading.Lock()
        self._tty = tty
        self._tee = tee
        self.encoding = "utf-8"

    def write(self, text):
        if self._tee is not None:
            self._tee.write(text)
        key = threading.get_ident()
        with self._lock:
            pending = self._partial.pop(key, "") + str(text)
            *lines, rest = pending.split("\n")
            if rest:
                self._partial[key] = rest
        for line in lines:
            line = _visible(line)
            if line.strip():
                self._deliver(line)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return self._tty


_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")


def _visible(line):
    """What a terminal would show: the text after the last carriage return, without escapes."""
    return _ESCAPE.sub("", line.rstrip("\r").rsplit("\r", 1)[-1])


class CapturePrintMixin:
    """For an ``App``: send ``print`` output from any thread to ``on_captured_line(text, stderr)``.

    Code written for a terminal prints freely; under a full-screen UI those
    lines would be lost or garble the screen, so they become log records.
    Lines are delivered synchronously in the printing thread, so they keep
    their order relative to other messages that thread reports. Textual
    restores the original streams when the app exits.
    """

    def capture_prints(self, *, stdout_tty=False, tee=None):
        """``stdout_tty``: report stdout as a terminal, for commands that only
        ask questions when it is one (their questions then go through ``tee``,
        a ``PromptBridge`` output, and the bridge's input)."""
        sys.stdout = _LineWriter(lambda line: self.on_captured_line(line, False), tty=stdout_tty, tee=tee)
        sys.stderr = _LineWriter(lambda line: self.on_captured_line(line, True), tee=tee)

    def on_captured_line(self, text, stderr):  # pragma: no cover - overridden
        pass


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class ConfirmScreen(ModalScreen[bool]):
    """Ask a yes/no question; ``y``/Enter confirms, ``n``/Escape declines."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]

    def __init__(self, message, *, title="Confirm", yes="Yes", no="No"):
        super().__init__()
        self.message, self.title_text, self.yes, self.no = message, title, yes, no

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(self.message)
            with Horizontal(classes="dialog-buttons"):
                yield Button(self.no, id="no")
                yield Button(self.yes, id="yes", variant="primary")

    def on_mount(self):
        self.query_one("#yes", Button).focus()

    def action_answer(self, value):
        self.dismiss(bool(value))

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "yes")


class KeysScreen(ModalScreen):
    """List every key an app binds, including ones the footer has no room for."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("escape,question_mark,q", "close", "Close", priority=True)]

    def __init__(self, bindings, *, title="Keys"):
        super().__init__()
        self.title_text = title
        self.rows = []
        for binding in bindings:
            keys = ", ".join(_key_name(key) for key in binding.key.split(","))
            detail = binding.description
            if binding.tooltip and binding.tooltip != binding.description:
                detail += " — " + binding.tooltip
            self.rows.append((keys, detail))
        self.rows.append(("Tab", "Move between the panels"))

    def compose(self) -> ComposeResult:
        text = Text()
        width = max((len(keys) for keys, _ in self.rows), default=0)
        for keys, description in self.rows:
            text.append(keys.ljust(width + 2), style="bold")
            text.append(description + "\n")
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Static(text)
            yield Static("Escape closes", classes="dialog-help")

    def action_close(self):
        self.dismiss(None)


class DetailScreen(ModalScreen):
    """Show a long text (an error's saved detail, a record) in a scrollable dialog."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("escape,enter,q", "close", "Close", priority=True)]

    def __init__(self, title, text):
        super().__init__()
        self.title_text, self.text = title, text

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            with VerticalScroll():
                yield Static(Text(self.text), id="detail-text")
            yield Static("Escape closes", classes="dialog-help")

    def action_close(self):
        self.dismiss(None)


_KEY_NAMES = {"slash": "/", "space": "Space", "enter": "Enter", "escape": "Esc", "question_mark": "?",
              "tab": "Tab"}


def _key_name(key):
    key = key.strip()
    if key in _KEY_NAMES:
        return _KEY_NAMES[key]
    if key.startswith("ctrl+"):
        return "Ctrl+" + key[5:].upper()
    return key


class TextEditScreen(ModalScreen):
    """Edit multi-line text; Ctrl+S saves, Escape cancels (returns ``None``).

    ``validate(text)`` may raise ``ValueError``; its message is shown and the
    dialog stays open, so nothing invalid is ever handed back.
    """

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("ctrl+s", "save", "Save", priority=True),
                Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, title, value, *, help_text="", validate=None, single_line=False):
        super().__init__()
        self.title_text, self.value, self.help_text = title, value, help_text
        self.validate = validate
        self.single_line = single_line

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.single_line:
                yield Input(self.value, id="editor")
            else:
                yield TextArea(self.value, id="editor", soft_wrap=True)
            yield Static("", id="error", classes="dialog-error")
            yield Static(self.help_text or "Ctrl+S saves · Escape cancels", classes="dialog-help")

    def on_mount(self):
        self.query_one("#editor").focus()

    def text(self):
        editor = self.query_one("#editor")
        return editor.value if isinstance(editor, Input) else editor.text

    def action_save(self):
        text = self.text()
        if self.single_line:
            text = text.strip()
        if self.validate is not None:
            try:
                self.validate(text)
            except ValueError as error:
                self.query_one("#error", Static).update(str(error))
                return
        self.dismiss(text)

    def on_input_submitted(self, event):
        self.action_save()

    def action_cancel(self):
        self.dismiss(None)


class ChoiceScreen(ModalScreen):
    """Choose one of ``options`` (``[(value, label)]``); type to filter, Escape cancels."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True),
                Binding("down", "move(1)", "Next", show=False), Binding("up", "move(-1)", "Previous", show=False)]

    def __init__(self, title, options, *, help_text="", filterable=True):
        super().__init__()
        self.title_text, self.options, self.help_text = title, list(options), help_text
        self.filterable = filterable

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.filterable:
                yield Input(placeholder="Type to filter", id="choice-filter")
            yield OptionList(*self._options(""), id="choices")
            yield Static(self.help_text or "Enter chooses · Escape cancels", classes="dialog-help")

    def _options(self, text):
        words = text.casefold().split()
        return [Option(label, id=str(index)) for index, (_, label) in enumerate(self.options)
                if all(word in label.casefold() for word in words)]

    def on_mount(self):
        self.query_one("#choice-filter" if self.filterable else "#choices").focus()

    def action_move(self, delta):
        """Up/Down while typing in the filter move the list's highlight."""
        choices = self.query_one("#choices", OptionList)
        if choices.option_count:
            current = choices.highlighted if choices.highlighted is not None else -1 if delta > 0 else 0
            choices.highlighted = max(0, min(choices.option_count - 1, current + delta))

    def on_input_changed(self, event):
        choices = self.query_one("#choices", OptionList)
        choices.clear_options()
        choices.add_options(self._options(event.value))
        if choices.option_count:
            choices.highlighted = 0

    def on_input_submitted(self, event):
        choices = self.query_one("#choices", OptionList)
        if choices.option_count:
            index = choices.highlighted or 0
            self.dismiss(self.options[int(choices.get_option_at_index(index).id)][0])

    def on_option_list_option_selected(self, event):
        self.dismiss(self.options[int(event.option.id)][0])

    def action_cancel(self):
        self.dismiss(None)


class PromptScreen(ModalScreen):
    """Show a terminal prompt's text and return one typed line (Escape: empty answer)."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, text):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("Input needed", classes="dialog-title")
            yield Static(Text(self.text))
            yield Input(id="answer")
            yield Static("Enter answers · Escape answers with nothing", classes="dialog-help")

    def on_mount(self):
        self.query_one("#answer", Input).focus()

    def on_input_submitted(self, event):
        self.dismiss(event.value)

    def action_cancel(self):
        self.dismiss("")


class _BridgeOutput:
    def __init__(self, bridge):
        self._bridge = bridge

    def write(self, text):
        with self._bridge.lock:
            # Bounded: long runs print far more than any one question needs.
            self._bridge.pending = (self._bridge.pending + str(text))[-20000:]
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return True


class _BridgeInput:
    def __init__(self, bridge):
        self._bridge = bridge

    def readline(self):
        return self._bridge.ask() + "\n"

    def isatty(self):
        return True


class PromptBridge:
    """Line prompts from a worker thread, answered in a ``PromptScreen``.

    Text written to ``output`` accumulates until the worker calls
    ``input.readline()``; the accumulated text becomes the dialog's question.
    Must never be used from the UI thread itself (that would deadlock).
    """

    CONTEXT_LINES = 30

    def __init__(self, app, *, on_answer=None):
        self.app = app
        self.lock = threading.Lock()
        self.pending = ""
        self.on_answer = on_answer
        self.output = _BridgeOutput(self)
        self.input = _BridgeInput(self)

    def ask(self):
        with self.lock:
            # Only the latest lines are the question's context; the full text is in the log.
            lines = [_visible(line) for line in self.pending.split("\n")]
            text = "\n".join(line for line in lines if line.strip())
            text = "\n".join(text.split("\n")[-self.CONTEXT_LINES:])
            self.pending = ""
        answered = threading.Event()
        result = {"value": ""}

        def done(value):
            result["value"] = value or ""
            answered.set()

        self.app.call_from_thread(self.app.push_screen, PromptScreen(text), done)
        answered.wait()
        if self.on_answer is not None:
            self.on_answer(result["value"])
        return result["value"]


def sync_table(table: DataTable, rows):
    """Replace a table's rows (``[(key, cells)]``) and keep the cursor on the same key."""
    current = None
    if table.row_count:
        try:
            current = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            current = None
    table.clear()
    for key, cells in rows:
        table.add_row(*cells, key=key)
    keys = [key for key, _ in rows]
    if current in keys:
        table.move_cursor(row=keys.index(current))
    return keys


def highlighted_key(table: DataTable):
    if not table.row_count:
        return None
    try:
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
    except Exception:
        return None


class MessageLog(RichLog):
    """A log fed from ``(time, severity, text)`` records; only new records are written."""

    def __init__(self, **kwargs):
        super().__init__(wrap=True, markup=False, highlight=False, **kwargs)
        self._written = 0
        self._started = time.monotonic()

    def sync(self, messages, *, started=None):
        messages = list(messages)
        if self._written > len(messages):
            self._written = 0
        origin = self._started if started is None else started
        for stamp, severity, text in messages[self._written:]:
            line = Text(format_seconds(stamp - origin) + " ", style="dim")
            line.append(text, style=SEVERITY_STYLES.get(severity, ""))
            self.write(line)
        self._written = len(messages)


class JobsTable(DataTable):
    """Background jobs: state, elapsed time, label, and a short error."""

    def on_mount(self):
        self.cursor_type = "row"
        self.add_columns("State", "Time", "Job", "Error")

    def show(self, jobs, now=None):
        now = time.monotonic() if now is None else now
        rows = []
        for job in jobs:
            state = Text(job.state, style=JOB_STYLES.get(job.state, ""))
            rows.append((str(job.job_id), (state, format_seconds(job.elapsed(now)), job.label,
                                           (job.error or "").splitlines()[0][:120] if job.error else "")))
        return sync_table(self, rows)
