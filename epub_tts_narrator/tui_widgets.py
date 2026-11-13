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


class Dialog(ModalScreen):
    """A modal screen that closes once.

    Two selections can reach a dialog before it closes (a held or bouncing
    Enter, a double click). Textual would then run the result callback a
    second time and fail popping a screen that is already gone, so every
    dismissal after the first is ignored. Nothing awaits ``dismiss`` here.
    """

    _dismissed_once = False

    def dismiss(self, result=None):
        if self._dismissed_once:
            return None
        self._dismissed_once = True
        return super().dismiss(result)


class ConfirmScreen(Dialog):
    """Ask a yes/no question; ``y``/Enter confirms, ``n``/Escape declines."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n,escape", "answer(False)", "No")]

    def __init__(self, message, *, title="Confirm", yes="Yes", no="No", danger=False):
        """``danger``: something that stops work; No is focused, so Enter or Escape keeps it running."""
        super().__init__()
        self.message, self.title_text, self.yes, self.no = message, title, yes, no
        self.danger = danger

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Static(Text(self.message))
            with Horizontal(classes="dialog-buttons"):
                yield Button(self.no, id="no", variant="primary" if self.danger else "default")
                yield Button(self.yes, id="yes", variant="error" if self.danger else "primary")
            if self.danger:
                yield Static("Press y (or choose the button) to confirm; Enter or Escape keeps running.",
                             classes="dialog-help")

    def on_mount(self):
        self.query_one("#no" if self.danger else "#yes", Button).focus()

    def action_answer(self, value):
        self.dismiss(bool(value))

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "yes")


class KeysScreen(Dialog):
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
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Static(text)
            yield Static("Escape closes", classes="dialog-help")

    def action_close(self):
        self.dismiss(None)


class DetailScreen(Dialog):
    """Show a long text (an error's saved detail, a record) in a scrollable dialog."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("escape,enter,q", "close", "Close", priority=True)]

    def __init__(self, title, text):
        super().__init__()
        self.title_text, self.text = title, text

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
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


class TextEditScreen(Dialog):
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
            yield Label(Text(self.title_text), classes="dialog-title")
            if self.single_line:
                yield Input(self.value, id="editor")
            else:
                yield TextArea(self.value, id="editor", soft_wrap=True)
            yield Static("", id="error", classes="dialog-error")
            yield Static(Text(self.help_text or "Ctrl+S saves · Escape cancels"), classes="dialog-help")

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
                self.query_one("#error", Static).update(Text(str(error)))
                return
        self.dismiss(text)

    def on_input_submitted(self, event):
        self.action_save()

    def action_cancel(self):
        self.dismiss(None)


class ChoiceScreen(Dialog):
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
            yield Label(Text(self.title_text), classes="dialog-title")
            if self.filterable:
                yield Input(placeholder="Type to filter", id="choice-filter")
            yield OptionList(*self._options(""), id="choices")
            yield Static(Text(self.help_text or "Enter chooses · Escape cancels"), classes="dialog-help")

    def _options(self, text):
        words = text.casefold().split()
        return [Option(Text(label), id=str(index)) for index, (_, label) in enumerate(self.options)
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


_CHOICE = re.compile(r"\[([A-Za-z0-9])\]\s*([^/\[]*?)\s*(?=/|\[|$)")
_YES_NO = re.compile(r"\[(y/N|Y/n|y/n)\]\s*:?\s*$")


def prompt_choices(question):
    """``[(key, label)]`` from a question like ``[A] apply the new directions / [L] leave it``.

    A yes/no question (``… now? [y/N]``) gets Yes and No, the default first.
    """
    yes_no = _YES_NO.search(question.rstrip())
    if yes_no:
        choices = [("y", "Yes"), ("n", "No")]
        return choices[::-1] if yes_no.group(1) == "y/N" else choices
    text = question.rstrip().rstrip(":").rstrip()
    choices = [(key, label.strip(" :")) for key, label in _CHOICE.findall(text)]
    return choices if len(choices) >= 2 else []


class PromptScreen(Dialog):
    """A question the command asks on its terminal, answered in a dialog.

    The command's recent output scrolls above; the question itself always
    stays visible at the bottom. Lettered choices (``[A] … / [L] …``) become
    buttons, and pressing a choice's letter answers at once; any other
    question takes a typed line. Escape ends input, exactly like Ctrl-D on a
    terminal (the commands treat it as leaving things unchanged); it
    dismisses with ``None``.
    """

    DEFAULT_CSS = BASE_CSS + """
    PromptScreen .dialog { height: 90%; }
    PromptScreen #prompt-context { height: 1fr; border: round $primary-background; padding: 0 1; }
    PromptScreen #prompt-question { text-style: bold; margin-top: 1; }
    PromptScreen #prompt-choices { height: auto; margin-top: 1; }
    PromptScreen #prompt-choices Button { margin-right: 2; }
    """
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, text, question=None):
        super().__init__()
        if question is None:
            lines = text.rstrip("\n").split("\n")
            question, text = lines[-1], "\n".join(lines[:-1])
        self.text, self.question = text, question.strip()
        self.choices = prompt_choices(self.question)

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll
        with Vertical(classes="dialog"):
            yield Label("The command asks", classes="dialog-title")
            if self.text.strip():
                with VerticalScroll(id="prompt-context"):
                    yield Static(Text(self.text))
            yield Static(Text(self.question.strip() or "Answer:"), id="prompt-question")
            if self.choices:
                with Horizontal(id="prompt-choices"):
                    for key, label in self.choices:
                        yield Button(Text(f"[{key}] {label}"), id=f"choice-{key}")
            yield Input(id="answer", placeholder="Or type an answer and press Enter" if self.choices else
                        "Type an answer and press Enter")
            yield Static(Text(("Press a choice's letter (or its button) to answer · " if self.choices else "")
                              + "Escape ends input, which leaves things unchanged"), classes="dialog-help")

    def on_mount(self):
        if self.text.strip():
            self.query_one("#prompt-context").scroll_end(animate=False)
        if self.choices:
            self.query_one(f"#choice-{self.choices[0][0]}", Button).focus()
        else:
            self.query_one("#answer", Input).focus()

    def on_key(self, event):
        # A choice's letter answers at once, unless the person is typing in the box.
        if not self.choices or isinstance(self.focused, Input):
            return
        key = (event.character or "").casefold()
        match = next((choice for choice, _ in self.choices if choice.casefold() == key), None)
        if match is not None:
            event.stop()
            self.dismiss(match)

    def on_button_pressed(self, event):
        if event.button.id and event.button.id.startswith("choice-"):
            self.dismiss(event.button.id.removeprefix("choice-"))

    def on_input_submitted(self, event):
        self.dismiss(event.value)

    def action_cancel(self):
        self.dismiss(None)


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
        answer = self._bridge.ask()
        # Escape in the dialog ends input, as end-of-file does on a terminal.
        return "" if answer is None else answer + "\n"

    def isatty(self):
        return True


class PromptBridge:
    """Line prompts from a worker thread, answered in a ``PromptScreen``.

    Text written to ``output`` accumulates until the worker calls
    ``input.readline()``; the accumulated text becomes the dialog's question.
    Must never be used from the UI thread itself (that would deadlock).
    """

    CONTEXT_LINES = 400

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
        result = {"value": None}

        def done(value):
            result["value"] = value
            answered.set()

        self.app.call_from_thread(self.app.push_screen, PromptScreen(text), done)
        answered.wait()
        if self.on_answer is not None:
            self.on_answer(result["value"] if result["value"] is not None else "(input ended)")
        return result["value"]

    def show(self, make_screen):
        """Open ``make_screen()`` (built on the UI thread) and wait for its dismissal value."""
        answered = threading.Event()
        result = {"value": None}

        def done(value):
            result["value"] = value
            answered.set()

        self.app.call_from_thread(lambda: self.app.push_screen(make_screen(), done))
        answered.wait()
        return result["value"]


def _cell(value):
    # Plain strings are shown literally: a book or voice name may contain [brackets].
    return Text(value) if isinstance(value, str) else value


def _same(before, after):
    """Whether a cell would look the same (rich's ``Text ==`` ignores the text's own style)."""
    if isinstance(before, Text) and isinstance(after, Text):
        return before == after and str(before.style) == str(after.style) and before.justify == after.justify
    return type(before) is type(after) and before == after


def sync_table(table: DataTable, rows):
    """Show ``rows`` (``[(key, cells)]``) in ``table`` without moving what the person is looking at.

    The same keys in the same order (most updates: a status changes) only
    update the cells that changed, so the cursor and scroll position stay
    exactly where they are. Added, removed or reordered rows rebuild the
    table, then keep the highlighted row on the same line of the screen
    (``clear`` alone would scroll back to the top).
    """
    keys = [key for key, _ in rows]
    current = [row_key.value for row_key in table.rows]
    if current == keys:
        columns = [column.key for column in table.ordered_columns]
        for key, cells in rows:
            shown = table.get_row(key)
            for column, before, value in zip(columns, shown, map(_cell, cells)):
                if not _same(before, value):
                    table.update_cell(key, column, value)
        return keys
    cursor, line = None, 0
    if table.row_count:
        try:
            cursor = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
            line = table.cursor_row - round(table.scroll_y)
        except Exception:
            cursor = None
    scroll_y = table.scroll_y
    table.clear()
    for key, cells in rows:
        table.add_row(*map(_cell, cells), key=key)
    if cursor in keys:
        index = keys.index(cursor)
        table.move_cursor(row=index, scroll=False)
        target = max(0, index - line)
    else:
        target = scroll_y
    # The new rows' heights are measured on the next refresh; scroll then.
    table.call_after_refresh(table.scroll_to, y=target, animate=False)
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
