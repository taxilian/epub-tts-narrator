"""Reusable Textual pieces for every full-screen interface in this project.

Import this module only after ``tui_support.wants_tui`` said yes: Textual is an
optional extra, and nothing else in the package imports it at module level.

* ``ConfirmScreen``, ``TextEditScreen``, ``ChoiceScreen``, ``MultiChoiceScreen``,
  ``PromptScreen`` — modal dialogs that return their answer through ``dismiss``.
* ``PromptBridge`` — file-like input/output streams that turn a worker
  thread's line prompt (``readline``) into a ``PromptScreen``, so code written
  for a terminal (model download confirmation, speaker choices) can run under a
  full-screen UI unchanged.
* ``sync_table`` — refill a ``DataTable`` by row key, keeping the cursor on
  the same row.
* ``MessageLog`` — a scrolling, severity-colored log fed from ``(time,
  severity, text)`` records.
* ``JobsTable`` — a ``DataTable`` showing a ``tui_jobs.JobQueue``.
* ``VimKeysMixin`` — for an ``App``: vim movement keys do what the arrow,
  page and home/end keys do wherever the person is.
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


# Vim movement: each key does what the key it stands for does in the focused list, table or pane.
VIM_KEYS = {"j": "down", "k": "up", "h": "left", "l": "right", "ctrl+d": "pagedown", "ctrl+u": "pageup",
            "g": "top", "G": "bottom"}
_VIM_ACTION = "vim_key"


def _binding_keys(binding):
    return {key.strip() for key in binding.key.split(",")}


class VimKeysMixin:
    """For an ``App`` or a screen: j/k/h/l, Ctrl+D/Ctrl+U and g/G move like the arrow, page and home/end keys.

    Apps get it, and so does every ``Dialog``, because Textual checks no app
    bindings under a modal screen. A vim key the class (or a base) already
    binds keeps that meaning (the cast's ``g`` generates; a takes dialog's
    ``g`` marks a take good), and a subclass's own key replaces an
    inherited vim key; a screen that binds or handles a key itself still
    wins over the app's, because app bindings are checked last; and text
    boxes still type the letters. Each key is replayed as the key it stands for, so it
    does exactly what that key does where the person is: the cursor of a
    table or list, a dialog's own up/down, or scrolling a pane. ``g``/``G``
    go to the first/last row of a table (Ctrl+Home/End there) and the
    top/bottom of a list or pane.
    """

    def __init_subclass__(cls, **kwargs):
        own = list(cls.__dict__.get("BINDINGS", []))
        inherited = kwargs.get("inherit_bindings", True)
        classes = cls.__mro__[1:] if inherited else ()
        taken = {key for klass in (cls, *classes) for binding in
                 (own if klass is cls else klass.__dict__.get("BINDINGS", []))
                 if isinstance(binding, Binding) for key in _binding_keys(binding)}
        added = [Binding(key, f"{_VIM_ACTION}({key!r})", f"Vim {motion}", show=False)
                 for key, motion in VIM_KEYS.items() if key not in taken]
        if added:
            cls.BINDINGS = [*own, *added]
        super().__init_subclass__(**kwargs)

    def action_vim_key(self, key):
        from textual.app import App
        app = self.app
        motion = VIM_KEYS[key]
        if motion in {"top", "bottom"}:
            # A table's Home/End scroll sideways; its first/last row is Ctrl+Home/End.
            wanted = "ctrl+home" if motion == "top" else "ctrl+end"
            node, binds = app.focused, False
            while node is not None and not isinstance(node, App) and not binds:
                binds = any(wanted in _binding_keys(binding) for binding in getattr(node, "BINDINGS", [])
                            if isinstance(binding, Binding))
                node = node.parent
            motion = wanted if binds else ("home" if motion == "top" else "end")
        app.simulate_key(motion)


def vim_help(bindings):
    """The one key-help row for the vim keys among ``bindings`` (an app's), else None."""
    keys = [key for binding in bindings if isinstance(binding, Binding) and binding.action.startswith(_VIM_ACTION)
            for key in _binding_keys(binding)]
    if not keys:
        return None
    names = {"j": "j/k", "k": "j/k", "h": "h/l", "l": "h/l", "ctrl+d": "Ctrl+D/U", "ctrl+u": "Ctrl+D/U",
             "g": "g/G", "G": "g/G"}
    shown = []
    for key in keys:
        if names[key] not in shown:
            shown.append(names[key])
    # A pair whose second key the app uses for something else is shown with only the free one.
    free = set(keys)
    shown = [name if all(part in free for part in _pair(name)) else next(part for part in _pair(name) if part in free)
             for name in shown]
    return (", ".join(_key_name(name) if "/" not in name else name for name in shown),
            "Vim movement: down/up, left/right, page down/up, first/last (as the arrow, page and Home/End keys)")


def _pair(name):
    return {"j/k": ("j", "k"), "h/l": ("h", "l"), "Ctrl+D/U": ("ctrl+d", "ctrl+u"), "g/G": ("g", "G")}[name]


class Dialog(VimKeysMixin, ModalScreen):
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


class QuitWaitScreen(Dialog):
    """Quitting waits for work in progress: say what it is, live, and offer to quit at once.

    ``describe()`` returns the lines saying what is still running; it is called
    every half second on the screen's thread, so it must be cheap. When it
    returns nothing the work is done and the dialog closes with ``"done"``.
    Quit now (``Q``) closes it with ``"now"`` and Don't quit (Escape, when
    ``can_cancel``) with ``"cancel"``. ``consequence`` says what Quit now loses.
    """

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("Q", "answer('now')", "Quit now"), Binding("escape,n", "answer('cancel')", "Don't quit")]

    def __init__(self, describe, *, title="Quitting…", waiting="Quitting when this finishes:",
                 consequence="Quit now ends at once: work in progress is lost, and everything already saved stays.",
                 can_cancel=True, clock=time.monotonic):
        super().__init__()
        self.describe, self.title_text, self.waiting = describe, title, waiting
        self.consequence, self.can_cancel, self.clock = consequence, can_cancel, clock
        self.started = clock()

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Static(Text(self.waiting))
            yield Static("", id="quit-wait-work")
            yield Static("", id="quit-wait-elapsed", classes="dialog-help")
            with Horizontal(classes="dialog-buttons"):
                if self.can_cancel:
                    yield Button("Don't quit", id="cancel", variant="primary")
                yield Button("Quit now", id="now", variant="error")
            yield Static(Text(self.consequence + " Q quits now" + ("; Escape keeps working." if self.can_cancel
                                                                     else ".")), classes="dialog-help")

    def on_mount(self):
        if self._never_composed():
            return
        self.query_one("#cancel" if self.can_cancel else "#now", Button).focus()
        self._refresh()
        self.set_interval(0.5, self._refresh)

    def _never_composed(self):
        """The work ended and the app is closing while this dialog opens: nothing is left to show."""
        return not self.query("#now")

    def _refresh(self):
        if self._dismissed_once or self._never_composed():
            return
        try:
            lines = list(self.describe())
        except Exception as error:  # noqa: BLE001 - the dialog must stay usable to quit now
            lines = [f"(could not tell what is running: {type(error).__name__})"]
        if not lines:
            self.dismiss("done")
            return
        self.query_one("#quit-wait-work", Static).update(Text("\n".join("• " + line for line in lines)))
        self.query_one("#quit-wait-elapsed", Static).update(
            Text(f"Waiting {int(self.clock() - self.started)} s"))

    def action_answer(self, value):
        if value == "cancel" and not self.can_cancel:
            return
        self.dismiss(value)

    def on_button_pressed(self, event):
        self.action_answer(event.button.id)


class KeysScreen(Dialog):
    """List every key an app binds, including ones the footer has no room for."""

    DEFAULT_CSS = BASE_CSS
    BINDINGS = [Binding("escape,question_mark,q", "close", "Close", priority=True)]

    def __init__(self, bindings, *, title="Keys"):
        super().__init__()
        self.title_text = title
        self.rows = []
        bindings = list(bindings)
        for binding in bindings:
            if binding.action.startswith(_VIM_ACTION):
                continue  # Summed up in one row below.
            keys = ", ".join(_key_name(key) for key in binding.key.split(","))
            detail = binding.description
            if binding.tooltip and binding.tooltip != binding.description:
                detail += " — " + binding.tooltip
            self.rows.append((keys, detail))
        self.rows.append(("Tab", "Move between the panels"))

    def compose(self) -> ComposeResult:
        # The vim keys are the app's (``VimKeysMixin``), whichever screen's keys are listed.
        vim = vim_help(binding for klass in type(self.app).__mro__ for binding in klass.__dict__.get("BINDINGS", []))
        if vim is not None and vim not in self.rows:
            self.rows.append(vim)
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


class MultiChoiceScreen(Dialog):
    """Choose one option, or mark several (Space) and choose them together; Escape cancels.

    ``options`` is ``[(value, label)]``; a value of ``None`` means "none of
    them" and always returns ``[]``. Enter returns the highlighted option
    alone, unless it is marked or two or more are marked: then it returns the
    marked ones, in the order they were marked. Typing filters the list; Down
    (or Tab) moves into it, where Space marks.
    """

    DEFAULT_CSS = BASE_CSS + """
    MultiChoiceScreen OptionList { height: auto; max-height: 20; }
    """
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True),
                Binding("space", "toggle", "Mark", show=False)]

    def __init__(self, title, options, *, selected=(), help_text=""):
        super().__init__()
        self.title_text, self.options, self.help_text = title, list(options), help_text
        self.marked = [value for value in selected if value is not None]
        self._shown = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Input(placeholder="Type to filter", id="choice-filter")
            yield OptionList(id="choices")
            yield Static(Text(self.help_text or "Enter chooses the highlighted one, or the marked ones · "
                                                "Space marks (in the list) · Escape cancels"),
                         classes="dialog-help")

    def _label(self, value, label):
        if value is None:
            return Text(("[x] " if not self.marked else "[ ] ") + label, style="italic")
        return Text(("[x] " if value in self.marked else "[ ] ") + label)

    def _fill(self, text, keep=None):
        words = text.casefold().split()
        self._shown = [index for index, (_, label) in enumerate(self.options)
                       if all(word in label.casefold() for word in words)]
        choices = self.query_one("#choices", OptionList)
        choices.clear_options()
        choices.add_options([Option(self._label(*self.options[index]), id=str(index)) for index in self._shown])
        if self._shown:
            position = self._shown.index(keep) if keep in self._shown else 0
            choices.highlighted = position

    def on_mount(self):
        self._fill("")
        first = next((index for index, (value, _) in enumerate(self.options) if value in self.marked), None)
        if first is not None:
            self.query_one("#choices", OptionList).highlighted = self._shown.index(first)
        self.query_one("#choice-filter", Input).focus()

    def _highlighted(self):
        choices = self.query_one("#choices", OptionList)
        if not choices.option_count or choices.highlighted is None:
            return None
        return int(choices.get_option_at_index(choices.highlighted).id)

    def on_key(self, event):
        if event.key == "down" and self.focused is self.query_one("#choice-filter", Input):
            self.query_one("#choices", OptionList).focus()
            event.stop()

    def on_input_changed(self, event):
        self._fill(event.value)

    def action_toggle(self):
        if self.focused is self.query_one("#choice-filter", Input):
            # Space types in the filter; it marks only in the list.
            self.query_one("#choice-filter", Input).insert_text_at_cursor(" ")
            return
        index = self._highlighted()
        if index is None:
            return
        value = self.options[index][0]
        if value is None:
            self.marked.clear()
        elif value in self.marked:
            self.marked.remove(value)
        else:
            self.marked.append(value)
        self._fill(self.query_one("#choice-filter", Input).value, keep=index)

    def _choose(self, index):
        value = self.options[index][0]
        if value is None:
            self.dismiss([])
        elif len(self.marked) >= 2 or value in self.marked:
            self.dismiss(list(self.marked))
        else:
            self.dismiss([value])

    def on_input_submitted(self, event):
        index = self._highlighted()
        if index is not None:
            self._choose(index)

    def on_option_list_option_selected(self, event):
        self._choose(int(event.option.id))

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
                    # Widen the column when the value grows (1421/5830, not 1421/5).
                    table.update_cell(key, column, value, update_width=True)
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
