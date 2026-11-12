"""Render repair dialogs for the run shell (Textual).

* ``FailureScreen`` — one failure's saved detail, a plain-language hint, and
  the repairs that apply to it; returns the chosen action name.
* ``TakesScreen`` — every retained take of a failed passage: play them, compare
  what the recognizer heard with the passage, and accept one a person judges
  good (optionally teaching the recognizer); returns ``{"attempt", "learn"}``.

Import only after ``tui_support.wants_tui`` said yes.
"""

from __future__ import annotations

import json

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Label, OptionList, Static
from textual.widgets.option_list import Option

from .tui_widgets import BASE_CSS, ChoiceScreen, highlighted_key, sync_table


ACTIONS = {
    "retry": "Retry this chapter now (reuses every verified clip)",
    "takes": "Listen to the takes and accept one that sounds right",
    "resume_book": "Resume this book after the run (completed work is reused)",
    "close": "Close",
}


def failure_text(item, hint):
    scope = item["scope"]
    lines = []
    if item.get("error"):
        lines.append(item["error"])
    lines += ["", "Suggestion: " + hint, ""]
    if item.get("state") and item["state"] != "failed":
        lines.append(f"State: {item['state']}")
    lines.append("Saved record:")
    lines.append(json.dumps(item["record"], indent=2, ensure_ascii=False, default=str))
    title = " ".join(part for part in (f"[{scope['book']}]" if scope.get("book") else "",
                                       f"Chapter {scope['chapter']}" if scope.get("chapter") else "",
                                       "failed" if item["kind"] == "chapter" else "book failed") if part)
    return title, "\n".join(lines)


class FailureScreen(ModalScreen):
    """A failure's detail and the repairs that apply to it."""

    DEFAULT_CSS = BASE_CSS + """
    FailureScreen #failure-detail { height: auto; max-height: 20; }
    FailureScreen OptionList { height: auto; margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close", priority=True)]

    def __init__(self, item, hint, actions):
        super().__init__()
        self.item, self.hint, self.actions = item, hint, [*actions, "close"]

    def compose(self) -> ComposeResult:
        title, text = failure_text(self.item, self.hint)
        with Vertical(classes="dialog"):
            yield Label(Text(title), classes="dialog-title")
            with VerticalScroll(id="failure-detail"):
                yield Static(Text(text), id="detail-text")
            yield OptionList(*[Option(Text(ACTIONS[name]), id=name) for name in self.actions], id="actions")
            yield Static("Enter chooses · Escape closes", classes="dialog-help")

    def on_mount(self):
        self.query_one("#actions", OptionList).focus()

    def on_option_list_option_selected(self, event):
        self.dismiss(None if event.option.id == "close" else event.option.id)

    def action_close(self):
        self.dismiss(None)


def _seconds(value):
    return "" if value is None else f"{value:.1f}s"


class TakesScreen(ModalScreen):
    """Play every take of a failed passage and accept one a person judges good."""

    DEFAULT_CSS = BASE_CSS + """
    TakesScreen #takes-expected { height: auto; max-height: 6; }
    TakesScreen #takes-table { height: auto; max-height: 14; margin-top: 1; }
    TakesScreen #takes-heard { height: auto; max-height: 6; margin-top: 1; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("enter,p", "play", "Play"),
        Binding("space", "stop", "Stop"),
        Binding("a", "accept", "Accept"),
    ]

    def __init__(self, title, repair, player):
        super().__init__()
        self.title_text, self.repair, self.player = title, repair, player
        self.rows = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Static("", id="takes-expected")
            yield DataTable(id="takes-table", cursor_type="row")
            yield Static("", id="takes-heard")
            yield Static("", id="takes-error", classes="dialog-error")
            yield Static("Enter/p plays · Space stops · a accepts the highlighted take · Escape closes",
                         classes="dialog-help")

    def on_mount(self):
        table = self.query_one("#takes-table", DataTable)
        table.add_columns("Take", "Result", "Length", "Why it failed", "Usable")
        try:
            self.rows = self.repair.takes()
        except (ValueError, OSError) as error:
            self.query_one("#takes-error", Static).update(Text(f"Could not read the takes: {error}"))
            return
        expected = next((row["expected_text"] for row in self.rows if row["expected_text"]), "")
        self.query_one("#takes-expected", Static).update(Text("Passage: " + expected) if expected else "")
        sync_table(table, [(str(row["attempt"]), (
            f"{row['attempt']}{' ●' if row['accepted'] else ''}",
            "passed" if row["passed"] else "failed", _seconds(row["duration_seconds"]),
            ", ".join(row["reasons"]) or "—",
            "yes" if row["eligible"] else f"no: {row['ineligible_reason']}")) for row in self.rows])
        table.focus()
        self._show_heard()

    def current(self):
        key = highlighted_key(self.query_one("#takes-table", DataTable))
        return next((row for row in self.rows if str(row["attempt"]) == key), None)

    def _show_heard(self):
        row = self.current()
        self.query_one("#takes-heard", Static).update(
            Text("Heard: " + row["transcript"]) if row and row["transcript"] else "")

    def on_data_table_row_highlighted(self, event):
        self._show_heard()

    def on_data_table_row_selected(self, event):
        self.action_play()

    def action_play(self):
        row = self.current()
        if row is None:
            return
        self.query_one("#takes-error", Static).update("")
        self.run_worker(lambda: self._play(row["audio_path"]), thread=True, exclusive=True, group="takes-play")

    def _play(self, path):
        try:
            self.player.start(path)
        except (ValueError, OSError) as error:
            self.app.call_from_thread(self.query_one("#takes-error", Static).update, Text(f"Could not play: {error}"))

    def action_stop(self):
        try:
            self.player.stop()
        except (ValueError, OSError):
            pass

    def action_accept(self):
        row = self.current()
        if row is None:
            return
        if not row["eligible"]:
            self.query_one("#takes-error", Static).update(
                Text(f"Take {row['attempt']} cannot be accepted: {row['ineligible_reason']}"))
            return
        options = []
        if row["learnable"]:
            options.append(("learn", "Accept it, and teach the recognizer how it heard these words"))
        options.append(("accept", "Accept it" + ("" if row["learnable"] else
                                                 " (its differences are not ones the recognizer can learn)")))

        def chosen(value):
            if value is not None:
                self.action_stop()
                self.dismiss({"attempt": row["attempt"], "learn": value == "learn"})
        self.app.push_screen(ChoiceScreen(
            f"Take {row['attempt']} sounds right: accept it for this passage? The chapter then renders again "
            "from its verified clips with this take.", options, filterable=False), chosen)

    def action_close(self):
        self.action_stop()
        self.dismiss(None)
