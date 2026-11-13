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
from textual.widgets import DataTable, Label, OptionList, Static
from textual.widgets.option_list import Option

from .tui_widgets import BASE_CSS, ChoiceScreen, Dialog, highlighted_key, sync_table


ACTIONS = {
    "retry": "Retry this chapter now, ahead of everything waiting (reuses every verified clip)",
    "retry_book": "Put this book back in the queue now, at the front (completed work is reused)",
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


class FailureScreen(Dialog):
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


class TakesScreen(Dialog):
    """Every take of one passage, to compare by ear, choose the one used, and judge verification.

    Dismisses with ``{"attempt", "learn"}`` when a person chooses a take to
    use; marks (good/bad) are saved at once and the dialog stays open.
    """

    DEFAULT_CSS = BASE_CSS + """
    TakesScreen #takes-expected { height: auto; max-height: 6; }
    TakesScreen #takes-table { height: auto; max-height: 14; margin-top: 1; }
    TakesScreen #takes-heard { height: auto; max-height: 6; margin-top: 1; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("enter,p", "play", "Play"),
        Binding("space", "stop", "Stop"),
        Binding("u,a", "use", "Use this take"),
        Binding("g", "mark('good')", "Good"),
        Binding("b", "mark('bad')", "Bad"),
        Binding("c", "mark(None)", "Clear mark"),
    ]

    def __init__(self, title, review, player, *, highlight=None):
        super().__init__()
        self.title_text, self.review, self.player = title, review, player
        self.highlight = highlight
        self.rows = []

    # The name the failed-passage dialog first used for its handle.
    @property
    def repair(self):
        return self.review

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            yield Static("", id="takes-expected")
            yield DataTable(id="takes-table", cursor_type="row")
            yield Static("", id="takes-heard")
            yield Static("", id="takes-error", classes="dialog-error")
            yield Static(Text("Enter/p plays · Space stops · u uses this take · g marks it good (wrongly rejected) · "
                              "b marks it bad · c clears the mark · Escape closes"), classes="dialog-help")

    def on_mount(self):
        self.query_one("#takes-table", DataTable).add_columns("Take", "Used", "Verification", "Length",
                                                              "Why it failed", "Your verdict", "Usable")
        self.reload()
        self.query_one("#takes-table", DataTable).focus()

    def reload(self):
        table = self.query_one("#takes-table", DataTable)
        try:
            self.rows = self.review.takes()
        except (ValueError, OSError) as error:
            self.query_one("#takes-error", Static).update(Text(f"Could not read the takes: {error}"))
            return
        expected = next((row["expected_text"] for row in self.rows if row["expected_text"]), "")
        self.query_one("#takes-expected", Static).update(Text("Passage: " + expected) if expected else "")
        verdicts = {"good": Text("good", style="green"), "bad": Text("bad", style="red"), None: ""}
        sync_table(table, [(str(row["attempt"]), (
            str(row["attempt"]), Text("●", style="bold green") if row["accepted"] else "",
            "passed" if row["passed"] else "rejected", _seconds(row["duration_seconds"]),
            ", ".join(row["reasons"]) or "—", verdicts.get(row.get("verdict"), ""),
            "yes" if row["eligible"] else f"no: {row['ineligible_reason']}")) for row in self.rows])
        if self.highlight is not None:
            keys = [str(row["attempt"]) for row in self.rows]
            if str(self.highlight) in keys:
                table.move_cursor(row=keys.index(str(self.highlight)))
            self.highlight = None
        self._show_heard()

    def current(self):
        key = highlighted_key(self.query_one("#takes-table", DataTable))
        return next((row for row in self.rows if str(row["attempt"]) == key), None)

    def _show_heard(self):
        row = self.current()
        self.query_one("#takes-heard", Static).update(
            Text("Heard: " + row["transcript"]) if row and row["transcript"] else "")

    def _error(self, text):
        self.query_one("#takes-error", Static).update(Text(text))

    def on_data_table_row_highlighted(self, event):
        self._show_heard()

    def on_data_table_row_selected(self, event):
        self.action_play()

    def action_play(self):
        row = self.current()
        if row is None:
            return
        self._error("")
        self.run_worker(lambda: self._play(row["audio_path"]), thread=True, exclusive=True, group="takes-play")

    def _play(self, path):
        try:
            self.player.start(path)
        except (ValueError, OSError) as error:
            self.app.call_from_thread(self._error, f"Could not play: {error}")

    def action_stop(self):
        try:
            self.player.stop()
        except (ValueError, OSError):
            pass

    def action_use(self):
        row = self.current()
        if row is None:
            return
        if not row["eligible"]:
            self._error(f"Take {row['attempt']} cannot be used: {row['ineligible_reason']}")
            return
        if row["accepted"]:
            self._error(f"Take {row['attempt']} is already the one used.")
            return
        options = []
        if row["learnable"]:
            options.append(("learn", "Use it, and teach the recognizer how it heard these words"))
        options.append(("use", "Use it" + ("" if row["learnable"] or row["passed"] else
                                          " (its differences are not ones the recognizer can learn)")))

        def chosen(value):
            if value is not None:
                self.action_stop()
                self.dismiss({"attempt": row["attempt"], "learn": value == "learn"})
        self.app.push_screen(ChoiceScreen(
            f"Use take {row['attempt']} for this passage? Its chapter is rebuilt from its verified clips with it.",
            options, filterable=False), chosen)

    # The failed-passage dialog's original key name.
    action_accept = action_use

    def action_mark(self, verdict):
        row = self.current()
        if row is None:
            return
        if verdict == "good" and row["learnable"]:
            def chosen(value):
                if value is not None:
                    self._mark(row, "good", value == "learn")
            self.app.push_screen(ChoiceScreen(
                f"Take {row['attempt']} sounds right. Teach the recognizer how it heard these words too?",
                [("learn", "Mark it good and teach the recognizer"), ("good", "Only mark it good")],
                filterable=False), chosen)
            return
        self._mark(row, verdict, False)

    def _mark(self, row, verdict, learn):
        self.run_worker(lambda: self._save_mark(row, verdict, learn), thread=True, group="takes-mark")

    def _save_mark(self, row, verdict, learn):
        try:
            outcome = self.review.mark(row["attempt"], verdict, learn=learn)
        except (ValueError, OSError) as error:
            self.app.call_from_thread(self._error, f"Could not save the mark: {error}")
            return
        learned = "; ".join(f"{expected} → {heard}" for expected, heard in outcome["learned"])
        self.app.call_from_thread(self._error, f"Take {row['attempt']}: " + (
            f"marked {verdict}" if verdict else "mark cleared") + (f"; learned {learned}" if learned else ""))
        self.app.call_from_thread(self.reload)

    def action_close(self):
        self.action_stop()
        self.dismiss(None)
