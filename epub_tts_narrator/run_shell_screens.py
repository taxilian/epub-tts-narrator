"""Render repair dialogs for the run shell (Textual).

* ``FailureScreen`` — one failure's saved detail (for a book, also its failed
  chapters), a plain-language hint, and the repairs that apply to it; returns
  the chosen action name.
* ``TakesScreen`` — every retained take of a failed passage: play them, compare
  what the recognizer heard with the passage, and accept one a person judges
  good (optionally teaching the recognizer); returns ``{"attempt", "learn"}``.
  Marking the used take bad retires it at once and calls ``on_retired`` so its
  chapter renders the passage again.

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
    "retry_book": ("Resume this book now in its command-line place: it goes ahead of books listed after it as soon "
                   "as their current chapters are done (completed work is reused)"),
    "takes": "Listen to the takes and accept one that sounds right",
    "resume_book": "Resume this book (completed work is reused)",
    "resume": "Run it again (completed work is reused)",
    "chapters": "Show this book's chapters",
    "close": "Close",
}


def failure_text(item, hint, details=(), summary=""):
    """``(title, text)``: the plain-words summary first, then the whole error, the details, the hint and record."""
    scope = item["scope"]
    lines = [summary, ""] if summary else []
    if item.get("error"):
        lines.append(("Error: " if summary else "") + item["error"])
    else:
        lines.append("No error message was reported; the Log (4) has this run's messages.")
    if details:
        lines += ["", *details]
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
    FailureScreen #failure-detail { height: auto; max-height: 60vh; }
    FailureScreen OptionList { height: auto; max-height: 12; margin-top: 1; }
    """
    BINDINGS = [Binding("escape", "close", "Close", priority=True),
                Binding("pageup", "scroll_detail(-1)", "Scroll up", show=False),
                Binding("pagedown", "scroll_detail(1)", "Scroll down", show=False)]

    def __init__(self, item, hint, actions, *, details=(), summary=""):
        """``actions`` are names in ``ACTIONS`` or ``(name, label)`` pairs; Close is always last."""
        super().__init__()
        self.labels = {}
        names = []
        for action in actions:
            name, label = action if isinstance(action, tuple) else (action, ACTIONS[action])
            names.append(name)
            self.labels[name] = label
        self.item, self.hint, self.actions = item, hint, [*names, "close"]
        self.labels["close"] = ACTIONS["close"]
        self.details, self.summary = list(details), summary

    def compose(self) -> ComposeResult:
        title, text = failure_text(self.item, self.hint, self.details, self.summary)
        with Vertical(classes="dialog"):
            yield Label(Text(title), classes="dialog-title")
            with VerticalScroll(id="failure-detail"):
                yield Static(Text(text), id="detail-text")
            yield OptionList(*[Option(Text(self.labels[name]), id=name) for name in self.actions], id="actions")
            yield Static("Enter chooses · PageUp/PageDown scroll the detail · Escape closes", classes="dialog-help")

    def action_scroll_detail(self, direction):
        detail = self.query_one("#failure-detail", VerticalScroll)
        (detail.scroll_page_down if direction > 0 else detail.scroll_page_up)()

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
    use; marks (good/bad) are saved at once and the dialog stays open. A bad
    mark on the used take retires it, and ``on_retired(attempt)`` (called on a
    worker thread) arranges for the passage to render again and returns the
    message to show.
    """

    DEFAULT_CSS = BASE_CSS + """
    TakesScreen .dialog { width: 95%; max-width: 220; height: 95%; max-height: 95%; }
    TakesScreen #takes-passage { height: 1fr; min-height: 3; }
    TakesScreen #takes-table { height: auto; max-height: 12; margin-top: 1; }
    TakesScreen #takes-detail { height: 1fr; min-height: 3; margin-top: 1; }
    TakesScreen #takes-differs { margin-top: 1; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("enter,p", "play", "Play"),
        Binding("space", "stop", "Stop"),
        Binding("u,a", "use", "Use this take"),
        Binding("g", "mark('good')", "Good"),
        Binding("b", "mark('bad')", "Bad"),
        Binding("t", "mark('truncated')", "Truncated"),
        Binding("c", "mark(None)", "Clear mark"),
        # Before the table's own paging: the table is short, the passage may not be.
        Binding("pageup", "scroll_pane('passage', -1)", "Scroll passage up", show=False, priority=True),
        Binding("pagedown", "scroll_pane('passage', 1)", "Scroll passage down", show=False, priority=True),
        Binding("shift+pageup", "scroll_pane('detail', -1)", "Scroll heard up", show=False, priority=True),
        Binding("shift+pagedown", "scroll_pane('detail', 1)", "Scroll heard down", show=False, priority=True),
        Binding("greater_than_sign", "speed", "Speed", tooltip="Play at 2x or 1x (pitch unchanged), here and everywhere"),
    ]

    def __init__(self, title, review, player, *, highlight=None, on_retired=None):
        super().__init__()
        self.title_text, self.review, self.player = title, review, player
        self.highlight = highlight
        self.on_retired = on_retired
        self.rows = []

    # The name the failed-passage dialog first used for its handle.
    @property
    def repair(self):
        return self.review

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            # Never cut off: the whole passage and transcript scroll when they do not fit.
            with VerticalScroll(id="takes-passage"):
                yield Static("", id="takes-expected")
            yield DataTable(id="takes-table", cursor_type="row")
            with VerticalScroll(id="takes-detail"):
                yield Static("", id="takes-heard")
                yield Static("", id="takes-differs")
            yield Static("", id="takes-error", classes="dialog-error")
            yield Static(self._help(), id="takes-help", classes="dialog-help")

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
        verdicts = {"good": Text("good", style="green"), "bad": Text("bad", style="red"),
                    "truncated": Text("truncated", style="yellow"), "redo": Text("set aside", style="magenta"),
                    None: ""}
        sync_table(table, [(str(row["attempt"]), (
            str(row["attempt"]), Text("●", style="bold green") if row["accepted"] else "",
            ("passed" if row["passed"] else "rejected")
            + (f" · finished take {row['finished_from']}" if row.get("finished_from") else ""),
            _seconds(row["duration_seconds"]),
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
        from .renderer.take_review import word_differences
        row = self.current()
        self.query_one("#takes-heard", Static).update(
            Text("Heard: " + row["transcript"]) if row and row["transcript"] else "")
        differs = Text()
        if row and row.get("tone") and not row["passed"]:
            # Rejected before recognition: there is no transcript to compare.
            differs.append(f"Rejected for {row['tone']} over the voice.\n", style="yellow")
        if row and row["transcript"] and not row["passed"]:
            if row.get("stopped") == "silence":
                differs.append("The take went silent and stopped before the end.\n", style="yellow")
            if row.get("misread"):
                differs.append(f"Differs: {row['misread']} (not the pronunciation you chose)\n", style="yellow")
            for line in word_differences(row["expected_text"], row["transcript"]):
                differs.append("Differs: " + line + "\n", style="yellow")
        self.query_one("#takes-differs", Static).update(differs)

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

    def action_scroll_pane(self, pane, direction):
        view = self.query_one(f"#takes-{pane}", VerticalScroll)
        (view.scroll_page_down if direction > 0 else view.scroll_page_up)(animate=False)

    def _help(self):
        from .renderer import playback_speed
        return Text("PgUp/PgDn scroll the passage, Shift+PgUp/PgDn what was heard · "
                    "Enter/p plays · Space stops · u uses this take · g marks it good (wrongly rejected) · "
                    "b marks it bad (the used take is then rendered again) · t marks it truncated "
                    "(right but cut short: the rest is generated after it) · c clears the mark · "
                    f"{playback_speed.help_text(self.player)} · Escape closes")

    def action_speed(self):
        from .renderer import playback_speed
        try:
            playback_speed.toggle(self.player)  # A playing take starts again at the new speed.
        except (ValueError, OSError) as error:
            self._error(f"Could not play: {error}")
        self.query_one("#takes-help", Static).update(self._help())

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
        message = f"Take {row['attempt']}: " + (
            f"marked {verdict}" if verdict else "mark cleared") + (f"; learned {learned}" if learned else "")
        if outcome.get("retired"):
            follow = self.on_retired(row["attempt"]) if callable(self.on_retired) else None
            message += "; it is no longer used. " + (follow or "The next render makes a new take of this passage.")
        self.app.call_from_thread(self._error, message)
        self.app.call_from_thread(self.reload)

    def action_close(self):
        self.action_stop()
        self.dismiss(None)


def _check_cell(word):
    """What the name check made of the problem word in one take."""
    if not word:
        return Text("—", style="dim")
    if word.get("uncertain"):
        return Text("uncertain", style="yellow")
    return Text("said right", style="green") if word.get("ok") else Text("MISREAD", style="bold red")


class HistoryTakesScreen(Dialog):
    """Takes around a passage blocked on a name (``renderer.history_review``): listen, judge, and act.

    ``mode`` is ``heard`` (what the passage heard as history), ``word`` (the
    voice's takes of the problem word) or ``stream`` (the voice's chain of
    takes in the chapter). Rows load on a worker (``loader``). Good and bad
    marks are saved at once; ``on_retired(row)`` (on a worker thread) arranges
    for a retired take's chapter to render again and returns the message to
    show. Dismisses with ``{"action": "avoid", "rows"}`` (the passage never
    hears them again), ``{"action": "stream", "start"}`` (render the stream
    again from that take) or ``None``.
    """

    DEFAULT_CSS = BASE_CSS + """
    HistoryTakesScreen .dialog { width: 95%; max-width: 220; height: 95%; max-height: 95%; }
    HistoryTakesScreen #history-intro { height: auto; max-height: 6; }
    HistoryTakesScreen #history-table { height: 1fr; min-height: 5; margin-top: 1; }
    HistoryTakesScreen #history-detail { height: auto; max-height: 8; margin-top: 1; }
    """
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("enter,p", "play(False)", "Play"),
        Binding("f", "play(True)", "Play the whole take"),
        Binding("space", "stop", "Stop"),
        Binding("g", "mark('good')", "Good"),
        Binding("b", "mark('bad')", "Bad"),
        Binding("c", "mark(None)", "Clear mark"),
        Binding("a", "avoid(False)", "Not for this passage", show=False),
        Binding("A", "avoid(True)", "None of these for this passage", show=False),
        Binding("r", "stream_from", "Render again from here", show=False),
        Binding("h", "immediate", "Only this passage's history", show=False),
        Binding("greater_than_sign", "speed", "Speed", tooltip="Play at 2x or 1x (pitch unchanged), here and everywhere"),
    ]

    def __init__(self, title, intro, history, loader, player, *, mode, on_retired=None):
        super().__init__()
        self.title_text, self.intro, self.history, self.loader = title, intro, history, loader
        self.player, self.mode, self.on_retired = player, mode, on_retired
        self.rows = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.title_text), classes="dialog-title")
            with VerticalScroll(id="history-intro"):
                yield Static(Text(self.intro))
            yield DataTable(id="history-table", cursor_type="row")
            with VerticalScroll(id="history-detail"):
                yield Static("", id="history-text")
            yield Static(Text("Loading the takes and hearing the name in each…"), id="history-error",
                         classes="dialog-error")
            yield Static(self._help(), id="history-help", classes="dialog-help")

    def on_mount(self):
        self.query_one("#history-table", DataTable).add_columns(
            "Where", "Role", "Name heard as", "Name check", "In the book", "Your verdict", "Text")
        self.query_one("#history-table", DataTable).focus()
        self.run_worker(self._load, thread=True, exclusive=True, group="history-load")

    def _load(self):
        try:
            rows = self.loader()
        except (ValueError, OSError) as error:
            self.app.call_from_thread(self._error, f"Could not read the takes: {error}")
            return
        self.app.call_from_thread(self._show, rows)

    def _verdict(self, row):
        from .renderer.take_feedback import verdicts
        try:
            return verdicts(row["work"], row["request_hash"]).get(row["attempt"])
        except (ValueError, OSError):
            return None

    def _show(self, rows):
        self.rows = rows
        marks = {"good": Text("good", style="green"), "bad": Text("bad", style="red"),
                 "truncated": Text("truncated", style="yellow"), "redo": Text("set aside", style="magenta")}
        table = self.query_one("#history-table", DataTable)
        sync_table(table, [(str(index), (
            row["where"], row["relation"], (row["word"] or {}).get("heard") or "—", _check_cell(row["word"]),
            "yes" if row["in_use"] else "no", marks.get(self._verdict(row), ""),
            (row["text"][:70] + "…") if len(row["text"]) > 70 else row["text"])) for index, row in enumerate(rows)])
        self._error("" if rows else "No takes to show.")
        self._detail()

    def current(self):
        key = highlighted_key(self.query_one("#history-table", DataTable))
        return self.rows[int(key)] if key is not None and key.isdigit() and int(key) < len(self.rows) else None

    def _detail(self):
        row = self.current()
        self.query_one("#history-text", Static).update(Text(f"{row['where']}: {row['text']}") if row else "")

    def _error(self, text):
        self.query_one("#history-error", Static).update(Text(text))

    def _help(self):
        from .renderer import playback_speed
        extra = {"heard": " · a: this passage never hears it again · A: none of these for this passage",
                 "stream": " · r: render the stream again from this take · h: change only this passage's history"
                 }.get(self.mode, "")
        return Text("Enter/p plays (around the name) · f plays the whole take · Space stops · g good · b bad "
                    "(a take in the book renders again) · c clears" + extra +
                    f" · {playback_speed.help_text(self.player)} · Escape closes")

    def on_data_table_row_highlighted(self, event):
        self._detail()

    def on_data_table_row_selected(self, event):
        self.action_play(False)

    def action_play(self, whole):
        row = self.current()
        if row is None or not row.get("audio_path"):
            return
        self._error("")
        self.run_worker(lambda: self._play(row, whole), thread=True, exclusive=True, group="history-play")

    def _play(self, row, whole):
        try:
            path = row["audio_path"] if whole else self.history.excerpt(row)
            self.player.start(path)
        except Exception as error:  # an excerpt or player failure is shown, never fatal
            self.app.call_from_thread(self._error, f"Could not play: {error}")

    def action_stop(self):
        try:
            self.player.stop()
        except (ValueError, OSError):
            pass

    def action_speed(self):
        from .renderer import playback_speed
        try:
            playback_speed.toggle(self.player)
        except (ValueError, OSError) as error:
            self._error(f"Could not play: {error}")
        self.query_one("#history-help", Static).update(self._help())

    def action_mark(self, verdict):
        row = self.current()
        if row is None or type(row.get("attempt")) is not int:
            return
        self.run_worker(lambda: self._save_mark(row, verdict), thread=True, group="history-mark")

    def _save_mark(self, row, verdict):
        try:
            outcome = self.history.review_of(row).mark(row["attempt"], verdict)
        except (ValueError, OSError) as error:
            self.app.call_from_thread(self._error, f"Could not save the mark: {error}")
            return
        message = f"{row['where']}: " + (f"marked {verdict}" if verdict else "mark cleared")
        if outcome.get("retired"):
            follow = self.on_retired(row) if callable(self.on_retired) else None
            message += "; it is no longer used. " + (follow or "The next render makes a new take of it.")
            row["in_use"] = False
        self.app.call_from_thread(self._error, message)
        self.app.call_from_thread(self._show, self.rows)

    def action_avoid(self, every):
        if self.mode != "heard":
            return
        rows = list(self.rows) if every else [row for row in [self.current()] if row is not None]
        if rows:
            self.action_stop()
            self.dismiss({"action": "avoid", "rows": rows})

    def action_stream_from(self):
        row = self.current()
        if self.mode == "stream" and row is not None:
            self.action_stop()
            self.dismiss({"action": "stream", "start": row})

    def action_immediate(self):
        if self.mode == "stream":
            self.action_stop()
            self.dismiss({"action": "immediate"})

    def action_close(self):
        self.action_stop()
        self.dismiss(None)
