"""The full-screen (Textual) review of a published book: ``review-ui`` in a terminal.

Chapters on the left, the highlighted chapter's passages on the right (each
speaker in its own color) and the highlighted passage's metadata below them;
a log of what happened at the bottom. Every decision is ``review_tui.ReviewApp``'s
(the controller the curses view also drives): reassigning a passage to existing
cast members (several make a simultaneous line), its delivery direction, its
review flag, ignoring chapters, saving through ``revise_directories``. Takes
open in the run shell's ``TakesScreen`` over ``renderer/take_review``, so there
is one take player. No voices are created and no model is called.

Import this module only after ``tui_support.wants_tui`` said yes.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from .review_tui import _KIND_BADGES, _flat
from .review_tui_model import palette
from .tui_widgets import (
    BASE_CSS, CapturePrintMixin, ChoiceScreen, VimKeysMixin, ConfirmScreen, KeysScreen, MessageLog, MultiChoiceScreen,
    TextEditScreen, highlighted_key, sync_table,
)


PASSAGE_CHARACTERS = 300
CHAPTER_TITLE_CHARACTERS = 28
_FIELD_KEYS = {"speaker": "e", "direction": "d", "review": "r", "takes": "t"}
_TROUBLE = ("Save failed", "Ignore save failed", "Could not", "Playback failed", "Checkpoint archive failed")


def _severity(notice, ok=True):
    if notice.startswith(_TROUBLE):
        return "error"
    return "info" if ok else "warning"


class ReviewTuiApp(VimKeysMixin, CapturePrintMixin, App):
    """Browse and edit a published book. ``controller`` owns all state; this class only displays it."""

    TITLE = "Review"
    CSS = BASE_CSS + """
    #main { height: 1fr; }
    #left { width: 36%; min-width: 36; }
    #right { width: 1fr; }
    #chapters, #passages { height: 1fr; }
    #detail-scroll { height: 14; border: round $primary-background; padding: 0 1; }
    #log { height: 7; border: round $primary-background; }
    """
    BINDINGS = [
        Binding("q", "quit_review", "Quit", tooltip="Quit; asks first when edits are unsaved"),
        # Textual's own Ctrl-Q would close at once, discarding unsaved edits.
        Binding("ctrl+q", "quit_review", "Quit", show=False, priority=True),
        Binding("question_mark", "keys", "Keys", tooltip="Show every key"),
        Binding("S,ctrl+s", "save", "Save", tooltip="Publish the edits (backups kept; series books rebound)"),
        Binding("e,f2", "speaker", "Speaker", tooltip="Reassign the passage to existing cast members"),
        Binding("d,f3", "direction", "Direction", tooltip="Edit the delivery direction (at most 240 characters)"),
        Binding("r,f4", "review", "Review flag", tooltip="Turn the passage's review flag on or off"),
        Binding("p", "play", "Play", tooltip="Play the passage's accepted take"),
        Binding("space", "stop", "Stop", show=False, tooltip="Stop playback"),
        Binding("greater_than_sign", "speed", "Speed",
                tooltip="Play at 2x or 1x (pitch unchanged); also in Takes and every other listening screen"),
        Binding("t", "takes", "Takes", tooltip="Listen to every take of the passage and change which one is used"),
        Binding("f,F", "failed", "Failed", tooltip="Go to the chapter's first failed passage"),
        Binding("i,I", "ignore", "Ignore", tooltip="Ignore the chapter in the next render and assemble, or restore it"),
        Binding("h,H", "show_ignored", "Ignored", show=False, tooltip="Show or hide ignored chapters"),
        Binding("escape", "back", "Chapters", show=False, tooltip="Back to the chapter list"),
    ]

    def __init__(self, controller, *, title=None):
        super().__init__()
        self.controller = controller
        self.book_title = title or controller.document["source"]["book"]["title"]
        self.role_colors = palette(len(controller.roles))
        self.messages = []
        self.playing = None
        self._started = time.monotonic()
        self._said = ""
        self._chapter_keys = []
        self._passage_keys = []

    # Layout --------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield DataTable(id="chapters", cursor_type="row", zebra_stripes=True)
            with Vertical(id="right"):
                yield DataTable(id="passages", cursor_type="row")
                with VerticalScroll(id="detail-scroll"):
                    yield Static("", id="detail")
        yield MessageLog(id="log")
        yield Footer()

    def on_mount(self):
        self.capture_prints()
        self.title = f"Review — {_flat(self.book_title)}"
        chapters = self.query_one("#chapters", DataTable)
        chapters.add_columns("#", "Chapter", "Passages", "Needs")
        chapters.border_title = "Chapters"
        self.query_one("#passages", DataTable).add_columns("", "Speaker", "Passage")
        self.query_one("#passages", DataTable).border_title = "Passages"
        self.query_one("#detail-scroll").border_title = "Passage"
        self.query_one("#log", MessageLog).border_title = "Log"
        self._say()  # A notice from loading (an unreadable ignore file).
        self.render_chapters()
        chapters.focus()
        self.set_interval(0.3, self._poll_playback)

    def on_captured_line(self, text, stderr):
        from .run_events import classify
        self._log(text, classify(text))

    # Drawing -------------------------------------------------------------------

    def _log(self, text, severity="info"):
        self.messages.append((time.monotonic(), severity, _flat(text)))
        self.query_one("#log", MessageLog).sync(self.messages, started=self._started)

    def _say(self, ok=True):
        """Log the controller's latest notice once."""
        notice = self.controller.notice
        if notice and notice != self._said:
            self._log(notice, _severity(notice, ok))
        self._said = notice
        for line in self.controller.progress:
            self._log(line)
        self.controller.progress.clear()
        self._subtitle()

    def _subtitle(self):
        pending = len(self.controller.pending)
        parts = [f"{pending} unsaved edit(s) · S saves" if pending else "no unsaved edits"]
        if self.controller.show_ignored:
            parts.append("showing ignored chapters")
        if self.playing is not None:
            parts.append("playing (space stops)")
        from .renderer import playback_speed
        parts.append(playback_speed.help_text(self.controller.player))
        self.sub_title = " · ".join(parts)

    def render_chapters(self):
        controller = self.controller
        rows = controller.chapter_rows()
        table = self.query_one("#chapters", DataTable)
        cells = []
        for row in rows:
            needs = []
            if row.get("failure"):
                needs.append(("FAILED", "bold red"))  # Its type and message: the passages title and detail.
            if row.get("ignored"):
                needs.append(("IGNORED", "yellow"))
            if row["unassigned_count"]:
                needs.append((f"{row['unassigned_count']} unassigned", "red"))
            if row["review_count"]:
                needs.append((f"{row['review_count']} to review", "yellow"))
            state = Text(" · ").join(Text(text, style=style) for text, style in needs)
            title = _flat(row["title"])
            if len(title) > CHAPTER_TITLE_CHARACTERS:
                title = title[:CHAPTER_TITLE_CHARACTERS - 1] + "…"
            cells.append((str(row["id"]), (str(row["id"] + 1), title, str(row["passage_count"]), state)))
        self._chapter_keys = sync_table(table, cells)
        key = str(controller.chapter_index)
        if key in self._chapter_keys and highlighted_key(table) != key:
            table.move_cursor(row=self._chapter_keys.index(key))
        if not rows:
            self.query_one("#detail", Static).update(Text(
                "All chapters are ignored. Press H to show them and restore any."
                if controller.ignored_chapters and not controller.show_ignored else "No chapters with passages."))
        self.render_passages()

    def _speakers(self, row):
        if not row["speaker_labels"]:
            return Text("Unassigned", style="bold red")
        text = Text()
        for index, (label, slot) in enumerate(zip(row["speaker_labels"], row["role_indices"])):
            if index:
                text.append(", ")
            text.append(_flat(label), style=f"bold color({self.role_colors[slot % len(self.role_colors)]})"
                        if self.role_colors else "bold")
        return text

    def render_passages(self):
        controller = self.controller
        table = self.query_one("#passages", DataTable)
        if not self._chapter_keys:
            sync_table(table, [])
            self._passage_keys = []
            return
        rows = controller.passage_rows()
        cells = []
        for row in rows:
            badge = Text(_KIND_BADGES.get(row["kind"], "?"))
            if row["review_required"]:
                badge.append("!", style="bold yellow")
            if row.get("failed"):
                badge.append(" FAILED", style="bold red")
            passage = _flat(row["text"])
            if len(passage) > PASSAGE_CHARACTERS:
                passage = passage[:PASSAGE_CHARACTERS - 1] + "…"
            cells.append((row["segment_id"], (badge, self._speakers(row), passage)))
        self._passage_keys = sync_table(table, cells)
        chapter = next((row for row in controller.chapter_rows() if row["id"] == controller.chapter_index), None)
        if chapter is not None:
            title = f"{_flat(chapter['title'])} — {len(rows)} passages"
            if chapter.get("failure"):
                title += " — FAILED: " + _flat(chapter["failure"]["error"])
            table.border_title = title
        segment = highlighted_key(table)
        controller.focus(segment_id=segment)
        self.render_detail()

    def render_detail(self):
        controller = self.controller
        if controller.info_segment_id is None or controller.info_segment_id not in self._passage_keys:
            self.query_one("#detail", Static).update("")
            return
        text = Text()
        for field, line in controller.detail_rows():
            if not line:
                text.append("\n")
                continue
            if field is not None:
                text.append(_flat(line), style="bold")
                text.append(f"  ({_FIELD_KEYS[field]})\n", style="dim")
            else:
                text.append(_flat(line) + "\n")
        self.query_one("#detail", Static).update(text)

    def refresh_all(self, ok=True):
        self._say(ok)
        self.render_chapters()

    # Navigation ----------------------------------------------------------------

    def on_data_table_row_highlighted(self, event):
        key = event.row_key.value if event.row_key is not None else None
        if event.data_table.id == "chapters" and key is not None:
            if int(key) != self.controller.chapter_index:
                self.controller.focus(chapter_index=int(key))
                self.query_one("#passages", DataTable).move_cursor(row=0)
                self.render_passages()
        elif event.data_table.id == "passages" and key is not None:
            self.controller.focus(segment_id=key)
            self.render_detail()

    def on_data_table_row_selected(self, event):
        if event.data_table.id == "chapters":
            self.query_one("#passages", DataTable).focus()
        elif event.data_table.id == "passages":
            self._passage_menu()

    def action_back(self):
        self.query_one("#chapters", DataTable).focus()

    def action_keys(self):
        self.push_screen(KeysScreen(self.BINDINGS, title="Review keys"))

    def _passage(self):
        """The highlighted passage's ID (the controller is focused on it), or None with a hint."""
        if self.controller.info_segment_id is None or self.controller.info_segment_id not in self._passage_keys:
            self._log("Choose a passage first.", "warning")
            return None
        return self.controller.info_segment_id

    def _passage_menu(self):
        if self._passage() is None:
            return
        options = [("speaker", "Change the speaker (existing cast only)"), ("direction", "Edit the delivery direction"),
                   ("review", "Turn the review flag on or off"), ("play", "Play the accepted take"),
                   ("takes", "Listen to every take; change which one is used")]

        def chosen(value):
            if value is not None:
                getattr(self, "action_" + value)()
        self.push_screen(ChoiceScreen("This passage", options, filterable=False), chosen)

    # Editing -------------------------------------------------------------------

    def action_speaker(self):
        if self._passage() is None:
            return
        options, selected = self.controller.speaker_choices()
        labels = [(role_id, label + (f" — {_flat(reason)}" if reason else "")) for role_id, label, reason in options]
        labels[0] = (None, "Unresolved (clears the assignment)")

        def chosen(role_ids):
            if role_ids is not None:
                ok = self.controller.set_speakers(role_ids)
                self.refresh_all(ok)
        self.push_screen(MultiChoiceScreen(
            "Who speaks this passage?", labels, selected=selected,
            help_text="Likely speakers are first. Enter chooses the highlighted one · Space (in the list) marks "
                      "two or more for a line said together, then Enter · Escape cancels"), chosen)

    def action_direction(self):
        if self._passage() is None:
            return

        def check(text):
            from .attribution_edits import edit_direction
            edit_direction(self.controller._segment(self.controller.info_segment_id), text)

        def chosen(text):
            if text is not None:
                ok = self.controller.set_direction(text)
                self.refresh_all(ok)
        self.push_screen(TextEditScreen("Delivery direction", self.controller.direction(), validate=check,
                                        single_line=True,
                                        help_text="At most 240 characters; empty means none · Enter saves · "
                                                  "Escape cancels"), chosen)

    def action_review(self):
        if self._passage() is not None:
            ok = self.controller.toggle_review()
            self.refresh_all(ok)

    def action_save(self):
        self._log("Saving…")
        self.run_worker(self._save, thread=True, exclusive=True, group="save")

    def _save(self, then_exit=False):
        ok = self.controller.save()

        def done():
            self.refresh_all(ok)
            if ok and then_exit:
                self.exit()
        self.call_from_thread(done)

    def action_quit_review(self):
        if not self.controller.pending:
            self.action_stop()
            self.exit()
            return

        def chosen(value):
            if value == "save":
                self._log("Saving…")
                self.run_worker(lambda: self._save(then_exit=True), thread=True, exclusive=True, group="save")
            elif value == "discard":
                self.action_stop()
                self.exit()
        count = len(self.controller.pending)
        self.push_screen(ChoiceScreen(
            f"{count} unsaved edit(s). Save them before quitting?",
            [("save", "Save and quit"), ("discard", "Discard the edits and quit"), ("stay", "Keep reviewing")],
            filterable=False), chosen)

    # Chapters ------------------------------------------------------------------

    def action_failed(self):
        target = self.controller.failed_passage()
        if target is None:
            self._say(ok=False)
            return
        table = self.query_one("#passages", DataTable)
        table.focus()
        table.move_cursor(row=target)

    def action_ignore(self):
        if not self._chapter_keys:
            return
        index = self.controller.chapter_index
        ignore = index not in self.controller.ignored_chapters
        action = "Ignore" if ignore else "Restore"

        def chosen(yes):
            if yes:
                ok = self.controller.set_ignored(index, ignore)
                if ok:
                    self.controller._sync_toc_selection()
                self.refresh_all(ok)
        self.push_screen(ConfirmScreen(
            f"{action} chapter {index + 1}? This saves now and applies to the next render and assemble; "
            "unsaved attribution edits are unchanged.", title=f"{action} chapter {index + 1}", yes=action), chosen)

    def action_show_ignored(self):
        self.controller.show_ignored = not self.controller.show_ignored
        self.controller._sync_toc_selection()
        self.refresh_all()

    # Listening -----------------------------------------------------------------

    def action_play(self):
        if self._passage() is None:
            return
        row = self.controller.accepted_take()
        if row is None:
            self._say(ok=False)
            return
        if not row.get("audio_path"):
            self._log(f"Take {row.get('attempt', '?')} has no playable audio file.", "warning")
            return
        self._start(row["audio_path"], f"Playing take {row.get('attempt', '?')} (space stops)")

    def _start(self, path, message):
        self.action_stop(quiet=True)
        try:
            self.controller.player.start(str(path))
        except (ValueError, OSError) as error:
            self._log("Playback failed: " + _flat(error), "error")
            return
        self.playing = path
        self._log(message)
        self._subtitle()

    def action_stop(self, quiet=False):
        if self.playing is None:
            return
        try:
            self.controller.player.stop()
        except (ValueError, OSError):
            pass
        self.playing = None
        if not quiet:
            self._log("Playback stopped")
        self._subtitle()

    def action_speed(self):
        from .renderer import playback_speed
        try:
            playback_speed.toggle(self.controller.player)  # A playing take starts again at the new speed.
        except (ValueError, OSError) as error:
            self.playing = None
            self._log("Playback failed: " + _flat(error), "error")
        self._log(playback_speed.toggled_notice(self.controller.player))
        self._subtitle()

    def _poll_playback(self):
        if self.playing is None:
            return
        try:
            still = self.controller.player.poll()
        except (ValueError, OSError) as error:
            self.playing = None
            self._log("Playback failed: " + _flat(error), "error")
            self._subtitle()
            return
        if not still:
            self.playing = None
            self._log("Playback finished")
            self._subtitle()

    def action_takes(self):
        if self._passage() is None:
            return
        chapter, clips = self.controller.passage_clips()
        if not clips:
            self._say(ok=False)
            return
        if len(clips) == 1:
            self._open_takes(chapter, clips[0], "")
            return
        options = [(index, f"Chunk {index + 1}/{len(clips)}  {clip['chunk_id']}") for index, clip in enumerate(clips)]

        def chosen(index):
            if index is not None:
                self._open_takes(chapter, clips[index], f" — chunk {index + 1}/{len(clips)}")
        self.push_screen(ChoiceScreen("This passage was rendered in several chunks. Which one?", options,
                                      filterable=False), chosen)

    def _open_takes(self, chapter, clip, part):
        from .run_shell_screens import TakesScreen
        self.action_stop(quiet=True)
        controller = self.controller
        review = controller.take_review(chapter, clip)
        number = controller.chapter_index + 1

        def retired(attempt):
            return controller.rerender_notice(chapter, attempt)

        def chosen(choice):
            if choice is not None:
                self.run_worker(lambda: self._use_take(chapter, clip, review, choice["attempt"]),
                                thread=True, group="takes")
        self.push_screen(TakesScreen(f"Takes — chapter {number}{part}", review, controller.player,
                                     on_retired=retired), chosen)

    def _use_take(self, chapter, clip, review, attempt):
        ok = self.controller.accept_take(chapter, clip, attempt)
        if ok:
            try:
                review.mark(attempt, "good")  # Choosing a take also judges it good.
            except (ValueError, OSError):
                pass
        self.call_from_thread(self.refresh_all, ok)


def run_review_tui(directory, *, player_command=None):
    """Open a completed book in the full-screen review; never creates voices or calls models."""
    from pathlib import Path
    from .analysis_directory import _completed_directory
    from .analysis_revision import related_directories, revise_directories
    from .review_tui import ReviewApp, _audio_player
    directory = Path(directory).resolve()
    document = _completed_directory(directory)[0]
    controller = ReviewApp(directory, document, revise_directories=revise_directories,
                           related_directories=related_directories,
                           player=_audio_player(player_command or ["/usr/bin/afplay"]))
    ReviewTuiApp(controller).run()
    return controller
