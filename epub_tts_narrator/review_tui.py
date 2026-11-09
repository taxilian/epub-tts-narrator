"""Interactive terminal review for published analyses; existing cast members only."""
# noqa: SIZE_OK — the pure terminal state machine and its sole curses adapter
# share one event/draw contract and are deliberately independent of the model.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from .analysis_directory import _completed_directory
from .analysis_revision import related_directories, revise_directories
from .review_tui_model import (
    apply_edits, edit_direction, edit_review, edit_speaker, metadata, palette, passages, toc,
)
from .speaker_decisions import _display, _role_labels
from .speaker_suggestions import rank_speakers


_KIND_BADGES = {"narration": "N", "dialogue": "D", "thought": "T",
                "letter": "L", "heading": "H", "other": "O"}


def _flat(value):
    """Make untrusted book/display text safe and single-line for a terminal."""
    return " ".join(_display(value).split())


def _selected(selected, role=None):
    return {"reverse": bool(selected), "role": role}


class ReviewApp:
    """Curses-independent review state machine with receipt-bound saves."""

    def __init__(self, directory, document, *, revise_directories, related_directories):
        self.directory = Path(directory).resolve()
        self.document = document
        self._revise = revise_directories
        self.directories = [Path(path).resolve() for path in related_directories([self.directory])]
        self.expected_manifest_hashes = self._manifest_hashes()
        self.pending = {}
        self.running = True
        self.view = "toc"
        self.chapter_index = 0
        self.passage_index = 0
        self.passage_scroll = 0
        self.info_segment_id = None
        self.info_cursor = 0
        self.info_scroll = 0
        self.speaker_options = []
        self.speaker_index = 0
        self.speaker_selected = []
        self.speaker_candidates = []
        self.speaker_reasons = {}
        self.direction_buffer = ""
        self.return_view = "toc"
        self.notice = ""
        self.progress = []
        self.page_size = 10
        self.roles = {role["id"]: role for role in document["cast"]["roles"]}
        self.labels = _role_labels(document["cast"]["roles"],
                                   {row["id"]: row for row in document["cast"]["characters"]},
                                   document["source"]["book"])

    def _manifest_hashes(self):
        return {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                for path in self.directories}

    def _effective_document(self):
        return apply_edits(self.document, self.pending) if self.pending else self.document

    def _toc_rows(self):
        return toc(self._effective_document())

    def _passage_rows(self):
        return passages(self._effective_document(), self.chapter_index)

    def _segment(self, segment_id):
        if segment_id in self.pending:
            return self.pending[segment_id]
        return next(segment for segment in self.document["segments"] if segment["id"] == segment_id)

    def _current_passage_id(self):
        rows = self._passage_rows()
        return rows[self.passage_index]["segment_id"] if rows else None

    def _store_edit(self, segment):
        original = next(row for row in self.document["segments"] if row["id"] == segment["id"])
        if segment == original:
            self.pending.pop(segment["id"], None)
        else:
            self.pending[segment["id"]] = segment
        self.notice = "Unsaved changes" if self.pending else "No unsaved changes"

    def _move_chapter(self, amount):
        count = len(self._toc_rows())
        self.chapter_index = max(0, min(count - 1, self.chapter_index + amount))

    def _move_passage(self, amount):
        count = len(self._passage_rows())
        self.passage_index = max(0, min(count - 1, self.passage_index + amount))

    def _open_info(self):
        segment_id = self._current_passage_id()
        if segment_id is not None:
            self.info_segment_id = segment_id
            self.info_cursor = 0
            self.info_scroll = 0
            self.view = "info"

    def _open_speaker(self):
        document = self._effective_document()
        segment = self._segment(self.info_segment_id)
        ranked = rank_speakers(document, segment)
        self.speaker_candidates = [row["role_id"] for row in ranked if row["role_id"] in self.roles]
        self.speaker_reasons = {row["role_id"]: row.get("reason", "") for row in ranked}
        ordered = list(self.speaker_candidates)
        ordered.extend(role["id"] for role in document["cast"]["roles"] if role["id"] not in ordered)
        self.speaker_options = [None, *ordered]
        current: list = metadata(document, self.info_segment_id)["role_ids"]
        self.speaker_selected = list(current)
        self.speaker_index = self.speaker_options.index(current[0]) if current else 0
        self.view = "speaker"

    def _commit_speaker(self, role_ids):
        try:
            edited = edit_speaker(self._segment(self.info_segment_id), role_ids,
                                  self.roles, self.labels, candidates=self.speaker_candidates)
        except ValueError as exc:
            self.notice = str(exc)
            return
        self._store_edit(edited)
        self.view = "info"

    def _toggle_review(self):
        segment = self._segment(self.info_segment_id)
        try:
            edited = edit_review(segment, not segment["attribution"]["review_required"])
        except ValueError as exc:
            self.notice = str(exc)
            return
        self._store_edit(edited)

    def _activate_info_field(self, field):
        if field == "speaker":
            self._open_speaker()
        elif field == "direction":
            self.direction_buffer = self._segment(self.info_segment_id)["delivery"]["direction"]
            self.view = "direction"
        elif field == "review":
            self._toggle_review()

    def _request_quit(self):
        if not self.pending:
            self.running = False
            return
        self.return_view = self.view
        self.view = "confirm_quit"

    def save(self):
        """Publish pending edits without printing through the active curses screen."""
        if not self.pending:
            self.notice = "No unsaved changes"
            return True
        revised = apply_edits(self.document, self.pending)
        messages = []
        try:
            result = self._revise(
                {self.directory: revised}, directories=self.directories,
                expected_manifest_hashes=self.expected_manifest_hashes,
                scope="attribution", progress=messages.append,
            )
        except (ValueError, OSError) as exc:
            self.notice = "Save failed: " + str(exc)
            return False
        self.document = revised
        self.pending.clear()
        self.expected_manifest_hashes = self._manifest_hashes()
        self.progress.extend(messages)
        backup = result.get("backup_directory") if isinstance(result, dict) else None
        if backup:
            self.progress.append("Original analysis files and revision journal: " + str(backup))
        self.notice = "Saved revision"
        return True

    def _handle_confirm_quit(self, key):
        if key == "y":
            if self.save():
                self.running = False
            else:
                self.view = self.return_view
        elif key == "n":
            self.running = False
        elif key in {"ESC", "q", "Q"}:
            self.view = self.return_view

    def _handle_direction(self, key):
        if key == "ENTER":
            try:
                edited = edit_direction(self._segment(self.info_segment_id), self.direction_buffer)
            except ValueError as exc:
                self.notice = str(exc)
                return
            self._store_edit(edited)
            self.view = "info"
        elif key == "ESC":
            self.view = "info"
        elif key == "BACKSPACE":
            self.direction_buffer = self.direction_buffer[:-1]
        elif isinstance(key, str) and len(key) == 1 and key.isprintable():
            self.direction_buffer += key

    def _handle_speaker(self, key):
        if key == "ESC":
            self.view = "info"
            return
        if key in {"UP", "k"}:
            self.speaker_index = max(0, self.speaker_index - 1)
            return
        if key in {"DOWN", "j"}:
            self.speaker_index = min(len(self.speaker_options) - 1, self.speaker_index + 1)
            return
        role_id = self.speaker_options[self.speaker_index]
        if key == "SPACE" and role_id is not None:
            if role_id in self.speaker_selected:
                self.speaker_selected.remove(role_id)
            else:
                self.speaker_selected.append(role_id)
            return
        if key != "ENTER":
            return
        if role_id is None:
            self._commit_speaker([])
        elif len(self.speaker_selected) >= 2 or role_id in self.speaker_selected:
            self._commit_speaker(self.speaker_selected)
        else:
            self._commit_speaker([role_id])

    def handle_key(self, key):
        """Apply one symbolic key and return whether the app continues running."""
        if self.view == "confirm_quit":
            self._handle_confirm_quit(key)
            return self.running
        if self.view == "direction":
            self._handle_direction(key)
            return self.running
        if self.view == "speaker":
            self._handle_speaker(key)
            return self.running
        if key in {"s", "S"}:
            self.save()
            return self.running
        if key in {"q", "Q"}:
            self._request_quit()
            return self.running
        if self.view == "toc":
            if key in {"UP", "k"}:
                self._move_chapter(-1)
            elif key in {"DOWN", "j"}:
                self._move_chapter(1)
            elif key == "PPAGE":
                self._move_chapter(-self.page_size)
            elif key == "NPAGE":
                self._move_chapter(self.page_size)
            elif key == "ENTER":
                self.passage_index = self.passage_scroll = 0
                self.view = "passages"
        elif self.view == "passages":
            if key in {"UP", "k"}:
                self._move_passage(-1)
            elif key in {"DOWN", "j"}:
                self._move_passage(1)
            elif key == "PPAGE":
                self._move_passage(-self.page_size)
            elif key == "NPAGE":
                self._move_passage(self.page_size)
            elif key == "ENTER":
                self._open_info()
            elif key == "ESC":
                self.view = "toc"
        elif self.view == "info":
            rows = self._info_rows()
            if key in {"UP", "k"}:
                self.info_cursor = max(0, self.info_cursor - 1)
            elif key in {"DOWN", "j"}:
                self.info_cursor = min(len(rows) - 1, self.info_cursor + 1)
            elif key == "PPAGE":
                self.info_scroll = max(0, self.info_scroll - self.page_size)
            elif key == "NPAGE":
                self.info_scroll = min(max(0, len(rows) - 1), self.info_scroll + self.page_size)
            elif key == "ENTER":
                self._activate_info_field(rows[self.info_cursor][0])
            elif key == "F2":
                self._open_speaker()
            elif key == "F3":
                self._activate_info_field("direction")
            elif key in {"F4", "r"}:
                self._toggle_review()
            elif key == "ESC":
                self.view = "passages"
        return self.running

    def _info_rows(self):
        data: dict = metadata(self._effective_document(), self.info_segment_id)
        attribution = data["attribution"]
        delivery = data["delivery"]
        speaker = ", ".join(data["speaker_labels"]) if data["speaker_labels"] else "Unassigned"
        rows = [
            ("speaker", "Speaker(s): " + speaker),
            ("direction", "Direction: " + (delivery["direction"] or "(none)")),
            ("review", "Review required: " + ("yes" if attribution["review_required"] else "no")),
            (None, ""),
            (None, "Passage: " + data["text"]),
            (None, f"ID: {data['id']}  Block: {data['block_id']}  Chapter: {data['chapter_id']}"),
            (None, f"Sequence: {data['sequence']}  Kind: {data['kind']}  "
                   f"Offsets: {data['offsets']['start']}-{data['offsets']['end']}"),
            (None, "Role IDs: " + (", ".join(data["role_ids"]) or "(none)")),
            (None, "Voice IDs: " + (", ".join(data["voice_ids"]) or "(none)")),
        ]
        for index, description in enumerate(data["voice_descriptions"], 1):
            rows.append((None, f"Voice {index} description: {description}"))
        rows.extend([
            (None, "Attribution basis: " + attribution["basis"]),
            (None, "Attribution reason: " + attribution["reason"]),
            (None, "Attribution candidates: " + (", ".join(attribution["candidates"]) or "(none)")),
        ])
        for item in attribution["evidence"]:
            rows.append((None, f"Attribution evidence {item['block_id']}: {item['quote']}"))
        for item in delivery["evidence"]:
            rows.append((None, f"Delivery evidence {item['block_id']}: {item['quote']}"))
        controls = data["performance_controls"]
        kinds = ", ".join(f"{kind}={count}" for kind, count in sorted(controls["kinds"].items())) or "none"
        rows.append((None, f"Performance controls (read-only): {controls['total']} ({kinds})"))
        if "simultaneous_speakers" in data:
            rows.append((None, "Simultaneous: " + json.dumps(data["simultaneous_speakers"], ensure_ascii=False)))
        if data["additional"]:
            rows.append((None, "Additional metadata: " + json.dumps(data["additional"], ensure_ascii=False)))
        return rows

    def _header(self, screen, title):
        _, width = screen.size()
        screen.write(0, 0, _flat(title), {"reverse": True})
        dirty = f"  {len(self.pending)} unsaved edit(s)" if self.pending else ""
        screen.write(1, 0, _flat(self.document["source"]["book"]["title"] + dirty), None)

    def _footer(self, screen, controls):
        height, _ = screen.size()
        if self.notice:
            screen.write(max(0, height - 2), 0, _flat(self.notice), None)
        screen.write(max(0, height - 1), 0, controls, {"reverse": True})

    def _draw_toc(self, screen):
        height, _ = screen.size()
        rows = self._toc_rows()
        self._header(screen, f"Review chapters  {self.chapter_index + 1}/{len(rows)}")
        self.page_size = max(1, height - 5)
        start = max(0, min(self.chapter_index - self.page_size + 1, len(rows) - self.page_size))
        start = max(0, start)
        for line, row in enumerate(rows[start:start + self.page_size], 3):
            index = start + line - 3
            marker = "> " if index == self.chapter_index else "  "
            text = (f"{marker}{row['title']}  {row['passage_count']} passages; "
                    f"{row['review_count']} review; {row['unassigned_count']} unassigned")
            screen.write(line, 0, text, _selected(index == self.chapter_index))
        self._footer(screen, "Up/Down select  Enter passages  S save  Q quit")

    def _draw_passages(self, screen):
        height, _ = screen.size()
        rows = self._passage_rows()
        chapter = self._toc_rows()[self.chapter_index]
        self._header(screen, chapter["title"] + f"  Passage {self.passage_index + 1}/{len(rows)}")
        self.page_size = max(1, height - 5)
        if self.passage_index < self.passage_scroll:
            self.passage_scroll = self.passage_index
        if self.passage_index >= self.passage_scroll + self.page_size:
            self.passage_scroll = self.passage_index - self.page_size + 1
        for line, row in enumerate(rows[self.passage_scroll:self.passage_scroll + self.page_size], 3):
            index = self.passage_scroll + line - 3
            selected = index == self.passage_index
            badge = f"{_KIND_BADGES.get(row['kind'], '?')}{'!' if row['review_required'] else ' '}"
            screen.write(line, 0, ("> " if selected else "  ") + badge + " ", _selected(selected))
            x = 5
            if row["speaker_labels"]:
                for speaker_index, (label, role_index) in enumerate(zip(row["speaker_labels"], row["role_indices"])):
                    if speaker_index:
                        screen.write(line, x, ", ", _selected(selected))
                        x += 2
                    screen.write(line, x, _flat(label), _selected(selected, role_index))
                    x += len(_flat(label))
            else:
                screen.write(line, x, "Unassigned", _selected(selected))
                x += len("Unassigned")
            screen.write(line, x, "  " + _flat(row["text"]), _selected(selected))
        self._footer(screen, "Up/Down/PgUp/PgDn  Enter info  Esc chapters  S save  Q quit")

    def _draw_info(self, screen):
        height, _ = screen.size()
        self._header(screen, "Passage metadata")
        rows = self._info_rows()
        self.page_size = max(1, height - 5)
        if self.info_cursor < self.info_scroll:
            self.info_scroll = self.info_cursor
        if self.info_cursor >= self.info_scroll + self.page_size:
            self.info_scroll = self.info_cursor - self.page_size + 1
        for line, (field, text) in enumerate(rows[self.info_scroll:self.info_scroll + self.page_size], 3):
            index = self.info_scroll + line - 3
            marker = "> " if index == self.info_cursor and field else "  "
            screen.write(line, 0, marker + _flat(text), _selected(index == self.info_cursor and field is not None))
        self._footer(screen, "Enter edit  F2 speaker  F3 direction  F4/R review  PgUp/PgDn scroll  Esc back  S save")

    def _draw_speaker(self, screen):
        height, _ = screen.size()
        self._header(screen, "Choose existing speaker(s)")
        self.page_size = max(1, height - 5)
        start = max(0, min(self.speaker_index - self.page_size + 1, len(self.speaker_options) - self.page_size))
        start = max(0, start)
        for line, role_id in enumerate(self.speaker_options[start:start + self.page_size], 3):
            index = start + line - 3
            label = "Unresolved" if role_id is None else self.labels[role_id]
            chosen = role_id in self.speaker_selected if role_id is not None else not self.speaker_selected
            reason = self.speaker_reasons.get(role_id, "")
            suffix = "  " + _flat(reason) if reason else ""
            marker = "> " if index == self.speaker_index else "  "
            screen.write(line, 0, f"{marker}[{'x' if chosen else ' '}] {_flat(label)}{suffix}",
                         _selected(index == self.speaker_index,
                                   list(self.roles).index(role_id) if role_id is not None else None))
        self._footer(screen, "Space toggle group  Enter confirm/choose  Unresolved clears assignment  Esc cancel")

    def _draw_direction(self, screen):
        self._header(screen, "Edit delivery direction")
        screen.write(3, 0, self.direction_buffer, None)
        screen.write(4, 0, f"{len(self.direction_buffer)}/240 characters", None)
        self._footer(screen, "Printable text  Backspace delete  Enter apply  Esc cancel")

    def draw(self, screen):
        """Render the current state through a curses-independent screen protocol."""
        screen.clear()
        if self.view == "toc":
            self._draw_toc(screen)
        elif self.view == "passages":
            self._draw_passages(screen)
        elif self.view == "info":
            self._draw_info(screen)
        elif self.view == "speaker":
            self._draw_speaker(screen)
        elif self.view == "direction":
            self._draw_direction(screen)
        elif self.view == "confirm_quit":
            self._header(screen, "Unsaved changes")
            screen.write(3, 0, "Save changes before quitting?", None)
            screen.write(4, 0, "Y save and quit; N discard and quit; Esc return", None)
            self._footer(screen, "Y save  N discard  Esc cancel")
        screen.refresh()


class CursesScreen:
    """The only curses-aware adapter; attributes and keys remain symbolic outside."""

    def __init__(self, curses_module, screen, role_colors):
        self.curses = curses_module
        self.screen = screen
        self.color_pairs = {}
        self.screen.keypad(True)
        try:
            self.curses.curs_set(0)
        except self.curses.error:
            pass
        if not role_colors or not self.curses.has_colors():
            return
        try:
            self.curses.start_color()
            self.curses.use_default_colors()
            if self.curses.COLORS <= max(role_colors):
                return
            for slot, color in enumerate(role_colors[:self.curses.COLOR_PAIRS - 1], 1):
                self.curses.init_pair(slot, color, -1)
                self.color_pairs[slot - 1] = slot
        except self.curses.error:
            self.color_pairs.clear()

    def size(self):
        return self.screen.getmaxyx()

    def clear(self):
        self.screen.erase()

    def write(self, y, x, text, attributes):
        height, width = self.size()
        if y < 0 or y >= height or x < 0 or x >= width or not text:
            return
        attributes = attributes or {}
        attr = self.curses.A_REVERSE if attributes.get("reverse") else 0
        role = attributes.get("role")
        if role is not None and self.color_pairs:
            attr |= self.curses.color_pair(self.color_pairs[role % len(self.color_pairs)])
        available = width - x - (1 if y == height - 1 else 0)
        if available <= 0:
            return
        try:
            self.screen.addnstr(y, x, str(text), available, attr)
        except self.curses.error:
            # Wide Unicode at the last terminal cell has no safe partial write.
            return

    def refresh(self):
        self.screen.refresh()

    def read_key(self):
        key = self.screen.get_wch()
        if isinstance(key, int):
            mapping = {
                self.curses.KEY_UP: "UP", self.curses.KEY_DOWN: "DOWN",
                self.curses.KEY_PPAGE: "PPAGE", self.curses.KEY_NPAGE: "NPAGE",
                self.curses.KEY_ENTER: "ENTER", self.curses.KEY_BACKSPACE: "BACKSPACE",
                self.curses.KEY_F2: "F2", self.curses.KEY_F3: "F3", self.curses.KEY_F4: "F4",
                self.curses.KEY_RESIZE: "RESIZE",
            }
            return mapping.get(key, "UNKNOWN")
        if key in {"\n", "\r"}:
            return "ENTER"
        if key == "\x1b":
            return "ESC"
        if key in {"\b", "\x7f"}:
            return "BACKSPACE"
        if key == " ":
            return "SPACE"
        return key


def _drive(app, screen):
    while app.running:
        app.draw(screen)
        key = screen.read_key()
        if key is None:
            raise ValueError("review-ui input ended")
        app.handle_key(key)
    return app


def run_review(directory, *, screen=None):
    """Open a completed book in the review UI; never create voices or call models."""
    directory = Path(directory).resolve()
    if screen is None and (not sys.stdin.isatty() or not sys.stdout.isatty() or not os.environ.get("TERM")):
        raise ValueError("review-ui requires an interactive terminal")
    document = _completed_directory(directory)[0]
    app = ReviewApp(directory, document, revise_directories=revise_directories,
                    related_directories=related_directories)
    if screen is not None:
        return _drive(app, screen)
    import curses

    def wrapped(stdscr):
        adapter = CursesScreen(curses, stdscr, palette(len(app.roles)))
        return _drive(app, adapter)

    return curses.wrapper(wrapped)
