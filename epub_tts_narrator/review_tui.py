"""Interactive terminal review for published analyses; existing cast members only."""
# noqa: SIZE_OK — the pure terminal state machine and its sole curses adapter
# share one event/draw contract and are deliberately independent of the model.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from .analysis_directory import _completed_directory
from .analysis_revision import related_directories, revise_directories
from .renderer.terminal import play_sample
from .review_tui_model import (
    apply_edits, edit_direction, edit_review, edit_speaker, metadata, palette, passages, toc,
)
from .review_tui_render import (
    ChapterIndex, ClipRef, RenderIndex, TakeRow, archive_checkpoint, format_take, playback_notice,
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


def _audio_player(command):
    from .renderer.audition import AudioPlayer
    return AudioPlayer(command)


class ReviewApp:
    """Curses-independent review state machine with receipt-bound saves."""

    def __init__(self, directory, document, *, revise_directories, related_directories,
                 player=None, play_sample=play_sample, list_takes=None, select_take=None, render_config=None):
        self.directory = Path(directory).resolve()
        self.document = document
        self._revise = revise_directories
        self.directories = [Path(path).resolve() for path in related_directories([self.directory])]
        self.expected_manifest_hashes = self._manifest_hashes()
        self.player = player if player is not None else _audio_player(["/usr/bin/afplay"])
        self._play_sample = play_sample
        self._list_takes = list_takes
        self._select_take = select_take
        self.render_index = RenderIndex(self.directory, config=render_config)
        self.pending = {}
        self.running = True
        self.view = "toc"
        self.chapter_index = 0
        self.passage_index = 0
        self.passage_scroll = 0
        self.info_segment_id: str | None = None
        self.info_cursor = 0
        self.info_scroll = 0
        self.speaker_options = []
        self.speaker_index = 0
        self.speaker_selected = []
        self.speaker_candidates = []
        self.speaker_reasons = {}
        self.direction_buffer = ""
        self.takes_segment_id: str | None = None
        self.takes_chapter: ChapterIndex | None = None
        self.takes_clips: list[ClipRef] = []
        self.takes_chunk_index = 0
        self.takes_rows: list[TakeRow] = []
        self.takes_index = 0
        self.takes_scroll = 0
        self.confirm_take_attempt: int | None = None
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

    def _chapter_failure(self, chapter_index=None):
        """Return the persisted render failure for one chapter, if any."""
        index = self.chapter_index if chapter_index is None else chapter_index
        try:
            return self.render_index.failures().get(index + 1)
        except (ValueError, OSError):
            return None

    def _toc_rows(self):
        rows = toc(self._effective_document())
        for row in rows:
            row["failure"] = self._chapter_failure(row["id"])
        return rows

    def _passage_rows(self):
        rows = passages(self._effective_document(), self.chapter_index)
        failure = self._chapter_failure()
        known_ids = {row["segment_id"] for row in rows}
        failed_ids = set(failure["segment_ids"]) & known_ids if failure else set()
        for row in rows:
            row["failed"] = row["segment_id"] in failed_ids
        return rows

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

    def _jump_to_failure(self):
        """Open the chapter at its first known failed passage, when linkage exists."""
        failure = self._chapter_failure()
        if failure is None:
            self.notice = "This chapter has no recorded render failure."
            return
        rows = self._passage_rows()
        target = next((index for index, row in enumerate(rows) if row["failed"]), None)
        if target is None:
            self.notice = (f"Chapter {self.chapter_index + 1} render failed "
                           f"({failure['error_type']}); no exact failed-passage linkage was recorded.")
            return
        self.passage_index = target
        self.passage_scroll = 0
        self.view = "passages"

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
        current: list[str] = metadata(document, self.info_segment_id)["role_ids"]
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
        elif field == "takes":
            self._open_takes()

    def _render_chapter(self) -> ChapterIndex:
        try:
            return self.render_index.chapter(self.chapter_index + 1)
        except (ValueError, OSError) as exc:
            return {"status": "error", "work_directory": self.directory / "render-work",
                    "segments": {}, "error": _flat(exc)}

    def _not_rendered_notice(self, chapter: ChapterIndex):
        failure = self._chapter_failure()
        if failure is not None:
            # The failure record alone never proves audio absence; only the
            # empty segment index does. Rejected takes can stay playable.
            return (f"Render failed for chapter {self.chapter_index + 1} "
                    f"({failure['error_type']}): {_flat(failure['error'])} "
                    "No completed audio is indexed for this passage; "
                    "rerun render to regenerate it.")
        status = chapter["status"]
        if status == "render_work_missing":
            return "Not rendered: this book has no render-work directory."
        if status == "checkpoint_missing":
            return f"Not rendered: chapter {self.chapter_index + 1} has no completed-chapter checkpoint."
        if status == "error":
            return "Could not read rendered takes: " + chapter.get("error", "unknown error")
        return "Not rendered: no clip for this passage."

    def _take_lister(self):
        if self._list_takes is None:
            from .renderer.take_selection import list_takes
            self._list_takes = list_takes
        return self._list_takes

    def _take_selector(self):
        if self._select_take is None:
            from .renderer.take_selection import select_take
            self._select_take = select_take
        return self._select_take

    def _clip_takes(self, chapter: ChapterIndex, clip: ClipRef):
        clips = chapter["work_directory"] / "clips"
        return self._take_lister()(clips, clip["request_hash"])

    def _open_takes(self):
        self.takes_segment_id = self.info_segment_id
        self.takes_chapter = self._render_chapter()
        clips = self.takes_chapter["segments"].get(self.takes_segment_id, [])
        self.takes_clips = list(clips)
        self.takes_chunk_index = 0
        self.takes_index = self.takes_scroll = 0
        self.takes_rows = []
        self.view = "takes"
        if not self.takes_clips:
            self.notice = self._not_rendered_notice(self.takes_chapter)
            return
        self._load_takes()

    def _load_takes(self):
        if not self.takes_clips:
            self.takes_rows = []
            return
        clip = self.takes_clips[self.takes_chunk_index]
        assert self.takes_chapter is not None
        try:
            self.takes_rows = list(self._clip_takes(self.takes_chapter, clip))
        except (ValueError, OSError) as exc:
            self.takes_rows = []
            self.notice = "Could not list takes: " + _flat(exc)
            return
        self.takes_index = max(0, min(self.takes_index, len(self.takes_rows) - 1))

    def _current_take(self) -> TakeRow | None:
        if not self.takes_rows:
            return None
        return self.takes_rows[self.takes_index]

    def _play_audio(self, row: TakeRow, screen):
        attempt = row.get("attempt", "?")
        if not row.get("audio_path"):
            self.notice = f"Take {attempt} has no playable audio file."
            return
        if screen is not None:
            screen.clear()
            self._header(screen, f"Playing take {attempt}")
            screen.write(3, 0, f"Playing take {attempt} — any key stops", None)
            self._footer(screen, "Any key stops playback")
            screen.refresh()
        try:
            outcome = self._play_sample(self.player, row["audio_path"])
        except (ValueError, RuntimeError, OSError) as exc:
            outcome = exc
        self.notice = playback_notice(outcome)

    def _play_info_accepted(self, screen):
        chapter = self._render_chapter()
        clips = chapter["segments"].get(self.info_segment_id, [])
        if not clips:
            self.notice = self._not_rendered_notice(chapter)
            return
        if len(clips) > 1:
            self._open_takes()
            self.notice = "This passage has multiple rendered chunks; choose a chunk and take to play."
            return
        try:
            rows = self._clip_takes(chapter, clips[0])
        except (ValueError, OSError) as exc:
            self.notice = "Could not list takes: " + _flat(exc)
            return
        accepted = next((row for row in rows if row.get("accepted")), None)
        if accepted is None:
            self.notice = "Not rendered: this passage has no accepted take."
            return
        self._play_audio(accepted, screen)

    def _request_take_acceptance(self):
        assert self.takes_chapter is not None
        row = self._current_take()
        if row is None:
            self.notice = self._not_rendered_notice(self.takes_chapter)
            return
        attempt = row["attempt"]
        if row.get("echo"):
            reason = row.get("ineligible_reason") or "history echo"
            self.notice = f"Take {attempt} cannot be accepted: {_flat(reason)}."
            return
        if not row.get("eligible"):
            reason = row.get("ineligible_reason") or "not eligible"
            self.notice = f"Take {attempt} cannot be accepted: {_flat(reason)}."
            return
        self.confirm_take_attempt = attempt
        self.view = "confirm_take"

    def _confirm_take_acceptance(self):
        assert self.takes_chapter is not None and self.confirm_take_attempt is not None
        attempt = self.confirm_take_attempt
        clip = self.takes_clips[self.takes_chunk_index]
        clips = self.takes_chapter["work_directory"] / "clips"
        self.view = "takes"
        try:
            self._take_selector()(clips, clip["request_hash"], attempt, note="")
        except ValueError as exc:
            self.notice = _flat(exc)
            return
        chapter_number = self.chapter_index + 1
        try:
            archived = archive_checkpoint(self.takes_chapter["work_directory"], chapter_number, token=attempt)
            archive_error = ""
        except OSError as exc:
            archived = None
            archive_error = _flat(exc)
        self._load_takes()
        book = shlex.quote(str(self.directory))
        command = (f"audiobook-render render {book} --chapters-only {chapter_number} && "
                   f"audiobook-render assemble {book}")
        archived_notice = "Chapter checkpoint archived. " if archived else ""
        if archive_error:
            archived_notice = "Checkpoint archive failed: " + archive_error + ". "
        self.notice = f"Accepted take {attempt}. {archived_notice}Rerun: {command}"

    def _handle_takes(self, key, screen):
        if key == "ESC":
            self.view = "info"
        elif key in {"UP", "k"} and self.takes_rows:
            self.takes_index = max(0, self.takes_index - 1)
        elif key in {"DOWN", "j"} and self.takes_rows:
            self.takes_index = min(len(self.takes_rows) - 1, self.takes_index + 1)
        elif key in {"LEFT", "RIGHT"} and self.takes_clips:
            amount = -1 if key == "LEFT" else 1
            self.takes_chunk_index = max(0, min(len(self.takes_clips) - 1, self.takes_chunk_index + amount))
            self.takes_index = self.takes_scroll = 0
            self._load_takes()
        elif key in {"ENTER", "p", "P"}:
            row = self._current_take()
            if row is not None:
                self._play_audio(row, screen)
        elif key in {"a", "A"}:
            self._request_take_acceptance()

    def _handle_confirm_take(self, key):
        if key in {"y", "Y"}:
            self._confirm_take_acceptance()
        elif key in {"n", "ESC", "q", "Q"}:
            self.view = "takes"

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
        self.render_index.reset()
        self.expected_manifest_hashes = self._manifest_hashes()
        self.progress.extend(messages)
        backup = result.get("backup_directory") if isinstance(result, dict) else None
        if backup:
            self.progress.append("Original analysis files and revision journal: " + str(backup))
        self.notice = "Saved revision"
        if self.render_index.failures():
            self.progress.append("Render failures are unchanged: failed chapters still need a successful rerender.")
            self.notice = "Saved revision; recorded render failures still need a successful rerender"
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

    def handle_key(self, key, *, screen=None):
        """Apply one symbolic key and return whether the app continues running."""
        if self.view == "confirm_quit":
            self._handle_confirm_quit(key)
            return self.running
        if self.view == "confirm_take":
            self._handle_confirm_take(key)
            return self.running
        if self.view == "direction":
            self._handle_direction(key)
            return self.running
        if self.view == "speaker":
            self._handle_speaker(key)
            return self.running
        if self.view == "takes":
            self._handle_takes(key, screen)
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
            elif key in {"f", "F"}:
                self._jump_to_failure()
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
            elif key in {"f", "F"}:
                self._jump_to_failure()
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
            elif key in {"t", "T"}:
                self._open_takes()
            elif key in {"p", "P"}:
                self._play_info_accepted(screen)
            elif key == "ESC":
                self.view = "passages"
        return self.running

    def _info_rows(self):
        data: dict[str, Any] = metadata(self._effective_document(), self.info_segment_id)
        attribution = data["attribution"]
        delivery = data["delivery"]
        speaker = ", ".join(data["speaker_labels"]) if data["speaker_labels"] else "Unassigned"
        rows = [
            ("speaker", "Speaker(s): " + speaker),
            ("direction", "Direction: " + (delivery["direction"] or "(none)")),
            ("review", "Review required: " + ("yes" if attribution["review_required"] else "no")),
            ("takes", "Takes: view and play rendered attempts"),
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
        failure = self._chapter_failure()
        if failure is not None:
            rows.append((None, "Render: FAILED — " + failure["error_type"] + ": " + _flat(failure["error"])))
            linked = set(failure["segment_ids"])
            if data["id"] in linked:
                rows.append((None, "The recorded failure is linked to this passage."))
                rows.append((None, "These edits stay pending until a successful rerender."))
            elif linked:
                rows.append((None, "The recorded failure is linked to another passage, not this one."))
                rows.append((None, "This chapter still needs a successful rerender before edits "
                                   "reach audio."))
            else:
                rows.append((None, "The render failure has no exact passage linkage."))
                rows.append((None, "Which passages lack completed audio is unknown until a "
                                   "successful rerender."))
                rows.append((None, "These edits stay pending until then."))
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
            prefix = f"FAILED {row['failure']['error_type']}  " if row.get("failure") else ""
            text = (f"{marker}{prefix}{row['title']}  {row['passage_count']} passages; "
                    f"{row['review_count']} review; {row['unassigned_count']} unassigned")
            screen.write(line, 0, text, _selected(index == self.chapter_index))
        self._footer(screen, "Up/Down select  Enter passages  F failed passage  S save  Q quit")

    def _draw_passages(self, screen):
        height, _ = screen.size()
        rows = self._passage_rows()
        chapter = self._toc_rows()[self.chapter_index]
        title = chapter["title"] + f"  Passage {self.passage_index + 1}/{len(rows)}"
        if chapter.get("failure"):
            title = "FAILED  " + title + "  " + _flat(chapter["failure"]["error"])
        self._header(screen, title)
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
            if row.get("failed"):
                screen.write(line, x, "[failed] ", _selected(selected))
                x += len("[failed] ")
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
        self._footer(screen, "Up/Down/PgUp/PgDn  Enter info  F failed passage  Esc chapters  S save  Q quit")

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
        self._footer(screen, "Enter edit/action  P play  T takes  F2 speaker  F3 direction  F4/R review  Esc back  S save")

    def _draw_takes(self, screen):
        height, _ = screen.size()
        chapter_number = self.chapter_index + 1
        self._header(screen, f"Rendered takes  Chapter {chapter_number}")
        if not self.takes_clips:
            screen.write(3, 0, "No rendered clips for this passage.", None)
            failure = self._chapter_failure()
            if failure is not None:
                screen.write(4, 0, "Render failed: " + _flat(failure["error_type"]) + ": "
                             + _flat(failure["error"]), None)
                screen.write(5, 0, "No completed audio is indexed for this passage.", None)
            self._footer(screen, "Esc passage")
            return
        clip = self.takes_clips[self.takes_chunk_index]
        chunk = f"Chunk {self.takes_chunk_index + 1}/{len(self.takes_clips)}  {clip['chunk_id']}"
        screen.write(3, 0, chunk, None)
        screen.write(4, 0, "* accepted", None)
        self.page_size = max(1, height - 9)
        if self.takes_index < self.takes_scroll:
            self.takes_scroll = self.takes_index
        if self.takes_index >= self.takes_scroll + self.page_size:
            self.takes_scroll = self.takes_index - self.page_size + 1
        for line, row in enumerate(self.takes_rows[self.takes_scroll:self.takes_scroll + self.page_size], 6):
            index = self.takes_scroll + line - 6
            selected = index == self.takes_index
            screen.write(line, 0, ("> " if selected else "  ") + format_take(row), _selected(selected))
        chunks = "  Left/Right chunk" if len(self.takes_clips) > 1 else ""
        self._footer(screen, "Up/Down take  Enter/P play  A accept" + chunks + "  Esc passage")

    def _draw_confirm_take(self, screen):
        self._header(screen, "Accept rendered take")
        row = next(row for row in self.takes_rows if row["attempt"] == self.confirm_take_attempt)
        screen.write(3, 0, f"Accept take {self.confirm_take_attempt}?", None)
        if row.get("passed") is not True:
            screen.write(4, 0, "It did not pass ordinary verification.", None)
        self._footer(screen, "Y accept  N/Esc cancel")

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
        elif self.view == "takes":
            self._draw_takes(screen)
        elif self.view == "confirm_take":
            self._draw_confirm_take(screen)
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
                self.curses.KEY_LEFT: "LEFT", self.curses.KEY_RIGHT: "RIGHT",
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
        app.handle_key(key, screen=screen)
    return app


def run_review(directory, *, screen=None, player_command=None):
    """Open a completed book in the review UI; never create voices or call models."""
    directory = Path(directory).resolve()
    if screen is None and (not sys.stdin.isatty() or not sys.stdout.isatty() or not os.environ.get("TERM")):
        raise ValueError("review-ui requires an interactive terminal")
    document = _completed_directory(directory)[0]
    player = _audio_player(player_command or ["/usr/bin/afplay"])
    app = ReviewApp(directory, document, revise_directories=revise_directories,
                    related_directories=related_directories, player=player)
    if screen is not None:
        return _drive(app, screen)
    import curses

    def wrapped(stdscr):
        adapter = CursesScreen(curses, stdscr, palette(len(app.roles)))
        return _drive(app, adapter)

    return curses.wrapper(wrapped)
