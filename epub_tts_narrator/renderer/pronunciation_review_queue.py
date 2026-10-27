"""Interruptible listening queue flow and terminal presentation.

The queue speaks each queued word's current choice in isolation, twice, with
an interruptible one-second gap after each play. An uninterrupted pair
auto-accepts durably against the hash-bound played receipt; any key falls back
five words (including the current one) so a late interruption can be
corrected. Progress lives in the review-state sidecar owned by
``pronunciation_review_state``; only an acceptance writes this book's
pronunciation registry, and ancestor registries are never touched.
"""

from __future__ import annotations

import re
import shutil
import sys

from ..pronunciation_groups import entry_words, family_words, find_pronunciation
from .audition import display
from .pronunciation_review import _preferred, _pronunciation, _show
from .pronunciation_review_state import ReviewState


_GAP_SECONDS = 1.0
# An interruption returns to the last five words, including the current one.
_JUMP_BACK = 5


def run_pronunciation_queue(session, player, playback_fn, *, input_fn, output=print,
                            interactive=None, redraw=None, gap_seconds=_GAP_SECONDS):
    """Run the count-only menu and the interruptible listening queue.

    ``interactive`` gates the walk: without an interactive terminal a
    completed play cannot be attributed to a listener, so the queue refuses to
    walk (explicit per-word actions remain available). ``redraw`` receives each
    queue line so a terminal can rewrite the current line instead of scrolling.
    """
    from ..pronunciations import set_pronunciation
    from ..provider import ProviderError
    from .diagnostics import configured_secrets, safe_error_text
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if redraw is None:
        def redraw(value):
            if interactive:
                sys.stdout.write(value)
                sys.stdout.flush()
    bundle = session.bundle
    state = ReviewState(session.directory, bundle["source"]["book"]["id"])
    width = shutil.get_terminal_size((80, 24)).columns
    secrets = configured_secrets(session.config)
    last = None

    def say(value):
        output(display(value))

    def draw(value):
        text = display(value)
        redraw("\r" + text[: max(1, width - 1)] + "\x1b[K" + ("\n" if not interactive else ""))

    def stop_draw():
        if interactive:
            redraw("\n")

    def require_interactive():
        if not interactive:
            raise ValueError("The listening queue needs an interactive terminal so auto-acceptance "
                             "means a person heard the pair; use the explicit commands or "
                             "--action play/set instead")

    def play_excerpt(row, proposed):
        description = session.description(row, proposed, prepare=True)
        preview = session.audition(row["word"], proposed)
        say(f"Auditioning {row['word']}: {description}.")
        say("Listening preview — wording and pronunciation are for you to judge; this is not accepted book audio.")
        say(f"{preview['context']['chapter_title']}: {preview['request']['text']}")
        say("Press any key to stop.")
        playback_fn(player, preview["path"])
        session.mark_played(preview)
        return preview

    def decision_words(row):
        """An explicit decision resolves its family, including old sidecar keys.

        Reopening remains restricted to the displayed group; an actual new
        decision can resolve previously conflicting capitalization choices.
        """
        rows = [row, *session.rows()]
        spellings = set(state.entries)
        for item in rows:
            spellings.update(entry_words(item))
        return family_words(row["word"], rows, spellings)

    def autoaccept(row, choice, receipt, index, total):
        """Apply a completed, receipt-verified pair; the receipt binds this exact audio."""
        if choice is None:
            session.ignore(row["word"])
        else:
            set_pronunciation(bundle["directory"], bundle, row["word"], choice, status="confirmed")
            session.clear_drafts(row)
        state.record(row["word"], choice, receipt, accepted=True, variants=decision_words(row))
        draw(f"[{index + 1}/{total}] {row['word']} — accepted ({session.description(row, choice)})")

    def walk(start):
        nonlocal last
        from .terminal import wait_key
        index = start
        previous_notify = session.notify
        try:
            session.notify = lambda message: draw(message)  # keep generation notices on the queue line
            while True:
                rows = session.rows()
                if not rows or index >= len(rows):
                    state.resume_word = None
                    state.save()
                    stop_draw()
                    say("Listening queue complete — every queued word was heard without interruption.")
                    return
                row = rows[index]
                if not state.queued(row):
                    index += 1
                    continue
                word = row["word"]
                state.resume_word = word  # saved before the expensive preview work
                state.save()
                last = word
                choice = session.selection(row)
                description = session.description(row, choice)
                if row.get("case_conflict"):
                    description += " (conflicting capitalization choices)"
                if choice is not None and row["status"] == "pending" and choice == row.get("proposal"):
                    description += " (guessed; not verified)"
                draw(f"[{index + 1}/{len(rows)}] {word} — {description}")
                try:
                    preview = session.audition_isolated(word, choice)
                    if not playback_fn(player, preview["path"]):
                        return interrupted(rows, index)
                    if wait_key(gap_seconds) is not None:
                        return interrupted(rows, index)
                    if not playback_fn(player, preview["path"]):
                        return interrupted(rows, index)
                    if wait_key(gap_seconds) is not None:
                        return interrupted(rows, index)
                    autoaccept(row, choice, session.mark_played(preview), index, len(rows))
                except (ValueError, RuntimeError, OSError, ProviderError, KeyboardInterrupt):
                    stop_draw()
                    raise
                # Accepting a capitalization choice may collapse earlier
                # conflicting rows. Resume after the accepted group in the
                # fresh list, rather than skipping its new next neighbour.
                current = session.rows()
                accepted = find_pronunciation(current, word)
                if accepted is None:
                    raise ValueError("Pronunciation entries changed during playback; reopen the queue to resume")
                index = current.index(accepted) + 1
        finally:
            session.notify = previous_notify

    def interrupted(rows, index):
        first = max(0, index - (_JUMP_BACK - 1))
        state.resume_word = rows[first]["word"]
        state.save()
        nonlocal last
        last = state.resume_word
        stop_draw()
        say(f"Interrupted at word {index + 1}/{len(rows)}: {rows[index]['word']}. Last five words:")
        for position in range(first, index + 1):
            row = rows[position]
            say(f"  {position - first + 1}. [{position + 1}/{len(rows)}] {row['word']} — {row['status']}; "
                f"{session.description(row)}")
            if row.get("case_conflict"):
                say("     Conflicting capitalization choices remain separate until you choose a pronunciation.")
            if row.get("preparation_error"):
                say("     Audio unavailable: " + row["preparation_error"])
        say(f"Enter resumes the queue at [{first + 1}/{len(rows)}] {rows[first]['word']}; "
            "E/P/B/C/I/R [N] adjusts a word by its list number (L); Q quits.")

    def reopen(spec):
        rows = session.rows()
        if spec is None:
            raise ValueError("Use O all, O N, or O N-M (entry numbers from the list)")
        if spec.casefold() == "all":
            chosen = list(rows)
        else:
            match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", spec.strip())
            if match is None:
                raise ValueError("Use O all, O N, or O N-M (entry numbers from the list)")
            first, final = int(match.group(1)), int(match.group(2) or match.group(1))
            if not 1 <= first <= final <= len(rows):
                raise ValueError(f"Reopen range must stay within 1..{len(rows)}")
            chosen = rows[first - 1:final]
        for row in chosen:
            state.mark_reopened(row["word"], _preferred(row), variants=entry_words(row))
        say(f"Marked {len(chosen)} entr{'y' if len(chosen) == 1 else 'ies'} for re-listening; "
            "registry decisions are unchanged, including inherited ones.")

    def compact_list():
        rows = session.rows()
        if len(rows) <= 10:
            _show(rows, output, formatter=session.description)
            return
        items = []
        for index, row in enumerate(rows, 1):
            marker = "*" if state.queued(row) else " "
            if row.get("preparation_error"):
                marker += "!"
            if row.get("case_conflict"):
                marker += "?"
            items.append(display(f"{index:>3}{marker} {row['word']} — {row['status']}"))
        cell = max(2, min(width, max(len(item) for item in items) + 2))
        columns = max(1, width // cell)
        for offset in range(0, len(items), columns):
            say("".join(item[: cell - 2].ljust(cell) for item in items[offset:offset + columns]).rstrip()[:width])
        if any(row.get("preparation_error") for row in rows):
            say("! Audio unavailable; choose the entry number to retry its preview.")
        if any(row.get("case_conflict") for row in rows):
            say("? Conflicting capitalization choices remain separate until you choose a pronunciation.")

    def queued_count():
        return sum(1 for row in session.rows() if state.queued(row))

    def resume_index(rows):
        if state.resume_word is not None:
            selected = find_pronunciation(rows, state.resume_word)
            if selected is not None:
                return rows.index(selected)
        return next((index for index, row in enumerate(rows) if state.queued(row)), None)

    def help_text():
        rows = session.rows()
        if not rows:
            say("No pronunciation entries. Use pronunciations --action discover to find candidates.")
            return False
        say(f"{queued_count()} of {len(rows)} pronunciation entries await listening.")
        unavailable = sum(bool(row.get("preparation_error")) for row in rows)
        if unavailable:
            say(f"{unavailable} audio preview(s) unavailable; edits remain available, and numbered playback retries.")
        say("S [N]: listen and auto-accept, starting at entry N (default: saved position or first queued word)")
        say("Number: play that entry's excerpt preview   L: compact list   O <all|N|N-M>: mark for re-listening")
        say("B/E/P/C/I/R [N]: baseline / readable hint / CMU phonemes / confirm / keep default / reset   Q: quit")
        say("During playback or the gap after it, any key interrupts and returns to the last five words.")
        return True

    def start_walk(rows, parts):
        require_interactive()
        if len(parts) > 1:
            if not parts[1].isdecimal():
                raise ValueError("Use S, or S N with an entry number from the list (L)")
            start = int(parts[1]) - 1
            if not 0 <= start < len(rows):
                raise ValueError(f"Start at an entry from 1 to {len(rows)}")
        else:
            start = resume_index(rows)
            if start is None:
                say("Nothing is awaiting listening.")
                return
        walk(start)

    if not help_text():
        return 0
    while True:
        try:
            command = input_fn("> ").strip()
        except EOFError:
            state.save()
            say("Pronunciation review closed; decisions, drafts, and queue position are saved.")
            return 0
        if not command:
            if state.resume_word is not None:
                try:
                    start_walk(session.rows(), ["s"])
                except (ValueError, RuntimeError, OSError, ProviderError) as exc:
                    say("Action stopped: " + safe_error_text(exc, secrets=secrets))
            else:
                help_text()
            continue
        if command.casefold() == "q":
            state.save()
            return 0
        try:
            parts = command.split(maxsplit=2)
            verb = "play" if parts[0].isdecimal() else parts[0].casefold()
            if verb == "s":
                start_walk(session.rows(), parts)
                continue
            if verb == "l":
                compact_list()
                continue
            if verb == "o":
                reopen(parts[1] if len(parts) > 1 else None)
                continue
            rows = session.rows()
            selection = parts[0] if verb == "play" else (parts[1] if len(parts) > 1 else None)
            if selection is None:
                selected_word = last or state.resume_word
                row = find_pronunciation(rows, selected_word) if selected_word else None
                if row is None:
                    raise ValueError("Enter a number from the pronunciation list (L), or S to start the queue")
            elif not selection.isdecimal() or not 1 <= int(selection) <= len(rows):
                raise ValueError("Enter a number from the pronunciation list (L), or S to start the queue")
            else:
                row = rows[int(selection) - 1]
            word = last = row["word"]
            if verb == "play":
                play_excerpt(row, session.selection(row))
            elif verb == "b":
                play_excerpt(row, None)
            elif verb in {"e", "p"}:
                say(f"Current pronunciation for {word}: {session.description(row, prepare=True)}")
                prompt = ("Say it as (plain spelling; empty cancels): " if verb == "e" else
                          "CMU phonemes (for example M AA1 R AH0; empty cancels): ")
                value = parts[2] if len(parts) > 2 else input_fn(prompt).strip()
                if not value:
                    continue
                if verb == "e":
                    proposed = session.propose(word, value)
                else:
                    proposed = _pronunciation(phonemes=value)
                    session.edit(word, proposed)
                say("Draft saved for audition; C confirms it for future rendering.")
                play_excerpt(row, proposed)
            elif verb == "c":
                session.confirm(row)
                state.clear(decision_words(row))
                say("Pronunciation confirmed. Existing audio is unchanged.")
            elif verb == "i":
                session.ignore(word)
                state.clear(decision_words(row))
                say("Default pronunciation selected.")
                try:
                    say(session.description(row, None, prepare=True))
                except (ValueError, RuntimeError, OSError, ProviderError) as exc:
                    say("CMU estimate unavailable; B retries it: " + safe_error_text(exc, secrets=secrets))
            elif verb == "r":
                session.reset(word)
                state.clear(decision_words(row))
                say("Local decision and audition draft removed; inherited decisions apply.")
            else:
                raise ValueError("Use S, L, O, a number, B, E, P, C, I, R, or Q")
        except EOFError:
            state.save()
            say("Pronunciation review closed; decisions, drafts, and queue position are saved.")
            return 0
        except (ValueError, RuntimeError, OSError, ProviderError) as exc:
            say("Action stopped: " + safe_error_text(exc, secrets=secrets))
