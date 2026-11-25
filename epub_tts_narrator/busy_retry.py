"""A step of ``analyze`` that finds a book in use waits, retries or is skipped instead of ending the run.

A change to a saved book is refused while a book it rewrites is in use: a
render of it, a casting session, another save (``analysis_revision.BookBusy``).
Nothing has been changed then, and the step's model answers are cached, so
running it again costs no new requests. In a terminal the person chooses:

* ``W`` wait: the step is retried every ``WAIT_SECONDS`` until the book is
  free (stop the render from its own shell, or let it finish); Ctrl-C stops.
* ``R`` retry now (after stopping the render or closing the cast screen).
* ``S`` skip this step for now: the run goes on and a later rerun of
  ``analyze BOOK`` does it.

Without a terminal the refusal ends the run as before; rerunning resumes.
"""

from __future__ import annotations

import sys
import time

WAIT_SECONDS = 30.0
SKIPPED = object()


def interactive():
    return bool(getattr(sys.stdin, "isatty", lambda: False)() and getattr(sys.stdout, "isatty", lambda: False)())


def run_when_free(step, label, *, ask=None, output=print, sleep=None, wait_seconds=None, is_interactive=None):
    """``step()``'s result; ``SKIPPED`` when the person skips it while its book is in use.

    ``label`` names the step in plain words ("cast entries that are one person").
    """
    from .analysis_revision import BookBusy
    sleep = time.sleep if sleep is None else sleep
    wait_seconds = WAIT_SECONDS if wait_seconds is None else wait_seconds
    is_interactive = interactive if is_interactive is None else is_interactive
    waiting = False
    while True:
        try:
            return step()
        except BookBusy as error:
            if waiting:
                sleep(wait_seconds)
                continue
            if not is_interactive():
                raise
            output(f"{label[:1].upper()}{label[1:]}: a book this needs is in use, so nothing was saved yet. {error}")
            answer = _choice(ask)
            if answer == "s":
                output(f"Skipped for now: {label}. Rerunning analyze on the book does it later.")
                return SKIPPED
            if answer == "w":
                output(f"Waiting until the book is free, trying again every {int(wait_seconds)} seconds "
                       "(stop the render, or let it finish; Ctrl-C stops waiting).")
                waiting = True
                sleep(wait_seconds)
            # "r": try again at once


def _choice(ask):
    prompt = "[W] wait until it is free and continue / [R] retry now / [S] skip this step for now: "
    while True:
        try:
            answer = (ask or input)(prompt)
        except EOFError:
            return "s"
        answer = (answer or "").strip().casefold()[:1]
        if answer in {"w", "r", "s"}:
            return answer
