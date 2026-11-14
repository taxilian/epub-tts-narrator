"""A person's change to one failed passage's delivery direction, saved to the book.

The run shell offers this for a passage no take could pass. The change is a
manual attribution edit (``attribution_edits.edit_direction``, ``delivery.basis
= "user"``) published through ``analysis_revision.revise_directories`` scope
``attribution``, exactly as ``review-ui`` saves one: backups, receipts and
descendant rebinding included, and never refused for a live render's lease.

A running render keeps rendering from the script it loaded, so the edit is
used the next time the book renders (the run shell's ``R`` after the run, or
the next ``render``); the changed passage is a new request and gets new takes.

The spoken words are not editable here: they are the book's own text, and a
change to them is a reviewed source substitution, never a free edit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


DIRECTION_LIMIT = 240


def check_direction(text):
    """The trimmed direction, or ``ValueError`` naming the limit."""
    value = str(text or "").strip()
    if len(value) > DIRECTION_LIMIT:
        raise ValueError(f"A direction is at most {DIRECTION_LIMIT} characters (this one has {len(value)})")
    return value


def save_direction(book, segment_ids, direction, *, progress=None):
    """Set ``direction`` on each of ``segment_ids`` in the book at ``book``; returns the revision result.

    Every segment must exist; the book's manifest (and related books') must
    be unchanged since they were read, as for any attribution revision.
    """
    from ..analysis_directory import _completed_directory
    from ..analysis_revision import related_directories, revise_directories
    from ..attribution_edits import apply_edits, edit_direction
    book = Path(book).resolve()
    value = check_direction(direction)
    wanted = [segment_id for segment_id in segment_ids if isinstance(segment_id, str)]
    if not wanted:
        raise ValueError("This failure names no passage of the book, so there is nothing to edit")
    directories = [Path(path).resolve() for path in related_directories([book])]
    hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
              for path in directories}
    document = _completed_directory(book)[0]
    segments = {segment["id"]: segment for segment in document["segments"]}
    missing = [segment_id for segment_id in wanted if segment_id not in segments]
    if missing:
        raise ValueError("The book no longer has this passage (it was revised since the render started): "
                         + ", ".join(missing))
    edits = {segment_id: edit_direction(segments[segment_id], value) for segment_id in wanted}
    messages = []
    result = revise_directories({book: apply_edits(document, edits)}, directories=directories,
                                expected_manifest_hashes=hashes, scope="attribution",
                                progress=progress or messages.append)
    return result
