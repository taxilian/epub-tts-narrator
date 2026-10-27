"""Book-local listening-queue sidecar; acceptance is bound to the accepted choice.

The sidecar stores per-word entries with the accepted choice, the hash-bound
played receipt, and a reopen marker. The pronunciation registry is the only
synthesis policy; reopening marks entries in this sidecar instead of touching
the registry, so confirmed/ignored decisions (local or inherited) stay
byte-identical until the listener actually accepts a pair.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from ..pronunciation_groups import entry_words
from .common import digest, read_json, regular_file, write_json
from .pronunciation_review import _preferred
from .run_locks import acquire_directory_lock


STATE_FORMAT = "epub-tts-pronunciation-review"
_RECEIPT_KEYS = ("request_sha256", "audio_sha256")


def _validate_choice(choice):
    if choice is None:
        return None
    from ..pronunciations import validate_pronunciation
    return validate_pronunciation(choice)


def _validate_receipt(receipt):
    if receipt is None:
        return None
    if (not isinstance(receipt, dict) or set(receipt) != set(_RECEIPT_KEYS)
            or any(not isinstance(receipt[key], str) or re.fullmatch(r"[a-f0-9]{64}", receipt[key]) is None
                   for key in _RECEIPT_KEYS)):
        raise ValueError("Pronunciation review playback receipt is invalid")
    return dict(receipt)


class ReviewState:
    """Book-local queue sidecar; accepted flags are bound to the accepted choice."""

    def __init__(self, directory, book_id):
        path = Path(directory)
        if path.is_symlink():
            raise ValueError("Pronunciation review state directory must not be a symlink")
        self.path = path / "review-state.json"
        if self.path.is_symlink():
            raise ValueError("Pronunciation review state must not be a symlink")
        self.book_id = book_id
        self.entries = {}
        self.resume_word = None
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        from ..pronunciations import validate_word
        value = read_json(regular_file(self.path))
        if (not isinstance(value, dict) or value.get("format") != STATE_FORMAT or value.get("version") != 1
                or value.get("book_id") != self.book_id or not isinstance(value.get("entries"), dict)):
            raise ValueError("Pronunciation review state is invalid or belongs to a different book")
        body = {"entries": value["entries"], "resume_word": value.get("resume_word")}
        if value.get("sha256") != digest(body):
            raise ValueError("Pronunciation review state changed; remove it to discard queue progress")
        entries = {}
        for word, entry in value["entries"].items():
            validate_word(word)
            if (not isinstance(entry, dict) or set(entry) - {"choice", "receipt", "accepted", "reopened"}
                    or not entry or not isinstance(entry.get("accepted"), bool)
                    or not isinstance(entry.get("reopened", False), bool)):
                raise ValueError("Pronunciation review state entry is invalid")
            entries[word] = {"choice": _validate_choice(entry.get("choice")),
                             "receipt": _validate_receipt(entry.get("receipt")),
                             "accepted": entry["accepted"],
                             "reopened": entry.get("reopened", False)}
        self.entries = entries
        resume = value.get("resume_word")
        if resume is not None:
            validate_word(resume)
            self.resume_word = resume

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = acquire_directory_lock(self.path.parent / ".review-state.lock")
        try:
            body = {"format": STATE_FORMAT, "version": 1, "book_id": self.book_id,
                    "entries": self.entries, "resume_word": self.resume_word}
            write_json(self.path, dict(body, sha256=digest({key: body[key] for key in ("entries", "resume_word")})))
        finally:
            lock.rmdir()

    def accepted(self, row):
        """Reuse a group member's acceptance only for the same selected choice.

        Receipts retain their original request/audio hashes; grouping does not
        manufacture a played receipt for the displayed spelling's preview.
        """
        choice = _preferred(row)
        return next((self.entries[word] for word in entry_words(row)
                     if word in self.entries and self.entries[word]["accepted"]
                     and self.entries[word]["choice"] == choice), None)

    def queued(self, row):
        """Whether the walk should (re-)listen to this row.

        Registry-pending rows queue until accepted for their current choice;
        confirmed/ignored rows queue only when the effective choice changed
        (for example a new draft on a confirmed word); reopened rows queue
        regardless of status, including inherited decisions.
        """
        if any(self.entries.get(word, {}).get("reopened") for word in entry_words(row)):
            return True
        choice = _preferred(row)
        if row["status"] == "pending":
            return self.accepted(row) is None
        return choice is not None and choice != row.get("pronunciation")

    def record(self, word, choice, receipt, *, accepted, variants=()):
        entry = {"choice": deepcopy(choice), "receipt": _validate_receipt(receipt),
                 "accepted": bool(accepted), "reopened": False}
        for member in (word, *variants):
            self.entries.pop(member, None)
        self.entries[word] = entry
        self.save()

    def mark_reopened(self, word, choice, *, variants=()):
        """Queue a row for re-listening without changing the pronunciation registry."""
        prior = self.entries.get(word, {})
        for member in (word, *variants):
            self.entries.pop(member, None)
        self.entries[word] = {"choice": deepcopy(choice),
                              "receipt": prior.get("receipt") if prior.get("choice") == choice else None,
                              "accepted": False, "reopened": True}
        self.save()

    def clear(self, words):
        for word in words:
            self.entries.pop(word, None)
        self.save()
