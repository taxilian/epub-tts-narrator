"""Bounded content-addressed reuse of validated reference audio for the serve endpoint."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading

MAX_REFERENCE_CACHE_ENTRIES = 64
MAX_REFERENCE_CACHE_BYTES = 256 * 1024 * 1024
# Stable wire marker: the client retries inline audio when it matches a 400.
UNKNOWN_REFERENCE_MARKER = "reference_id is not registered"
UNKNOWN_REFERENCE_MESSAGE = UNKNOWN_REFERENCE_MARKER + "; send this reference inline again"


class ReferenceCache:
    """Thread-safe bounded LRU mapping SHA-256 content ids to immutable audio.

    Only audio that already passed server-side WAV inspection enters the
    cache, so a hit never revalidates bytes. The cache lives in memory: a
    restart loses every id, and clients resend inline audio when an id is
    unknown.
    """

    def __init__(self, *, max_entries: int = MAX_REFERENCE_CACHE_ENTRIES,
                 max_bytes: int = MAX_REFERENCE_CACHE_BYTES):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Reference cache must allow at least one entry")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Reference cache must allow at least one byte")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._stored_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def content_id(audio: bytes) -> str:
        return hashlib.sha256(audio).hexdigest()

    def store(self, audio: bytes, content_id: str | None = None) -> str:
        """Keep validated audio for reuse; a claimed id must match the content."""
        if content_id is None:
            content_id = self.content_id(audio)
        elif content_id != self.content_id(audio):
            raise ValueError("Reference cache id does not match its audio content")
        with self._lock:
            displaced = self._entries.pop(content_id, None)
            if displaced is not None:
                self._stored_bytes -= len(displaced)
            if len(audio) <= self._max_bytes:
                self._entries[content_id] = audio
                self._stored_bytes += len(audio)
                while len(self._entries) > 1 and self._stored_bytes > self._max_bytes:
                    _, evicted = self._entries.popitem(last=False)
                    self._stored_bytes -= len(evicted)
                while len(self._entries) > self._max_entries:
                    _, evicted = self._entries.popitem(last=False)
                    self._stored_bytes -= len(evicted)
        return content_id

    def load(self, content_id: str) -> bytes | None:
        with self._lock:
            audio = self._entries.get(content_id)
            if audio is not None:
                self._entries.move_to_end(content_id)
            return audio
