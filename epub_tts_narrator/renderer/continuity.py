"""Mandatory chapter history: Fish's pinned first turn plus the latest accepted turn.

Every chapter carries Drama 3's generated performance context across its
requests. The fixed voice references say who is speaking; the history says
where the performance is. Following Fish's own long-text policy
(``fish_omni/chunking.py``), the first accepted single-voice narration take of
the chapter stays pinned and the latest accepted take follows it. Mixed-voice
paragraphs take part with their speaker tags renumbered per request; every
history voice is sent as a reference. When a prompt overflows, candidates drop
the latest turn, then the pinned one, exactly as Fish does.

Only the sequential chapter coordinator advances history. Inference receives
an immutable snapshot through a context variable, so recovery callbacks,
pool lanes and HTTP workers see the same context without extra plumbing.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
import re

from .common import digest, file_hash, regular_file


_CURRENT = ContextVar("accepted_chapter_continuity", default=None)
POLICY = "fish-pinned-first-narration-latest"
MAX_SPEAKERS = 16
SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
# Joins, repairs and best-effort choices remain accepted audio but are not one
# generation of the exact logical text, so they never become history.
_NOT_HISTORY = {"recovery", "clip_repair", "speech_resolution", "fragment_recovery",
                "delivery_recovery", "direction_repair", "exhaustion_selection"}


def current():
    return _CURRENT.get()


def require_support(engine):
    """Every renderer engine must carry chapter history; there is no fallback."""
    if getattr(engine, "capabilities", {}).get("chapter_continuity") is not True:
        raise ValueError("The configured speech engine does not support chapter continuity; "
                         "every renderer engine must accept history turns and return generated codes")
    check = getattr(engine, "check_continuity", None)
    if callable(check):
        check()


def _segments(chunk):
    if "paragraph_turns" in chunk:
        return [turn["segment"] for turn in chunk["paragraph_turns"]]
    return [chunk["segment"]]


def narrator_asset(chapter, voices):
    """The voice speaking the most narration in this chapter, or None."""
    totals = {}
    for chunk in chapter["chunks"]:
        if chunk.get("silent_control"):
            continue
        for segment in _segments(chunk):
            voice = voices.get(segment.get("voice_id"))
            if voice is not None and segment.get("kind") == "narration":
                asset = voice["asset_hash"]
                totals[asset] = totals.get(asset, 0) + len(segment.get("text", ""))
    return max(sorted(totals), key=totals.get) if totals else None


def is_narration(chunk, voices, narrator):
    """Single-voice narration by the chapter narrator, suitable for the pinned turn."""
    if narrator is None or chunk.get("silent_control"):
        return False
    segments = _segments(chunk)
    return all(segment.get("kind") == "narration" and "simultaneous_speakers" not in segment
               and voices.get(segment.get("voice_id"), {}).get("asset_hash") == narrator
               for segment in segments)


class ChapterContinuity:
    """Chapter-owned pinned/latest accepted takes and pool worker affinity."""

    def __init__(self, engine_identity, voices, narrator=None):
        # Engine compatibility (transport, runtime updates, model aliases) is
        # the speech cache's decision through the request's own engine
        # identity; history identity names only the exact accepted takes.
        self.voices_by_asset = {voice["asset_hash"]: voice for voice in voices.values()}
        self.narrator = narrator
        self.pinned = None
        self.latest = None
        # A pool keeps a chapter on the worker that produced its first take.
        self.worker = None

    def takes(self):
        return [entry for entry in (self.pinned, self.latest) if entry is not None]

    @contextmanager
    def transaction(self, request, *, narration=False):
        takes = deepcopy(self.takes())
        identity = {"version": 2, "policy": POLICY, "takes": [take["identity"] for take in takes]}
        bound = dict(request, continuity=identity)
        scope = {"chapter": self, "identity": identity, "takes": takes,
                 "request_binding": digest(bound), "narration": narration,
                 # Simultaneous lanes overlap in time; none of them is the
                 # next turn of the chapter conversation.
                 "history": "performer" not in request}
        token = _CURRENT.set(scope)
        try:
            yield bound
        finally:
            _CURRENT.reset(token)

    def bind(self, identity_hash):
        """Record the final cache identity of this transaction's request."""
        current()["request_binding"] = identity_hash

    def accept(self, path, accepted):
        from .continuity_codes import load_codes
        scope = current()
        metadata = accepted.get("engine_metadata", {})
        if not scope["history"] or _NOT_HISTORY.intersection(metadata):
            return
        marker = metadata.get("continuity_codes")
        if marker is None:
            raise ValueError("Accepted take has no generated codec evidence for chapter continuity")
        if (accepted.get("report", {}).get("passed") is not True
                or file_hash(regular_file(path)) != accepted.get("audio_sha256")):
            raise ValueError("Invalid continuity accepted audio")
        # The speech cache already proved this accepted take is the current
        # request (possibly through a compatible transport or alias); its
        # codes must have been generated after exactly the current history.
        if (marker.get("context") != scope["identity"]
                or marker.get("accepted_audio_sha256") != accepted["audio_sha256"]):
            raise ValueError("Continuity generation does not match the accepted request")
        unknown = [asset for asset in marker["speakers"] if asset not in self.voices_by_asset]
        if unknown:
            raise ValueError("Continuity history names a voice outside this chapter")
        codes_path = Path(path).parent / marker["file"]
        load_codes(codes_path, marker)
        entry = {"path": str(codes_path), "marker": deepcopy(marker), "identity": {
            "codes_sha256": marker["sha256"], "text_sha256": digest(marker["text"]),
            "speakers": list(marker["speakers"]), "request_hash": accepted["request_hash"],
            "audio_sha256": accepted["audio_sha256"]}}
        if self.pinned is None and scope["narration"]:
            # The pinned take is also the newest; Fish starts its recent
            # window only after the pinned chunk.
            self.pinned, self.latest = entry, None
        else:
            self.latest = entry


def _retag(text, mapping):
    def replace(match):
        index = int(match.group(1))
        if index not in mapping:
            raise ValueError("Continuity history text uses an undeclared speaker tag")
        return f"<|speaker:{mapping[index]}|>"
    return SPEAKER_TAG.sub(replace, text)


def compose(text, voices):
    """Wire request for tagged ``text`` spoken by ``voices`` (index = speaker id).

    Returns ``(references, history)``: reference voices carrying explicit
    ``speaker`` ids (the request's own voices first, then voices heard only in
    history) and ``(tagged_text, codes)`` history turns in chronological order.
    Outside a chapter transaction, such as casting previews, history is empty.
    """
    from .continuity_codes import load_codes
    used = {int(index) for index in SPEAKER_TAG.findall(text)}
    if not used or max(used) >= len(voices):
        raise ValueError("Every request speaker tag must name one of its voices")
    order = [voice.get("asset_hash") for voice in voices]
    references = [dict(voice, speaker=index) for index, voice in enumerate(voices)]
    scope = current()
    if scope is None:
        return references, []
    chapter = scope["chapter"]
    history = []
    for take in scope["takes"]:
        marker = take["marker"]
        mapping = {}
        for index, asset in enumerate(marker["speakers"]):
            if asset not in order:
                order.append(asset)
                references.append(dict(chapter.voices_by_asset[asset], speaker=len(order) - 1))
            mapping[index] = order.index(asset)
        payload = load_codes(take["path"], marker)
        history.append((_retag(marker["text"], mapping), payload["codes"]))
    return references, history


def candidates(text, references, history):
    """Fish's fallback order: all history, drop the latest, then the pinned turn.

    Each candidate keeps only the references whose speaker tag appears in its
    text, so a dropped turn also drops the voices only it needed. Candidates
    over Drama 3's 16 speaker tokens are skipped; the empty history always fits.
    """
    turns = list(history)
    options = [turns[:size] for size in range(len(turns), -1, -1)]
    result = []
    for kept in options:
        tags = {int(index) for item in [text, *[turn[0] for turn in kept]]
                for index in SPEAKER_TAG.findall(item)}
        selected = [reference for reference in references if reference["speaker"] in tags]
        if len(selected) <= MAX_SPEAKERS:
            result.append((selected, kept))
    return result


def generated_codes(codes, codebook_size, text, voices):
    """Engine metadata for exactly what was generated; the workflow persists it."""
    return {"generated_codes": {"codes": codes, "codebook_size": codebook_size, "text": text,
                                "speakers": [voice.get("asset_hash") for voice in voices]}}
