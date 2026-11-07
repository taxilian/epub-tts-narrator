"""Mandatory same-voice chapter history: Fish's pinned first turn plus the latest.

Every chapter carries Drama 3's generated performance context across its
requests. The fixed voice references say who is speaking; the history says
where that voice's performance is. Following Fish's own long-text policy
(``fish_omni/chunking.py``), a request hears the first accepted take (pinned)
and the latest accepted take of the same voice in this chapter. Only takes
whose speakers are all among the request's own speakers are eligible, so no
other character's audio sits in the prompt. Speakers are the script's voice
ids, not recordings: a narrator linked to a character's voice clones the same
reference but keeps its own history, so narration and that character's
dialogue stay distinct performances: Drama 3 documents history for
continuing one voice's script or revising a take, and another voice's recent
audio competes with the reference for identity and accent. Chapter headings
never become the pinned take. When a prompt overflows, candidates drop the
latest turn, then the pinned one, exactly as Fish does.

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
# A history-echo retry's ladder rung (see history_echo); None is ordinary history.
_RECOVERY = ContextVar("history_echo_recovery", default=None)
POLICY = "fish-first-latest-same-voice"
MAX_SPEAKERS = 16
SPEAKER_TAG = re.compile(r"<\|speaker:(\d+)\|>")
# Joins, repairs and best-effort choices remain accepted audio but are not one
# generation of the exact logical text, so they never become history.
_NOT_HISTORY = {"recovery", "clip_repair", "speech_resolution", "fragment_recovery",
                "delivery_recovery", "direction_repair", "exhaustion_selection"}


def current():
    return _CURRENT.get()


def recovery_mode():
    """The requested history-echo ladder rung for generation in this context."""
    return _RECOVERY.get()


@contextmanager
def history_recovery(mode):
    """Generate inside ``mode``'s rung; engines record what they actually applied."""
    token = _RECOVERY.set(mode)
    try:
        yield
    finally:
        _RECOVERY.reset(token)


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


def is_pinnable(chunk):
    """A take may become its voice's pinned first turn unless it is only headings."""
    return not all(segment.get("kind") == "heading" for segment in _segments(chunk))


def request_speakers(request):
    """The script voice ids speaking this request (a simultaneous lane: its performer)."""
    if "performer" in request:
        return frozenset({request["performer"]["voice_id"]})
    from .bundle import chunk_speakers
    return frozenset(speaker["voice_id"] for speaker in chunk_speakers(request["chunk"]))


class ChapterContinuity:
    """Chapter-owned accepted takes for per-request history composition."""

    def __init__(self, engine_identity, voices):
        # Engine compatibility (transport, runtime updates, model aliases) is
        # the speech cache's decision through the request's own engine
        # identity; history identity names only the exact accepted takes.
        self.voices_by_asset = {voice["asset_hash"]: voice for voice in voices.values()}
        self.entries = []

    def takes(self, speakers):
        """Fish's first + latest, from this chapter's takes by these voice ids only."""
        eligible = [entry for entry in self.entries if set(entry["voices"]) <= speakers]
        if not eligible:
            return []
        pinned = next((entry for entry in eligible if entry["pinnable"]), None)
        latest = eligible[-1]
        return [latest] if pinned is None or pinned is latest else [pinned, latest]

    @contextmanager
    def transaction(self, request, *, pinnable=True):
        voices = request_speakers(request)
        takes = deepcopy(self.takes(voices))
        identity = {"version": 3, "policy": POLICY, "takes": [take["identity"] for take in takes]}
        bound = dict(request, continuity=identity)
        segments = _segments(request["chunk"]) if isinstance(request.get("chunk"), dict) else []
        scope = {"chapter": self, "identity": identity, "takes": takes, "voices": sorted(voices),
                 "request_binding": digest(bound), "pinnable": pinnable,
                 # The line's role, for a history-echo retry's role tag.
                 "kind": segments[0].get("kind") if segments else None,
                 # Simultaneous lanes overlap in time; none of them is the
                 # next turn of the chapter.
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
        # Selection uses the script voices; the take identity (and so every
        # request's cache identity) still names only the exact accepted take.
        self.entries.append({"path": str(codes_path), "marker": deepcopy(marker), "pinnable": scope["pinnable"],
                             "voices": list(scope["voices"]), "identity": {
            "codes_sha256": marker["sha256"], "text_sha256": digest(marker["text"]),
            "speakers": list(marker["speakers"]), "request_hash": accepted["request_hash"],
            "audio_sha256": accepted["audio_sha256"]}})


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
