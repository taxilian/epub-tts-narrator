"""Source-selected, frame-preserving muting of accepted speech.

This module never changes speech acceptance, source text, or saved reference audio.
Its timestamps must describe the original clip, before assembly edge trimming.
"""

from array import array
from collections.abc import Mapping
import hashlib
import json
import math
import re
import sys


VERSION = 1
DEFAULT_POLICY = {
    "enabled": False,
    "words": ["fuck", "fucks", "fucked", "fucking", "fucker", "fuckers", "fuckhead", "fuckheads", "fuckwit", "fuckwits", "motherfucker", "motherfuckers", "motherfucking", "shit", "shits", "shitting", "shitty", "bullshit", "horseshit", "batshit", "apeshit", "chickenshit", "shithead", "shitheads", "shitshow", "shitshows", "shitload", "shitloads"],
    "preserve_ms": 45,
    "max_preserved_fraction": 0.2,
    "fade_ms": 5,
    "padding_ms": 15,
}
_TOKEN = re.compile(r"[^\W_]+(?:['’\-‐‑][^\W_]+)*", re.UNICODE)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CensorshipAlignmentRequired(ValueError):
    """Accepted speech needs alignment; retrying synthesis is not the remedy."""

    def __init__(self, reason, targets=None):
        self.reason = reason
        self.matches = targets or []
        super().__init__("Censorship requires reliable word alignment: " + reason)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normal(text):
    return "".join(character for character in text.casefold() if character.isalnum())


def validate_policy(policy=None):
    """Return a canonical policy; configuration values never enter speech identity."""
    if policy is None:
        policy = {}
    if not isinstance(policy, Mapping):
        raise ValueError("censorship must be a table")
    if set(policy) - (set(DEFAULT_POLICY) | {"version"}):
        raise ValueError("Unknown configuration key in censorship")
    if "version" in policy and (type(policy["version"]) is not int or policy["version"] != VERSION):
        raise ValueError("Unsupported censorship policy version")
    result = dict(DEFAULT_POLICY, **policy)
    result.pop("version", None)
    if type(result["enabled"]) is not bool:
        raise ValueError("censorship.enabled must be a boolean")
    words = result["words"]
    if not isinstance(words, list) or len(words) > 1000:
        raise ValueError("censorship.words must be an array of at most 1000 explicit words")
    if any(not isinstance(word, str) or not 1 <= len(word) <= 100 or _TOKEN.fullmatch(word) is None for word in words):
        raise ValueError("censorship.words must contain nonempty whole words, optionally joined by apostrophes or hyphens")
    result["words"] = sorted({_normal(word) for word in words})
    for field, maximum in (("preserve_ms", 250), ("fade_ms", 20), ("padding_ms", 100)):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"censorship.{field} must be between 0 and {maximum}")
    value = result["max_preserved_fraction"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 0.5:
        raise ValueError("censorship.max_preserved_fraction must be between 0 and 0.5")
    return result


def policy_identity(policy=None):
    policy = validate_policy(policy)
    return dict(version=VERSION, **policy) if policy["enabled"] else None


def _tokens(text):
    return [{"text": match.group(), "start": match.start(), "end": match.end(), "normalized": _normal(match.group())} for match in _TOKEN.finditer(text)]


def matches(text, policy=None):
    """Match explicit complete lexical tokens, never an innocent substring."""
    if not isinstance(text, str):
        raise ValueError("Censorship source text must be a string")
    policy = validate_policy(policy)
    if not policy["enabled"]:
        return []
    words = set(policy["words"])
    return [token for token in _tokens(text) if token["normalized"] in words]


def _targets(text, policy, targets):
    if targets is None:
        return matches(text, policy)
    if not isinstance(targets, list):
        raise ValueError("Censorship targets must be an array")
    source_tokens = _tokens(text)
    starts, ends = {token["start"] for token in source_tokens}, {token["end"] for token in source_tokens}
    result, previous_end = [], 0
    for target in targets:
        if not isinstance(target, dict) or set(target) != {"text", "start", "end", "normalized"}:
            raise ValueError("Censorship target must contain exact text, start, end, and normalized source term")
        start, end = target["start"], target["end"]
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(text) or start not in starts or end not in ends:
            raise ValueError("Censorship target ranges must be ordered complete spoken tokens")
        if target["text"] != text[start:end] or target["normalized"] not in policy["words"]:
            raise ValueError("Censorship target must bind exact spoken text and a configured source term")
        previous_end = end
        result.append(dict(target))
    return result


def _stream(tokens):
    text, starts, ends = "", {}, {}
    for index, token in enumerate(tokens):
        starts[len(text)] = index
        text += token["normalized"]
        ends[len(text)] = index + 1
    return text, starts, ends


def _occurrences(haystack, needle, starts, ends):
    found, offset = [], 0
    while needle:
        index = haystack.find(needle, offset)
        if index < 0:
            break
        end = index + len(needle)
        if index in starts and end in ends:
            found.append((index, end))
        offset = index + 1
    return found


def _timed_words(report, sample_rate, frames, targets):
    if not isinstance(report, dict) or report.get("word_timing_status") == "unavailable" or not isinstance(report.get("words"), list) or not report["words"]:
        raise CensorshipAlignmentRequired("word timestamps are unavailable", targets)
    words, previous = [], 0.0
    for word in report["words"]:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            raise CensorshipAlignmentRequired("word timestamps have an invalid structure", targets)
        start, end = word.get("start"), word.get("end")
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in (start, end)) or not 0 <= start < end <= (frames + 1) / sample_rate or start < previous:
            raise CensorshipAlignmentRequired("word timestamps overlap or leave the audio", targets)
        previous = end
        normalized = "".join(token["normalized"] for token in _tokens(word["word"]))
        if not normalized:
            continue
        probability = word.get("probability")
        if probability is not None and (isinstance(probability, bool) or not isinstance(probability, (float, int)) or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise CensorshipAlignmentRequired("word confidence is invalid", targets)
        words.append(dict(word, normalized=normalized))
    if not words:
        raise CensorshipAlignmentRequired("word timestamps contain no lexical words", targets)
    return words


def _locate(source_tokens, words, target, targets):
    source, source_starts, source_ends = _stream(source_tokens)
    heard, heard_starts, heard_ends = _stream(words)
    first = next(index for index, token in enumerate(source_tokens) if token["start"] == target["start"])
    last = next(index + 1 for index, token in enumerate(source_tokens) if token["end"] == target["end"])
    offset = sum(len(token["normalized"]) for token in source_tokens[:first])
    length = sum(len(token["normalized"]) for token in source_tokens[first:last])
    if source == heard:
        if offset not in heard_starts or offset + length not in heard_ends:
            raise CensorshipAlignmentRequired("target boundaries fall inside an untimed joined word", targets)
        return heard_starts[offset], heard_ends[offset + length], "exact_transcript"

    # A uniquely matching phrase anchors the target without assuming that ASR
    # word N corresponds to source word N. Word splits/joins outside the target
    # are harmless; ambiguous repeated phrases require realignment.
    candidates = set()
    for left in range(min(first, 3) + 1):
        for right in range(min(len(source_tokens) - last, 3) + 1):
            if left + right < min(2, len(source_tokens) - (last - first)) or not left + right:
                continue
            before = "".join(token["normalized"] for token in source_tokens[first - left:first])
            after = "".join(token["normalized"] for token in source_tokens[last:last + right])
            phrase = before + source[offset:offset + length] + after
            source_hits = _occurrences(source, phrase, source_starts, source_ends)
            heard_hits = _occurrences(heard, phrase, heard_starts, heard_ends)
            if len(source_hits) != 1 or len(heard_hits) != 1:
                continue
            start = heard_hits[0][0] + len(before)
            end = start + length
            if start in heard_starts and end in heard_ends:
                candidates.add((heard_starts[start], heard_ends[end]))
    if len(candidates) != 1:
        raise CensorshipAlignmentRequired("target has no unique exact surrounding-word anchor", targets)
    first, last = candidates.pop()
    return first, last, "unique_context"


def _audio_identity(sample_rate, frames, audio_sha256):
    if type(sample_rate) is not int or sample_rate <= 0 or type(frames) is not int or frames < 0 or not isinstance(audio_sha256, str) or not _SHA256.fullmatch(audio_sha256):
        raise ValueError("Censorship needs a valid sample rate, frame count, and SHA-256 of the original audio")


def plan_masks(text, report, policy=None, *, sample_rate, frames, audio_sha256, targets=None):
    """Plan exact frame intervals, or request alignment without changing speech."""
    if not isinstance(text, str):
        raise ValueError("Censorship source text must be a string")
    policy = validate_policy(policy)
    _audio_identity(sample_rate, frames, audio_sha256)
    selected = _targets(text, policy, targets) if policy["enabled"] else []
    audit = {"version": VERSION, "status": "complete", "text_sha256": _text_hash(text), "audio_sha256": audio_sha256, "policy": policy_identity(policy), "sample_rate": sample_rate, "frames": frames, "targets": selected, "masks": []}
    if not selected:
        return audit
    words, tokens = _timed_words(report, sample_rate, frames, selected), _tokens(text)
    previous_end = 0
    for target in selected:
        first, last, method = _locate(tokens, words, target, selected)
        if first < previous_end or any(word.get("probability") is not None and word["probability"] < 0.2 for word in words[first:last]):
            raise CensorshipAlignmentRequired("target alignment is unordered or has very low confidence", selected)
        previous_end = last
        # Whisper's alignment grid is approximately 20 ms. A shorter reported
        # target is not evidence that the actual spoken word was that brief;
        # muting it would risk leaving most of the word audible.
        if words[last - 1]["end"] - words[first]["start"] < 0.02 - 1e-9:
            raise CensorshipAlignmentRequired("target duration is below reliable word-timing resolution", selected)
        start = max(0, round(words[first]["start"] * sample_rate))
        end = min(frames, round(words[last - 1]["end"] * sample_rate))
        duration = end - start
        if duration <= 0:
            raise CensorshipAlignmentRequired("target has no measurable audio frames", selected)
        fade = round(policy["fade_ms"] * sample_rate / 1000)
        preserved = min(round(policy["preserve_ms"] * sample_rate / 1000), math.floor(duration * policy["max_preserved_fraction"]))
        if duration <= max(2 * fade, 2) or preserved <= fade:
            preserved = 0
        fade_out = min(fade, preserved)
        next_start = round(words[last]["start"] * sample_rate) if last < len(words) else frames
        padded_end = max(end, min(frames, next_start, end + round(policy["padding_ms"] * sample_rate / 1000)))
        fade_in = min(fade, padded_end - end)
        alignment = "mlx_whisper_attention_dtw" if report.get("alignment_method") == "mlx_whisper_attention_dtw" else "asr_word_timestamps"
        audit["masks"].append({"source_start": target["start"], "source_end": target["end"], "target": target["text"], "normalized": target["normalized"], "method": alignment + ":" + method, "word_start_frame": start, "word_end_frame": end, "preserved_frames": preserved, "mute_start_frame": start + preserved - fade_out, "silence_start_frame": start + preserved, "silence_end_frame": padded_end - fade_in, "mute_end_frame": padded_end, "fade_out_frames": fade_out, "fade_in_frames": fade_in})
    audit["masks_sha256"] = _hash(audit["masks"])
    return audit


def validate_mask_plan(audit, text, policy=None, *, sample_rate, frames, audio_sha256, targets=None):
    """Validate persisted masks against current source, policy, and original audio."""
    if not isinstance(text, str):
        raise ValueError("Censorship source text must be a string")
    policy = validate_policy(policy)
    _audio_identity(sample_rate, frames, audio_sha256)
    selected = _targets(text, policy, targets) if policy["enabled"] else []
    expected = {"version": VERSION, "status": "complete", "text_sha256": _text_hash(text), "audio_sha256": audio_sha256, "policy": policy_identity(policy), "sample_rate": sample_rate, "frames": frames, "targets": selected}
    if not isinstance(audit, dict) or any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Censorship mask plan is stale or does not match its source audio and policy")
    masks = audit.get("masks")
    if not isinstance(masks, list) or len(masks) != len(selected) or (masks and audit.get("masks_sha256") != _hash(masks)):
        raise ValueError("Censorship mask plan is incomplete or its intervals have changed")
    previous_end = 0
    for target, mask in zip(selected, masks):
        if not isinstance(mask, dict) or any(mask.get(field) != target[key] for field, key in (("source_start", "start"), ("source_end", "end"), ("target", "text"), ("normalized", "normalized"))):
            raise ValueError("Censorship interval does not match its source target")
        fields = ("word_start_frame", "word_end_frame", "preserved_frames", "mute_start_frame", "silence_start_frame", "silence_end_frame", "mute_end_frame", "fade_out_frames", "fade_in_frames")
        if any(type(mask.get(field)) is not int for field in fields):
            raise ValueError("Censorship interval frame positions must be integers")
        start, end = mask["word_start_frame"], mask["word_end_frame"]
        mute_start, silence_start, silence_end, mute_end = (mask[field] for field in ("mute_start_frame", "silence_start_frame", "silence_end_frame", "mute_end_frame"))
        if not previous_end <= start <= mute_start <= silence_start < end <= silence_end <= mute_end <= frames or not start < end:
            raise ValueError("Censorship interval leaves its word or overlaps another word")
        if end - start < max(1, math.floor(0.02 * sample_rate)):
            raise ValueError("Censorship target duration is below reliable word-timing resolution")
        if mask["preserved_frames"] != silence_start - start or mask["fade_out_frames"] != silence_start - mute_start or mask["fade_in_frames"] != mute_end - silence_end:
            raise ValueError("Censorship fade envelope is inconsistent")
        limit = min(round(policy["preserve_ms"] * sample_rate / 1000), math.floor((end - start) * policy["max_preserved_fraction"]))
        fade = round(policy["fade_ms"] * sample_rate / 1000)
        if mask["preserved_frames"] > limit or max(mask["fade_out_frames"], mask["fade_in_frames"]) > fade or mute_end - end > round(policy["padding_ms"] * sample_rate / 1000):
            raise ValueError("Censorship interval exceeds its configured envelope")
        if mask.get("method") not in {"asr_word_timestamps:exact_transcript", "asr_word_timestamps:unique_context", "mlx_whisper_attention_dtw:exact_transcript", "mlx_whisper_attention_dtw:unique_context"}:
            raise ValueError("Censorship interval has an unsupported alignment method")
        previous_end = mute_end
    return audit


def apply_masks(pcm, start_frame, intervals, *, channels=1):
    """Apply silence/fades to PCM16LE, preserving byte count and all other frames."""
    if not isinstance(pcm, bytes) or type(channels) is not int or channels < 1 or len(pcm) % (2 * channels) or type(start_frame) is not int or start_frame < 0:
        raise ValueError("Censorship requires whole PCM16 frames and a nonnegative frame offset")
    if not pcm or not intervals:
        return pcm
    values = array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    end_frame = start_frame + len(values) // channels
    for mask in intervals:
        start, end = max(start_frame, mask["mute_start_frame"]), min(end_frame, mask["mute_end_frame"])
        for frame in range(start, end):
            if frame < mask["silence_start_frame"]:
                gain = (mask["silence_start_frame"] - frame - 1) / max(1, mask["fade_out_frames"])
            elif frame >= mask["silence_end_frame"]:
                gain = (frame - mask["silence_end_frame"] + 1) / max(1, mask["fade_in_frames"])
            else:
                gain = 0.0
            offset = (frame - start_frame) * channels
            for channel in range(channels):
                values[offset + channel] = round(values[offset + channel] * gain)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()
