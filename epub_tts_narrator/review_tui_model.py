"""Pure document and editing logic for the interactive published-book review UI."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .attribution_edits import apply_edits, edit_direction, edit_review, edit_speaker
from .contracts import has_assigned_speakers
from .speaker_decisions import _role_labels


# Exact xterm-256 RGB derivation: the standard base palette, the 6x6x6 color
# cube, then the 24-step grayscale. Keeping these inputs explicit makes the
# contrast calculations independently testable without a terminal or curses.
_BASE16_RGB = (
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
)
_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)
XTERM_RGB = _BASE16_RGB + tuple(
    (_CUBE_LEVELS[red], _CUBE_LEVELS[green], _CUBE_LEVELS[blue])
    for red in range(6) for green in range(6) for blue in range(6)
) + tuple((8 + 10 * step, 8 + 10 * step, 8 + 10 * step) for step in range(24))

# Pleasant mid-luminance choices from the eligible cube. Remaining eligible
# indices follow in numeric order, so large casts stay deterministic too.
_PREFERRED_COLORS = (
    28, 31, 33, 64, 69, 92, 99, 100, 105, 126, 129,
    135, 136, 139, 160, 165, 166, 171, 196, 201, 202,
)
def relative_luminance(rgb):
    """Return WCAG relative luminance for an sRGB 8-bit RGB triple."""
    def linear(channel):
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = rgb
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def contrast_ratio(foreground, background):
    """Return the WCAG contrast ratio between two sRGB RGB triples."""
    first, second = relative_luminance(foreground), relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def contrast_ok(index):
    """Return whether an xterm color is readable at 3:1 on black and white."""
    if type(index) is not int or not 0 <= index < len(XTERM_RGB):
        raise ValueError(f"xterm color index must be from 0 through {len(XTERM_RGB) - 1}")
    rgb = XTERM_RGB[index]
    return (contrast_ratio(rgb, (0, 0, 0)) >= 3.0
            and contrast_ratio(rgb, (255, 255, 255)) >= 3.0)


_ELIGIBLE_COLORS = tuple(index for index in range(16, len(XTERM_RGB)) if contrast_ok(index))
_PREFERRED = tuple(index for index in _PREFERRED_COLORS if contrast_ok(index))
_PALETTE = _PREFERRED + tuple(index for index in _ELIGIBLE_COLORS if index not in _PREFERRED)


def palette(n):
    """Return n deterministic, dual-background-readable xterm color indices."""
    if type(n) is not int or n < 0:
        raise ValueError("Palette size must be a nonnegative integer")
    return [_PALETTE[index % len(_PALETTE)] for index in range(n)]


def _role_ids(segment):
    if not has_assigned_speakers(segment):
        return []
    if "simultaneous_speakers" in segment:
        return [speaker["role_id"] for speaker in segment["simultaneous_speakers"]]
    return [segment["role_id"]]


def _labels(document):
    return _role_labels(document["cast"]["roles"],
                        {row["id"]: row for row in document["cast"]["characters"]},
                        document["source"]["book"])


def _needs_review(segment):
    attribution = segment["attribution"]
    return bool(attribution["review_required"] or attribution["basis"] == "unresolved")


def _chapter_spans(document):
    """Return (partition, sequence_start, segments) per rendered chapter.

    Source chapters are EPUB resources that may hold several story chapters;
    ``script_partitions`` defines the boundaries the exported scripts and the
    rendered M4B actually use, so the review UI navigates those, not the
    raw spine chapters.
    """
    from .cli import script_partitions
    segments = document["segments"]
    spans = []
    cursor = 0
    for partition in script_partitions(document):
        block_ids = set(partition["block_ids"])
        start = cursor
        while cursor < len(segments) and segments[cursor]["block_id"] in block_ids:
            cursor += 1
        spans.append((partition, start, segments[start:cursor]))
    return spans


def toc(document):
    """Summarize passages and outstanding review work per rendered chapter."""
    rows = []
    for index, (partition, _, chapter_segments) in enumerate(_chapter_spans(document)):
        rows.append({
            "id": index,
            "title": partition["title"],
            "passage_count": len(chapter_segments),
            "review_count": sum(_needs_review(segment) for segment in chapter_segments),
            "unassigned_count": sum(not has_assigned_speakers(segment) for segment in chapter_segments),
        })
    return rows


def passages(document, chapter_index):
    """Return display rows in global script order for one rendered chapter."""
    labels = _labels(document)
    slots = {role["id"]: index for index, role in enumerate(document["cast"]["roles"])}
    _, start, chapter_segments = _chapter_spans(document)[chapter_index]
    rows = []
    for sequence, segment in enumerate(chapter_segments, start):
        role_ids = _role_ids(segment)
        speaker_labels = [labels[role_id] for role_id in role_ids]
        rows.append({
            "segment_id": segment["id"],
            "text": segment["text"],
            "kind": segment["kind"],
            "from_label": ", ".join(speaker_labels) if speaker_labels else "Unassigned",
            "speaker_labels": speaker_labels,
            "role_index": slots[role_ids[0]] if role_ids else None,
            "role_indices": [slots[role_id] for role_id in role_ids],
            "review_required": _needs_review(segment),
            "basis": segment["attribution"]["basis"],
            "sequence": sequence,
        })
    return rows


def metadata(document, segment_id) -> dict:
    """Return the complete display/edit metadata for one segment."""
    labels = _labels(document)
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    voices = {voice["id"]: voice for voice in document["cast"]["voices"]}
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    for sequence, segment in enumerate(document["segments"]):
        if segment["id"] != segment_id:
            continue
        role_ids = _role_ids(segment)
        voice_ids = [roles[role_id]["voice_id"] for role_id in role_ids]
        controls = segment.get("performance_controls", {}).get("controls", [])
        speakers = [{"role_id": role_id, "voice_id": roles[role_id]["voice_id"],
                     "label": labels[role_id],
                     "voice_description": voices[roles[role_id]["voice_id"]]["description"]}
                    for role_id in role_ids]
        result = {
            "id": segment["id"],
            "block_id": segment["block_id"],
            "chapter_id": blocks[segment["block_id"]]["chapter_id"],
            "offsets": {"start": segment["start"], "end": segment["end"]},
            "sequence": sequence,
            "kind": segment["kind"],
            "text": segment["text"],
            "speaker_labels": [speaker["label"] for speaker in speakers],
            "role_ids": role_ids,
            "voice_ids": voice_ids,
            "voice_descriptions": [speaker["voice_description"] for speaker in speakers],
            "speakers": speakers,
            "attribution": deepcopy(segment["attribution"]),
            "delivery": deepcopy(segment["delivery"]),
            "performance_controls": {
                "total": len(controls),
                "kinds": dict(Counter(control["kind"] for control in controls)),
                "read_only": True,
            },
            "editable": ["speaker", "direction", "review_required"],
        }
        if "simultaneous_speakers" in segment:
            result["simultaneous_speakers"] = deepcopy(segment["simultaneous_speakers"])
        core = {"id", "block_id", "start", "end", "text", "kind", "role_id", "voice_id",
                "attribution", "delivery", "performance_controls", "simultaneous_speakers"}
        result["additional"] = {key: deepcopy(value) for key, value in segment.items() if key not in core}
        return result
    raise ValueError(f"Unknown segment id: {segment_id}")





