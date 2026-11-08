"""Pure selection and diff helpers for published-book attribution verification."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from .speaker_decisions import _role_labels


def _role_ids(segment):
    if segment.get("simultaneous_speakers"):
        return tuple(speaker["role_id"] for speaker in segment["simultaneous_speakers"])
    role_id = segment.get("role_id")
    return (role_id,) if role_id else ()


def risk_blocks(document):
    """Return source-ordered blocks whose dialogue needs another attribution look."""
    by_block = {block["id"]: [] for block in document["source"]["blocks"]}
    for segment in document["segments"]:
        by_block[segment["block_id"]].append(segment)
    selected = []
    blocks = document["source"]["blocks"]
    for index, block in enumerate(blocks):
        segments = by_block[block["id"]]
        flagged = any(segment["attribution"].get("review_required")
                      or segment["attribution"].get("basis") == "unresolved"
                      for segment in segments)
        neighbors = [segment for neighbor in blocks[max(0, index - 1):index + 2]
                     for segment in by_block[neighbor["id"]]]
        speakers = {role_id for segment in neighbors for role_id in _role_ids(segment)}
        contextual = any(segment["kind"] in {"dialogue", "thought"}
                         and segment["attribution"].get("basis") == "context"
                         for segment in segments)
        if flagged or (contextual and len(speakers) >= 2):
            selected.append(block["id"])
    return selected


def chapter_block_ids(document, chapter_ids):
    """Map requested chapter IDs to block IDs in source chapter order."""
    requested = set(chapter_ids)
    known = {chapter["id"] for chapter in document["source"]["chapters"]}
    unknown = requested - known
    if unknown:
        raise ValueError("Unknown chapter ids: " + ", ".join(sorted(unknown)))
    return {chapter["id"]: list(chapter["block_ids"])
            for chapter in document["source"]["chapters"] if chapter["id"] in requested}


def sample_blocks(block_ids, count):
    """Select a deterministic SHA256-ranked sample while retaining input order."""
    if type(count) is not int or count < 1:
        raise ValueError("sample count must be an integer of at least 1")
    ordered = list(dict.fromkeys(block_ids))
    ranked = sorted(sorted(ordered),
                    key=lambda block_id: hashlib.sha256(block_id.encode("utf-8")).digest())
    chosen = set(ranked[:min(count, len(ranked))])
    return [block_id for block_id in ordered if block_id in chosen]


def findings(original, audited):
    """Describe same-segment speaker corrections and newly raised review flags."""
    originals = {segment["id"]: segment for segment in original["segments"]}
    chapters = {block["id"]: block["chapter_id"] for block in original["source"]["blocks"]}
    characters = {character["id"]: character for character in original["cast"]["characters"]}
    labels = _role_labels(original["cast"]["roles"], characters, original["source"]["book"])

    def label(segment):
        role_ids = _role_ids(segment)
        return ", ".join(labels[role_id] for role_id in role_ids) if role_ids else "unassigned"

    result = []
    for segment in audited["segments"]:
        before = originals.get(segment["id"])
        if before is None:
            continue
        before_roles, after_roles = _role_ids(before), _role_ids(segment)
        before_attribution, after_attribution = before["attribution"], segment["attribution"]
        newly_flagged = (not before_attribution["review_required"]
                         and after_attribution["review_required"])
        if before_roles == after_roles and not newly_flagged:
            continue
        text = segment["text"]
        result.append({
            "segment_id": segment["id"], "chapter_id": chapters[segment["block_id"]],
            "block_id": segment["block_id"], "text": text[:99] + "…" if len(text) > 100 else text,
            "from_label": label(before), "to_label": label(segment),
            "basis": after_attribution["basis"],
            "evidence": [{"block_id": item["block_id"], "quote": item["quote"]}
                         for item in after_attribution["evidence"]],
            "review_only": before_roles == after_roles and newly_flagged,
        })
    return result


def restrict_findings(original, audited, chosen_segment_ids):
    """Return the original document with only chosen same-ID segments replaced."""
    originals = {segment["id"]: segment for segment in original["segments"]}
    replacements = {segment["id"]: segment for segment in audited["segments"]}
    chosen = list(dict.fromkeys(chosen_segment_ids))
    unknown = [segment_id for segment_id in chosen
               if segment_id not in originals or segment_id not in replacements]
    if unknown:
        raise ValueError("Chosen segment IDs must exist in both documents: " + ", ".join(unknown))
    selected = set(chosen)
    revised = deepcopy(original)
    applied = []
    for index, segment in enumerate(revised["segments"]):
        if segment["id"] in selected:
            revised["segments"][index] = deepcopy(replacements[segment["id"]])
            applied.append(segment["id"])
    return revised, applied
