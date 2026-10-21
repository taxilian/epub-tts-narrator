"""Validate an analyzer bundle and construct a lossless speech plan."""

from __future__ import annotations

import json
from pathlib import Path
import re

from epub_audiobook.contracts import validate_document, review_items
from epub_audiobook.cli import script_partitions
from epub_audiobook.pipeline import fingerprint
from .common import digest, file_hash, read_json, regular_file
from .quality import normalize_words


def load_bundle(directory):
    directory = Path(directory).resolve()
    manifest = read_json(regular_file(directory / "manifest.json"))
    if manifest.get("format") != "epub-audiobook-bundle" or manifest.get("version") != 2:
        raise ValueError("Renderer requires a version-2 chapter bundle; re-export older analyses")
    files = manifest.get("files", {})
    if {k: files.get(k) for k in ("source", "cast", "review")} != {
        "source": "source.json", "cast": "cast.json", "review": "review.json"
    } or set(files) != {"source", "cast", "review", "scripts"}:
        raise ValueError("Invalid bundle file inventory")
    scripts = files["scripts"]
    if not isinstance(scripts, list) or not scripts:
        raise ValueError("Missing chapter scripts")
    names = ["source.json", "cast.json", "review.json"]
    for number, entry in enumerate(scripts, 1):
        expected = f"scripts/chapter-{number:04d}.jsonl"
        if not isinstance(entry, dict) or entry.get("path") != expected:
            raise ValueError("Invalid chapter filename or order")
        names.append(expected)
    if set(manifest.get("sha256", {})) != set(names):
        raise ValueError("Bundle checksum inventory is incomplete")
    for name in names:
        path = directory / name
        if any(p.is_symlink() for p in path.parents if p != directory.parent):
            raise ValueError("Bundle paths must not use symlinks")
        if file_hash(regular_file(path)) != manifest["sha256"][name]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    source, cast, reviews = (read_json(directory / name) for name in names[:3])
    if not isinstance(reviews, list) or fingerprint(source) != manifest.get("source_fingerprint"):
        raise ValueError("Invalid review list or source fingerprint")
    if source["book"] != manifest.get("book") or source["chapters"] != manifest.get("chapters"):
        raise ValueError("Manifest source metadata mismatch")
    chapters, rows, cursor = [], [], 0
    for entry in scripts:
        chapter_rows = []
        with (directory / entry["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                def reject(_):
                    raise ValueError("Non-finite JSON in script")
                row = json.loads(line, parse_constant=reject)
                if type(row.get("sequence")) is not int or row.get("sequence") != cursor or row.get("chapter_id") != entry.get("chapter_id"):
                    raise ValueError("Invalid chapter row order or membership")
                chapter_rows.append(row)
                rows.append(row)
                cursor += 1
        if (type(entry.get("segment_count")) is not int or
                entry["segment_count"] != len(chapter_rows) or
                type(entry.get("sequence_start")) is not int or
                type(entry.get("sequence_end")) is not int or
                entry["sequence_start"] != cursor - len(chapter_rows) or entry["sequence_end"] != cursor):
            raise ValueError("Invalid chapter sequence range")
        chapters.append({"entry": entry, "rows": chapter_rows})
    document = {"format": "epub-audiobook-analysis", "version": 1,
                "source": source, "cast": cast, "segments": rows, "issues": [],
                "provenance": manifest.get("provenance", {})}
    errors = validate_document(document)
    if errors:
        raise ValueError("Invalid source/cast/script: " + "; ".join(errors[:3]))
    blocks = {b["id"]: b for b in source["blocks"]}
    for row in rows:
        block = blocks[row["block_id"]]
        if row["chapter_id"] != block["chapter_id"]:
            raise ValueError("Script chapter differs from source block")
        if row.get("break_after") != ("paragraph" if row["end"] == len(block["text"]) else "none"):
            raise ValueError("Script paragraph break differs from source")
        if row.get("break_before") != (block.get("break_before") if row["start"] == 0 else None):
            raise ValueError("Script preceding break differs from source")
    partitions = script_partitions(document)
    if len(partitions) != len(chapters):
        raise ValueError("Script partitions differ from source chapter headings")
    for expected, chapter in zip(partitions, chapters):
        entry = chapter["entry"]
        if any(entry.get(key) != expected[key] for key in ("chapter_id", "title", "heading_block_id")):
            raise ValueError("Script chapter title or heading differs from source")
        block_ids = list(dict.fromkeys(row["block_id"] for row in chapter["rows"]))
        if block_ids != expected["block_ids"]:
            raise ValueError("Script partition does not match source chapter boundaries")
    return {"directory": str(directory), "manifest": manifest, "source": source,
            "cast": cast, "reviews": reviews, "chapters": chapters,
            "implied_reviews": review_items(document),
            "bundle_hash": digest(manifest)}


def spoken_mapping(row):
    """Map spoken code points to canonical block offsets, without guessing edits."""
    text = row["text"]
    spoken = row.get("spoken_text", text)
    if spoken == text:
        return spoken, list(range(row["start"], row["end"]))
    # Analyzer's sole text transformation removes square-bracket delimiters.
    indices = [i for i, char in enumerate(text) if char not in "[]"]
    if "".join(text[i] for i in indices) != spoken:
        raise ValueError("Unsupported spoken_text transformation; explicit source mapping is required")
    return spoken, [row["start"] + i for i in indices]


def split_text(text, maximum):
    """Contiguous, exact slices; prefer sentence/word boundaries."""
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            piece = text[start:end]
            boundaries = [m.end() for m in re.finditer(r'[.!?][”"\']?\s+', piece)]
            cut = boundaries[-1] if boundaries and boundaries[-1] >= maximum // 3 else piece.rfind(" ") + 1
            if cut > 0:
                end = start + cut
        yield start, end
        start = end


def ranges(indices):
    result = []
    for index in indices:
        if result and result[-1]["end"] == index:
            result[-1]["end"] += 1
        else:
            result.append({"start": index, "end": index + 1})
    return result


def make_plan(bundle, config):
    from .book_context import resolve_book_config
    config = resolve_book_config(bundle, config)
    settings = config["render"]
    selected = settings["chapter_numbers"] or list(range(1, len(bundle["chapters"]) + 1))
    if any(n > len(bundle["chapters"]) for n in selected):
        raise ValueError("Configured chapter number is outside the bundle")
    selected = set(selected)
    chapters, errors, used = [], [], set()
    established_series = bundle["manifest"].get("provenance", {}).get("series", {}).get("series_id")
    if established_series and config["voices"]["series_id"] != established_series:
        errors.append("Configured voices.series_id differs from the analyzer series ID; reuse the exact series ID")
    state = bundle["manifest"].get("readiness", {})
    if (state.get("ready") is not True or bundle["reviews"] or bundle.get("implied_reviews")) and not settings["allow_review"]:
        errors.append("Bundle has outstanding review items; resolve them or pass --allow-review for this run")
    for number, item in enumerate(bundle["chapters"], 1):
        if number not in selected:
            continue
        chunks = []
        for row in item["rows"]:
            spoken, mapping = spoken_mapping(row)
            if normalize_words(spoken) and row.get("voice_id") is None:
                errors.append(f"Missing voice for segment {row['id']}; resolve attribution before synthesis")
            spans = list(split_text(spoken, settings["max_chunk_chars"])) or [(0, 0)]
            for start, end in spans:
                speech = spoken[start:end]
                chunk = {
                    "segment": row, "spoken_text": speech, "spoken_start": start, "spoken_end": end,
                    "source_ranges": ranges(mapping[start:end]),
                    "silent_control": not normalize_words(speech),
                }
                chunk["chunk_id"] = digest({"source": bundle["manifest"]["source_fingerprint"],
                                            "segment": row["id"], "start": start, "end": end,
                                            "spoken_text": speech})[:32]
                if not chunk["silent_control"] and row.get("voice_id"):
                    used.add(row["voice_id"])
                chunks.append(chunk)
        chapters.append({"number": number, "id": f"chapter-{number:04d}", "title": item["entry"]["title"],
                         "script_path": item["entry"]["path"], "chunks": chunks})
    return {"format": "audiobook-render-plan", "version": 1,
            "bundle_hash": bundle["bundle_hash"], "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "book": bundle["source"]["book"], "series_id": config["voices"]["series_id"],
            "voice_ids": sorted(used), "chapters": chapters, "errors": errors,
            "review_items": max(len(bundle["reviews"]), len(bundle.get("implied_reviews", []))),
            "partial_book": len(selected) != len(bundle["chapters"])}
