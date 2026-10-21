"""Independent validation of native v1/v2 renderer handoffs.

No provider configuration is read. Manifest paths are accepted only when they
match the format's fixed relative filenames, before any path is opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from epub_tts_narrator.contracts import readiness, review_items
from epub_tts_narrator.pipeline import fingerprint

FIXED_FILES = ("source.json", "cast.json", "review.json")
CREDENTIAL_KEYS = {"apikey", "authorization", "password", "accesstoken", "refreshtoken",
                   "authtoken", "clientsecret", "secretaccesskey", "awssecretaccesskey"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_json(data):
    def reject_constant(_):
        raise ValueError("Non-finite numbers are not valid JSON")
    return json.loads(data, parse_constant=reject_constant)


def regular_file(path):
    path = Path(path)
    require(not path.is_symlink() and not any(p.is_symlink() for p in path.parents),
            "Bundle inputs must not use symlinks")
    require(path.is_file(), "Required bundle input is missing or not a regular file")
    return path.read_bytes()


def no_credentials(value):
    """Reject obvious credential fields without inspecting or echoing values."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = "".join(c for c in key.casefold() if c.isalnum())
            require(normalized not in CREDENTIAL_KEYS, "Credential field found in handoff metadata")
            no_credentials(child)
    elif isinstance(value, list):
        for child in value:
            no_credentials(child)


def bundle_names(manifest):
    """Return a safe ordered inventory; never trust arbitrary manifest paths."""
    require(isinstance(manifest, dict) and manifest.get("format") == "epub-audiobook-bundle"
            and type(manifest.get("version")) is int and manifest["version"] in (1, 2),
            "Bundle format or version is unsupported")
    files = manifest.get("files")
    require(isinstance(files, dict), "Bundle file inventory is invalid")
    fixed = {name.split(".")[0]: name for name in FIXED_FILES}
    if manifest["version"] == 1:
        require(files == dict(fixed, script="script.jsonl"), "Legacy bundle file inventory is invalid")
        data_names = (*FIXED_FILES, "script.jsonl")
    else:
        require(set(files) == {*fixed, "scripts"} and all(files[k] == v for k, v in fixed.items()),
                "Chapter bundle file inventory is invalid")
        scripts = files["scripts"]
        chapters = manifest.get("chapters")
        require(isinstance(chapters, list) and isinstance(scripts, list) and len(scripts) >= len(chapters),
                "Chapter script inventory does not match chapter inventory")
        chapter_ids = [chapter.get("id") for chapter in chapters if isinstance(chapter, dict)]
        require(len(chapter_ids) == len(chapters) and len(set(chapter_ids)) == len(chapter_ids)
                and all(isinstance(value, str) and value for value in chapter_ids),
                "Native chapter inventory is invalid")
        cursor, paths, seen_chapters = 0, [], []
        for number, entry in enumerate(scripts, 1):
            require(isinstance(entry, dict) and set(entry) ==
                    {"chapter_id", "title", "heading_block_id", "path",
                     "segment_count", "sequence_start", "sequence_end"},
                    "Chapter script entry is invalid")
            path = f"scripts/chapter-{number:04d}.jsonl"
            require(entry["chapter_id"] in chapter_ids and entry["path"] == path,
                    "Chapter script chapter or path is invalid")
            if not seen_chapters or seen_chapters[-1] != entry["chapter_id"]:
                seen_chapters.append(entry["chapter_id"])
            require(isinstance(entry["title"], str) and
                    (entry["heading_block_id"] is None or
                     isinstance(entry["heading_block_id"], str) and bool(entry["heading_block_id"])),
                    "Chapter script title or heading reference is invalid")
            require(all(type(entry[k]) is int and entry[k] >= 0
                        for k in ("segment_count", "sequence_start", "sequence_end")),
                    "Chapter script count or sequence bounds are invalid")
            require(entry["sequence_start"] == cursor
                    and entry["sequence_end"] == cursor + entry["segment_count"],
                    "Chapter script sequence ranges are not contiguous")
            cursor = entry["sequence_end"]
            paths.append(path)
        require(seen_chapters == chapter_ids, "Chapter script inventory is not in native source chapter order")
        data_names = (*FIXED_FILES, *paths)
    hashes = manifest.get("sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(data_names),
            "Bundle checksum inventory is invalid")
    require(all(isinstance(value, str) and len(value) == 64 and
                all(c in "0123456789abcdef" for c in value) for value in hashes.values()),
            "Bundle checksum value is invalid")
    return ("manifest.json", *data_names)


def load_bundle(folder):
    """Read only the validated fixed manifest inventory, refusing extra files."""
    folder = Path(folder)
    manifest_bytes = regular_file(folder / "manifest.json")
    names = bundle_names(parse_json(manifest_bytes))
    # Unexpected files can hide stale monolithic scripts or accidental secrets.
    actual = set()
    for path in folder.rglob("*"):
        require(not path.is_symlink(), "Bundle inputs must not use symlinks")
        require(path.is_file() or path.is_dir(), "Bundle contains a nonregular entry")
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    require(actual == set(names), "Bundle directory contains unexpected or missing files")
    return {name: manifest_bytes if name == "manifest.json" else regular_file(folder / name)
            for name in names}


def rendered_rows(document):
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    rows = []
    for index, segment in enumerate(document["segments"]):
        block = blocks[segment["block_id"]]
        expected = dict(segment, sequence=index, chapter_id=block["chapter_id"],
                        break_after="paragraph" if segment["end"] == len(block["text"]) else "none")
        if segment["start"] == 0 and block.get("break_before"):
            expected["break_before"] = block["break_before"]
        rows.append(expected)
    return rows


def validate_native_bundle(entries, prefix, document):
    """Compare all manifest metadata and every row to the current analysis."""
    manifest = parse_json(entries[prefix + "manifest.json"])
    names = bundle_names(manifest)
    require({key for key in entries if key.startswith(prefix)} ==
            {prefix + name for name in names}, "Native bundle member inventory differs from manifest")
    for name in names[1:]:
        require(sha(entries[prefix + name]) == manifest["sha256"][name],
                "Native bundle checksum differs from manifest")
    rows = rendered_rows(document)
    files = {name.split(".")[0]: name for name in FIXED_FILES}
    if manifest["version"] == 1:
        files["script"] = "script.jsonl"
        expected_scripts = [("script.jsonl", rows)]
    else:
        from epub_tts_narrator.cli import script_partitions
        partitions = script_partitions(document)
        require([block_id for part in partitions for block_id in part["block_ids"]] ==
                [block["id"] for block in document["source"]["blocks"]],
                "Logical chapter partition does not preserve exact native block coverage")
        scripts, expected_scripts, cursor = [], [], 0
        for number, chapter in enumerate(partitions, 1):
            block_ids = set(chapter["block_ids"])
            chapter_rows = [row for row in rows if row["block_id"] in block_ids]
            path = f"scripts/chapter-{number:04d}.jsonl"
            end = cursor + len(chapter_rows)
            scripts.append({"chapter_id": chapter["chapter_id"], "title": chapter["title"],
                            "heading_block_id": chapter["heading_block_id"], "path": path,
                            "segment_count": len(chapter_rows),
                            "sequence_start": cursor, "sequence_end": end})
            expected_scripts.append((path, chapter_rows))
            cursor = end
        files["scripts"] = scripts
    expected_manifest = {
        "format": "epub-audiobook-bundle", "version": manifest["version"],
        "book": document["source"]["book"], "chapters": document["source"]["chapters"],
        "source_fingerprint": fingerprint(document["source"]), "readiness": readiness(document),
        "offsets": "Zero-based Unicode code points within source block text; end exclusive.",
        "files": files, "sha256": {name: sha(entries[prefix + name]) for name in names[1:]},
        "provenance": document["provenance"],
    }
    require(manifest == expected_manifest, "Native manifest differs from current analysis export metadata")
    no_credentials(manifest)
    for name, expected in (("source.json", document["source"]), ("cast.json", document["cast"]),
                           ("review.json", review_items(document))):
        actual = parse_json(entries[prefix + name])
        no_credentials(actual)
        require(actual == expected, "Native source, cast, or review differs from current analysis")
    for path, expected in expected_scripts:
        actual = [parse_json(line) for line in entries[prefix + path].decode("utf-8").splitlines()]
        no_credentials(actual)
        require(actual == expected, "Native script rows, order, chapter, or performance differ from current analysis")
    return {"version": manifest["version"], "files_checked": len(names) - 1,
            "script_files": len(expected_scripts), "segments": len(rows)}
