"""Offline, rollback-protected migration of this seven-book run to v2 bundles.

All analyses, source text, casting, review judgments and the published registry
remain byte-for-byte unchanged. Each original bundle is retained beside its
replacement as bundle-before-chapter-split-v1. Existing ZIPs are never changed.
No configuration or provider is loaded. This helper refuses to run twice.
The CLI requires --root pointing to the existing legacy run directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from epub_tts_narrator.cli import export_bundle, read_analysis
from epub_tts_narrator.pipeline import fingerprint
from bundle_validation import load_bundle, parse_json, regular_file, require, sha, validate_native_bundle

BACKUP = "bundle-before-chapter-split-v1"
LEDGER = "chapter-split-migration.json"


def snapshot(root):
    """Snapshot only authoritative read-only inputs, never configs or caches."""
    paths = [root / name for name in ("series-cast.json", "inventory.json")]
    paths.extend(root.glob("*.zip"))
    for number in range(1, 8):
        folder = root / f"book-{number:02d}"
        paths.extend(folder / name for name in
                     ("analysis.json", "source.json", "registry-before.json", "review.json", "summary.json")
                     if (folder / name).exists())
        final = folder / "final-source-audit.json"
        paths.append(final if final.exists() else folder / "post-review-source-audit.json")
    return {path.relative_to(root).as_posix(): sha(regular_file(path)) for path in sorted(paths)}


def verify_snapshot(root, expected):
    require(snapshot(root) == expected, "Authoritative run inputs changed during bundle migration")


def file_hashes(entries):
    return {name: sha(data) for name, data in entries.items()}


def script_rows(entries):
    manifest = parse_json(entries["manifest.json"])
    files = (["script.jsonl"] if manifest["version"] == 1 else
             [entry["path"] for entry in manifest["files"]["scripts"]])
    return [parse_json(line) for filename in files
            for line in entries[filename].decode("utf-8").splitlines()]


def migrate(root):
    root = Path(root).absolute()
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.parents),
            "Migration refuses symlink roots")
    ledger_path = root / LEDGER
    require(not ledger_path.exists() and not ledger_path.is_symlink(),
            "Chapter migration ledger already exists; refusing to replay")
    lock = root / ".chapter-split-migration.lock"
    lock.mkdir()  # Exclusive lock: existing ownership is never removed.
    stage = None
    moved = []
    try:
        protected = snapshot(root)
        documents, old_entries = {}, {}
        for number in range(1, 8):
            folder = root / f"book-{number:02d}"
            backup = folder / BACKUP
            require(not backup.exists() and not backup.is_symlink(), "A chapter migration backup already exists")
            document = read_analysis(folder / "analysis.json")
            entries = load_bundle(folder / "bundle")
            require(validate_native_bundle(entries, "", document)["version"] == 1,
                    "Migration requires all seven original version 1 bundles")
            documents[number], old_entries[number] = document, entries
        stage = Path(tempfile.mkdtemp(prefix=".chapter-split-stage-", dir=root))
        new_entries = {}
        for number, document in documents.items():
            output = stage / f"book-{number:02d}"
            export_bundle(document, output, allow_review=True)
            entries = load_bundle(output)
            require(validate_native_bundle(entries, "", document)["version"] == 2,
                    "Current exporter did not produce a version 2 chapter bundle")
            require(script_rows(entries) == script_rows(old_entries[number]),
                    "Flattened chapter script differs from the original version 1 rows")
            new_entries[number] = entries
        verify_snapshot(root, protected)
        for number in documents:
            folder = root / f"book-{number:02d}"
            require(load_bundle(folder / "bundle") == old_entries[number],
                    "An original bundle changed during staging")
        # All seven are validated before the first current bundle is moved.
        for number in documents:
            folder = root / f"book-{number:02d}"
            backup = folder / BACKUP
            require(not backup.exists() and not backup.is_symlink(), "Backup name became occupied")
            os.rename(folder / "bundle", backup)
            moved.append(number)
            require(not (folder / "bundle").exists(), "Bundle destination became occupied")
            os.rename(stage / f"book-{number:02d}", folder / "bundle")
        verify_snapshot(root, protected)
        books = []
        for number, document in documents.items():
            folder = root / f"book-{number:02d}"
            require(load_bundle(folder / BACKUP) == old_entries[number], "Original backup changed during installation")
            require(load_bundle(folder / "bundle") == new_entries[number], "New bundle changed during installation")
            books.append({
                "book": number, "analysis_fingerprint": fingerprint(document),
                "source_fingerprint": fingerprint(document["source"]),
                "cast_fingerprint": fingerprint(document["cast"]),
                "segments_fingerprint": fingerprint(document["segments"]),
                "segments": len(document["segments"]),
                "flattened_script_unchanged": True,
                "chapter_scripts": len(parse_json(new_entries[number]["manifest.json"])["files"]["scripts"]),
                "backup": f"book-{number:02d}/{BACKUP}",
                "before_sha256": file_hashes(old_entries[number]),
                "after_sha256": file_hashes(new_entries[number]),
            })
        ledger = {"format": "soulship-chapter-bundle-migration", "version": 1,
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "from_bundle_version": 1, "to_bundle_version": 2,
                  "authoritative_inputs_unchanged_sha256": protected, "books": books,
                  "provider_calls": 0, "analysis_mutations": 0,
                  "note": "Only renderer bundle layout changed. Sequence ranges have exclusive ends. Original ZIPs and version 1 bundles are retained."}
        staged_ledger = stage / LEDGER
        with staged_ledger.open("x", encoding="utf-8") as stream:
            json.dump(ledger, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        verify_snapshot(root, protected)
        os.link(staged_ledger, ledger_path)  # Exclusive publication, never replaces.
        return ledger
    except BaseException:
        # Roll back ordinary failures and interruptions, in reverse book order.
        # Do not delete a destination modified independently after installation.
        for number in reversed(moved):
            folder = root / f"book-{number:02d}"
            destination, backup = folder / "bundle", folder / BACKUP
            if destination.exists():
                require(load_bundle(destination) == new_entries[number],
                        "Concurrent modification blocks automatic rollback; original backup is retained")
                os.rename(destination, stage / f"rollback-book-{number:02d}")
            os.rename(backup, destination)
        raise
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing legacy seven-book run directory")
    args = parser.parse_args()
    result = migrate(args.root)
    print(json.dumps({"ledger": str(Path(args.root) / LEDGER),
                      "books": len(result["books"]),
                      "chapter_scripts": sum(row["chapter_scripts"] for row in result["books"]),
                      "segments": sum(row["segments"] for row in result["books"]),
                      "provider_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
