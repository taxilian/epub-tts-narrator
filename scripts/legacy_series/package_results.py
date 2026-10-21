"""Create a new, verified TTS handoff ZIP only after all seven results are final.

Native bundle bytes and provenance are preserved. This helper reads only its
explicit handoff/validation inputs: never provider configuration, caches or logs.
It does not update analyses, audits, reports, registry, or bundle files.
The CLI requires --root pointing to the existing legacy run directory.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import validate_registry
from audit_continuity import audit_book
from report_series import summarize
from bundle_validation import (bundle_names, load_bundle, no_credentials, parse_json,
                               regular_file, sha, validate_native_bundle)

SHARED_FILES = ("series-cast.json", "voice-continuity.csv", "series-report.txt")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_report(text):
    lines = text.splitlines()
    require(len(lines) > 1 and lines[1].startswith("Generated: "), "Series report has an unexpected header")
    return lines[:1] + lines[2:]


def verify_inputs(bindings):
    for path, expected in bindings.items():
        require(sha(regular_file(path)) == expected, "Packaging input changed during validation or staging")


def archive_inventory(entries):
    """Derive an ordered allowlist only from seven validated manifest layouts."""
    allowed = []
    for number in range(1, 8):
        prefix = f"book-{number:02d}/bundle/"
        manifest = parse_json(entries[prefix + "manifest.json"])
        names = bundle_names(manifest)
        allowed.extend(prefix + name for name in names)
        for name in names[1:]:
            require(sha(entries[prefix + name]) == manifest["sha256"][name],
                    "Archive data differs from manifest checksum")
    allowed.extend((*SHARED_FILES, "FORMAT.md", "README.txt"))
    require(set(entries) == set(allowed) and len(entries) == len(allowed),
            "Unexpected handoff inventory")
    return tuple(allowed)


def collect_entries(root, *, format_path=PROJECT / "FORMAT.md"):
    """Validate all seven published drafts, then return exact handoff bytes."""
    root = Path(root).absolute()
    format_path = Path(format_path).absolute()
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.parents),
            "Packaging refuses symlink input roots")
    final_audits = {}
    # Fail early on an unfinished run; no archive or stage exists at this point.
    for number in range(1, 8):
        folder = root / f"book-{number:02d}"
        require((folder / "analysis.json").is_file(), f"Book {number:02d} is incomplete: analysis is missing")
        final = folder / "final-source-audit.json"
        if not final.is_file():
            final = folder / "post-review-source-audit.json"
        require(final.is_file(), f"Book {number:02d} has no final source audit")
        final_audits[number] = final

    entries = {}
    for number in range(1, 8):
        prefix = f"book-{number:02d}/bundle/"
        native = load_bundle(root / prefix)
        entries.update({prefix + name: data for name, data in native.items()})
    entries.update({name: regular_file(root / name) for name in SHARED_FILES})
    input_paths = {root / name for name in entries}
    input_paths.add(format_path)
    input_paths.add(root / "inventory.json")
    for number in range(1, 8):
        folder = root / f"book-{number:02d}"
        input_paths.update((folder / "analysis.json", folder / "source.json",
                            folder / "registry-before.json", final_audits[number]))
    bindings = {path: sha(regular_file(path)) for path in sorted(input_paths)}
    require(all(sha(data) == bindings[root / name] for name, data in entries.items()),
            "Packaging input changed while collecting the bundle")
    entries["FORMAT.md"] = regular_file(format_path)
    require(sha(entries["FORMAT.md"]) == bindings[format_path],
            "Packaging format documentation changed while collecting inputs")
    registry = json.loads(entries["series-cast.json"])
    validate_registry(registry)
    no_credentials(registry)
    require(len(registry["books"]) == 7, "Registry must contain exactly seven published books")
    source_order = []
    states = []
    for number in range(1, 8):
        folder = root / f"book-{number:02d}"
        document = json.loads(regular_file(folder / "analysis.json"))
        source_hash, cast_hash = fingerprint(document["source"]), fingerprint(document["cast"])
        source_order.append(source_hash)
        audit = json.loads(regular_file(final_audits[number]))
        require(audit.get("status") in {"complete", "complete_with_review_needed"},
                f"Book {number:02d} final source audit is not complete")
        analysis_key = "analysis_fingerprint" if "analysis_fingerprint" in audit else "reviewed_analysis_fingerprint"
        require(audit.get("source_fingerprint") == source_hash and audit.get(analysis_key) == fingerprint(document),
                f"Book {number:02d} final source audit is stale or unbound")
        for key, value in (("cast_fingerprint", cast_hash),
                           ("script_fingerprint", fingerprint(document["segments"])),
                           ("segments_fingerprint", fingerprint(document["segments"]))):
            require(key not in audit or audit[key] == value, f"Book {number:02d} final audit has a stale component binding")
        published = registry["books"][number - 1]
        require(published["source_fingerprint"] == source_hash and published["cast"] == document["cast"]
                and published["book"] == document["source"]["book"],
                f"Book {number:02d} registry snapshot differs from current source/cast")
        require(published.get("issues") == document["issues"] and published.get("readiness") == readiness(document),
                f"Book {number:02d} registry review state differs from current analysis")
        continuity = audit_book(root, number)
        require(continuity.get("passed_mechanical_checks") is True and not continuity.get("errors")
                and continuity.get("bundle", {}).get("present") is True,
                f"Book {number:02d} continuity or bundle checks failed")
        prefix = f"book-{number:02d}/bundle/"
        validate_native_bundle(entries, prefix, document)
        states.append((number, readiness(document)))
    require(len(set(source_order)) == 7, "Seven distinct ordered books are required")
    fresh_report, fresh_csv, status = summarize(root)
    require(status == 0, "Recomputed series report has failures or incomplete publication")
    require(stable_report(entries["series-report.txt"].decode("utf-8")) == stable_report(fresh_report),
            "Saved series report is stale; regenerate it before packaging")
    require(entries["voice-continuity.csv"] == fresh_csv.encode("utf-8"),
            "Saved voice continuity table is stale; regenerate it before packaging")
    lines = [
        "SOULSHIP: SEVEN-BOOK TTS HANDOFF", "",
        "This archive contains engine-independent text, casting directions and speaker tags.",
        "It contains no generated audio or .m4b audiobook.", "",
        "Read FORMAT.md first. Each book-NN/bundle/ contains a native manifest, source,",
        "cast, review list and ordered chapter scripts. Read files.scripts in manifest order.",
        "Each script row keeps its global sequence and source block offsets; chapter",
        "sequence ranges use an inclusive start and exclusive end. Empty chapters",
        "have an explicit empty script file. Bundle bytes are preserved unchanged.",
        "Native audit metadata may reference the original workspace; those references",
        "do not include provider configuration files, caches, logs or the source EPUBs.", "",
        "Treat these as drafts requiring review, not human-approved literary assignments.",
        "Check manifest readiness and review.json before production synthesis. Null role/",
        "voice IDs and review_required flags remain unresolved; do not silently substitute",
        "a collective or guessed voice. Historical disagreement flags can remain even when",
        "a later reviewed passage changed; consult the retained provenance.", "",
        "Use a persistent voice_id -> engine voice mapping across the seven books.",
        "Stable character and voice IDs support continuity; role IDs are book-local.",
        "Names/aliases alone do not establish identity. Narration may use a separate voice.",
        "The series registry and voice-continuity.csv summarize the existing decisions;",
        "they do not grant new identity approvals or resolve outstanding review flags.", "",
        "Per-book mechanical readiness (does not certify perfect attribution):",
    ]
    lines.extend(f"  Book {number:02d}: ready={str(state['ready']).lower()}, review_items={state['review_count']}"
                 for number, state in states)
    lines += ["", "series-report.txt records the current checks and their limits.",
              "Keep the selected engine voice map separate from this portable analysis.", ""]
    entries["README.txt"] = "\n".join(lines).encode("utf-8")
    archive_inventory(entries)
    verify_inputs(bindings)
    return entries, bindings


def write_archive(entries, bindings, root, filename="soulship-tts-handoff-chapters.zip"):
    """Stage and CRC/hash-check, then atomically install without overwriting."""
    root = Path(root).absolute()
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.parents),
            "Packaging refuses symlink output roots")
    require(isinstance(filename, str) and filename == Path(filename).name and "/" not in filename
            and "\\" not in filename and filename.endswith(".zip"), "Archive name must be a simple .zip filename")
    allowlist = archive_inventory(entries)
    destination = root / filename
    require(not destination.exists() and not destination.is_symlink(), "Archive already exists; refusing to replace it")
    verify_inputs(bindings)
    descriptor, stage_name = tempfile.mkstemp(prefix=".package-results-", suffix=".zip.tmp", dir=root)
    stage = Path(stage_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name in allowlist:
                    archive.writestr(name, entries[name])
            stream.flush()
            os.fsync(stream.fileno())
        with zipfile.ZipFile(stage) as archive:
            require(archive.namelist() == list(allowlist), "Archive member inventory changed")
            require(archive.testzip() is None, "Archive CRC check failed")
            for name in allowlist:
                require(sha(archive.read(name)) == sha(entries[name]), "Archive member hash check failed")
        verify_inputs(bindings)
        # Same-directory hardlink publishes complete bytes atomically and fails
        # if another writer has occupied the name; rename could overwrite it.
        os.link(stage, destination)
        return {"filename": filename, "members": len(allowlist), "sha256": sha(stage.read_bytes())}
    finally:
        stage.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing legacy seven-book run directory")
    parser.add_argument("--output", default="soulship-tts-handoff-chapters.zip", help="New .zip filename under the run root")
    args = parser.parse_args()
    entries, bindings = collect_entries(args.root)
    print(json.dumps(write_archive(entries, bindings, args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
