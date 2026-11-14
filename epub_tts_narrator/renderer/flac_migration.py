"""Convert a book's kept render WAVs to FLAC (``audiobook-render convert-flac``, ``just convert-flac``).

Each WAV becomes the ``stored_audio`` FLAC beside it, which restores the WAV
byte for byte; the WAV is removed only after the finished FLAC has been
decoded back to identical bytes. Records, checksums and every cache, voice
and request identity therefore stay exactly as they were, and nothing is
regenerated. WAVs FLAC cannot hold exactly (the engines' 32-bit float raw
output) stay WAV and are listed as kept.

Groups, all on by default; ``--only`` picks a subset:

- ``clips``: speech takes, selections, recoveries and raw audio under
  ``render-work/clips``.
- ``chapters``: lossless chapter audio under ``render-work/chapters``.
- ``mixes``: simultaneous-speaker mixes under ``render-work/mixes``.
- ``previews``: loudness-levelled listening copies under
  ``render-work/voice-previews``.

Casting takes and voice reference recordings are not converted: engines,
engine plugins and remote workers read a voice's reference WAV by its path.

Never traverses the published output, never follows symlinks and skips
hidden staging files. Holds the same locks as ``prune``, so it refuses while
a render, assembly, casting session or cleanup uses the book. An interrupted
conversion resumes on rerun: each file is converted on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import stored_audio


GROUPS = ("clips", "chapters", "mixes", "previews")
_WORK_GROUPS = {"clips": ("clips",), "chapters": ("chapters",), "mixes": ("mixes",),
                "previews": ("voice-previews",)}


def parse_groups(text):
    """``None`` → every group; else a comma-separated subset, in canonical order."""
    if text is None:
        return GROUPS
    names = {part.strip() for part in text.split(",") if part.strip()}
    if not names or names - set(GROUPS):
        raise ValueError(f"--only takes one or more of {', '.join(GROUPS)}, separated by commas")
    return tuple(group for group in GROUPS if group in names)


def _wavs(root):
    """Every visible regular ``.wav`` below ``root``, never following symlinks or entering hidden staging."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        return []
    found, pending = [], [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.name.startswith(".") or entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(".wav"):
                    found.append(Path(entry.path))
    return sorted(found)


def _candidates(work, groups):
    for group in groups:
        for name in _WORK_GROUPS[group]:
            for path in _wavs(Path(work) / name):
                yield group, path


def convert_book(bundle, config, *, groups=GROUPS, dry_run=False, confirm=None, progress=None):
    """Convert the selected groups' WAVs; return the report (``converted`` says whether it did).

    ``confirm(report)`` sees the dry-run report and returns True to convert;
    without it, conversion follows ``dry_run``. ``progress(done, total)`` is
    called after each file.
    """
    from .book_context import resolve_book_config
    from .run_locks import acquire_directory_lock
    from .workflow import RenderBlocked, _lock_book_audio
    groups = tuple(group for group in GROUPS if group in groups)
    config = resolve_book_config(bundle, config)
    book = Path(bundle["directory"]).resolve()
    work = Path(config["render"]["work_dir"])
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise RenderBlocked("Conversion work directory must not use symlinks")
    locks = []
    busy = ("Another render, assembly, casting session or cleanup is using this book; "
            "retry after it finishes: {}")
    try:
        if work.is_dir():
            try:
                locks.append(acquire_directory_lock(work / ".render.lock"))
            except FileExistsError as exc:
                raise RenderBlocked(busy.format(work / ".render.lock")) from exc
        locks.append(_lock_book_audio(bundle))
        files = list(_candidates(work, groups))
        report = _report(book, groups, files, dry_run=True)
        convert = (not dry_run and bool(files)
                   and (confirm is None or confirm(report)))
        if not convert:
            return dict(report, converted=False)
        return dict(_convert(book, groups, files, progress), converted=True)
    finally:
        for lock in reversed(locks):
            lock.rmdir()


def _label(book, path):
    try:
        return str(Path(path).resolve().relative_to(book))
    except (ValueError, OSError):
        return str(path)


def _report(book, groups, files, *, dry_run):
    summary = {group: {"files": 0, "bytes": 0} for group in groups}
    for group, path in files:
        summary[group]["files"] += 1
        summary[group]["bytes"] += path.stat().st_size
    return {"book": str(book), "dry_run": dry_run, "groups": summary,
            "files": sum(entry["files"] for entry in summary.values()),
            "bytes": sum(entry["bytes"] for entry in summary.values())}


def _convert(book, groups, files, progress):
    summary = {group: {"converted": 0, "kept": 0, "failed": 0, "wav_bytes": 0, "flac_bytes": 0}
               for group in groups}
    kept, failed = [], []
    for done, (group, path) in enumerate(files, 1):
        entry = summary[group]
        try:
            size = path.stat().st_size
            stored = stored_audio.store(path)
        except (stored_audio.StoredAudioError, OSError, ValueError, RuntimeError) as exc:
            entry["failed"] += 1
            failed.append({"path": _label(book, path), "error": str(exc)})
        else:
            if stored is None:
                entry["kept"] += 1
                kept.append(_label(book, path))
            else:
                entry["converted"] += 1
                entry["wav_bytes"] += size
                entry["flac_bytes"] += stored.stat().st_size
        if progress is not None:
            progress(done, len(files))
    totals = {"files_" + key if key == "converted" else key: sum(entry[key] for entry in summary.values())
              for key in ("converted", "kept", "failed", "wav_bytes", "flac_bytes")}
    return {"book": str(book), "dry_run": False, "groups": summary, **totals,
            "kept_files": kept, "failed_files": failed}


def _size(count):
    for unit in ("bytes", "KiB", "MiB", "GiB"):
        if count < 1024 or unit == "GiB":
            return f"{count:.0f} {unit}" if unit == "bytes" else f"{count:.1f} {unit}"
        count /= 1024


def summary_lines(report):
    lines = [f"Book: {report['book']}"]
    if not report.get("converted"):
        for group, entry in report["groups"].items():
            lines.append(f"  {group:<9}{entry['files']:>7} WAV file(s) {_size(entry['bytes']):>11}")
        lines.append(f"  Total {report['files']} WAV file(s), {_size(report['bytes'])}; "
                     "FLAC usually needs a third to a half of that.")
        return lines
    for group, entry in report["groups"].items():
        lines.append(f"  {group:<9}{entry['converted']:>7} converted {_size(entry['wav_bytes']):>11} → "
                     f"{_size(entry['flac_bytes'])}"
                     + (f", {entry['kept']} kept as WAV" if entry["kept"] else "")
                     + (f", {entry['failed']} failed" if entry["failed"] else ""))
    lines.append(f"  Total {report['files_converted']} file(s): {_size(report['wav_bytes'])} → "
                 f"{_size(report['flac_bytes'])}, freeing about "
                 f"{_size(max(report['wav_bytes'] - report['flac_bytes'], 0))}.")
    if report["kept"]:
        lines.append(f"  Kept {report['kept']} WAV file(s) FLAC cannot hold exactly (--json lists them).")
    for item in report["failed_files"][:10]:
        lines.append(f"  Failed, WAV kept: {item['path']}: {item['error']}")
    return lines


def convert_command(args, config):
    import json
    import sys
    from .book_context import resolve_book_config
    from .bundle import load_bundle
    books = args.bundle
    if len(books) > 1 and (args.work_dir or args.output or args.epub):
        raise ValueError("--work-dir, --output and --epub apply to one book; name a single book with them")
    groups = parse_groups(args.only)
    asking = not args.yes and not args.dry_run
    interactive = sys.stdin.isatty()
    say = sys.stderr if args.json else sys.stdout
    failed = 0
    for path in books:
        try:
            bundle = load_bundle(path)
            book_config = resolve_book_config(bundle, config, output=args.output,
                                              work_dir=args.work_dir, epub=args.epub)

            def confirm(report):
                for line in summary_lines(report):
                    print(line, file=say)
                if not interactive:
                    print("  Not converted: no terminal to confirm on. Rerun with --yes to convert.", file=say)
                    return False
                print("Convert these WAV files to FLAC? [y/N] ", end="", file=say, flush=True)
                return sys.stdin.readline().strip().lower() in {"y", "yes"}

            def progress(done, total):
                if done == total or done % 500 == 0:
                    print(f"  Converted {done} of {total}", file=say, flush=True)

            report = convert_book(bundle, book_config, groups=groups, dry_run=args.dry_run,
                                  confirm=confirm if asking else None, progress=progress)
        except (ValueError, OSError) as exc:
            failed += 1
            print(f"{path}: {exc}", file=sys.stderr)
            continue
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["converted"] or not asking or not report["files"]:
            for line in summary_lines(report):
                print(line, file=say)
        if report["converted"]:
            failed += bool(report["failed"])
        elif args.dry_run:
            print("  Dry run: nothing was converted.", file=say)
        elif not report["files"]:
            print("  No WAV files to convert.", file=say)
        else:
            print("  Nothing was converted.", file=say)
    return 2 if failed else 0
