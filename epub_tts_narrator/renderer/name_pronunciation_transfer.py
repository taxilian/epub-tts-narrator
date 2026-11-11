"""`pronounce BOOK --export FILE` / `--import FILE`: move confirmed choices between books."""

from pathlib import Path

from epub_tts_narrator.name_pronunciation_yaml import export_file, import_file


def transfer_command(bundle, args):
    """Export (returns an exit code) or import (returns None: tagging the uses follows)."""
    if args.name is not None or args.sound is not None or args.phonemes or args.parts_of_speech:
        raise ValueError("--export and --import take no WORD, HOW_IT_SOUNDS, --phonemes or --as")
    directory = Path(bundle["directory"])
    if args.export:
        if args.replace:
            raise ValueError("--replace applies only to --import")
        count = export_file(directory, args.export, inherited=not args.local_only)
        print(f"Exported {count} confirmed pronunciation(s) to {args.export}.")
        return 0
    if args.local_only:
        raise ValueError("--local-only applies only to --export")
    outcome = import_file(directory, args.import_file, replace=args.replace)
    labels = {"added": "added", "overrode_inherited": "now chosen here instead of inherited",
              "replaced": "replaced", "unchanged": "already the same", "conflict": "kept this book's own choice"}
    for key, label in labels.items():
        if outcome[key]:
            print(f"{len(outcome[key])} {label}: {', '.join(outcome[key])}")
    if outcome["conflict"]:
        print("Rerun with --replace to use the file's choice for those word(s).")
    if not any(outcome[key] for key in ("added", "overrode_inherited", "replaced")):
        print("No pronunciations changed.")
    print("Tagging the uses of every confirmed word in this book next.")
    return None
