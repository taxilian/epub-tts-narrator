"""`refresh-directions`: re-direct a published book's lines, keeping its speakers."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
import re
import sys
import uuid

from .analysis_directory import _completed_directory
from .analysis_progress import analysis_cache_directory
from .analysis_revision import related_directories, reserve_revision_scope, revise_directories
from .attribution_verification import chapter_block_ids
from .direction_refresh import refresh_directions
from .pipeline import CachedRequests


EXAMPLES = 8


def _chapter_ids(args, document):
    if args.chapter is None and args.chapters is None:
        return None
    try:
        numbers = [args.chapter] if args.chapter is not None else [int(part) for part in args.chapters.split(",")]
    except ValueError:
        raise ValueError("--chapters must be a comma-separated list of positive chapter numbers") from None
    if not numbers or any(type(number) is not int or number < 1 for number in numbers):
        raise ValueError("--chapter/--chapters must contain positive chapter numbers")
    chapters = document["source"]["chapters"]
    unknown = [number for number in numbers if number > len(chapters)]
    if unknown:
        raise ValueError("Unknown chapter number(s): " + ", ".join(str(number) for number in unknown))
    return [chapters[number - 1]["id"] for number in dict.fromkeys(numbers)]


def _ask_apply():
    while True:
        print("[A] apply the new directions / [L] leave the book unchanged: ", end="", flush=True)
        try:
            answer = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt, OSError):
            answer = ""
        if answer == "":
            print("\nInput ended; leaving the book unchanged.")
            return False
        answer = answer.strip().casefold()
        if answer in {"a", "l", "q"}:
            return answer == "a"
        print("Please enter A or L.")


def run_direction_refresh(args, config, client_factory, progress):
    """Refresh directions; parsed CLI input, configuration and I/O factories stay at the boundary."""
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
    if type(args.workers) is not int or not 1 <= args.workers <= 5:
        raise ValueError("workers must be an integer from 1 to 5")
    instructions = args.instructions or ""
    if args.instructions is not None and not 1 <= len(instructions) <= 2000:
        raise ValueError("--instructions must contain from 1 to 2000 characters")
    directory = args.directory.resolve()
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != original_manifest_hash:
        raise ValueError(f"Analysis changed while loading the saved book: {directory}")
    chapter_ids = _chapter_ids(args, document)
    block_ids = (None if chapter_ids is None else
                 [block_id for ids in chapter_block_ids(document, chapter_ids).values() for block_id in ids])
    chapters = len(chapter_ids) if chapter_ids is not None else len(document["source"]["chapters"])
    print(f"Direction refresh: strategy={document['provenance'].get('strategy', 'consensus')}; {chapters} "
          f"chapter(s); instructions={'yes' if instructions else 'no'}. Speakers, cast and voices stay as they are.")
    write_capable = bool(args.apply or (sys.stdin.isatty() and sys.stdout.isatty()))
    with ExitStack() as stack:
        if write_capable:
            reserved = stack.enter_context(reserve_revision_scope([directory]))
            print(f"Publication ownership reserved before the refresh: {len(reserved)} book(s) in the update chain "
                  f"are locked against concurrent analysis revisions.", file=sys.stderr)
        else:
            reserved = {}
            print("Report-only invocation: no publication ownership is taken and the book is not changed.",
                  file=sys.stderr)
        # Every run asks afresh: directions are a performance choice worth
        # retrying. A run keeps its own responses only so an interrupted run
        # can be finished with an explicit --resume RUN_ID.
        cache_root = analysis_cache_directory(directory, settings["cache_dir"]) / "direction-refresh"
        run_id = args.resume if args.resume is not None else uuid.uuid4().hex
        cache = cache_root / run_id
        if args.resume is None:
            cache.mkdir(parents=True, exist_ok=False)
            print(f"Fresh direction run: {run_id}", file=sys.stderr)
            print(f"To finish this run after an interruption, use --resume {run_id}.", file=sys.stderr)
        else:
            if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
                raise ValueError("--resume must be a direction run ID printed by an earlier invocation")
            if not cache.is_dir():
                raise ValueError(f"Unknown direction run ID for this book: {run_id}")
            print(f"Resuming direction run: {run_id}", file=sys.stderr)
        client = client_factory(config["provider"])
        print(f"Directing lines with {client.model} at {client.base_url}; selected book passages will be sent to "
              f"this endpoint.", file=sys.stderr)
        requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        revised, report = refresh_directions(document, requests, block_ids=block_ids, notes=instructions,
                                             workers=args.workers, progress=progress)
        print(f"{report['directed']}/{report['lines']} spoken line(s) directed; {report['changes']} mid-line "
              f"change(s); {report['narration_reads']} narration direction(s); {report['italic_emphasis']} "
              f"italic emphasis; {report['notes_attached']} other in-line note(s) proposed, "
              f"{report['review'].get('kept', 0)} kept by review.")
        if report["kept_pronunciation_rows"]:
            print(f"{report['kept_pronunciation_rows']} line(s) with pronunciation controls kept them and took "
                  f"no new in-line notes.")
        old = {segment["id"]: segment["delivery"]["direction"] for segment in document["segments"]}
        shown = 0
        for segment in revised["segments"]:
            if shown >= EXAMPLES or segment["kind"] not in {"dialogue", "thought", "letter"}:
                continue
            if old.get(segment["id"]) != segment["delivery"]["direction"]:
                print(f"  {segment['id']}: {old.get(segment['id']) or '(none)'} -> "
                      f"{segment['delivery']['direction'] or '(none)'}")
                shown += 1
        print(f"{report['changed_segments']} segment(s) would change.")
        if not report["changed_segments"]:
            print("No changes applied.")
            return 0
        if not args.apply:
            if not write_capable:
                print("Report only: no changes applied. Run it in a terminal to be asked, or pass --apply.")
                return 0
            if not _ask_apply():
                print("No changes applied.")
                return 0
        _publish(directory, revised, report, client, chapter_ids, original_manifest_hash, reserved,
                 f"rerunning with --resume {run_id}")
        return 0


def _publish(directory, revised, report, client, chapter_ids, original_manifest_hash, reserved, retry):
    revised["provenance"].setdefault("direction_refreshes", []).append(
        {**report, "provider": client.identity, "chapters": chapter_ids})
    try:
        directories = related_directories([directory])
        hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                  for path in directories}
        if hashes[str(directory)] != original_manifest_hash:
            raise ValueError(f"Analysis changed while directions were being refreshed: {directory}")
        result = revise_directories({directory: revised}, directories=directories,
                                    expected_manifest_hashes=hashes, scope="attribution", progress=print,
                                    reserved_update_locks=reserved or None)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Direction publication did not succeed: {exc} This run's responses remain; "
                         f"publish them without new requests by {retry}.") from exc
    print(f"Original analysis files and revision journal: {result['backup_directory']}")
    print(f"Applied: {report['changed_segments']} segment(s) changed. Their audio renders again on the next "
          f"render of this book.")


def refresh_outdated_directions(directory, config, client_factory, progress):
    """``analyze BOOK``: refresh the whole book's directions when an older version of the rules made them.

    Returns True when it published new directions. Directions a person typed
    are kept. The run ID comes from the saved book and the direction rules,
    so rerunning ``analyze`` after an interruption replays this run's
    responses instead of asking again; it is also accepted by
    ``refresh-directions --resume``.
    """
    from .direction_policy import VERSION, consensus_only, outdated
    directory = Path(directory).resolve()
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    if not outdated(document):
        if consensus_only(document):
            print("Line directions come from the consensus analysis; run refresh-directions on this book to "
                  "direct its lines with the narrator's rules (reading the narration's cues for each line).")
        return False
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("This book's line directions were made with older rules and are refreshed by analyze; "
                         "set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
    print("This book's line directions were made with older direction rules; directing every line again "
          "(speakers, cast and voices stay as they are; directions a person typed are kept).")
    with ExitStack() as stack:
        reserved = stack.enter_context(reserve_revision_scope([directory]))
        run_id = hashlib.sha256(f"analyze-directions:{VERSION}:{original_manifest_hash}".encode()).hexdigest()[:32]
        cache = analysis_cache_directory(directory, settings["cache_dir"]) / "direction-refresh" / run_id
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Direction run: {run_id}", file=sys.stderr)
        client = client_factory(config["provider"])
        print(f"Directing lines with {client.model} at {client.base_url}; book passages will be sent to "
              f"this endpoint.", file=sys.stderr)
        requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        revised, report = refresh_directions(document, requests, workers=min(5, settings["attribution_workers"]),
                                             progress=progress)
        print(f"{report['directed']}/{report['lines']} spoken line(s) directed; {report['changes']} mid-line "
              f"change(s); {report['narration_reads']} narration direction(s); {report['kept_typed_directions']} "
              f"typed direction(s) kept.")
        # Even with nothing changed, the record says these directions now follow the current rules.
        _publish(directory, revised, report, client, None, original_manifest_hash, reserved, "rerunning analyze")
    return True
