"""Fresh, explicitly resumable attribution audits with post-audit decisions."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import re
import sys
import uuid
from typing import Literal, TypeAlias, assert_never

from .analysis_directory import _completed_directory
from .analysis_progress import analysis_cache_directory
from .analysis_revision import related_directories, reserve_revision_scope, revise_directories
from .attribution_manual_review import edit_findings, print_finding
from .attribution_review import audit_attribution
from .attribution_verification import chapter_block_ids, findings, risk_blocks, sample_blocks
from .pipeline import CachedRequests


VerificationAction: TypeAlias = Literal["apply", "edit", "leave"]


def run_attribution_verification(args, config, client_factory, progress):
    """Run one audit; parsed CLI input, configuration, and I/O factories stay at the boundary."""
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
    risk = risk_blocks(document)
    if not risk:
        print("No context-basis or flagged dialogue lines to verify.")
        print("No changes applied.")
        return 0
    selected = risk
    if args.chapter is not None or args.chapters is not None:
        try:
            numbers = ([args.chapter] if args.chapter is not None else
                       [int(part.strip()) for part in args.chapters.split(",")])
        except ValueError:
            raise ValueError("--chapters must be a comma-separated list of positive chapter numbers") from None
        if not numbers or any(type(number) is not int or number < 1 for number in numbers):
            raise ValueError("--chapter/--chapters must contain positive chapter numbers")
        chapters = document["source"]["chapters"]
        unknown = [number for number in numbers if number > len(chapters)]
        if unknown:
            raise ValueError("Unknown chapter number(s): " + ", ".join(str(number) for number in unknown))
        chapter_ids = [chapters[number - 1]["id"] for number in dict.fromkeys(numbers)]
        scoped = chapter_block_ids(document, chapter_ids)
        in_chapters = {block_id for block_ids in scoped.values() for block_id in block_ids}
        selected = [block_id for block_id in risk if block_id in in_chapters]
        print(f"Chapter selection: {sum(len(block_ids) for block_ids in scoped.values())} block(s); "
              f"{len(selected)} risk-set block(s).")
        if not selected:
            print("No risk-set lines in the selected chapter(s).")
            print("No changes applied.")
            return 0
    elif args.sample is not None:
        selected = sample_blocks(risk, args.sample)
    block_chapters = {block["id"]: block["chapter_id"] for block in document["source"]["blocks"]}
    covered = list(dict.fromkeys(block_chapters[block_id] for block_id in selected))
    strategy = document["provenance"].get("strategy", "consensus")
    print(f"Verification selection: strategy={strategy}; {len(selected)}/{len(document['source']['blocks'])} "
          f"blocks across {len(covered)} chapter(s); instructions={'yes' if instructions else 'no'}.")
    # A write-capable invocation secures the whole publication scope before any
    # model work: stale revision locks fail here, and later renders refuse
    # themselves through the .analysis-update.lock handshake, so Accept/Edit can
    # never hit a late lock. Live renderers are never a blocker for an
    # attribution save; the running render keeps its loaded snapshot.
    write_capable = bool(args.interactive or args.apply
                         or (sys.stdin.isatty() and sys.stdout.isatty()))
    with ExitStack() as stack:
        if write_capable:
            reserved = stack.enter_context(reserve_revision_scope([directory]))
            print(f"Publication ownership reserved before verification: {len(reserved)} book(s) in the "
                  f"update chain are locked against concurrent analysis revisions.", file=sys.stderr)
        else:
            reserved = {}
            print("Report-only invocation: no publication ownership is taken and no changes can be applied.",
                  file=sys.stderr)
        cache_root = analysis_cache_directory(directory, settings["cache_dir"]) / "attribution-verification"
        run_id = args.resume if args.resume is not None else uuid.uuid4().hex
        model_cache = cache_root / run_id
        if args.resume is None:
            model_cache.mkdir(parents=True, exist_ok=False)
            print(f"Fresh verification run: {run_id}", file=sys.stderr)
            print("No earlier verification run is reused.", file=sys.stderr)
            print(f"To resume this run explicitly after interruption, use --resume {run_id}.", file=sys.stderr)
        else:
            if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
                raise ValueError("--resume must be a verification run ID printed by an earlier invocation")
            if not model_cache.is_dir():
                raise ValueError(f"Unknown verification run ID for this book: {run_id}")
            print(f"Resuming verification run: {run_id}", file=sys.stderr)
            print("Only compatible completed, validated responses from this explicitly selected run are reused.",
                  file=sys.stderr)
        print(f"Verification request cache: {model_cache}", file=sys.stderr)

        client = client_factory(config["provider"])
        print(f"Verifying attribution with {client.model} at {client.base_url}; selected book passages will be sent to this endpoint.",
              file=sys.stderr)
        requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        audited = audit_attribution(document, requests, workers=args.workers, review_notes=instructions,
                                    target_block_ids=selected, progress=print)
        found = findings(document, audited)

        def publish(candidate):
            try:
                directories = related_directories([directory])
                hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                          for path in directories}
                if hashes[str(directory)] != original_manifest_hash:
                    raise ValueError(f"Analysis changed while attribution was being reviewed: {directory}")
                result = revise_directories({directory: candidate}, directories=directories,
                                            expected_manifest_hashes=hashes, scope="attribution", progress=print,
                                            reserved_update_locks=reserved or None)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Attribution publication did not succeed: {exc} The completed audit remains cached; "
                    f"republish the same findings without new verification calls by rerunning this command "
                    f"with --resume {run_id}.") from exc
            print(f"Original analysis files and revision journal: {result['backup_directory']}")

        def outcome(mode, applied):
            left = len(found) - applied
            if applied:
                print(f"{mode}: {applied} applied / {left} left unchanged.")
            else:
                print(f"{mode}: No changes applied; {left} finding(s) left unchanged.")

        for index, finding in enumerate(found, 1):
            print_finding(finding, index)
        print(f"{len(found)} attribution finding(s).")
        if not found:
            print("No attribution corrections or new review flags. No changes applied.")
            return 0

        action: VerificationAction
        if args.interactive:
            action = "edit"
        elif args.apply:
            action = "apply"
        elif not (sys.stdin.isatty() and sys.stdout.isatty()):
            outcome("Report only", 0)
            return 0
        else:
            actions: dict[str, VerificationAction] = {"a": "apply", "e": "edit", "l": "leave", "q": "leave"}
            while True:
                print("[A] accept all / [E] edit-review individual corrections / [L] leave unchanged (quit): ",
                      end="", flush=True)
                try:
                    answer = sys.stdin.readline()
                except (EOFError, KeyboardInterrupt, OSError):
                    answer = ""
                if answer == "":
                    print("\nInput ended; leaving every finding unchanged.")
                    action = "leave"
                    break
                answer = answer.strip().casefold()
                if answer in actions:
                    action = actions[answer]
                    break
                print("Please enter A, E, or L.")

        match action:
            case "leave":
                outcome("Leave unchanged", 0)
                return 0
            case "apply":
                print(f"Applying all {len(found)} finding(s).")
                publish(audited)
                outcome("Accept all", len(found))
                return 0
            case "edit":
                pass
            case unreachable:
                assert_never(unreachable)

        revised, proposals, manual = edit_findings(document, audited, found)
        applied = proposals + manual
        if applied:
            publish(revised)
        outcome("Edit-review", applied)
        print(f"  {proposals} accepted proposal(s); {manual} manual edit(s).")
        return 0
