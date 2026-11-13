"""Command-line boundary: extraction, analysis, corrections, and TTS handoff."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from . import __version__
from .config import load_config, resolve_config_path
from .contracts import readiness, review_items, validate_document
from .epub import extract_epub
from .source_epub import SOURCE_EPUB_HINT, copy_source_epub_hint, remember_source_epub
from .pipeline import (CachedRequests, analyze, apply_edits, check_cast, fingerprint,
                       seed_cast, write_json)
from .attribution_review import audit_attribution
from .speaker_decisions import prompt_speaker_decisions
from .jev import JevClient
from .provider import OpenAICompatibleClient, ProviderError, set_default_stream_observer
from .stream_progress import StreamConsole
from .series import link_analysis, load_registry, save_registry
from .analysis_directory import (
    default_output, validate_output, continuation_registry, fresh_registry,
    publish_directory, continuation_directories,
)
from .analysis_progress import (
    STAGES, analysis_cache_directory, analysis_options_fingerprint, create_pending,
    is_pending_directory, pending_identity, resume_document,
)


def read_json(path):
    def reject_constant(value):
        raise ValueError("Non-finite numbers are not valid JSON")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def _speaker_completion_status(document):
    """Advisory notes need no approval; missing performers need a real choice."""
    from .speaker_completion import unassigned_segments
    remaining = len(unassigned_segments(document))
    if remaining:
        print(f"Speaker selection incomplete: {remaining} spoken passage(s) still need a voice. "
              "The book and completed decisions are saved. Audio generation cannot use these "
              "passages until their speakers are selected. Exiting with status 2.", file=sys.stderr)
        return 2
    print("All spoken passages have assigned voices. No review step is required.")
    return 0


def _speaker_judge(config):
    """Reuse optional global JEV settings; constructing a client sends no text."""
    settings = config.get("jev", {})
    if not settings.get("enabled", False):
        return None
    return JevClient(settings["api_key"], model=settings["model"],
                     timeout_seconds=settings["timeout_seconds"])


def _run_directory_command(args):
    """Dispatch completed-book commands that need no shared pipeline state."""
    if args.command == "review-ui":
        from .review_tui import run_review
        config = load_config(args.config)
        player_command = config.get("casting", {}).get("player", ["/usr/bin/afplay"])
        run_review(args.directory.resolve(), player_command=player_command)
        return 0
    from .voice_profile_refresh import refresh_book_voice_profiles
    refresh_book_voice_profiles(args.directory, args.config, workers=args.workers)
    return 0


def read_analysis(path):
    document = read_json(path)
    errors = validate_document(document)
    if errors:
        raise ValueError("Invalid analysis: " + "; ".join(errors[:12]))
    return document


def source_from(path, include_auxiliary=False):
    if Path(path).suffix.lower() == ".epub":
        source = extract_epub(path, include_auxiliary=include_auxiliary)
    else:
        wrapper = read_json(path)
        if not isinstance(wrapper, dict) or wrapper.get("format") != "epub-audiobook-source" or wrapper.get("version") != 1:
            raise ValueError("Input must be an EPUB or a source JSON file made by extract")
        if include_auxiliary:
            raise ValueError("extraction.include_auxiliary applies when extracting an EPUB, not to a saved source")
        source = wrapper.get("source")
    if not isinstance(source, dict):
        raise ValueError("Invalid extracted source")
    check_cast(source, seed_cast())
    return source


_CHAPTER_NUMBER = (
    r"(?:[0-9]+|[IVXLCDM]+|"
    r"ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|"
    r"THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|"
    r"(?:TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY)"
    r"(?:[ -](?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE))?)"
)
_CHAPTER_LABEL = re.compile(rf"CHAPTER\s+{_CHAPTER_NUMBER}\s*[:.]?", re.IGNORECASE)


def script_partitions(document, *, prefer_source_titles=True):
    """Find conservative display chapters without changing source coordinates.

    Source chapters are EPUB resources, which can contain several story chapters.
    Only standalone numbered headings with structural/annotation support create
    extra boundaries. Resources resembling a contents list remain one section.
    """
    blocks = {block["id"]: block for block in document["source"]["blocks"]}
    kinds = {}
    for segment in document["segments"]:
        kinds.setdefault(segment["block_id"], set()).add(segment["kind"])

    def is_heading(block):
        return (re.fullmatch(r"h[1-6]", block["source"]["tag"], re.IGNORECASE) is not None
                or kinds.get(block["id"]) == {"heading"})

    def is_label(block):
        return _CHAPTER_LABEL.fullmatch(block["text"].strip()) is not None

    def has_prose(items):
        return any(block["text"].strip() and not is_label(block) and not is_heading(block)
                   for block in items)

    result = []
    for chapter in document["source"]["chapters"]:
        chapter_blocks = [blocks[block_id] for block_id in chapter["block_ids"]]
        labels = [index for index, block in enumerate(chapter_blocks)
                  if is_label(block) and is_heading(block)]
        list_like = len(labels) >= 2 and (
            len(labels) / len(chapter_blocks) >= 0.75
            or not any(has_prose(chapter_blocks[left + 1:right])
                       for left, right in zip(labels, labels[1:])))
        starts = [0]
        headings = {}
        if labels and not list_like:
            first = labels[0]
            if first and has_prose(chapter_blocks[:first]):
                starts.append(first)
                headings[first] = chapter_blocks[first]
            else:
                headings[0] = chapter_blocks[first]
            for left, right in zip(labels, labels[1:]):
                if has_prose(chapter_blocks[left + 1:right]):
                    starts.append(right)
                    headings[right] = chapter_blocks[right]
        for part, start in enumerate(starts):
            end = starts[part + 1] if part + 1 < len(starts) else len(chapter_blocks)
            heading = headings.get(start)
            title = chapter["title"]
            if list_like:
                title = "Contents"
            elif prefer_source_titles and part == 0 and title.strip().casefold() not in {"", "unknown", "untitled"}:
                heading = None
            elif heading is not None:
                title = heading["text"]
            elif title.strip().casefold() in {"", "unknown", "untitled"}:
                first_heading = next((block for block in chapter_blocks[start:end]
                                      if is_heading(block)), None)
                if first_heading is not None:
                    heading = first_heading
                    title = heading["text"]
            result.append({
                "chapter_id": chapter["id"], "title": title,
                "heading_block_id": heading["id"] if heading is not None else None,
                "block_ids": [block["id"] for block in chapter_blocks[start:end]],
            })
    return result


def export_bundle(document, output, *, allow_review=False, source_epub=None, source_epub_from=None):
    if source_epub is not None and source_epub_from is not None:
        raise ValueError("Specify an original EPUB or a source hint directory, not both")
    state = readiness(document)
    if state["errors"]:
        raise ValueError("Invalid analysis: " + "; ".join(state["errors"][:12]))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir()
    staging = output / ".export-staging"
    staging.mkdir()
    published = []
    try:
        write_json(staging / "source.json", document["source"])
        write_json(staging / "cast.json", document["cast"])
        write_json(staging / "review.json", review_items(document))
        blocks = {b["id"]: b for b in document["source"]["blocks"]}
        (staging / "scripts").mkdir()
        scripts = []
        segments = document["segments"]
        sequence = 0
        for chapter_number, chapter in enumerate(script_partitions(document), start=1):
            # Filenames use only the trusted ordinal, never EPUB titles or IDs.
            name = f"scripts/chapter-{chapter_number:04d}.jsonl"
            start = sequence
            chapter_blocks = set(chapter["block_ids"])
            with (staging / name).open("x", encoding="utf-8") as stream:
                while sequence < len(segments):
                    item = segments[sequence]
                    block = blocks[item["block_id"]]
                    if block["id"] not in chapter_blocks:
                        break
                    rendered = dict(item, sequence=sequence, chapter_id=block["chapter_id"],
                                    break_after="paragraph" if item["end"] == len(block["text"]) else "none")
                    if item["start"] == 0 and block.get("break_before"):
                        rendered["break_before"] = block["break_before"]
                    stream.write(json.dumps(rendered, ensure_ascii=False, allow_nan=False) + "\n")
                    sequence += 1
            scripts.append({
                "chapter_id": chapter["chapter_id"], "title": chapter["title"],
                "heading_block_id": chapter["heading_block_id"], "path": name,
                "segment_count": sequence - start,
                "sequence_start": start, "sequence_end": sequence,
            })
        names = ("source.json", "cast.json", "review.json", *(item["path"] for item in scripts))
        manifest = {
            "format": "epub-audiobook-bundle", "version": 2,
            "book": document["source"]["book"], "chapters": document["source"]["chapters"],
            "source_fingerprint": fingerprint(document["source"]), "readiness": state,
            "offsets": "Zero-based Unicode code points within source block text; end exclusive.",
            "files": {"source": "source.json", "cast": "cast.json", "review": "review.json",
                      "scripts": scripts},
            "sha256": {name: hashlib.sha256((staging / name).read_bytes()).hexdigest() for name in names},
            "provenance": document["provenance"],
        }
        write_json(staging / "manifest.json", manifest)
        if {path.name for path in output.iterdir()} - {staging.name}:
            raise FileExistsError(f"Output directory changed during export: {output}")
        hint_names = []
        if source_epub is not None:
            remember_source_epub(staging, source_epub, document["source"]["book"]["id"])
            hint_names.append(SOURCE_EPUB_HINT)
        elif source_epub_from is not None and copy_source_epub_hint(
                source_epub_from, staging, document["source"]["book"]["id"]):
            hint_names.append(SOURCE_EPUB_HINT)
        # Reserve the destination exclusively. POSIX rename() can replace a
        # concurrently created empty directory, so it is not a no-clobber commit.
        # Publish the manifest last: consumers must require it before reading.
        (output / "scripts").mkdir()
        for name in (*names, *hint_names, "manifest.json"):
            os.link(staging / name, output / name)
            published.append(output / name)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            (output / "scripts").rmdir()
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            output.rmdir()
        except OSError:
            pass
    return state


def parser():
    root = argparse.ArgumentParser(prog="epub-tts-narrator", description="Analyze an EPUB into a portable multi-voice audiobook script.")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)

    def input_command(name, help_text):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", type=Path)
        command.add_argument("--config", type=Path,
                             help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
        return command

    def output_options(command):
        command.add_argument("-o", "--output", type=Path, required=True)
        command.add_argument("--force", action="store_true", help="Replace an existing output file")

    extract = input_command("extract", "Extract and inspect source text without model calls")
    output_options(extract)
    plan = input_command("plan", "Show extraction counts and automatic analysis work estimates")
    plan.add_argument("-o", "--output", type=Path,
                      help="Proposed output directory; defaults to ./<input stem>.audiobook")
    run = input_command("analyze", "Analyze an EPUB, or resume unfinished speaker choices in an existing book directory")
    run.add_argument("-o", "--output", type=Path,
                     help="Output directory; defaults to ./<input stem>.audiobook. A .json path selects legacy file output.")
    run.add_argument("--force", action="store_true",
                     help="Replace legacy .json output only; completed directories are never overwritten")
    run.add_argument("--edits", type=Path, help="Reapply source-bound human corrections after analysis")
    run.add_argument("--strategy", choices=("consensus", "narrator"), default=None,
                     help="Analysis strategy: narrator (default for a new book) or consensus. "
                          "Narrator uses one sequential prep read, then terse chapter markup. "
                          "Updating a saved book from a new EPUB defaults to the strategy it was analyzed with")
    passage_review = input_command("review-attribution", "Review an existing script with source evidence and three-way consensus")
    output_options(passage_review)
    verification = commands.add_parser(
        "verify-attribution", help="Freshly re-read risky speaker assignments, then accept, edit-review, or leave findings unchanged")
    verification.add_argument("directory", type=Path, help="Completed analysis directory to verify")
    verification.add_argument("--config", type=Path,
                              help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
    selection = verification.add_mutually_exclusive_group()
    selection.add_argument("--chapters", help="Comma-separated source chapter numbers, such as 3,5")
    selection.add_argument("--chapter", type=int, help="One source chapter number")
    selection.add_argument("--sample", type=int, help="Deterministically sample N blocks from the risk set")
    verification.add_argument("--instructions", help="Review guidance passed to the attribution engine (1-2000 characters)")
    verification.add_argument("--apply", action="store_true",
                              help="Apply all findings without a post-audit prompt; publication ownership over the "
                                   "book and its descendants is reserved before any model request")
    verification.add_argument("--interactive", action="store_true",
                              help="Accept, manually edit, or leave each finding; takes precedence over --apply and "
                                   "reserves publication ownership before any model request")
    verification.add_argument("--resume", metavar="RUN_ID",
                              help="Explicitly resume one printed verification run; ordinary invocations always start fresh")
    verification.add_argument("--workers", type=int, default=3, help="Concurrent model requests, 1-5 (default 3)")
    directions = commands.add_parser(
        "refresh-directions",
        help="Rewrite how each line is performed in a published book, keeping its speakers, cast and voices")
    directions.add_argument("directory", type=Path, help="Completed analysis directory to update in place; originals are backed up")
    directions.add_argument("--config", type=Path,
                            help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
    direction_scope = directions.add_mutually_exclusive_group()
    direction_scope.add_argument("--chapters", help="Comma-separated source chapter numbers, such as 3,5")
    direction_scope.add_argument("--chapter", type=int, help="One source chapter number")
    directions.add_argument("--instructions", help="Performance guidance for the director (1-2000 characters)")
    directions.add_argument("--apply", action="store_true",
                            help="Publish the new directions without asking (a terminal run otherwise asks); publication ownership over the book "
                                 "and its descendants is reserved before any model request")
    directions.add_argument("--resume", metavar="RUN_ID",
                            help="Finish one printed direction run after an interruption; ordinary runs always ask afresh")
    directions.add_argument("--workers", type=int, default=3, help="Concurrent model requests, 1-5 (default 3)")
    review_ui = commands.add_parser(
        "review-ui", help="Browse and edit passages in a published book with an interactive terminal UI")
    review_ui.add_argument("directory", type=Path, help="Completed analysis directory to review")
    review_ui.add_argument("--config", type=Path,
                           help="Global TOML file (accepted for command consistency; review-ui makes no model calls)")
    speakers = commands.add_parser("resolve-speakers", help="Resolve unassigned passages by consensus and refresh existing book directories")
    speakers.add_argument("directories", nargs="+", type=Path, help="Analyzed book directories to repair in place; originals are backed up")
    speakers.add_argument("--config", type=Path, help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
    refresh_profiles = commands.add_parser(
        "refresh-voice-profiles",
        help="Refresh existing voice-profile descriptions, traits, and speechTraits in a completed book directory")
    refresh_profiles.add_argument("directory", type=Path,
                                  help="Completed analysis directory to update in place; originals are backed up")
    refresh_profiles.add_argument("--config", type=Path,
                                  help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
    refresh_profiles.add_argument("--workers", type=int, default=1,
                                  help="Concurrent model requests, 1-5 (default 1; capped by analysis.attribution_workers)")
    for command in (plan, run):
        command.add_argument("--continue-from", type=Path, metavar="DIRECTORY",
                             help="Continue a series from a previous analysis output directory, without modifying it")
        command.add_argument("--series", type=Path,
                             help="Legacy JSON output: explicit series registry; use --continue-from for directory output")
    register = commands.add_parser("series-import", help="Register an existing analysis in a series without model calls")
    register.add_argument("input", type=Path)
    register.add_argument("--series", type=Path, required=True, help="Series registry JSON to create or update")
    register.add_argument("-o", "--output", type=Path, help="Optional analysis copy with stable series identities")
    register.add_argument("--force", action="store_true", help="Replace the optional output copy")

    validate = commands.add_parser("validate", help="Verify schema, source coverage, and cast references")
    validate.add_argument("input", type=Path)
    validate.add_argument("--ready", action="store_true", help="Also fail if spoken passages have no assigned voice")
    review = commands.add_parser("review", help="Print unresolved attribution and casting issues as JSON")
    review.add_argument("input", type=Path)
    apply = commands.add_parser("apply", help="Apply human corrections without changing source text")
    apply.add_argument("input", type=Path)
    apply.add_argument("--edits", type=Path, required=True)
    output_options(apply)
    export = commands.add_parser("export", help="Export a documented bundle for an audio-generation CLI")
    export.add_argument("input", type=Path)
    export.add_argument("-o", "--output", type=Path, required=True)
    export.add_argument("--allow-review", action="store_true", help="Compatibility option; analysis notes do not block export")
    from .tui_support import add_plain_option
    for name in SHELL_COMMANDS:
        add_plain_option(commands.choices[name])
    return root


# Long-running model commands that open the full-screen run shell on an
# interactive terminal (with Textual installed and without --plain).
SHELL_COMMANDS = ("analyze", "review-attribution", "verify-attribution", "refresh-directions",
                  "resolve-speakers", "refresh-voice-profiles")
# Commands whose ordinary rerun starts fresh; the shell resumes them with the
# run ID they printed (--resume RUN_ID).
_RUN_ID = {"verify-attribution": re.compile(r"(?:Fresh|Resuming) verification run: (\S+)"),
           "refresh-directions": re.compile(r"(?:Fresh|Resuming) direction run: (\S+)")}


def _shell(argv, args):
    """Run this same command inside the run shell; its questions become dialogs."""
    from .run_shell import run_in_shell
    argv = list(sys.argv[1:] if argv is None else argv)
    target = getattr(args, "input", None) or getattr(args, "directory", None) or (getattr(args, "directories", None) or [""])[0]
    resume = None
    if args.command in _RUN_ID and "--resume" not in argv:
        def resume(model, books=None):
            found = next((match.group(1) for _, event in reversed(model.logs)
                          if (match := _RUN_ID[args.command].search(event.text))), None)
            return (lambda: main([*argv, "--resume", found], in_shell=True)) if found else None
    return run_in_shell(lambda: main(argv, in_shell=True), title=f"{args.command} — {Path(target).name}",
                        interactive_input=True, resume_command=resume)


def main(argv=None, *, in_shell=False):
    args = parser().parse_args(argv)
    if not in_shell and args.command in SHELL_COMMANDS:
        from .tui_support import wants_tui
        if wants_tui(args):
            return _shell(argv, args)
    model_cache = None
    title_only_resume = False
    original_stderr = sys.stderr
    from .run_events import active_bus
    from .stream_progress import BusStreamConsole
    bus = active_bus()
    # Inside the full-screen run shell, the same live status goes to its screen.
    stream_console = BusStreamConsole(bus) if bus is not None else StreamConsole(original_stderr)
    previous_observer = set_default_stream_observer(stream_console.observe)
    if stream_console.enabled and bus is None:
        # (The run shell already captures stderr, including speaker questions.)
        sys.stderr = stream_console

    def progress(message):
        stream_console.progress(message)

    def provider_client(settings=None, **kwargs):
        options = dict(settings or {})
        options.update(kwargs)
        return OpenAICompatibleClient(**options, stream_observer=stream_console.observe)

    def announce_cache(path):
        resolved = Path(path).resolve()
        print(f"Request cache: {resolved}", file=sys.stderr)
        print("Completed, validated responses are reused automatically; cached stages are replayed and revalidated locally.",
              file=sys.stderr)
        return resolved

    def resume_guidance():
        if model_cache is not None:
            print(f"Resume cache: {model_cache}", file=sys.stderr)
            print("Completed, validated responses remain cached. After resolving the error or interruption, "
                  "rerun the same command with the same input, output, and configuration; "
                  "incomplete requests will be retried.", file=sys.stderr)

    def with_name_tags(directory, config, status):
        """Tag every untagged use of names whose pronunciation this book defines or inherits."""
        if title_only_resume:
            from .name_pronunciations import pending_tagging
            pending = pending_tagging(directory, read_json(directory / "source.json"))
            if pending:
                names = ", ".join(definition["name"] for definition, _ in pending)
                print(f"Chapter titles are current. Pronunciation uses still need decisions for {names}; "
                      f"run audiobook-render pronounce {directory} before rendering. Exiting with status 2.",
                      file=sys.stderr)
                return 2
            return status
        from .name_pronunciation_model import tag_defined_names
        unresolved = tag_defined_names(directory, config, client_factory=provider_client, progress=print)
        if unresolved:
            print(f"Pronunciation tagging incomplete for {', '.join(sorted(unresolved))}; rendering stays "
                  "blocked until a person decides those uses. Exiting with status 2.", file=sys.stderr)
            return 2
        return status

    def resume_book(directory, config):
        """Reuse the saved cast; repair narration boundaries and remaining speakers."""
        nonlocal model_cache, title_only_resume
        from .analysis_directory import _completed_directory
        from .analysis_revision import related_directories, revise_directories
        from .audition_scripts import prepare_document_samples, audition_preparation_pending
        from .narration_boundaries import repair_narration_boundaries
        from .speaker_completion import automatic_completion_targets, complete_speakers, unassigned_segments

        original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if args.input.is_dir():
            try:
                original = _completed_directory(directory)[0]
            except ValueError as error:
                raise ValueError(f"Cannot load the saved analysis in {directory}; supply the authoritative EPUB "
                                 f"so analyze can validate and recover it: {error}") from error
        else:
            source = source_from(args.input, config["extraction"]["include_auxiliary"]
                                 if args.input.suffix.lower() == ".epub" else False)
            from .analysis_recovery import recover_analysis_directory
            recovery = recover_analysis_directory(directory, source)
            if recovery is not None:
                print(f"Recovered a verified analysis baseline for {directory}; preserved drifted files in "
                      f"{recovery.backup_directory}.")
                if recovery.source_matches_preserved_edit:
                    print("The supplied source contains the preserved source edit; the normal source update will reapply it.")
                else:
                    print("The supplied source does not contain every preserved source edit; those bytes remain in the "
                          "backup while the supplied source remains authoritative.")
            if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != original_manifest_hash:
                raise ValueError(f"Analysis recovery changed the saved manifest: {directory}")
            original = _completed_directory(directory)[0]
        ancestors = continuation_directories(directory)
        if args.series is not None or config["analysis"]["series_registry"] is not None:
            raise ValueError("Existing book directories retain their saved series context; remove --series or analysis.series_registry")
        if args.continue_from is not None:
            if not ancestors or args.continue_from.resolve() != ancestors[0]:
                raise ValueError("--continue-from differs from this book's saved continuation parent; resume with the recorded parent or omit the option")
        if not args.input.is_dir():
            from .book_update import (accepted_epub_digests, epub_digest, latest_epub_sha256,
                                      merged_source, update_book)
            from .chapter_title_update import source_for_title_refresh
            title_source = source_for_title_refresh(
                original, source,
                epub_sha256=epub_digest(args.input) if args.input.suffix.lower() == ".epub" else None)
            updates = original["provenance"].get("book_updates") or []
            last_update = updates[-1] if updates else {}
            previous_title_repair = (last_update.get("retitled_chapters")
                                     and not last_update.get("added_chapters")
                                     and not last_update.get("edited_chapters")
                                     and last_update.get("delta_source_fingerprint") is None)
            if (title_source is not None and not args.edits
                    and (previous_title_repair
                         or fingerprint(title_source) != fingerprint(original["source"]))):
                title_only_resume = True
                if fingerprint(title_source) == fingerprint(original["source"]):
                    print("Saved analysis is complete; no source text or chapter-title changes. "
                          "The saved analysis is unchanged.")
                    return _speaker_completion_status(original)
                updated = update_book(
                    directory, title_source, config=config, input_path=args.input, strategy=args.strategy,
                    client_factory=provider_client, announce_cache=announce_cache, progress=progress)
                return _speaker_completion_status(updated if updated is not None else original)
            same_chapters = (fingerprint(source) == fingerprint(original["source"])
                             or fingerprint(merged_source(original["source"], source)) == fingerprint(original["source"]))
            if not same_chapters:
                # A serial's new download: add its new and edited chapters, or
                # explain exactly which saved chapter cannot be matched.
                if args.edits:
                    raise ValueError("--edits cannot be combined with updating the book from a new EPUB; "
                                     "update the book first")
                updated = update_book(
                    directory, source, config=config, input_path=args.input, strategy=args.strategy,
                    client_factory=provider_client, speaker_judge=_speaker_judge(config),
                    announce_cache=announce_cache, progress=progress)
                return _speaker_completion_status(updated if updated is not None else original)
            if args.input.suffix.lower() == ".epub":
                accepted = accepted_epub_digests(original)
                if epub_digest(args.input) in accepted:
                    remember_source_epub(directory, args.input, original["source"]["book"]["id"],
                                         epub_sha256=latest_epub_sha256(original))
                else:
                    # Identical chapters from a different download of the book.
                    print("No new or edited chapters: the EPUB's chapters match the saved book. The saved "
                          "analysis and its recorded EPUB location are unchanged.")

        if not args.edits:
            # A saved split made before quotation marks were read the current way
            # (mixed curly and straight marks, a paragraph opening inside speech)
            # is read again; the model-free tag repair goes first.
            from .split_check import misread_blocks
            repaired = repair_narration_boundaries(original)
            misread = misread_blocks(repaired)
            if misread:
                from .book_update import update_book
                print(f"{len(misread)} paragraph(s) are split differently from their quotation marks (speech "
                      "read by the narrator, or narration read in a character's voice); reading them again.")
                update_book(directory, original["source"], config=config, strategy=args.strategy,
                            client_factory=provider_client, speaker_judge=_speaker_judge(config),
                            announce_cache=announce_cache, progress=progress, reread=misread,
                            base_document=repaired)
                original = _completed_directory(directory)[0]
                original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        revised = apply_edits(original, read_json(args.edits)) if args.edits else original
        if (revised["source"] != original["source"] or revised["cast"] != original["cast"]
                or revised["provenance"].get("series") != original["provenance"].get("series")):
            raise ValueError("Resuming an existing book accepts attribution edits only; cast or series changes require a separate output")
        revised = repair_narration_boundaries(revised, progress=print)
        remaining = unassigned_segments(revised)
        print(f"Resuming saved analysis in {directory}; {len(remaining)} unassigned spoken passage(s).")
        if not remaining and revised == original and not audition_preparation_pending(revised):
            if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != original_manifest_hash:
                raise ValueError("Analysis changed while loading the saved book; rerun to use its latest assignments")
            print("Saved analysis is complete; no model calls or analysis changes are needed.")
            return _speaker_completion_status(original)

        directories = related_directories([directory])
        hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                  for path in directories}
        if hashes[str(directory)] != original_manifest_hash:
            raise ValueError("Analysis changed while loading the saved book; rerun to use its latest assignments")
        settings = config["analysis"]
        decisions_cache = analysis_cache_directory(directory, settings["cache_dir"])
        targets = automatic_completion_targets(revised, retry_exhausted=True) if remaining else []
        requests = None
        if targets:
            print(f"Checking remaining speakers in {len(targets)} source block(s); "
                  "saved assignments will be retained.", file=sys.stderr, flush=True)
        if targets or audition_preparation_pending(revised):
            cache = analysis_cache_directory(directory, settings["cache_dir"])
            if not config["provider"]["model"]:
                raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
            model_cache = announce_cache(cache)
            client = provider_client(config["provider"])
            print(f"Updating analysis with {client.model} at {client.base_url}; book passages will be sent to this endpoint.", file=sys.stderr)
            requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"],
                                      progress=progress)
        if targets:
            revised = complete_speakers(revised, requests, workers=min(5, settings["attribution_workers"]),
                                        target_block_ids=targets, retry_unresolved=True,
                                        progress=progress)
            revised = repair_narration_boundaries(revised, progress=print)
        speaker_input = deepcopy(revised)
        revised = prompt_speaker_decisions(revised, book_path=directory, jev_client=_speaker_judge(config),
                                           decisions_cache=decisions_cache)
        if "voice_samples" in original["provenance"] or audition_preparation_pending(revised):
            # Keep successful auditions and retry only missing ones. Optional
            # casting text must not prevent publishing finished book analysis.
            revised = prepare_document_samples(revised, requests,
                workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                progress=progress)
        if revised != original:
            result = revise_directories({directory: revised}, directories=directories,
                                        expected_manifest_hashes=hashes, scope="speaker-decisions",
                                        speaker_decision_inputs={directory: speaker_input})
            print(f"Original analysis files and revision journal: {result['backup_directory']}")
            print(f"Updated saved analysis and chapter scripts in {directory}.")
        else:
            print("No analysis changes; saved files are unchanged.")
        return _speaker_completion_status(revised)

    try:
        if args.command == "resolve-speakers":
            from .analysis_directory import _completed_directory
            from .analysis_revision import related_directories, revise_directories
            from .speaker_completion import automatic_completion_targets, complete_speakers, unassigned_segments
            from .narration_boundaries import repair_narration_boundaries
            config = load_config(args.config)
            settings = config["analysis"]
            selected = {path.resolve() for path in args.directories}
            directories = related_directories(selected)
            hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                      for path in directories}
            documents = {path: _completed_directory(path)[0] for path in selected}
            revised, requests = {}, None
            for directory in directories:
                if directory not in selected:
                    continue
                revised[directory] = repair_narration_boundaries(documents[directory], progress=print)
                targets = automatic_completion_targets(revised[directory], retry_exhausted=True)
                if not targets:
                    continue
                if requests is None:
                    if not config["provider"]["model"]:
                        raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
                    model_cache = announce_cache(analysis_cache_directory(directories[0], settings["cache_dir"]))
                    client = provider_client(config["provider"])
                    print(f"Resolving speakers with {client.model} at {client.base_url}; selected book passages will be sent to this endpoint.", file=sys.stderr)
                    requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                              max_request_chars=settings["max_request_chars"],
                                              progress=progress)
                print(f"Checking remaining speakers in {len(targets)} source block(s): {directory}; "
                      "saved assignments will be retained.", file=sys.stderr, flush=True)
                revised[directory] = complete_speakers(
                    revised[directory], requests, workers=min(5, settings["attribution_workers"]),
                    target_block_ids=targets, retry_unresolved=True,
                    progress=progress)
                revised[directory] = repair_narration_boundaries(revised[directory], progress=print)
            judge = _speaker_judge(config)
            decisions_cache = analysis_cache_directory(directories[0], settings["cache_dir"])
            speaker_inputs = {}
            for directory in revised:
                speaker_inputs[directory] = deepcopy(revised[directory])
                revised[directory] = prompt_speaker_decisions(revised[directory], book_path=directory,
                                                               jev_client=judge, decisions_cache=decisions_cache)
                if (revised[directory]["cast"] != documents[directory]["cast"]
                        and "voice_samples" in documents[directory]["provenance"]):
                    from .audition_scripts import prepare_document_samples
                    revised[directory] = prepare_document_samples(revised[directory], requests,
                        workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                        progress=progress)
            changed = {path: document for path, document in revised.items() if document != documents[path]}
            if changed:
                result = revise_directories(changed, directories=directories, expected_manifest_hashes=hashes,
                                            scope="speaker-decisions",
                                            speaker_decision_inputs={path: speaker_inputs[path] for path in changed})
                print(f"Original analysis files and revision journal: {result['backup_directory']}")
            else:
                print("No speaker decisions changed; saved files are unchanged.")
            remaining = sum(len(unassigned_segments(document)) for document in revised.values())
            print(f"Speaker completion finished: {remaining} unassigned spoken passage(s) remain.")
            return 2 if remaining else 0
        if args.command == "verify-attribution":
            from .attribution_verification_command import run_attribution_verification
            return run_attribution_verification(args, load_config(args.config), provider_client, progress)
        if args.command == "refresh-directions":
            from .direction_refresh_command import run_direction_refresh
            return run_direction_refresh(args, load_config(args.config), provider_client, progress)
        if args.command in {"refresh-voice-profiles", "review-ui"}:
            return _run_directory_command(args)
        directory_mode = (args.command == "analyze" and
                          (args.input.is_dir() or args.output is None or args.output.suffix.lower() != ".json"))
        if args.command == "analyze" and directory_mode:
            if args.force:
                raise ValueError("--force applies only to legacy .json output; choose a new output directory")
            if args.input.is_dir():
                output = args.input.resolve()
                if args.output is not None and args.output.resolve() != output:
                    raise ValueError("When analyzing an existing book directory, omit --output or use that same directory")
            else:
                output = args.output or default_output(args.input)
            if output.is_dir():
                args.output = output.resolve()
                if not is_pending_directory(args.output):
                    config = load_config(args.config)
                    return with_name_tags(args.output, config, resume_book(args.output, config))
                # A checkpointed run resumes below, after source extraction and
                # identity computation prove it belongs to this command.
            else:
                args.output = validate_output(output)
        if (args.command in {"extract", "analyze", "review-attribution"}
                and args.output.resolve() == resolve_config_path(args.config).resolve()):
            raise ValueError("The output must use a separate path from configuration")
        if (args.command in {"extract", "analyze", "apply", "review-attribution"}
                and not directory_mode and args.output.exists() and not args.force):
            raise FileExistsError(f"Output already exists: {args.output}; choose another path or use --force")
        if args.command == "analyze" and args.strategy is None:
            # Completed books returned above with their saved strategy; only
            # new analyses receive the new-book default.
            args.strategy = "narrator"
        if args.command in {"extract", "plan", "analyze"}:
            config = load_config(args.config)
            include_auxiliary = config["extraction"]["include_auxiliary"] if args.input.suffix.lower() == ".epub" else False
            source = source_from(args.input, include_auxiliary)
            if args.command == "analyze" and args.strategy == "narrator":
                from .front_matter import drop_front_matter
                source = drop_front_matter(source)
            for warning in source["warnings"]:
                print(f"Extraction: {warning}", file=sys.stderr)
            if args.command == "extract":
                write_json(args.output, {"format": "epub-audiobook-source", "version": 1, "source": source}, overwrite=args.force)
                print(f"Extracted {len(source['blocks'])} blocks to {args.output}")
                return 0
            series_path = args.series or config["analysis"]["series_registry"]
            registry = None
            if args.continue_from is not None and series_path is not None:
                raise ValueError("--continue-from cannot be combined with --series or analysis.series_registry")
            if args.command == "plan":
                # Keep explicit registry planning available for legacy JSON
                # workflows; otherwise plan the same directory run as analyze.
                directory_mode = series_path is None
                existing = args.output or default_output(args.input)
                if directory_mode and args.continue_from is None and (existing / "manifest.json").is_file():
                    # Estimate only the chapters a serial update would add.
                    from .book_update import plan_update
                    print(json.dumps(plan_update(existing, source, config), indent=2, ensure_ascii=False))
                    return 0
                if directory_mode:
                    args.output = validate_output(args.output or default_output(args.input))
                elif args.output is not None:
                    raise ValueError("For directory planning use --continue-from instead of --series or analysis.series_registry")
            if directory_mode:
                if series_path is not None:
                    raise ValueError("Directory output uses --continue-from for series context; remove --series or analysis.series_registry")
                cache = analysis_cache_directory(args.output, config["analysis"]["cache_dir"])
                output = args.output.resolve()
                if args.continue_from is not None:
                    previous = args.continue_from.resolve()
                    ancestors = [previous, *continuation_directories(previous)]
                    if any(output == ancestor or ancestor in output.parents
                           or cache == ancestor or ancestor in cache.parents
                           for ancestor in ancestors):
                        raise ValueError("Output and cache directories must be outside --continue-from and every ancestor book; "
                                         "prior analyses stay unchanged")
                    registry = continuation_registry(args.continue_from, source)
                else:
                    # A pending directory's registry does not exist yet; load_registry
                    # derives the same stable identity it would get on a fresh run.
                    registry = (load_registry(args.output / "series-cast.json")
                                if is_pending_directory(args.output) else fresh_registry(args.output))
            elif args.continue_from is not None:
                raise ValueError("--continue-from requires directory output, not a legacy .json output file")
            elif series_path is not None:
                series_path = Path(series_path)
                protected = [args.input, resolve_config_path(args.config)]
                if args.command == "analyze":
                    protected.append(args.output)
                if any(series_path.resolve() == path.resolve() for path in protected):
                    raise ValueError("The series registry must use a separate path from input, output, and configuration")
                registry = load_registry(series_path)
            if args.command == "plan":
                from .analysis_plan import plan_analysis
                planned = plan_analysis(source, config["analysis"], config["performance_policy"], registry)
                if directory_mode:
                    planned["output_directory"] = str(args.output)
                    planned["continue_from"] = str(args.continue_from.resolve()) if args.continue_from else None
                print(json.dumps(planned, indent=2, ensure_ascii=False))
                return 0
            if not config["provider"]["model"]:
                raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
            client = provider_client(config["provider"])
            print(f"Analyzing with {client.model} at {client.base_url}; book text will be sent to this endpoint.", file=sys.stderr)
            settings = config["analysis"]
            cache = (analysis_cache_directory(args.output, settings["cache_dir"])
                     if directory_mode else Path(settings["cache_dir"] or args.output.parent / ".analysis-cache"))
            model_cache = announce_cache(cache)
            edits = read_json(args.edits) if args.edits else None
            pending = None
            document = None
            resume_index = 0
            if directory_mode:
                options_fingerprint = analysis_options_fingerprint(
                    settings, config["performance_policy"],
                    fingerprint(edits) if edits is not None else None)
                if args.strategy not in (None, "consensus"):
                    # A different strategy must never resume another's checkpoints.
                    options_fingerprint = fingerprint([options_fingerprint, args.strategy])
                identity = pending_identity(source, client.identity, options_fingerprint)
                resumed = resume_document(args.output, identity)
                if resumed is not None:
                    document, next_stage = resumed
                    resume_index = len(STAGES) if next_stage is None else STAGES.index(next_stage)
                pending = create_pending(args.output, identity)

            def begin(stage):
                if pending is not None:
                    pending.begin_stage(stage)

            def complete(stage):
                if pending is not None:
                    pending.complete_stage(stage, document)

            try:
                if resume_index <= 0:
                    begin("analyzed")
                    if args.strategy == "narrator":
                        from .narrator_read import narrated_analyze, settings_notes
                        document = narrated_analyze(
                            source, client, model_cache, series_registry=registry,
                            policy=config["performance_policy"], workers=settings["attribution_workers"],
                            repair_attempts=settings["repair_attempts"],
                            max_request_chars=settings["max_request_chars"], progress=progress,
                            meter_callback=stream_console.set_progress, **settings_notes(settings))
                    else:
                        document = analyze(source, client, model_cache, chunk_chars=settings["chunk_chars"],
                            repair_attempts=settings["repair_attempts"], max_request_chars=settings["max_request_chars"],
                            whole_book_discovery=settings["whole_book_discovery"], attribution_workers=settings["attribution_workers"],
                            identity_consensus=settings["identity_consensus"],
                            attribution_review=settings["attribution_review"],
                            attribution_review_chunk_chars=settings["attribution_review_chunk_chars"],
                            attribution_review_notes=settings["attribution_review_notes"],
                            cast_review_notes=settings["cast_review_notes"],
                            cast_reconciliation_rounds=settings["cast_reconciliation_rounds"],
                            cast_reconciliation_notes=settings["cast_reconciliation_notes"],
                            series_registry=registry,
                            policy=config["performance_policy"], progress=progress,
                            meter=stream_console.set_progress)
                    if edits is not None:
                        document = apply_edits(document, edits)
                    complete("analyzed")
                from .audition_scripts import prepare_document_samples
                sample_requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                                 max_request_chars=settings["max_request_chars"],
                                                 progress=progress)
                if resume_index <= 1:
                    # Keep persisted checkpoint names resumable; spelling cleanup is retired.
                    begin("speech-normalized-1")
                    complete("speech-normalized-1")
                if resume_index <= 2:
                    begin("samples-prepared-1")
                    document = prepare_document_samples(document, sample_requests,
                        workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                        progress=progress)
                    complete("samples-prepared-1")
                if resume_index <= 3:
                    begin("speakers-decided")
                    document = prompt_speaker_decisions(document, book_path=args.output, jev_client=_speaker_judge(config),
                                                        decisions_cache=model_cache)
                    complete("speakers-decided")
                if resume_index <= 4:
                    begin("speech-normalized-2")
                    complete("speech-normalized-2")
                if resume_index <= 5:
                    # Human choices may make additional real dialogue available. Keep
                    # optional missing auditions pending without another request pass.
                    begin("samples-prepared-2")
                    document = prepare_document_samples(document, None,
                        workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                        progress=progress)
                    complete("samples-prepared-2")
                if directory_mode:
                    state = publish_directory(document, registry, args.output, export_bundle,
                                              continue_from=args.continue_from,
                                              source_epub=args.input if args.input.suffix.lower() == ".epub" else None)
            except BaseException as error:
                if pending is not None:
                    pending.mark_failed(error, interrupted=isinstance(error, KeyboardInterrupt))
                    pending.close()
                raise
            if pending is not None:
                pending.close()
            if directory_mode:
                print(f"Wrote analysis, chapter scripts, cast, and series continuity to {args.output}")
                print(f"{len(document['segments'])} segments, {len(document['cast']['characters'])} characters; "
                      f"{state['review_count']} advisory notes. Bundle created.")
                return with_name_tags(args.output, config, _speaker_completion_status(document))
            write_json(args.output, document, overwrite=args.force)
            if registry is not None:
                save_registry(series_path, registry, document)
            state = readiness(document)
            print(f"Wrote {len(document['segments'])} segments, {len(document['cast']['characters'])} characters; {state['review_count']} advisory notes. {args.output}")
            return _speaker_completion_status(document)
        if args.command == "validate":
            state = readiness(read_json(args.input))
            print(json.dumps(state, indent=2))
            return 1 if state["errors"] else (2 if args.ready and not state["ready"] else 0)
        document = read_analysis(args.input)
        if args.command == "review-attribution":
            config = load_config(args.config)
            if not config["provider"]["model"]:
                raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
            settings = config["analysis"]
            protected = [resolve_config_path(args.config)]
            if settings["series_registry"] is not None:
                protected.append(Path(settings["series_registry"]))
            if any(args.output.resolve() == path.resolve() for path in protected):
                raise ValueError("The reviewed analysis must use a separate path from configuration and series registry")
            client = provider_client(config["provider"])
            print(f"Reviewing attribution with {client.model} at {client.base_url}; book text will be sent to this endpoint.", file=sys.stderr)
            model_cache = announce_cache(settings["cache_dir"] or args.output.parent / ".epub-audiobook-cache")
            requests = CachedRequests(client, model_cache,
                                      repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"],
                                      progress=progress)
            result = audit_attribution(document, requests, workers=min(5, settings["attribution_workers"]),
                                       review_chunk_chars=settings["attribution_review_chunk_chars"],
                                       review_notes=settings["attribution_review_notes"] or settings["cast_review_notes"],
                                       progress=progress)
            from .narration_boundaries import repair_narration_boundaries
            result = repair_narration_boundaries(result, progress=print)
            write_json(args.output, result, overwrite=args.force)
            print(f"Saved attribution review to {args.output}; {readiness(result)['review_count']} review items remain")
        elif args.command == "series-import":
            if args.series.resolve() == args.input.resolve():
                raise ValueError("The series registry must not replace the original analysis")
            if args.output:
                if args.output.resolve() in {args.series.resolve(), args.input.resolve()}:
                    raise ValueError("The linked analysis must use a separate output path")
                if args.output.exists() and not args.force:
                    raise FileExistsError("The linked analysis output already exists; choose another path or use --force")
            registry = load_registry(args.series)
            linked = link_analysis(document, registry)
            if args.output:
                write_json(args.output, linked, overwrite=args.force)
            save_registry(args.series, registry, linked)
            print(f"Registered {len(linked['cast']['characters'])} characters in series {registry['series_id']}; "
                  f"{readiness(linked)['review_count']} review items. Original analysis preserved.")
        elif args.command == "review":
            print(json.dumps({"source_fingerprint": fingerprint(document["source"]),
                              "cast_fingerprint": fingerprint(document["cast"]),
                              "items": review_items(document)}, indent=2, ensure_ascii=False))
        elif args.command == "apply":
            result = apply_edits(document, read_json(args.edits))
            write_json(args.output, result, overwrite=args.force)
            print(f"Saved corrections to {args.output}; {readiness(result)['review_count']} review items remain")
        elif args.command == "export":
            state = export_bundle(document, args.output, allow_review=args.allow_review,
                                  source_epub_from=args.input.parent if args.input.name == "analysis.json" else None)
            print(f"Exported bundle to {args.output}; {state['review_count']} advisory notes.")
            return _speaker_completion_status(document)
        return 0
    except (ValueError, OSError, ProviderError, KeyError, TypeError, IndexError, RecursionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        resume_guidance()
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed model requests remain cached; checkpointed analysis progress "
              "is kept in the output directory; rerun the same command to resume.", file=sys.stderr)
        resume_guidance()
        return 130
    finally:
        if sys.stderr is stream_console:
            sys.stderr = original_stderr
        set_default_stream_observer(previous_observer)
        stream_console.close()
