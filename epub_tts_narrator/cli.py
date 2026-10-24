"""Command-line boundary: extraction, analysis, corrections, and TTS handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from . import __version__
from .config import load_config, resolve_config_path
from .contracts import readiness, review_items, validate_document
from .epub import extract_epub
from .pipeline import (CachedRequests, analyze, apply_edits, check_cast, fingerprint,
                       make_units, seed_cast, write_json)
from .attribution_review import audit_attribution, count_review_batches
from .speaker_decisions import prompt_speaker_decisions
from .jev import JevClient
from .provider import OpenAICompatibleClient, ProviderError
from .series import link_analysis, load_registry, save_registry, series_context
from .analysis_directory import (
    default_output, validate_output, continuation_registry, fresh_registry,
    publish_directory, continuation_directories,
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


def script_partitions(document):
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


def export_bundle(document, output, *, allow_review=False):
    state = readiness(document)
    if state["errors"]:
        raise ValueError("Invalid analysis: " + "; ".join(state["errors"][:12]))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".audiobook-export-", dir=output.parent))
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
        # Reserve the destination exclusively. POSIX rename() can replace a
        # concurrently created empty directory, so it is not a no-clobber commit.
        # Publish the manifest last: consumers must require it before reading.
        output.mkdir()
        (output / "scripts").mkdir()
        for name in (*names, "manifest.json"):
            os.link(staging / name, output / name)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
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
    plan = input_command("plan", "Show extraction counts and baseline model request count")
    plan.add_argument("-o", "--output", type=Path,
                      help="Proposed output directory; defaults to ./<input stem>.audiobook")
    run = input_command("analyze", "Analyze an EPUB, or resume unfinished speaker choices in an existing book directory")
    run.add_argument("-o", "--output", type=Path,
                     help="Output directory; defaults to ./<input stem>.audiobook. A .json path selects legacy file output.")
    run.add_argument("--force", action="store_true",
                     help="Replace legacy .json output only; completed directories are never overwritten")
    run.add_argument("--edits", type=Path, help="Reapply source-bound human corrections after analysis")
    passage_review = input_command("review-attribution", "Review an existing script with source evidence and three-way consensus")
    output_options(passage_review)
    speakers = commands.add_parser("resolve-speakers", help="Resolve unassigned passages by consensus and refresh existing book directories")
    speakers.add_argument("directories", nargs="+", type=Path, help="Analyzed book directories to repair in place; originals are backed up")
    speakers.add_argument("--config", type=Path, help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
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
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    model_cache = None

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

    def resume_book(directory, config):
        """Reuse the saved cast; repair narration boundaries and remaining speakers."""
        nonlocal model_cache
        from .analysis_directory import _completed_directory
        from .analysis_revision import related_directories, revise_directories
        from .audition_scripts import prepare_document_samples, audition_preparation_pending
        from .narration_boundaries import repair_narration_boundaries
        from .speaker_completion import automatic_completion_targets, complete_speakers, unassigned_segments
        from .speech_normalization import normalize_document_speech, normalization_pending

        original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        original = _completed_directory(directory)[0]
        ancestors = continuation_directories(directory)
        if args.series is not None or config["analysis"]["series_registry"] is not None:
            raise ValueError("Existing book directories retain their saved series context; remove --series or analysis.series_registry")
        if args.continue_from is not None:
            if not ancestors or args.continue_from.resolve() != ancestors[0]:
                raise ValueError("--continue-from differs from this book's saved continuation parent; resume with the recorded parent or omit the option")
        if not args.input.is_dir():
            source = source_from(args.input, config["extraction"]["include_auxiliary"]
                                 if args.input.suffix.lower() == ".epub" else False)
            if fingerprint(source) != fingerprint(original["source"]):
                raise ValueError("Input source does not match the existing book directory. Choose another output directory; the saved book was not changed")

        revised = apply_edits(original, read_json(args.edits)) if args.edits else original
        if (revised["source"] != original["source"] or revised["cast"] != original["cast"]
                or revised["provenance"].get("series") != original["provenance"].get("series")):
            raise ValueError("Resuming an existing book accepts attribution edits only; cast or series changes require a separate output")
        revised = repair_narration_boundaries(revised, progress=print)
        remaining = unassigned_segments(revised)
        print(f"Resuming saved analysis in {directory}; {len(remaining)} unassigned spoken passage(s).")
        if (not remaining and revised == original and not normalization_pending(revised)
                and not audition_preparation_pending(revised)):
            if hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() != original_manifest_hash:
                raise ValueError("Analysis changed while loading the saved book; rerun to use its latest assignments")
            print("Saved analysis is complete; no model calls or file changes are needed.")
            return _speaker_completion_status(original)

        directories = related_directories([directory])
        hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                  for path in directories}
        if hashes[str(directory)] != original_manifest_hash:
            raise ValueError("Analysis changed while loading the saved book; rerun to use its latest assignments")
        settings = config["analysis"]
        targets = automatic_completion_targets(revised, retry_exhausted=True) if remaining else []
        requests = None
        if targets:
            print(f"Checking remaining speakers in {len(targets)} source block(s); "
                  "saved assignments will be retained.", file=sys.stderr, flush=True)
        if targets or normalization_pending(revised) or audition_preparation_pending(revised):
            cache = Path(settings["cache_dir"] or directory.parent / ".epub-audiobook-cache").resolve()
            if any(cache == book or book in cache.parents for book in [directory, *ancestors, *directories]):
                raise ValueError("The cache directory must be outside the existing book and its series directories")
            if not config["provider"]["model"]:
                raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
            model_cache = announce_cache(cache)
            client = OpenAICompatibleClient(**config["provider"])
            print(f"Updating analysis with {client.model} at {client.base_url}; book passages will be sent to this endpoint.", file=sys.stderr)
            requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"],
                                      progress=lambda message: print(message, file=sys.stderr))
        if targets:
            revised = complete_speakers(revised, requests, workers=min(5, settings["attribution_workers"]),
                                        target_block_ids=targets, retry_unresolved=True,
                                        progress=lambda message: print(message, file=sys.stderr))
            revised = repair_narration_boundaries(revised, progress=print)
        revised = prompt_speaker_decisions(revised, book_path=directory, jev_client=_speaker_judge(config))
        revised = normalize_document_speech(revised, requests,
            workers=min(5, settings["attribution_workers"]),
            progress=lambda message: print(message, file=sys.stderr))
        if "voice_samples" in original["provenance"] or audition_preparation_pending(revised):
            # Keep successful auditions and retry only missing ones. Optional
            # casting text must not prevent publishing finished book analysis.
            revised = prepare_document_samples(revised, requests,
                workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                progress=lambda message: print(message, file=sys.stderr))
        if revised != original:
            result = revise_directories({directory: revised}, directories=directories,
                                        expected_manifest_hashes=hashes)
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
                    model_cache = announce_cache(settings["cache_dir"] or directories[0].parent / ".epub-audiobook-cache")
                    client = OpenAICompatibleClient(**config["provider"])
                    print(f"Resolving speakers with {client.model} at {client.base_url}; selected book passages will be sent to this endpoint.", file=sys.stderr)
                    requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                              max_request_chars=settings["max_request_chars"],
                                              progress=lambda message: print(message, file=sys.stderr))
                print(f"Checking remaining speakers in {len(targets)} source block(s): {directory}; "
                      "saved assignments will be retained.", file=sys.stderr, flush=True)
                revised[directory] = complete_speakers(
                    revised[directory], requests, workers=min(5, settings["attribution_workers"]),
                    target_block_ids=targets, retry_unresolved=True,
                    progress=lambda message: print(message, file=sys.stderr))
                revised[directory] = repair_narration_boundaries(revised[directory], progress=print)
            judge = _speaker_judge(config)
            for directory in revised:
                revised[directory] = prompt_speaker_decisions(revised[directory], book_path=directory,
                                                             jev_client=judge)
            changed = {path: document for path, document in revised.items() if document != documents[path]}
            if changed:
                result = revise_directories(changed, directories=directories, expected_manifest_hashes=hashes)
                print(f"Original analysis files and revision journal: {result['backup_directory']}")
            else:
                print("No speaker decisions changed; saved files are unchanged.")
            remaining = sum(len(unassigned_segments(document)) for document in revised.values())
            print(f"Speaker completion finished: {remaining} unassigned spoken passage(s) remain.")
            return 2 if remaining else 0
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
                return resume_book(args.output, load_config(args.config))
            args.output = validate_output(output)
        if (args.command in {"extract", "analyze", "review-attribution"}
                and args.output.resolve() == resolve_config_path(args.config).resolve()):
            raise ValueError("The output must use a separate path from configuration")
        if (args.command in {"extract", "analyze", "apply", "review-attribution"}
                and not directory_mode and args.output.exists() and not args.force):
            raise FileExistsError(f"Output already exists: {args.output}; choose another path or use --force")
        if args.command in {"extract", "plan", "analyze"}:
            config = load_config(args.config)
            include_auxiliary = config["extraction"]["include_auxiliary"] if args.input.suffix.lower() == ".epub" else False
            source = source_from(args.input, include_auxiliary)
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
                if directory_mode:
                    args.output = validate_output(args.output or default_output(args.input))
                elif args.output is not None:
                    raise ValueError("For directory planning use --continue-from instead of --series or analysis.series_registry")
            if directory_mode:
                if series_path is not None:
                    raise ValueError("Directory output uses --continue-from for series context; remove --series or analysis.series_registry")
                cache = config["analysis"]["cache_dir"] or args.output.parent / ".epub-audiobook-cache"
                cache = Path(cache).resolve()
                output = args.output.resolve()
                if cache == output or output in cache.parents:
                    raise ValueError("The cache directory must be outside the output directory")
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
                    registry = fresh_registry(args.output)
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
                batches = make_units(source, config["analysis"]["chunk_chars"])
                whole_book = config["analysis"]["whole_book_discovery"]
                reconciliation_rounds = config["analysis"]["cast_reconciliation_rounds"]
                consensus = config["analysis"]["identity_consensus"]
                identity_scans = (1 if whole_book else len(batches)) if consensus else 0
                context = series_context(registry, source) if registry is not None else None
                series_scans = 1 if consensus and context and context["prior_books"] else 0
                attribution_review = config["analysis"]["attribution_review"]
                review_scans = (count_review_batches(source, config["analysis"]["attribution_review_chunk_chars"])
                                if attribution_review else 0)
                base_requests = (len(batches) + 1 if whole_book else 2 * len(batches)) + reconciliation_rounds
                planned = {"book": source["book"], "chapters": len(source["chapters"]),
                                  "blocks": len(source["blocks"]), "text_characters": sum(len(b["text"]) for b in source["blocks"]),
                                  "analysis_batches": len(batches),
                                  "cast_reconciliation_rounds": reconciliation_rounds,
                                  "identity_consensus": consensus,
                                  "identity_scan_requests": identity_scans,
                                  "series_identity_scan_requests_minimum": series_scans,
                                  "attribution_review": attribution_review,
                                  "attribution_review_scan_requests": review_scans,
                                  "voice_ensemble_requests": 1,
                                  "speech_spelling_scan_requests_minimum": 1 if source["blocks"] else 0,
                                  "baseline_model_requests": (base_requests + identity_scans + series_scans
                                                              + review_scans + 1 + bool(source["blocks"])),
                                  "note": ("Before cache reuse and bounded validation repairs; character count is not a token estimate. "
                                           + f"Baseline includes {reconciliation_rounds} cumulative cast reconciliation round(s). "
                                           + "Baseline includes one whole-cast voice design request after the cast is final; "
                                             "it is skipped when discovery yields fewer than two distinct voices or only "
                                             "established series profiles. "
                                           + ("Baseline includes identity scans. Identity case count is unknown before analysis: "
                                              "allow three verdict requests per case, including series matches, "
                                              "plus one preparation request per proposed split."
                                              if consensus else "Identity consensus is disabled.")
                                           + ((" Baseline budgets one dedicated series continuity scan assuming a nonempty discovered cast; "
                                               "an empty cast needs no scan. "
                                               + ("Whole-book mode uses one complete-source continuity scan."
                                                  if whole_book else "Bounded mode can require additional character/context scan batches "
                                                  "if the complete source exceeds the request budget; this count is unknown until cast discovery.")
                                               + " A series jury without three approvals, or scanner uncertainty with nonempty prior candidates, "
                                               "can add one additional continuity round "
                                               "(two rounds maximum): another scan, or bounded-mode scan batches, plus three fresh jury "
                                               "requests for every match proposed in that round, including previously approved matches. "
                                               "New-character and uncertain decisions without prior candidates do not trigger this retry.")
                                              if series_scans else "")
                                           + (" Baseline includes attribution review scans. Add three jury requests per review batch "
                                              "that proposes corrections; the number is unknown before review."
                                              if attribution_review else " Attribution review is disabled.")
                                           + " Baseline includes one spoken-spelling scan; the final segment count determines additional bounded batches."
                                             " Add three independent reviews for each batch proposing spelling edits."),
                                  "exclusions": source["exclusions"]}
                if registry is not None:
                    planned["series"] = {"series_id": registry["series_id"], "books": len(context["prior_books"]),
                                         "characters": len(context["characters"])}
                if directory_mode:
                    planned["output_directory"] = str(args.output)
                    planned["continue_from"] = str(args.continue_from.resolve()) if args.continue_from else None
                print(json.dumps(planned, indent=2, ensure_ascii=False))
                return 0
            if not config["provider"]["model"]:
                raise ValueError("Set provider.model in epub-tts-narrator.toml, or select a TOML file with --config")
            client = OpenAICompatibleClient(**config["provider"])
            print(f"Analyzing with {client.model} at {client.base_url}; book text will be sent to this endpoint.", file=sys.stderr)
            settings = config["analysis"]
            cache = settings["cache_dir"] or args.output.parent / ".epub-audiobook-cache"
            model_cache = announce_cache(cache)
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
                policy=config["performance_policy"], progress=lambda message: print(message, file=sys.stderr))
            if args.edits:
                document = apply_edits(document, read_json(args.edits))
            from .audition_scripts import prepare_document_samples
            from .speech_normalization import normalize_document_speech
            sample_requests = CachedRequests(client, model_cache, repair_attempts=settings["repair_attempts"],
                                             max_request_chars=settings["max_request_chars"],
                                             progress=lambda message: print(message, file=sys.stderr))
            document = normalize_document_speech(document, sample_requests,
                workers=min(5, settings["attribution_workers"]),
                progress=lambda message: print(message, file=sys.stderr))
            document = prepare_document_samples(document, sample_requests,
                workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                progress=lambda message: print(message, file=sys.stderr))
            document = prompt_speaker_decisions(document, book_path=args.output, jev_client=_speaker_judge(config))
            document = normalize_document_speech(document, sample_requests,
                workers=min(5, settings["attribution_workers"]),
                progress=lambda message: print(message, file=sys.stderr))
            # Human choices may make additional real dialogue available. Keep
            # optional missing auditions pending without another request pass.
            document = prepare_document_samples(document, None,
                workers=min(5, settings["attribution_workers"]), allow_incomplete=True,
                progress=lambda message: print(message, file=sys.stderr))
            if directory_mode:
                state = publish_directory(document, registry, args.output, export_bundle,
                                          continue_from=args.continue_from)
                print(f"Wrote analysis, chapter scripts, cast, and series continuity to {args.output}")
                print(f"{len(document['segments'])} segments, {len(document['cast']['characters'])} characters; "
                      f"{state['review_count']} advisory notes. Bundle created.")
                return _speaker_completion_status(document)
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
            client = OpenAICompatibleClient(**config["provider"])
            print(f"Reviewing attribution with {client.model} at {client.base_url}; book text will be sent to this endpoint.", file=sys.stderr)
            model_cache = announce_cache(settings["cache_dir"] or args.output.parent / ".epub-audiobook-cache")
            requests = CachedRequests(client, model_cache,
                                      repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"],
                                      progress=lambda message: print(message, file=sys.stderr))
            result = audit_attribution(document, requests, workers=min(5, settings["attribution_workers"]),
                                       review_chunk_chars=settings["attribution_review_chunk_chars"],
                                       review_notes=settings["attribution_review_notes"] or settings["cast_review_notes"],
                                       progress=lambda message: print(message, file=sys.stderr))
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
            state = export_bundle(document, args.output, allow_review=args.allow_review)
            print(f"Exported bundle to {args.output}; {state['review_count']} advisory notes.")
            return _speaker_completion_status(document)
        return 0
    except (ValueError, OSError, ProviderError, KeyError, TypeError, IndexError, RecursionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        resume_guidance()
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed model requests remain cached; rerun the same command to resume.", file=sys.stderr)
        resume_guidance()
        return 130
