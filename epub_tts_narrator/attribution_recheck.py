"""Rerunning ``analyze BOOK`` double-checks who speaks every paragraph and where its speech starts and ends.

The first rerun of a saved book audits every paragraph with the attribution
review (``attribution_review.audit_attribution``: a scan of the whole book,
then three independent jurors for each proposed correction, one of them
checking the boundaries between speech and its tags and action beats). Only
corrections all three approve are applied, through the ``attribution``
revision scope with backups. Each checked paragraph is recorded in
``provenance.attribution_checks`` by its text and its split (kinds, spans,
speakers; never directions), so a later rerun checks again only paragraphs
whose text or split changed since, or every paragraph when ``VERSION`` or the
review prompts change. The run ID comes from the saved book, so an
interrupted rerun replays its responses instead of asking again.

A paragraph with a person's speaker choice is never changed: it is sent only as context.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

from .analysis_directory import _completed_directory
from .analysis_progress import analysis_cache_directory
from .analysis_revision import related_directories, reserve_revision_scope, revise_directories
from .attribution_review_prompts import REVIEW_VERSION
from .pipeline import CachedRequests

VERSION = f"1:{REVIEW_VERSION}"


def _role_ids(segment):
    members = [member["role_id"] for member in segment.get("simultaneous_speakers", [])]
    return sorted(members) if members else [segment.get("role_id")]


def block_states(document):
    """``{block_id: hash}`` of each paragraph's text and split (kinds, spans, speakers)."""
    rows = {}
    for segment in document["segments"]:
        rows.setdefault(segment["block_id"], []).append(
            [segment["start"], segment["end"], segment["kind"], _role_ids(segment)])
    return {block["id"]: hashlib.sha256(json.dumps(
        [block["text"], sorted(rows.get(block["id"], []))], ensure_ascii=False).encode()).hexdigest()
        for block in document["source"]["blocks"]}


def unchecked_blocks(document):
    """Source-ordered IDs of paragraphs not checked at this version since they last changed.

    A paragraph with a person's speaker choice is never one: the audit may not
    change it, so it is sent only as context around the others.
    """
    record = document["provenance"].get("attribution_checks")
    checked = record.get("blocks", {}) if isinstance(record, dict) and record.get("version") == VERSION else {}
    chosen = {row["block_id"] for row in document["segments"] if row["attribution"]["basis"] == "user"}
    return [block_id for block_id, state in block_states(document).items()
            if block_id not in chosen and checked.get(block_id) != state]


def record_checks(document, block_ids):
    """Remember that ``block_ids`` were checked in the state ``document`` now has."""
    record = document["provenance"].get("attribution_checks")
    blocks = dict(record.get("blocks", {})) if isinstance(record, dict) and record.get("version") == VERSION else {}
    states = block_states(document)
    blocks.update({block_id: states[block_id] for block_id in block_ids})
    document["provenance"]["attribution_checks"] = {"version": VERSION, "blocks": dict(sorted(blocks.items()))}


def recheck_attributions(directory, config, client_factory, progress):
    """``analyze BOOK``: check the speakers and splits of every paragraph not yet checked; True when it published."""
    from .attribution_manual_review import print_finding
    from .attribution_review import audit_attribution
    from .attribution_verification import findings
    directory = Path(directory).resolve()
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    targets = unchecked_blocks(document)
    total = len(document["source"]["blocks"])
    if not targets:
        print(f"Speakers and speech splits: all {total} paragraph(s) were already checked and have not "
              "changed since.")
        return False
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("Rerunning analyze checks every paragraph's speakers; set provider.model in "
                         "epub-tts-narrator.toml, or select a TOML file with --config")
    print(f"Checking speakers and speech splits in {len(targets)}/{total} paragraph(s) not checked since they "
          "last changed; only corrections three reviewers agree on are applied, and your own speaker "
          "choices are kept.")
    with ExitStack() as stack:
        reserved = stack.enter_context(reserve_revision_scope([directory]))
        run_id = hashlib.sha256(f"analyze-attribution:{VERSION}:{original_manifest_hash}".encode()).hexdigest()[:32]
        cache = analysis_cache_directory(directory, settings["cache_dir"]) / "attribution-check" / run_id
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Attribution check run: {run_id}", file=sys.stderr)
        client = client_factory(config["provider"])
        print(f"Checking attribution with {client.model} at {client.base_url}; book passages will be sent to "
              "this endpoint.", file=sys.stderr)
        requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        audited = audit_attribution(document, requests, workers=min(5, settings["attribution_workers"]),
                                    target_block_ids=targets, progress=print)
        found = findings(document, audited)
        for index, finding in enumerate(found, 1):
            print_finding(finding, index)
        # Every target was checked; a corrected paragraph is recorded in its corrected state.
        record_checks(audited, targets)
        try:
            directories = related_directories([directory])
            hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                      for path in directories}
            if hashes[str(directory)] != original_manifest_hash:
                raise ValueError(f"Analysis changed while its attribution was being checked: {directory}")
            result = revise_directories({directory: audited}, directories=directories,
                                        expected_manifest_hashes=hashes, scope="attribution", progress=print,
                                        reserved_update_locks=reserved or None)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Attribution check publication did not succeed: {exc} This run's responses "
                             "remain; rerun analyze to publish them without new requests.") from exc
    review = audited["provenance"]["attribution_review"]
    flagged = sum(1 for finding in found if finding["review_only"])
    print(f"Speakers and speech splits checked in {len(targets)} paragraph(s): {review['applied_blocks']} "
          f"paragraph(s) corrected, {review['rejected_correction_blocks']} proposed correction(s) not agreed, "
          f"{flagged} line(s) newly flagged for review. Original analysis files and revision journal: "
          f"{result['backup_directory']}")
    return True
