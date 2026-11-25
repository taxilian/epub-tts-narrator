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

A character added since the last check (a person's new character, for
instance) makes the paragraphs that mention it, and those around the lines it
speaks, be checked again with it among the choices: its other lines may be
its too. The record keeps the cast's character IDs for this.

A paragraph with a person's speaker choice is never changed: it is sent only as context.
With ``reconsider_choices`` (``analyze BOOK --reconsider-choices``) those
paragraphs are checked too, the reviewers told which lines a person chose, and
may be corrected like any other.

A person's instructions (``analyze BOOK --instructions``) reach the scanner and
jurors as review notes. Paragraphs checked under other notes (or none) are
checked again once under new ones: the record keeps the notes' digest.
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


NEARBY = 3


def _record(document):
    record = document["provenance"].get("attribution_checks")
    return record if isinstance(record, dict) and record.get("version") == VERSION else {}


def new_characters(document):
    """Cast characters added since the last check (a person's new character, a merge's survivor stays)."""
    known = _record(document).get("characters")
    if known is None:
        return []
    return [row for row in document["cast"]["characters"] if row["id"] not in set(known)]


def near_new_characters(document):
    """Paragraphs that mention a newly added character or lie within ``NEARBY`` of a line they speak.

    Their other lines may be theirs too, so these are checked again with the
    new character among the choices.
    """
    import re
    added = new_characters(document)
    if not added:
        return set()
    blocks = document["source"]["blocks"]
    position = {block["id"]: index for index, block in enumerate(blocks)}
    roles = {role["id"] for role in document["cast"]["roles"] if role.get("character_id") in {row["id"] for row in added}}
    names = [name for row in added for name in [row["name"], *row.get("aliases", [])] if name.strip()]
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(name) for name in names) + r")\b", re.I) if names else None
    found = {block["id"] for block in blocks if pattern is not None and pattern.search(block["text"])}
    for segment in document["segments"]:
        if set(_role_ids(segment)) & roles:
            index = position[segment["block_id"]]
            found.update(block["id"] for block in blocks[max(0, index - NEARBY):index + NEARBY + 1])
    return found


def notes_digest(notes):
    """The digest a check under ``notes`` is recorded with (None without notes)."""
    notes = (notes or "").strip()
    return hashlib.sha256(notes.encode("utf-8")).hexdigest() if notes else None


def unchecked_blocks(document, *, notes=None, reconsider_choices=False, everything=False):
    """Source-ordered IDs of paragraphs to check: not checked at this version since they last changed, or
    near a character added since the last check; every paragraph once under new ``notes``.

    A paragraph with a person's speaker choice is one only with
    ``reconsider_choices``; otherwise the audit may not change it, so it is
    sent only as context around the others. ``everything`` (a recheck) takes
    every paragraph, however recently it was checked.
    """
    record = _record(document)
    checked = {} if everything else record.get("blocks", {})
    digest = notes_digest(notes)
    if digest is not None and record.get("notes") != digest:
        checked = {}
    near = near_new_characters(document)
    chosen = set() if reconsider_choices else {
        row["block_id"] for row in document["segments"] if row["attribution"]["basis"] == "user"}
    return [block_id for block_id, state in block_states(document).items()
            if block_id not in chosen and (checked.get(block_id) != state or block_id in near)]


def record_checks(document, block_ids, *, notes=None):
    """Remember that ``block_ids`` were checked in the state ``document`` now has, with its current cast
    (and under ``notes``, whose digest is kept)."""
    record = _record(document)
    digest = notes_digest(notes)
    blocks = {} if digest is not None and record.get("notes") != digest else dict(record.get("blocks", {}))
    states = block_states(document)
    blocks.update({block_id: states[block_id] for block_id in block_ids})
    document["provenance"]["attribution_checks"] = {
        "version": VERSION, "characters": sorted(row["id"] for row in document["cast"]["characters"]),
        "blocks": dict(sorted(blocks.items()))}
    if digest is not None or record.get("notes"):
        document["provenance"]["attribution_checks"]["notes"] = digest or record["notes"]


def recheck_attributions(directory, config, client_factory, progress, *, notes=None, reconsider_choices=False,
                         recheck_cache=None):
    """``analyze BOOK``: check the speakers and splits of every paragraph not yet checked; True when it published."""
    from .attribution_manual_review import print_finding
    from .attribution_review import audit_attribution
    from .attribution_verification import findings
    directory = Path(directory).resolve()
    original_manifest_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    document, _, _ = _completed_directory(directory)
    targets = unchecked_blocks(document, notes=notes, reconsider_choices=reconsider_choices,
                               everything=recheck_cache is not None)
    total = len(document["source"]["blocks"])
    if not targets:
        print(f"Speakers and speech splits: all {total} paragraph(s) were already checked and have not "
              "changed since.")
        return False
    settings = config["analysis"]
    if not config["provider"]["model"]:
        raise ValueError("Rerunning analyze checks every paragraph's speakers; set provider.model in "
                         "epub-tts-narrator.toml, or select a TOML file with --config")
    added = new_characters(document)
    if added:
        print("New in the cast since the last check: " + ", ".join(row["name"] for row in added)
              + "; the paragraphs that mention them and those around their lines are checked again with them "
                "among the choices.")
    print(f"Checking speakers and speech splits in {len(targets)}/{total} paragraph(s); only corrections three "
          "reviewers agree on are applied, and "
          + ("your own speaker choices may be corrected too (--reconsider-choices)." if reconsider_choices
             else "your own speaker choices are kept.")
          + (" Your instructions are given to the reviewers." if notes_digest(notes) else ""))
    with ExitStack() as stack:
        reserved = stack.enter_context(reserve_revision_scope([directory]))
        identity = f"analyze-attribution:{VERSION}:{original_manifest_hash}"
        if notes_digest(notes) or reconsider_choices:
            identity += f":{notes_digest(notes)}:{reconsider_choices}"
        run_id = hashlib.sha256(identity.encode()).hexdigest()[:32]
        # A recheck asks afresh in its own cache, whose answers an interrupted recheck replays.
        cache = (recheck_cache / "attribution-check" if recheck_cache is not None
                 else analysis_cache_directory(directory, settings["cache_dir"]) / "attribution-check" / run_id)
        cache.mkdir(parents=True, exist_ok=True)
        print(f"Attribution check run: {run_id}", file=sys.stderr)
        client = client_factory(config["provider"])
        print(f"Checking attribution with {client.model} at {client.base_url}; book passages will be sent to "
              "this endpoint.", file=sys.stderr)
        requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                  max_request_chars=settings["max_request_chars"], progress=progress)
        audited = audit_attribution(document, requests, workers=min(5, settings["attribution_workers"]),
                                    target_block_ids=targets, progress=print,
                                    review_notes=(notes or "").strip(), reconsider_choices=reconsider_choices)
        found = findings(document, audited)
        for index, finding in enumerate(found, 1):
            print_finding(finding, index)
        # Every target was checked; a corrected paragraph is recorded in its corrected state.
        record_checks(audited, targets, notes=notes)
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
