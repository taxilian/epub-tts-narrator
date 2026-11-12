"""Model-free chapter-title repairs for an already analyzed book.

The ordinary serial updater owns comparison and publication. This helper proves
that a title delta changes no source content or layout, builds the exact saved
source plus new titles, and rebinds only source-bound local sidecars.
"""

from copy import deepcopy
from pathlib import Path
import tempfile

from .pipeline import fingerprint, write_json


_INCIDENTAL_SOURCE_FIELDS = frozenset({"book", "chapters", "blocks", "warnings", "exclusions"})


def source_for_title_refresh(document, source, *, epub_sha256=None):
    """Recognize title equivalence without restoring recorded narrator omissions."""
    from .book_update import accepted_epub_digests, saved_strategy
    saved = document["source"]
    current = deepcopy(source)
    if (saved_strategy(document) == "narrator" and epub_sha256 is not None
            and epub_sha256 == source["book"]["id"]
            and epub_sha256 in accepted_epub_digests(document)):
        omissions = [(entry.get("href"), entry.get("block_id")) for entry in saved["exclusions"]
                     if entry.get("reason") == "publisher apparatus omitted by narrator-read analysis"]
        blocks = {block["id"]: block for block in current["blocks"]}
        chapters = {chapter["id"]: chapter for chapter in current["chapters"]}
        for href, identifier in omissions:
            block = blocks.get(identifier)
            chapter = chapters.get(block["chapter_id"]) if block is not None else None
            if (block is None or chapter is None or block["source"]["href"] != href
                    or chapter["source_href"] != href or identifier not in chapter["block_ids"]):
                return None
        omitted = {identifier for _, identifier in omissions}
        current["blocks"] = [block for block in current["blocks"] if block["id"] not in omitted]
        for chapter in current["chapters"]:
            chapter["block_ids"] = [identifier for identifier in chapter["block_ids"] if identifier not in omitted]
    if len(current["chapters"]) != len(saved["chapters"]):
        return None
    current["book"]["id"] = saved["book"]["id"]
    current["warnings"] = deepcopy(saved["warnings"])
    current["exclusions"] = deepcopy(saved["exclusions"])
    previous = deepcopy(current)
    for old, new in zip(saved["chapters"], previous["chapters"]):
        new["title"] = old["title"]
    if previous != saved:
        return None
    return current


def _title_source(saved, comparison):
    """Return saved source plus current titles, or None for any other change."""
    if (comparison["appended_chapters"] or comparison["edited_chapters"]
            or not comparison["retitled_chapters"] or comparison["changed_blocks"]
            or comparison["removed_blocks"] or comparison["block_map"]):
        return None
    current = comparison["source"]
    if current["blocks"] != saved["blocks"] or len(current["chapters"]) != len(saved["chapters"]):
        return None
    for old, new in zip(saved["chapters"], current["chapters"]):
        if dict(new, title=old["title"]) != old:
            return None
    old_fields = {key for key in saved if key not in _INCIDENTAL_SOURCE_FIELDS}
    new_fields = {key for key in current if key not in _INCIDENTAL_SOURCE_FIELDS}
    if old_fields != new_fields or any(current[key] != saved[key] for key in old_fields):
        return None
    source = deepcopy(saved)
    titles = {chapter["id"]: chapter["title"] for chapter in current["chapters"]}
    for chapter in source["chapters"]:
        chapter["title"] = titles[chapter["id"]]
    return source


def retitled_document(saved, comparison, *, epub_sha256=None):
    """Build the metadata-only candidate through the ordinary merge invariants."""
    source = _title_source(saved["source"], comparison)
    if source is None:
        return None
    from .book_update import merge_update
    merged = deepcopy(comparison)
    merged["source"] = source
    document = merge_update(saved, None, merged, epub_sha256=epub_sha256)
    samples = document["provenance"].get("voice_samples")
    if isinstance(samples, dict) and "source_fingerprint" in samples:
        samples["source_fingerprint"] = fingerprint(source)
    return document


def _rebind_performance(directory, document):
    """Validate and rebind only the book-local performance plan, if present."""
    from .performance_planning import (
        MAX_PLAN_BYTES, _read_bounded, _safe_path, load_prepared_performance,
    )
    path = directory / "performance-plan.json"
    if not path.exists() and not path.is_symlink():
        return None
    record = _read_bounded(_safe_path(path), MAX_PLAN_BYTES)
    chapters = record.get("chapters") if isinstance(record, dict) else None
    from .renderer.bundle import load_bundle
    record = load_prepared_performance(load_bundle(directory), path, chapters=chapters)
    bound = deepcopy(record)
    source_fingerprint = fingerprint(document["source"])
    bound["source_fingerprint"] = source_fingerprint
    provenance = {key: deepcopy(value) for key, value in bound["provenance"].items() if key != "inputs_sha256"}
    bound["provenance"]["inputs_sha256"] = fingerprint({
        "source": source_fingerprint, "chapters": bound["chapters"], **provenance})
    from .cli import export_bundle
    with tempfile.TemporaryDirectory(prefix=".title-update-", dir=directory) as staging:
        bundle_directory = Path(staging) / "bundle"
        export_bundle(document, bundle_directory, allow_review=True)
        plan_path = bundle_directory / path.name
        write_json(plan_path, bound)
        load_prepared_performance(load_bundle(bundle_directory), plan_path, chapters=bound["chapters"])
    return bound


def title_update_sidecars(directory, saved, document):
    """Rebind the existing local source-bound sidecars for a title-only update."""
    from .book_update import _rebind_sidecars
    directory = Path(directory).resolve()
    records = _rebind_sidecars(directory, saved, document)
    performance = _rebind_performance(directory, document)
    if performance is not None:
        records["performance-plan.json"] = performance
    return records
