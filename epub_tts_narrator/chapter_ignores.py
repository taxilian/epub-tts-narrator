"""Persistent, reversible, book-local chapter ignores.

``review-ui`` can exclude whole script partitions from rendering and
assembly. The choice lives in a book-local ``chapter-ignores.json``
sidecar whose every entry binds the book ID plus the ignored partition's
exact chapter ID and complete ordered source block IDs, so a title or
ordinal change never transfers a choice, and a partition whose layout
moved or disappeared fails closed instead of ignoring different text.

The renderer plan emits ignored chapters as zero-chunk rows (retaining
id, number, title, and script path) and every synthesis, readiness, and
provenance path skips them; unignoring restores the ordinary plan and any
pre-ignore checkpoint byte for byte. The sidecar never propagates to
series descendants and never rewrites analysis, manifest, or source.
"""

from copy import deepcopy
from pathlib import Path

from .renderer.common import digest, read_json, regular_file, write_json

FILENAME = "chapter-ignores.json"
SNAPSHOT_KEY = "_chapter_ignore_snapshot"
UPDATE_LOCK = ".analysis-update.lock"
_FORMAT = "epub-audiobook-chapter-ignores"
_VERSION = 1


def bundle_document(bundle):
    """The analyzer locator view of a loaded renderer bundle.

    ``script_partitions`` reads only source block/chapter structure and
    segment kinds, so this projection — not a second locator scheme —
    resolves sidecar entries against exactly the partitions the bundle was
    validated to contain.
    """
    rows = [row for chapter in bundle["chapters"] for row in chapter["rows"]]
    return {"source": bundle["source"], "segments": rows}


def _partitions(document):
    from .chapter_partitions import script_partitions
    return script_partitions(document)


def _book_id(document):
    book = document["source"].get("book") if isinstance(document.get("source"), dict) else None
    value = book.get("id") if isinstance(book, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{FILENAME}: the book source carries no ID to bind chapter ignores to")
    return value


def _entry(value):
    if (not isinstance(value, dict) or set(value) != {"chapter_id", "block_ids"}
            or not isinstance(value["chapter_id"], str) or not value["chapter_id"]
            or not isinstance(value["block_ids"], list)
            or any(not isinstance(block, str) or not block for block in value["block_ids"])
            or len(set(value["block_ids"])) != len(value["block_ids"])):
        raise ValueError(f"{FILENAME}: every ignore entry must bind chapter_id and an ordered block_ids list "
                         "(an empty list addresses a chapter with no passages)")
    return {"chapter_id": value["chapter_id"], "block_ids": list(value["block_ids"])}


def _load(directory, book_id):
    path = Path(directory) / FILENAME
    if not (path.exists() or path.is_symlink()):
        return None
    if path.is_symlink():
        raise ValueError(f"{FILENAME} must be a regular file, not a symlink")
    record = read_json(regular_file(path))
    if not isinstance(record, dict):
        raise ValueError(f"{FILENAME} must contain a JSON object")
    if set(record) != {"format", "version", "book_id", "ignored"}:
        raise ValueError(f"{FILENAME} fields must be exactly format, version, book_id, and ignored; fix or remove it")
    if (record["format"] != _FORMAT or type(record["version"]) is not int
            or record["version"] != _VERSION):
        raise ValueError(f"{FILENAME} is not a version-{_VERSION} chapter-ignore sidecar; fix or remove it")
    if record["book_id"] != book_id:
        raise ValueError(f"{FILENAME} belongs to book {record['book_id']!r}, not {book_id!r}; "
                         "chapter ignores are book-local and never inherited")
    if not isinstance(record["ignored"], list):
        raise ValueError(f"{FILENAME} ignored must be a list of chapter locators; fix or remove it")
    return dict(record, ignored=[_entry(entry) for entry in record["ignored"]])


def _locator(partition):
    return {"chapter_id": partition["chapter_id"], "block_ids": list(partition["block_ids"])}


def _locator_key(entry):
    return (entry["chapter_id"], tuple(entry["block_ids"]))


def _resolve(entries, document):
    """Map every saved locator to exactly one current partition index.

    Locators are complete ordered block-ID lists, which partition the source
    disjointly, so an unmatched entry means the source's chapter layout
    changed; fail closed instead of ignoring different text.
    """
    partitions = _partitions(document)
    by_locator = {}
    for index, partition in enumerate(partitions):
        by_locator.setdefault(_locator_key(partition), []).append(index)
    saved = []
    for entry in entries:
        matches = by_locator.get(_locator_key(entry))
        if not matches:
            raise ValueError(f"{FILENAME}: an ignored chapter ({entry['chapter_id']}, "
                             f"{len(entry['block_ids'])} blocks) no longer matches any chapter partition of "
                             "this book; its layout changed. Review the choice again before rendering — "
                             "rerun review-ui, re-ignore the chapter you want, and never keep a stale choice")
        if len(matches) > 1:
            raise ValueError(f"{FILENAME}: an ignored chapter ({entry['chapter_id']}, "
                             f"{len(entry['block_ids'])} blocks) matches {len(matches)} chapter partitions; "
                             "the choice is ambiguous and nothing may be ignored")
        saved.append(matches[0])
    return frozenset(saved)


def ignored_indices(directory, document):
    """Zero-based script-partition indices this book ignores, or an empty set.

    ``document`` is the analyzer locator view (``bundle_document``); indices
    address the original ``script_partitions`` order the bundle's chapter
    scripts follow. A missing sidecar or a sidecar with no ignores returns an
    empty set without touching source coordinates.
    """
    record = _load(directory, _book_id(document))
    if record is None or not record["ignored"]:
        return frozenset()
    return _resolve(record["ignored"], document)


def set_ignored(directory, document, chapter_index, ignored, *, expected_manifest_hash):
    """Toggle one chapter partition's ignore choice and return the saved indices.

    ``expected_manifest_hash`` is the canonical digest of ``manifest.json``
    (the ``bundle_hash`` ``renderer.bundle.load_bundle`` reports). The current
    manifest must still match it, so a caller working from a stale book view
    writes nothing. The source and partition boundaries must also match the
    saved bundle, so a stale document paired with a newer receipt is refused.
    The toggle runs under the book's ``.analysis-update.lock``
    (serialized against analysis updates), rereads the latest sidecar under
    that lock so concurrent toggles never lose a choice, and replaces the
    sidecar atomically. Choices never propagate to descendant books.
    """
    directory = Path(directory)
    partitions = _partitions(document)
    if type(chapter_index) is not int or not 0 <= chapter_index < len(partitions):
        raise ValueError(f"chapter_index must be a zero-based chapter partition index "
                         f"between 0 and {len(partitions) - 1}")
    book_id = _book_id(document)
    entry = _locator(partitions[chapter_index])
    from .renderer.run_locks import acquire_directory_lock
    lock_path = directory / UPDATE_LOCK
    try:
        lock = acquire_directory_lock(lock_path)
    except FileExistsError as exc:
        raise ValueError(f"An analysis revision of this book is already active; retry the "
                         f"chapter ignore when it finishes: {lock_path}") from exc
    try:
        manifest = read_json(regular_file(directory / "manifest.json"))
        if digest(manifest) != expected_manifest_hash:
            raise ValueError("This book changed after it was loaded; reload it and retry the "
                             "chapter ignore. Nothing was saved")
        if read_json(regular_file(directory / "source.json")) != document["source"]:
            raise ValueError("This book's source changed after it was loaded; reload it and retry the "
                             "chapter ignore. Nothing was saved")
        scripts = manifest.get("files", {}).get("scripts")
        if not isinstance(scripts, list) or len(scripts) != len(partitions):
            raise ValueError("This book's chapter partitions changed after it was loaded; reload it "
                             "and retry the chapter ignore. Nothing was saved")
        previous = None
        for partition, script in zip(partitions, scripts):
            chapter_id = partition["chapter_id"]
            if (script.get("chapter_id") != chapter_id
                    or (chapter_id == previous
                        and script.get("heading_block_id") != partition["heading_block_id"])):
                raise ValueError("This book's chapter partitions changed after it was loaded; reload it "
                                 "and retry the chapter ignore. Nothing was saved")
            previous = chapter_id
        record = _load(directory, book_id)
        entries = record["ignored"] if record is not None else []
        if ignored and entry not in entries:
            entries = [*entries, entry]
        elif not ignored:
            entries = [saved for saved in entries if saved != entry]
        # A stale locator already in the sidecar refuses the whole toggle here,
        # before anything is written: nothing is saved until every entry,
        # old and new, resolves against the current partitions.
        saved_indices = _resolve(entries, document)
        write_json(directory / FILENAME,
                   {"format": _FORMAT, "version": _VERSION, "book_id": book_id, "ignored": entries})
    finally:
        lock.rmdir()
    return saved_indices


def freeze(bundle, config):
    """Freeze this invocation's ignored partition indices into ``config``.

    Renderer CLI and render-queue prologues call this once, early, before any
    readiness or planning work; recovery and paragraph-fallback replans
    inside the same invocation reuse the snapshot, so a choice saved while a
    render is running applies to the next invocation, never mid-run.
    """
    config[SNAPSHOT_KEY] = ignored_indices(Path(bundle["directory"]), bundle_document(bundle))
    return config[SNAPSHOT_KEY]


def included_source(bundle, ignored):
    """The analyzer source view restricted to chapters the plan still renders.

    Only readiness scans (name-pronunciation labelling and tagged rendering)
    see this view; the sidecar's saved decisions and every other consumer
    keep the full source, so no evidence is invalidated. An ignored chapter's
    untagged or unsure name uses cannot block rendering a book that will not
    speak them.
    """
    if not ignored:
        return bundle["source"]
    partitions = _partitions(bundle_document(bundle))
    keep = {block_id for index, partition in enumerate(partitions) if index not in ignored
            for block_id in partition["block_ids"]}
    source = dict(bundle["source"])
    source["blocks"] = [block for block in source["blocks"] if block["id"] in keep]
    source["chapters"] = [chapter for chapter in (dict(item, block_ids=[b for b in item["block_ids"] if b in keep])
                                                  for item in source["chapters"]) if chapter["block_ids"]]
    return source


def rebind_validated(directory, saved_document, new_document):
    """Revalidate the sidecar's entries against an updated source document.

    A title or heading repair keeps every exact locator; a source update that
    changes an ignored partition's chapter ID or block layout fails closed
    before the update publishes, so a stale choice can never ignore different
    text. Returns a copy of the unchanged record, or ``None`` without one.
    """
    record = _load(directory, _book_id(saved_document))
    if record is None:
        return None
    _ = _resolve(record["ignored"], new_document)
    return deepcopy(record)
