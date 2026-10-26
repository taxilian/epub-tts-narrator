"""Resume pronunciation discovery independently of completed book analysis."""

from pathlib import Path

from .analysis_directory import _completed_directory, continuation_directories
from .pipeline import CachedRequests
from .provider import OpenAICompatibleClient


def prepare_book_pronunciations(directory, config, *, document=None, requests=None, progress=print,
                               client_factory=None):
    """Publish candidates only after every source batch has a validated response.

    The existing request cache retains completed batches through API failures.
    Cast, scripts, voices, and continuation snapshots are never rewritten here.
    """
    from .pronunciations import discovery_pending, load_pronunciations, save_candidates
    from .pronunciation_discovery import discover_pronunciations

    directory = Path(directory).resolve()
    saved = _completed_directory(directory)[0]
    if document is not None and (document["source"] != saved["source"] or document["cast"] != saved["cast"]):
        raise ValueError("Book source or cast changed before pronunciation discovery; rerun against the saved book")
    document = saved
    if discovery_pending(directory, document):
        settings = config["analysis"]
        if requests is None:
            cache = Path(settings["cache_dir"] or directory.parent / ".epub-audiobook-cache").resolve()
            if any(cache == book or book in cache.parents for book in [directory, *continuation_directories(directory)]):
                raise ValueError("The request cache must be outside this book and its continuation ancestors")
            if not config["provider"]["model"]:
                raise ValueError("Set provider.model in the global TOML file before discovering pronunciations")
            client = (client_factory or OpenAICompatibleClient)(**config["provider"])
            progress(f"Discovering pronunciation candidates with {client.model} at {client.base_url}; "
                     "book text will be sent to this endpoint.")
            progress(f"Request cache: {cache}; completed discovery batches are reused on retry.")
            requests = CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                                      max_request_chars=settings["max_request_chars"], progress=progress)
        candidates = discover_pronunciations(document, requests, progress=progress,
                                            workers=min(5, settings["attribution_workers"]))
        save_candidates(directory, document, candidates)
    entries = load_pronunciations(directory)
    pending = sum(entry["status"] == "pending" for entry in entries)
    progress(f"Pronunciations: {pending} word(s) to review, {len(entries)} total; "
             "use audiobook-render pronunciations BOOK. Pending words keep the engine's default pronunciation.")
    return entries
