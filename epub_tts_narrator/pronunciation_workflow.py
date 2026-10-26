"""Resume pronunciation discovery and name proposals after book analysis."""

from pathlib import Path

from .analysis_directory import _completed_directory, continuation_directories
from .pipeline import CachedRequests
from .provider import OpenAICompatibleClient


def _requests(directory, config, progress, client_factory=None):
    settings = config["analysis"]
    cache = Path(settings["cache_dir"] or directory.parent / ".epub-audiobook-cache").resolve()
    if any(cache == book or book in cache.parents for book in [directory, *continuation_directories(directory)]):
        raise ValueError("The request cache must be outside this book and its continuation ancestors")
    if not config["provider"]["model"]:
        raise ValueError("Set provider.model in the global TOML file before preparing pronunciations")
    client = (client_factory or OpenAICompatibleClient)(**config["provider"])
    progress(f"Preparing pronunciation suggestions with {client.model} at {client.base_url}; "
             "book text will be sent to this endpoint.")
    progress(f"Request cache: {cache}; completed discovery and pronunciation requests are reused on retry.")
    return CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                          max_request_chars=settings["max_request_chars"], progress=progress)


def create_pronunciation_proposer(directory, config_path=None, *, progress=print):
    """Lazily create a cached proposal client for interactive readable corrections.

    Listing, replaying a prepared proposal, or saving explicit CMU needs no LLM
    configuration. Book source and effective choices are reread when requested.
    """
    from .config import load_config
    from .pronunciation_proposals import propose_pronunciation
    from .pronunciations import WORD_PATTERN, load_pronunciations, validate_word

    directory = Path(directory).resolve()
    requests = None

    def propose(word, respelling=None):
        nonlocal requests
        validate_word(word)
        document = _completed_directory(directory)[0]
        evidence = next(({"block_id": block["id"], "quote": word}
                         for block in document["source"]["blocks"]
                         if any(token.group() == word for token in WORD_PATTERN.finditer(block["text"]))), None)
        if evidence is None:
            raise ValueError("Pronunciation word must occur exactly in this book's source")
        entry = next((item for item in load_pronunciations(directory) if item["word"] == word),
                     {"word": word, "kind": "other", "reason": "Manually selected for pronunciation review.",
                      "evidence": [evidence]})
        if requests is None:
            requests = _requests(directory, load_config(config_path), progress)
        return propose_pronunciation(document, entry, requests, respelling=respelling)

    return propose


def prepare_book_pronunciations(directory, config, *, document=None, requests=None, progress=print,
                               client_factory=None):
    """Complete source discovery, then checkpoint each pending name's proposal.

    The request cache retains completed batches and phoneme replies through API
    failures. A failed proposal never repeats finished discovery. Cast, scripts,
    voices, and continuation snapshots are never rewritten here.
    """
    from .pronunciations import discovery_pending, load_pronunciations, save_candidates, save_proposal
    from .pronunciation_discovery import discover_pronunciations
    from .pronunciation_proposals import propose_name_pronunciations

    directory = Path(directory).resolve()
    saved = _completed_directory(directory)[0]
    if document is not None and (document["source"] != saved["source"] or document["cast"] != saved["cast"]):
        raise ValueError("Book source or cast changed before pronunciation discovery; rerun against the saved book")
    document = saved
    settings = config["analysis"]
    if discovery_pending(directory, document):
        if requests is None:
            requests = _requests(directory, config, progress, client_factory)
        candidates = discover_pronunciations(document, requests, progress=progress,
                                            workers=min(5, settings["attribution_workers"]))
        save_candidates(directory, document, candidates)
    entries = load_pronunciations(directory)
    if any(entry["status"] == "pending" and entry["kind"] in {"name", "place"}
           and not entry.get("proposal") for entry in entries):
        if requests is None:
            requests = _requests(directory, config, progress, client_factory)
        propose_name_pronunciations(
            document, entries, requests, progress=progress, workers=min(5, settings["attribution_workers"]),
            on_proposal=lambda entry, proposal: save_proposal(directory, document, entry["word"], proposal))
        entries = load_pronunciations(directory)
    pending = sum(entry["status"] == "pending" for entry in entries)
    proposals = sum(entry["status"] == "pending" and bool(entry.get("proposal")) for entry in entries)
    progress(f"Pronunciations: {pending} word(s) to review ({proposals} with CMU suggestions), {len(entries)} total; "
             "use audiobook-render pronunciations BOOK. Pending words keep the engine's default pronunciation.")
    return entries
