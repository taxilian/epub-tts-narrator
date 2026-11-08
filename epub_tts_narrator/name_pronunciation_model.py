"""LLM help for a user-chosen name pronunciation: guesses and use tagging.

Every request goes through ``CachedRequests`` inside the book's own analysis
cache, so an interrupted run resumes without repeating completed requests, and
tagging checkpoints each batch's decisions into ``name-pronunciations.json``.
Nothing here confirms a pronunciation; the listener does that.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .name_pronunciation_prompts import GUESS_SYSTEM, TAG_SYSTEM
from .name_pronunciations import (DECISIONS, name_key, normalize_phonemes, validate_name_phonemes,
                                  validate_respelling)
from .pronunciation_dictionary import dictionary_readings
from .pronunciations import WORD_PATTERN


GUESS_VERSION = 1
TAG_VERSION = 1
MAX_CONTEXTS = 4
CONTEXT_BEFORE = 320
CONTEXT_AFTER = 200
BATCH_OCCURRENCES = 40
MAX_BATCH_CHARS = 24000


def make_requests(directory, config, *, progress=print, client_factory=None):
    """Cached model access inside the book directory, created only when needed."""
    from .analysis_progress import analysis_cache_directory
    from .pipeline import CachedRequests
    from .provider import OpenAICompatibleClient
    if not config["provider"]["model"]:
        raise ValueError("Set provider.model in the TOML configuration before asking the model about names")
    settings = config["analysis"]
    cache = analysis_cache_directory(directory, settings["cache_dir"])
    client = (client_factory or OpenAICompatibleClient)(**config["provider"])
    progress(f"Using {client.model} at {client.base_url}; book excerpts containing the name are sent to this endpoint.")
    return CachedRequests(client, cache, repair_attempts=settings["repair_attempts"],
                          max_request_chars=settings["max_request_chars"], progress=progress)


def _excerpt(text, start, end):
    """Exact source around one occurrence, trimmed to word boundaries."""
    left = max(0, start - CONTEXT_BEFORE)
    if left:
        space = text.find(" ", left, start)
        left = space + 1 if space != -1 else left
    right = min(len(text), end + CONTEXT_AFTER)
    if right < len(text):
        space = text.rfind(" ", end, right)
        right = space if space != -1 else right
    return text[left:start], text[start:end], text[end:right]


def character_context(document, name):
    """Cast entries carrying the name, with descriptions; never the whole cast."""
    key = name_key(name)
    found = []
    for character in document.get("cast", {}).get("characters", []):
        names = [character.get("name", ""), *character.get("aliases", [])]
        if any(token.group().casefold() == key for value in names if isinstance(value, str)
               for token in WORD_PATTERN.finditer(value)):
            found.append({"name": character.get("name", ""),
                          "aliases": [value for value in character.get("aliases", []) if isinstance(value, str)][:8],
                          "description": str(character.get("description", ""))[:600]})
        if len(found) == 4:
            break
    return found


def guess(document, name, requests, occurrences, *, current=None, instruction=None, rejected=()):
    """One proposed reading; a revision follows the listener's instruction."""
    blocks = {block["id"]: block["text"] for block in document["source"]["blocks"]}
    contexts = []
    for item in occurrences:
        if item["form"] != "base":
            continue
        before, word, after = _excerpt(blocks[item["block_id"]], item["start"], item["end"])
        contexts.append(before + "⟦" + word + "⟧" + after)
        if len(contexts) == MAX_CONTEXTS:
            break
    payload = {"mode": "name_pronunciation_guess", "version": GUESS_VERSION, "name": name,
               "book_language": document["source"]["book"].get("language"),
               "dictionary_readings": list(dictionary_readings(name)),
               "characters": character_context(document, name), "excerpts": contexts}
    if current is not None:
        payload["current"] = current
    if instruction is not None:
        payload["listener_instruction"] = instruction
    rejected = [value for value in dict.fromkeys(rejected) if value != current]
    if rejected:
        payload["rejected"] = rejected[-12:]
    excluded = set(rejected) | ({current} if current and instruction is not None else set())

    def validate(response):
        if not isinstance(response, dict) or not {"phonemes"} <= set(response) or set(response) - {"phonemes", "respelling"}:
            raise ValueError("Return only phonemes and an optional respelling")
        phonemes = validate_name_phonemes(normalize_phonemes(response["phonemes"]))
        if phonemes in excluded:
            raise ValueError("That reading was already heard; return a different one that follows the instruction")
        respelling = response.get("respelling")
        try:
            respelling = validate_respelling(respelling)
        except ValueError:
            respelling = None  # Display-only hint; never worth a repair request.
        return {"phonemes": phonemes, "respelling": respelling}
    return requests.run(GUESS_SYSTEM, payload, validate)


def _batches(items, document):
    blocks = {block["id"]: block["text"] for block in document["source"]["blocks"]}
    batch, size = [], 0
    for item in items:
        before, word, after = _excerpt(blocks[item["block_id"]], item["start"], item["end"])
        text = before + "⟦" + word + "⟧" + after
        if batch and (len(batch) == BATCH_OCCURRENCES or size + len(text) > MAX_BATCH_CHARS):
            yield batch
            batch, size = [], 0
        batch.append((item, text))
        size += len(text)
    if batch:
        yield batch


def tag(document, name, requests, pending, *, on_batch, workers=1, progress=print):
    """Classify every pending occurrence; ``on_batch`` checkpoints each batch."""
    characters = character_context(document, name)
    batches = list(_batches(pending, document))
    total = len(pending)

    def run(batch):
        payload = {"mode": "name_use_tagging", "version": TAG_VERSION, "name": name, "characters": characters,
                   "occurrences": [{"n": index + 1, "text": text} for index, (_, text) in enumerate(batch)]}
        if len(TAG_SYSTEM) + len(json.dumps(payload, ensure_ascii=False)) > requests.max_request_chars:
            raise ValueError("Name tagging batch exceeds analysis.max_request_chars; increase the request budget")

        def validate(response):
            if (not isinstance(response, dict) or set(response) != {"decisions"}
                    or not isinstance(response["decisions"], list) or len(response["decisions"]) != len(batch)):
                raise ValueError(f"Return exactly {len(batch)} decisions, one per occurrence in order")
            if any(not isinstance(value, str) or value not in DECISIONS for value in response["decisions"]):
                raise ValueError('Each decision must be "name", "other" or "unsure"')
            return response["decisions"]
        decisions = requests.run(TAG_SYSTEM, payload, validate)
        return [{key: item[key] for key in ("block_id", "start", "end", "word", "block_sha256")}
                | {"decision": decision, "basis": "model"} for (item, _), decision in zip(batch, decisions)]

    done = 0
    if workers <= 1 or len(batches) <= 1:
        for batch in batches:
            on_batch(run(batch))
            done += len(batch)
            progress(f"Tagged {done}/{total} occurrences of {name}")
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as executor:
        futures = [executor.submit(run, batch) for batch in batches]
        try:
            for future in as_completed(futures):
                result = future.result()
                on_batch(result)
                done += len(result)
                progress(f"Tagged {done}/{total} occurrences of {name}")
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def directory_document(directory):
    from .analysis_directory import _completed_directory
    return _completed_directory(Path(directory).resolve())[0]
