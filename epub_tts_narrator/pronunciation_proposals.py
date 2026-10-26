"""Cached pronunciation proposals, without approval, book edits, or synthesis."""

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from copy import deepcopy
import json

from .performance_plan import validate_phonemes
from .pronunciations import WORD_PATTERN, validate_word
from .pronunciation_proposal_prompts import SYSTEM


PROPOSAL_VERSION = 1
MAX_CONTEXTS = 3
MAX_CONTEXT_CHARS = 600
MAX_REQUEST_CHARS = 16000
_VOWELS = frozenset("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())


def _validate_response(response):
    if not isinstance(response, dict) or set(response) != {"phonemes"}:
        raise ValueError("Pronunciation proposal requires only a phonemes string")
    value = validate_phonemes(response["phonemes"])
    tokens = value.split(" ")
    if any(token in _VOWELS for token in tokens):
        raise ValueError("Pronunciation proposal must mark every CMU vowel with stress 0, 1, or 2")
    if not any(token[:-1] in _VOWELS and token[-1:] == "1" for token in tokens):
        raise ValueError("Pronunciation proposal needs at least one CMU vowel with primary stress 1")
    return {"alphabet": "cmu_arpabet", "value": value}


def _contexts(source, word):
    """Find current-book evidence; inherited entry citations belong elsewhere."""
    contexts = []
    blocks = source.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("Pronunciation proposal requires the current book's source blocks")
    for block in blocks:
        if (not isinstance(block, dict) or not isinstance(block.get("id"), str)
                or not 1 <= len(block["id"]) <= 200 or not isinstance(block.get("text"), str)):
            raise ValueError("Pronunciation proposal source blocks need bounded IDs and exact text")
        match = next((match for match in WORD_PATTERN.finditer(block["text"]) if match.group() == word), None)
        if match is None:
            continue
        # Keep exact source substrings, not paraphrases or synthesized context.
        start = max(0, match.start() - (MAX_CONTEXT_CHARS - len(word)) // 2)
        end = min(len(block["text"]), start + MAX_CONTEXT_CHARS)
        contexts.append({"block_id": block["id"], "start": start,
                         "text": block["text"][start:end],
                         "evidence": {"start": match.start(), "end": match.end(), "quote": word}})
        if len(contexts) == MAX_CONTEXTS:
            break
    if not contexts:
        raise ValueError("Pronunciation target must occur as an exact whole word in the current book's source")
    return contexts


def _cast_context(document, word):
    cast = document.get("cast", {})
    if not isinstance(cast, dict):
        raise ValueError("Pronunciation proposal cast context must be an object")
    characters = cast.get("characters", [])
    if not isinstance(characters, list):
        raise ValueError("Pronunciation proposal cast context must be a character list")
    selected = []
    for character in characters:
        if not isinstance(character, dict):
            raise ValueError("Pronunciation proposal cast context contains an invalid character")
        name, aliases = character.get("name"), character.get("aliases", [])
        if not isinstance(name, str) or not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError("Pronunciation proposal cast names and aliases must be text")
        matches = [value for value in [name, *aliases]
                   if any(match.group() == word for match in WORD_PATTERN.finditer(value))]
        if not matches:
            continue
        # Include only the associated exact names, never an entire cast dump.
        bounded = [value for value in dict.fromkeys(matches) if len(value) <= 256][:4]
        if bounded:
            selected.append({"names": bounded})
        if len(selected) == 8:
            break
    return selected


def _payload(document, entry, respelling, maximum):
    if not isinstance(document, dict) or not isinstance(document.get("source"), dict):
        raise ValueError("Pronunciation proposal requires a loaded current-book document")
    if not isinstance(entry, dict):
        raise ValueError("Pronunciation proposal requires a target word entry")
    word = validate_word(entry.get("word"))
    if respelling is not None:
        validate_word(respelling)
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Pronunciation request budget must be a positive integer")
    source = document["source"]
    book = source.get("book", {})
    language = book.get("language") if isinstance(book, dict) else None
    if language is not None and (not isinstance(language, str) or len(language) > 128):
        raise ValueError("Pronunciation proposal book language must be bounded text")
    result = {"mode": "pronunciation_proposal", "version": PROPOSAL_VERSION, "word": word,
              "book_language": language, "context": _contexts(source, word),
              "cast_context": _cast_context(document, word)}
    if respelling is not None:
        result["respelling"] = respelling
    budget = min(maximum, MAX_REQUEST_CHARS)
    def fits():
        return len(SYSTEM) + len(json.dumps(result, ensure_ascii=False)) <= budget
    # Reduce optional examples before rejecting a small configured context.
    # The target and its intended respelling are never shortened or removed.
    while not fits() and len(result["cast_context"]) > 1:
        result["cast_context"].pop()
    while not fits() and len(result["context"]) > 1:
        result["context"].pop()
    if not fits():
        raise ValueError("Pronunciation proposal exceeds analysis.max_request_chars; increase the request budget")
    return result


def propose_pronunciation(document, entry, requests, *, respelling=None):
    """Propose one stressed CMU reading through existing cache/repair machinery.

    Returns a proposal only. Even a well-formed response needs a listening
    audition and explicit user confirmation before it becomes render policy.
    """
    payload = _payload(document, entry, respelling, requests.max_request_chars)
    return requests.run(SYSTEM, payload, _validate_response)


def propose_name_pronunciations(document, entries, requests, progress=None, workers=1, on_proposal=None):
    """Checkpoint each newly requested pending name/place proposal as it arrives.

    The optional callback receives (entry, pronunciation) serially. On a batch
    failure, queued requests are cancelled and valid results already in flight
    still reach the callback before the first failure is raised. Cached replies
    are revalidated, so retrying without a callback also preserves model work.
    """
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("Pronunciation proposal workers must be from 1 through 5")
    if not isinstance(entries, list) or len(entries) > 100000:
        raise ValueError("Pronunciation proposals require a bounded entry list")
    if on_proposal is not None and not callable(on_proposal):
        raise ValueError("Pronunciation proposal checkpoint must be callable")
    selected, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Pronunciation proposals require word entries")
        if (entry.get("status") != "pending" or entry.get("kind") not in {"name", "place"}
                or "proposal" in entry):
            continue
        word = validate_word(entry.get("word"))
        if word in seen:
            raise ValueError("Pronunciation proposal entries repeat an exact word")
        selected.append(deepcopy(entry))
        seen.add(word)
    progress = progress or (lambda message: None)
    result = {}
    def retain(entry, proposal):
        if on_proposal is not None:
            on_proposal(deepcopy(entry), deepcopy(proposal))
        result[entry["word"]] = proposal
        progress(f"Pronunciation proposals: {len(result)}/{len(selected)} complete")
    if workers == 1:
        for entry in selected:
            retain(entry, propose_pronunciation(document, entry, requests))
    elif selected:
        with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as executor:
            futures = {executor.submit(propose_pronunciation, document, entry, requests): entry for entry in selected}
            failure = None
            try:
                for future in as_completed(futures):
                    try:
                        retain(futures[future], future.result())
                    except CancelledError:
                        if failure is None:
                            raise
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
                            for pending in futures:
                                pending.cancel()
            except BaseException:
                # Ctrl-C while waiting must not start every queued proposal.
                # Running requests still retain validated replies in the cache.
                for pending in futures:
                    pending.cancel()
                raise
            if failure is not None:
                raise failure
    return result
