"""Bounded, cached discovery of exact source words needing pronunciation review."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json

from .pipeline import fingerprint
from .pronunciations import DISCOVERY_VERSION, WORD_PATTERN, _evidence_contains_word, validate_entry, validate_word, word_id
from .pronunciation_discovery_prompts import SYSTEM


MAX_BATCH_CHARS = 12000

# Cast aliases also contain descriptions such as "the second guard" and
# "Jasper's mother". These words alone are not proper names. This conservative
# filter is only for automatic cast seeding; the contextual source scan may
# still identify a title-shaped word as a genuine name in the story.
_GENERIC_NAME_WORDS = frozenset("""
a an the and or but nor of to for from in on at by with without as into onto
over under between through before after out up down about around along near
is are was were be been being has have had do does did can could would should
i me my mine myself you your yours yourself he him his himself she her hers
herself it its itself we us our ours ourselves they them their theirs themselves
who whom whose which that this these those someone somebody anyone anybody
everyone everybody no one none another other others any all both each some
narrator narration author character characters voice voices speaker speakers
unknown unnamed unidentified anonymous unassigned unrecognized unseen unheard
mysterious familiar unfamiliar distant disembodied inner internal external
first second third fourth fifth last next previous former latter younger older
young old elderly little small tall large big male female man woman men women
boy girl child children adult person people human humans stranger strangers
father mother parent parents dad mom mum son daughter brother sister sibling
grandfather grandmother grandparent grandparents grandson granddaughter uncle
aunt cousin husband wife nephew niece friend companion partner family relatives
mr mrs ms miss dr doctor professor sir madam ma'am lord lady king queen prince
princess duke duchess emperor empress count countess baron baroness saint pope
captain commander general admiral lieutenant sergeant corporal colonel major
chief leader master mistress elder high grand great senior junior
guard guards soldier soldiers warrior warriors rider riders knight knights
priest priests priestess monk nun servant servants slave slaves attendant
merchant trader innkeeper bartender teacher student healer cook farmer sailor
messenger scout assassin enemy opponent ally visitor passenger customer officer
apprentice disciple instructor assistant subordinate receptionist technician
spectator bystander passerby henchman bandit thug bully
""".split())


def character_name_candidates(document):
    """Queue literal cast-name words even when the model considers them obvious."""
    source = document["source"]
    labels = []
    for character in document.get("cast", {}).get("characters", []):
        for label in [character.get("name", ""), *(character.get("aliases") or [])]:
            if not isinstance(label, str):
                continue
            words = []
            for token in WORD_PATTERN.finditer(label):
                word = token.group()
                generic_form = word.casefold().replace("’", "'")
                if generic_form.endswith("'s"):
                    generic_form = generic_form[:-2]
                if generic_form in _GENERIC_NAME_WORDS:
                    continue
                try:
                    validate_word(word)
                except ValueError:
                    continue
                words.append(word)
            labels.append((words, len(list(WORD_PATTERN.finditer(label))) == 1))
    # A phrase alias can contain ordinary descriptors ("voice in Mara's head").
    # Preserve lowercase exact variants when explicitly named on their own or
    # when their capitalization counterpart is also a cast name; never merge
    # their identities or infer any spelling resemblance.
    proper_forms = {word.casefold() for words, single_word in labels for word in words
                    if single_word or any(character.isupper() for character in word)}
    named = {word for words, _ in labels for word in words if word.casefold() in proper_forms}
    evidence = {}
    for block in source["blocks"]:
        for token in WORD_PATTERN.finditer(block["text"]):
            word = token.group()
            if word in named and word not in evidence:
                evidence[word] = {"block_id": block["id"], "quote": word}
    return [validate_entry({"id": word_id(word), "word": word, "status": "pending", "kind": "name",
                            "reason": "Character name or alias queued for pronunciation review, even if its reading seems familiar.",
                            "evidence": [evidence[word]]}, source)
            for word in sorted(evidence, key=lambda item: (item.casefold(), item))]


def _context(source, cast):
    return {"mode": "pronunciation_discovery", "version": DISCOVERY_VERSION,
            "source_fingerprint": fingerprint(source),
            "book": {key: deepcopy(source.get("book", {}).get(key)) for key in ("id", "title", "authors", "language")},
            "cast_context": [{key: deepcopy(character.get(key)) for key in ("id", "name", "aliases")}
                             for character in (cast or {}).get("characters", [])]}


def _split(text, limit):
    offset = 0
    while offset < len(text):
        end = min(len(text), offset + limit)
        if end < len(text):
            # Never split a lexical word or invent evidence across a window.
            for token in WORD_PATTERN.finditer(text, offset):
                if token.start() >= end:
                    break
                if token.start() < end < token.end():
                    end = token.start()
                    break
            if end <= offset:
                raise ValueError("Source word exceeds the pronunciation discovery request budget")
        yield offset, text[offset:end]
        offset = end


def _batches(source, cast, max_request_chars):
    context = _context(source, cast)
    overhead = len(SYSTEM) + len(json.dumps(dict(context, target_blocks=[]), ensure_ascii=False))
    available = max_request_chars - overhead - 128
    if available < 512:
        raise ValueError("Pronunciation cast context exceeds analysis.max_request_chars")
    window_limit = min(MAX_BATCH_CHARS, available // 2)
    batches, current, characters = [], [], 0
    for block in source["blocks"]:
        for offset, text in _split(block["text"], window_limit):
            item = {"block_id": block["id"], "start": offset, "text": text}
            candidate = dict(context, target_blocks=current + [item])
            if current and (characters + len(text) > MAX_BATCH_CHARS
                            or len(SYSTEM) + len(json.dumps(candidate, ensure_ascii=False)) > max_request_chars):
                batches.append(dict(context, target_blocks=current))
                current, characters = [], 0
            current.append(item)
            characters += len(text)
            if len(SYSTEM) + len(json.dumps(dict(context, target_blocks=current), ensure_ascii=False)) > max_request_chars:
                raise ValueError("Pronunciation source window exceeds analysis.max_request_chars")
    if current:
        batches.append(dict(context, target_blocks=current))
    return batches


def count_discovery_batches(source, *, cast=None, max_request_chars=200000):
    return len(_batches(source, cast, max_request_chars))


def _validator(source, payload):
    visible = {}
    for window in payload["target_blocks"]:
        visible.setdefault(window["block_id"], []).append(window["text"])
    def validate(response):
        if not isinstance(response, dict) or set(response) != {"candidates"} or not isinstance(response["candidates"], list) or len(response["candidates"]) > 256:
            raise ValueError("Pronunciation discovery requires only a bounded candidates array")
        checked, seen = [], set()
        for index, candidate in enumerate(response["candidates"]):
            if not isinstance(candidate, dict) or set(candidate) != {"word", "kind", "reason", "evidence"}:
                raise ValueError(f"Pronunciation candidate {index} requires only word, kind, reason, and evidence")
            entry = validate_entry(dict(candidate, id=word_id(candidate["word"]), status="pending"), source)
            if entry["id"] in seen:
                raise ValueError("Pronunciation discovery repeated an exact word")
            for evidence in entry["evidence"]:
                if not any(_evidence_contains_word(entry["word"], evidence["quote"], text)
                           for text in visible.get(evidence["block_id"], [])):
                    raise ValueError("Pronunciation evidence must be quoted from its visible source window")
            seen.add(entry["id"])
            checked.append(entry)
        return checked
    return validate


def discover_pronunciations(document, requests, progress=None, workers=1):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Pronunciation discovery workers must be from 1 through 8")
    source, cast = document["source"], document.get("cast", {})
    batches = _batches(source, cast, requests.max_request_chars)
    progress = progress or (lambda message: None)
    results = [None] * len(batches)
    progress(f"Discovering pronunciation candidates: {len(batches)} source batch(es)")
    def run(index):
        payload = batches[index]
        return requests.run(SYSTEM, payload, _validator(source, payload))
    if workers == 1:
        for index in range(len(batches)):
            results[index] = run(index)
            progress(f"Pronunciation discovery {index + 1}/{len(batches)} complete")
    else:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(batches)))) as executor:
            futures = {executor.submit(run, index): index for index in range(len(batches))}
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    results[futures[future]] = future.result()
                    progress(f"Pronunciation discovery {completed}/{len(batches)} complete")
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    merged = {}
    for batch in [character_name_candidates(document), *results]:
        for entry in batch:
            if entry["id"] not in merged:
                merged[entry["id"]] = deepcopy(entry)
            else:
                existing = merged[entry["id"]]["evidence"]
                for evidence in entry["evidence"]:
                    if evidence not in existing and len(existing) < 32:
                        existing.append(deepcopy(evidence))
    return sorted(merged.values(), key=lambda entry: (entry["word"].casefold(), entry["word"]))
