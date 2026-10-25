"""Resumable source-bound performance planning without rewriting an analysis.

The model proposes exact quoted spans. Runtime validation resolves their
Unicode offsets, verifies source evidence, and an independent remove-only
review screens their interpretation. Complete batches are atomically saved.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from threading import Lock
import unicodedata

from .attribution_review import _bounded_map
from .performance_plan import validate_performance
from .performance_planning_prompts import SYSTEM, REVIEW_SYSTEM
from .pipeline import CachedRequests, fingerprint
from .provider import MAX_RESPONSE_BYTES, OpenAICompatibleClient
from .spoken_text import build_spoken
from .renderer.common import write_json
from .renderer.run_locks import acquire_directory_lock


FORMAT = "epub-tts-performance-plan"
VERSION = 1
PLANNER_VERSION = 1
MAX_PLAN_BYTES = 64 * 1024 * 1024
BATCH_ROWS = 12
BATCH_CHARACTERS = 6000
_PROVIDER_KEYS = {"model", "base_url", "json_mode", "max_output_tokens", "token_limit_field", "reasoning_effort"}
_TAGS = {"style", "whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful"}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(value, maximum):
    return (isinstance(value, str) and 0 < len(value) <= maximum
            and value == value.strip()
            and not any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value))


def _read_bounded(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Performance planning state must be a regular nonsymlink file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Performance planning state exceeds its size limit")
    def reject(_):
        raise ValueError("Performance planning state contains nonfinite JSON")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=reject)
    except (UnicodeError, RecursionError, json.JSONDecodeError):
        raise ValueError("Performance planning state is not valid JSON") from None


def _safe_path(path):
    path = Path(path).absolute()
    # Resolve the existing parent only after checking each user-selected path
    # component. Never follow a redirected output or cache directory.
    if any(item.is_symlink() for item in [path, *path.parents]):
        raise ValueError("Performance planning paths must not use symlinks")
    return path


class _BoundedRequests(CachedRequests):
    """Reuse the normal bounded repair transport with bounded cache replay too."""

    def _cached_response(self, path, validate):
        if not path.exists() and not path.is_symlink():
            return False, None
        try:
            accepted = validate(_read_bounded(path, MAX_RESPONSE_BYTES))
        except (ValueError, TypeError, KeyError, IndexError):
            with self.progress_lock:
                self.progress("Performance cache failed current validation; requesting a replacement.")
            return False, None
        with self.progress_lock:
            self.cache_hits += 1
        return True, accepted


def _selected(bundle, chapters):
    count = len(bundle["chapters"])
    selected = list(range(1, count + 1)) if chapters is None else list(chapters)
    if (not selected or any(type(number) is not int or not 1 <= number <= count for number in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Performance planning chapters must be distinct valid one-based chapter numbers")
    selected.sort()
    rows = [row for number in selected for row in bundle["chapters"][number - 1]["rows"]]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Performance planning requires distinct segment IDs")
    return selected, rows


def _quote_span(text, quote, occurrence):
    if not isinstance(quote, str) or not quote or type(occurrence) is not int or not 1 <= occurrence <= len(text):
        raise ValueError("Performance control requires an exact quote and positive occurrence")
    cursor = 0
    for _ in range(occurrence):
        start = text.find(quote, cursor)
        if start < 0:
            raise ValueError("Performance control quote occurrence is not in the target spoken text")
        cursor = start + len(quote)
    return start, cursor


def _evidence(value, blocks):
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("Performance controls require one to three source evidence quotations")
    for entry in value:
        if (not isinstance(entry, dict) or set(entry) != {"block_id", "quote"}
                or not isinstance(entry["block_id"], str) or entry["block_id"] not in blocks
                or not _plain(entry["quote"], 2000)
                or entry["quote"] not in blocks[entry["block_id"]]):
            raise ValueError("Performance evidence must quote an exact visible source block substring")
    return deepcopy(value)


def _targets(rows):
    return [{"segment_id": row["id"], "block_id": row["block_id"],
             "kind": row["kind"], "voice_id": row.get("voice_id"),
             "source_text": row["text"], "spoken_text": build_spoken(row)[0],
             "existing_delivery": deepcopy(row.get("delivery", {"direction": "", "evidence": []}))}
            for row in rows]


def _validate_proposal(value, payload):
    targets = payload["target_segments"]
    if (not isinstance(value, dict) or set(value) != {"segments"}
            or not isinstance(value["segments"], list) or len(value["segments"]) != len(targets)):
        raise ValueError("Performance proposal must return every target segment exactly once in input order")
    blocks = {entry["block_id"]: entry["text"] for entry in payload["source_blocks"]}
    records = {}
    for target, proposed in zip(targets, value["segments"]):
        if (not isinstance(proposed, dict) or set(proposed) != {"segment_id", "controls"}
                or proposed["segment_id"] != target["segment_id"]
                or not isinstance(proposed["controls"], list) or len(proposed["controls"]) > 4):
            raise ValueError("Performance proposal has invalid target coverage or controls")
        text = target["spoken_text"]
        controls = []
        for proposal in proposed["controls"]:
            if not isinstance(proposal, dict):
                raise ValueError("Performance control must be an object")
            kind = proposal.get("kind")
            common = {"kind", "occurrence", "evidence", "reason"}
            expected = (common | {"quote", "tag", "direction"} if kind == "delivery"
                        else common | {"after", "duration_ms"} if kind == "pause" else set())
            if not expected or set(proposal) != expected:
                raise ValueError("Automatic performance controls support only exact delivery spans and internal pauses")
            if not _plain(proposal["reason"], 1000):
                raise ValueError("Performance control requires a brief plain reason")
            control = {"kind": kind, "evidence": _evidence(proposal["evidence"], blocks), "reason": proposal["reason"]}
            if kind == "delivery":
                start, end = _quote_span(text, proposal["quote"], proposal["occurrence"])
                tag, direction = proposal["tag"], proposal["direction"]
                if (not isinstance(tag, str) or tag not in _TAGS or not isinstance(direction, str)
                        or (tag == "style" and not _plain(direction, 240))
                        or (tag != "style" and direction != "")
                        or any(char in direction for char in "[]<>`")):
                    raise ValueError("Performance delivery requires a supported tag and plain compatible direction")
                control.update(start=start, end=end, text=text[start:end], tag=tag, direction=direction)
            else:
                _, offset = _quote_span(text, proposal["after"], proposal["occurrence"])
                duration = proposal["duration_ms"]
                if not 0 < offset < len(text) or type(duration) is not int or not 100 <= duration <= 3000:
                    raise ValueError("Automatic pauses require an internal offset and 100..3000 milliseconds")
                control.update(offset=offset, duration_ms=duration)
            controls.append(control)
        record = {"version": 1, "text_sha256": _sha(text), "controls": controls}
        validate_performance(record, text)
        records[target["segment_id"]] = record
    return records


def _validate_review(value, proposals, targets):
    if (not isinstance(value, dict) or set(value) != {"segments"}
            or not isinstance(value["segments"], list) or len(value["segments"]) != len(targets)):
        raise ValueError("Performance review must return every target segment exactly once")
    result = {}
    for target, decision in zip(targets, value["segments"]):
        identifier = target["segment_id"]
        if (not isinstance(decision, dict) or set(decision) != {"segment_id", "keep", "reason"}
                or decision["segment_id"] != identifier or not isinstance(decision["keep"], list)
                or not _plain(decision["reason"], 1000)):
            raise ValueError("Performance review has invalid target coverage or decision")
        indexes = decision["keep"]
        controls = proposals[identifier]["controls"]
        if (any(type(index) is not int or not 0 <= index < len(controls) for index in indexes)
                or indexes != sorted(set(indexes))):
            raise ValueError("Performance review may only retain unique existing control indexes in order")
        result[identifier] = deepcopy(proposals[identifier])
        result[identifier]["controls"] = [deepcopy(controls[index]) for index in indexes]
        result[identifier]["reason"] = decision["reason"]
        validate_performance(result[identifier], target["spoken_text"])
    return result


def _payload(bundle, rows, source_positions):
    source_blocks = bundle["source"]["blocks"]
    needed = set()
    target_ids = {row["block_id"] for row in rows}
    for identifier in target_ids:
        position = source_positions[identifier]
        chapter = source_blocks[position]["chapter_id"]
        for neighbor in range(max(0, position - 1), min(len(source_blocks), position + 2)):
            if source_blocks[neighbor]["chapter_id"] == chapter:
                needed.add(neighbor)
    context = [{"block_id": source_blocks[index]["id"], "text": source_blocks[index]["text"]}
               for index in sorted(needed)]
    return {"mode": "performance_planning", "version": PLANNER_VERSION,
            "target_segments": _targets(rows), "source_blocks": context}


def _batches(bundle, rows, maximum):
    positions = {block["id"]: index for index, block in enumerate(bundle["source"]["blocks"])}
    batches, current, chars = [], [], 0
    for row in rows:
        size = len(build_spoken(row)[0])
        trial = current + [row]
        payload = _payload(bundle, trial, positions)
        fits = len(SYSTEM) + len(json.dumps(payload, ensure_ascii=False)) <= maximum // 2
        # Reserve half of the configured request space for proposals and review.
        if current and (len(trial) > BATCH_ROWS or chars + size > BATCH_CHARACTERS or not fits):
            batches.append(_payload(bundle, current, positions))
            current, chars = [], 0
        current.append(row)
        chars += size
    if current:
        batches.append(_payload(bundle, current, positions))
    if any(len(SYSTEM) + len(json.dumps(payload, ensure_ascii=False)) > maximum // 2 for payload in batches):
        raise ValueError("Performance planning source context exceeds analysis.max_request_chars; increase its budget")
    return batches


def _validate_state(value, bundle, chapters, rows, *, complete):
    if (not isinstance(value, dict) or value.get("format") != FORMAT or type(value.get("version")) is not int
            or value["version"] != VERSION or value.get("book_id") != bundle["source"]["book"]["id"]
            or value.get("source_fingerprint") != bundle["manifest"]["source_fingerprint"]
            or value.get("chapters") != chapters or value.get("status") not in {"pending", "complete"}
            or not isinstance(value.get("segments"), dict) or not isinstance(value.get("provenance"), dict)
            or value["provenance"].get("script_sha256") != fingerprint(rows)
            or value["provenance"].get("cast_sha256") != fingerprint(bundle["cast"])):
        raise ValueError("Performance plan does not match the current book, source, scripts, or chapters")
    expected = {row["id"]: row for row in rows}
    blocks = {block["id"]: block["text"] for block in bundle["source"]["blocks"]}
    if set(value["segments"]) - set(expected):
        raise ValueError("Performance plan contains segments outside its selected chapters")
    for identifier, record in value["segments"].items():
        validate_performance(record, build_spoken(expected[identifier])[0])
        for control in record["controls"]:
            if "evidence" in control:
                _evidence(control["evidence"], blocks)
    if complete and (value["status"] != "complete" or set(value["segments"]) != set(expected)):
        raise ValueError("Performance plan is incomplete; rerun prepare-performance to resume")
    return value


def load_prepared_performance(bundle, path, *, chapters=None):
    """Read a completed source/script-bound sidecar without consulting an LLM."""
    value = _read_bounded(_safe_path(path), MAX_PLAN_BYTES)
    selected, rows = _selected(bundle, value.get("chapters") if isinstance(value, dict) else [])
    _validate_state(value, bundle, selected, rows, complete=True)
    requested, _ = _selected(bundle, chapters)
    if not set(requested).issubset(selected):
        raise ValueError("Performance plan does not cover all requested chapters")
    return value


def prepare_performance(bundle, analysis_config, output_path, *, chapters=None, client=None, notify=print):
    """Prepare selected chapters with bounded concurrency and atomic resumability.

    Model calls are isolated from canonical book files and never load TTS.
    An unchanged complete plan returns without calls; interrupted runs restore
    each completed batch and validated proposal/review request independently.
    """
    selected, rows = _selected(bundle, chapters)
    settings = analysis_config["analysis"]
    workers = settings.get("attribution_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Performance planning workers must be between one and eight")
    workers = min(workers, 5)
    client = client or OpenAICompatibleClient(**analysis_config["provider"])
    provider = {key: deepcopy(value) for key, value in client.identity.items() if key in _PROVIDER_KEYS}
    provenance = {"planner_version": PLANNER_VERSION, "provider": provider,
                  "prompt_sha256": _sha(SYSTEM), "review_prompt_sha256": _sha(REVIEW_SYSTEM),
                  "script_sha256": fingerprint(rows), "cast_sha256": fingerprint(bundle["cast"]),
                  "review_method": "independent_remove_only"}
    provenance["inputs_sha256"] = fingerprint({"source": bundle["manifest"]["source_fingerprint"],
                                               "chapters": selected, **provenance})
    path = _safe_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    work = _safe_path(path.with_name(path.name + ".work"))
    work.mkdir(mode=0o700, exist_ok=True)
    cache = _safe_path(work / "requests")
    cache.mkdir(mode=0o700, exist_ok=True)
    _safe_path(cache / "failed")
    lock = acquire_directory_lock(path.with_name(path.name + ".lock"))
    try:
        state = {"format": FORMAT, "version": VERSION, "book_id": bundle["source"]["book"]["id"],
                 "source_fingerprint": bundle["manifest"]["source_fingerprint"], "status": "pending",
                 "chapters": selected, "segments": {}, "provenance": provenance}
        if path.exists():
            prior = _read_bounded(path, MAX_PLAN_BYTES)
            if (isinstance(prior, dict) and isinstance(prior.get("provenance"), dict)
                    and prior["provenance"].get("inputs_sha256") == provenance["inputs_sha256"]):
                state = _validate_state(prior, bundle, selected, rows, complete=prior.get("status") == "complete")
                if state["status"] == "complete":
                    notify(f"Performance plan reused: {len(rows)} segment(s); no API calls.")
                    return deepcopy(state)
            else:
                notify("Performance inputs changed; preparing a newly bound plan.")
        # Invalidate before expensive work. A failed refresh can never expose an
        # old complete plan as current, even when the original file existed.
        write_json(path, state)
        requests = _BoundedRequests(client, cache, repair_attempts=settings.get("repair_attempts", 1),
                                    max_request_chars=settings.get("max_request_chars", 200000), progress=notify)
        batches = _batches(bundle, rows, requests.max_request_chars)
        pending = [payload for payload in batches if any(target["segment_id"] not in state["segments"]
                                                         for target in payload["target_segments"])]
        state_lock = Lock()
        total = len(rows)
        notify(f"Preparing performance: {len(state['segments'])}/{total} segment(s), {len(pending)} batch(es).")

        def run(payload, cancelled):
            proposed = requests.run(SYSTEM, payload, lambda value: _validate_proposal(value, payload), cancelled=cancelled)
            if any(record["controls"] for record in proposed.values()):
                review = {**payload, "mode": "performance_review", "proposals": proposed}
                accepted = requests.run(REVIEW_SYSTEM, review,
                                        lambda value: _validate_review(value, proposed, payload["target_segments"]), cancelled=cancelled)
            else:
                accepted = proposed
            with state_lock:
                state["segments"].update(accepted)
                write_json(path, state)
                notify(f"Preparing performance: {len(state['segments'])}/{total} segment(s) saved.")
            return True

        _bounded_map(pending, run, workers)
        state["status"] = "complete"
        _validate_state(state, bundle, selected, rows, complete=True)
        write_json(path, state)
        notify(f"Performance plan complete: {total} segment(s), "
               f"{sum(len(record['controls']) for record in state['segments'].values())} control(s).")
        return deepcopy(state)
    finally:
        lock.rmdir()
